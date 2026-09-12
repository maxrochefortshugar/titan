# SPDX-License-Identifier: Apache-2.0
"""Concurrent MTP for Qwen3.8-Flash-Next (qwen4_exp): one fused verify forward
for B sequences.

Today two concurrent requests get no MTP at all. oMLX has a row-wise batched
MTP path (``_MtpBatchState`` / ``_mtp_batch_next``,
omlx/patches/mlx_lm_mtp/batch_generator.py:2590-2629) but it is opt-in behind
OMLX_MTP_ROWWISE_BATCH and loses to plain batched decode, because it runs one
backbone forward per row per cycle and copies every row's KV cache out and
back with ``extract_cache`` / ``merge`` on every cycle.

This patch replaces the body of that loop with a single fused verify:

    inputs (B, k+1) -> ONE backbone forward against the batch cache
    -> per-row greedy acceptance in graph, ONE host sync
    -> ONE batched rollback (rollback_speculative_cache already takes a
       per-row accepted list)
    -> per-row MTP head chain (unchanged, still the stock
       ``_chain_next_drafts``, so copy-lane and mtp-depth still compose)

Rows scale with B, which is the lever: the expert gather runs at 300 GB/s at
one row and 549 at eight (kernels/AUDIT-2026-09-12.md:95). The extract/merge
copies disappear entirely because the fused path writes straight into
``gen_batch.prompt_cache``.

Installs (all import time, module function swaps, idempotent):

  install_batched_mtp()
      Gate OMLX_MTP_BATCHED=1. Swaps three module functions in
      omlx/patches/mlx_lm_mtp/batch_generator.py:
        _rowwise_batch_mtp_enabled  (:432-451)  -> True, so multi-row batches
                                                   activate row MTP state
        _mtp_batch_next             (:2590-2629) -> fused verify
        _generation_batch_has_active_mtp (:351-373) -> False, so a request
                                                   arriving mid-MTP merges
                                                   instead of being pinned out
                                                   (item 3 of the brief)

  install_batched_mtp_scheduler()
      Gate OMLX_MTP_BATCHED=1. Lifts the "scheduler contention" refusal in
      omlx/scheduler.py:9113-9130 (an explicit preference, not a correctness
      constraint) on the external-drafter vlm_mtp path. The "drafter is busy"
      refusal at :9096-9107 stays unless OMLX_MTP_BATCHED_MULTI_DRAFTER=1,
      because the drafter really does keep per-request state on the module
      instance. Not used by Flash-Next, which drafts with its own MTP head
      through batch_generator, but the brief asks for it.

Env

  OMLX_MTP_BATCHED               0     master gate
  OMLX_MTP_BATCHED_MAX_B         4     do not fuse above this many rows
  OMLX_MTP_BATCHED_MAX_ROWS      16    cap k so B*(k+1) stays at or under this
  OMLX_MTP_BATCHED_MAX_CTX       16384 do not fuse past this cache offset; the
                                       batched QSA cache fails the strict
                                       ``type(c) is QSAKVCache`` test so verify
                                       drops to dense attention, which is flat
                                       cheap short and ruinous long
                                       (round3/qsa-verify/REPORT.md section 2)
  OMLX_MTP_BATCHED_TRACE_EVERY   0     log a fused-cycle line every N cycles
  OMLX_MTP_BATCHED_MULTI_DRAFTER 0     also lift the vlm_mtp "drafter is busy"
                                       refusal (unsafe, see above)
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BG = "omlx.patches.mlx_lm_mtp.batch_generator"

_INSTALLED = False
_SCHED_INSTALLED = False

# module-level trace counters
_TRACE = {"cycles": 0, "rows": 0, "accepted": 0, "tokens": 0, "fallbacks": 0}


def _envflag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return _envflag("OMLX_MTP_BATCHED")


# ---------------------------------------------------------------------------
# the fused cycle
# ---------------------------------------------------------------------------


def _cache_offset(prompt_cache: List[Any]) -> int:
    """Largest offset over the cache list, 0 when nothing reports one."""
    best = 0
    pending = list(prompt_cache or ())
    while pending:
        c = pending.pop()
        pending.extend(getattr(c, "caches", ()) or ())
        off = getattr(c, "offset", None)
        if off is None:
            continue
        try:
            if hasattr(off, "reshape"):
                off = max(int(v) for v in off.reshape(-1).tolist())
            best = max(best, int(off))
        except Exception:
            continue
    return best


def _set_batch_mrope_deltas(bg: Any, gen_batch: Any) -> None:
    """Bind per-row mRoPE deltas for the fused verify forward.

    Mirrors scheduler._bind_step_rope_deltas rather than
    ``_set_singleton_mrope_delta`` (batch_generator.py:1235-1257), which
    refuses anything but one uid. Note the consequence, spelled out in the
    report: ``set_step_rope_deltas`` only keeps Qwen4's rank-two text
    positions at len(uids) == 1, so a fused batch is on rank-three mRoPE ids,
    which is a second reason verify takes the dense arm.
    """
    import mlx.core as mx

    model = getattr(gen_batch, "model", None)
    uids = list(getattr(gen_batch, "uids", None) or ())
    if model is None or not uids:
        return
    if not getattr(model, "_uses_mrope", False):
        return
    table = getattr(model, "_uid_rope_deltas", None)
    if not table:
        return
    deltas = mx.array([float(table.get(u, 0.0)) for u in uids])
    step_setter = getattr(type(model), "set_step_rope_deltas", None)
    if callable(step_setter):
        step_setter(model, deltas, uids)
    elif hasattr(model, "set_batch_rope_deltas"):
        model.set_batch_rope_deltas(deltas)


def _fusable(bg: Any, gen_batch: Any, batch_state: Any) -> Optional[List[Any]]:
    """Return the per-row (idx, uid, state) triples to fuse, or None.

    None means "hand this cycle back to the stock row-wise loop". Every test
    here is a precondition of the fused arithmetic, not a preference.
    """
    uids = list(getattr(gen_batch, "uids", None) or ())
    if len(uids) < 2:
        return None
    max_b = _envint("OMLX_MTP_BATCHED_MAX_B", 4)
    if len(uids) > max_b:
        return None

    # Lockstep, and it has to be. The batch cache has one row per sequence, so
    # a forward covers all B rows or none: a subset cannot be advanced. Rows
    # drain their queue at one token per next() call but refill m+1 at a time,
    # so once acceptance differs they are never simultaneously dry again.
    # Every row therefore advances on every fused cycle, and a row that is
    # ahead builds an emit queue. ``_queue_cap`` bounds that queue by clamping
    # the ahead row's accepted count, which is always correct (accepting fewer
    # verified drafts is legal) and bounds how far a row can over-run a stop
    # token.
    rows = []
    for idx, uid in enumerate(uids):
        state = batch_state.states.get(uid)
        if state is None:
            return None
        if state.next_main is None or state.drafts is None:
            return None
        if not getattr(state, "chain", False):
            return None
        rows.append((idx, uid, state))
    if min(len(s.queue) for _, _, s in rows) > 0:
        # Nothing is dry: emit from the queues, do not spend a forward.
        return None

    # Greedy only: the stochastic acceptance walk needs per-row residual
    # sampling against per-row draft densities, which the fused sync would
    # have to serialise anyway. Stock loop keeps those rows exact.
    if not bg._is_greedy(gen_batch):
        return None
    if bg._proc_list(gen_batch) is not None:
        return None
    if int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0):
        return None
    if callable(getattr(gen_batch.model, "mtp_clamp_accept", None)):
        return None

    ks = [int(s.drafts.shape[0]) for _, _, s in rows]
    k = min(ks)
    if k < 1:
        return None
    max_rows = _envint("OMLX_MTP_BATCHED_MAX_ROWS", 16)
    while k >= 1 and len(rows) * (k + 1) > max_rows:
        k -= 1
    if k < 1:
        return None

    max_ctx = _envint("OMLX_MTP_BATCHED_MAX_CTX", 16384)
    if max_ctx and _cache_offset(gen_batch.prompt_cache) > max_ctx:
        return None

    return [(idx, uid, state, k) for (idx, uid, state) in rows]


def _mtp_batch_next_fused(gen_batch: Any, batch_state: Any) -> Any:
    """One fused verify cycle over every row of a multi-row MTP batch."""
    import mlx.core as mx

    bg = importlib.import_module(_BG)
    orig = _ORIGINALS.get("_mtp_batch_next")

    plan = _fusable(bg, gen_batch, batch_state)
    if plan is None:
        _TRACE["fallbacks"] += 1
        return orig(gen_batch, batch_state)

    k = plan[0][3]
    B = len(plan)
    cycle_t0 = time.perf_counter()

    _set_batch_mrope_deltas(bg, gen_batch)

    # --- build the (B, k+1) verify window -------------------------------
    drafts = mx.stack([s.drafts[:k].astype(mx.uint32) for _, _, s, _ in plan])
    next_main = mx.concatenate([s.next_main.reshape(1) for _, _, s, _ in plan])
    inputs = mx.concatenate([next_main[:, None], drafts], axis=1)  # (B, k+1)

    # --- ONE backbone forward -------------------------------------------
    t0 = time.perf_counter()
    logits, hidden, gdn_states = bg._call_backbone(
        gen_batch.model,
        inputs,
        gen_batch.prompt_cache,
        n_confirmed=1,
    )
    if logits.shape[0] != B or logits.shape[1] != k + 1:
        # The model did not honour the batch dimension. Nothing has been
        # committed yet beyond the cache write, which the stock fallback
        # cannot undo, so this must raise rather than silently degrade.
        raise bg._MtpStepFallback(
            f"fused verify returned {tuple(logits.shape)[:2]}, expected {(B, k + 1)}"
        )

    # --- acceptance, in graph, ONE host sync ----------------------------
    targets = mx.argmax(logits, axis=-1).astype(mx.int32)  # (B, k+1)
    matches = (targets[:, :k] == drafts.astype(mx.int32)).astype(mx.int32)
    m_arr = mx.cumprod(matches, axis=1).sum(axis=1)  # (B,)
    host = mx.concatenate(
        [m_arr.reshape(-1), targets.reshape(-1), drafts.astype(mx.int32).reshape(-1)]
    ).tolist()
    backbone_ms = (time.perf_counter() - t0) * 1000.0

    ms = [int(v) for v in host[:B]]
    tgt_flat = host[B : B + B * (k + 1)]
    drf_flat = host[B + B * (k + 1) :]
    targets_per_row = [tgt_flat[b * (k + 1) : (b + 1) * (k + 1)] for b in range(B)]
    drafts_per_row = [drf_flat[b * k : (b + 1) * k] for b in range(B)]

    # Bound the emit queue of rows that are running ahead. Accepting fewer
    # verified drafts is always correct (the stock cycle does the same for
    # ``mtp_clamp_accept`` and boundary alignment), and ``targets[m]`` stays
    # the right token at the shortened position because it is the target's
    # own argmax there.
    cap = _envint("OMLX_MTP_BATCHED_QUEUE_CAP", 8)
    if cap > 0:
        ms = [
            min(m, max(0, cap - 1 - len(plan[b][2].queue)))
            for b, m in enumerate(ms)
        ]

    combined_lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # (B,k+1,V)

    # --- ONE batched rollback -------------------------------------------
    t0 = time.perf_counter()
    if all(m == k for m in ms):
        bg._clear_rollback(gen_batch.prompt_cache)
    else:
        rolled = False
        if gdn_states is not None and hasattr(
            gen_batch.model, "rollback_speculative_cache"
        ):
            try:
                gen_batch.model.rollback_speculative_cache(
                    gen_batch.prompt_cache, gdn_states, list(ms), k + 1
                )
                rolled = True
            except Exception as exc:  # pragma: no cover - guarded fallback
                logger.debug("fused rollback_speculative_cache failed: %s", exc)
        if not rolled:
            raise bg._MtpStepFallback("fused cycle: batched rollback unavailable")
    cache_ms = (time.perf_counter() - t0) * 1000.0

    # --- per-row commit --------------------------------------------------
    for b, (idx, uid, state, _) in enumerate(plan):
        m = ms[b]
        draft_ids = drafts_per_row[b]
        emit_last_id = targets_per_row[b][m]
        emit_last_lp = combined_lp[b, m]

        state.stats.cycles += 1
        state.stats.backbone_ms += backbone_ms / B
        state.stats.cache_ops_ms += cache_ms / B
        if len(state.stats.depth_drafted) < state.depth:
            pad = state.depth - len(state.stats.depth_drafted)
            state.stats.depth_drafted.extend([0] * pad)
            state.stats.depth_accepted.extend([0] * pad)
        for j in range(k):
            state.stats.depth_drafted[j] += 1
            if j < m:
                state.stats.depth_accepted[j] += 1
            else:
                break
        state.stats.accepts += m
        if m < k:
            state.stats.rejects += 1

        for j in range(m):
            state.queue.append((int(draft_ids[j]), state.draft_lps[j], "draft"))
        state.queue.append(
            (int(emit_last_id), emit_last_lp, "bonus" if m == k else "verify")
        )

        # --- head history + next chain, per row, stock code -------------
        t0 = time.perf_counter()
        if not state.head_clone:
            bg._mtp_head_trim_to(state.mtp_cache, state.hist_offset)
        committed = mx.array(
            [int(d) for d in draft_ids[:m]] + [int(emit_last_id)], dtype=mx.uint32
        )
        row = bg._make_row_batch(
            gen_batch, idx, prompt_cache=gen_batch.prompt_cache, state=state
        )
        bg._chain_next_drafts(row, state, hidden[b : b + 1, : m + 1], committed, None)
        state.next_main = committed[-1:]
        state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000.0

        _TRACE["accepted"] += m
        _TRACE["tokens"] += m + 1

    # --- depth controllers ----------------------------------------------
    cycle_ms = (time.perf_counter() - cycle_t0) * 1000.0
    for b, (_, _, state, _) in enumerate(plan):
        if state.controller is None:
            continue
        keepalive = bool(getattr(state.mtp_cache, "fold_keepalive", False))
        if keepalive:
            state.mtp_cache.fold_keepalive = False
        # Every row paid the whole fused cycle, so every row's cost model
        # sees the whole fused cycle. That is what makes the controller shrink
        # depth under concurrency, which is the right answer (see the report's
        # break-even table).
        state.controller.observe(k, ms[b], cycle_ms, time_sample=not keepalive)

    _TRACE["cycles"] += 1
    _TRACE["rows"] += B
    every = _envint("OMLX_MTP_BATCHED_TRACE_EVERY", 0)
    if every and _TRACE["cycles"] % every == 0:
        logger.info(
            "MTP batched: cycles=%d mean_B=%.2f k=%d accept=%d/%d tok/cycle=%.2f "
            "last_cycle=%.1fms fallbacks=%d",
            _TRACE["cycles"],
            _TRACE["rows"] / max(1, _TRACE["cycles"]),
            k,
            _TRACE["accepted"],
            _TRACE["cycles"] * B * k,
            _TRACE["tokens"] / max(1, _TRACE["cycles"]),
            cycle_ms,
            _TRACE["fallbacks"],
        )

    return bg._emit_batch_responses(gen_batch, batch_state)


# ---------------------------------------------------------------------------
# installs
# ---------------------------------------------------------------------------

_ORIGINALS: Dict[str, Any] = {}


def install_batched_mtp() -> bool:
    """Import-time module function swap in batch_generator. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not enabled():
        return False
    try:
        bg = importlib.import_module(_BG)
    except Exception as exc:
        logger.warning("MTP batched: %s unavailable (%s)", _BG, exc)
        return False

    required = (
        "_mtp_batch_next",
        "_rowwise_batch_mtp_enabled",
        "_generation_batch_has_active_mtp",
        "_emit_batch_responses",
        "_make_row_batch",
        "_chain_next_drafts",
        "_call_backbone",
        "_clear_rollback",
        "_mtp_head_trim_to",
        "_is_greedy",
        "_proc_list",
        "_MtpStepFallback",
    )
    missing = [n for n in required if not hasattr(bg, n)]
    if missing:
        logger.warning("MTP batched: %s missing %s", _BG, ", ".join(missing))
        return False

    for name in ("_mtp_batch_next", "_rowwise_batch_mtp_enabled",
                 "_generation_batch_has_active_mtp"):
        _ORIGINALS[name] = getattr(bg, name)

    bg._mtp_batch_next = _mtp_batch_next_fused
    # Multi-row batches now activate row MTP state without the opt-in env var
    # the stock gate demands (its own docstring says row-wise loses; the fused
    # verify is why that judgement changes).
    bg._rowwise_batch_mtp_enabled = lambda: True
    # Item 3: a request arriving mid-MTP is no longer pinned out. Returning
    # False here lets BatchGenerator._next keep its real completion_batch_size,
    # so the late arrival merges through patched_extend, which reconciles the
    # active MTP state to standard first and then re-activates every row on
    # the next call via _prepare_mtp_batch_state_for_next.
    bg._generation_batch_has_active_mtp = lambda gen_batch: False

    _INSTALLED = True
    logger.info(
        "MTP batched verify installed (max_B=%d max_rows=%d max_ctx=%d)",
        _envint("OMLX_MTP_BATCHED_MAX_B", 4),
        _envint("OMLX_MTP_BATCHED_MAX_ROWS", 16),
        _envint("OMLX_MTP_BATCHED_MAX_CTX", 16384),
    )
    return True


def install_batched_mtp_scheduler() -> bool:
    """Lift the vlm_mtp contention refusals (external-drafter path). Idempotent."""
    global _SCHED_INSTALLED
    if _SCHED_INSTALLED:
        return True
    if not enabled():
        return False
    try:
        sched_mod = importlib.import_module("omlx.scheduler")
    except Exception as exc:
        logger.warning("MTP batched: omlx.scheduler unavailable (%s)", exc)
        return False
    cls = getattr(sched_mod, "Scheduler", None)
    route = getattr(cls, "_route_to_vlm_mtp", None) if cls is not None else None
    if route is None:
        logger.warning("MTP batched: Scheduler._route_to_vlm_mtp not found")
        return False

    multi = _envflag("OMLX_MTP_BATCHED_MULTI_DRAFTER")

    def patched_route(self, *args, **kwargs):
        # The two refusals live inside the method body and read scheduler
        # attributes, so the cheapest exact lift is to hide those attributes
        # for the duration of the call. ``waiting`` / ``running`` /
        # ``prefilling`` only feed the contention preference; the real
        # scheduling structures are untouched.
        saved = {}
        for attr in ("waiting", "running", "prefilling"):
            if hasattr(self, attr):
                saved[attr] = getattr(self, attr)
                object.__setattr__(self, attr, ())
        active = None
        if multi:
            active = self._vlm_mtp_active
            object.__setattr__(self, "_vlm_mtp_active", {})
        try:
            return route(self, *args, **kwargs)
        finally:
            for attr, value in saved.items():
                object.__setattr__(self, attr, value)
            if active is not None:
                object.__setattr__(self, "_vlm_mtp_active", active)

    cls._route_to_vlm_mtp = patched_route
    _SCHED_INSTALLED = True
    logger.info(
        "MTP batched: vlm_mtp contention refusal lifted (multi_drafter=%s)", multi
    )
    return True


def install(model: Any = None) -> bool:  # convenience for bootstrap
    ok = install_batched_mtp()
    try:
        ok = install_batched_mtp_scheduler() or ok
    except Exception:
        pass
    return ok
