# SPDX-License-Identifier: Apache-2.0
"""Per-cycle confidence-gated MTP draft depth for Qwen3.8-Flash-Next (qwen4_exp).

OMLX_MTP_CONF_DEPTH=1 replaces the fixed per-cycle draft depth with a gate:
draft up to a ceiling, but stop as soon as the running product of the
drafter's own proposal probabilities falls under a floor. The verify batch
is then sized per cycle (M = k + 1 varies) instead of being fixed at
depth + 1.

Two installs, mirroring round-2's contract:

  install_conf_depth()            import time, module function swap
      Replaces omlx/patches/mlx_lm_mtp/batch_generator.py:2303-2434
      (``_chain_next_drafts``) and wraps
      batch_generator.py:2169-2178 (``_DepthController._best``).

  install_conf_depth_ceiling(model)   AFTER load, needs the model instance
      Raises the chain depth marker stamped at
      .../vendor/mlx_vlm/models/qwen4_exp/language.py:3060
      (``_omlx_mtp_depth``) to OMLX_MTP_CONF_MAX_DEPTH, so the controller,
      ``_MtpState.depth`` and the per-depth stats arrays are all sized for
      the higher ceiling.

Composition with the round-2 shortlist drafter (OMLX_MTP_SHORTLIST_DRAFT=1):
both patches own ``_chain_next_drafts``, so this module subsumes it. When
OMLX_MTP_SHORTLIST_DRAFT=1 the loop below routes draft steps
>= OMLX_MTP_SHORTLIST_FROM_STEP through round-2's shortlist head, reusing
that module's helpers verbatim (imported by path). The probability the gate
reads is always the row the sampler actually drew from on that step, so on a
shortlisted step it is the shortlist head's own softmax, never a stale
full-vocabulary value.

Env vars

  OMLX_MTP_CONF_DEPTH        0    master gate
  OMLX_MTP_CONF_PMIN         0.25 static running-product floor
  OMLX_MTP_CONF_MAX_DEPTH    5    draft ceiling
  OMLX_MTP_CONF_MIN_DEPTH    1    never gate below this many drafts
  OMLX_MTP_CONF_ADAPT        1    also apply the measured break-even floor
  OMLX_MTP_CONF_PSTEP        0.0  optional per-step floor (llama.cpp form)
  OMLX_MTP_CONF_PFLOOR       0.02 hard clamp, low side
  OMLX_MTP_CONF_PCEIL        0.90 hard clamp, high side
  OMLX_MTP_CONF_TRACE_EVERY  0    log the chosen-depth histogram every N cycles
  OMLX_MTP_SHORTLIST_*            round-2 knobs, honoured unchanged
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_BG = "omlx.patches.mlx_lm_mtp.batch_generator"
_R2 = Path(__file__).resolve().parents[2] / "round2" / "mtp" / "patch.py"

_CHAIN_INSTALLED = False
_CEILING_INSTALLED = False
_BEST_WRAPPED = False


def _envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envfloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def conf_enabled() -> bool:
    return _enabled("OMLX_MTP_CONF_DEPTH")


def _load_round2():
    """Round-2's shortlist helpers, imported by path (never by module name)."""
    if not _R2.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_omlx_mtp_r2", _R2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


class _GateCfg:
    """Snapshot of the env knobs, read once at install."""

    __slots__ = (
        "pmin",
        "max_depth",
        "min_depth",
        "adapt",
        "pstep",
        "pfloor",
        "pceil",
        "trace_every",
        "shortlist",
        "sl_k",
        "sl_from",
        "sl_refresh",
    )

    def __init__(self):
        self.pmin = _envfloat("OMLX_MTP_CONF_PMIN", 0.25)
        self.max_depth = max(1, _envint("OMLX_MTP_CONF_MAX_DEPTH", 5))
        self.min_depth = max(0, _envint("OMLX_MTP_CONF_MIN_DEPTH", 1))
        self.adapt = _enabled("OMLX_MTP_CONF_ADAPT", "1")
        self.pstep = _envfloat("OMLX_MTP_CONF_PSTEP", 0.0)
        self.pfloor = _envfloat("OMLX_MTP_CONF_PFLOOR", 0.02)
        self.pceil = _envfloat("OMLX_MTP_CONF_PCEIL", 0.90)
        self.trace_every = max(0, _envint("OMLX_MTP_CONF_TRACE_EVERY", 0))
        self.shortlist = _enabled("OMLX_MTP_SHORTLIST_DRAFT")
        self.sl_k = _envint("OMLX_MTP_SHORTLIST_K", 2048)
        self.sl_from = 1 if _envint("OMLX_MTP_SHORTLIST_FROM_STEP", 1) else 0
        self.sl_refresh = max(1, _envint("OMLX_MTP_SHORTLIST_REFRESH", 1))


def _break_even_floor(cfg: _GateCfg, ctl, j: int, expected: float) -> float:
    """Running-product floor for taking draft step ``j + 1``.

    Drafting one more token buys ``P_j * p[j]`` extra expected tokens for
    ``D_{j+1} = C(j+1) - C(j)`` extra milliseconds, against a cycle that
    currently yields ``expected`` tokens in ``C(j)`` ms. Taking the step is
    worth it exactly while

        P_j * p[j] / D_{j+1}  >  expected / C(j)

    i.e. while ``P_j`` exceeds ``D_{j+1} * expected / (p[j] * C(j))``. Both
    ``C`` and ``p`` come from the controller's own measurements, so this is
    the same break-even oMLX already scores, evaluated against this cycle's
    live confidence instead of an acceptance EMA.
    """
    if ctl is None or not cfg.adapt:
        return cfg.pmin
    try:
        c_j = float(ctl._t_est(j))
        c_j1 = float(ctl._t_est(j + 1))
        p_bar = float(ctl.p[j]) if j < len(ctl.p) else float(ctl.p[-1])
    except Exception:  # noqa: BLE001
        return cfg.pmin
    d_marg = c_j1 - c_j
    if not (c_j > 0.0) or not math.isfinite(d_marg):
        return cfg.pmin
    if d_marg <= 0.0:
        return cfg.pfloor  # a deeper cycle measured no slower: always draft
    if p_bar <= 1e-6:
        return cfg.pceil
    f = (d_marg * expected) / (p_bar * c_j)
    if not math.isfinite(f):
        return cfg.pmin
    return min(cfg.pceil, max(cfg.pfloor, f))


class _GateStats:
    __slots__ = ("hist", "cycles", "syncs")

    def __init__(self, width: int):
        self.hist = [0] * (width + 1)
        self.cycles = 0
        self.syncs = 0

    def note(self, k: int):
        if k < len(self.hist):
            self.hist[k] += 1
        self.cycles += 1


def _make_gated_chain(bg, orig, r2, cfg: _GateCfg):
    import mlx.core as mx

    stats = _GateStats(max(cfg.max_depth, 8))

    def patched(gen_batch, state, hidden_rows, committed, prev_buf):
        model = gen_batch.model
        if bg._dspark_host(model) is not None:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        ctl = getattr(state, "controller", None)
        cur = int(ctl.cur) if ctl is not None else int(state.depth)
        ceiling = max(0, min(cur, int(state.depth)))

        # Warmup sweeps and probe bursts are the controller measuring a
        # specific depth. Gating them would poison t[] with a depth it did
        # not run, so those cycles stay fixed-depth and double as the
        # in-flight fixed-vs-gated comparison.
        gate_on = ceiling > 0 and (
            ctl is None or (not ctl._warmup and ctl.probe_left == 0)
        )

        # --- shortlist plumbing (round-2), only when it is enabled ---------
        use_sl = False
        head = lm = mtp = sl = None
        if cfg.shortlist and r2 is not None:
            head, lm = r2._head_of(model)
            if head is not None and r2._is_affine_qlinear(head):
                getter = getattr(lm, "get_mtp_module", None)
                mtp = getter() if callable(getter) else None
                use_sl = mtp is not None and ceiling > cfg.sl_from
        if not use_sl:
            lm = model
            for attr in ("language_model", "_language_model"):
                inner = getattr(model, attr, None)
                if inner is not None and hasattr(inner, "mtp_forward"):
                    lm = inner
                    break

        if ceiling == 0 and not state.mtp_cache:
            state.drafts = mx.zeros((0,), dtype=mx.uint32)
            state.draft_lps = []
            state.draft_accept_lps = []
            stats.note(0)
            return

        sampler = bg._resolve_draft_sampler(gen_batch, state)
        procs = bg._proc_list(gen_batch)

        if use_sl:
            sl = getattr(state, "_omlx_shortlist", None)
            if sl is None:
                sl = r2._Shortlist()
                state._omlx_shortlist = sl
            sl.age += 1
            full_step0 = (
                cfg.sl_from >= 1 or sl.ids is None or sl.age >= cfg.sl_refresh
            )
        else:
            full_step0 = True

        head_prenorm = getattr(model, "_omlx_mtp_head_prenorm", False) or getattr(
            getattr(model, "_language_model", None), "_omlx_mtp_head_prenorm", False
        )
        if bg._HEAD_HIDDEN_POST_NORM and not head_prenorm and hidden_rows.ndim == 3:
            hidden_rows = bg._trunk_norm_module(model)(hidden_rows)

        begin = getattr(model, "mtp_begin_cycle", None) or getattr(
            getattr(model, "_language_model", None), "mtp_begin_cycle", None
        )
        if begin is not None:
            begin(state.mtp_cache, ceiling)

        n = committed.shape[0]
        lp_next = None
        logits = None
        if full_step0:
            logits, head_hidden = lm.mtp_forward(
                hidden_rows,
                committed.reshape(1, n),
                state.mtp_cache,
                return_hidden=True,
                logits_keep=1,
            )
        else:
            mtp_out, head_hidden = mtp(
                hidden_rows,
                committed.reshape(1, n),
                lm.model.embed_tokens,
                state.mtp_cache,
            )
            lp_next = r2._shortlist_logprobs(mx, sl, head, mtp_out[:, -1, :])
        state.hist_offset += int(n)

        draft_toks: List[Any] = []
        draft_lps: List[Any] = []
        draft_accept_lps: List[Any] = []

        chain_prefix = committed[-1:]
        h = head_hidden[:, -1:]
        chain_cache = state.mtp_cache
        if state.head_clone and ceiling > 1:
            chain_cache = bg._clone_mtp_head_cache(state.mtp_cache)

        snap = bg._snap_snapshotable(procs)
        embed = getattr(getattr(lm, "model", None), "embed_tokens", None)

        run_prod = 1.0  # product of proposal probabilities so far
        expected = 1.0  # 1 + sum of running products = expected tokens

        for j in range(ceiling):
            normalised = lp_next is not None
            logits_2d = lp_next if normalised else logits[:, -1, :]
            if procs is not None and prev_buf is not None:
                prev = mx.concatenate(
                    [prev_buf.astype(mx.int32), chain_prefix.astype(mx.int32)]
                    + [t.reshape(1).astype(mx.int32) for t in draft_toks]
                )
                logits_2d = bg._apply_processors(procs, prev, logits_2d)
            lp_2d = logits_2d if normalised else bg._logprobs(logits_2d)
            tok = bg._ensure_uint32(sampler(lp_2d))
            draft_toks.append(tok)
            draft_lps.append(lp_2d.squeeze(0))
            draft_accept_lps.append(bg._accept_lp_for(sampler, lp_2d).squeeze(0))
            if j + 1 == ceiling:
                break

            # ---- the gate -------------------------------------------------
            # ``lp_2d`` is the row this step's sampler actually drew from,
            # after the logit processors and (on a shortlisted step) after
            # the shortlist renormalisation. That is the drafter's proposal
            # probability for the token it just proposed.
            if gate_on and (j + 1) >= max(1, cfg.min_depth):
                q = mx.exp(
                    mx.take(lp_2d.reshape(-1), tok.reshape(1).astype(mx.int32))
                )
                mx.eval(q, tok)  # one host round-trip; see REPORT section 4
                stats.syncs += 1
                q_val = float(q.reshape(-1).tolist()[0])
                run_prod *= q_val
                expected += run_prod
                if cfg.pstep > 0.0 and q_val < cfg.pstep:
                    break
                floor = _break_even_floor(cfg, ctl, j + 1, expected)
                if run_prod < floor:
                    break

            if not normalised and use_sl:
                r2._refresh_shortlist(mx, sl, head, lp_2d, cfg.sl_k)
            if use_sl:
                mtp_out, head_hidden = mtp(h, tok.reshape(1, 1), embed, chain_cache)
                lp_next = r2._shortlist_logprobs(mx, sl, head, mtp_out[:, -1, :])
                h = head_hidden[:, -1:]
            else:
                logits, head_hidden = lm.mtp_forward(
                    h, tok.reshape(1, 1), chain_cache, return_hidden=True
                )
                h = head_hidden[:, -1:]

        bg._restore_snapshotable(procs, snap)

        if draft_toks:
            state.drafts = mx.concatenate(draft_toks)
            mx.async_eval(state.drafts)
        else:
            state.drafts = mx.zeros((0,), dtype=mx.uint32)
        state.draft_lps = draft_lps
        state.draft_accept_lps = draft_accept_lps

        stats.note(len(draft_toks))
        if cfg.trace_every and stats.cycles % cfg.trace_every == 0:
            logger.info(
                "MTP conf-depth: cycles=%d depth_hist=%s syncs/cycle=%.2f "
                "ceiling=%d p=%s t=%s",
                stats.cycles,
                stats.hist[: cfg.max_depth + 1],
                stats.syncs / max(1, stats.cycles),
                ceiling,
                [round(v, 3) for v in (ctl.p if ctl is not None else [])],
                (
                    {d: round(v, 2) for d, v in sorted(ctl.t.items())}
                    if ctl is not None
                    else {}
                ),
            )

    patched._omlx_mtp_conf_depth = True
    patched._omlx_mtp_shortlist = bool(cfg.shortlist)
    patched._omlx_conf_stats = stats
    return patched


# ---------------------------------------------------------------------------
# controller hand-off
# ---------------------------------------------------------------------------


def _wrap_controller_best(bg) -> bool:
    """Stop oMLX's controller choosing a depth, keep it choosing whether to
    speculate at all.

    ``_DepthController._best`` (batch_generator.py:2169-2178) is the only
    place the post-warmup depth is chosen; warmup and probe bursts assign
    ``self.cur`` directly and are left alone on purpose. Under the flag the
    wrapper collapses the controller's answer to two outcomes: 0, meaning its
    depth-0 escape hatch and the ``_park_mtp_to_standard`` hand-off still fire
    exactly as before, or ``max_depth``, meaning the per-cycle gate picks the
    depth. Everything else the controller does (acceptance EMA, per-depth cost
    EMA, staleness probes, the exit streak) keeps running on real samples,
    because ``observe`` is called with the gate's actual ``k``.
    """
    global _BEST_WRAPPED
    if _BEST_WRAPPED:
        return True
    ctrl = getattr(bg, "_DepthController", None)
    if ctrl is None:
        return False
    if getattr(ctrl, "_omlx_conf_depth", False):
        _BEST_WRAPPED = True
        return True
    orig_best = ctrl._best

    def _best(self):
        d = orig_best(self)
        if not conf_enabled():
            return d
        return 0 if d == 0 else self.max_depth

    ctrl._best = _best
    ctrl._omlx_conf_depth = True
    _BEST_WRAPPED = True
    return True


def install_conf_depth() -> bool:
    """Install the per-cycle confidence gate. Import time (function swap).

    Must run AFTER round-2's ``install_shortlist_draft()``: both own
    ``_chain_next_drafts`` and this one subsumes it.
    """
    global _CHAIN_INSTALLED
    if not conf_enabled():
        return False
    if _CHAIN_INSTALLED:
        return True
    try:
        bg = importlib.import_module(_BG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP conf depth: %s unavailable (%s)", _BG, exc)
        return False
    orig = getattr(bg, "_chain_next_drafts", None)
    if orig is None:
        return False
    if getattr(orig, "_omlx_mtp_conf_depth", False):
        _CHAIN_INSTALLED = True
        return True
    cfg = _GateCfg()
    r2 = None
    if cfg.shortlist:
        try:
            r2 = _load_round2()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTP conf depth: round-2 shortlist helpers: %s", exc)
            r2 = None
        if r2 is None:
            logger.warning(
                "MTP conf depth: OMLX_MTP_SHORTLIST_DRAFT=1 but %s could not be "
                "loaded; drafting on the full vocabulary",
                _R2,
            )
            cfg.shortlist = False
    # fall back to the pre-shortlist stock function when unwinding
    base = getattr(bg, "_omlx_mtp_shortlist_orig", None) or orig
    try:
        bg._chain_next_drafts = _make_gated_chain(bg, base, r2, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP conf depth not installed: %s", exc)
        return False
    if not _wrap_controller_best(bg):
        logger.warning(
            "MTP conf depth: _DepthController not found; the gate will run "
            "under whatever depth the stock policy picks"
        )
    bg._omlx_mtp_conf_orig = orig
    _CHAIN_INSTALLED = True
    logger.info(
        "MTP conf depth installed (pmin=%.3f max_depth=%d min_depth=%d "
        "adapt=%s pstep=%.3f shortlist=%s)",
        cfg.pmin,
        cfg.max_depth,
        cfg.min_depth,
        cfg.adapt,
        cfg.pstep,
        cfg.shortlist,
    )
    return True


def install_conf_depth_ceiling(model: Any = None) -> bool:
    """Raise the chain depth marker to OMLX_MTP_CONF_MAX_DEPTH. AFTER load.

    ``_resolve_mtp_chain_depth`` (batch_generator.py:1609-1628) reads
    ``_omlx_mtp_depth`` off the model and clamps it to 8; that value becomes
    ``_MtpState.depth``, ``_DepthController.max_depth`` and the width of the
    per-depth stats arrays. Without this the gate can never draft past the
    load-time depth of 3.
    """
    global _CEILING_INSTALLED
    if not conf_enabled() or model is None:
        return False
    if _CEILING_INSTALLED:
        return True
    want = max(1, min(8, _envint("OMLX_MTP_CONF_MAX_DEPTH", 5)))
    touched = 0
    for cand in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    ):
        if cand is None:
            continue
        if not getattr(cand, "_omlx_mtp_chain", False):
            continue
        have = int(getattr(cand, "_omlx_mtp_depth", 1) or 1)
        if have >= want:
            continue
        cand._omlx_mtp_depth = want
        touched += 1
    if not touched:
        logger.info("MTP conf depth ceiling: nothing to raise (already >= %d)", want)
        return False
    _CEILING_INSTALLED = True
    logger.info("MTP conf depth ceiling raised to %d on %d module(s)", want, touched)
    return True


def uninstall() -> bool:
    """Restore whatever owned ``_chain_next_drafts`` before this module."""
    global _CHAIN_INSTALLED
    try:
        bg = importlib.import_module(_BG)
    except Exception:  # noqa: BLE001
        return False
    prev = getattr(bg, "_omlx_mtp_conf_orig", None)
    if prev is None:
        return False
    bg._chain_next_drafts = prev
    _CHAIN_INSTALLED = False
    return True


def gate_stats() -> Optional[dict]:
    """Chosen-depth histogram of the live gate. Read-only."""
    try:
        bg = importlib.import_module(_BG)
    except Exception:  # noqa: BLE001
        return None
    st = getattr(getattr(bg, "_chain_next_drafts", None), "_omlx_conf_stats", None)
    if st is None:
        return None
    return {
        "cycles": st.cycles,
        "hist": list(st.hist),
        "syncs_per_cycle": st.syncs / max(1, st.cycles),
    }


def install(model: Any = None) -> bool:
    done = install_conf_depth()
    done |= install_conf_depth_ceiling(model)
    return bool(done)


__all__ = [
    "install",
    "install_conf_depth",
    "install_conf_depth_ceiling",
    "uninstall",
    "gate_stats",
    "conf_enabled",
]
