# SPDX-License-Identifier: Apache-2.0
"""MTP decode patches for Qwen3.8-Flash-Next (qwen4_exp) on oMLX 0.7.0.dev2.

Three independent, env-gated installs. Each returns True only when it
actually changed something, and leaves the stock path untouched otherwise.

  install_shortlist_draft()   OMLX_MTP_SHORTLIST_DRAFT=1   import time
      Replaces the full-vocabulary lm_head pass on draft chain steps
      2..depth with a top-K shortlist taken from step 1's full-vocabulary
      row. Stock code: omlx/patches/mlx_lm_mtp/batch_generator.py:2300-2422
      (_chain_next_drafts), whose per-step logits come from
      LanguageModel.mtp_forward -> self.lm_head(...) at
      .../vendor/mlx_vlm/models/qwen4_exp/language.py:3101-3126.

  install_verify_gate_up()    OMLX_MTP_VERIFY_GATE_UP=1    post-load
      Guarantees the MTP verify MoE path uses the fused gate_up_proj
      instead of separate gate/up gathers. Stock code:
      mlx_vlm/models/qwen3_5_moe/language.py:15-32
      (_target_verify_switch_glu).

  install_depth_trace()       OMLX_MTP_DEPTH_TRACE=1       import time
      Read-only instrumentation of the existing depth controller
      (batch_generator.py:1880-2180) so the break-even can be checked
      against measured per-depth cycle costs on the workbench.

Nothing here touches the verify forward's numerics: the verify pass keeps
its exact full-vocabulary logits, so the accepted token stream is
unchanged under greedy decoding and remains distributionally exact under
sampling (the shortlisted proposal density q is what the Leviathan/Chen
ratio is given).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_BG = "omlx.patches.mlx_lm_mtp.batch_generator"
_NEG_INF = -3.0e38

_SHORTLIST_INSTALLED = False
_GATE_UP_INSTALLED = False
_TRACE_INSTALLED = False


def _envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# (a) shortlist drafter
# ---------------------------------------------------------------------------


class _Shortlist:
    """Per-sequence shortlist state: candidate ids and the gathered head rows."""

    __slots__ = ("ids", "weight", "scales", "biases", "age", "neg")

    def __init__(self):
        self.ids = None
        self.weight = None
        self.scales = None
        self.biases = None
        self.age = 0
        self.neg = None


def _head_of(model: Any):
    """The lm_head (or tied embedding) the MTP head samples through."""
    lm = model
    for attr in ("language_model", "_language_model"):
        inner = getattr(lm, attr, None)
        if inner is not None and hasattr(inner, "mtp_forward"):
            lm = inner
            break
    args = getattr(lm, "args", None)
    if args is not None and getattr(args, "tie_word_embeddings", False):
        return None, lm  # tied heads are not shortlisted (embedding, not linear)
    return getattr(lm, "lm_head", None), lm


def _is_affine_qlinear(head) -> bool:
    import mlx.nn as nn

    return (
        isinstance(head, nn.QuantizedLinear)
        and getattr(head, "mode", "affine") == "affine"
        and getattr(head, "biases", None) is not None
        and "bias" not in head
    )


def _refresh_shortlist(mx, state_sl: _Shortlist, head, logits_1d, k: int) -> None:
    """Pick the top-k candidates and gather their packed head rows."""
    v = int(logits_1d.shape[-1])
    k = max(16, min(int(k), v))
    ids = mx.argpartition(logits_1d, kth=v - k, axis=-1)[..., -k:]
    ids = ids.reshape(-1).astype(mx.int32)
    state_sl.ids = ids
    state_sl.weight = head.weight[ids]
    state_sl.scales = head.scales[ids]
    state_sl.biases = head.biases[ids]
    state_sl.age = 0
    if state_sl.neg is None or state_sl.neg.shape[-1] != v:
        state_sl.neg = mx.full((1, v), _NEG_INF, dtype=mx.float32)
    mx.eval(state_sl.ids, state_sl.weight, state_sl.scales, state_sl.biases)


def _shortlist_logprobs(mx, state_sl: _Shortlist, head, hidden_2d):
    """Full-vocabulary log-prob row of the shortlist-restricted proposal.

    Positions outside the shortlist carry -inf, which is exactly the
    proposal density of a drafter that can only emit shortlist tokens.
    Downstream acceptance (greedy compare, or the Leviathan/Chen ratio and
    residual) therefore stays correct with no other change.
    """
    small = mx.quantized_matmul(
        hidden_2d,
        state_sl.weight,
        state_sl.scales,
        state_sl.biases,
        transpose=True,
        group_size=head.group_size,
        bits=head.bits,
    ).astype(mx.float32)
    small = small - mx.logsumexp(small, axis=-1, keepdims=True)
    return mx.put_along_axis(state_sl.neg, state_sl.ids[None, :], small, axis=-1)


def _make_chain_next_drafts(bg, orig):
    import mlx.core as mx

    shortlist_k = _envint("OMLX_MTP_SHORTLIST_K", 2048)
    from_step = _envint("OMLX_MTP_SHORTLIST_FROM_STEP", 1)
    if from_step not in (0, 1):
        logger.warning(
            "OMLX_MTP_SHORTLIST_FROM_STEP=%d unsupported, clamping to 1", from_step
        )
        from_step = 1
    refresh_every = max(1, _envint("OMLX_MTP_SHORTLIST_REFRESH", 1))

    def patched(gen_batch, state, hidden_rows, committed, prev_buf):
        model = gen_batch.model
        if bg._dspark_host(model) is not None:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        head, lm = _head_of(model)
        if head is None or not _is_affine_qlinear(head):
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)
        mtp = getattr(lm, "get_mtp_module", None)
        mtp = mtp() if callable(mtp) else None
        if mtp is None:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        sampler = bg._resolve_draft_sampler(gen_batch, state)
        procs = bg._proc_list(gen_batch)
        depth = state.controller.cur if state.controller is not None else state.depth
        if depth <= from_step:
            # Only the fold's own logits are consumed: nothing to shortlist.
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        sl = getattr(state, "_omlx_shortlist", None)
        if sl is None:
            sl = _Shortlist()
            state._omlx_shortlist = sl
        # Step 0 keeps the full vocabulary whenever from_step >= 1, and also
        # on the cycles where the shortlist has to be rebuilt.
        sl.age += 1
        full_step0 = from_step >= 1 or sl.ids is None or sl.age >= refresh_every

        head_prenorm = getattr(model, "_omlx_mtp_head_prenorm", False) or getattr(
            getattr(model, "_language_model", None), "_omlx_mtp_head_prenorm", False
        )
        if bg._HEAD_HIDDEN_POST_NORM and not head_prenorm and hidden_rows.ndim == 3:
            hidden_rows = bg._trunk_norm_module(model)(hidden_rows)

        begin = getattr(model, "mtp_begin_cycle", None) or getattr(
            getattr(model, "_language_model", None), "mtp_begin_cycle", None
        )
        if begin is not None:
            begin(state.mtp_cache, depth)

        n = committed.shape[0]
        if full_step0:
            logits, head_hidden = lm.mtp_forward(
                hidden_rows,
                committed.reshape(1, n),
                state.mtp_cache,
                return_hidden=True,
                logits_keep=1,
            )
            lp_next = None
        else:
            mtp_out, head_hidden = mtp(
                hidden_rows, committed.reshape(1, n), lm.model.embed_tokens,
                state.mtp_cache,
            )
            logits = None
            lp_next = _shortlist_logprobs(mx, sl, head, mtp_out[:, -1, :])
        state.hist_offset += int(n)

        draft_toks: List[Any] = []
        draft_lps: List[Any] = []
        draft_accept_lps: List[Any] = []

        chain_prefix = committed[-1:]
        h = head_hidden[:, -1:]
        chain_cache = state.mtp_cache
        if state.head_clone and depth > 1:
            chain_cache = bg._clone_mtp_head_cache(state.mtp_cache)

        snap = bg._snap_snapshotable(procs)
        embed = lm.model.embed_tokens

        for j in range(depth):
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
            if j + 1 == depth:
                break
            if not normalised:
                # this row came from the full vocabulary: rebuild the shortlist
                _refresh_shortlist(mx, sl, head, lp_2d, shortlist_k)
            mtp_out, head_hidden = mtp(h, tok.reshape(1, 1), embed, chain_cache)
            h = head_hidden[:, -1:]
            lp_next = _shortlist_logprobs(mx, sl, head, mtp_out[:, -1, :])

        bg._restore_snapshotable(procs, snap)

        if draft_toks:
            state.drafts = mx.concatenate(draft_toks)
            mx.async_eval(state.drafts)
        else:
            state.drafts = mx.zeros((0,), dtype=mx.uint32)
        state.draft_lps = draft_lps
        state.draft_accept_lps = draft_accept_lps

    patched._omlx_mtp_shortlist = True
    return patched


def install_shortlist_draft() -> bool:
    """Route draft chain steps >= OMLX_MTP_SHORTLIST_FROM_STEP through a
    top-K shortlist head. Install at import time (module function swap)."""
    global _SHORTLIST_INSTALLED
    if not _enabled("OMLX_MTP_SHORTLIST_DRAFT"):
        return False
    if _SHORTLIST_INSTALLED:
        return True
    try:
        bg = importlib.import_module(_BG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP shortlist drafter: %s unavailable (%s)", _BG, exc)
        return False
    orig = getattr(bg, "_chain_next_drafts", None)
    if orig is None:
        return False
    if getattr(orig, "_omlx_mtp_shortlist", False):
        _SHORTLIST_INSTALLED = True
        return True
    try:
        bg._chain_next_drafts = _make_chain_next_drafts(bg, orig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP shortlist drafter not installed: %s", exc)
        return False
    bg._omlx_mtp_shortlist_orig = orig
    _SHORTLIST_INSTALLED = True
    logger.info(
        "MTP shortlist drafter installed (K=%d, from step %d, refresh %d)",
        _envint("OMLX_MTP_SHORTLIST_K", 2048),
        _envint("OMLX_MTP_SHORTLIST_FROM_STEP", 1),
        max(1, _envint("OMLX_MTP_SHORTLIST_REFRESH", 1)),
    )
    return True


# ---------------------------------------------------------------------------
# (b) fused gate_up in MTP verify
# ---------------------------------------------------------------------------


def verify_gate_up_status(model: Any = None) -> dict:
    """Report whether the MTP verify MoE path is fused. Read-only."""
    status = {"helper_patched": False, "fused_layers": 0, "unfused_layers": 0}
    try:
        module = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")
    except Exception:  # noqa: BLE001
        return status
    status["helper_patched"] = bool(
        getattr(module, "_omlx_gate_up_fused_verify", False)
    )
    if model is None:
        return status
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
    except Exception:  # noqa: BLE001
        return status
    for _, m in model.named_modules():
        if isinstance(m, SwitchGLU):
            if getattr(m, "gate_up_proj", None) is not None:
                status["fused_layers"] += 1
            else:
                status["unfused_layers"] += 1
    return status


def install_verify_gate_up(model: Any = None) -> bool:
    """Make the MTP verify MoE helper fused-aware. Install AFTER load.

    Returns False when oMLX's own gate/up fusion already did this, which is
    the expected outcome on a stock 0.7.0.dev2 server.
    """
    global _GATE_UP_INSTALLED
    if not _enabled("OMLX_MTP_VERIFY_GATE_UP"):
        return False
    if _GATE_UP_INSTALLED:
        return True
    status = verify_gate_up_status(model)
    if status["helper_patched"]:
        logger.info(
            "MTP verify gate_up: already fused by oMLX (%d fused / %d unfused "
            "SwitchGLU)", status["fused_layers"], status["unfused_layers"]
        )
        return False
    if model is not None and status["fused_layers"] == 0:
        logger.info("MTP verify gate_up: no fused SwitchGLU to route to")
        return False
    try:
        from omlx.patches.qwen35_moe_gate_up import _make_patched_target_verify

        module = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")
        orig = getattr(module, "_target_verify_switch_glu", None)
        if orig is None:
            return False
        module._target_verify_switch_glu = _make_patched_target_verify(orig)
        module._omlx_gate_up_fused_verify = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP verify gate_up patch failed: %s", exc)
        return False
    _GATE_UP_INSTALLED = True
    logger.info("MTP verify gate_up: fused-aware helper installed")
    return True


# ---------------------------------------------------------------------------
# (c) depth-controller instrumentation
# ---------------------------------------------------------------------------


def install_depth_trace() -> bool:
    """Log the depth controller's measured per-depth cost and acceptance.

    oMLX already implements adaptive depth (batch_generator.py:1880-2180):
    ``_score(d) = (1 + sum_j prod_{i<=j} p_i) / t[d]`` with 3% hysteresis,
    which is the break-even test. This install only surfaces the inputs so
    the choice can be audited. Install at import time.
    """
    global _TRACE_INSTALLED
    if not _enabled("OMLX_MTP_DEPTH_TRACE"):
        return False
    if _TRACE_INSTALLED:
        return True
    try:
        bg = importlib.import_module(_BG)
    except Exception:  # noqa: BLE001
        return False
    ctrl = None
    for name in dir(bg):
        obj = getattr(bg, name)
        if isinstance(obj, type) and hasattr(obj, "_score") and hasattr(obj, "observe"):
            ctrl = obj
            break
    if ctrl is None or getattr(ctrl, "_omlx_depth_trace", False):
        return False
    every = max(1, _envint("OMLX_MTP_DEPTH_TRACE_EVERY", 64))
    orig_observe = ctrl.observe

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        orig_observe(self, used, accepted, cycle_ms, time_sample=time_sample)
        if self.cycles % every:
            return
        marg = self._marginal_est()
        logger.info(
            "MTP depth: cur=%d cycles=%d p=%s t=%s scores=%s marginal=%.2fms",
            self.cur,
            self.cycles,
            [round(v, 3) for v in self.p],
            {d: round(v, 2) for d, v in sorted(self.t.items())},
            {d: round(self._score(d), 5) for d in range(0, self.max_depth + 1)},
            marg,
        )

    ctrl.observe = observe
    ctrl._omlx_depth_trace = True
    _TRACE_INSTALLED = True
    logger.info("MTP depth trace installed (every %d cycles)", every)
    return True


# ---------------------------------------------------------------------------


def install(model: Any = None) -> bool:
    """Apply every enabled patch. True if at least one changed something."""
    done = False
    done |= install_shortlist_draft()
    done |= install_depth_trace()
    done |= install_verify_gate_up(model)
    return bool(done)


__all__ = [
    "install",
    "install_shortlist_draft",
    "install_verify_gate_up",
    "install_depth_trace",
    "verify_gate_up_status",
]
