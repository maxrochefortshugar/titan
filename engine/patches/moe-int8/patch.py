# SPDX-License-Identifier: Apache-2.0
"""oMLX-style runtime patch: route the sorted routed-expert prefill matmul of
Qwen3.8-Flash-Next (``qwen4_exp``) through the int8 x int4 NAX kernel.

Two hooks, mirroring how oMLX layers its own MoE patches:

* ``SwitchGLU.__call__`` is wrapped only to record the token count and top-k of
  the call in progress, so the ``min_tokens`` floor can be expressed in tokens.
  It delegates straight to whatever ``__call__`` is installed, so it composes
  with ``omlx.patches.qwen35_moe_gate_up`` (which replaces gate/up with a fused
  ``gate_up_proj``) instead of fighting it.
* ``mx.gather_qmm`` is wrapped the way ``omlx/patches/m5_gather_qmm.py`` wraps
  it.  The previous implementation is captured as the fallback, so installing
  after oMLX's M5 reroute keeps that reroute for everything this kernel does
  not take.

Routed only when ALL of: ``sorted_indices=True``, ``rhs_indices`` present and
``lhs_indices`` absent, ``transpose=True``, affine 4-bit with ``group_size=64``,
bf16 activations of shape ``[rows, 1, K]``, ``K % 64 == 0``, ``N % 256 == 0``
(or a smaller column tile that divides N), and ``tokens >= min_tokens``.
Everything else - every decode shape, every MTP verify shape, the 5/6/8-bit
tensors - falls through untouched.

    OMLX_MOE_INT8_PREFILL=1      enable (default off; this is opt-in)
    OMLX_MOE_INT8_MIN_TOKENS=512 token floor for the chunk (default 512)
    OMLX_MOE_INT8_MIN_ROWS=4096  routed-row floor used when the token count is
                                 unknown (default 4096)

Memory: each routed weight tensor gets [E, G, N] scale, folded-bias and
nibble-sum tables - three arrays the size of the existing ``scales``, i.e.
~157 MB per fused gate_up and ~79 MB per down projection, ~11 GB across the
48 MoE layers of Flash-Next.  ``warmup(model)`` builds them up front;
``clear_cache()`` drops them.  Nothing under /Applications is modified.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import mlx.core as mx

if str(Path(__file__).resolve().parent) not in sys.path:      # standalone module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from . import kernel as _k                                # imported as a package
except ImportError:
    import kernel as _k

logger = logging.getLogger(__name__)

_original_gather_qmm = None
_original_switchglu_call = None
_ctx = {"tokens": None}
_stats = {"routed": 0, "fallback": 0}


def _enabled() -> bool:
    return os.environ.get("OMLX_MOE_INT8_PREFILL", "0") == "1"


def _min_tokens() -> int:
    return int(os.environ.get("OMLX_MOE_INT8_MIN_TOKENS", "512"))


def _min_rows() -> int:
    return int(os.environ.get("OMLX_MOE_INT8_MIN_ROWS", "4096"))


def _arg(args, kwargs, pos, name, default=None):
    if len(args) > pos:
        return args[pos]
    return kwargs.get(name, default)


def _should_route(x, w, args, kwargs) -> bool:
    if not _enabled() or not kwargs.get("sorted_indices"):
        return False
    # positional layout after (x, w): scales, biases, lhs_indices, rhs_indices,
    # transpose, group_size, bits, mode
    scales = _arg(args, kwargs, 0, "scales")
    biases = _arg(args, kwargs, 1, "biases")
    lhs = _arg(args, kwargs, 2, "lhs_indices")
    rhs = _arg(args, kwargs, 3, "rhs_indices")
    transpose = _arg(args, kwargs, 4, "transpose", True)
    group_size = _arg(args, kwargs, 5, "group_size", 64)
    bits = _arg(args, kwargs, 6, "bits", 4)
    mode = _arg(args, kwargs, 7, "mode", "affine")
    if lhs is not None or mode != "affine":
        return False
    # gate_up-only mode: skip the down projections (K == intermediate 640) to save their tables
    if os.environ.get("OMLX_MOE_INT8_SKIP_DOWN", "0") == "1" and x.shape[-1] <= 1024:
        return False
    tokens = _ctx.get("tokens")
    if tokens is not None:
        if tokens < _min_tokens():
            return False
    elif x.ndim < 1 or x.shape[0] < _min_rows():
        return False
    return _k.supported(x, w, scales, biases, rhs, transpose, group_size, bits)


def _gather_qmm(x, w, *args, **kwargs):
    if _should_route(x, w, args, kwargs):
        try:
            out = _k.gather_qmm_sorted(
                x, w,
                _arg(args, kwargs, 0, "scales"),
                _arg(args, kwargs, 1, "biases"),
                _arg(args, kwargs, 3, "rhs_indices"),
                group_size=_arg(args, kwargs, 5, "group_size", 64),
                bits=_arg(args, kwargs, 6, "bits", 4),
                fallback=_original_gather_qmm,
            )
            _stats["routed"] += 1
            return out
        except Exception as exc:                       # noqa: BLE001
            logger.warning("moe int8 prefill kernel failed, falling back: %s", exc)
    _stats["fallback"] += 1
    return _original_gather_qmm(x, w, *args, **kwargs)


_gather_qmm._omlx_moe_int8 = True


def _switchglu_call(self, x, indices, *a, **kw):
    prev = _ctx.get("tokens")
    try:
        _ctx["tokens"] = int(indices.size // indices.shape[-1]) if indices.ndim else None
    except Exception:                                  # noqa: BLE001
        _ctx["tokens"] = None
    try:
        return _original_switchglu_call(self, x, indices, *a, **kw)
    finally:
        _ctx["tokens"] = prev


def install() -> bool:
    """Idempotent. Returns True when this call installed the hooks."""
    global _original_gather_qmm, _original_switchglu_call
    if getattr(mx.gather_qmm, "_omlx_moe_int8", False):
        return False
    _original_gather_qmm = mx.gather_qmm
    mx.gather_qmm = _gather_qmm
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
        if not getattr(SwitchGLU, "_omlx_moe_int8", False):
            _original_switchglu_call = SwitchGLU.__call__
            SwitchGLU.__call__ = _switchglu_call
            SwitchGLU._omlx_moe_int8 = True
    except Exception as exc:                           # noqa: BLE001
        logger.debug("SwitchGLU token hook not installed (%s); row floor only", exc)
    logger.info("moe int8 prefill kernel installed (enabled=%s)", _enabled())
    return True


def uninstall() -> None:
    global _original_gather_qmm, _original_switchglu_call
    if _original_gather_qmm is not None:
        mx.gather_qmm = _original_gather_qmm
        _original_gather_qmm = None
    if _original_switchglu_call is not None:
        from mlx_lm.models.switch_layers import SwitchGLU
        SwitchGLU.__call__ = _original_switchglu_call
        SwitchGLU._omlx_moe_int8 = False
        _original_switchglu_call = None


def clear_cache() -> None:
    _k.clear_cache()


def stats() -> dict:
    return dict(_stats)


def warmup(model) -> int:
    """Pre-build the per-tensor tables so the first prefill chunk is not charged."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    n = 0
    total = 0
    for _, m in model.named_modules():
        if not isinstance(m, QuantizedSwitchLinear):
            continue
        if (m.bits, m.group_size, getattr(m, "mode", "affine")) != (4, 64, "affine"):
            continue
        if _k.pick_cfg(m["scales"].shape[1]) is None:
            continue
        if os.environ.get("OMLX_MOE_INT8_SKIP_DOWN", "0") == "1" and m["weight"].shape[-1] * 32 // m.bits <= 1024:
            continue
        tabs = _k._prepared(m["weight"], m["scales"], m["biases"])
        total += sum(t.nbytes for t in tabs)
        n += 1
    logger.info("moe int8 prefill: %d expert tensors prepared, %.2f GB of tables",
                n, total / 1e9)
    return n
