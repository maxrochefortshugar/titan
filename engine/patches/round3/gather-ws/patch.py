# SPDX-License-Identifier: Apache-2.0
"""oMLX-style runtime patch: route the sorted routed-expert prefill matmul of
Qwen3.8-Flash-Next (``qwen4_exp``) through the weight-stationary bf16 kernel.

Two hooks, the same two ``../../moe-int8/patch.py`` uses, for the same reasons:

* ``SwitchGLU.__call__`` is wrapped only to record the token count of the call
  in progress, so the ``min_tokens`` floor can be expressed in tokens.  It
  delegates to whatever ``__call__`` is installed, so it composes with
  ``omlx.patches.qwen35_moe_gate_up`` (fused ``gate_up_proj``) rather than
  fighting it, and chains on top of the int8 patch's identical hook when that
  one is already present.
* ``mx.gather_qmm`` is wrapped the way ``omlx/patches/m5_gather_qmm.py`` wraps
  it.  Whatever was installed before is captured as the fallback, so installing
  after oMLX's M5 reroute keeps that reroute for everything this kernel does
  not take.

Routed only when ALL of: ``sorted_indices=True``, ``rhs_indices`` present and
``lhs_indices`` absent, ``transpose=True``, affine 4-bit ``group_size=64``,
bf16 activations of shape ``[rows, 1, K]``, ``K % 64 == 0``, ``N % 256 == 0``
(or a smaller column block that divides N), and ``tokens >= min_tokens``.
Everything else falls through untouched: every decode shape, every MTP verify
shape, the 5/6/8-bit tensors.

Precedence against the int8 patch (``OMLX_MOE_INT8_PREFILL=1``).  The int8
kernel is faster wherever it applies (1.42x against this kernel's 1.14x at
T=2048), so **int8 wins**.  Two cases:

* int8 installed first, this patch second: this wrapper sits on top, sees the
  ``_omlx_moe_int8`` marker on its own fallback, and declines any call the int8
  kernel's own ``supported()`` accepts, so the call falls through to it.
* this patch installed first, int8 second: the int8 wrapper is on top and this
  one is its fallback, so int8 already takes what it wants and hands the rest
  down.  No coordination needed.

If the int8 module cannot be imported the marker check alone is used and every
supported call is handed down while int8 is enabled.

    OMLX_MOE_GATHER_WS=1              enable (default off; opt-in)
    OMLX_MOE_GATHER_WS_MIN_TOKENS=2048   token floor (default 2048; at 1024
                                      tokens and below the kernel is a wash on
                                      gate_up and a loss on down)
    OMLX_MOE_GATHER_WS_MIN_ROWS=8192  routed-row floor when the token count is
                                      unknown

Memory: none.  No per-tensor tables are built, so there is nothing to warm up
and nothing to evict.  Install at import time (it monkeypatches module
functions, not model instances); it does not need the model to be loaded.
Nothing under /Applications is modified.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

import mlx.core as mx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from . import kernel as _k
except ImportError:
    import kernel as _k

logger = logging.getLogger(__name__)

_original_gather_qmm = None
_original_switchglu_call = None
_switchglu_hooked_here = False
_ctx = {"tokens": None}
_stats = {"routed": 0, "fallback": 0, "to_int8": 0}
_int8_kernel = None
_int8_probed = False


def _enabled() -> bool:
    return os.environ.get("OMLX_MOE_GATHER_WS", "0") == "1"


def _min_tokens() -> int:
    return int(os.environ.get("OMLX_MOE_GATHER_WS_MIN_TOKENS", "2048"))


def _min_rows() -> int:
    return int(os.environ.get("OMLX_MOE_GATHER_WS_MIN_ROWS", "8192"))


def _int8_active() -> bool:
    """True when the int8 patch is installed below us and switched on."""
    return (os.environ.get("OMLX_MOE_INT8_PREFILL", "0") == "1"
            and getattr(_original_gather_qmm, "_omlx_moe_int8", False))


def _int8_module():
    """The int8 kernel module, imported read-only by path.  None if absent."""
    global _int8_kernel, _int8_probed
    if _int8_probed:
        return _int8_kernel
    _int8_probed = True
    path = _HERE.parent.parent / "moe-int8" / "kernel.py"
    try:
        spec = importlib.util.spec_from_file_location("_moe_int8_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _int8_kernel = mod
    except Exception as exc:                            # noqa: BLE001
        logger.debug("int8 kernel not importable for precedence check (%s)", exc)
        _int8_kernel = None
    return _int8_kernel


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
    tokens = _ctx.get("tokens")
    if tokens is not None:
        if tokens < _min_tokens():
            return False
    elif x.ndim < 1 or x.shape[0] < _min_rows():
        return False
    if not _k.supported(x, w, scales, biases, rhs, transpose, group_size, bits):
        return False
    if _int8_active():
        m = _int8_module()
        if m is None or m.supported(x, w, scales, biases, rhs, transpose, group_size, bits):
            _stats["to_int8"] += 1
            return False
    return True


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
        except Exception as exc:                        # noqa: BLE001
            logger.warning("moe gather-ws kernel failed, falling back: %s", exc)
    _stats["fallback"] += 1
    return _original_gather_qmm(x, w, *args, **kwargs)


_gather_qmm._omlx_moe_gather_ws = True


def _switchglu_call(self, x, indices, *a, **kw):
    prev = _ctx.get("tokens")
    try:
        _ctx["tokens"] = int(indices.size // indices.shape[-1]) if indices.ndim else None
    except Exception:                                   # noqa: BLE001
        _ctx["tokens"] = None
    try:
        return _original_switchglu_call(self, x, indices, *a, **kw)
    finally:
        _ctx["tokens"] = prev


def install() -> bool:
    """Idempotent. Returns True when this call installed the hooks."""
    global _original_gather_qmm, _original_switchglu_call, _switchglu_hooked_here
    if getattr(mx.gather_qmm, "_omlx_moe_gather_ws", False):
        return False
    _original_gather_qmm = mx.gather_qmm
    mx.gather_qmm = _gather_qmm
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
        # chain on top of whatever is installed (the int8 patch's hook included);
        # ours delegates to it, so both token contexts stay correct.
        if not getattr(SwitchGLU, "_omlx_moe_gather_ws", False):
            _original_switchglu_call = SwitchGLU.__call__
            SwitchGLU.__call__ = _switchglu_call
            SwitchGLU._omlx_moe_gather_ws = True
            _switchglu_hooked_here = True
    except Exception as exc:                            # noqa: BLE001
        logger.debug("SwitchGLU token hook not installed (%s); row floor only", exc)
    logger.info("moe gather-ws prefill kernel installed (enabled=%s)", _enabled())
    return True


def uninstall() -> None:
    global _original_gather_qmm, _original_switchglu_call, _switchglu_hooked_here
    if _original_gather_qmm is not None:
        mx.gather_qmm = _original_gather_qmm
        _original_gather_qmm = None
    if _switchglu_hooked_here and _original_switchglu_call is not None:
        from mlx_lm.models.switch_layers import SwitchGLU
        SwitchGLU.__call__ = _original_switchglu_call
        SwitchGLU._omlx_moe_gather_ws = False
        _original_switchglu_call = None
        _switchglu_hooked_here = False


def clear_cache() -> None:
    _k.clear_cache()


def stats() -> dict:
    return dict(_stats)


def warmup(model=None) -> int:
    """Nothing to precompute; present so bootstrap can call it uniformly."""
    return 0
