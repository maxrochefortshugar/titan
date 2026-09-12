# SPDX-License-Identifier: Apache-2.0
"""Extend the top_k=10 weighted-sum kernel to the MTP target-verify layout.

``kernels/round2/wsum10`` fuses ``_scatter_unsort`` + ``(y * scores).sum(-2)``
into one launch, but its gate refuses ``target_verify=True``
(``round2/wsum10/kernel.py`` ``_should_route``), so the MTP verify pass runs
the stock three-pass tail on every decode cycle.

The verify layout is different but easier: ``_target_verify_switch_glu``
(``mlx_vlm/models/qwen3_5_moe/language.py:15-32``) calls the three
``SwitchLinear`` projections with ``sorted_indices=False`` and returns
``[B, T, k, D]`` in token order, so the fused kernel runs with an identity
permutation and no ``inv_order`` gather at all.

This module wraps ``Qwen3_5MoeSparseMoeBlock.__call__`` once more and claims
only ``target_verify=True`` with ``T`` inside ``[OMLX_WSUM_VERIFY_MIN_T,
OMLX_WSUM_VERIFY_MAX_T]`` (default 2..15, the MTP verify widths).  Everything
else, prefill included, is handed to whatever call was installed before this
one, so it composes with round2/wsum10 in either order.  Nothing under
round2/wsum10 is modified: its ``weighted_sum`` is imported by path.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)

ENV_ENABLE = "OMLX_WSUM_TOPK10_VERIFY"
ENV_MODE = "OMLX_WSUM_VERIFY_MODE"        # clone (default) / ops / fast
ENV_MIN_T = "OMLX_WSUM_VERIFY_MIN_T"
ENV_MAX_T = "OMLX_WSUM_VERIFY_MAX_T"
ENV_TOPK = "OMLX_WSUM_VERIFY_K"

_WSUM10 = os.path.expanduser("~/inference-server/kernels/round2/wsum10/kernel.py")
_WSUM10_MODULE = "omlx_wsum_topk10_kernel"   # the name round2/wsum10/patch.py uses

_CFG = {"mode": "clone", "min_t": 2, "max_t": 15, "topk": (10,)}
_PATCHED = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def wsum10():
    """The round2/wsum10 kernel module, loaded by path and shared by name."""
    mod = sys.modules.get(_WSUM10_MODULE)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(_WSUM10_MODULE, _WSUM10)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_WSUM10_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_tail(y: mx.array, scores: mx.array, mode: str | None = None) -> mx.array:
    """``(y * scores[..., None]).sum(-2)`` for an unsorted [B, T, k, D] y."""
    mode = mode or _CFG["mode"]
    out_shape = (*y.shape[:-2], y.shape[-1])
    return wsum10().weighted_sum(y, None, scores, out_shape, mode=mode)


def _supported_topk() -> tuple[int, ...]:
    raw = os.environ.get(ENV_TOPK, "10")
    out = []
    for piece in raw.replace(",", " ").split():
        try:
            out.append(int(piece))
        except ValueError:
            pass
    return tuple(out) or (10,)


def _target_verify_arg(args, kwargs) -> bool:
    if bool(kwargs.get("target_verify", False)):
        return True
    return bool(args and isinstance(args[0], bool) and args[0])


def _should_route(self, x: mx.array) -> bool:
    if x.ndim != 3:
        return False
    t = int(x.shape[1])
    if t < _CFG["min_t"] or t > _CFG["max_t"]:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if getattr(self, "sharding_group", None) is not None:
        return False
    if int(getattr(self, "top_k", 0)) not in _CFG["topk"]:
        return False
    switch_mlp = getattr(self, "switch_mlp", None)
    return switch_mlp is not None and hasattr(switch_mlp, "down_proj")


def _fast_verify(self, x: mx.array, lang) -> mx.array:
    # Router chain reproduced verbatim from Qwen3_5MoeSparseMoeBlock.__call__
    # so expert sets and score values are bit-identical to the stock path.
    tvl = lang._target_verify_linear
    gates = tvl(self.gate, x, True)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(self, "norm_topk_prob", True):
        scores = scores / scores.sum(axis=-1, keepdims=True)

    y = lang._target_verify_switch_glu(self.switch_mlp, x, inds, True)
    y = verify_tail(y, scores)

    if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
        try:
            shared_y = self.shared_expert(x, True)
        except TypeError:
            shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(tvl(self.shared_expert_gate, x, True)) * shared_y
        y = y + shared_y
    return y


def _make_patched_call(orig_call, lang):
    def patched(self, x: mx.array, *args, **kwargs):
        if not _target_verify_arg(args, kwargs):
            return orig_call(self, x, *args, **kwargs)
        if not _should_route(self, x):
            return orig_call(self, x, *args, **kwargs)
        try:
            return _fast_verify(self, x, lang)
        except Exception:
            logger.warning("MTP verify weighted-sum fast path failed",
                           exc_info=True)
            return orig_call(self, x, *args, **kwargs)

    patched._omlx_wsum_verify = True
    return patched


_FLAG = "_wsum_verify_patched"


def _self_check() -> bool:
    """Unsorted-layout kernel against the stock ops tail at the verify widths."""
    mode = _CFG["mode"]
    k = _CFG["topk"][0]
    for b, t, d in ((1, 2, 256), (1, 4, 2560), (1, 13, 2560)):
        y = mx.random.normal((b, t, k, d)).astype(mx.bfloat16)
        sc = mx.random.uniform(shape=(b, t, k)).astype(mx.bfloat16)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        got = verify_tail(y, sc, mode)
        ref = (y * sc[..., None]).sum(axis=-2)
        mx.eval(got, ref)
        if mode == "clone":
            if not bool(mx.all(got == ref).item()):
                return False
        else:
            g, r = got.astype(mx.float32), ref.astype(mx.float32)
            scale = float(mx.max(mx.abs(r)).item()) + 1e-6
            if float(mx.max(mx.abs(g - r)).item()) > 0.02 * scale:
                return False
    return True


def install() -> bool:
    """Idempotent; returns False and leaves the stock path on any failure.

    Post-load, after round2/wsum10, so that mlx_vlm is already imported by
    oMLX's own compat layer and this wrapper ends up outermost.  It patches a
    class and no instances, so the order against oMLX's patches is free.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get(ENV_ENABLE, "0") != "1":
        return False
    if not mx.metal.is_available():
        return False
    mode = os.environ.get(ENV_MODE, "clone").lower()
    _CFG["mode"] = mode if mode in ("clone", "ops", "fast") else "clone"
    _CFG["min_t"] = max(2, _env_int(ENV_MIN_T, 2))
    _CFG["max_t"] = max(_CFG["min_t"], _env_int(ENV_MAX_T, 15))
    _CFG["topk"] = _supported_topk()
    try:
        if not _self_check():
            logger.warning("MTP verify weighted-sum self-check failed; skipped")
            return False
    except Exception:
        logger.warning("MTP verify weighted-sum self-check raised", exc_info=True)
        return False

    try:
        lang = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("MTP verify weighted-sum: %s", exc)
        return False
    cls = getattr(lang, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or not hasattr(lang, "_target_verify_switch_glu"):
        return False
    if getattr(cls, _FLAG, False):
        _PATCHED = True
        return True
    orig = cls.__call__
    cls.__call__ = _make_patched_call(orig, lang)
    setattr(cls, _FLAG, True)
    setattr(cls, "_wsum_verify_original_call", orig)
    _PATCHED = True
    logger.info("MTP verify weighted-sum installed (mode=%s, T %d..%d, top_k=%s)",
                _CFG["mode"], _CFG["min_t"], _CFG["max_t"], _CFG["topk"])
    return True


__all__ = ["install", "verify_tail", "wsum10"]
