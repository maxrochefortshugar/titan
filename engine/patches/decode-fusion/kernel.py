# SPDX-License-Identifier: Apache-2.0
"""Single-token MoE decode fusion for Qwen3.8-Flash-Next style routed experts.

At T<=2 the routed-expert path is launch-latency bound: on this M5 Max a Metal
dispatch costs ~15-20 us no matter how small it is, and the stock chain issues
gather_qmm(gate_up) -> split -> silu*mul -> gather_qmm(down) -> mul -> sum.
This module replaces that with two JIT Metal kernels:

    h = silu(Wg[e] @ x) * (Wu[e] @ x)         # fused gather gate+up + SiLU*mul
    y = sum_j score_j * (Wd[e_j] @ h_j)       # fused gather down + weighted sum

Both dequantize MLX affine 4-bit weights on the fly and accumulate in fp32.

Enable with ``OMLX_MOE_DECODE_FUSED=1`` and call :func:`apply`.  Anything the
kernels do not cover -- longer sequences, non-affine or non-4-bit weights,
unfused gate/up projections, sharded experts -- falls through to the stock
implementation unchanged.

Prerequisite: the oMLX gate+up fusion (``qwen35_moe_gate_up.py``) must have run,
so ``switch_mlp`` exposes a single ``gate_up_proj`` of shape [E, 2*I, K].  Blocks
with separate ``gate_proj``/``up_proj`` fall back.
"""
from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from fused_kernels import down_wsum, gate_up_silu, router_logits, router_topk

logger = logging.getLogger(__name__)

_PATCHED = False
_ENV = "OMLX_MOE_DECODE_FUSED"
_ENV_MAX_T = "OMLX_MOE_DECODE_FUSED_MAX_TOKENS"
_ENV_ROUTER = "OMLX_MOE_DECODE_FUSED_ROUTER"
_DEFAULT_MAX_T = 2

_TARGETS = (
    ("mlx_lm.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"),
    ("mlx_lm.models.qwen3_5", "SparseMoeBlock"),
    ("mlx_lm.models.qwen3_next", "Qwen3NextSparseMoeBlock"),
    ("mlx_vlm.models.qwen3_5_moe.language", "Qwen3_5MoeSparseMoeBlock"),
    ("moe_ref", "SparseMoeBlock"),
)


# --------------------------------------------------------------- eligibility --
def _switch_linear_ok(sl: Any, bits: int = 4) -> bool:
    if sl is None:
        return False
    if getattr(sl, "mode", None) != "affine" or getattr(sl, "bits", None) != bits:
        return False
    if getattr(sl, "group_size", None) not in (64, 128):
        return False
    if "bias" in sl:
        return False
    w, s, b = sl.get("weight"), sl.get("scales"), sl.get("biases")
    if w is None or s is None or b is None:
        return False
    if w.ndim != 3 or s.ndim != 3 or b.ndim != 3 or w.dtype != mx.uint32:
        return False
    if s.shape != b.shape or s.shape[:2] != w.shape[:2]:
        return False
    return w.shape[2] * 8 == s.shape[2] * sl.group_size


def _eligible(self: Any, x: mx.array, max_t: int) -> bool:
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] > max_t:
        return False
    if getattr(self, "sharding_group", None) is not None:
        return False
    sm = getattr(self, "switch_mlp", None)
    if sm is None:
        return False
    gu = getattr(sm, "gate_up_proj", None)
    dn = getattr(sm, "down_proj", None)
    if not (_switch_linear_ok(gu) and _switch_linear_ok(dn)):
        return False
    if gu.group_size != dn.group_size:
        return False
    E, twoI, KW = gu["weight"].shape
    Ed, D, IW = dn["weight"].shape
    if E != Ed or twoI % 2 != 0:
        return False
    I = twoI // 2
    return KW * 8 == x.shape[-1] and IW * 8 == I and D == x.shape[-1]


# ---------------------------------------------------------------- fast path --
def _router_gate_fusable(gate: Any) -> bool:
    """Quantized affine 4-bit router gate, shaped [E, K] -- the fused path."""
    if getattr(gate, "mode", None) != "affine" or getattr(gate, "bits", None) != 4:
        return False
    if getattr(gate, "group_size", None) not in (64, 128):
        return False
    if "bias" in gate:
        return False
    w, sc, b = gate.get("weight"), gate.get("scales"), gate.get("biases")
    if w is None or sc is None or b is None or w.dtype != mx.uint32:
        return False
    return w.ndim == 2 and sc.shape == b.shape and w.shape[0] % 32 == 0


def fused_routed_mlp(self: Any, x: mx.array) -> mx.array:
    """Router + the two fused expert kernels.  x is [1, T, K]."""
    sm = self.switch_mlp
    gu, dn = sm.gate_up_proj, sm.down_proj
    gs = gu.group_size
    k = self.top_k
    T, K = x.shape[-2], x.shape[-1]
    xf = x.reshape(T, K)

    norm = bool(getattr(self, "norm_topk_prob", True))
    if (os.environ.get(_ENV_ROUTER, "0") == "1"
            and _router_gate_fusable(self.gate)):
        lg = router_logits(xf, self.gate["weight"], self.gate["scales"],
                           self.gate["biases"], self.gate.group_size)
        idx, sc = router_topk(lg, k, norm)
    else:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if norm:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        idx = inds.reshape(T, k).astype(mx.uint32)
        sc = scores.reshape(T, k).astype(mx.float32)

    h = gate_up_silu(xf, gu["weight"], gu["scales"], gu["biases"], idx, gs)
    y = down_wsum(h, dn["weight"], dn["scales"], dn["biases"], idx, sc, gs,
                  out_dtype=x.dtype)
    return y.reshape(1, T, K)


def _make_patched(orig_call: Callable[..., mx.array]):
    def patched(self, x: mx.array, *args, **kwargs):
        # One shape test on the hot path before touching env or module state.
        if x.ndim != 3 or x.shape[-2] > _max_tokens():
            return orig_call(self, x, *args, **kwargs)
        if os.environ.get(_ENV, "0") != "1":
            return orig_call(self, x, *args, **kwargs)
        if args or kwargs:  # target-verify / extra flags: stay on the stock path
            return orig_call(self, x, *args, **kwargs)
        if not _eligible(self, x, _max_tokens()):
            return orig_call(self, x, *args, **kwargs)
        try:
            y = fused_routed_mlp(self, x)
        except Exception:
            logger.warning("MoE decode fusion failed; using stock path",
                           exc_info=True)
            return orig_call(self, x, *args, **kwargs)
        if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return y

    return patched


_MAX_T_CACHE: int | None = None


def _max_tokens() -> int:
    global _MAX_T_CACHE
    if _MAX_T_CACHE is None:
        try:
            _MAX_T_CACHE = int(os.environ.get(_ENV_MAX_T, _DEFAULT_MAX_T))
        except ValueError:
            _MAX_T_CACHE = _DEFAULT_MAX_T
    return _MAX_T_CACHE


def apply() -> int:
    """Patch every reachable MoE block class.  Returns how many were patched."""
    global _PATCHED, _MAX_T_CACHE
    _MAX_T_CACHE = None
    if _PATCHED:
        return 0
    n = 0
    for mod_name, cls_name in _TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, "_omlx_moe_decode_fused", False):
            continue
        cls._omlx_moe_decode_fused_original_call = cls.__call__
        cls.__call__ = _make_patched(cls.__call__)
        cls._omlx_moe_decode_fused = True
        n += 1
    _PATCHED = n > 0
    if n:
        logger.info("MoE decode fusion patched %d block classes (T<=%d)",
                    n, _max_tokens())
    return n


__all__ = ["apply", "fused_routed_mlp"]
