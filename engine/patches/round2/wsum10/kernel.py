# SPDX-License-Identifier: Apache-2.0
"""Fused MoE unsort + routed weighted sum for Qwen3.8-Flash-Next (top_k=10).

oMLX ships a native weighted-sum kernel for this exact spot but whitelists
``top_k in (6, 8)`` (``omlx/patches/qwen35_moe_weighted_sum.py:58``), so
Flash-Next (``qwen4_exp``, top_k=10, hidden 2560) falls back to generic MLX
ops.  The stock chain in
``mlx_vlm/models/qwen3_5_moe/language.py:64-65`` is

    y = _target_verify_switch_glu(self.switch_mlp, x, inds, target_verify)
    y = (y * scores[..., None]).sum(axis=-2)

where ``SwitchGLU.__call__`` (``mlx_lm/models/switch_layers.py:195-198``)
first runs ``_scatter_unsort`` on the sorted expert output.  That is three
full passes over a [T, k, D] bf16 tensor (105 MB at T=2048, k=10, D=2560):
scatter, multiply, reduce.

This module replaces the tail with a single JIT ``mx.fast.metal_kernel``
launch that consumes the SORTED expert output directly, applies
``inv_order`` as a gather while it accumulates, and writes [T, D].  One read
of the 105 MB and one 10 MB write instead of ~520 MB of traffic.

bf16 in, bf16 out, fp32 accumulate.  Two accumulation modes:

* ``fast``   - fp32 scores, fp32 products (more accurate than the ops path).
* ``ops``    - each product rounded to bf16 before accumulating, which is
               what the ops path does; bit-identical in practice.

``ops`` is the default so the exactness bar is met by construction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

ENV_ENABLE = "OMLX_WSUM_TOPK10"
ENV_MODE = "OMLX_WSUM_TOPK10_MODE"          # "ops" (default) or "fast"
ENV_MIN_TOKENS = "OMLX_WSUM_TOPK10_MIN_TOKENS"   # 1 = decode too
ENV_TOPK = "OMLX_WSUM_TOPK10_K"             # which top_k values to claim
ENV_VEC = "OMLX_WSUM_TOPK10_VEC"            # outputs per thread (tuning)
ENV_CONTIG = "OMLX_WSUM_TOPK10_CONTIG"      # 1 = adjacent columns per thread

_SOURCE = r"""
    constexpr uint DPV = uint(DIM) / uint(VEC);
    const uint dx = thread_position_in_grid.x;   // 0 .. DPV-1
    const uint r  = thread_position_in_grid.y;   // output row
    if (dx >= DPV || r >= uint(ROWS)) {
        return;
    }
    const uint base = r * uint(K);
    // CONTIG != 0: each thread owns VEC adjacent columns (vector loads).
    // CONTIG == 0: columns strided by DPV (one coalesced load per step).
    constexpr uint STRIDE = (CONTIG != 0) ? 1u : DPV;
    const uint d0 = (CONTIG != 0) ? (dx * uint(VEC)) : dx;

    // NACC partial accumulators break the k-long FMA dependency chain (worth
    // 1.45x here).  NACC == 8 with BF_ACC also reproduces exactly the partial-
    // accumulator layout of MLX's own bf16 axis reduce (see REPORT.md).
    float acc[VEC][NACC];
    for (uint v = 0; v < uint(VEC); ++v) {
        for (uint a = 0; a < uint(NACC); ++a) {
            acc[v][a] = 0.0f;
        }
    }

    for (uint j = 0; j < uint(K); ++j) {
        const uint p = base + j;
        const uint src = (USE_INV != 0) ? uint(inv[p]) : p;
        const float s = float(sc[p]);
        const uint slot = (uint(NACC) == 1u) ? 0u : (j % uint(NACC));
        const device T* xr = xs + src * uint(DIM) + d0;
        for (uint v = 0; v < uint(VEC); ++v) {
            float prod = s * float(xr[v * STRIDE]);
            if (ROUND_PROD != 0) {
                prod = float(T(prod));
            }
            float t = acc[v][slot] + prod;
            acc[v][slot] = (BF_ACC != 0) ? float(T(t)) : t;
        }
    }

    device T* orow = out + r * uint(DIM) + d0;
    for (uint v = 0; v < uint(VEC); ++v) {
        float a = acc[v][0];
        for (uint j = 1; j < uint(NACC); ++j) {
            a = a + acc[v][j];
            if (BF_ACC != 0) {
                a = float(T(a));
            }
        }
        orow[v * STRIDE] = T(a);
    }
"""

_KERNEL = None
_IDENTITY: dict = {}

# mode -> (NACC, ROUND_PROD, BF_ACC, scores carried in the input dtype)
_MODES = {
    # fp32 scores, fp32 products, eight fp32 partial accumulators.
    "fast": (8, 0, 0, False),
    # fp32 accumulation of bf16-rounded products.
    "ops": (8, 1, 0, True),
    # Bit-exact clone of the stock chain: bf16 products, eight bf16 partial
    # accumulators, bf16 sequential combine - what MLX's own axis reduce does.
    "clone": (8, 1, 1, True),
}
_DEFAULT_MODE = "clone"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="wsum_topk_unsort",
            input_names=["xs", "inv", "sc"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def _pick_vec_tg(dim: int):
    """Choose (VEC, threadgroup width) so DIM/VEC is a multiple of the width."""
    want = _env_int(ENV_VEC, 4)
    for vec in (want, 4, 2, 1):
        if vec <= 0 or dim % vec:
            continue
        dpv = dim // vec
        for tg in (256, 128, 64, 32):
            if dpv % tg == 0:
                return vec, tg
    return 1, 32


def _identity(n: int) -> mx.array:
    arr = _IDENTITY.get(n)
    if arr is None:
        arr = mx.arange(n, dtype=mx.uint32)
        mx.eval(arr)
        _IDENTITY[n] = arr
    return arr


def weighted_sum(xs, inv_order, scores, out_shape, mode=_DEFAULT_MODE):
    """Unsort + routed weighted sum in one launch.

    ``xs``        sorted expert output, any shape flattening to [R*K, D].
    ``inv_order`` uint32 [R*K] permutation from ``_gather_sort``, or None when
                  the expert output is already in token-major order.
    ``scores``    router weights, shape [..., K], already normalised.
    ``out_shape`` shape of the result, e.g. (B, T, D).
    """
    nacc, round_prod, bf_acc, sc_native = _MODES[mode]
    dim = out_shape[-1]
    rows = 1
    for d in out_shape[:-1]:
        rows *= d
    k = scores.shape[-1]

    dtype = xs.dtype
    xs = mx.contiguous(xs.reshape(rows * k, dim))
    sc = mx.contiguous(
        scores.reshape(rows * k).astype(dtype if sc_native else mx.float32)
    )

    if inv_order is None:
        inv = _identity(rows * k)
        use_inv = 0
    else:
        inv = mx.contiguous(inv_order.reshape(rows * k).astype(mx.uint32))
        use_inv = 1

    vec, tgx = _pick_vec_tg(dim)
    (out,) = _kernel()(
        inputs=[xs, inv, sc],
        template=[
            ("T", dtype),
            ("DIM", dim),
            ("ROWS", rows),
            ("K", k),
            ("VEC", vec),
            ("CONTIG", _env_int(ENV_CONTIG, 1)),
            ("NACC", nacc),
            ("USE_INV", use_inv),
            ("ROUND_PROD", round_prod),
            ("BF_ACC", bf_acc),
        ],
        grid=(dim // vec, rows, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[tuple(out_shape)],
        output_dtypes=[dtype],
    )
    return out


# --------------------------------------------------------------------------
# The MoE block replacement
# --------------------------------------------------------------------------


def fused_switch_weighted_sum(
    switch_mlp: Any,
    x: mx.array,
    inds: mx.array,
    scores: mx.array,
    mode: str,
) -> mx.array:
    """SwitchGLU forward whose tail is the fused kernel.

    Mirrors ``SwitchGLU.__call__`` and oMLX's gate_up-fused variant, but the
    final ``_scatter_unsort`` + ``(y * scores).sum(-2)`` become one launch.
    """
    from mlx_lm.models.switch_layers import _gather_sort

    do_sort = inds.size >= 64
    xe = mx.expand_dims(x, (-2, -3))
    idx = inds
    inv_order = None
    if do_sort:
        xe, idx, inv_order = _gather_sort(xe, inds)
    if switch_mlp.training:
        idx = mx.stop_gradient(idx)

    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    if gate_up is not None:
        x_gate_up = gate_up(xe, idx, sorted_indices=do_sort)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
    else:
        x_up = switch_mlp.up_proj(xe, idx, sorted_indices=do_sort)
        x_gate = switch_mlp.gate_proj(xe, idx, sorted_indices=do_sort)
    ys = switch_mlp.down_proj(
        switch_mlp.activation(x_up, x_gate), idx, sorted_indices=do_sort
    )

    out_shape = (*x.shape[:-1], ys.shape[-1])
    return weighted_sum(ys, inv_order, scores, out_shape, mode=mode)


def _target_verify_arg(args, kwargs) -> bool:
    if bool(kwargs.get("target_verify", False)):
        return True
    return bool(args and isinstance(args[0], bool) and args[0])


def _supported_topk() -> tuple[int, ...]:
    raw = os.environ.get(ENV_TOPK, "10")
    out = []
    for piece in raw.replace(",", " ").split():
        try:
            out.append(int(piece))
        except ValueError:
            pass
    return tuple(out)


def _mode() -> str:
    m = os.environ.get(ENV_MODE, _DEFAULT_MODE).lower()
    return m if m in _MODES else _DEFAULT_MODE


def _min_tokens() -> int:
    return _env_int(ENV_MIN_TOKENS, 1)


# Config is snapshotted at install() so the hot path (48 MoE blocks per token
# at decode) does no environment lookups.
_CFG = {"mode": _DEFAULT_MODE, "min_tokens": 1, "topk": (10,)}


def _should_route(self: Any, x: mx.array, target_verify: bool, min_tokens: int) -> bool:
    # Shape gate first: this runs on every MoE block call of every decode step.
    if x.ndim != 3 or x.shape[-2] < min_tokens:
        return False
    if target_verify:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if getattr(self, "sharding_group", None) is not None:
        return False
    if int(getattr(self, "top_k", 0)) not in _CFG["topk"]:
        return False
    switch_mlp = getattr(self, "switch_mlp", None)
    if switch_mlp is None or not hasattr(switch_mlp, "down_proj"):
        return False
    return hasattr(switch_mlp, "gate_up_proj") or (
        hasattr(switch_mlp, "up_proj") and hasattr(switch_mlp, "gate_proj")
    )


def _fast_moe(self: Any, x: mx.array, target_verify: bool, mode: str) -> mx.array:
    # Router chain reproduced verbatim from
    # mlx_vlm/models/qwen3_5_moe/language.py:56-62 so the expert set and the
    # score values are bit-identical to the stock path.
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(self, "norm_topk_prob", True):
        scores = scores / scores.sum(axis=-1, keepdims=True)

    y = fused_switch_weighted_sum(self.switch_mlp, x, inds, scores, mode)

    if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
        try:
            shared_y = self.shared_expert(x, target_verify)
        except TypeError:
            shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        y = y + shared_y
    return y


def _make_patched_call(orig_call):
    def patched(self, x: mx.array, *args, **kwargs):
        cfg = _CFG
        if x.ndim != 3 or x.shape[-2] < cfg["min_tokens"]:
            return orig_call(self, x, *args, **kwargs)
        target_verify = _target_verify_arg(args, kwargs)
        if not _should_route(self, x, target_verify, cfg["min_tokens"]):
            return orig_call(self, x, *args, **kwargs)
        try:
            return _fast_moe(self, x, target_verify, cfg["mode"])
        except Exception:
            logger.warning("top_k=10 MoE weighted-sum fast path failed", exc_info=True)
            return orig_call(self, x, *args, **kwargs)

    return patched


_TARGETS = (
    ("mlx_vlm.models.qwen3_5_moe.language", "Qwen3_5MoeSparseMoeBlock"),
    ("mlx_lm.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"),
    ("mlx_lm.models.qwen3_5", "SparseMoeBlock"),
    ("mlx_lm.models.qwen3_next", "Qwen3NextSparseMoeBlock"),
)

_FLAG = "_wsum_topk10_patched"


def _patch_class(module_name: str, class_name: str) -> bool:
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    if getattr(cls, _FLAG, False):
        return True
    orig = cls.__call__
    cls.__call__ = _make_patched_call(orig)
    setattr(cls, _FLAG, True)
    setattr(cls, "_wsum_topk10_original_call", orig)
    return True


_PATCHED = False


def install() -> bool:
    """Idempotent; returns False and leaves the stock path on any failure."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get(ENV_ENABLE, "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    _CFG["mode"] = _mode()
    _CFG["min_tokens"] = max(1, _min_tokens())
    _CFG["topk"] = _supported_topk() or (10,)
    if not _CFG["topk"]:
        return False
    try:
        # Compile-and-verify probe at a tiny shape so a broken kernel never
        # reaches the model.
        if not _self_check():
            logger.warning("top_k=10 weighted-sum self-check failed; patch skipped")
            return False
    except Exception:
        logger.warning("top_k=10 weighted-sum self-check raised", exc_info=True)
        return False

    patched = False
    for module_name, class_name in _TARGETS:
        patched |= _patch_class(module_name, class_name)
    _PATCHED = patched
    if patched:
        logger.info(
            "top_k=%s MoE weighted-sum kernel installed (mode=%s, min_tokens=%d)",
            _CFG["topk"],
            _CFG["mode"],
            _CFG["min_tokens"],
        )
    return patched


def _self_check() -> bool:
    """Compile the kernel and verify it at a small and a hidden-size shape.

    In ``clone`` mode the bar is bit-identity with the stock chain; the fp32
    modes get a tight relative bound instead (there the stock chain is the
    looser side, see REPORT.md).
    """
    import random

    mode = _CFG["mode"]
    ks = _CFG["topk"]
    k = ks[0] if ks else 10
    dim2 = _env_int("OMLX_WSUM_TOPK10_SELFCHECK_DIM", 2560)
    for rows, dim in ((9, 256), (128, dim2)):
        n = rows * k
        ys = mx.random.normal((n, 1, dim)).astype(mx.bfloat16)
        perm = mx.array(random.Random(0).sample(range(n), n), dtype=mx.uint32)
        scores = mx.random.uniform(shape=(1, rows, k)).astype(mx.bfloat16)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        got = weighted_sum(ys, perm, scores, (1, rows, dim), mode=mode)
        ref = (
            ys.reshape(n, dim)[perm].reshape(1, rows, k, dim) * scores[..., None]
        ).sum(axis=-2)
        mx.eval(got, ref)
        if mode == "clone":
            if not bool(mx.all(got == ref).item()):
                return False
        else:
            g = got.astype(mx.float32)
            rf = ref.astype(mx.float32)
            scale = float(mx.max(mx.abs(rf)).item()) + 1e-6
            if float(mx.max(mx.abs(g - rf)).item()) > 0.02 * scale:
                return False
    return True


__all__ = ["install", "weighted_sum", "fused_switch_weighted_sum"]
