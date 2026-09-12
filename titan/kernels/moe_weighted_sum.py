# SPDX-License-Identifier: MIT
"""MoE unsort + routed weighted sum in one launch.

Ported from ``engine/patches/round2/wsum10/kernel.py`` and the target-verify
extension in ``engine/patches/round3/small-items/wsum_verify.py``.

The reference is the stock tail of a SwitchGLU block: scatter the sorted expert
output back to token order, multiply by the router scores, reduce over the k
axis. Three full passes over a [T, k, D] bf16 tensor, 105 MB at T = 2048,
k = 10, D = 2560, about 520 MB of traffic. The Metal implementation consumes
the sorted output directly, applies the inverse permutation as a gather while
it accumulates, and writes [T, D]: one read and one write.

Both layouts are supported. ``inv_order = None`` is the unsorted (token-major)
layout the MTP target-verify pass produces, where no gather is needed at all;
otherwise ``inv_order`` is the permutation from the sort.

Three accumulation modes, all fp32 registers:

``clone`` (default)
    bf16 products, eight bf16 partial accumulators combined sequentially in
    bf16, which is exactly what MLX's own bf16 axis reduce does. Bit-identical
    to the reference.
``ops``
    bf16-rounded products, fp32 accumulation.
``fast``
    fp32 scores and fp32 products throughout. More accurate than the reference,
    and therefore not bit-identical to it.

The eight partial accumulators are not a numerical accident: they break the
k-long FMA dependency chain (worth 1.45x) and they reproduce the partial
accumulator layout of MLX's reduce, which is what makes ``clone`` exact.
"""

from __future__ import annotations

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["MODES", "OP", "key", "metal", "reference", "supports"]

# mode -> (NACC, ROUND_PROD, BF_ACC, carry scores in the input dtype)
MODES = {
    "fast": (8, 0, 0, False),
    "ops": (8, 1, 0, True),
    "clone": (8, 1, 1, True),
}
DEFAULT_MODE = "clone"

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
_IDENTITY: dict[int, mx.array] = {}


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="titan_moe_weighted_sum",
            input_names=["xs", "inv", "sc"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def _pick_vec_tg(dim: int, want: int = 4) -> tuple[int, int]:
    """Choose (VEC, threadgroup width) so DIM/VEC is a multiple of the width."""
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


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(xs, inv_order, scores, out_shape, mode=DEFAULT_MODE):
    """The stock three-pass tail, in plain MLX ops.

    ``xs``        expert output, any shape flattening to [R*K, D].
    ``inv_order`` uint32 [R*K] permutation, or None when ``xs`` is already in
                  token-major order.
    ``scores``    router weights [..., K], already normalised.
    ``out_shape`` shape of the result, e.g. (B, T, D).
    """
    dim = out_shape[-1]
    rows = 1
    for d in out_shape[:-1]:
        rows *= d
    k = scores.shape[-1]
    flat = xs.reshape(rows * k, dim)
    if inv_order is not None:
        flat = flat[inv_order.reshape(rows * k).astype(mx.uint32)]
    y = flat.reshape(*out_shape[:-1], k, dim)
    return (y * scores.reshape(*out_shape[:-1], k, 1)).sum(axis=-2)


def metal(xs, inv_order, scores, out_shape, mode=DEFAULT_MODE):
    """Unsort + routed weighted sum in one launch. Same signature as reference."""
    nacc, round_prod, bf_acc, sc_native = MODES[mode]
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
        inv, use_inv = _identity(rows * k), 0
    else:
        inv = mx.contiguous(inv_order.reshape(rows * k).astype(mx.uint32))
        use_inv = 1

    vec, tgx = _pick_vec_tg(dim)
    (out,) = _kernel()(
        inputs=[xs, inv, sc],
        template=[("T", dtype), ("DIM", dim), ("ROWS", rows), ("K", k),
                  ("VEC", vec), ("CONTIG", 1), ("NACC", nacc),
                  ("USE_INV", use_inv), ("ROUND_PROD", round_prod),
                  ("BF_ACC", bf_acc)],
        grid=(dim // vec, rows, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[tuple(out_shape)],
        output_dtypes=[dtype],
    )
    return out


def key(xs, inv_order, scores, out_shape, mode=DEFAULT_MODE) -> ShapeClass:
    return shape_class(
        xs, inv_order, scores,
        extra=(tuple(out_shape), mode, inv_order is None),
    )


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or not k.extra:
        return False
    out_shape, mode = k.extra[0], k.extra[1]
    if mode not in MODES:
        return False
    if k.dtypes[0] not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if len(out_shape) < 2 or out_shape[-1] % 32:
        return False
    return True


OP = KernelOp(
    name="moe_weighted_sum",
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=0.0,
    shapes=(
        {"T": 9, "K": 10, "D": 256},
        {"T": 13, "K": 10, "D": 2560},
        {"T": 1, "K": 10, "D": 2560},
        {"T": 2048, "K": 10, "D": 2560},
    ),
    exactness="bit-identical in clone mode",
    source="engine/patches/round2/wsum10/REPORT.md, "
           "engine/patches/round3/small-items/REPORT.md item 3",
)
