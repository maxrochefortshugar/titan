# SPDX-License-Identifier: MIT
"""Grouped RMSNorm over a wide residual stream, bf16 in and bf16 out.

Ported from ``engine/patches/ple-fix/norm_patch.py`` (which routed the stock
prefill hyper-connection norm onto this arithmetic) and the grouped-norm Metal
source it selected.

The stream is ``[rows, groups * hidden]``: ``groups`` independent sub-vectors
per row, each normalised on its own and scaled by ``1 + weight`` over the full
width. A grouped weight cannot be handed to ``mx.fast.rms_norm``, so the
reference has to do

    x.astype(float32) -> rms_norm -> * scale -> astype(bf16)

which at [2048, 10240] is 42 MB in, an 84 MB fp32 intermediate and about
420 MB of traffic. The Metal implementation reads bf16, accumulates the sum of
squares in fp32 per stream, and writes bf16; no fp32 tensor exists.

Exactness: within 1 bf16 ULP, never more. Same arithmetic, different fp32
rounding order (``metal::rsqrt`` and a simd-tree reduction against mlx's
``rms_norm``). Measured on a real 2048-token chunk in the ple-fix report: no
element anywhere differs by more than one bf16 ULP.
"""

from __future__ import annotations

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["OP", "key", "metal", "reference", "supports"]

_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="titan_grouped_rmsnorm_bf16",
            input_names=["x", "w", "eps"],
            output_names=["xn"],
            source=_SOURCE,
        )
    return _KERNEL


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(x, weight, *, groups, eps):
    """Canonical grouped norm. ``x``: [rows, groups*hidden], ``weight``: [width]."""
    rows, width = x.shape
    hidden = width // groups
    dtype = x.dtype
    scale = (1.0 + weight).astype(mx.float32)
    y = x.astype(mx.float32).reshape(rows, groups, hidden)
    y = mx.fast.rms_norm(y, None, eps)
    y = y * scale.reshape(groups, hidden)
    return y.reshape(rows, width).astype(dtype)


def metal(x, weight, *, groups, eps):
    """One dispatch per (row, stream), bf16 throughout. Same signature."""
    rows, width = x.shape
    hidden = width // groups
    return _kernel()(
        inputs=[x, weight, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("K", width), ("H", hidden)],
        grid=(256, groups, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[x.dtype],
    )[0]


def key(x, weight, *, groups, eps) -> ShapeClass:
    return shape_class(x, weight, extra=(groups,))


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) != 2 or not k.extra:
        return False
    xs, ws = k.shapes
    xd, wd = k.dtypes
    if len(xs) != 2 or len(ws) != 1 or ws[0] != xs[1]:
        return False
    if xd not in (mx.bfloat16, mx.float16) or wd != xd:
        return False
    groups = k.extra[0]
    if groups < 1 or xs[1] % groups:
        return False
    # the kernel reduces over 8 simdgroups of 256 threads
    return (xs[1] // groups) >= 1


OP = KernelOp(
    name="grouped_rmsnorm_bf16",
    aliases=("rms_norm_grouped",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=1.0,
    shapes=(
        {"rows": 8, "groups": 4, "hidden": 2560},
        {"rows": 64, "groups": 4, "hidden": 2560},
        {"rows": 2048, "groups": 4, "hidden": 2560},
    ),
    exactness="<= 1 bf16 ULP",
    source="engine/patches/ple-fix/REPORT.md section 4.2",
)
