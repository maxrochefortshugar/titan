# SPDX-License-Identifier: Apache-2.0
"""Prefill matmul on native int4/int8 tensor operands (M5 NAX, macOS 26.2+).

Stock mlx 0.32.2 sends 4-bit prefill through the ``*_nax`` kernels, which
dequantize the weights to bf16 in threadgroup memory and then call
``mpp::tensor_ops::matmul2d`` with bf16 x bf16 operands.  Measured on this
machine the bf16 tensor path peaks at 65 TFLOP/s and stock
``quantized_matmul`` already reaches 56.9 TFLOP/s of it, so no bf16-operand
rewrite can win more than ~15%.

macOS 26.2 exposes ``int8_t`` and ``metal::int4b_format`` tensor operands, and
those run at roughly 2x the bf16 unit (measured: 124 TOP/s int8 x int8,
119 TOP/s int8 x int4, vs 65 TFLOP/s bf16).  This module takes that path:

    y[m,n] = sum_g  s[n,g] * xs[m,g] * (xi[m,:] . qs[n,:])_g
                  + (8*s[n,g] + b[n,g]) * rowsum[m,g]

``qs`` is the stored affine nibble re-centred to signed int4.  ``q ^ 8`` is
exactly ``q - 8`` in 4-bit two's complement, so the packed mlx weight tensor
needs one XOR at load time and is then handed to the tensor unit verbatim -
no unpacking, no dequantization, no threadgroup round trip, and the weight
operand costs 0.5 bytes/element in the tensor load path instead of 2.

``xi``/``xs`` are a per-row-group (group_size 64) symmetric int8 quantization
of the activations, produced by a fused Metal kernel.  ``rowsum`` is the exact
fp32 group sum of the *unquantized* activations, so the affine bias term
carries no quantization error at all.  That bias term is itself a rank-G
matmul and is folded into the same fp32 accumulator as one extra bf16
``matmul2d`` before the main loop.

This is NOT bit-exact with the stock path: the int8 activation quantization is
a real approximation.  See test_exact.py for measured error.

Enable with ``OMLX_PREFILL_INT4=1`` (default on once installed); ``=0`` or any
unsupported shape falls back to stock ``mx.quantized_matmul``.
"""

from __future__ import annotations

import os
import mlx.core as mx

GROUP = 64
BITS = 4
TM = TN = 32            # per-simdgroup matmul2d tile
SGM, SGN = 2, 4         # simdgroups per threadgroup
NSG = SGM * SGN
TMT = TM * SGM          # 64  rows per threadgroup
TNT = TN * SGN          # 128 cols per threadgroup
MIN_M = 512

_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_KERNEL_CACHE: dict = {}


def _gp(G: int) -> int:
    """matmul2d requires K % 16 == 0, so the bias-term rank is padded."""
    return (G + 15) // 16 * 16


# ---------------------------------------------------------------------------
# fused activation quantizer: x[M,K] bf16 -> xq int8, xscale[M,G], rowsum[M,GP]
# ---------------------------------------------------------------------------

_QUANT_SRC = """
    // one simdgroup per (row, group); each lane owns 2 of the 64 values
    const uint gid = thread_position_in_grid.x / 32;
    const uint lane = thread_position_in_grid.x % 32;
    const uint m = gid / GSZ;
    const uint g = gid % GSZ;
    if (m >= MSZ) return;

    const device uint* xw = (const device uint*)x;
    uint w = xw[(m * KSZ + g * 64) / 2 + lane];
    bfloat2 v = as_type<bfloat2>(w);
    float a = float(v.x), b = float(v.y);

    float amax = simd_max(metal::max(metal::abs(a), metal::abs(b)));
    float sum = simd_sum(a + b);
    float sc = metal::max(amax, 1e-30f) / 127.0f;
    float inv = 1.0f / sc;

    char2 q;
    q.x = char(metal::clamp(metal::rint(a * inv), -127.0f, 127.0f));
    q.y = char(metal::clamp(metal::rint(b * inv), -127.0f, 127.0f));
    ((device char2*)xq)[(m * KSZ + g * 64) / 2 + lane] = q;

    if (lane == 0) {
        xscale[m * GSZ + g] = sc;
        rowsum[m * GPSZ + g] = bfloat(sum);
    }
    if (lane >= 1 && lane < 1 + (GPSZ - GSZ) && g == 0) {
        rowsum[m * GPSZ + GSZ + lane - 1] = bfloat(0.0f);
    }
"""


def _quant_kernel():
    k = _KERNEL_CACHE.get("quant")
    if k is None:
        k = mx.fast.metal_kernel(
            name="prefill_actquant",
            input_names=["x"],
            output_names=["xq", "xscale", "rowsum"],
            header=_HEADER,
            source=_QUANT_SRC,
        )
        _KERNEL_CACHE["quant"] = k
    return k


def quantize_activations(x):
    """Per-row-group symmetric int8 quantization + exact fp32 group sums."""
    M, K = x.shape
    G = K // GROUP
    GP = _gp(G)
    k = _quant_kernel()
    nthreads = M * G * 32
    return k(
        inputs=[x],
        output_shapes=[(M, K), (M, G), (M, GP)],
        output_dtypes=[mx.int8, mx.float32, mx.bfloat16],
        grid=(nthreads, 1, 1),
        threadgroup=(256, 1, 1),
        template=[("MSZ", M), ("KSZ", K), ("GSZ", G), ("GPSZ", GP)],
    )


# ---------------------------------------------------------------------------
# main matmul
# ---------------------------------------------------------------------------

def _source(K: int, N: int, G: int, GP: int) -> str:
    return f"""
    const uint tid  = thread_position_in_threadgroup.x;
    const uint sgid = tid / 32;
    const uint sgm  = sgid % {SGM};
    const uint sgn  = sgid / {SGM};
    const uint m0 = threadgroup_position_in_grid.y * {TMT} + sgm * {TM};
    const uint n0 = threadgroup_position_in_grid.x * {TNT} + sgn * {TN};

    // ---- affine bias term: acc = rowsum[m,:] . (8s+b)[n,:]  (rank-G bf16 matmul)
    constexpr auto bdesc = matmul2d_descriptor({TM}, {TN}, {GP}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<bdesc, metal::execution_simdgroup> bop;
    auto tR = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(
        (device bfloat*)rowsum + m0 * {GP}, dextents<int32_t,2>({GP}, {TM}), array<int32_t,2>{{1, {GP}}});
    auto tBs = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(
        (device bfloat*)bscale + n0 * {GP}, dextents<int32_t,2>({GP}, {TN}), array<int32_t,2>{{1, {GP}}});
    auto acc = bop.get_destination_cooperative_tensor<decltype(tR), decltype(tBs), float>();
    bop.run(tR, tBs, acc);

    // ---- main loop: int8 activations x int4 weights -> int32, rescaled per group
    constexpr auto qdesc = matmul2d_descriptor({TM}, {TN}, {GROUP}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qdesc, metal::execution_simdgroup> qop;
    auto ct0 = qop.get_destination_cooperative_tensor<
        tensor<device int8_t, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::int4b_format, dextents<int32_t,2>, tensor_inline>, int32_t>();

    // This lane owns 32 destination elements spanning only 4 rows and 8 columns.
    // Cache those offsets so each group costs 12 scale loads, not 64.
    ushort mrow[4], ncol[8];
    {{
        const ushort mi[4] = {{0, 4, 16, 20}};
        const ushort ni[8] = {{0, 1, 2, 3, 8, 9, 10, 11}};
        #pragma unroll
        for (uint j = 0; j < 4; ++j) mrow[j] = ct0.get_multidimensional_index(mi[j])[1];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) ncol[j] = ct0.get_multidimensional_index(ni[j])[0];
    }}
    float sv[8], xv[4];

    for (uint g = 0; g < {G}; ++g) {{
        #pragma unroll
        for (uint j = 0; j < 8; ++j) sv[j] = float(scales[(n0 + ncol[j]) * {G} + g]);
        #pragma unroll
        for (uint j = 0; j < 4; ++j) xv[j] = xscale[(m0 + mrow[j]) * {G} + g];

        const uint k0 = g * {GROUP};
        auto tA = tensor<device int8_t, dextents<int32_t,2>, tensor_inline>(
            (device int8_t*)xq + m0 * {K} + k0, dextents<int32_t,2>({GROUP}, {TM}), array<int32_t,2>{{1, {K}}});
        auto tB = tensor<device metal::int4b_format, dextents<int32_t,2>, tensor_inline>(
            (device uchar*)wq + (n0 * {K} + k0) / 2, dextents<int32_t,2>({GROUP}, {TN}), array<int32_t,2>{{1, {K}}});
        auto ct = qop.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
        qop.run(tA, tB, ct);

        #pragma unroll
        for (uint16_t i = 0; i < 32; ++i) {{
            float t = float(ct[i]) * xv[((i >> 2) & 1) + 2 * ((i >> 4) & 1)];
            acc[i] = fma(t, sv[(i & 3) + 4 * ((i >> 3) & 1)], acc[i]);
        }}
    }}

    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        auto ix = acc.get_multidimensional_index(i);
        y[(m0 + ix[1]) * {N} + n0 + ix[0]] = bfloat(acc[i]);
    }}
"""


def _kernel(K: int, N: int, G: int, GP: int):
    key = (K, N, G, GP)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"prefill_i8i4_k{K}_n{N}",
            input_names=["xq", "xscale", "rowsum", "wq", "scales", "bscale"],
            output_names=["y"],
            header=_HEADER,
            source=_source(K, N, G, GP),
        )
        _KERNEL_CACHE[key] = k
    return k


# ---------------------------------------------------------------------------
# host-side preparation
# ---------------------------------------------------------------------------

def prepare_weights(w_q, scales, biases):
    """One-time repack: signed nibbles, plus the folded affine bias scale.

    ``q ^ 8`` maps the stored unsigned nibble q in [0,15] onto the 4-bit two's
    complement encoding of ``q - 8``, so ``w = (q-8)*s + (8s+b)``.
    """
    wq_s = w_q.astype(mx.uint32) ^ mx.array(0x88888888, dtype=mx.uint32)
    bscale = (biases.astype(mx.float32) + 8.0 * scales.astype(mx.float32)).astype(mx.bfloat16)
    N, G = scales.shape
    GP = _gp(G)
    if GP != G:
        bscale = mx.concatenate([bscale, mx.zeros((N, GP - G), dtype=mx.bfloat16)], axis=1)
    return wq_s, scales.astype(mx.bfloat16), bscale


def prefill_qmm(x, wq_s, scales, bscale):
    """y[M,N] = x[M,K] @ dequant(wq)[N,K]^T via int8 x int4 tensor ops."""
    M, K = x.shape
    N = scales.shape[0]
    G = K // GROUP
    xq, xscale, rowsum = quantize_activations(x)
    k = _kernel(K, N, G, _gp(G))
    (y,) = k(
        inputs=[xq, xscale, rowsum, wq_s, scales, bscale],
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSG * (N // TNT), M // TMT, 1),
        threadgroup=(32 * NSG, 1, 1),
    )
    return y


def supported(x, w_q, scales, transpose, group_size, bits) -> bool:
    if not (transpose and group_size == GROUP and bits == BITS):
        return False
    if x.ndim != 2 or w_q.ndim != 2 or scales.ndim != 2:
        return False
    M, K = x.shape
    N = scales.shape[0]
    return (M >= MIN_M and M % TMT == 0 and N % TNT == 0 and K % GROUP == 0
            and x.dtype == mx.bfloat16)


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------

_original_qmm = None
_WCACHE: dict = {}


def _prepared(w_q, scales, biases):
    key = (id(w_q), w_q.shape)
    hit = _WCACHE.get(key)
    if hit is None:
        hit = prepare_weights(w_q, scales, biases)
        mx.eval(*hit)
        _WCACHE[key] = hit
    return hit


def _patched_qmm(x, w_q, scales, biases, transpose=True, group_size=64, bits=4, **kw):
    if os.environ.get("OMLX_PREFILL_INT4", "1") != "0" and supported(
            x, w_q, scales, transpose, group_size, bits):
        wq_s, sc, bs = _prepared(w_q, scales, biases)
        return prefill_qmm(x, wq_s, sc, bs)
    return _original_qmm(x, w_q, scales, biases, transpose=transpose,
                         group_size=group_size, bits=bits, **kw)


def install() -> bool:
    global _original_qmm
    if _original_qmm is not None:
        return True
    _original_qmm = mx.quantized_matmul
    mx.quantized_matmul = _patched_qmm
    return True


def uninstall() -> None:
    global _original_qmm
    if _original_qmm is not None:
        mx.quantized_matmul = _original_qmm
        _original_qmm = None
