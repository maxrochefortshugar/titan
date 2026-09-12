# SPDX-License-Identifier: Apache-2.0
"""Sorted-gather MoE prefill matmul on M5 NAX int8 x int4 tensor operands.

Companion to ``../prefill-int4/kernel.py``, which did the dense case.  The
routed-expert GEMMs are ~79% of Qwen3.8-Flash-Next prefill compute and go
through ``mx.gather_qmm(..., sorted_indices=True)``, which on mlx 0.32.2
lands in the NAX ``*_gather_qmm_rhs_nax`` kernel: weights are dequantized to
bf16 in threadgroup memory and fed to ``matmul2d`` as bf16 x bf16.

Two things are left on the table there, and this module takes both:

1. bf16 x bf16 peaks at 65.7 TFLOP/s on this machine; uint8 x uint4 peaks at
   113 TOP/s.  Feeding the *packed* 4-bit weights straight to the tensor unit
   also removes the dequantize round trip and cuts weight traffic in the
   tensor load path from 2 bytes/weight to 0.5.
2. Stock tiles the row axis at BM=32 with one tile per (expert, block).  A
   2048-token top-10 chunk is ~40 rows per expert, so stock runs two 32-row
   tiles per expert and reads every expert weight *twice*.  This kernel uses
   a device-built ragged tile table so a 48- or 64-row tile covers a whole
   expert: the weights stream once and the row waste drops from 1.6x to
   ~1.2-1.3x.

Unsigned, not signed
--------------------
The dense kernel re-centred the stored nibble with ``q ^ 8`` so it could use
signed ``int4b_format`` with int8 activations.  That is not affordable here:
the XOR produces a second copy of the weights, and the routed experts *are*
the model (1.26 GB per layer, 60 GB total).  So this kernel keeps the stored
unsigned nibbles verbatim as ``metal::uint4b_format`` and quantizes the
activations to **uint8 with a fixed zero point of 128**:

    x[m,k] = sc[m,g] * (xu[m,k] - 128)
    w[n,k] = q[n,k] * s[n,g] + b[n,g]

    y[m,n] = sum_g  sc[m,g] * s[n,g] * ( D[m,n,g] - 128 * qsum[n,g] )
                  + rowsum[m,g] * b[n,g]

``D`` is the uint8 x uint4 -> int32 tensor-unit dot product over the 64-wide
group and ``qsum[n,g] = sum_k q[n,k]`` is a per (expert, row, group) integer
in [0, 960].  ``D - 128*qsum`` is a difference of two large nearly-equal
integers, so it is done in the **integer** domain (exact); folding the
correction into a bf16 scale table loses ~0.4% of the whole product to
cancellation, which is why ``qsum`` is stored as uint16 and not pre-scaled.
``rowsum`` is the exact fp32 group sum of the *unquantized* activations, so
the affine bias term carries no activation-quantization error.

``qsum`` costs one uint16 per (expert, out-row, group) - the same size as the
existing ``scales`` array, i.e. ~78 MB per MoE layer, 3.7 GB across the
48 layers of Flash-Next.  That is the price of not duplicating the weights.

Not bit-exact.  See test_exact.py.
"""

from __future__ import annotations

import os
import mlx.core as mx

GROUP = 64
BITS = 4

_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# destination-layout probe
# ---------------------------------------------------------------------------

def _probe_layout(TM: int, TN: int):
    """Element -> (row slot, col slot) map of the matmul2d destination tile.

    The absolute row/col a lane owns is lane dependent, but the *rank* of an
    element's row among that lane's distinct rows is uniform across the
    simdgroup (verified here, for every element and every lane).  Baking the
    ranks in as compile-time constants keeps the rescale's register arrays
    statically indexed; querying the absolute rows at kernel start keeps the
    kernel honest about the real layout.
    """
    key = ("probe", TM, TN)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    src = f"""
    const uint lane = thread_position_in_threadgroup.x;
    constexpr auto d = matmul2d_descriptor({TM}, {TN}, 64, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, metal::execution_simdgroup> op;
    auto ct = op.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>, int32_t>();
    uint C = ct.get_capacity();
    if (lane == 0) cap[0] = C;
    for (uint i = 0; i < C; ++i) {{
        auto ix = ct.get_multidimensional_index(i);
        out[(lane * 128 + i) * 2 + 0] = ix[0];
        out[(lane * 128 + i) * 2 + 1] = ix[1];
    }}
    """
    k = mx.fast.metal_kernel(name=f"moe_probe_{TM}_{TN}", input_names=[],
                             output_names=["out", "cap"], header=_HEADER, source=src)
    out, cap = k(inputs=[], output_shapes=[(32 * 128 * 2,), (1,)],
                 output_dtypes=[mx.int32, mx.int32], grid=(32, 1, 1), threadgroup=(32, 1, 1))
    mx.eval(out, cap)
    o = out.tolist()
    C = cap.item()
    lane0 = [(o[i * 2], o[i * 2 + 1]) for i in range(C)]
    rows = sorted({m for _, m in lane0})
    cols = sorted({n for n, _ in lane0})
    ranks = [(rows.index(m), cols.index(n)) for n, m in lane0]
    for L in range(1, 32):
        p = [(o[(L * 128 + i) * 2], o[(L * 128 + i) * 2 + 1]) for i in range(C)]
        r_, c_ = sorted({m for _, m in p}), sorted({n for n, _ in p})
        if [(r_.index(m), c_.index(n)) for n, m in p] != ranks:
            raise RuntimeError(f"matmul2d destination layout is not simd-uniform for {TM}x{TN}")
    # a representative element index for each row slot / col slot
    rep_row = [next(i for i in range(C) if ranks[i][0] == j) for j in range(len(rows))]
    rep_col = [next(i for i in range(C) if ranks[i][1] == j) for j in range(len(cols))]
    hit = (C, len(rows), len(cols), ranks, rep_row, rep_col)
    _CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------
# fused activation quantizer: bf16 [R,K] -> uint8 codes, fp32 scales, fp32 rowsums
# ---------------------------------------------------------------------------

_QUANT_SRC = """
    const uint gid = thread_position_in_grid.x / 32;
    const uint lane = thread_position_in_grid.x % 32;
    const uint m = gid / GSZ;
    if (gid >= RPAD * GSZ) return;
    const uint g = gid % GSZ;
    if (m >= RPAD) return;
    if (m >= RSZ) {                       // padding rows: zero, never stored
        ((device uchar2*)xq)[(m * KSZ + g * 64) / 2 + lane] = uchar2(128, 128);
        if (lane == 0) { xsc[m*GSZ+g] = 0.0f; xrs[m*GSZ+g] = 0.0f; xav[m*GSZ+g] = 0.0f; }
        return;
    }
    const device uint* xw = (const device uint*)x;
    uint w = xw[(m * KSZ + g * 64) / 2 + lane];
    bfloat2 v = as_type<bfloat2>(w);
    float a = float(v.x), b = float(v.y);

    float amax = simd_max(metal::max(metal::abs(a), metal::abs(b)));
    float sum = simd_sum(a + b);
    float sc = metal::max(amax, 1e-30f) / 127.0f;
    float inv = 1.0f / sc;

    uchar2 q;
    q.x = uchar(metal::clamp(metal::rint(a * inv) + 128.0f, 0.0f, 255.0f));
    q.y = uchar(metal::clamp(metal::rint(b * inv) + 128.0f, 0.0f, 255.0f));
    ((device uchar2*)xq)[(m * KSZ + g * 64) / 2 + lane] = q;
    float csum = simd_sum(float(q.x) + float(q.y));

    if (lane == 0) {
        xsc[m * GSZ + g] = sc;
        xrs[m * GSZ + g] = sum;
        xav[m * GSZ + g] = 65536.0f - 8.0f * csum;   // exact: |.| < 2^18
    }
"""


def _quant_kernel():
    k = _CACHE.get("quant")
    if k is None:
        k = mx.fast.metal_kernel(name="moe_actquant_u8", input_names=["x"],
                                 output_names=["xq", "xsc", "xrs", "xav"],
                                 header=_HEADER, source=_QUANT_SRC)
        _CACHE["quant"] = k
    return k


def quantize_activations(x, pad: int):
    """uint8 codes (zero point 128), scales, exact group sums, and the -8*code-sum term."""
    R, K = x.shape
    G = K // GROUP
    RP = R + pad
    return _quant_kernel()(
        inputs=[x], output_shapes=[(RP, K), (RP, G), (RP, G), (RP, G)],
        output_dtypes=[mx.uint8, mx.float32, mx.float32, mx.float32],
        grid=(((RP * G * 32 + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        template=[("RSZ", R), ("RPAD", RP), ("KSZ", K), ("GSZ", G)])


# ---------------------------------------------------------------------------
# per (expert, row, group) nibble sums - computed once per weight tensor
# ---------------------------------------------------------------------------

_QSUM_SRC = """
    const uint i = thread_position_in_grid.x;      // (e*N + n)*G + g
    if (i >= TOTAL) return;
    const device uint* w = (const device uint*)wq;
    uint acc = 0;
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
        uint v = w[i * 8 + j];
        #pragma unroll
        for (uint t = 0; t < 8; ++t) acc += (v >> (4 * t)) & 0xf;
    }
    const uint g = i % GSZ, n = (i / GSZ) % NSZ, e = i / (GSZ * NSZ);
    qsum[(e * GSZ + g) * NSZ + n] = ushort(acc);     // [E, G, N]
"""


def compute_qsum(wq):
    """Sum of the 64 stored nibbles of each affine group, as [E, G, N] uint16."""
    E, N, KW = wq.shape
    G = KW // 8
    total = E * N * G
    k = _CACHE.get("qsum")
    if k is None:
        k = mx.fast.metal_kernel(name="moe_qsum", input_names=["wq"], output_names=["qsum"],
                                 header=_HEADER, source=_QSUM_SRC)
        _CACHE["qsum"] = k
    (q,) = k(inputs=[wq], output_shapes=[(E, G, N)], output_dtypes=[mx.uint16],
             grid=(((total + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
             template=[("TOTAL", total), ("GSZ", G), ("NSZ", N)])
    return q


# ---------------------------------------------------------------------------
# expert row offsets + ragged tile table (device side, no host sync)
# ---------------------------------------------------------------------------

_TILE_SRC = """
    const uint t = thread_position_in_grid.x;
    for (uint tt = t; tt < NEXP + 1; tt += THREADS) {
        const uint e_ = tt;                      // binary search: first row with idx >= t
        uint lo = 0, hi = ROWS;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (uint(idx[mid]) < e_) lo = mid + 1; else hi = mid;
        }
        offsets[e_] = lo;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (t != 0) return;
    uint n = 0;
    for (uint e = 0; e < NEXP; ++e) {
        uint s = offsets[e], q = offsets[e + 1];
        for (uint r = s; r < q; r += TMT) {
            if (n < MAXT) { tile_row[n] = r; tile_exp[n] = e; }
            ++n;
        }
    }
    ntiles[0] = metal::min(n, uint(MAXT));
    for (uint j = ntiles[0]; j < MAXT; ++j) { tile_row[j] = 0; tile_exp[j] = 0; }
"""


def build_tiles(indices, E: int, TMT: int, maxt: int):
    k = _CACHE.get("tiles")
    if k is None:
        k = mx.fast.metal_kernel(
            name="moe_tiles", input_names=["idx"],
            output_names=["offsets", "tile_row", "tile_exp", "ntiles"],
            header=_HEADER, source=_TILE_SRC)
        _CACHE["tiles"] = k
    R = indices.shape[0]
    # one threadgroup only: the offsets->tile-table handoff uses a threadgroup barrier
    tg = 1024
    return k(inputs=[indices.astype(mx.uint32)],
             output_shapes=[(E + 1,), (maxt,), (maxt,), (1,)],
             output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
             grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
             template=[("NEXP", E), ("ROWS", R), ("TMT", TMT), ("MAXT", maxt), ("THREADS", tg)])


# ---------------------------------------------------------------------------
# main kernel
# ---------------------------------------------------------------------------

def _source(cfg, K, N, G):
    TM, TN, SGM, SGN = cfg
    C, NR, NC, ranks, rep_row, rep_col = _probe_layout(TM, TN)
    TMT, TNT = TM * SGM, TN * SGN
    ri = ", ".join(str(r) for r, _ in ranks)
    ci = ", ".join(str(c) for _, c in ranks)
    rr = ", ".join(str(i) for i in rep_row)
    rc = ", ".join(str(i) for i in rep_col)
    consts = (f"constant uchar RI[{C}] = {{{ri}}};\n"
              f"constant uchar CI[{C}] = {{{ci}}};\n"
              f"constant uchar RREP[{NR}] = {{{rr}}};\n"
              f"constant uchar CREP[{NC}] = {{{rc}}};\n")
    return consts, f"""
    const uint t = threadgroup_position_in_grid.y;
    if (t >= ntiles[0]) return;
    const uint e     = tile_exp[t];
    const uint row0  = tile_row[t];
    const uint rowend = offsets[e + 1];

    const uint tid  = thread_position_in_threadgroup.x;
    const uint sgid = tid / 32;
    const uint sgm  = sgid % {SGM};
    const uint sgn  = sgid / {SGM};
    const uint m0 = row0 + sgm * {TM};
    const uint n0 = threadgroup_position_in_grid.x * {TNT} + sgn * {TN};

    constexpr auto qdesc = matmul2d_descriptor({TM}, {TN}, {GROUP}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qdesc, metal::execution_simdgroup> qop;
    auto ct0 = qop.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>, int32_t>();

    ushort mrow[{NR}], ncol[{NC}];
    #pragma unroll
    for (uint j = 0; j < {NR}; ++j) mrow[j] = ct0.get_multidimensional_index(RREP[j])[1];
    #pragma unroll
    for (uint j = 0; j < {NC}; ++j) ncol[j] = ct0.get_multidimensional_index(CREP[j])[0];

    float acc[{C}];
    #pragma unroll
    for (uint i = 0; i < {C}; ++i) acc[i] = 0.0f;

    const device bfloat*  sc_e = (const device bfloat*)scales + (size_t)e * {N} * {G};
    const device bfloat*  bi_e = (const device bfloat*)biases + (size_t)e * {N} * {G};
    const device ushort*  qs_e = (const device ushort*)qsum   + (size_t)e * {N} * {G};
    const device uchar*   wq_e = (const device uchar*)wq + (size_t)e * ({N} * {K} / 2);

    float sv[{NC}], bv[{NC}], qv[{NC}], xv[{NR}], uv[{NR}], av[{NR}];

    for (uint g = 0; g < {G}; ++g) {{
        // [E, G, N] tables: the {NC} columns a lane owns are contiguous, so the
        // whole simdgroup's scale/bias/qsum loads are one cache line each.
        #pragma unroll
        for (uint j = 0; j < {NC}; ++j) {{
            size_t o = (size_t)g * {N} + n0 + ncol[j];
            sv[j] = float(sc_e[o]);
            bv[j] = float(bi_e[o]);
            qv[j] = sv[j] * (128.0f * float(qs_e[o]));   // fp32: no cancellation
        }}
        #pragma unroll
        for (uint j = 0; j < {NR}; ++j) {{
            uint o = (m0 + mrow[j]) * {G} + g;
            xv[j] = xsc[o];
            uv[j] = xrs[o];
            av[j] = xav[o];
        }}
        const uint k0 = g * {GROUP};
        auto tA = tensor<device uchar, dextents<int32_t,2>, tensor_inline>(
            (device uchar*)xq + (size_t)m0 * {K} + k0,
            dextents<int32_t,2>({GROUP}, {TM}), array<int32_t,2>{{1, {K}}});
        auto tB = tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>(
            (device uchar*)wq_e + ((size_t)n0 * {K} + k0) / 2,
            dextents<int32_t,2>({GROUP}, {TN}), array<int32_t,2>{{1, {K}}});
        auto ct = qop.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
        qop.run(tA, tB, ct);

        #pragma unroll
        for (uint16_t i = 0; i < {C}; ++i) {{
            const uint r = RI[i], c = CI[i];
            // sc*s*sum_k (xu-128)*(q-8)  +  (8s+b)*rowsum_exact
            float t = float(ct[i]) + av[r];
            acc[i] = fma(xv[r], fma(t, sv[c], -qv[c]), acc[i]);
            acc[i] = fma(uv[r], bv[c], acc[i]);
        }}
    }}

    #pragma unroll
    for (uint16_t i = 0; i < {C}; ++i) {{
        uint m = m0 + mrow[RI[i]];
        if (m < rowend) y[(size_t)m * {N} + n0 + ncol[CI[i]]] = bfloat(acc[i]);
    }}
"""


def _kernel(cfg, K, N, G):
    key = (cfg, K, N, G)
    k = _CACHE.get(key)
    if k is None:
        consts, src = _source(cfg, K, N, G)
        k = mx.fast.metal_kernel(
            name=f"moe_i8i4_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}_k{K}_n{N}",
            input_names=["xq", "xsc", "xrs", "xav", "wq", "scales", "biases", "qsum",
                         "offsets", "tile_row", "tile_exp", "ntiles"],
            output_names=["y"], header=_HEADER + consts, source=src)
        _CACHE[key] = k
    return k


# TM, TN, SGM, SGN.  TM=48 covers a whole expert's ~40 rows in one tile, so each
# expert weight is streamed exactly once; TN=16 keeps the per-lane accumulator at 24
# floats (no spill) and the per-group column loads at 4.  Swept, see REPORT.md.
DEFAULT_CFG = (48, 16, 1, 16)


def gather_qmm_int8(x_sorted, wq, scales_t, biases_t, qsum_t, expert_offsets,
                    tile_row, tile_exp, ntiles, maxt, cfg=DEFAULT_CFG):
    """y[R, N] = x_sorted[R, K] @ dequant(w[expert(row)])[N, K]^T on the tensor units.

    ``x_sorted`` must already be grouped by expert; ``expert_offsets[e]`` is the first
    row of expert ``e`` and ``expert_offsets[E]`` == R.  ``scales_t``/``biases_t``/
    ``qsum_t`` are the [E, G, N] tables from :func:`prepare_weights`.
    """
    TM, TN, SGM, SGN = cfg
    TMT, TNT = TM * SGM, TN * SGN
    R, K = x_sorted.shape
    G, N = scales_t.shape[1], scales_t.shape[2]
    xq, xsc, xrs, xav = quantize_activations(x_sorted, TMT)
    k = _kernel(cfg, K, N, G)
    (y,) = k(inputs=[xq, xsc, xrs, xav, wq, scales_t, biases_t, qsum_t,
                     expert_offsets, tile_row, tile_exp, ntiles],
             output_shapes=[(R, N)], output_dtypes=[mx.bfloat16],
             grid=(32 * SGM * SGN * (N // TNT), maxt, 1),
             threadgroup=(32 * SGM * SGN, 1, 1))
    return y


def max_tiles(R: int, E: int, TMT: int) -> int:
    """Static upper bound on sum(ceil(rows_e / TMT)); surplus tiles exit immediately."""
    return (R + TMT - 1) // TMT + E


def pick_cfg(N: int):
    """Widest threadgroup tile whose column count divides N."""
    TM, TN, SGM = DEFAULT_CFG[0], DEFAULT_CFG[1], DEFAULT_CFG[2]
    for SGN in (16, 8, 4, 2, 1):
        if N % (TN * SGN) == 0:
            return (TM, TN, SGM, SGN)
    return None


def prepare_weights(wq, scales, biases):
    """One-time per-tensor tables: [E, G, N] scales/biases plus the nibble sums.

    Transposing puts the columns a simdgroup needs for one affine group next to each
    other; with the stock [E, N, G] layout every column is its own cache line and the
    rescale's scale traffic dominates (measured: 1.35x -> 1.67x).
    """
    bs = biases.astype(mx.float32) + 8.0 * scales.astype(mx.float32)
    return (mx.contiguous(mx.swapaxes(scales.astype(mx.bfloat16), 1, 2)),
            mx.contiguous(mx.swapaxes(bs.astype(mx.bfloat16), 1, 2)),
            compute_qsum(wq))


# ---------------------------------------------------------------------------
# drop-in wrapper
# ---------------------------------------------------------------------------

_PREP: dict = {}


def _prepared(wq, scales, biases):
    key = (id(wq), wq.shape)
    hit = _PREP.get(key)
    if hit is None:
        hit = prepare_weights(wq, scales, biases)
        mx.eval(*hit)
        _PREP[key] = hit
    return hit


def clear_cache() -> None:
    _PREP.clear()


def supported(x, wq, scales, biases, rhs_indices, transpose, group_size, bits):
    if not (transpose and group_size == GROUP and bits == BITS):
        return False
    if biases is None or wq is None or rhs_indices is None:
        return False
    if wq.ndim != 3 or scales.ndim != 3 or rhs_indices.ndim != 1:
        return False
    if x.dtype != mx.bfloat16 or x.ndim != 3 or x.shape[1] != 1:
        return False
    if x.shape[0] != rhs_indices.shape[0]:
        return False
    if x.shape[-1] % GROUP:
        return False
    return pick_cfg(scales.shape[1]) is not None


def gather_qmm_sorted(x, w, s, b, rhs_indices, group_size=64, bits=4,
                      transpose=True, fallback=None):
    """Drop-in for ``mx.gather_qmm(..., sorted_indices=True)``.

    ``x`` is [R, 1, K] already sorted by expert, ``rhs_indices`` the matching sorted
    expert ids.  Unsupported bits/shapes go to ``fallback`` (default ``mx.gather_qmm``).
    """
    if not supported(x, w, s, b, rhs_indices, transpose, group_size, bits):
        fb = fallback or mx.gather_qmm
        return fb(x, w, s, b, rhs_indices=rhs_indices, transpose=transpose,
                  group_size=group_size, bits=bits, sorted_indices=True)
    E, N = w.shape[0], s.shape[1]
    cfg = pick_cfg(N)
    TMT = cfg[0] * cfg[2]
    R, K = x.shape[0], x.shape[-1]
    maxt = max_tiles(R, E, TMT)
    st, bt, qt = _prepared(w, s, b)
    offs, trow, texp, nt = build_tiles(rhs_indices, E, TMT, maxt)
    y = gather_qmm_int8(x.reshape(R, K), w, st, bt, qt, offs, trow, texp, nt, maxt, cfg)
    return y.reshape(R, 1, N)
