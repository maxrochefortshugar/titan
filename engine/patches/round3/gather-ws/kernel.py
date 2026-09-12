# SPDX-License-Identifier: Apache-2.0
"""Weight-stationary bf16 sorted-gather MoE prefill matmul for M5 NAX.

Replaces ``mx.gather_qmm(..., sorted_indices=True)`` on the routed-expert
prefill GEMMs of Qwen3.8-Flash-Next.  Same operands as stock (int4 affine
weights, bf16 activations, fp32 accumulate), same output layout and dtype,
**zero precomputed tables**.

What stock does wrong (AUDIT-2026-09-12 section C): the sorted NAX path
``affine_gather_qmm_rhs_nax`` picks BM=32 when ``rows_per_expert < 64``.  A
2048-token top-10 chunk gives ~40 rows per expert, so every expert runs two
32-row tiles and **every expert weight is dequantised and streamed twice**.

What this kernel does: a device-built ragged tile table gives one tile per
expert whose row extent (TM=48) covers the whole expert for the overwhelming
majority of experts, so each expert's int4 tile is read from DRAM, dequantised
into threadgroup bf16 once, and consumed by every row of that expert before it
is evicted.  Experts with more rows than TM (the skew tail) get further tiles;
experts with fewer waste only the unused row slots of one tile.

Per (tile, column block) threadgroup, for each 64-wide affine group g:

    1. cooperatively unpack ``TNT x 64`` nibbles, apply ``q*s + b``, write bf16
       into ``threadgroup Ws[TNT][64]``;
    2. every simdgroup runs one ``matmul2d<TM, TN, 64>`` of the device bf16
       activation tile against its own slice of ``Ws``, fp32 destination,
       ``relaxed_precision = false``, ``mode::multiply_accumulate`` so the
       running sum lives in the cooperative tensor across all 40 groups.

Because each simdgroup stages only the columns it consumes, the handoff needs a
``simdgroup_barrier``, not a threadgroup one, and the device loads for group
``g+1`` are hoisted into registers above the matmul for group ``g``.

No activation quantisation, no scale transposition, no nibble-sum table: the
arithmetic is bit-for-bit the same operation stock performs, only the summation
order over the 40 groups and the tiling differ.
"""

from __future__ import annotations

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
# destination-layout probe (fp32 destination, bf16 x bf16)
# ---------------------------------------------------------------------------

def _probe_layout(TM: int, TN: int):
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
        tensor<device bfloat, dextents<int32_t,2>, tensor_inline>,
        tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>, float>();
    uint C = ct.get_capacity();
    if (lane == 0) cap[0] = C;
    for (uint i = 0; i < C; ++i) {{
        auto ix = ct.get_multidimensional_index(i);
        out[(lane * 128 + i) * 2 + 0] = ix[0];
        out[(lane * 128 + i) * 2 + 1] = ix[1];
    }}
    """
    k = mx.fast.metal_kernel(name=f"ws_probe_{TM}_{TN}", input_names=[],
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
    rep_row = [next(i for i in range(C) if ranks[i][0] == j) for j in range(len(rows))]
    rep_col = [next(i for i in range(C) if ranks[i][1] == j) for j in range(len(cols))]
    hit = (C, len(rows), len(cols), ranks, rep_row, rep_col)
    _CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------
# expert row offsets + ragged tile table (device side, no host sync)
# ---------------------------------------------------------------------------

_TILE_SRC = """
    const uint t = thread_position_in_grid.x;
    for (uint tt = t; tt < NEXP + 1; tt += THREADS) {
        const uint e_ = tt;                      // first row whose expert id >= e_
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
    """Segment boundaries plus a ragged (row_start, expert) tile list, on device."""
    k = _CACHE.get("tiles")
    if k is None:
        k = mx.fast.metal_kernel(
            name="ws_tiles", input_names=["idx"],
            output_names=["offsets", "tile_row", "tile_exp", "ntiles"],
            header=_HEADER, source=_TILE_SRC)
        _CACHE["tiles"] = k
    R = indices.shape[0]
    tg = 1024
    return k(inputs=[indices.astype(mx.uint32)],
             output_shapes=[(E + 1,), (maxt,), (maxt,), (1,)],
             output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
             grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
             template=[("NEXP", E), ("ROWS", R), ("TMT", TMT), ("MAXT", maxt), ("THREADS", tg)])


def max_tiles(R: int, E: int, TMT: int) -> int:
    """Static upper bound on sum(ceil(rows_e / TMT)); surplus tiles exit at once."""
    return (R + TMT - 1) // TMT + E


# ---------------------------------------------------------------------------
# main kernel
# ---------------------------------------------------------------------------




def _MM_TAIL(ACC: int, C: int) -> str:
    if ACC:
        return "      qop.run(tA, tB, acc); }"
    bs = " \\" + "\n"
    return (
        "      auto ct = qop.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();" + bs
        + "      qop.run(tA, tB, ct);" + bs
        + '      _Pragma("unroll")' + bs
        + f"      for (uint16_t i = 0; i < {C}; ++i) accr[i] += ct[i]; " + "}")


def _ACC_DECL(ACC: int, C: int) -> str:
    """Accumulator.  ACC=1 accumulates inside the cooperative tensor
    (mode::multiply_accumulate), which removes C fp32 adds per lane per group;
    ACC=0 keeps a separate register array and a plain multiply."""
    if ACC:
        return ("    auto acc = qop.get_destination_cooperative_tensor<\n"
                "        tensor<device bfloat, dextents<int32_t,2>, tensor_inline>,\n"
                "        tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>, float>();\n"
                f"    #pragma unroll\n"
                f"    for (uint i = 0; i < {C}; ++i) acc[i] = 0.0f;")
    return (f"    float accr[{C}];\n"
            f"    #pragma unroll\n"
            f"    for (uint i = 0; i < {C}; ++i) accr[i] = 0.0f;")


def _PIPE_SRC(DB: int, G: int) -> str:
    """Software-pipelined stage/consume loop.  DB=1 double-buffers the staging
    slice so the next group's dequantise-and-store overlaps this group's matmul;
    DB=0 keeps one buffer and only hoists the device loads above the matmul."""
    if DB:
        return f"""
    WS_FETCH(0u)
    WS_STORE(0u)
    for (uint g = 0; g < {G}; ++g) {{
        if (g + 1 < {G}) WS_FETCH(g + 1)
        simdgroup_barrier(mem_flags::mem_threadgroup);
        WS_MM(g & 1u, g)
        if (g + 1 < {G}) {{
            simdgroup_barrier(mem_flags::mem_threadgroup);
            WS_STORE((g + 1) & 1u)
        }}
    }}"""
    return f"""
    WS_FETCH(0u)
    for (uint g = 0; g < {G}; ++g) {{
        simdgroup_barrier(mem_flags::mem_threadgroup);
        WS_STORE(0u)
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (g + 1 < {G}) WS_FETCH(g + 1)
        WS_MM(0u, g)
    }}"""


def _source(cfg, K, N, G):
    TM, TN, SGN, DB, ACC = cfg
    C, NR, NC, ranks, rep_row, rep_col = _probe_layout(TM, TN)
    TNT = TN * SGN
    NBUF = 2 if DB else 1
    WPL = TN * 8 // 32
    MODE = 'multiply_accumulate' if ACC else 'multiply'
    ACCN = 'acc' if ACC else 'accr'
    ri = ", ".join(str(r) for r, _ in ranks)
    ci = ", ".join(str(c) for _, c in ranks)
    rr = ", ".join(str(i) for i in rep_row)
    rc = ", ".join(str(i) for i in rep_col)
    consts = (f"constant uchar RI[{C}] = {{{ri}}};\n"
              f"constant uchar CI[{C}] = {{{ci}}};\n"
              f"constant uchar RREP[{NR}] = {{{rr}}};\n"
              f"constant uchar CREP[{NC}] = {{{rc}}};\n")
    return consts, f"""
    threadgroup bfloat Ws[{TNT} * {GROUP} * {NBUF}];

    const uint t = threadgroup_position_in_grid.y;
    if (t >= ntiles[0]) return;
    const uint e      = tile_exp[t];
    const uint row0   = tile_row[t];
    const uint rowend = offsets[e + 1];

    const uint tid  = thread_position_in_threadgroup.x;
    const uint sgn  = tid / 32;
    // last tile of the whole array: slide the row window back so the activation
    // tensor never reads past row R.  Rows below row0 are computed and discarded.
    const uint m0 = metal::min(row0, uint(ROWS) - {TM}u);
    const uint nt0 = threadgroup_position_in_grid.x * {TNT};
    const uint n0  = nt0 + sgn * {TN};

    constexpr auto qdesc = matmul2d_descriptor({TM}, {TN}, {GROUP}, false, true, false,
        matmul2d_descriptor::mode::{MODE});
    matmul2d<qdesc, metal::execution_simdgroup> qop;
    auto ct0 = qop.get_destination_cooperative_tensor<
        tensor<device bfloat, dextents<int32_t,2>, tensor_inline>,
        tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>, float>();

    ushort mrow[{NR}], ncol[{NC}];
    #pragma unroll
    for (uint j = 0; j < {NR}; ++j) mrow[j] = ct0.get_multidimensional_index(RREP[j])[1];
    #pragma unroll
    for (uint j = 0; j < {NC}; ++j) ncol[j] = ct0.get_multidimensional_index(CREP[j])[0];

{_ACC_DECL(ACC, C)}

    const device uint*   wq_e = (const device uint*)wq + (size_t)e * ({N} * {K} / 8);
    const device bfloat* sc_e = (const device bfloat*)scales + (size_t)e * {N} * {G};
    const device bfloat* bi_e = (const device bfloat*)biases + (size_t)e * {N} * {G};

    // each simdgroup owns its own [TN x 64] slice of the staging buffer, so the
    // stage/consume handoff needs only a simdgroup barrier, not a threadgroup one.
    const uint lane = tid & 31;
    threadgroup bfloat* Wsg = Ws + sgn * ({TN} * {GROUP} * {NBUF});
    const device uint* wq_c = wq_e + (size_t)n0 * ({K} / 8);

    // per-lane register staging of the WPL = TN*8/32 uints this lane unpacks
    uint pk[{WPL}];
    float sf[{WPL}], bf_[{WPL}];

#define WS_FETCH(gg)                                                              \\
    {{ _Pragma("unroll")                                                          \\
      for (uint w = 0; w < {WPL}; ++w) {{                                         \\
        const uint i = lane + w * 32;                                             \\
        const uint col = i >> 3, j = i & 7;                                       \\
        pk[w] = wq_c[(size_t)col * ({K} / 8) + (gg) * 8 + j];                      \\
        sf[w] = float(sc_e[(size_t)(n0 + col) * {G} + (gg)]);                      \\
        bf_[w] = float(bi_e[(size_t)(n0 + col) * {G} + (gg)]);                     \\
      }} }}

#define WS_STORE(buf)                                                             \\
    {{ _Pragma("unroll")                                                          \\
      for (uint w = 0; w < {WPL}; ++w) {{                                         \\
        const uint i = lane + w * 32;                                             \\
        const uint col = i >> 3, j = i & 7;                                       \\
        bfloat4 v0, v1;                                                           \\
        _Pragma("unroll")                                                         \\
        for (uint u = 0; u < 4; ++u) {{                                           \\
            v0[u] = bfloat(fma(float((pk[w] >> (4 * u)) & 0xf), sf[w], bf_[w]));   \\
            v1[u] = bfloat(fma(float((pk[w] >> (4*u+16)) & 0xf), sf[w], bf_[w]));  \\
        }}                                                                        \\
        threadgroup bfloat4* dst = (threadgroup bfloat4*)                         \\
            (Wsg + (buf) * ({TN} * {GROUP}) + col * {GROUP} + j * 8);              \\
        dst[0] = v0; dst[1] = v1;                                                 \\
      }} }}

#define WS_MM(buf, gg)                                                            \\
    {{ auto tA = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(       \\
          (device bfloat*)x + (size_t)m0 * {K} + (gg) * {GROUP},                  \\
          dextents<int32_t,2>({GROUP}, {TM}), array<int32_t,2>{{1, {K}}});        \\
      auto tB = tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>(   \\
          Wsg + (buf) * ({TN} * {GROUP}),                                          \\
          dextents<int32_t,2>({GROUP}, {TN}), array<int32_t,2>{{1, {GROUP}}});    \\
{_MM_TAIL(ACC, C)}

{_PIPE_SRC(DB, G)}
#undef WS_FETCH
#undef WS_STORE
#undef WS_MM

    #pragma unroll
    for (uint16_t i = 0; i < {C}; ++i) {{
        uint m = m0 + mrow[RI[i]];
        if (m >= row0 && m < rowend)
            y[(size_t)m * {N} + n0 + ncol[CI[i]]] = bfloat({ACCN}[i]);
    }}
"""


def _kernel(cfg, K, N, G):
    key = (cfg, K, N, G)
    k = _CACHE.get(key)
    if k is None:
        consts, src = _source(cfg, K, N, G)
        k = mx.fast.metal_kernel(
            name=f"moe_ws_bf16_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}{cfg[4]}_k{K}_n{N}",
            input_names=["x", "wq", "scales", "biases", "offsets",
                         "tile_row", "tile_exp", "ntiles"],
            output_names=["y"], header=_HEADER + consts, source=src)
        _CACHE[key] = k
    return k


# TM=48 covers a whole expert's ~40 rows in one tile so the weight tile is
# streamed once; TN=16 keeps the fp32 destination at 24 registers per lane.
# SGN=8 keeps the threadgroup staging buffer at 16 KB.  Swept, see REPORT.md.
DEFAULT_CFG = (48, 16, 16, 0, 1)


def pick_cfg(N: int, cfg=DEFAULT_CFG):
    """Widest column block that divides N."""
    TM, TN, SGN, DB, ACC = cfg
    for s in (SGN, 8, 4, 2, 1):
        if s <= SGN and N % (TN * s) == 0:
            return (TM, TN, s, DB, ACC)
    return None


def gather_ws(x_sorted, wq, scales, biases, offsets, tile_row, tile_exp,
              ntiles, maxt, cfg=DEFAULT_CFG):
    """y[R, N] = x_sorted[R, K] @ dequant(w[expert(row)])[N, K]^T, weight-stationary."""
    TM, TN, SGN, DB, ACC = cfg
    TNT = TN * SGN
    R, K = x_sorted.shape
    N, G = scales.shape[1], scales.shape[2]
    k = _kernel(cfg, K, N, G)
    (y,) = k(inputs=[x_sorted, wq, scales, biases, offsets, tile_row, tile_exp, ntiles],
             output_shapes=[(R, N)], output_dtypes=[mx.bfloat16],
             grid=(32 * SGN * (N // TNT), maxt, 1),
             threadgroup=(32 * SGN, 1, 1),
             template=[("ROWS", R)])
    return y


_TILES: dict = {}


def _tiles_memo(indices, E: int, TMT: int, maxt: int):
    """Both projections of one SwitchGLU share ``rhs_indices``, so segment the
    sorted indices once per layer instead of once per matmul.  The key array is
    kept alive by the cache entry, so its ``id`` cannot be recycled underneath
    us.  Two entries, which covers gate_up plus down of the layer in flight."""
    key = (id(indices), indices.shape, E, TMT)
    hit = _TILES.get(key)
    if hit is not None:
        return hit[1]
    out = build_tiles(indices, E, TMT, maxt)
    if len(_TILES) >= 2:
        _TILES.pop(next(iter(_TILES)))
    _TILES[key] = (indices, out)
    return out


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
    if x.shape[-1] % GROUP or scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        return False
    if x.shape[0] < DEFAULT_CFG[0]:
        return False
    return pick_cfg(scales.shape[1]) is not None


def gather_qmm_sorted(x, w, s, b, rhs_indices, group_size=64, bits=4,
                      transpose=True, fallback=None):
    """Drop-in for ``mx.gather_qmm(..., sorted_indices=True)``.

    ``x`` is [R, 1, K] already sorted by expert, ``rhs_indices`` the matching
    sorted expert ids.  Unsupported shapes go to ``fallback``.
    """
    if not supported(x, w, s, b, rhs_indices, transpose, group_size, bits):
        fb = fallback or mx.gather_qmm
        return fb(x, w, s, b, rhs_indices=rhs_indices, transpose=transpose,
                  group_size=group_size, bits=bits, sorted_indices=True)
    E, N = w.shape[0], s.shape[1]
    cfg = pick_cfg(N)
    TMT = cfg[0]
    R, K = x.shape[0], x.shape[-1]
    maxt = max_tiles(R, E, TMT)
    offs, trow, texp, nt = _tiles_memo(rhs_indices, E, TMT, maxt)
    y = gather_ws(mx.contiguous(x.reshape(R, K)), w, s, b, offs, trow, texp, nt, maxt, cfg)
    return y.reshape(R, 1, N)


def clear_cache() -> None:
    """No per-tensor state is held; this drops only the tile-table memo."""
    _TILES.clear()
