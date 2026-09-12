# SPDX-License-Identifier: MIT
"""Weight-stationary bf16 sorted-gather MoE matmul.

Ported from ``engine/patches/round3/gather-ws/kernel.py``.

Replaces ``mx.gather_qmm(..., sorted_indices=True)`` on the routed-expert
prefill GEMMs: same operands (int4 affine weights, bf16 activations, fp32
accumulate), same output layout and dtype, no precomputed tables.

What the stock sorted NAX path leaves on the table: it picks BM = 32 when
``rows_per_expert < 64``, and a 2048-token top-10 chunk gives about 40 rows per
expert, so every expert runs two 32-row tiles and every expert weight is
dequantised and streamed twice. This kernel builds a ragged tile table on the
device so one tile (TM = 48) covers a whole expert for the overwhelming
majority of experts: each expert's int4 tile is read from DRAM, dequantised
into threadgroup bf16 once, and consumed by every row of that expert before it
is evicted. Experts past TM rows get further tiles; experts under it waste only
the unused row slots of one tile.

Per (tile, column block) threadgroup, for each 64-wide affine group: unpack
``TN*SGN x 64`` nibbles cooperatively, apply ``q*s + b``, write bf16 into
threadgroup memory; then every simdgroup runs one ``matmul2d<TM, TN, 64>`` of
the device bf16 activation tile against its own slice, fp32 destination,
``relaxed_precision = false``, ``mode::multiply_accumulate`` so the running sum
lives in the cooperative tensor across all groups. Each simdgroup stages only
the columns it consumes, so the handoff needs a simdgroup barrier and not a
threadgroup one.

Exactness: bit-identical to the reference at every shape tested. That is not
luck: both paths dequantise to bf16 and feed bf16 x bf16 tensor ops with an
fp32 accumulator, and the summation order over the 64-wide groups is the same.
No activation quantisation, no scale transposition, no nibble-sum table.

State: the ragged tile table is a :class:`TileTable`, built by
:func:`build_tiles` and owned by the caller. Both projections of one SwitchGLU
share ``rhs_indices``, so the caller segments once per layer instead of once
per matmul; the module keeps no cache of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["DEFAULT_CFG", "GROUP", "BITS", "OP", "TileTable", "build_tiles",
           "key", "max_tiles", "metal", "pick_cfg", "reference", "supports"]

GROUP = 64
BITS = 4

# TM = 48 covers a whole expert's ~40 rows in one tile so the weight tile is
# streamed once; TN = 16 keeps the fp32 destination at 24 registers per lane;
# SGN = 16 keeps the threadgroup staging buffer small. Swept in the report.
DEFAULT_CFG = (48, 16, 16, 0, 1)   # TM, TN, SGN, double-buffer, accumulate-in-CT

_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_CACHE: dict = {}   # compiled kernels and the destination-layout probe only


# ---------------------------------------------------------------------------
# destination-layout probe (fp32 destination, bf16 x bf16)
# ---------------------------------------------------------------------------


def _probe_layout(TM: int, TN: int):
    """Element -> (row slot, col slot) map of the matmul2d destination tile.

    The absolute row/col a lane owns is lane dependent, but the *rank* of an
    element's row among that lane's distinct rows is uniform across the
    simdgroup, which is verified here for every element and every lane. Baking
    the ranks in as compile-time constants keeps the epilogue's register arrays
    statically indexed.
    """
    hit = _CACHE.get(("probe", TM, TN))
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
    k = mx.fast.metal_kernel(name=f"titan_ws_probe_{TM}_{TN}", input_names=[],
                             output_names=["out", "cap"], header=_HEADER, source=src)
    out, cap = k(inputs=[], output_shapes=[(32 * 128 * 2,), (1,)],
                 output_dtypes=[mx.int32, mx.int32],
                 grid=(32, 1, 1), threadgroup=(32, 1, 1))
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
            raise RuntimeError(
                f"matmul2d destination layout is not simd-uniform for {TM}x{TN}"
            )
    rep_row = [next(i for i in range(C) if ranks[i][0] == j) for j in range(len(rows))]
    rep_col = [next(i for i in range(C) if ranks[i][1] == j) for j in range(len(cols))]
    hit = (C, len(rows), len(cols), ranks, rep_row, rep_col)
    _CACHE[("probe", TM, TN)] = hit
    return hit


# ---------------------------------------------------------------------------
# caller-owned tile table
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


@dataclass(frozen=True)
class TileTable:
    """Segment boundaries plus a ragged (row_start, expert) tile list.

    Built on the device with no host sync. The caller owns it and reuses it
    across both projections of one SwitchGLU, which share ``rhs_indices``.
    """

    offsets: mx.array     # [E + 1] uint32, first row of each expert
    tile_row: mx.array    # [maxt] uint32
    tile_exp: mx.array    # [maxt] uint32
    ntiles: mx.array      # [1] uint32
    maxt: int
    tile_rows: int
    num_experts: int


def max_tiles(rows: int, num_experts: int, tile_rows: int) -> int:
    """Static upper bound on sum(ceil(rows_e / tile_rows)); surplus tiles exit."""
    return (rows + tile_rows - 1) // tile_rows + num_experts


def build_tiles(rhs_indices, num_experts: int, tile_rows: int = DEFAULT_CFG[0]
                ) -> TileTable:
    """Segment the sorted expert ids into a :class:`TileTable`, on the device."""
    k = _CACHE.get("tiles")
    if k is None:
        k = mx.fast.metal_kernel(
            name="titan_ws_tiles", input_names=["idx"],
            output_names=["offsets", "tile_row", "tile_exp", "ntiles"],
            header=_HEADER, source=_TILE_SRC)
        _CACHE["tiles"] = k
    rows = int(rhs_indices.shape[0])
    maxt = max_tiles(rows, num_experts, tile_rows)
    tg = 1024
    offsets, tile_row, tile_exp, ntiles = k(
        inputs=[rhs_indices.astype(mx.uint32)],
        output_shapes=[(num_experts + 1,), (maxt,), (maxt,), (1,)],
        output_dtypes=[mx.uint32] * 4,
        grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
        template=[("NEXP", num_experts), ("ROWS", rows), ("TMT", tile_rows),
                  ("MAXT", maxt), ("THREADS", tg)])
    return TileTable(offsets, tile_row, tile_exp, ntiles, maxt, tile_rows,
                     num_experts)


# ---------------------------------------------------------------------------
# main kernel source
# ---------------------------------------------------------------------------


def _MM_TAIL(ACC: int, C: int) -> str:
    if ACC:
        return "      qop.run(tA, tB, acc); }"
    bs = " \\" + "\n"
    return (
        "      auto ct = qop.get_destination_cooperative_tensor<decltype(tA), "
        "decltype(tB), float>();" + bs
        + "      qop.run(tA, tB, ct);" + bs
        + '      _Pragma("unroll")' + bs
        + f"      for (uint16_t i = 0; i < {C}; ++i) accr[i] += ct[i]; " + "}")


def _ACC_DECL(ACC: int, C: int) -> str:
    """ACC = 1 accumulates inside the cooperative tensor
    (``mode::multiply_accumulate``), removing C fp32 adds per lane per group."""
    if ACC:
        return ("    auto acc = qop.get_destination_cooperative_tensor<\n"
                "        tensor<device bfloat, dextents<int32_t,2>, tensor_inline>,\n"
                "        tensor<threadgroup bfloat, dextents<int32_t,2>, "
                "tensor_inline>, float>();\n"
                "    #pragma unroll\n"
                f"    for (uint i = 0; i < {C}; ++i) acc[i] = 0.0f;")
    return (f"    float accr[{C}];\n"
            "    #pragma unroll\n"
            f"    for (uint i = 0; i < {C}; ++i) accr[i] = 0.0f;")


def _PIPE_SRC(DB: int, G: int) -> str:
    """DB = 1 double-buffers the staging slice so the next group's
    dequantise-and-store overlaps this group's matmul; DB = 0 keeps one buffer
    and only hoists the device loads above the matmul."""
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
    MODE = "multiply_accumulate" if ACC else "multiply"
    ACCN = "acc" if ACC else "accr"
    consts = (
        f"constant uchar RI[{C}] = {{{', '.join(str(r) for r, _ in ranks)}}};\n"
        f"constant uchar CI[{C}] = {{{', '.join(str(c) for _, c in ranks)}}};\n"
        f"constant uchar RREP[{NR}] = {{{', '.join(str(i) for i in rep_row)}}};\n"
        f"constant uchar CREP[{NC}] = {{{', '.join(str(i) for i in rep_col)}}};\n"
    )
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
    // tensor never reads past row R. Rows below row0 are computed and discarded.
    const uint m0 = metal::min(row0, uint(ROWS) - {TM}u);
    const uint nt0 = threadgroup_position_in_grid.x * {TNT};
    const uint n0  = nt0 + sgn * {TN};

    constexpr auto qdesc = matmul2d_descriptor({TM}, {TN}, {GROUP}, false, true,
        false, matmul2d_descriptor::mode::{MODE});
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
    // stage/consume handoff needs only a simdgroup barrier.
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
    key_ = (cfg, K, N, G)
    k = _CACHE.get(key_)
    if k is None:
        consts, src = _source(cfg, K, N, G)
        k = mx.fast.metal_kernel(
            name=f"titan_moe_ws_bf16_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}{cfg[4]}"
                 f"_k{K}_n{N}",
            input_names=["x", "wq", "scales", "biases", "offsets",
                         "tile_row", "tile_exp", "ntiles"],
            output_names=["y"], header=_HEADER + consts, source=src)
        _CACHE[key_] = k
    return k


def pick_cfg(N: int, cfg=DEFAULT_CFG):
    """Widest column block that divides N, or None."""
    TM, TN, SGN, DB, ACC = cfg
    for s in (SGN, 8, 4, 2, 1):
        if s <= SGN and N % (TN * s) == 0:
            return (TM, TN, s, DB, ACC)
    return None


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(x, wq, scales, biases, rhs_indices, tiles=None, *,
              group_size=GROUP, bits=BITS):
    """``y[r] = x[r] @ dequant(w[expert(r)]).T`` through MLX's own gather.

    ``x`` is [R, 1, K] already sorted by expert, ``rhs_indices`` the matching
    sorted expert ids. ``tiles`` is ignored here and exists so the two
    implementations share a signature.
    """
    return mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=rhs_indices, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )


def metal(x, wq, scales, biases, rhs_indices, tiles=None, *,
          group_size=GROUP, bits=BITS):
    """Weight-stationary tiled gather. Same signature as :func:`reference`.

    ``tiles`` is the caller's :class:`TileTable`; when omitted one is built for
    this call, which is correct but wastes the segmentation the sibling
    projection could have shared.
    """
    E, N = int(wq.shape[0]), int(scales.shape[1])
    cfg = pick_cfg(N)
    if cfg is None:
        raise ValueError(f"no weight-stationary tiling for N = {N}")
    R, K = int(x.shape[0]), int(x.shape[-1])
    if tiles is None:
        tiles = build_tiles(rhs_indices, E, cfg[0])
    if tiles.tile_rows != cfg[0] or tiles.num_experts != E:
        raise ValueError("tile table does not match this call's tiling")
    TM, TN, SGN, _DB, _ACC = cfg
    G = int(scales.shape[2])
    k = _kernel(cfg, K, N, G)
    (y,) = k(
        inputs=[mx.contiguous(x.reshape(R, K)), wq, scales, biases,
                tiles.offsets, tiles.tile_row, tiles.tile_exp, tiles.ntiles],
        output_shapes=[(R, N)], output_dtypes=[mx.bfloat16],
        grid=(32 * SGN * (N // (TN * SGN)), tiles.maxt, 1),
        threadgroup=(32 * SGN, 1, 1),
        template=[("ROWS", R)],
    )
    return y.reshape(R, 1, N)


def key(x, wq, scales, biases, rhs_indices, tiles=None, *,
        group_size=GROUP, bits=BITS) -> ShapeClass:
    return shape_class(x, wq, scales, biases, rhs_indices,
                       extra=(group_size, bits))


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) != 5 or len(k.extra) != 2:
        return False
    if k.extra != (GROUP, BITS):
        return False
    xs, ws, ss, bs, idx = k.shapes
    xd, _wd, sd, bd, _id = k.dtypes
    if len(ws) != 3 or len(ss) != 3 or len(idx) != 1:
        return False
    if xd != mx.bfloat16 or sd != mx.bfloat16 or bd != mx.bfloat16:
        return False
    if len(xs) != 3 or xs[1] != 1 or xs[0] != idx[0]:
        return False
    if xs[-1] % GROUP or xs[0] < DEFAULT_CFG[0]:
        return False
    return pick_cfg(ss[1]) is not None


OP = KernelOp(
    name="moe_gather_ws",
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=0.0,
    shapes=(
        {"R": 64, "K": 256, "N": 256, "E": 8, "bits": 4},
        {"R": 512, "K": 2560, "N": 1280, "E": 32, "bits": 4},
        {"R": 20480, "K": 2560, "N": 1280, "E": 512, "bits": 4},
    ),
    exactness="bit-identical",
    source="engine/patches/round3/gather-ws/REPORT.md section 2",
)
