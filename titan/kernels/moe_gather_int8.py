# SPDX-License-Identifier: MIT
"""int8 x int4 sorted-gather MoE matmul on the tensor units.

Ported from ``engine/patches/moe-int8/kernel.py``.

The routed-expert GEMMs are most of prefill compute and normally go through
``mx.gather_qmm(..., sorted_indices=True)``, which dequantises weights to bf16
in threadgroup memory and feeds ``matmul2d`` bf16 x bf16. Two things are left
there: bf16 x bf16 peaks at 65.7 TFLOP/s on this machine while uint8 x uint4
peaks at 113 TOP/s, and feeding the packed nibbles straight to the tensor unit
removes the dequantise round trip and cuts weight traffic from 2 bytes to 0.5
per weight. The row axis is tiled ragged (see :mod:`titan.kernels.moe_gather_ws`
for the same argument) so an expert's weights stream once.

Unsigned, not signed. Re-centring the stored nibble with ``q ^ 8`` to use
signed int4 would produce a second copy of the weights, and the routed experts
*are* the model. So the stored unsigned nibbles are kept verbatim and the
activations are quantised to uint8 with a fixed zero point of 128::

    x[m,k] = sc[m,g] * (xu[m,k] - 128)
    w[n,k] = q[n,k] * s[n,g] + b[n,g]

    y[m,n] = sum_g sc[m,g] * s[n,g] * (D[m,n,g] - 128 * qsum[n,g])
                 + rowsum[m,g] * b[n,g]

``D`` is the uint8 x uint4 -> int32 tensor-unit dot product over the 64-wide
group and ``qsum[n,g] = sum_k q[n,k]`` is a per (expert, row, group) integer in
[0, 960]. ``D - 128*qsum`` is a difference of two large nearly-equal integers,
so it stays in the integer domain where it is exact; folding the correction
into a bf16 scale table loses about 0.4% of the product to cancellation.
``rowsum`` is the exact fp32 group sum of the *unquantised* activations, so the
affine bias term carries no activation-quantisation error.

Exactness: **not bit-exact, and does not try to be.** The activation
quantisation costs about 0.64 to 0.70% of output RMS, matching the dense int8
result on both random and real checkpoint tensors. This is the one op in the
library whose fast path changes the numbers, so it is opt-in: :func:`supports`
returns True only when the caller passes prepared tables, and the exactness
test asserts the relative-RMS bound rather than a ULP bound.

State: :class:`Int8Tables` holds the transposed [E, G, N] scales and biases and
the nibble sums, and is built once per weight tensor by :func:`build_tables`.
The caller owns it. The tile table is the caller's too. There is no hidden
per-tensor cache: ``qsum`` alone is about the size of the scales array, and a
module-level dict keyed on array identity is not a thing to hide from whoever
is accounting for memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from titan.kernels.moe_gather_ws import build_tiles
from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["BITS", "DEFAULT_CFG", "GROUP", "Int8Tables", "OP", "build_tables",
           "key", "metal", "pick_cfg", "quantize_activations", "reference",
           "supports"]

GROUP = 64
BITS = 4

# TM = 48 covers a whole expert's ~40 rows in one tile; TN = 16 keeps the
# per-lane accumulator at 24 floats. Swept in the report.
DEFAULT_CFG = (48, 16, 1, 16)   # TM, TN, SGM, SGN

_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_CACHE: dict = {}   # compiled kernels and the destination-layout probe only


def _probe_layout(TM: int, TN: int):
    """Element -> (row slot, col slot) map of the int32 destination tile."""
    hit = _CACHE.get(("probe", TM, TN))
    if hit is not None:
        return hit
    src = f"""
    const uint lane = thread_position_in_threadgroup.x;
    constexpr auto d = matmul2d_descriptor({TM}, {TN}, 64, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, metal::execution_simdgroup> op;
    auto ct = op.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>,
        int32_t>();
    uint C = ct.get_capacity();
    if (lane == 0) cap[0] = C;
    for (uint i = 0; i < C; ++i) {{
        auto ix = ct.get_multidimensional_index(i);
        out[(lane * 128 + i) * 2 + 0] = ix[0];
        out[(lane * 128 + i) * 2 + 1] = ix[1];
    }}
    """
    k = mx.fast.metal_kernel(name=f"titan_i8_probe_{TM}_{TN}", input_names=[],
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
# fused activation quantizer
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


def quantize_activations(x, pad: int):
    """uint8 codes (zero point 128), scales, exact group sums, and -8*code-sum."""
    k = _CACHE.get("quant")
    if k is None:
        k = mx.fast.metal_kernel(name="titan_i8_actquant", input_names=["x"],
                                 output_names=["xq", "xsc", "xrs", "xav"],
                                 header=_HEADER, source=_QUANT_SRC)
        _CACHE["quant"] = k
    R, K = x.shape
    G = K // GROUP
    RP = R + pad
    return k(inputs=[x], output_shapes=[(RP, K), (RP, G), (RP, G), (RP, G)],
             output_dtypes=[mx.uint8, mx.float32, mx.float32, mx.float32],
             grid=(((RP * G * 32 + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
             template=[("RSZ", R), ("RPAD", RP), ("KSZ", K), ("GSZ", G)])


# ---------------------------------------------------------------------------
# caller-owned weight tables
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


@dataclass(frozen=True)
class Int8Tables:
    """Per-weight-tensor tables. Built once by :func:`build_tables`, owned by
    the caller.

    Transposing to [E, G, N] puts the columns a simdgroup needs for one affine
    group next to each other; with the stock [E, N, G] layout every column is
    its own cache line and the rescale's scale traffic dominates (1.35x becomes
    1.67x). ``qsum`` costs one uint16 per (expert, out-row, group), about the
    size of the scales array. That is the price of not duplicating the weights,
    and it is the caller's to account for.
    """

    scales_t: mx.array   # [E, G, N] bf16
    biases_t: mx.array   # [E, G, N] bf16, already folded with 8 * scale
    qsum: mx.array       # [E, G, N] uint16

    @property
    def nbytes(self) -> int:
        return self.scales_t.nbytes + self.biases_t.nbytes + self.qsum.nbytes


def build_tables(wq, scales, biases) -> Int8Tables:
    """Transpose the scale tables and compute the nibble sums. One-time."""
    E, N, KW = wq.shape
    G = KW // 8
    total = E * N * G
    k = _CACHE.get("qsum")
    if k is None:
        k = mx.fast.metal_kernel(name="titan_i8_qsum", input_names=["wq"],
                                 output_names=["qsum"], header=_HEADER,
                                 source=_QSUM_SRC)
        _CACHE["qsum"] = k
    (qsum,) = k(inputs=[wq], output_shapes=[(E, G, N)], output_dtypes=[mx.uint16],
                grid=(((total + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                template=[("TOTAL", total), ("GSZ", G), ("NSZ", N)])
    folded = biases.astype(mx.float32) + 8.0 * scales.astype(mx.float32)
    tables = Int8Tables(
        scales_t=mx.contiguous(mx.swapaxes(scales.astype(mx.bfloat16), 1, 2)),
        biases_t=mx.contiguous(mx.swapaxes(folded.astype(mx.bfloat16), 1, 2)),
        qsum=qsum,
    )
    mx.eval(tables.scales_t, tables.biases_t, tables.qsum)
    return tables


# ---------------------------------------------------------------------------
# main kernel
# ---------------------------------------------------------------------------


def _source(cfg, K, N, G):
    TM, TN, SGM, SGN = cfg
    C, NR, NC, ranks, rep_row, rep_col = _probe_layout(TM, TN)
    TNT = TN * SGN
    consts = (
        f"constant uchar RI[{C}] = {{{', '.join(str(r) for r, _ in ranks)}}};\n"
        f"constant uchar CI[{C}] = {{{', '.join(str(c) for _, c in ranks)}}};\n"
        f"constant uchar RREP[{NR}] = {{{', '.join(str(i) for i in rep_row)}}};\n"
        f"constant uchar CREP[{NC}] = {{{', '.join(str(i) for i in rep_col)}}};\n"
    )
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

    constexpr auto qdesc = matmul2d_descriptor({TM}, {TN}, {GROUP}, false, true,
        false, matmul2d_descriptor::mode::multiply);
    matmul2d<qdesc, metal::execution_simdgroup> qop;
    auto ct0 = qop.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>,
        int32_t>();

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
        // [E, G, N] tables: the columns a lane owns are contiguous, so the
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
        auto tB = tensor<device metal::uint4b_format, dextents<int32_t,2>,
                         tensor_inline>(
            (device uchar*)wq_e + ((size_t)n0 * {K} + k0) / 2,
            dextents<int32_t,2>({GROUP}, {TN}), array<int32_t,2>{{1, {K}}});
        auto ct = qop.get_destination_cooperative_tensor<decltype(tA), decltype(tB),
                                                         int32_t>();
        qop.run(tA, tB, ct);

        #pragma unroll
        for (uint16_t i = 0; i < {C}; ++i) {{
            const uint r = RI[i], c = CI[i];
            // sc*s*sum_k (xu-128)*(q-8)  +  (8s+b)*rowsum_exact
            float tt = float(ct[i]) + av[r];
            acc[i] = fma(xv[r], fma(tt, sv[c], -qv[c]), acc[i]);
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
    key_ = (cfg, K, N, G)
    k = _CACHE.get(key_)
    if k is None:
        consts, src = _source(cfg, K, N, G)
        k = mx.fast.metal_kernel(
            name=f"titan_moe_i8i4_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}_k{K}_n{N}",
            input_names=["xq", "xsc", "xrs", "xav", "wq", "scales", "biases",
                         "qsum", "offsets", "tile_row", "tile_exp", "ntiles"],
            output_names=["y"], header=_HEADER + consts, source=src)
        _CACHE[key_] = k
    return k


def pick_cfg(N: int):
    """Widest threadgroup tile whose column count divides N."""
    TM, TN, SGM = DEFAULT_CFG[0], DEFAULT_CFG[1], DEFAULT_CFG[2]
    for SGN in (16, 8, 4, 2, 1):
        if N % (TN * SGN) == 0:
            return (TM, TN, SGM, SGN)
    return None


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(x, wq, scales, biases, rhs_indices, tables=None, tiles=None, *,
              group_size=GROUP, bits=BITS):
    """``y[r] = x[r] @ dequant(w[expert(r)]).T`` through MLX's own gather.

    ``tables`` and ``tiles`` are ignored here and exist so the two
    implementations share a signature.
    """
    return mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=rhs_indices, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )


def metal(x, wq, scales, biases, rhs_indices, tables=None, tiles=None, *,
          group_size=GROUP, bits=BITS):
    """uint8 x uint4 tensor-unit gather. Same signature as :func:`reference`.

    ``tables`` must be the caller's :class:`Int8Tables` for ``wq``; passing
    ``None`` builds them for this one call, which is correct and wasteful (they
    are large and meant to be built once per weight tensor).
    """
    E, N = int(wq.shape[0]), int(scales.shape[1])
    cfg = pick_cfg(N)
    if cfg is None:
        raise ValueError(f"no int8 tiling for N = {N}")
    TM, TN, SGM, SGN = cfg
    TMT = TM * SGM
    R, K = int(x.shape[0]), int(x.shape[-1])
    G = int(scales.shape[2])
    if tables is None:
        tables = build_tables(wq, scales, biases)
    if tiles is None:
        tiles = build_tiles(rhs_indices, E, TMT)
    if tiles.tile_rows != TMT or tiles.num_experts != E:
        raise ValueError("tile table does not match this call's tiling")
    xq, xsc, xrs, xav = quantize_activations(mx.contiguous(x.reshape(R, K)), TMT)
    k = _kernel(cfg, K, N, G)
    (y,) = k(
        inputs=[xq, xsc, xrs, xav, wq, tables.scales_t, tables.biases_t,
                tables.qsum, tiles.offsets, tiles.tile_row, tiles.tile_exp,
                tiles.ntiles],
        output_shapes=[(R, N)], output_dtypes=[mx.bfloat16],
        grid=(32 * SGM * SGN * (N // (TN * SGN)), tiles.maxt, 1),
        threadgroup=(32 * SGM * SGN, 1, 1),
    )
    return y.reshape(R, 1, N)


def key(x, wq, scales, biases, rhs_indices, tables=None, tiles=None, *,
        group_size=GROUP, bits=BITS) -> ShapeClass:
    return shape_class(x, wq, scales, biases, rhs_indices,
                       extra=(group_size, bits, tables is not None))


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) != 5 or len(k.extra) != 3:
        return False
    group_size, bits, has_tables = k.extra
    # opt-in: this is the one fast path that changes the numbers, so it is
    # never selected unless the caller has built and passed the tables.
    if not has_tables or (group_size, bits) != (GROUP, BITS):
        return False
    xs, ws, ss, _bs, idx = k.shapes
    if len(ws) != 3 or len(ss) != 3 or len(idx) != 1:
        return False
    if k.dtypes[0] != mx.bfloat16:
        return False
    if len(xs) != 3 or xs[1] != 1 or xs[0] != idx[0] or xs[-1] % GROUP:
        return False
    return pick_cfg(ss[1]) is not None


OP = KernelOp(
    name="moe_gather_int8",
    default_off=True,
    default_off_reason=(
        "ROUND4: -3.7% at 64k against the reference control, at pinned "
        "drafter depth, and its cold 65k prefill is below the control too, "
        "so nothing is traded for it. It also holds 11.3 GB of qsum tables "
        "(VENDORED.md) and quantises activations for 0.65% of output RMS."
    ),
    aliases=("moe_gather_gate_up",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=0.01,   # relative RMS, not ULPs: see the module docstring
    shapes=(
        {"R": 64, "K": 256, "N": 256, "E": 8, "bits": 4},
        {"R": 512, "K": 2560, "N": 1280, "E": 32, "bits": 4},
        {"R": 20480, "K": 2560, "N": 1280, "E": 512, "bits": 4},
    ),
    exactness="not exact: ~0.65% of output RMS from activation quantisation",
    source="engine/patches/moe-int8/REPORT.md section 5",
)
