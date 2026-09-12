# SPDX-License-Identifier: Apache-2.0
"""JIT Metal kernels for single-token MoE decode fusion (affine 4-bit, gs 64/128).

Two kernels replace the five-kernel routed-expert chain at T<=2:

  moe_gate_up_silu : gather fused gate+up expert matvec + SiLU*mul
                     x[T,K] , W[E,2I,K] -> h[T,topk,I]     (fp32 accumulation)
  moe_down_wsum    : gather down expert matvec + router-weighted sum
                     h[T,topk,I], W[E,D,I] -> y[T,D]       (fp32 accumulation)

Both read MLX affine-quantized weights (uint32 packed, 8 nibbles/word,
w = scale*q + bias) and dequantize on the fly.  One SIMD group produces one
output element; the K reduction is a simd_sum.
"""
from __future__ import annotations

import mlx.core as mx

SIMD = 32
TG = 256  # threads per threadgroup (8 simd groups)

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# ---------------------------------------------------------------- gate+up ---
# One simd group -> one output element h[t, j, i].
# It reduces two expert rows at once: gate row i and up row I+i, sharing the
# activation loads and the per-word sum of x.
_SRC_GATE_UP = """
  const uint lane = thread_position_in_threadgroup.x & 31u;
  const uint sgid = thread_position_in_grid.x >> 5;          // global simd id
  if (sgid >= (uint)(NT * TOPK * I)) return;

  const uint i  = sgid % (uint)I;
  const uint jt = sgid / (uint)I;
  const uint j  = jt % (uint)TOPK;
  const uint t  = jt / (uint)TOPK;

  const uint e  = idx[t * TOPK + j];
  const uint NW = (uint)(K / 8);                 // packed words per row
  const uint NG = (uint)(K / GS);                // quant groups per row
  const uint WPG = (uint)(GS / 8);               // words per quant group

  const ulong rg = (ulong)e * (ulong)(2 * I) + (ulong)i;          // gate row
  const ulong ru = rg + (ulong)I;                                 // up row
  device const uint* wg = w + rg * (ulong)NW;
  device const uint* wu = w + ru * (ulong)NW;
  device const ST* sg = scales + rg * (ulong)NG;
  device const ST* su = scales + ru * (ulong)NG;
  device const ST* bg = biases + rg * (ulong)NG;
  device const ST* bu = biases + ru * (ulong)NG;
  device const XT* xp = x + (ulong)t * (ulong)K;

  float acc_g = 0.0f, acc_u = 0.0f;
  for (uint word = lane; word < NW; word += 32u) {
    const uint g = word / WPG;
    const uint base = word * 8u;
    float xs[8];
    float sx = 0.0f;
    {
      device const vec<XT, 4>* xv = (device const vec<XT, 4>*)(xp + base);
      vec<XT, 4> x0 = xv[0], x1 = xv[1];
      xs[0]=(float)x0[0]; xs[1]=(float)x0[1]; xs[2]=(float)x0[2]; xs[3]=(float)x0[3];
      xs[4]=(float)x1[0]; xs[5]=(float)x1[1]; xs[6]=(float)x1[2]; xs[7]=(float)x1[3];
      sx = ((xs[0]+xs[1])+(xs[2]+xs[3])) + ((xs[4]+xs[5])+(xs[6]+xs[7]));
    }

    const uint vg = wg[word];
    const uint vu = wu[word];
    float qg = 0.0f, qu = 0.0f;
    #pragma unroll
    for (uint c = 0; c < 8u; ++c) {
      qg = fma(xs[c], (float)((vg >> (4u * c)) & 0xFu), qg);
      qu = fma(xs[c], (float)((vu >> (4u * c)) & 0xFu), qu);
    }
    acc_g = fma((float)sg[g], qg, fma((float)bg[g], sx, acc_g));
    acc_u = fma((float)su[g], qu, fma((float)bu[g], sx, acc_u));
  }

  acc_g = simd_sum(acc_g);
  acc_u = simd_sum(acc_u);
  if (lane == 0) {
    float s = acc_g / (1.0f + metal::exp(-acc_g));   // SiLU
    h[sgid] = (OT)(s * acc_u);
  }
"""

# ------------------------------------------------------- down + weighted sum ---
# One simd group -> one output element y[t, d]; loops the topk experts and
# folds the router score into the lane-local partial so only one simd_sum runs.
_SRC_DOWN_WSUM = """
  const uint lane = thread_position_in_threadgroup.x & 31u;
  const uint gsid = thread_position_in_grid.x >> 5;      // global simd id
  const uint block = gsid;                               // one block of ROWS d
  const uint nblk = (uint)(D / ROWS);
  if (block >= (uint)NT * nblk) return;
  const uint t  = block / nblk;
  const uint d0 = (block % nblk) * (uint)ROWS;

  constexpr uint NW = (uint)(I / 8);
  constexpr uint NG = (uint)(I / GS);
  constexpr uint WPG = (uint)(GS / 8);
  constexpr uint NITER = (NW + 31u) / 32u;

  float acc[ROWS];
  #pragma unroll
  for (uint r = 0; r < (uint)ROWS; ++r) acc[r] = 0.0f;

  const uint ibase = t * (uint)TOPK;
  for (uint j = 0; j < (uint)TOPK; ++j) {
    const uint e = idx[ibase + j];
    const float sc = scores[ibase + j];
    device const HT* hp = h + ((ulong)t * (ulong)TOPK + (ulong)j) * (ulong)I;

    // The lane's slice of h is the same for every output row, so load it once
    // per expert and reuse it across the ROWS rows of the block.
    float hs[NITER][8];
    float sh[NITER];
    #pragma unroll
    for (uint it = 0; it < NITER; ++it) {
      const uint word = lane + 32u * it;
      if (word < NW) {
        device const vec<HT, 4>* hv = (device const vec<HT, 4>*)(hp + word * 8u);
        vec<HT, 4> h0 = hv[0], h1 = hv[1];
        hs[it][0]=(float)h0[0]; hs[it][1]=(float)h0[1];
        hs[it][2]=(float)h0[2]; hs[it][3]=(float)h0[3];
        hs[it][4]=(float)h1[0]; hs[it][5]=(float)h1[1];
        hs[it][6]=(float)h1[2]; hs[it][7]=(float)h1[3];
        sh[it] = ((hs[it][0]+hs[it][1])+(hs[it][2]+hs[it][3]))
               + ((hs[it][4]+hs[it][5])+(hs[it][6]+hs[it][7]));
      } else {
        #pragma unroll
        for (uint c = 0; c < 8u; ++c) hs[it][c] = 0.0f;
        sh[it] = 0.0f;
      }
    }

    const ulong ebase = (ulong)e * (ulong)D + (ulong)d0;
    #pragma unroll
    for (uint r = 0; r < (uint)ROWS; ++r) {
      device const uint* wp = w + (ebase + r) * (ulong)NW;
      device const ST* sp = scales + (ebase + r) * (ulong)NG;
      device const ST* bp = biases + (ebase + r) * (ulong)NG;
      float a = 0.0f;
      #pragma unroll
      for (uint it = 0; it < NITER; ++it) {
        const uint word = lane + 32u * it;
        if (word < NW) {
          const uint v = wp[word];
          const uint g = word / WPG;
          float q = 0.0f;
          #pragma unroll
          for (uint c = 0; c < 8u; ++c)
            q = fma(hs[it][c], (float)((v >> (4u * c)) & 0xFu), q);
          a = fma((float)sp[g], q, fma((float)bp[g], sh[it], a));
        }
      }
      acc[r] = fma(sc, a, acc[r]);
    }
  }

  #pragma unroll
  for (uint r = 0; r < (uint)ROWS; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0) y[t * (uint)D + d0 + r] = (OT)v;
  }
"""

_K_GATE_UP = mx.fast.metal_kernel(
    name="omlx_moe_gate_up_silu",
    input_names=["x", "w", "scales", "biases", "idx"],
    output_names=["h"],
    header=_HEADER,
    source=_SRC_GATE_UP,
    ensure_row_contiguous=True,
)

_K_DOWN_WSUM = mx.fast.metal_kernel(
    name="omlx_moe_down_wsum",
    input_names=["h", "w", "scales", "biases", "idx", "scores"],
    output_names=["y"],
    header=_HEADER,
    source=_SRC_DOWN_WSUM,
    ensure_row_contiguous=True,
)


def _grid(n_simd):
    return (n_simd * SIMD, 1, 1)


def gate_up_silu(x, weight, scales, biases, idx, group_size=64, out_dtype=mx.float32):
    """x[T,K] (bf16/fp16), fused gate_up weight [E,2I,K] -> h[T,topk,I]."""
    T, K = x.shape
    twoI = weight.shape[1]
    I = twoI // 2
    topk = idx.shape[-1]
    n = T * topk * I
    (h,) = _K_GATE_UP(
        inputs=[x, weight, scales, biases, idx],
        output_shapes=[(T, topk, I)],
        output_dtypes=[out_dtype],
        grid=_grid(n),
        threadgroup=(TG, 1, 1),
        template=[("XT", x.dtype), ("ST", scales.dtype), ("OT", out_dtype),
                  ("GS", group_size), ("NT", T), ("K", K), ("I", I), ("TOPK", topk)],
    )
    return h


def down_wsum(h, weight, scales, biases, idx, scores, group_size=64,
              out_dtype=mx.bfloat16, rows=2):
    """h[T,topk,I], down weight [E,D,I] , scores[T,topk] fp32 -> y[T,D]."""
    T, topk, I = h.shape
    D = weight.shape[1]
    n = T * (D // rows)
    (y,) = _K_DOWN_WSUM(
        inputs=[h, weight, scales, biases, idx, scores],
        output_shapes=[(T, D)],
        output_dtypes=[out_dtype],
        grid=_grid(n),
        threadgroup=(TG, 1, 1),
        template=[("HT", h.dtype), ("ST", scales.dtype), ("OT", out_dtype),
                  ("GS", group_size), ("NT", T), ("I", I), ("D", D),
                  ("TOPK", topk), ("ROWS", rows)],
    )
    return y


# ------------------------------------------------------------------ router ---
# R1: quantized router matvec, one simd group per (token, expert) logit.
_SRC_ROUTER_LOGITS = """
  const uint lane = thread_position_in_threadgroup.x & 31u;
  const uint gsid = thread_position_in_grid.x >> 5;
  if (gsid >= (uint)(NT * E)) return;
  const uint e = gsid % (uint)E;
  const uint t = gsid / (uint)E;

  constexpr uint NW = (uint)(K / 8);
  constexpr uint WPG = (uint)(GS / 8);

  device const uint* wp = w + (ulong)e * (ulong)NW;
  device const ST* sp = scales + (ulong)e * (ulong)(K / GS);
  device const ST* bp = biases + (ulong)e * (ulong)(K / GS);
  device const XT* xp = x + (ulong)t * (ulong)K;

  float acc = 0.0f;
  for (uint word = lane; word < NW; word += 32u) {
    const uint base = word * 8u;
    device const vec<XT, 4>* xv = (device const vec<XT, 4>*)(xp + base);
    vec<XT, 4> x0 = xv[0], x1 = xv[1];
    float xs[8];
    xs[0]=(float)x0[0]; xs[1]=(float)x0[1]; xs[2]=(float)x0[2]; xs[3]=(float)x0[3];
    xs[4]=(float)x1[0]; xs[5]=(float)x1[1]; xs[6]=(float)x1[2]; xs[7]=(float)x1[3];
    const float sx = ((xs[0]+xs[1])+(xs[2]+xs[3])) + ((xs[4]+xs[5])+(xs[6]+xs[7]));
    const uint v = wp[word];
    float q = 0.0f;
    #pragma unroll
    for (uint c = 0; c < 8u; ++c)
      q = fma(xs[c], (float)((v >> (4u * c)) & 0xFu), q);
    const uint g = word / WPG;
    acc = fma((float)sp[g], q, fma((float)bp[g], sx, acc));
  }
  acc = simd_sum(acc);
  if (lane == 0) logits[gsid] = acc;
"""

# R2: top-k + score normalisation, one simd group per token.
# Selection is by logit, which is order-isomorphic to softmax, so the result
# matches argpartition(softmax(.)) exactly whenever there are no exact ties.
_SRC_ROUTER_TOPK = """
  const uint lane = thread_position_in_threadgroup.x & 31u;
  const uint t = thread_position_in_grid.x >> 5;
  if (t >= (uint)NT) return;

  constexpr uint PER = (uint)E / 32u;
  float vals[PER];
  device const float* lp = logits + (ulong)t * (ulong)E;
  #pragma unroll
  for (uint c = 0; c < PER; ++c) vals[c] = lp[lane * PER + c];

  float lmax = -INFINITY;
  #pragma unroll
  for (uint c = 0; c < PER; ++c) lmax = max(lmax, vals[c]);
  const float gmax = simd_max(lmax);

  float lsum = 0.0f;
  #pragma unroll
  for (uint c = 0; c < PER; ++c) lsum += metal::precise::exp(vals[c] - gmax);
  const float gsum = simd_sum(lsum);

  float sv[TOPK];
  for (uint j = 0; j < (uint)TOPK; ++j) {
    float best = -INFINITY;
    uint bl = 0;
    #pragma unroll
    for (uint c = 0; c < PER; ++c) {
      if (vals[c] > best) { best = vals[c]; bl = c; }
    }
    const float m = simd_max(best);
    const uint cand = (best == m) ? (lane * PER + bl) : 0xFFFFFFFFu;
    const uint sel = simd_min(cand);
    if (cand == sel) vals[bl] = -INFINITY;
    sv[j] = m;
    if (lane == 0) idx[t * (uint)TOPK + j] = sel;
  }

  if (lane == 0) {
    if (NORM) {
      float s = 0.0f;
      float p[TOPK];
      #pragma unroll
      for (uint j = 0; j < (uint)TOPK; ++j) {
        p[j] = metal::precise::exp(sv[j] - sv[0]);
        s += p[j];
      }
      #pragma unroll
      for (uint j = 0; j < (uint)TOPK; ++j) scores[t * (uint)TOPK + j] = p[j] / s;
    } else {
      #pragma unroll
      for (uint j = 0; j < (uint)TOPK; ++j)
        scores[t * (uint)TOPK + j] = metal::precise::exp(sv[j] - gmax) / gsum;
    }
  }
"""

_K_ROUTER_LOGITS = mx.fast.metal_kernel(
    name="omlx_moe_router_logits",
    input_names=["x", "w", "scales", "biases"],
    output_names=["logits"],
    header=_HEADER,
    source=_SRC_ROUTER_LOGITS,
    ensure_row_contiguous=True,
)

_K_ROUTER_TOPK = mx.fast.metal_kernel(
    name="omlx_moe_router_topk",
    input_names=["logits"],
    output_names=["idx", "scores"],
    header=_HEADER,
    source=_SRC_ROUTER_TOPK,
    ensure_row_contiguous=True,
)


def router_logits(x, weight, scales, biases, group_size=64):
    T, K = x.shape
    E = weight.shape[0]
    (lg,) = _K_ROUTER_LOGITS(
        inputs=[x, weight, scales, biases],
        output_shapes=[(T, E)],
        output_dtypes=[mx.float32],
        grid=_grid(T * E),
        threadgroup=(TG, 1, 1),
        template=[("XT", x.dtype), ("ST", scales.dtype), ("GS", group_size),
                  ("NT", T), ("K", K), ("E", E)],
    )
    return lg


def router_topk(logits, topk, norm_topk_prob=True):
    T, E = logits.shape
    idx, scores = _K_ROUTER_TOPK(
        inputs=[logits],
        output_shapes=[(T, topk), (T, topk)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=_grid(T),
        threadgroup=(SIMD, 1, 1),
        template=[("NT", T), ("E", E), ("TOPK", topk),
                  ("NORM", bool(norm_topk_prob))],
    )
    return idx, scores
