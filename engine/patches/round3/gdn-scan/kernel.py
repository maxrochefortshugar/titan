# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Gated DeltaNet chunked prefill scan, ported from mlx PR #4020 to a JIT
``mx.fast.metal_kernel`` so it runs on the app's pinned mlx 0.32.2 without a rebuild.

Source: ml-explore/mlx pull/4020, commit c7e1a2a,
``mlx/backend/metal/kernels/gated_delta_update.h`` :: ``gated_delta_fused_chunk``.
The template parameters (InT, Dk, Dv, Hk, Hv, C) become mlx template args, the
``[[kernel]]`` signature and buffer indices are supplied by metal_kernel, and the
macro block is inlined verbatim.

Semantics are identical to ``mlx_lm.models.gated_delta.gated_delta_kernel``:
    q, k : [B, T, Hk, Dk]   bf16/fp16/fp32
    v    : [B, T, Hv, Dv]
    g    : [B, T, Hv]       per-head decay in linear space (already exp'd)
    beta : [B, T, Hv]       already sigmoid'd
    state: [B, Hv, Dv, Dk]  float32, carried between prefill chunks and into decode
returns (y [B, T, Hv, Dv] in q.dtype, state_out [B, Hv, Dv, Dk] float32).
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

#define AT(TILE, IDX) TILE.thread_elements()[IDX]
#define SUB(TILE0, TILE1, TILE2)                \\
  {                                             \\
    AT(TILE0, 0) = AT(TILE1, 0) - AT(TILE2, 0); \\
    AT(TILE0, 1) = AT(TILE1, 1) - AT(TILE2, 1); \\
  }
#define FMA(TILE0, S, TILE1, TILE2)                 \\
  {                                                 \\
    AT(TILE0, 0) = S * AT(TILE1, 0) + AT(TILE2, 0); \\
    AT(TILE0, 1) = S * AT(TILE1, 1) + AT(TILE2, 1); \\
  }
#define SCALE(TILE0, S) \\
  {                     \\
    AT(TILE0, 0) *= S;  \\
    AT(TILE0, 1) *= S;  \\
  }
#define SCALE2(TILE0, S0, S1) \\
  {                           \\
    AT(TILE0, 0) *= S0;       \\
    AT(TILE0, 1) *= S1;       \\
  }
#define SCALE_TRI(TILE0, S0, S1)            \\
  {                                         \\
    AT(TILE0, 0) *= fn > fm ? 0.f : S0;     \\
    AT(TILE0, 1) *= fn + 1 > fm ? 0.f : S1; \\
  }
#define SCALE_TRIEQ(TILE0, S0, S1)           \\
  {                                          \\
    AT(TILE0, 0) *= fn >= fm ? 0.f : S0;     \\
    AT(TILE0, 1) *= fn + 1 >= fm ? 0.f : S1; \\
  }

#define LOAD_M(M, SRC, LD, B)                                                  \\
  if (B) {                                                                     \\
    AT(M, 0) =                                                                 \\
        static_cast<float>((fm < valid_rows) ? ((SRC)[fm * (LD) + fn]) : 0.f); \\
    AT(M, 1) = static_cast<float>(                                             \\
        (fm < valid_rows) ? ((SRC)[fm * (LD) + fn + 1]) : 0.f);                \\
  } else {                                                                     \\
    AT(M, 0) = static_cast<float>((SRC)[fm * (LD) + fn]);                      \\
    AT(M, 1) = static_cast<float>((SRC)[fm * (LD) + fn + 1]);                  \\
  }

#define LOAD_MT(M, SRC, LD, B)                                                 \\
  if (B) {                                                                     \\
    AT(M, 0) =                                                                 \\
        static_cast<float>((fn < valid_rows) ? ((SRC)[fn * (LD) + fm]) : 0.f); \\
    AT(M, 1) = static_cast<float>(                                             \\
        (fn + 1 < valid_rows) ? ((SRC)[(fn + 1) * (LD) + fm]) : 0.f);          \\
  } else {                                                                     \\
    AT(M, 0) = static_cast<float>((SRC)[fn * (LD) + fm]);                      \\
    AT(M, 1) = static_cast<float>((SRC)[(fn + 1) * (LD) + fm]);                \\
  }

#define PROCESS_CHUNK_SG(B, S_tile, VALID)                                     \\
  {                                                                            \\
    const short valid_rows = (VALID);                                          \\
                                                                               \\
    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)               \\
        ? metal::fast::log(                                                    \\
              metal::max(                                                      \\
                  static_cast<float>(                                          \\
                      g_[thread_index_in_simdgroup * Hv + hv_idx]),            \\
                  1e-6f))                                                      \\
        : 0.0f;                                                                \\
                                                                               \\
    float gamma_val = simd_prefix_inclusive_sum(g_val);                        \\
                                                                               \\
    if (thread_index_in_simdgroup < C) {                                       \\
      gamma[thread_index_in_simdgroup] = gamma_val;                            \\
    }                                                                          \\
    simdgroup_barrier(mem_flags::mem_threadgroup);                             \\
                                                                               \\
    float gamma_fm = metal::fast::exp(gamma[fm]);                              \\
    float gamma_fmdfn = metal::fast::exp(gamma[fm] - gamma[fn]);               \\
    float gamma_fmdfn1 = metal::fast::exp(gamma[fm] - gamma[fn + 1]);          \\
    float gamma_Cdfn = metal::fast::exp(gamma[C - 1] - gamma[fn]);             \\
    float gamma_Cdfn1 = metal::fast::exp(gamma[C - 1] - gamma[fn + 1]);        \\
    float gamma_C = metal::fast::exp(gamma[C - 1]);                            \\
                                                                               \\
    float beta_fm = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;        \\
                                                                               \\
    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \\
    MLX_MTL_PRAGMA_UNROLL                                                      \\
    for (int kk = 0; kk < Dk; kk += 8) {                                       \\
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \\
      LOAD_MT(KT_tile, k_ + kk, Dk * Hk, B)                                    \\
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);      \\
    }                                                                          \\
                                                                               \\
    KKtK_tile = KKt_tile;                                                      \\
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)                                   \\
                                                                               \\
    simdgroup_float8x8 Tinv, P;                                                \\
    AT(P, 0) = AT(KKtK_tile, 0);                                               \\
    AT(P, 1) = AT(KKtK_tile, 1);                                               \\
    SUB(Tinv, I_tile, KKtK_tile)                                               \\
                                                                               \\
    MLX_MTL_PRAGMA_UNROLL                                                      \\
    for (int step = 1; (1 << step) < C; step++) {                              \\
      simdgroup_multiply(P, P, P);                                             \\
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);                      \\
    }                                                                          \\
                                                                               \\
    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                     \\
    MLX_MTL_PRAGMA_UNROLL                                                      \\
    for (int kk = 0; kk < Dk; kk += 8) {                                       \\
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \\
      SCALE(K_tile, beta_fm)                                                   \\
      simdgroup_multiply(W_tile, Tinv, K_tile);                                \\
      SCALE(W_tile, gamma_fm)                                                  \\
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile); \\
    }                                                                          \\
                                                                               \\
    SCALE_TRI(Tinv, gamma_fmdfn, gamma_fmdfn1)                                 \\
                                                                               \\
    LOAD_M(V_tile, v_ + dv_idx, Dv * Hv, B)                                    \\
    SCALE(V_tile, beta_fm)                                                     \\
    simdgroup_multiply(U_tile, Tinv, V_tile);                                  \\
    SUB(delta_tile, U_tile, WS_tile)                                           \\
                                                                               \\
    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \\
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \\
    MLX_MTL_PRAGMA_UNROLL                                                      \\
    for (int kk = 0; kk < Dk; kk += 8) {                                       \\
      LOAD_M(Q_tile, q_ + kk, Hk * Dk, B)                                      \\
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \\
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);       \\
      SCALE(Q_tile, gamma_fm)                                                  \\
      simdgroup_multiply_accumulate(                                           \\
          tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);                         \\
    }                                                                          \\
                                                                               \\
    SCALE_TRI(QKt_tile, gamma_fmdfn, gamma_fmdfn1)                             \\
                                                                               \\
    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);   \\
                                                                               \\
    if (fm < valid_rows) {                                                     \\
      y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));       \\
      y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));   \\
    }                                                                          \\
                                                                               \\
    MLX_MTL_PRAGMA_UNROLL                                                      \\
    for (int kk = 0; kk < Dk; kk += 8) {                                       \\
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \\
      SCALE2(K_tile, gamma_Cdfn, gamma_Cdfn1)                                  \\
      simdgroup_multiply(KD_tile, K_tile, delta_tile);                         \\
      FMA(S_tile[kk / 8], gamma_C, S_tile[kk / 8], KD_tile)                    \\
    }                                                                          \\
  }
"""

_SOURCE = """
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  const short qid = thread_index_in_simdgroup / 4;
  const short fm = (qid & 4) + ((thread_index_in_simdgroup / 2) % 4);
  const short fn = (qid & 2) * 2 + (thread_index_in_simdgroup % 2) * 2;

  auto dv_idx = thread_position_in_grid.y * 8;
  const short sg_id = thread_position_in_threadgroup.y;

  auto g_ = g + b_idx * T * Hv;
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  simdgroup_float8x8 S_tile[Dk / 8];
  simdgroup_float8x8 V_tile, K_tile, KT_tile, Q_tile;
  simdgroup_float8x8 W_tile, U_tile;
  simdgroup_float8x8 WS_tile;
  simdgroup_float8x8 delta_tile;
  simdgroup_float8x8 tmp_tile;
  simdgroup_float8x8 QKt_tile;
  simdgroup_float8x8 out_tile;
  simdgroup_float8x8 KD_tile;
  simdgroup_float8x8 KKtK_tile, KKt_tile;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  simdgroup_float8x8 I_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
  AT(I_tile, 0) = (fm == fn) ? 1.0f : 0.0f;
  AT(I_tile, 1) = (fm == fn + 1) ? 1.0f : 0.0f;

  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_load(S_tile[kk / 8], i_state + kk, Dk, ulong2(0, 0), true);
  }

  int t = 0;
  for (; t + C <= T; t += C) {
    PROCESS_CHUNK_SG(false, S_tile, C);
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    PROCESS_CHUNK_SG(true, S_tile, short(T - t));
  }

  MLX_MTL_PRAGMA_UNROLL
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_store(S_tile[kk / 8], o_state + kk, Dk, ulong2(0, 0), true);
  }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="pr4020_gated_delta_fused_chunk",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


# Head configurations instantiated by PR #4020's gated_delta_update.metal.
SUPPORTED_HEADS = {(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)}
CHUNK = 8


def supported(q, k, v, g, beta, state) -> bool:
    if q.ndim != 4 or v.ndim != 4 or g.ndim != 3:
        return False
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if Dk != 128 or Dv != 128:
        return False
    if (Hk, Hv) not in SUPPORTED_HEADS:
        return False
    if Hv % Hk != 0:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if state is not None and state.dtype != mx.float32:
        return False
    return T > 1


def gated_delta_fused_chunk(q, k, v, g, beta, state=None):
    """PR #4020 chunked delta-rule scan. Drop-in for mlx_lm gated_delta_kernel."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    # fp32 gates: g sits near 1.0 where bf16 has only 8 mantissa bits, and a
    # fixed dtype keeps the JIT signature stable across callers.
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    return _kernel()(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", in_dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("C", CHUNK),
        ],
        grid=(32, Dv // CHUNK, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[in_dtype, mx.float32],
    )
