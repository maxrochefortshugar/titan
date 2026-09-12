# SPDX-License-Identifier: MIT
"""Chunked Gated DeltaNet prefill scan (mlx PR #4020, C = 8).

Ported from ``engine/patches/round3/gdn-scan/kernel.py`` and
``kernel_nax.py``. The Metal body is mlx PR #4020's
``gated_delta_fused_chunk`` (ml-explore/mlx pull/4020, commit c7e1a2a,
``mlx/backend/metal/kernels/gated_delta_update.h``), with the template
parameters turned into mlx template args so it JITs against a pinned mlx
without a rebuild.

Semantics::

    q, k  : [B, T, Hk, Dk]
    v     : [B, T, Hv, Dv]
    g     : [B, T, Hv]        per-head decay, already in linear space
    beta  : [B, T, Hv]        already sigmoid'd
    state : [B, Hv, Dv, Dk]   float32, carried between chunks and into decode

    returns (y [B, T, Hv, Dv] in q.dtype, state_out [B, Hv, Dv, Dk] float32)

and the recurrence the reference spells out per token is

    S     <- g_t * S
    u     <- beta_t * (v_t - S @ k_t)
    S     <- S + outer(u, k_t)
    y_t   <- S @ q_t

Exactness: not bit-identical, and not claimed to be. Judged on the state,
which is what survives the chunk boundary. Against the exact per-token
recurrence in fp32, the C = 8 kernel sits at state rrmse 4e-7 to 6e-7 at
T = 64, 512 and 2048, two orders worse than a sequential scan and far under a
bf16 ULP; the error does not compound over chained chunks. The bar is state
rrmse under 1e-5, cleared 16x. That the WY inverse is conditioned at all
depends on ``k`` being L2-normalised, which bounds ``KK^T`` by 1.

The NAX variant (C = 16, cooperative-tensor matmul2d) is behind
``variant="nax"`` and is **not exact**: state rrmse 5.4e-4, which misses the
bar by 50x, and feeding it fp32 does not help, so it is the algorithm and not
the dtype. The cause is ``matmul2d_descriptor``'s ``relaxed_precision``
argument, which the PR and mlx's own steel NAX GEMM leave true; setting it
false yields garbage (rrmse 1.1), so the fragment copies are tied to the
relaxed layout. It is kept for measurement and must not be selected by
default; :func:`supports` never picks it.
"""

from __future__ import annotations

import os
import re

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["CHUNK", "CHUNK_NAX", "OP", "SUPPORTED_HEADS", "key", "metal",
           "nax", "reference", "supports"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_METAL = os.path.join(_HERE, "_metal")

CHUNK = 8
CHUNK_NAX = 16

# Head configurations PR #4020's gated_delta_update.metal instantiates.
SUPPORTED_HEADS = {(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)}

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

_KERNELS: dict[str, object] = {}


def _kernel():
    k = _KERNELS.get("c8")
    if k is None:
        k = mx.fast.metal_kernel(
            name="titan_gdn_chunk_scan_c8",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            header=_HEADER,
            source=_SOURCE,
        )
        _KERNELS["c8"] = k
    return k


# ---------------------------------------------------------------------------
# NAX variant, not exact, behind a flag
# ---------------------------------------------------------------------------


def _mlx_include() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(mx.__file__)), "include")


def _flatten(path: str, inc: str, seen: set) -> str:
    """mlx's JIT has no include search path, so the steel NAX header is
    flattened by recursively inlining its ``mlx/...`` includes."""
    p = os.path.normpath(path)
    if p in seen:
        return ""
    seen.add(p)
    out = []
    with open(p) as fh:
        for line in fh:
            m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
            out.append(_flatten(os.path.join(inc, m.group(1)), inc, seen) if m else line)
    return "".join(out)


def _nax_header(relaxed: bool) -> str:
    inc = _mlx_include()
    steel = _flatten(
        os.path.join(inc, "mlx/backend/metal/kernels/steel/gemm/nax.h"), inc, set()
    )
    macros = open(os.path.join(_METAL, "_nax_macros.h")).read()
    if not relaxed:
        macros = macros.replace("transpose_b, true, Mode)", "transpose_b, false, Mode)")
    return (
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "#include <metal_tensor>\n"
        "using namespace metal;\n"
        "using namespace mpp;\n"
        "using namespace mpp::tensor_ops;\n" + steel + macros
    )


def nax_available() -> bool:
    return os.path.exists(os.path.join(_METAL, "_nax_body.metal")) and os.path.exists(
        os.path.join(_METAL, "_nax_macros.h")
    )


def _nax_kernel(relaxed: bool):
    name = f"nax{'' if relaxed else '_precise'}"
    k = _KERNELS.get(name)
    if k is None:
        body = open(os.path.join(_METAL, "_nax_body.metal")).read()
        k = mx.fast.metal_kernel(
            name="titan_gdn_chunk_scan_" + name,
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            header=_nax_header(relaxed),
            source=body,
        )
        _KERNELS[name] = k
    return k


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(q, k, v, g, beta, state=None, *, variant="c8"):
    """The exact per-token recurrence, fp32, in plain MLX ops.

    Slow by construction: one small matmul chain per token. This is the
    definition of correct the chunked kernel is measured against.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    groups = Hv // Hk
    in_dtype = q.dtype
    S = (mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32) if state is None
         else state.astype(mx.float32))
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    g32 = g.astype(mx.float32)
    b32 = beta.astype(mx.float32)
    ys = []
    for t in range(T):
        # expand the shared K/V heads out to the value-head count
        kt = mx.repeat(k32[:, t], groups, axis=1)          # [B, Hv, Dk]
        qt = mx.repeat(q32[:, t], groups, axis=1)          # [B, Hv, Dk]
        vt = v32[:, t]                                     # [B, Hv, Dv]
        S = S * g32[:, t][..., None, None]
        kv = mx.sum(S * kt[:, :, None, :], axis=-1)        # [B, Hv, Dv]
        u = b32[:, t][..., None] * (vt - kv)
        S = S + u[..., None] * kt[:, :, None, :]
        ys.append(mx.sum(S * qt[:, :, None, :], axis=-1))  # [B, Hv, Dv]
    y = mx.stack(ys, axis=1).astype(in_dtype)
    return y, S


def metal(q, k, v, g, beta, state=None, *, variant="c8"):
    """PR #4020's chunked scan. ``variant`` is ``"c8"`` (exact enough, the
    default) or ``"nax"`` / ``"nax_precise"`` (measurement only, not exact)."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    # fp32 gates: g sits near 1.0 where bf16 has only 8 mantissa bits, and a
    # fixed dtype keeps the JIT signature stable across callers.
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    if variant == "c8":
        kernel, chunk = _kernel(), CHUNK
    elif variant in ("nax", "nax_precise"):
        kernel, chunk = _nax_kernel(variant == "nax"), CHUNK_NAX
    else:
        raise ValueError(f"unknown gdn_chunk_scan variant {variant!r}")
    return kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk),
                  ("Hv", Hv), ("C", chunk)],
        grid=(32, Dv // chunk, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[in_dtype, mx.float32],
    )


def nax(q, k, v, g, beta, state=None, *, relaxed=True):
    """The C = 16 NAX variant. Not exact; see the module docstring."""
    return metal(q, k, v, g, beta, state,
                 variant="nax" if relaxed else "nax_precise")


def key(q, k, v, g, beta, state=None, *, variant="c8") -> ShapeClass:
    return shape_class(q, k, v, g, beta, state, extra=(variant,))


def supports(sc: ShapeClass) -> bool:
    if sc.device != "gpu" or len(sc.shapes) < 5:
        return False
    # the NAX variant is never selected automatically: it misses the bar
    if sc.extra and sc.extra[0] != "c8":
        return False
    qs, ks, vs, gs = sc.shapes[0], sc.shapes[1], sc.shapes[2], sc.shapes[3]
    if len(qs) != 4 or len(vs) != 4 or len(gs) != 3 or ks != qs:
        return False
    _b, t, hk, dk = qs
    hv, dv = vs[-2], vs[-1]
    if dk != 128 or dv != 128 or (hk, hv) not in SUPPORTED_HEADS or hv % hk:
        return False
    if sc.dtypes[0] not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if sc.dtypes[1] != sc.dtypes[0] or sc.dtypes[2] != sc.dtypes[0]:
        return False
    if len(sc.shapes) > 5 and sc.dtypes[5] != mx.float32:
        return False
    return t > 1


OP = KernelOp(
    name="gdn_chunk_scan",
    aliases=("gdn_scan_chunked",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=1e-5,   # state rrmse against the fp32 per-token recurrence
    shapes=(
        {"B": 1, "T": 16, "Hk": 16, "Hv": 48, "Dk": 128, "Dv": 128},
        {"B": 1, "T": 64, "Hk": 16, "Hv": 48, "Dk": 128, "Dv": 128},
        {"B": 1, "T": 2048, "Hk": 16, "Hv": 48, "Dk": 128, "Dv": 128},
    ),
    exactness="state rrmse ~6e-7 vs the fp32 per-token recurrence "
              "(bar 1e-5); NAX variant 5.4e-4, not exact",
    source="engine/patches/round3/gdn-scan/REPORT.md section 4",
)
