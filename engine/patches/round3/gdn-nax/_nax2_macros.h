




// NAX MACROS I can probably do a nice template instead of doing this
// fm = base_fm + (idx >> 2) * 8;   // idx>>2 = idx/4  -> 0 for idx 0-3, 1 for
// idx 4-7 fn = base_fn + (idx % 4);        // 4 consecutive columns
#define AT_NAX(TILE, IDX) TILE.elems()[IDX]

#define SUB_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) - AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define ADD_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) + AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define FMA_NAX(TILE0, S, TILE1, TILE2)                                     \
  {                                                                         \
    STEEL_PRAGMA_UNROLL                                                     \
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) { \
      (TILE0)[_i] = (S) * (TILE1)[_i] + (TILE2)[_i];                        \
    }                                                                       \
  }

#define SCALE_NAX(TILE0, S)                                         \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) *= (S);                                     \
    }                                                               \
  }

#define SCALE_ROW_NAX(TILE0, S)                                            \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) *=                                                 \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
  }

#define SCALE_BETA_NAX(TILE0, BETA2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (BETA2)[_w >> 2];                        \
    }                                                               \
  }

#define SCALE2_NAX(TILE0, GAMMA)                                              \
  {                                                                           \
    STEEL_PRAGMA_UNROLL                                                       \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {           \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;           \
      const short _fm = mlx::steel::BaseNAXFrag::get_coord(_w).y;             \
      AT_NAX(TILE0, _i) *= metal::fast::exp((GAMMA)[(C) - 1] - (GAMMA)[_fm]); \
    }                                                                         \
  }

#define SCALE_TRI_NAX(TILE0, GAMMA)                                            \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x > _c.y)                                       \
          ? 0.f                                                                \
          : metal::fast::exp((GAMMA)[_c.y] - (GAMMA)[_c.x]);                   \
    }                                                                          \
  }

#define SCALE_TRIEQ_NAX1(TILE0, BETA)                                          \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x >= _c.y) ? 0.f : (BETA)[_i >> 2];             \
    }                                                                          \
  }


// ---------------------------------------------------------------------------
// Split-precision NAX matmul helpers.
//
// matmul2d with relaxed_precision=true rounds both operands to an 11-bit
// significand (measured: probe_format.py) and accumulates in fp32. Splitting
// each operand as x = hi + lo with hi = x masked to 11 significand bits, both
// exactly representable, and running hi*hi + hi*lo + lo*hi recovers ~22 bits.
// SPLIT=false reproduces the PR's original single pass exactly.
// ---------------------------------------------------------------------------

METAL_FUNC float nax_hi(float x) {
  return as_type<float>(as_type<uint>(x) & 0xFFFFE000u);
}

// bf16 operand storage halves the cooperative-tensor register footprint; the
// hi part is then only 8 significand bits, so the split carries ~16 bits.
template <bool BF>
METAL_FUNC float nax_hi_g(float x) {
  return as_type<float>(as_type<uint>(x) & (BF ? 0xFFFF0000u : 0xFFFFE000u));
}

namespace mlx {
namespace steel {

// M=16, N=16, K=32: A and B each two K-fragments, single 16x16 C.
template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,
    short SPLIT = 0,
    typename OpT = float>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      16,
      32,
      transpose_a,
      transpose_b,
      true,
      (SPLIT != 0)
          ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
          : Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  constexpr bool acc =
      Mode == mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate;
  // mode 0 none, 1 three-pass, 2 correct the left operand, 3 correct the right
  constexpr bool bf = metal::is_same_v<OpT, bfloat>;
  constexpr bool ANY = SPLIT != 0;
  constexpr bool SA = SPLIT == 1 || SPLIT == 2;
  constexpr bool SB = SPLIT == 1 || SPLIT == 3;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = OpT(SA ? nax_hi_g<bf>(A0[i]) : A0[i]);
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = OpT(SA ? nax_hi_g<bf>(A1[i]) : A1[i]);
    ct_b[i] = OpT(SB ? nax_hi_g<bf>(B0[i]) : B0[i]);
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(SB ? nax_hi_g<bf>(B1[i]) : B1[i]);
    ct_c[i] = (ANY && !acc) ? CType(0) : C[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  if (SB) { // + hi_a * lo_b
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_b[i] = OpT(B0[i] - nax_hi_g<bf>(B0[i]));
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(B1[i] - nax_hi_g<bf>(B1[i]));
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }
  if (SA) { // + lo_a * hi_b
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = OpT(A0[i] - nax_hi_g<bf>(A0[i]));
      ct_a[BaseNAXFrag::kElemsPerFrag + i] = OpT(A1[i] - nax_hi_g<bf>(A1[i]));
      ct_b[i] = OpT(SB ? nax_hi_g<bf>(B0[i]) : B0[i]);
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(SB ? nax_hi_g<bf>(B1[i]) : B1[i]);
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

// M=16, N=32, K=16 padded to carry a single 16x16 product.
template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,
    short SPLIT = 0,
    typename OpT = float>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      transpose_a,
      transpose_b,
      true,
      (SPLIT != 0)
          ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
          : Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  constexpr bool acc =
      Mode == mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate;
  // mode 0 none, 1 three-pass, 2 correct the left operand, 3 correct the right
  constexpr bool bf = metal::is_same_v<OpT, bfloat>;
  constexpr bool ANY = SPLIT != 0;
  constexpr bool SA = SPLIT == 1 || SPLIT == 2;
  constexpr bool SB = SPLIT == 1 || SPLIT == 3;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = OpT(SA ? nax_hi_g<bf>(A[i]) : A[i]);
    ct_b[i] = OpT(SB ? nax_hi_g<bf>(B[i]) : B[i]);
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(0.0);
    ct_c[i] = (ANY && !acc) ? CType(0) : C[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  if (SB) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_b[i] = OpT(B[i] - nax_hi_g<bf>(B[i]));
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }
  if (SA) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = OpT(A[i] - nax_hi_g<bf>(A[i]));
      ct_b[i] = OpT(SB ? nax_hi_g<bf>(B[i]) : B[i]);
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

// M=16, N=32, K=16: single A, B and C two N-fragments each.
template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,
    short SPLIT = 0,
    typename OpT = float>
METAL_FUNC static constexpr void mman(
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn0,
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn1,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      transpose_a,
      transpose_b,
      true,
      (SPLIT != 0)
          ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
          : Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<OpT, OpT, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  constexpr bool acc =
      Mode == mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate;
  // mode 0 none, 1 three-pass, 2 correct the left operand, 3 correct the right
  constexpr bool bf = metal::is_same_v<OpT, bfloat>;
  constexpr bool ANY = SPLIT != 0;
  constexpr bool SA = SPLIT == 1 || SPLIT == 2;
  constexpr bool SB = SPLIT == 1 || SPLIT == 3;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = OpT(SA ? nax_hi_g<bf>(A[i]) : A[i]);
    ct_b[i] = OpT(SB ? nax_hi_g<bf>(Bn0[i]) : Bn0[i]);
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(SB ? nax_hi_g<bf>(Bn1[i]) : Bn1[i]);
    ct_c[i] = (ANY && !acc) ? CType(0) : Cn0[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = (ANY && !acc) ? CType(0) : Cn1[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  if (SB) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_b[i] = OpT(Bn0[i] - nax_hi_g<bf>(Bn0[i]));
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(Bn1[i] - nax_hi_g<bf>(Bn1[i]));
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }
  if (SA) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = OpT(A[i] - nax_hi_g<bf>(A[i]));
      ct_b[i] = OpT(SB ? nax_hi_g<bf>(Bn0[i]) : Bn0[i]);
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = OpT(SB ? nax_hi_g<bf>(Bn1[i]) : Bn1[i]);
    }
    gemm_op.run(ct_a, ct_b, ct_c);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
  }
}

} // namespace steel
} // namespace mlx

#define SPB(BIT) (((SPLIT) >> (2 * (BIT))) & 3)

#define MM16x16x16(C, CO, A, TA, AO, B, TB, BO, BIT)         \
  mlx::steel::mma<                                           \
      float,                                                 \
      float,                                                 \
      float,                                                 \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply,  \
      SPB(BIT),                                              \
      metal::conditional_t<(BFOPS != 0), bfloat, float>>(                                             \
      C.frag_at(0, (CO)),                                    \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      metal::bool_constant<TB>{});

#define MMA16x16x16(C, CO, A, TA, AO, B, TB, BO, BIT)                   \
  mlx::steel::mma<                                                      \
      float,                                                            \
      float,                                                            \
      float,                                                            \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,  \
      SPB(BIT),                                              \
      metal::conditional_t<(BFOPS != 0), bfloat, float>>(                                                        \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      metal::bool_constant<TB>{});

#define MMA16x16x32(C, CO, A, TA, AO, B, TB, BO, BIT)                   \
  mlx::steel::mma<                                                      \
      float,                                                            \
      float,                                                            \
      float,                                                            \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,  \
      SPB(BIT),                                              \
      metal::conditional_t<(BFOPS != 0), bfloat, float>>(                                                        \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      A.frag_at(0, (AO) + 1),                                           \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});

#define MM16x32x16(C, CO, A, TA, AO, B, TB, BO, BIT)         \
  mlx::steel::mman<                                          \
      float,                                                 \
      float,                                                 \
      float,                                                 \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply,  \
      SPB(BIT),                                              \
      metal::conditional_t<(BFOPS != 0), bfloat, float>>(                                             \
      C.frag_at(0, (CO)),                                    \
      C.frag_at(0, (CO) + 1),                                \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      B.frag_at(0, (BO) + 1),                                \
      metal::bool_constant<TB>{});

#define MMA16x32x16(C, CO, A, TA, AO, B, TB, BO, BIT)                   \
  mlx::steel::mman<                                                     \
      float,                                                            \
      float,                                                            \
      float,                                                            \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate,  \
      SPB(BIT),                                              \
      metal::conditional_t<(BFOPS != 0), bfloat, float>>(                                                        \
      C.frag_at(0, (CO)),                                               \
      C.frag_at(0, (CO) + 1),                                           \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});
