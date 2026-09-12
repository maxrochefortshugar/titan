// generated from probe_layout.py; shared by probe_accum.py and probe_format.py

template <bool RLX>
METAL_FUNC void mman_p(thread metal::vec<float, 8>& C0,
                       thread metal::vec<float, 8>& C1,
                       const thread metal::vec<float, 8>& A,
                       const thread metal::vec<float, 8>& B0,
                       const thread metal::vec<float, 8>& B1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, RLX,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), float>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 8; i++) {
    ct_a[i] = A[i];
    ct_b[i] = B0[i];
    ct_b[8 + i] = B1[i];
    ct_c[i] = 0.0f;
    ct_c[8 + i] = 0.0f;
  }
  op.run(ct_a, ct_b, ct_c);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 8; i++) {
    C0[i] = ct_c[i];
    C1[i] = ct_c[8 + i];
  }
}
