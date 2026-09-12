# SPDX-License-Identifier: Apache-2.0
"""Read the per-thread element coordinates of each cooperative tensor directly,
for relaxed_precision true and false."""
import numpy as np
import mlx.core as mx
import naxhdr

EXTRA = r'''
template <bool RLX>
METAL_FUNC void dump(device int* out) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, RLX,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), float>();
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  const int base = (RLX ? 0 : 1) * 32 * 3 * 16 * 2 + lane * 3 * 16 * 2;
  for (short i = 0; i < 8; i++) {
    auto ca = ct_a.get_multidimensional_index(i);
    out[base + (0 * 16 + i) * 2 + 0] = (int)ca[0];
    out[base + (0 * 16 + i) * 2 + 1] = (int)ca[1];
  }
  for (short i = 0; i < 16; i++) {
    auto cb = ct_b.get_multidimensional_index(i);
    out[base + (1 * 16 + i) * 2 + 0] = (int)cb[0];
    out[base + (1 * 16 + i) * 2 + 1] = (int)cb[1];
    auto cc = ct_c.get_multidimensional_index(i);
    out[base + (2 * 16 + i) * 2 + 0] = (int)cc[0];
    out[base + (2 * 16 + i) * 2 + 1] = (int)cc[1];
  }
}
'''
BODY = r'''
  dump<true>(out);
  dump<false>(out);
'''
k = mx.fast.metal_kernel(name="nax_probe_coords", input_names=["d"], output_names=["out"],
                         header=naxhdr.header(EXTRA), source=BODY)
(out,) = k(inputs=[mx.zeros((1,), mx.float32)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
           output_shapes=[(2, 32, 3, 16, 2)], output_dtypes=[mx.int32])
mx.eval(out)
o = np.array(out)
names = ["A(16x16)", "B(16x32)", "C(16x32)"]
counts = [8, 16, 16]
for mi, mode in enumerate(["relaxed", "precise"]):
    for oi, nm in enumerate(names):
        print(f"[{mode}] {nm} lane0 idx->(d0,d1):",
              [tuple(o[mi, 0, oi, i]) for i in range(counts[oi])])
        print(f"[{mode}] {nm} lane5 idx->(d0,d1):",
              [tuple(o[mi, 5, oi, i]) for i in range(counts[oi])])
