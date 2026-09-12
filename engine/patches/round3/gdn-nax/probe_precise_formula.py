# SPDX-License-Identifier: Apache-2.0
"""Derive and verify closed-form precise-mode (relaxed_precision=false) layouts
for the two descriptors the GDN kernel uses: (16,32,16) and (16,16,32)."""
import numpy as np
import mlx.core as mx
import naxhdr

EXTRA = r'''
template <int M, int N, int K, bool RLX>
METAL_FUNC void dump(device int* out, int slot) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      M, N, K, false, false, RLX,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), float>();
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  device int* o = out + ((slot * 32 + lane) * 3) * 16 * 2;
  for (short i = 0; i < 16; i++) {
    auto ca = ct_a.get_multidimensional_index(i);
    auto cb = ct_b.get_multidimensional_index(i);
    auto cc = ct_c.get_multidimensional_index(i);
    o[(0 * 16 + i) * 2 + 0] = (int)ca[0]; o[(0 * 16 + i) * 2 + 1] = (int)ca[1];
    o[(1 * 16 + i) * 2 + 0] = (int)cb[0]; o[(1 * 16 + i) * 2 + 1] = (int)cb[1];
    o[(2 * 16 + i) * 2 + 0] = (int)cc[0]; o[(2 * 16 + i) * 2 + 1] = (int)cc[1];
  }
}
'''
BODY = r'''
  dump<16, 32, 16, false>(out, 0);
  dump<16, 16, 32, false>(out, 1);
  dump<16, 32, 16, true>(out, 2);
  dump<16, 16, 32, true>(out, 3);
'''
k = mx.fast.metal_kernel(name="nax_probe_formula", input_names=["d"], output_names=["out"],
                         header=naxhdr.header(EXTRA), source=BODY)
(out,) = k(inputs=[mx.zeros((1,), mx.float32)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
           output_shapes=[(4, 32, 3, 16, 2)], output_dtypes=[mx.int32])
mx.eval(out)
o = np.array(out)   # [slot, lane, operand, idx, (d0,d1)]

# element counts: (16,32,16): A 8, B 16, C 16 ; (16,16,32): A 16, B 16, C 8
COUNTS = {0: (8, 16, 16), 1: (16, 16, 8), 2: (8, 16, 16), 3: (16, 16, 8)}
SLOT = {0: "precise 16x32x16", 1: "precise 16x16x32", 2: "relaxed 16x32x16", 3: "relaxed 16x16x32"}
OPN = ["A", "B", "C"]


def coordset(slot, op):
    n = COUNTS[slot][op]
    return {(lane, i): tuple(o[slot, lane, op, i]) for lane in range(32) for i in range(n)}


def base(lane):
    qid = lane >> 2
    fm = (qid & 4) | ((lane >> 1) & 3)
    fn4 = ((qid & 2) | (lane & 1)) * 4
    fn2 = ((qid & 2) | (lane & 1)) * 2
    return fm, fn4, fn2


def predict_precise(lane, i, op, slot):
    fm, fn4, fn2 = base(lane)
    if slot == 0:   # (16,32,16): A is 16x16(MxK), B is 16x32(KxN)... report as (d0,d1)
        if op == 0:
            return (fn2 + (i & 1) + ((i >> 1) & 1) * 8, fm + (i >> 2) * 8)
        # B and C: 16 elems, extra N-half at bit 2
        return (fn2 + (i & 1) + ((i >> 1) & 1) * 8 + ((i >> 2) & 1) * 16, fm + (i >> 3) * 8)
    else:           # (16,16,32)
        if op == 2:  # C is 16x16
            return (fn2 + (i & 1) + ((i >> 1) & 1) * 8, fm + (i >> 2) * 8)
        return (fn2 + (i & 1) + ((i >> 1) & 1) * 8 + ((i >> 2) & 1) * 16, fm + (i >> 3) * 8)


def predict_relaxed(lane, i, op, slot):
    fm, fn4, fn2 = base(lane)
    small = (op == 0 and slot in (0, 2)) or (op == 2 and slot in (1, 3))
    if small:
        return (fn4 + (i % 4), fm + (i >> 2) * 8)
    return (fn4 + (i % 4), fm + ((i >> 2) & 1) * 8 + (i >> 3) * 16)


for slot in range(4):
    for op in range(3):
        cs = coordset(slot, op)
        pred = predict_precise if slot < 2 else predict_relaxed
        bad = [(l, i, c, pred(l, i, op, slot % 2)) for (l, i), c in cs.items()
               if c != pred(l, i, op, slot % 2)]
        print(f"{SLOT[slot]:<20} {OPN[op]}: {len(cs)-len(bad)}/{len(cs)} match", 
              "" if not bad else f" first mismatch {bad[0]}")
