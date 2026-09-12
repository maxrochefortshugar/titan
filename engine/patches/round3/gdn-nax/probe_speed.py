# SPDX-License-Identifier: Apache-2.0
"""Unreliable, kept for reference only: the compiler hoists the invariant operand
setup out of the loop, so the three modes are not comparable. Use bench_bits.py.

Throughput of one 16x32x16 matmul2d: relaxed single pass, relaxed 3-pass
hi/lo split, and precise (relaxed_precision=false) single pass.
Runs only while ~/inference-server/staging/GPU_FREE exists."""
import os, time, statistics as st
if not os.path.exists(os.path.expanduser("~/inference-server/staging/GPU_FREE")):
    raise SystemExit("GPU_FREE absent; not benchmarking")
import mlx.core as mx
import naxhdr

EXTRA = r'''
METAL_FUNC float nax_hi(float x) {
  return as_type<float>(as_type<uint>(x) & 0xFFFFE000u);
}
template <bool RLX, bool SPLIT>
METAL_FUNC void bench(thread metal::vec<float, 8>& C0,
                      thread metal::vec<float, 8>& C1,
                      thread metal::vec<float, 8>& A,
                      thread metal::vec<float, 8>& B0,
                      thread metal::vec<float, 8>& B1,
                      int reps) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, RLX,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), float>();
  for (int r = 0; r < reps; r++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 8; i++) {
      ct_a[i] = SPLIT ? nax_hi(A[i]) : A[i];
      ct_b[i] = SPLIT ? nax_hi(B0[i]) : B0[i];
      ct_b[8 + i] = SPLIT ? nax_hi(B1[i]) : B1[i];
      ct_c[i] = C0[i];
      ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    if (SPLIT) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_b[i] = B0[i] - nax_hi(B0[i]);
        ct_b[8 + i] = B1[i] - nax_hi(B1[i]);
      }
      op.run(ct_a, ct_b, ct_c);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_a[i] = A[i] - nax_hi(A[i]);
        ct_b[i] = nax_hi(B0[i]);
        ct_b[8 + i] = nax_hi(B1[i]);
      }
      op.run(ct_a, ct_b, ct_c);
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 8; i++) {
      C0[i] = ct_c[i] * 0.5f;
      C1[i] = ct_c[8 + i] * 0.5f;
    }
  }
}
'''

BODY = r'''
  mlx::steel::NAXTile<float, 1, 1> A, B0, B1, C0, C1;
  A.load(a, 16); B0.load(a, 16); B1.load(a, 16);
  C0.clear(); C1.clear();
  bench<RLX, SPLIT>(C0.frag_at(0, 0), C1.frag_at(0, 0), A.frag_at(0, 0),
                    B0.frag_at(0, 0), B1.frag_at(0, 0), REPS);
  if (thread_position_in_grid.z == 0) {
    C0.store(c, 32); C1.store(c + 16, 32);
  }
'''

k = mx.fast.metal_kernel(name="nax_mma_speed", input_names=["a"], output_names=["c"],
                         header=naxhdr.header(EXTRA), source=BODY)
a = mx.random.normal((16, 16))
mx.eval(a)
NSG, REPS = 1024, 400


def run(rlx, split):
    (c,) = k(inputs=[a], template=[("RLX", bool(rlx)), ("SPLIT", bool(split)), ("REPS", REPS)],
             grid=(32, 4, NSG // 4), threadgroup=(32, 4, 1),
             output_shapes=[(16, 32)], output_dtypes=[mx.float32])
    return c


def timeit(fn, iters=15):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


base = timeit(lambda: run(True, False))
print(f"{'mode':<28}{'us/kernel':>11}{'ns/mma-op':>11}{'x relaxed':>11}")
for name, rlx, split, npass in [("relaxed single pass", 1, 0, 1),
                                ("relaxed 3-pass hi/lo split", 1, 1, 3),
                                ("precise single pass", 0, 0, 1)]:
    t = timeit(lambda r=rlx, s=split: run(r, s))
    per = t / (NSG * REPS) * 1e9
    print(f"{name:<28}{t*1e6:>11.1f}{per:>11.3f}{t/base:>11.2f}")
