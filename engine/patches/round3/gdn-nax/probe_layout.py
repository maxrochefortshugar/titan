# SPDX-License-Identifier: Apache-2.0
"""Probe: does the BaseNAXFrag register layout survive relaxed_precision=false?

Uses the PR's padded 16x32x16 form (M=16,N=32,K=16), the descriptor behind
MM16x16x16 / MMA16x16x16 / MM16x32x16.

Test 1: A = I,  B = [X | 0]     -> C left half should equal X.
Test 2: A = X,  B = [I | 0]     -> C left half should equal X.
Values 0..255 are exact in bf16 and fp32, so a mismatch is layout, not rounding.
"""
import numpy as np
import mlx.core as mx
import naxhdr

EXTRA = r'''
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
'''

BODY = r'''
  mlx::steel::NAXTile<float, 1, 1> A, B0, B1, C0, C1;
  A.load(a, 16);
  B0.load(b, 32);
  B1.load(b + 16, 32);
  C0.clear();
  C1.clear();
  if (RLX) {
    mman_p<true>(C0.frag_at(0, 0), C1.frag_at(0, 0), A.frag_at(0, 0),
                 B0.frag_at(0, 0), B1.frag_at(0, 0));
  } else {
    mman_p<false>(C0.frag_at(0, 0), C1.frag_at(0, 0), A.frag_at(0, 0),
                  B0.frag_at(0, 0), B1.frag_at(0, 0));
  }
  C0.store(c, 32);
  C1.store(c + 16, 32);
'''

k = mx.fast.metal_kernel(
    name="nax_probe_layout16x32x16",
    input_names=["a", "b"],
    output_names=["c"],
    header=naxhdr.header(EXTRA),
    source=BODY,
)


def run(a, b, relaxed):
    (c,) = k(
        inputs=[mx.array(a), mx.array(b)],
        template=[("RLX", bool(relaxed))],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16, 32)],
        output_dtypes=[mx.float32],
    )
    mx.eval(c)
    return np.array(c)


I = np.eye(16, dtype=np.float32)
X = np.arange(256, dtype=np.float32).reshape(16, 16)
Z = np.zeros((16, 16), dtype=np.float32)

for relaxed in (True, False):
    tag = "relaxed" if relaxed else "precise"
    c1 = run(I, np.hstack([X, Z]), relaxed)[:, :16]
    c2 = run(X, np.hstack([I, Z]), relaxed)[:, :16]
    print(f"[{tag}] I@X == X : {np.array_equal(c1, X)}  max|d| {np.abs(c1-X).max():g}")
    print(f"[{tag}] X@I == X : {np.array_equal(c2, X)}  max|d| {np.abs(c2-X).max():g}")
    if not np.array_equal(c1, X):
        print("  I@X got[:4,:8]\n", c1[:4, :8])
    if not np.array_equal(c2, X):
        print("  X@I got[:4,:8]\n", c2[:4, :8])

# fp32 dynamic-range test: are the inputs silently truncated to bf16?
rng = np.random.default_rng(0)
A = rng.standard_normal((16, 16)).astype(np.float32)
B = rng.standard_normal((16, 16)).astype(np.float32)
ref = (A.astype(np.float64) @ B.astype(np.float64))
def rr(x, r):
    return float(np.linalg.norm(x - r) / np.linalg.norm(r))
for relaxed in (True, False):
    c = run(A, np.hstack([B, Z]), relaxed)[:, :16]
    tag = "relaxed" if relaxed else "precise"
    print(f"[{tag}] random fp32 16x16x16 rrmse vs fp64: {rr(c, ref):.3e}")
