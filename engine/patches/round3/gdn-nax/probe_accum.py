# SPDX-License-Identifier: Apache-2.0
"""Is relaxed_precision error input rounding (fixable by splitting) or accumulation?"""
import numpy as np
import mlx.core as mx
import naxhdr

EXTRA = open("probe_layout_extra.h").read()

BODY = r'''
  mlx::steel::NAXTile<float, 1, 1> A, B0, B1, C0, C1;
  A.load(a, 16);
  B0.load(b, 32);
  B1.load(b + 16, 32);
  C0.clear();
  C1.clear();
  mman_p<true>(C0.frag_at(0, 0), C1.frag_at(0, 0), A.frag_at(0, 0),
               B0.frag_at(0, 0), B1.frag_at(0, 0));
  C0.store(c, 32);
  C1.store(c + 16, 32);
'''

k = mx.fast.metal_kernel(
    name="nax_probe_accum",
    input_names=["a", "b"],
    output_names=["c"],
    header=naxhdr.header(EXTRA),
    source=BODY,
)


def run(a, b):
    (c,) = k(inputs=[mx.array(a), mx.array(b)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(16, 32)], output_dtypes=[mx.float32])
    mx.eval(c)
    return np.array(c)[:, :16]


def to_bf16(x):
    return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))


def rr(x, r):
    return float(np.linalg.norm(x - r) / np.linalg.norm(r))


rng = np.random.default_rng(0)
Z = np.zeros((16, 16), dtype=np.float32)
A = rng.standard_normal((16, 16)).astype(np.float32)
B = rng.standard_normal((16, 16)).astype(np.float32)
Ab, Bb = to_bf16(A), to_bf16(B)

print("fp32 in, ref fp64(fp32 in)      :", f"{rr(run(A, np.hstack([B, Z])), A.astype(np.float64) @ B.astype(np.float64)):.3e}")
print("bf16-exact in, ref fp64(same)   :", f"{rr(run(Ab, np.hstack([Bb, Z])), Ab.astype(np.float64) @ Bb.astype(np.float64)):.3e}")
print("fp32 in, ref fp64(bf16-rounded) :", f"{rr(run(A, np.hstack([B, Z])), Ab.astype(np.float64) @ Bb.astype(np.float64)):.3e}")

# 3-term split: A = Ah + Al, B = Bh + Bl (bf16 pieces), C ~ Ah Bh + Ah Bl + Al Bh
Ah, Bh = Ab, Bb
Al, Bl = to_bf16(A - Ah), to_bf16(B - Bh)
c3 = run(Ah, np.hstack([Bh, Z])) + run(Ah, np.hstack([Bl, Z])) + run(Al, np.hstack([Bh, Z]))
c2 = run(Ah, np.hstack([Bh, Z])) + run(Ah, np.hstack([Bl, Z]))
ref = A.astype(np.float64) @ B.astype(np.float64)
print("2-term split (AhBh+AhBl)        :", f"{rr(c2, ref):.3e}")
print("3-term split (+AlBh)            :", f"{rr(c3, ref):.3e}")
c4 = c3 + run(Al, np.hstack([Bl, Z]))
print("4-term split (+AlBl)            :", f"{rr(c4, ref):.3e}")
