# SPDX-License-Identifier: Apache-2.0
"""With B = I the relaxed matmul returns round_F(A), so C reveals F directly."""
import numpy as np
import mlx.core as mx
import naxhdr

EXTRA = open("probe_layout_extra.h").read()
BODY = r'''
  mlx::steel::NAXTile<float, 1, 1> A, B0, B1, C0, C1;
  A.load(a, 16);
  B0.load(b, 32);
  B1.load(b + 16, 32);
  C0.clear(); C1.clear();
  mman_p<true>(C0.frag_at(0, 0), C1.frag_at(0, 0), A.frag_at(0, 0),
               B0.frag_at(0, 0), B1.frag_at(0, 0));
  C0.store(c, 32); C1.store(c + 16, 32);
'''
k = mx.fast.metal_kernel(name="nax_probe_format", input_names=["a", "b"],
                         output_names=["c"], header=naxhdr.header(EXTRA), source=BODY)
Z = np.zeros((16, 16), dtype=np.float32)
I = np.eye(16, dtype=np.float32)


def quantize(vals):
    """Return round_F(vals) for a flat array of at most 256 values."""
    vals = np.asarray(vals, dtype=np.float32).ravel()
    out = []
    for s in range(0, len(vals), 256):
        blk = np.zeros(256, dtype=np.float32)
        n = min(256, len(vals) - s)
        blk[:n] = vals[s:s + n]
        (c,) = k(inputs=[mx.array(blk.reshape(16, 16)), mx.array(np.hstack([I, Z]))],
                 grid=(32, 1, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(16, 32)], output_dtypes=[mx.float32])
        mx.eval(c)
        out.append(np.array(c)[:, :16].ravel()[:n])
    return np.concatenate(out)


def bits(x):
    return np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]


rng = np.random.default_rng(1)
v = rng.standard_normal(256).astype(np.float32)
q = quantize(v)
mant_lost = []
for a, b in zip(v, q):
    d = bits(a) ^ bits(b)
    mant_lost.append(int(d).bit_length())
print("max mantissa bit index differing (0 = exact):", max(mant_lost))
rel = np.abs(q - v) / np.abs(v)
print(f"max relative quantization error: {rel.max():.3e}  (bf16 would be 3.9e-3, fp16 4.9e-4)")

# how many trailing mantissa bits are always zero after quantization?
zt = [int(bits(x) & 0x7FFFFF) for x in q]
tz = min((z & -z).bit_length() - 1 if z else 23 for z in zt)
print("min trailing zero mantissa bits in output:", tz, "-> significand bits kept:", 24 - tz)

# rounding mode: nearest or truncate?
print("signed error mean/std:", f"{(q - v).mean():.3e}", f"{(q - v).std():.3e}")

# exponent range: does it flush small or overflow large values?
mags = np.float32([1e-30, 1e-20, 1e-10, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8, 1e16, 1e30])
qm = quantize(mags)
for a, b in zip(mags, qm):
    print(f"  {a:>10.3e} -> {b:>12.6e}   rel {abs(b-a)/abs(a):.2e}")
