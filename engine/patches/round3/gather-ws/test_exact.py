# SPDX-License-Identifier: Apache-2.0
"""Exactness of the weight-stationary bf16 gather against mx.gather_qmm.

Both paths consume the same int4 affine weights and bf16 activations and
accumulate in fp32; only the tiling and the summation order over the 40 affine
groups differ.  So the comparison worth making is a bf16 ULP histogram against
stock, plus both paths against an fp32 reference to show neither is "the wrong
one" where they disagree.

Routing is Zipf-like (skewed expert popularity) as real top-10 routing is, not
uniform, so the ragged tail (experts with hundreds of rows) is exercised.
"""
import sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/round3/gather-ws')
import kernel as K

E, TOPK = 512, 10


def zipf_indices(R, E, a=0.7, seed=0):
    mx.random.seed(seed)
    p = 1.0 / mx.power(mx.arange(1, E + 1).astype(mx.float32), a)
    p = p / p.sum()
    perm = mx.random.permutation(E)
    cdf = mx.cumsum(p)
    u = mx.random.uniform(shape=(R,))
    pos = (u[:, None] > cdf[None, :]).sum(axis=1)
    return mx.sort(perm[mx.minimum(pos, E - 1)].astype(mx.uint32))


def bf16_ulp(a, b):
    """Signed ULP distance between two bf16 arrays, as an int32 numpy array."""
    ai = np.array(a.view(mx.uint16), copy=False).astype(np.int32)
    bi = np.array(b.view(mx.uint16), copy=False).astype(np.int32)
    key = lambda v: np.where(v & 0x8000, 0xFFFF - v, v + 0x8000)
    return np.abs(key(ai) - key(bi))


def fp32_ref(xf, wq, s, b, offs, N):
    parts = []
    for e in range(wq.shape[0]):
        lo, hi = offs[e], offs[e + 1]
        if hi <= lo:
            continue
        w = mx.dequantize(wq[e], s[e], b[e], group_size=64, bits=4).astype(mx.float32)
        parts.append(xf[lo:hi].astype(mx.float32) @ w.T)
        if len(parts) % 64 == 0:
            mx.eval(parts)
    mx.eval(parts)
    return mx.concatenate(parts, axis=0)


def case(name, N, Kd, T):
    R = T * TOPK
    w = (mx.random.normal((E, N, Kd)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    mx.eval(wq, s, b)
    mx.clear_cache()
    idx = zipf_indices(R, E)
    x = (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16)
    mx.eval(idx, x)

    stock = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                          group_size=64, bits=4, sorted_indices=True).reshape(R, N)
    ours = K.gather_qmm_sorted(x, wq, s, b, idx).reshape(R, N)
    mx.eval(stock, ours)

    u = bf16_ulp(ours, stock)
    d = mx.abs(ours.astype(mx.float32) - stock.astype(mx.float32))
    amax = mx.abs(stock.astype(mx.float32)).max().item()

    offs = K.build_tiles(idx, E, 48, K.max_tiles(R, E, 48))[0]
    mx.eval(offs)
    offs = offs.tolist()
    ref = fp32_ref(mx.contiguous(x.reshape(R, Kd)), wq, s, b, offs, N)
    mx.eval(ref)
    rms = mx.sqrt(mx.mean(ref * ref)).item()
    e_ours = mx.sqrt(mx.mean((ours.astype(mx.float32) - ref) ** 2)).item() / rms * 100
    e_stock = mx.sqrt(mx.mean((stock.astype(mx.float32) - ref) ** 2)).item() / rms * 100

    rows = np.diff(np.array(offs))
    print(f"{name} T={T} R={R} N={N} K={Kd}")
    print(f"  rows/expert  min {rows.min()} max {rows.max()} mean {rows.mean():.1f} "
          f"(experts over 48 rows: {(rows > 48).sum()}/{E})")
    print(f"  identical bf16 words   {100.0 * (u == 0).mean():.4f}%")
    print(f"  max ULP                {u.max()}   (1 ULP: {100.0*(u==1).mean():.4f}%, "
          f">1 ULP: {100.0*(u>1).mean():.6f}%)")
    print(f"  max abs diff vs stock  {d.max().item():.3e}   (|out|max {amax:.3f})")
    print(f"  RMS err vs fp32 ref    ours {e_ours:.4f}%   stock {e_stock:.4f}%")
    del wq, s, b, idx, x, stock, ours, ref
    mx.clear_cache()


for T in (512, 2048):
    case("gate_up (fused) [512,1280,2560]", 1280, 2560, T)
for T in (512, 2048):
    case("down            [512,2560, 640]", 2560, 640, T)
