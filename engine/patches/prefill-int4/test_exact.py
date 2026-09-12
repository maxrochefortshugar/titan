"""Accuracy of the int8-activation NAX prefill matmul.

This kernel is deliberately NOT bit-exact with the stock path: activations are
quantized to int8 per row-group of 64 before entering the tensor unit.  What
follows measures how big that is, against two references:

  stock  - mx.quantized_matmul (bf16 activations, weights dequantized to bf16,
           fp32 accumulation).  This is what production runs today.
  fp32   - float32 activations x float32 dequantized weights, accumulated in
           float64 on the host.  This is the true value of the quantized
           linear layer; both stock and this kernel are approximations of it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, numpy as np
import kernel


def fp32_reference(x, wq, s, b):
    w = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
    return np.array(x.astype(mx.float32), dtype=np.float64) @ np.array(w, dtype=np.float64).T


def stats(got, ref):
    d = np.abs(got - ref)
    scale = np.abs(ref).max()
    rms = np.sqrt((d ** 2).mean())
    rel = d / np.maximum(np.abs(ref), 1e-6)
    return d.max(), d.max() / scale, rms, np.sqrt((ref ** 2).mean()), np.median(rel)


def make_x(M, K, kind, key):
    g = mx.random.normal((M, K), key=key)
    if kind == "normal":
        return (g * 0.5).astype(mx.bfloat16)
    if kind == "outlier":
        # transformer-like: a few channels an order of magnitude hotter
        hot = (mx.arange(K) % 128 == 0).astype(mx.float32) * 9.0 + 1.0
        return (g * 0.5 * hot).astype(mx.bfloat16)
    if kind == "heavy":
        u = mx.random.uniform(shape=(M, K), key=key) * 0.98 + 0.01
        return (g * 0.5 / mx.sqrt(u)).astype(mx.bfloat16)   # heavy tail
    raise ValueError(kind)


def run_case(name, M, N, K, kind):
    keys = mx.random.split(mx.random.key(0xBEEF), 2)
    x = make_x(M, K, kind, keys[0])
    w = (mx.random.normal((N, K), key=keys[1]) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    wq_s, sc, bs = kernel.prepare_weights(wq, s, b)
    ours = np.array(kernel.prefill_qmm(x, wq_s, sc, bs).astype(mx.float32), dtype=np.float64)
    stock = np.array(mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64,
                                         bits=4).astype(mx.float32), dtype=np.float64)
    ref = fp32_reference(x, wq, s, b)
    # the yardstick: how much error 4-bit weight quantization already introduces
    full = (np.array(x.astype(mx.float32), dtype=np.float64)
            @ np.array(w.astype(mx.float32), dtype=np.float64).T)
    for label, got in (("ours vs stock", (ours, stock)), ("ours vs fp32", (ours, ref)),
                       ("stock vs fp32", (stock, ref)),
                       ("q4 vs unquantized", (ref, full)),
                       ("ours vs unquantized", (ours, full))):
        a, r = got
        mx_, rel, rms, rrms, medrel = stats(a, r)
        print(f"  {name:22s} {kind:8s} {label:14s} maxabs {mx_:9.5f}  "
              f"maxabs/|ref|max {rel:8.2e}  rms {rms:9.5f} ({rms/rrms:7.2e} of ref rms)")
    print()


def lm_head_argmax(M=1024, N=248320, K=2560):
    keys = mx.random.split(mx.random.key(0x1010), 2)
    x = (mx.random.normal((M, K), key=keys[0]) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((N, K), key=keys[1]) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    wq_s, sc, bs = kernel.prepare_weights(wq, s, b)
    mx.eval(wq_s, sc, bs)
    lo = kernel.prefill_qmm(x, wq_s, sc, bs).astype(mx.float32)
    ls = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
    wd = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
    lf = x.astype(mx.float32) @ wd.T
    a = np.array(mx.argmax(lo, axis=-1)); r = np.array(mx.argmax(ls, axis=-1))
    f = np.array(mx.argmax(lf, axis=-1))
    lfn = np.array(lf, dtype=np.float64)
    top = np.take_along_axis(lfn, f[:, None], 1)[:, 0]
    gap = top - np.take_along_axis(lfn, a[:, None], 1)[:, 0]
    print(f"  lm_head argmax, {M} rows, vocab {N} (random weights: near-degenerate logits)")
    print(f"    ours  vs stock : {(a==r).mean()*100:6.2f}%   ({int((a==r).sum())}/{M})")
    print(f"    stock vs fp32  : {(r==f).mean()*100:6.2f}%   ({int((r==f).sum())}/{M})")
    print(f"    ours  vs fp32  : {(a==f).mean()*100:6.2f}%   ({int((a==f).sum())}/{M})")
    print(f"    when ours differs from fp32, median fp32 logit gap to the true top-1: "
          f"{np.median(gap[a!=f]) if (a!=f).any() else 0.0:.4f} "
          f"(logit std {lfn.std():.3f})")
    return (a == r).mean()


if __name__ == "__main__":
    print("=== error vs stock and vs fp32 ===")
    for kind in ("normal", "outlier", "heavy"):
        run_case("q_proj M=2048", 2048, 6144, 2560, kind)
    run_case("o_proj M=512", 512, 2560, 6144, "normal")
    print("=== greedy decode agreement ===")
    lm_head_argmax()
