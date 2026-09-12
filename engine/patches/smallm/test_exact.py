#!/usr/bin/env python3
"""Correctness for the small-M quantized matmul kernel.

Compares ``qmm_smallm`` against ``mx.quantized_matmul`` on the Qwen3.8-Flash-
Next projection shapes, at M in {1,2,3,4,6,8,16}, for two weight
distributions:

  normal       -- N(0,1), the easy case
  heavy-tail   -- N(0,1) * Student-t-ish scale mixture, so a few channels carry
                  outliers 20-50x the mode.  This is what real transformer
                  weights look like after affine group quantization and it is
                  where a sloppy accumulation order shows up.

Both paths are also compared against an exact fp32 dequantize-and-matmul
reference, because the interesting question is not "do the two kernels agree
bit for bit" (they cannot -- the K reduction order differs) but "is the new
kernel at least as close to the truth as the stock one".

For lm_head it additionally reports greedy-argmax agreement over a 1000-row
sample, which is the property that actually decides token identity.

Usage: ~/inference-server/kdev/bin/python test_exact.py [--bits 4] [--gs 64]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import MAX_M, qmm_smallm, supported  # noqa: E402

SHAPES = (
    ("q_proj", 6144, 2560),
    ("o_proj", 2560, 6144),
    ("kv_proj", 512, 2560),
    ("shared_mlp", 640, 2560),
    ("lm_head", 248320, 2560),
)
MS = (1, 2, 3, 4, 6, 8, 16)


def make_weights(N, K, kind):
    if kind == "normal":
        return mx.random.normal((N, K)).astype(mx.bfloat16)
    # Heavy tail: per-element scale drawn from a lognormal, so a few percent of
    # channels dominate their quantization group.
    base = mx.random.normal((N, K))
    scale = mx.exp(mx.random.normal((N, K)) * 1.6)
    return (base * scale).astype(mx.bfloat16)


def errs(got, ref):
    """max abs, max *scale*-relative, mean abs.

    Relative error is taken against the largest magnitude in the same output
    row, not against the individual element.  Element-wise relative error is
    meaningless here: a logit that lands near zero by cancellation makes any
    two correct kernels look 1e6 apart.  Row scale is what decides softmax and
    argmax, so that is the denominator that matters.
    """
    g = got.astype(mx.float32)
    r = ref.astype(mx.float32)
    ae = mx.abs(g - r)
    scale = mx.maximum(mx.abs(r).max(axis=-1, keepdims=True), 1e-6)
    return float(ae.max()), float((ae / scale).max()), float(ae.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mx.random.seed(a.seed)

    print(f"mlx {mx.__version__}  bits={a.bits} gs={a.gs}")
    print("\nours vs stock mx.quantized_matmul, and both vs fp32 exact")
    print("(rel is against the row's max magnitude; M=1 and M=16 are outside")
    print(" the routed band and are where *stock* switches kernel, so the")
    print(" ours-vs-stock delta there is stock's own numerics changing)")
    print(f"{'shape':<12}{'dist':<12}{'M':>3}{'max|d|':>11}{'max rel':>10}"
          f"{'mean|d|':>11}{'ours-exact':>12}{'stock-exact':>13}")

    worst = 0.0
    for name, N, K in SHAPES:
        for kind in ("normal", "heavy-tail"):
            w = make_weights(N, K, kind)
            wq, s, b = mx.quantize(w, a.gs, a.bits)
            mx.eval(wq, s, b)
            del w
            # fp32 ground truth from the same dequantized weights.
            wd = mx.dequantize(
                wq, s, b, group_size=a.gs, bits=a.bits
            ).astype(mx.float32)
            mx.eval(wd)
            for M in MS:
                x = mx.random.normal((M, K)).astype(mx.bfloat16)
                ref = mx.quantized_matmul(
                    x, wq, s, b, transpose=True, group_size=a.gs, bits=a.bits
                )
                got = qmm_smallm(x, wq, s, b, a.gs, a.bits)
                exact = x.astype(mx.float32) @ wd.T
                mx.eval(ref, got, exact)
                mabs, mrel, meanabs = errs(got, ref)
                oe, _, _ = errs(got, exact)
                se, _, _ = errs(ref, exact)
                worst = max(worst, mrel)
                print(f"{name:<12}{kind:<12}{M:>3}{mabs:>11.3e}{mrel:>10.2e}"
                      f"{meanabs:>11.3e}{oe:>12.3e}{se:>13.3e}")
                del x, ref, got, exact
            del wq, s, b, wd
            mx.clear_cache()

    # ---- greedy argmax agreement on lm_head -------------------------------
    print("\nlm_head greedy-argmax agreement (1000 rows, batched at each M)")
    N, K = 248320, 2560
    w = make_weights(N, K, "heavy-tail")
    wq, s, b = mx.quantize(w, a.gs, a.bits)
    mx.eval(wq, s, b)
    del w
    mx.clear_cache()
    for M in (1, 2, 4, 8, 16):
        agree = 0
        total = 0
        for _ in range((1000 + M - 1) // M):
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(
                x, wq, s, b, transpose=True, group_size=a.gs, bits=a.bits
            )
            got = qmm_smallm(x, wq, s, b, a.gs, a.bits)
            eq = mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)
            mx.eval(eq)
            agree += int(eq.sum())
            total += M
        print(f"  M={M:<3} {agree}/{total} agree  ({100.0*agree/total:.2f}%)")
    del wq, s, b
    mx.clear_cache()

    # ---- fallback coverage ------------------------------------------------
    print("\nfallback: unsupported shapes must equal stock exactly")
    wq, s, b = mx.quantize(mx.random.normal((6144, 2560)).astype(mx.bfloat16), 64, 4)
    mx.eval(wq, s, b)
    cases = [("M>MAX_M", 32), ("M=17", 17)]
    for label, M in cases:
        x = mx.random.normal((M, 2560)).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        got = qmm_smallm(x, wq, s, b, 64, 4)
        mx.eval(ref, got)
        same = bool(mx.all(ref == got))
        print(f"  {label:<12} bit-identical to stock: {same}")
        assert same, label
    # 3-D batch-1 passthrough shape check
    x3 = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    y3 = qmm_smallm(x3, wq, s, b, 64, 4)
    print(f"  3-D batch-1  shape {y3.shape} (expect (1, 4, 6144))")
    assert y3.shape == (1, 4, 6144)
    # batch>1 must fall back
    x3b = mx.random.normal((2, 4, 2560)).astype(mx.bfloat16)
    yb = qmm_smallm(x3b, wq, s, b, 64, 4)
    rb = mx.quantized_matmul(x3b, wq, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(yb, rb)
    print(f"  3-D batch>1  bit-identical to stock: {bool(mx.all(yb == rb))}")
    print(f"  supported(M=17) = {supported(17, 2560, 6144, 4, 64, mx.bfloat16)}"
          f" (MAX_M={MAX_M})")
    print(f"  supported(K=2568) = {supported(4, 2568, 6144, 4, 64, mx.bfloat16)}")
    print(f"  supported(bits=3) = {supported(4, 2560, 6144, 3, 64, mx.bfloat16)}")

    print(f"\nworst relative error across all shapes/dists/M: {worst:.3e}")


if __name__ == "__main__":
    main()
