#!/usr/bin/env python3
"""Fused top-K against mx.topk / mx.argpartition at the lm_head width.

Median of 15, warm, CHAIN=10 per sample, mx.synchronize around each sample
(the kbench.py convention).  Peak allocation ~20 MB.
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import mlx.core as mx

import topk as T

CHAIN = 10
V = 248320


def timeit(fn, iters=15, warm=3):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)

    for _ in range(warm):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=41)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--ks", default="512,1024,2048,4096")
    a = ap.parse_args()
    dt = getattr(mx, a.dtype)

    x = (mx.random.normal((V,)) * 6.0).astype(dt)
    mx.eval(x)
    t_argmax = timeit(lambda: mx.argmax(x, axis=-1), a.iters)
    print(f"V={V} dtype={a.dtype}   mx.argmax reference: {t_argmax*1e3:.3f} ms")
    print(f"{'K':>6} {'mx.topk':>9} {'argpart':>9} {'fused':>9} "
          f"{'vs topk':>8} {'vs argpart':>11}")
    for k in [int(s) for s in a.ks.split(",")]:
        t_topk = timeit(lambda: mx.topk(x, k), a.iters)
        t_part = timeit(lambda: mx.argpartition(x, kth=V - k, axis=-1)[..., -k:],
                        a.iters)
        t_fast = timeit(lambda: T.fast_topk(x, k)[1], a.iters)
        print(f"{k:>6} {t_topk*1e3:9.3f} {t_part*1e3:9.3f} {t_fast*1e3:9.3f} "
              f"{t_topk/t_fast:7.2f}x {t_part/t_fast:10.2f}x")


if __name__ == "__main__":
    main()
