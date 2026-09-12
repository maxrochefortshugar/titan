#!/usr/bin/env python3
"""Fused verify tail against the stock ops tail at the MTP verify widths.

Shapes are the real ones: k=10, D=2560, unsorted [B, T, k, D] in token order.
Median of 15, warm, CHAIN=10 per sample, mx.synchronize around each sample.
Peak allocation under 30 MB.
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import mlx.core as mx

import wsum_verify as W

CHAIN = 10
K, D = 10, 2560


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
    ap.add_argument("--ts", default="2,3,4,5,6,8,10,13,15")
    ap.add_argument("--mode", default="clone")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--layers", type=int, default=48)
    a = ap.parse_args()

    print(f"unsorted verify tail, k={K}, D={D}, bf16, mode={a.mode}")
    print(f"{'T':>4} {'stock ms':>9} {'kernel ms':>10} {'speedup':>8} "
          f"{'saved/layer us':>15} {f'x{a.layers} layers ms':>18}")
    for t in [int(s) for s in a.ts.split(",")]:
        y = (mx.random.normal((1, t, K, D)) * 0.7).astype(mx.bfloat16)
        sc = mx.random.uniform(shape=(1, t, K)).astype(mx.bfloat16)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        mx.eval(y, sc)
        t_ops = timeit(lambda: (y * sc[..., None]).sum(axis=-2), a.iters)
        t_ker = timeit(lambda: W.verify_tail(y, sc, a.mode), a.iters)
        saved = (t_ops - t_ker) * 1e6
        print(f"{t:>4} {t_ops*1e3:9.4f} {t_ker*1e3:10.4f} {t_ops/t_ker:7.2f}x "
              f"{saved:15.1f} {saved*a.layers/1e3:18.3f}")


if __name__ == "__main__":
    main()
