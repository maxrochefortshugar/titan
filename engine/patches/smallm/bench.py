#!/usr/bin/env python3
"""Before/after benchmark for the small-M quantized matmul kernel.

Chains CHAIN ops per mx.eval to amortise command-buffer submission, reports
the median of ITERS timings, same protocol as ~/inference-server/kbench.py.

Usage: ~/inference-server/kdev/bin/python bench.py [--bits 4] [--gs 64] [--quick]
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import _pick_tile, qmm_smallm  # noqa: E402

CHAIN = 10

SHAPES = (
    ("q_proj      2560->6144", 6144, 2560),
    ("o_proj      6144->2560", 2560, 6144),
    ("kv_proj     2560->512", 512, 2560),
    ("shared mlp  2560->640", 640, 2560),
    ("lm_head     2560->248320", 248320, 2560),
)
MS = (1, 2, 3, 4, 6, 8, 16)


def timeit(fn, iters, warm=2):
    """Return (min, median) seconds per op.

    The GPU is shared with the live oMLX daemon, so the median is whatever the
    daemon happened to be doing.  Contention can only *add* time, so the min
    over enough samples is the number that reproduces -- it matches the stock
    figures in COMMON.md, while the median on this box drifts 2-5x run to run.
    Both are reported so the noise is visible rather than hidden.
    """
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
    return min(ts), st.median(ts)


def qbytes(n_elems, bits, gs):
    return n_elems * bits / 8 + (n_elems / gs) * 2 * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    iters = 8 if a.quick else 20

    print(f"mlx {mx.__version__}  bits={a.bits} gs={a.gs}  CHAIN={CHAIN} iters={iters}")
    print("times are min / median of the sample; min is the reproducible one")
    print(f"{'shape':<26}{'M':>4}{'stock ms':>10}{'ours ms':>10}{'speedup':>9}"
          f"{'GB/s':>8}{'tile':>12}{'vs M=1':>8}{'stock med':>11}{'our med':>10}")

    for name, N, K in SHAPES:
        w = mx.random.normal((N, K)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, a.gs, a.bits)
        mx.eval(wq, s, b)
        wbytes = qbytes(N * K, a.bits, a.gs)
        base_ours = None
        for M in MS:
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            mx.eval(x)
            t_stock, m_stock = timeit(
                lambda: mx.quantized_matmul(
                    x, wq, s, b, transpose=True, group_size=a.gs, bits=a.bits
                ),
                iters,
            )
            t_ours, m_ours = timeit(
                lambda: qmm_smallm(x, wq, s, b, a.gs, a.bits), iters
            )
            if base_ours is None:
                base_ours = t_ours
            by = wbytes + (M * K + M * N) * 2
            bn, kp, ng = _pick_tile(M, N, K, a.bits)
            print(
                f"{name:<26}{M:>4}{t_stock*1e3:>10.3f}{t_ours*1e3:>10.3f}"
                f"{t_stock/t_ours:>8.2f}x{by/t_ours/1e9:>8.0f}"
                f"{f'{bn}/{kp}/{ng}':>12}{t_ours/base_ours:>7.2f}x"
                f"{m_stock*1e3:>11.3f}{m_ours*1e3:>10.3f}"
            )
        del w, wq, s, b
        mx.clear_cache()


if __name__ == "__main__":
    main()
