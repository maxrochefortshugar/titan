#!/usr/bin/env python3
"""Cost of the host round-trip the confidence gate needs per draft step.

The stock draft chain dispatches every step lazily and syncs once, in the
next verify cycle. The gate has to read the drafted token's probability on
the host before it can decide whether to run the next step, so it forces one
extra evaluation per gated step.

This measures the bubble: a chain of D dependent steps, each sized to a
realistic draft step, evaluated once at the end (stock) against evaluated
step by step with a scalar read (gated). The difference divided by the
number of syncs is the per-step cost the break-even has to absorb.

Deliberately small: no model, no lm_head-sized weights, under 200 MB.

Run: ~/inference-server/kdev/bin/python bench_sync.py [--ms 1.9] [--reps 15]
"""

import argparse
import statistics
import time

import mlx.core as mx


def calibrate(target_ms: float, dtype=mx.bfloat16):
    """Square matmul size whose single launch costs about target_ms."""
    n = 512
    while n < 8192:
        a = mx.random.normal((n, n)).astype(dtype)
        b = mx.random.normal((n, n)).astype(dtype)
        mx.eval(a, b)
        for _ in range(3):
            mx.eval(a @ b)
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(a @ b)
        ms = (time.perf_counter() - t0) * 100.0
        if ms >= target_ms:
            return n, ms, a, b
        n = int(n * 1.35)
    return n, ms, a, b


def chain(a, b, depth, sync_each):
    """``depth`` dependent steps; optionally read one scalar per step."""
    x = a
    syncs = 0
    for j in range(depth):
        x = mx.tanh(x @ b) * 0.5
        if sync_each and j + 1 < depth:
            s = mx.sum(x[:1, :4])
            mx.eval(s)
            float(s.item())
            syncs += 1
    mx.eval(x)
    return syncs


def bench(fn, reps):
    xs = []
    for _ in range(3):
        fn()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=1.9,
                    help="target cost of one synthetic draft step")
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()

    n, ms, a, b = calibrate(args.ms)
    print(f"synthetic draft step: {n}x{n} bf16 matmul + tanh, "
          f"{ms:.3f} ms per step measured, {2 * n * n * 2 / 1e6:.0f} MB live")
    print(f"\n{'depth':>6}{'one sync (ms)':>16}{'per-step sync (ms)':>21}"
          f"{'syncs':>7}{'bubble/sync (ms)':>19}")
    for d in (2, 3, 4, 5):
        t_lazy = bench(lambda d=d: chain(a, b, d, False), args.reps)
        t_sync = bench(lambda d=d: chain(a, b, d, True), args.reps)
        syncs = d - 1
        print(f"{d:>6}{t_lazy:>16.3f}{t_sync:>21.3f}{syncs:>7}"
              f"{(t_sync - t_lazy) / max(1, syncs):>19.3f}")

    print("\nscalar round-trip alone (no dependent work):")
    x = mx.array([1.0])
    mx.eval(x)

    def one():
        y = mx.sum(x)
        mx.eval(y)
        float(y.item())

    print(f"  {bench(one, 200):.4f} ms per mx.eval + .item()")


if __name__ == "__main__":
    main()
