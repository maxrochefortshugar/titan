"""Microbench: the MTP draft block's projections at 4-bit vs 8-bit group 64.

Real shapes, synthetic random weights, M=1 (one decode row). The expert
tensors are built with a reduced expert count because gather_qmm at top-10
reads only the gathered experts, so the routed cost does not depend on how
many experts sit behind them; the allocation would.

    ~/inference-server/kdev/bin/python bench.py
"""

import statistics
import time

import mlx.core as mx

CHAIN, ITERS, WARM = 10, 15, 5
EXPERTS_ALLOC, TOPK = 32, 10
HIDDEN, INTER, HC, LOWRANK = 2560, 640, 4, 320


def q(shape, bits, gs=64):
    w = mx.random.normal(shape).astype(mx.bfloat16) * 0.02
    return mx.quantize(w, group_size=gs, bits=bits, mode="affine")


def timeit(fn):
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    out = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for _ in range(CHAIN):
            r = fn()
        mx.eval(r)
        mx.synchronize()
        out.append((time.perf_counter() - t0) * 1000 / CHAIN)
    return statistics.median(out)


def bench_linear(name, out_dims, in_dims, bits, gs=64):
    w, s, b = q((out_dims, in_dims), bits, gs)
    x = mx.random.normal((1, 1, in_dims)).astype(mx.bfloat16)
    mx.eval(w, s, b, x)
    fn = lambda: mx.quantized_matmul(  # noqa: E731
        x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits
    )
    ms = timeit(fn)
    nbytes = w.nbytes + s.nbytes + b.nbytes
    del w, s, b, x
    mx.clear_cache()
    return ms, nbytes


def bench_switch(name, out_dims, in_dims, bits, gs=64):
    w, s, b = q((EXPERTS_ALLOC, out_dims, in_dims), bits, gs)
    x = mx.random.normal((1, 1, 1, 1, in_dims)).astype(mx.bfloat16)
    idx = mx.array([[[i % EXPERTS_ALLOC for i in range(TOPK)]]], dtype=mx.uint32)
    mx.eval(w, s, b, x, idx)
    fn = lambda: mx.gather_qmm(  # noqa: E731
        x, w, s, b, rhs_indices=idx, transpose=True, group_size=gs, bits=bits
    )
    ms = timeit(fn)
    per_expert = (w.nbytes + s.nbytes + b.nbytes) / EXPERTS_ALLOC
    del w, s, b, x, idx
    mx.clear_cache()
    return ms, per_expert * TOPK


PARTS = [
    ("fc_embedding", "linear", HIDDEN, HIDDEN),
    ("fc_hidden", "linear", HIDDEN, HIDDEN),
    ("mixer.input_mix_weight_down", "linear", LOWRANK, HC * HIDDEN),
    ("mixer.input_mix_weight_up", "linear", HC * HIDDEN, LOWRANK),
    ("switch_mlp.gate_proj", "switch", INTER, HIDDEN),
    ("switch_mlp.up_proj", "switch", INTER, HIDDEN),
    ("switch_mlp.down_proj", "switch", HIDDEN, INTER),
]


def main():
    print(f"M=1, chain={CHAIN}, median of {ITERS}, top-{TOPK} routing\n")
    print("| part | 4-bit ms | 8-bit ms | delta ms | 4-bit read KB | 8-bit read KB |")
    print("|---|---|---|---|---|---|")
    t4 = t8 = 0.0
    r4 = r8 = 0.0
    for name, kind, o, i in PARTS:
        f = bench_linear if kind == "linear" else bench_switch
        m4, b4 = f(name, o, i, 4)
        m8, b8 = f(name, o, i, 8)
        t4 += m4
        t8 += m8
        r4 += b4
        r8 += b8
        print(f"| {name} | {m4:.4f} | {m8:.4f} | {m8 - m4:+.4f} "
              f"| {b4/1024:.0f} | {b8/1024:.0f} |")
    print(f"| **draft block total** | **{t4:.4f}** | **{t8:.4f}** "
          f"| **{t8 - t4:+.4f}** | {r4/1024:.0f} | {r8/1024:.0f} |")


if __name__ == "__main__":
    main()
