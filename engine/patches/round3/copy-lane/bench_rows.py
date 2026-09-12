#!/usr/bin/env python3
"""Cost of a verify forward's M-dependent parts at the real Flash-Next shapes.

The copy lane's whole economics is "one verify row costs X ms, an accepted
token is worth 24.4 ms", so X has to be measured, not assumed. This benches
the two pieces that scale with M and that fit in a couple of hundred MB:
the lm_head projection (2560 -> 248320) and one MoE expert gather
(top-10 of 512, up-proj [512,640,2560]).

Attention is not benched here: it needs a populated KV cache at a realistic
context length, which would mean loading the model. Its M scaling is bounded
instead by the fused-kernel envelopes read out of the oMLX sources, printed
at the end.

Run: ~/inference-server/kdev/bin/python bench_rows.py [--bits 4|8] [--maxm 16]
"""
import argparse
import statistics as st
import time

import mlx.core as mx

CHAIN = 10
ITERS = 21


def qlin(K, N, bits, gs=64):
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    return mx.quantize(w, group_size=gs, bits=bits)


def bench(fn, iters=ITERS):
    for _ in range(3):
        fn()
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(CHAIN):
            fn()
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000 / CHAIN)
    # The GPU is shared with the live daemon: the minimum is the only
    # statistic that survives another process's bursts.
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--maxm", type=int, default=16)
    a = ap.parse_args()

    K, N = 2560, 248320
    wq, sc, bi = qlin(K, N, a.bits)
    ms_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16]
    ms_list = [m for m in ms_list if m <= a.maxm]

    print(f"lm_head {K} -> {N}, {a.bits}-bit gs64, bf16 activations")
    print(f"{'M':>3s} {'ms':>8s} {'ms/row':>8s} {'d ms/row':>9s}")
    head = {}
    prev = None
    for m in ms_list:
        x = mx.random.normal((1, m, K)).astype(mx.bfloat16)
        f = lambda: mx.eval(mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                                group_size=64, bits=a.bits))
        t = bench(f)
        head[m] = t
        d = "" if prev is None else f"{(t - prev[1]) / (m - prev[0]):9.4f}"
        print(f"{m:3d} {t:8.4f} {t / m:8.4f} {d:>9s}")
        prev = (m, t)
    del wq, sc, bi

    E, D, Hs = 512, 640, 2560
    TOPK = 10
    wq, sc, bi = mx.quantize(
        mx.random.normal((E, D, Hs)).astype(mx.bfloat16), group_size=64, bits=4)
    print()
    print(f"MoE up-proj gather, {E} experts top-{TOPK}, [{E},{D},{Hs}], 4-bit gs64")
    print(f"{'M':>3s} {'ms':>8s} {'x48 layers':>11s} {'d/row x48':>10s}")
    prev = None
    for m in ms_list:
        x = mx.broadcast_to(
            mx.random.normal((1, m, 1, 1, Hs)).astype(mx.bfloat16),
            (1, m, TOPK, 1, Hs))
        idx = mx.sort(mx.random.randint(0, E, (1, m, TOPK)), axis=-1).astype(mx.uint32)
        mx.eval(idx, x)
        f = lambda: mx.eval(mx.gather_qmm(x, wq, sc, bi,
                                          rhs_indices=idx, transpose=True,
                                          group_size=64, bits=4))
        t = bench(f)
        d = "" if prev is None else f"{(t - prev[1]) / (m - prev[0]) * 48:10.3f}"
        print(f"{m:3d} {t:8.4f} {t * 48:11.3f} {d:>10s}")
        prev = (m, t)

    print()
    print("fused-kernel envelopes that bound M = block + 1 (read from oMLX):")
    print("  qwen35_verify_qmm.vk_eligible          3 <= M <= 6   (:421-430)")
    print("  qwen35_verify_qmm.patched_call gate    2 <= M <= 6   (:462)")
    print("  qwen35_verify_sdpa_split               q_len*gqa <= 32, rows <= 5/4 (:10-11,38)")
    print("  turboquant _DECODE_MULTIROW_MAX_Q_LEN  L <= 15       (:30)")
    print("  turboquant fold path n_repeats*L       <= 24         (:34,362)")
    print("  qwen4_exp _gathered_min_query_tokens   >= 16 but excluded on target_verify")
    print("  => hard cap M <= 15, i.e. OMLX_MTP_COPY_MAX <= 14")


if __name__ == "__main__":
    main()
