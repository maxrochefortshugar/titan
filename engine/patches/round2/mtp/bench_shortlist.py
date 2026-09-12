#!/usr/bin/env python3
"""Cost of a shortlisted draft lm_head pass against the full-vocabulary pass.

Full pass:      quantized_matmul(x[1,2560], W[248320,2560])       -> [1,248320]
Shortlist pass: argpartition top-Ks of a previous full row (once per cycle)
                + gather Ks rows of the packed head (once per cycle)
                + quantized_matmul(x[1,2560], W_s[Ks,2560])        per step
                + scatter back into a full -inf logprob row        per step
"""
import argparse, statistics as st, time
import mlx.core as mx
import mlx.nn as nn

CHAIN = 10
V, K = 248320, 2560


def timeit(fn, iters=15, warm=3):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)
    for _ in range(warm):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args()
    bits, gs = a.bits, a.gs

    w = mx.random.randint(0, 2**31 - 1, (V, K * bits // 32), dtype=mx.uint32)
    sc = (mx.random.normal((V, K // gs)) * 0.01).astype(mx.bfloat16)
    bi = (mx.random.normal((V, K // gs)) * 0.01).astype(mx.bfloat16)
    x = (mx.random.normal((1, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(w, sc, bi, x)

    full = lambda: mx.quantized_matmul(x, w, sc, bi, transpose=True,
                                       group_size=gs, bits=bits)
    t_full = timeit(full, a.iters)
    logits = full(); mx.eval(logits)
    print(f"lm_head {bits}-bit gs{gs}  full pass M=1: {t_full*1e3:.3f} ms")
    print(f"{'Ks':>6} {'topk ms':>9} {'gather ms':>10} {'qmm ms':>9} "
          f"{'scatter ms':>11} {'cycle2 ms':>10} {'vs 2 full':>10}")

    for Ks in (256, 512, 1024, 2048, 4096):
        t_topk = timeit(lambda: mx.argpartition(logits, kth=V - Ks,
                                                axis=-1)[..., -Ks:], a.iters)
        S = mx.argpartition(logits, kth=V - Ks, axis=-1)[0, -Ks:]
        mx.eval(S)
        gather = lambda: (w[S], sc[S], bi[S])
        t_gather = timeit(lambda: mx.concatenate(
            [w[S].reshape(-1)[:8].astype(mx.uint32),
             sc[S].reshape(-1)[:8].astype(mx.uint32),
             bi[S].reshape(-1)[:8].astype(mx.uint32)]), a.iters)
        ws, ss, bs = w[S], sc[S], bi[S]
        mx.eval(ws, ss, bs)
        t_qmm = timeit(lambda: mx.quantized_matmul(
            x, ws, ss, bs, transpose=True, group_size=gs, bits=bits), a.iters)
        small = mx.quantized_matmul(x, ws, ss, bs, transpose=True,
                                    group_size=gs, bits=bits)
        mx.eval(small)
        neg = mx.full((1, V), -3.0e38, dtype=mx.float32)

        def scat():
            return mx.put_along_axis(neg, S[None, :],
                                     small.astype(mx.float32), axis=-1)
        t_scat = timeit(scat, a.iters)
        # one cycle at depth 3 with step 0 full: topk + gather + 2*(qmm+scatter)
        cyc = t_topk + t_gather + 2 * (t_qmm + t_scat)
        print(f"{Ks:>6} {t_topk*1e3:9.3f} {t_gather*1e3:10.3f} {t_qmm*1e3:9.3f} "
              f"{t_scat*1e3:11.3f} {cyc*1e3:10.3f} {cyc/(2*t_full):10.2f}")


if __name__ == "__main__":
    main()
