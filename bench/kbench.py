#!/usr/bin/env python3
"""Roofline microbenchmark for the kernels that dominate Qwen3.8-Flash-Next on M5 Max.

Measures achieved bandwidth (GB/s) and compute (TFLOPS) for:
  - dense quantized matvec/matmul (qmm)     at M = 1, 2, 4, 8, 32, 2048   (attention/shared-expert/lm_head shapes)
  - MoE gather_qmm (E=512, top_k=10)        at T = 1, 2, 8 (decode/verify/batched) and 512, 2048 (prefill chunks)
  - copy bandwidth ceiling
Run:  ~/inference-server/kdev/bin/python kbench.py [--bits 4] [--quick]
Numbers are per-kernel, warm, median of N iterations. Compare against ~500 GB/s and ~61 TFLOPS on a 40-core M5 Max.
"""
import argparse, time, statistics as st
import mlx.core as mx

PEAK_BW = 614e9   # M5 Max spec
CHAIN = 10   # ops per eval, to amortise command-buffer submission
def timeit(fn, iters, warm=2):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)
    for _ in range(warm): run()
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize(); ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)

def qbytes(n_elems, bits, gs):
    return n_elems * bits / 8 + (n_elems / gs) * 2 * 2   # packed weights + fp16 scale & bias per group

def dense(M, N, K, bits, gs, iters):
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, gs, bits)
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)
    t = timeit(f, iters)
    by = qbytes(N * K, bits, gs) + (M * K + M * N) * 2
    return t, by / t / 1e9, 2 * M * N * K / t / 1e12

def moe(T, E, topk, D, I, bits, gs, iters, sorted_):
    # expert weights [E, I, D] for up-proj; gather over routed (token, expert) pairs
    w = mx.random.normal((E, I, D)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, gs, bits)
    x = mx.random.normal((T, 1, 1, D)).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (T, topk))
    if sorted_:
        flat = mx.sort(idx.reshape(-1)); idx = flat.reshape(T, topk)
    f = lambda: mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=gs, bits=bits, sorted_indices=sorted_)
    t = timeit(f, iters)
    touched = min(E, T * topk)           # distinct experts read, upper bound
    by = qbytes(touched * I * D, bits, gs) + T * D * 2 + T * topk * I * 2
    return t, by / t / 1e9, 2 * T * topk * I * D / t / 1e12

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bits", type=int, default=4); ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    iters = 5 if a.quick else 15
    print(f"mlx {mx.__version__}  bits={a.bits} gs={a.gs}  device={mx.device_info().get('device_name','?')}")
    # copy ceiling
    big = mx.zeros((512, 1024, 1024), dtype=mx.bfloat16); mx.eval(big)
    t = timeit(lambda: big + 1, iters); print(f"\ncopy ceiling: {2*big.nbytes/t/1e9:6.0f} GB/s  ({100*2*big.nbytes/t/PEAK_BW:.0f}% of spec)")
    print("\nDENSE quantized matmul (x[M,K] @ W[N,K]^T)")
    print(f"{'shape':<26}{'M':>6}{'ms':>9}{'GB/s':>8}{'TFLOPS':>9}")
    for name, N, K in (("q_proj 2560->6144", 6144, 2560), ("o_proj 6144->2560", 2560, 6144), ("shared mlp 2560->640", 640, 2560), ("lm_head 2560->248320", 248320, 2560)):
        for M in (1, 2, 4, 8, 32, 2048):
            if name.startswith("lm_head") and M > 8: continue
            t, gbs, tf = dense(M, N, K, a.bits, a.gs, iters)
            print(f"{name:<26}{M:>6}{t*1e3:>9.3f}{gbs:>8.0f}{tf:>9.2f}")
    print("\nMoE gather_qmm  E=512 top_k=10  up-proj [512, 640, 2560]")
    print(f"{'tokens':>8}{'rows/expert':>12}{'sorted':>8}{'ms':>9}{'GB/s':>8}{'TFLOPS':>9}")
    for T in (1, 2, 8, 512, 2048):
        for srt in ((False,) if T < 64 else (False, True)):
            t, gbs, tf = moe(T, 512, 10, 2560, 640, a.bits, a.gs, iters if T < 512 else 4, srt)
            print(f"{T:>8}{T*10/512:>12.2f}{str(srt):>8}{t*1e3:>9.3f}{gbs:>8.0f}{tf:>9.2f}")

if __name__ == "__main__":
    main()
