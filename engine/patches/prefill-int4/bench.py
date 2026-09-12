"""Before/after prefill matmul benchmark (stock mx.quantized_matmul vs int8xint4 NAX)."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, numpy as np
import kernel

def bench(fn, iters=5, chain=10):
    mx.eval(fn()); mx.synchronize(); ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); o = [fn() for _ in range(chain)]; mx.eval(o); mx.synchronize()
        ts.append((time.perf_counter() - t0) / chain)
    return float(np.median(ts))

SHAPES = [("q_proj", 6144, 2560), ("o_proj", 2560, 6144)]
print(f"{'shape':10s} {'M':>5s} {'N':>6s} {'K':>6s} {'stock ms':>9s} {'TF/s':>7s} "
      f"{'int4 ms':>9s} {'TF/s':>7s} {'prep ms':>8s} {'speedup':>8s}")
for name, N, K in SHAPES:
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    wq_s, sc, bs = kernel.prepare_weights(wq, s, b); mx.eval(wq_s, sc, bs)
    for M in (512, 2048):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16); mx.eval(x)
        t_stock = bench(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
        t_new = bench(lambda: kernel.prefill_qmm(x, wq_s, sc, bs))
        t_prep = bench(lambda: kernel.quantize_activations(x)[0])
        fl = 2 * M * N * K
        print(f"{name:10s} {M:5d} {N:6d} {K:6d} {t_stock*1e3:9.3f} {fl/t_stock/1e12:7.1f} "
              f"{t_new*1e3:9.3f} {fl/t_new/1e12:7.1f} {t_prep*1e3:8.3f} {t_stock/t_new:7.2f}x")
