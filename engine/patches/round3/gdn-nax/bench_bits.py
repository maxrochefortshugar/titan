"""Cost of the split at each matmul site, one bit at a time."""
import sys, os, time, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "gdn-scan"))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
if not os.path.exists(os.path.expanduser("~/inference-server/staging/GPU_FREE")):
    raise SystemExit("GPU_FREE absent")
import mlx.core as mx
from kernel_nax2 import gated_delta_fused_nax2
from test_exact import make_inputs

SITES = ["KK^T", "WY Neumann", "W panel", "U", "W S^T", "Q K^T", "Q S^T", "out+=QKtd", "state upd"]
CHAIN, ITERS = 4, 11


def timeit(fn):
    def run():
        outs = []
        for _ in range(CHAIN):
            outs.extend(fn())
        mx.eval(*outs)
    for _ in range(3):
        run()
    ts = []
    for _ in range(ITERS):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)


T = 2048
q, k, v, g, beta, s0 = make_inputs(T)
g = g.astype(mx.float32); beta = beta.astype(mx.float32); mx.eval(g, beta)
base = timeit(lambda: gated_delta_fused_nax2(q, k, v, g, beta, s0, split=0))
print(f"T={T}  no split: {base*1e3:.3f} ms/layer")
print(f"{'bit':>4} {'site':<12}{'alone ms':>10}{'delta':>9}{'from 0x1de ms':>15}{'delta':>9}")
full = 0x1DE
tfull = timeit(lambda: gated_delta_fused_nax2(q, k, v, g, beta, s0, split=full))
for b, name in enumerate(SITES):
    t1 = timeit(lambda b=b: gated_delta_fused_nax2(q, k, v, g, beta, s0, split=1 << b))
    if full & (1 << b):
        t2 = timeit(lambda b=b: gated_delta_fused_nax2(q, k, v, g, beta, s0, split=full & ~(1 << b)))
        d2 = f"{(tfull - t2)*1e3:>+9.3f}"
        s2 = f"{t2*1e3:>15.3f}"
    else:
        d2, s2 = f"{'-':>9}", f"{'-':>15}"
    print(f"{b:>4} {name:<12}{t1*1e3:>10.3f}{(t1-base)*1e3:>+9.3f}{s2}{d2}")
print(f"     {'0x1de':<12}{tfull*1e3:>10.3f}{(tfull-base)*1e3:>+9.3f}")
