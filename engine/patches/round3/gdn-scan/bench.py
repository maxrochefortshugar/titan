"""Per-layer microbench of the GDN prefill scan at the Flash-Next head shape."""
import sys, os, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax import gated_delta_fused_nax
from test_exact import make_inputs

try:
    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq
except Exception:
    gated_delta_blocked_seq = None

CHAIN = 4
ITERS = 15


def timeit(fn, chain=CHAIN, iters=ITERS):
    def run():
        outs = []
        for _ in range(chain):
            outs.extend(fn())
        mx.eval(*outs)
    for _ in range(3):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize()
        ts.append((time.perf_counter() - t0) / chain)
    return st.median(ts)


CHAIN = int(os.environ.get("CHAIN", "4"))
print(f"chain={CHAIN}")
print(f"{'T':>6}  {'path':<26}{'ms/layer':>10}{'x stock':>9}{'ms/chunk x36':>14}")
for T in (512, 2048):
    q, k, v, g, beta, s0 = make_inputs(T)
    g = g.astype(mx.float32); beta = beta.astype(mx.float32); mx.eval(g, beta)
    base = timeit(lambda: gated_delta_kernel(q, k, v, g, beta, s0))
    rows = [("stock seq kernel", base)]
    if gated_delta_blocked_seq is not None:
        rows.append(("oMLX blocked_seq", timeit(lambda: gated_delta_blocked_seq(q, k, v, g, beta, s0))))
    rows.append(("PR #4020 chunk C=8", timeit(lambda: gated_delta_fused_chunk(q, k, v, g, beta, s0))))
    rows.append(("PR #4020 NAX C=16", timeit(lambda: gated_delta_fused_nax(q, k, v, g, beta, s0))))
    for name, t in rows:
        print(f"{T:>6}  {name:<26}{t*1e3:>10.3f}{base/t:>9.2f}{t*1e3*36:>14.1f}")
    print()
