"""Per-layer microbench of the GDN prefill scan at the Flash-Next head shape.
Runs only while ~/inference-server/staging/GPU_FREE exists."""
import sys, os, time, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "gdn-scan"))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")

if not os.path.exists(os.path.expanduser("~/inference-server/staging/GPU_FREE")):
    raise SystemExit("GPU_FREE absent; not benchmarking")

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax import gated_delta_fused_nax
from kernel_nax2 import gated_delta_fused_nax2, mask, DEFAULT, FAST
from test_exact import make_inputs

try:
    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq
except Exception:
    gated_delta_blocked_seq = None

CHAIN = int(os.environ.get("CHAIN", "4"))
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


MASKS = [(0, "no split (= PR)"), (mask(wy=1, u=1, ws=1, state=1), "min sites"),
         (FAST, "ws A-corr (fast)"), (DEFAULT, "3-pass (default)")]
print(f"chain={CHAIN}")
print(f"{'T':>6}  {'path':<26}{'ms/layer':>10}{'x stock':>9}{'ms/chunk x36':>14}")
for T in (512, 2048):
    q, k, v, g, beta, s0 = make_inputs(T)
    g = g.astype(mx.float32); beta = beta.astype(mx.float32); mx.eval(g, beta)
    base = timeit(lambda: gated_delta_kernel(q, k, v, g, beta, s0))
    rows = [("stock seq kernel", base)]
    if gated_delta_blocked_seq is not None:
        rows.append(("oMLX blocked_seq", timeit(lambda: gated_delta_blocked_seq(q, k, v, g, beta, s0))))
    rows.append(("PR C=8", timeit(lambda: gated_delta_fused_chunk(q, k, v, g, beta, s0))))
    rows.append(("PR NAX C=16 (orig)", timeit(lambda: gated_delta_fused_nax(q, k, v, g, beta, s0))))
    for m, nm in MASKS:
        rows.append((f"NAX {nm}",
                     timeit(lambda m=m: gated_delta_fused_nax2(q, k, v, g, beta, s0, split=m))))
    for name, t in rows:
        print(f"{T:>6}  {name:<26}{t*1e3:>10.3f}{base/t:>9.2f}{t*1e3*36:>14.1f}")
    print()
