"""Worst-case state rrmse of candidate split masks over seeds and lengths."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "gdn-scan"))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax2 import gated_delta_fused_nax2, mask, DEFAULT, FAST
from test_exact import make_inputs, err

MASKS = [DEFAULT, FAST, mask(wy=1, u=1, ws=1, state=1), mask(u=1, state=1)]
CASES = [(512, 0), (512, 3), (2048, 0), (2048, 5), (2049, 1), (1024, 7)]
res = {m: [] for m in MASKS}
c8 = []
for T, seed in CASES:
    q, k, v, g, beta, st = make_inputs(T, seed=seed)
    g = g.astype(mx.float32); beta = beta.astype(mx.float32)
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    yr, sr = gated_delta_kernel(q32, k32, v32, g, beta, st); mx.eval(yr, sr)
    y, s = gated_delta_fused_chunk(q, k, v, g, beta, st); mx.eval(y, s)
    c8.append((err(y, yr)[2], err(s, sr)[2]))
    for m in MASKS:
        y, s = gated_delta_fused_nax2(q, k, v, g, beta, st, split=m); mx.eval(y, s)
        res[m].append((err(y, yr)[2], err(s, sr)[2]))
print(f"{'mask':<8}{'worst y rrmse':>15}{'worst state rrmse':>19}")
print(f"{'C=8':<8}{max(x[0] for x in c8):>15.3e}{max(x[1] for x in c8):>19.3e}")
for m in MASKS:
    print(f"{m:#09x}{'':<3}{max(x[0] for x in res[m]):>15.3e}{max(x[1] for x in res[m]):>19.3e}")
