"""State carry: 4 x 512-token chunks feeding state forward, versus one fp32
sequential pass over 2048 tokens. Does the split-precision NAX kernel drift?"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "gdn-scan"))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax import gated_delta_fused_nax
from kernel_nax2 import gated_delta_fused_nax2, mask, DEFAULT, FAST
from test_exact import make_inputs, err

T, NCH = 2048, 4
CS = T // NCH
q, k, v, g, beta, st0 = make_inputs(T)
g = g.astype(mx.float32); beta = beta.astype(mx.float32)
q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
y_ref, s_ref = gated_delta_kernel(q32, k32, v32, g, beta, st0)
mx.eval(y_ref, s_ref)

paths = [
    ("stock seq kernel", gated_delta_kernel),
    ("PR C=8", gated_delta_fused_chunk),
    ("PR NAX C=16 (orig)", gated_delta_fused_nax),
    ("NAX 3-pass (default)", lambda *a: gated_delta_fused_nax2(*a, split=DEFAULT)),
    ("NAX ws A-corr (fast)", lambda *a: gated_delta_fused_nax2(*a, split=FAST)),
]

print(f"{'path':<22}{'chunk':>7}{'y rrmse':>12}{'state rrmse':>13}")
for name, fn in paths:
    s = st0
    for c in range(NCH):
        sl = slice(c * CS, (c + 1) * CS)
        y, s = fn(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], s)
        mx.eval(y, s)
        sn = err(s, s_ref)[2] if c == NCH - 1 else float("nan")
        print(f"{name:<22}{c:>7}{err(y, y_ref[:, sl])[2]:>12.3e}{sn:>13.3e}")
