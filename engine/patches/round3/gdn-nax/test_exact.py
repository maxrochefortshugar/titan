"""Exactness of the split-precision NAX GDN kernel against the fp32 recurrence.

Reference = mlx_lm's sequential Metal kernel on fp32 q/k/v/g/beta and fp32
state, i.e. the exact per-token recurrence in fp32. Same input construction as
round3/gdn-scan/test_exact.py.
Run: ~/inference-server/kdev/bin/python test_exact.py
"""
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
from test_exact import make_inputs, err  # gdn-scan's builders

try:
    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq
except Exception:
    gated_delta_blocked_seq = None

MASKS = [
    ("NAX 3-pass (default)", DEFAULT),
    ("NAX ws A-corr (fast)", FAST),
    ("NAX min sites", mask(wy=1, u=1, ws=1, state=1)),
]


def run(T, seed=0):
    q, k, v, g, beta, st = make_inputs(T, seed=seed)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    y_ref, s_ref = gated_delta_kernel(q32, k32, v32, g, beta, st)
    mx.eval(y_ref, s_ref)

    rows = []
    y, s = gated_delta_kernel(q, k, v, g, beta, st); mx.eval(y, s)
    rows.append(("stock seq kernel", y, s))
    if gated_delta_blocked_seq is not None:
        y, s = gated_delta_blocked_seq(q, k, v, g, beta, st); mx.eval(y, s)
        rows.append(("oMLX blocked_seq", y, s))
    y, s = gated_delta_fused_chunk(q, k, v, g, beta, st); mx.eval(y, s)
    rows.append(("PR C=8", y, s))
    y, s = gated_delta_fused_nax(q, k, v, g, beta, st); mx.eval(y, s)
    rows.append(("PR NAX C=16 (orig)", y, s))
    for name, m in MASKS:
        y, s = gated_delta_fused_nax2(q, k, v, g, beta, st, split=m); mx.eval(y, s)
        rows.append((name, y, s))

    print(f"\nT = {T} seed={seed}  ref |y|max={float(mx.abs(y_ref).max()):.4g} "
          f"|state|max={float(mx.abs(s_ref).max()):.4g}")
    print(f"{'path':<24}{'y absmax':>11}{'y rrmse':>11}{'st absmax':>11}{'st rrmse':>11}")
    for name, y, s in rows:
        ya, _, yn = err(y, y_ref)
        sa, _, sn = err(s, s_ref)
        print(f"{name:<24}{ya:>11.3e}{yn:>11.3e}{sa:>11.3e}{sn:>11.3e}")


if __name__ == "__main__":
    for T in (512, 2048):
        run(T)
    run(512, seed=3)
    run(2049, seed=1)  # ragged tail chunk
