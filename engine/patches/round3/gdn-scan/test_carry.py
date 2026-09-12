"""State carry-over: 4 x 512-token chunks feeding state forward, versus one
fp32 sequential pass over the whole 2048 tokens. Answers whether the chunked
kernels' state error accumulates across prefill chunks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax import gated_delta_fused_nax
from test_exact import make_inputs, err

try:
    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq
except Exception:
    gated_delta_blocked_seq = None

T, NCH = 2048, 4
CS = T // NCH
q, k, v, g, beta, st0 = make_inputs(T)
g = g.astype(mx.float32); beta = beta.astype(mx.float32)
q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
y_ref, s_ref = gated_delta_kernel(q32, k32, v32, g, beta, st0)
mx.eval(y_ref, s_ref)

paths = [("stock seq kernel", gated_delta_kernel)]
if gated_delta_blocked_seq is not None:
    paths.append(("oMLX blocked_seq", gated_delta_blocked_seq))
paths += [("PR #4020 chunk C=8", gated_delta_fused_chunk),
          ("PR #4020 NAX C=16", gated_delta_fused_nax)]

print(f"{'path':<22}{'chunk':>7}{'y rrmse':>12}{'state rrmse':>13}{'state absmax':>14}")
for name, fn in paths:
    s = st0
    for c in range(NCH):
        sl = slice(c * CS, (c + 1) * CS)
        y, s = fn(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], s)
        mx.eval(y, s)
        yr = y_ref[:, sl]
        sa, sr, sn = err(s, s_ref) if c == NCH - 1 else (float("nan"),) * 3
        print(f"{name:<22}{c:>7}{err(y, yr)[2]:>12.3e}{sn:>13.3e}{sa:>14.3e}")
