"""Exactness of the PR #4020 chunked scan against the sequential recurrence.

Reference = mlx_lm's sequential Metal kernel run with fp32 q/k/v/g/beta and an
fp32 state, i.e. the exact per-token recurrence in fp32.
Compared: stock bf16 sequential kernel, oMLX's gated_delta_blocked_seq, PR #4020.
Run: ~/inference-server/kdev/bin/python test_exact.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_kernel
from kernel import gated_delta_fused_chunk
from kernel_nax import gated_delta_fused_nax

try:
    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq
except Exception:
    gated_delta_blocked_seq = None

B, Hk, Dk, Hv, Dv = 1, 16, 128, 48, 128


def make_inputs(T, seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    # q, k as the model produces them: L2-normalised, q scaled by 1/sqrt(Dk)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    q = q * mx.rsqrt(mx.sum(mx.square(q), -1, keepdims=True) + 1e-6) * (Dk ** -0.5)
    k = k * mx.rsqrt(mx.sum(mx.square(k), -1, keepdims=True) + 1e-6)
    v = mx.random.normal((B, T, Hv, Dv))
    # real gate construction: A ~ U(0,16), dt_bias = 1, a ~ N(0,1) -> g near 1
    A_log = mx.log(mx.random.uniform(low=1e-3, high=16.0, shape=(Hv,)))
    dt_bias = mx.ones((Hv,))
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv)) - 2.0  # small beta, sigmoid ~ 0.12
    g = mx.exp(-mx.exp(A_log) * nn.softplus(a + dt_bias))
    beta = mx.sigmoid(b)
    st = mx.random.normal((B, Hv, Dv, Dk)) * 0.05
    out = [x.astype(dtype) for x in (q, k, v, g, beta)] + [st.astype(mx.float32)]
    mx.eval(out)
    return out


def err(x, ref):
    x = x.astype(mx.float32)
    ref = ref.astype(mx.float32)
    d = mx.abs(x - ref)
    return float(d.max()), float((d / (mx.abs(ref) + 1e-6)).max()), float(
        mx.sqrt(mx.mean(mx.square(x - ref)) / (mx.mean(mx.square(ref)) + 1e-30))
    )


def run(T):
    q, k, v, g, beta, st = make_inputs(T)
    # production feeds fp32 gates (compute_g returns fp32); use them everywhere
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    q32, k32, v32, g32, b32 = (x.astype(mx.float32) for x in (q, k, v, g, beta))
    y_ref, s_ref = gated_delta_kernel(q32, k32, v32, g32, b32, st)
    mx.eval(y_ref, s_ref)

    rows = []
    y, s = gated_delta_kernel(q, k, v, g, beta, st)
    mx.eval(y, s)
    rows.append(("stock seq kernel (bf16)", y, s))

    if gated_delta_blocked_seq is not None:
        y, s = gated_delta_blocked_seq(q, k, v, g, beta, st)
        mx.eval(y, s)
        rows.append(("oMLX blocked_seq (bf16)", y, s))

    y, s = gated_delta_fused_chunk(q, k, v, g, beta, st)
    mx.eval(y, s)
    rows.append(("PR #4020 chunk C=8 (bf16)", y, s))

    y, s = gated_delta_fused_nax(q, k, v, g, beta, st)
    mx.eval(y, s)
    rows.append(("PR #4020 NAX C=16 (bf16)", y, s))

    y, s = gated_delta_fused_chunk(q32, k32, v32, g32, b32, st)
    mx.eval(y, s)
    rows.append(("PR #4020 chunk C=8 (fp32)", y, s))

    y, s = gated_delta_fused_nax(q32, k32, v32, g32, b32, st)
    mx.eval(y, s)
    rows.append(("PR #4020 NAX C=16 (fp32)", y, s))

    print(f"\nT = {T}   ref |y|max={float(mx.abs(y_ref).max()):.4g} "
          f"|state|max={float(mx.abs(s_ref).max()):.4g}")
    print(f"{'path':<28}{'y absmax':>11}{'y relmax':>11}{'y rrmse':>11}"
          f"{'st absmax':>11}{'st relmax':>11}{'st rrmse':>11}")
    for name, y, s in rows:
        ya, yr, yn = err(y, y_ref)
        sa, sr, sn = err(s, s_ref)
        print(f"{name:<28}{ya:>11.3e}{yr:>11.3e}{yn:>11.3e}{sa:>11.3e}{sr:>11.3e}{sn:>11.3e}")


if __name__ == "__main__":
    for T in (64, 512, 2048):
        run(T)
