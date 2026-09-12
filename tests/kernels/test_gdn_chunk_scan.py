# SPDX-License-Identifier: MIT
"""gdn_chunk_scan: state rrmse under 1e-5 against the fp32 per-token recurrence.

Bar from engine/patches/round3/gdn-scan/REPORT.md section 4. The judgement is
on the state, since that is what survives the chunk boundary and feeds decode.
The C = 8 kernel sits at about 6e-7 there, cleared 16x. The NAX C = 16 variant
misses the bar at 5.4e-4 and is asserted to miss it, so a future change that
quietly makes it the default fails here.

Inputs follow the real construction: q and k L2-normalised (which is what
conditions the WY inverse), q scaled by Dk**-0.5, A ~ U(0, 16) with dt_bias 1
so the decay sits just under 1, beta = sigmoid(b - 2) ~ 0.12, and a nonzero
random initial state.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from exactness import rrmse
from titan.kernels import gdn_chunk_scan as op

B, HK, HV, DK, DV = 1, 16, 48, 128, 128
STATE_BAR = 1e-5


def _inputs(T, seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, HK, DK))
    k = mx.random.normal((B, T, HK, DK))
    q = q * mx.rsqrt(mx.sum(mx.square(q), -1, keepdims=True) + 1e-6) * (DK ** -0.5)
    k = k * mx.rsqrt(mx.sum(mx.square(k), -1, keepdims=True) + 1e-6)
    v = mx.random.normal((B, T, HV, DV))
    a_log = mx.log(mx.random.uniform(low=1e-3, high=16.0, shape=(HV,)))
    g = mx.exp(-mx.exp(a_log) * nn.softplus(mx.random.normal((B, T, HV)) + 1.0))
    beta = mx.sigmoid(mx.random.normal((B, T, HV)) - 2.0)
    state = (mx.random.normal((B, HV, DV, DK)) * 0.05).astype(mx.float32)
    out = [x.astype(dtype) for x in (q, k, v)] + [g.astype(mx.float32),
                                                  beta.astype(mx.float32), state]
    mx.eval(*out)
    return out


@pytest.mark.exactness
@pytest.mark.parametrize("T", [8, 16, 64])
def test_state_error_under_bar(seeded, T):
    q, k, v, g, beta, state = _inputs(T)
    ref_y, ref_state = op.reference(*[x.astype(mx.float32) for x in (q, k, v)],
                                    g, beta, state)
    got_y, got_state = op.metal(*[x.astype(mx.float32) for x in (q, k, v)],
                                g, beta, state)
    assert rrmse(got_state, ref_state) < STATE_BAR
    assert rrmse(got_y, ref_y) < STATE_BAR


@pytest.mark.exactness
def test_bf16_inputs_match_the_bf16_rounding_floor(seeded):
    """Fed bf16, y carries bf16's own rounding (about 1.7e-3 rrmse) while the
    state stays at the kernel's own error, which is the point of the fp32
    state."""
    q, k, v, g, beta, state = _inputs(64)
    ref_y, ref_state = op.reference(q.astype(mx.float32), k.astype(mx.float32),
                                    v.astype(mx.float32), g, beta, state)
    got_y, got_state = op.metal(q, k, v, g, beta, state)
    assert rrmse(got_state, ref_state) < STATE_BAR
    assert rrmse(got_y, ref_y) < 5e-3


@pytest.mark.exactness
def test_ragged_tail_chunk(seeded):
    """T not a multiple of C = 8 takes the masked tail path."""
    q, k, v, g, beta, state = _inputs(21)
    ref_y, ref_state = op.reference(*[x.astype(mx.float32) for x in (q, k, v)],
                                    g, beta, state)
    got_y, got_state = op.metal(*[x.astype(mx.float32) for x in (q, k, v)],
                                g, beta, state)
    assert rrmse(got_state, ref_state) < STATE_BAR
    assert rrmse(got_y, ref_y) < STATE_BAR


@pytest.mark.exactness
def test_chunked_carry_does_not_compound(seeded):
    """Four chunks feeding state forward must land where one pass over the
    whole sequence lands. This is the failure mode that would bite at 65k."""
    q, k, v, g, beta, state = _inputs(64)
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    _, whole = op.reference(q32, k32, v32, g, beta, state)
    carried = state
    for i in range(4):
        s = slice(i * 16, (i + 1) * 16)
        _, carried = op.metal(q32[:, s], k32[:, s], v32[:, s], g[:, s],
                              beta[:, s], carried)
    assert rrmse(carried, whole) < STATE_BAR


@pytest.mark.exactness
def test_nax_variant_is_not_exact(seeded):
    """Documented as not exact; asserted so, so it cannot become the default by
    accident. The cause is matmul2d's relaxed_precision, which the source PR
    leaves on and which cannot be turned off without garbage."""
    if not op.nax_available():
        pytest.skip("NAX metal sources not present")
    q, k, v, g, beta, state = _inputs(64)
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    _, ref_state = op.reference(q32, k32, v32, g, beta, state)
    _, nax_state = op.nax(q32, k32, v32, g, beta, state)
    assert rrmse(nax_state, ref_state) > STATE_BAR
    # and the registry must never pick it
    assert not op.supports(op.key(q, k, v, g, beta, state, variant="nax"))


def test_supports_rejects_unsupported_head_counts(seeded):
    q, k, v, g, beta, state = _inputs(16)
    assert op.supports(op.key(q, k, v, g, beta, state))
    bad_q = mx.zeros((B, 16, 7, DK), dtype=mx.bfloat16)
    bad_g = mx.zeros((B, 16, 7), dtype=mx.float32)
    bad_v = mx.zeros((B, 16, 7, DV), dtype=mx.bfloat16)
    assert not op.supports(
        op.key(bad_q, bad_q, bad_v, bad_g, bad_g, None)
    )


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """A real prefill chunk: T = 2048 over 48 value heads. The reference is a
    2048-step Python loop, which is why this one is marked slow."""
    q, k, v, g, beta, state = _inputs(2048)
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    _, ref_state = op.reference(q32, k32, v32, g, beta, state)
    _, got_state = op.metal(q32, k32, v32, g, beta, state)
    assert rrmse(got_state, ref_state) < STATE_BAR
