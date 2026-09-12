# SPDX-License-Identifier: MIT
"""grouped_rmsnorm_bf16: within one bf16 ULP of the canonical grouped norm.

Bar from engine/patches/ple-fix/REPORT.md section 4.2: on a real 2048-token
chunk, no element anywhere differs by more than one bf16 ULP. The arithmetic is
the same; the fp32 rounding order is not (metal::rsqrt and a simd-tree
reduction against mlx's rms_norm).
"""

import mlx.core as mx
import pytest

from exactness import max_ulp, ulp_distance
from titan.kernels import grouped_rmsnorm_bf16 as op

GROUPS = 4


def _inputs(rows, hidden, groups=GROUPS, dtype=mx.bfloat16, scale=1.0):
    x = (scale * mx.random.normal((rows, groups * hidden))).astype(dtype)
    w = (0.1 * mx.random.normal((groups * hidden,))).astype(dtype)
    mx.eval(x, w)
    return x, w


@pytest.mark.exactness
@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("hidden", [256, 2560])
def test_within_one_ulp(seeded, rows, hidden):
    x, w = _inputs(rows, hidden)
    ref = op.reference(x, w, groups=GROUPS, eps=1e-6)
    got = op.metal(x, w, groups=GROUPS, eps=1e-6)
    assert max_ulp(got, ref) <= 1


@pytest.mark.exactness
@pytest.mark.parametrize("scale", [0.125, 8.0])
def test_stream_scale_does_not_widen_the_gap(seeded, scale):
    """The norm is scale invariant, so a stream eight times up or down must not
    move the ULP distance: if it does, the accumulator is the problem."""
    x, w = _inputs(32, 2560, scale=scale)
    ref = op.reference(x, w, groups=GROUPS, eps=1e-6)
    got = op.metal(x, w, groups=GROUPS, eps=1e-6)
    assert max_ulp(got, ref) <= 1


@pytest.mark.exactness
def test_most_elements_are_bit_identical(seeded):
    x, w = _inputs(64, 2560)
    ref = op.reference(x, w, groups=GROUPS, eps=1e-6)
    got = op.metal(x, w, groups=GROUPS, eps=1e-6)
    d = ulp_distance(got, ref)
    assert (d == 0).mean() > 0.5, f"only {(d == 0).mean():.3f} bit-identical"


def test_supports_rejects_fp32_and_ragged_groups(seeded):
    x, w = _inputs(8, 256)
    assert op.supports(op.key(x, w, groups=GROUPS, eps=1e-6))
    assert not op.supports(
        op.key(x.astype(mx.float32), w.astype(mx.float32), groups=GROUPS, eps=1e-6)
    )
    assert not op.supports(op.key(x, w, groups=3, eps=1e-6))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """The real hyper-connection stream: [2048, 4 x 2560] bf16."""
    x, w = _inputs(2048, 2560)
    ref = op.reference(x, w, groups=GROUPS, eps=1e-6)
    got = op.metal(x, w, groups=GROUPS, eps=1e-6)
    assert max_ulp(got, ref) <= 1
