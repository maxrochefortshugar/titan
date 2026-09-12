# SPDX-License-Identifier: MIT
"""gdn_norm_gate: bit-identical to the plain norm-and-gate chain.

Bar from engine/patches/round2/gdn-norm/REPORT.md: bit-identical, not merely
within 1 ULP, because the kernel reproduces mlx's own rms_single_row reduction.
"""

import mlx.core as mx
import pytest

from titan.kernels import gdn_norm_gate as op

from exactness import bit_identical, max_ulp

HV, DV = 48, 128       # the Flash-Next GDN value heads


def _inputs(seq, dtype=mx.bfloat16, hv=HV, dv=DV):
    x = mx.random.normal((1, seq, hv, dv)).astype(dtype)
    gate = (mx.random.normal((1, seq, hv, dv)) * 3.0).astype(dtype)
    w = (1.0 + 0.1 * mx.random.normal((dv,))).astype(dtype)
    mx.eval(x, gate, w)
    return x, gate, w


@pytest.mark.exactness
@pytest.mark.parametrize("seq", [1, 2, 8, 64])
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_bit_identical(seeded, seq, activation):
    x, gate, w = _inputs(seq)
    ref = op.reference(x, gate, w, eps=1e-6, activation=activation)
    got = op.metal(x, gate, w, eps=1e-6, activation=activation)
    assert bit_identical(got, ref), f"max {max_ulp(got, ref)} ULP at T={seq}"


@pytest.mark.exactness
def test_float16_also_exact(seeded):
    x, gate, w = _inputs(16, dtype=mx.float16)
    ref = op.reference(x, gate, w, eps=1e-6, activation="sigmoid")
    got = op.metal(x, gate, w, eps=1e-6, activation="sigmoid")
    assert bit_identical(got, ref)


@pytest.mark.exactness
def test_extreme_gate_values_do_not_overflow(seeded):
    """The kernel computes the sigmoid on the stable side; check both tails."""
    x = mx.random.normal((1, 8, HV, DV)).astype(mx.bfloat16)
    gate = mx.where(
        mx.random.uniform(shape=(1, 8, HV, DV)) > 0.5, 200.0, -200.0
    ).astype(mx.bfloat16)
    w = mx.ones((DV,), dtype=mx.bfloat16)
    mx.eval(x, gate, w)
    got = op.metal(x, gate, w, eps=1e-6, activation="sigmoid")
    ref = op.reference(x, gate, w, eps=1e-6, activation="sigmoid")
    mx.eval(got)
    assert bit_identical(got, ref)
    assert not bool(mx.any(mx.isnan(got.astype(mx.float32))).item())


def test_supports_rejects_mismatched_dtypes(seeded):
    x, gate, w = _inputs(8)
    assert op.supports(op.key(x, gate, w, eps=1e-6, activation="sigmoid"))
    assert not op.supports(
        op.key(x, gate, w.astype(mx.float32), eps=1e-6, activation="sigmoid")
    )
    assert not op.supports(op.key(x, gate, w, eps=1e-6, activation="relu"))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """The real prefill shape: [1, 2048, 48, 128] bf16, 25 MB per operand."""
    x, gate, w = _inputs(2048)
    ref = op.reference(x, gate, w, eps=1e-6, activation="sigmoid")
    got = op.metal(x, gate, w, eps=1e-6, activation="sigmoid")
    assert bit_identical(got, ref)
