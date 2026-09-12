# SPDX-License-Identifier: MIT
"""moe_weighted_sum: bit-identical in clone mode, both layouts.

Bar from engine/patches/round2/wsum10/REPORT.md: 0 of 5,242,880 elements differ
in clone mode, which reproduces MLX's own eight-partial-accumulator bf16 axis
reduce. The fp32 modes are more accurate than the reference, so they are held
to a relative bound instead, and clone is the default for exactly that reason.
"""

import mlx.core as mx
import pytest

from exactness import bit_identical, max_ulp, rrmse
from titan.kernels import moe_weighted_sum as op

K = 10   # Flash-Next top_k


def _inputs(rows, dim, k=K, sorted_layout=True, seed=0):
    mx.random.seed(seed)
    n = rows * k
    ys = mx.random.normal((n, 1, dim)).astype(mx.bfloat16)
    scores = mx.random.uniform(shape=(1, rows, k)).astype(mx.bfloat16)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    if sorted_layout:
        perm = mx.array(
            mx.random.permutation(n).tolist(), dtype=mx.uint32
        )
    else:
        perm = None
    mx.eval(ys, scores)
    if perm is not None:
        mx.eval(perm)
    return ys, perm, scores


@pytest.mark.exactness
@pytest.mark.parametrize("rows,dim", [(1, 256), (9, 256), (13, 2560), (128, 2560)])
@pytest.mark.parametrize("sorted_layout", [True, False])
def test_clone_is_bit_identical(seeded, rows, dim, sorted_layout):
    ys, perm, scores = _inputs(rows, dim, sorted_layout=sorted_layout)
    shape = (1, rows, dim)
    ref = op.reference(ys, perm, scores, shape, mode="clone")
    got = op.metal(ys, perm, scores, shape, mode="clone")
    assert bit_identical(got, ref), f"max {max_ulp(got, ref)} ULP"


@pytest.mark.exactness
def test_unsorted_layout_needs_no_permutation(seeded):
    """The MTP verify layout arrives in token order, so the kernel runs with an
    identity permutation and no gather at all."""
    ys, _perm, scores = _inputs(13, 2560, sorted_layout=False)
    shape = (1, 13, 2560)
    assert bit_identical(
        op.metal(ys, None, scores, shape, mode="clone"),
        op.reference(ys, None, scores, shape, mode="clone"),
    )


@pytest.mark.exactness
@pytest.mark.parametrize("mode", ["fast", "ops"])
def test_fp32_modes_are_close_not_identical(seeded, mode):
    """These accumulate more accurately than the reference does, so the gap is
    the reference's own error and is bounded relatively, not in ULPs."""
    ys, perm, scores = _inputs(128, 2560)
    shape = (1, 128, 2560)
    ref = op.reference(ys, perm, scores, shape)
    got = op.metal(ys, perm, scores, shape, mode=mode)
    assert rrmse(got, ref) < 2e-2


@pytest.mark.exactness
def test_top_k_six_and_eight(seeded):
    for k in (6, 8):
        ys, perm, scores = _inputs(32, 512, k=k)
        shape = (1, 32, 512)
        assert bit_identical(
            op.metal(ys, perm, scores, shape, mode="clone"),
            op.reference(ys, perm, scores, shape, mode="clone"),
        )


def test_supports_rejects_unknown_mode(seeded):
    ys, perm, scores = _inputs(8, 256)
    shape = (1, 8, 256)
    assert op.supports(op.key(ys, perm, scores, shape, mode="clone"))
    assert not op.supports(op.key(ys, perm, scores, shape, mode="nonsense"))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """The real prefill tail: T=2048, k=10, D=2560, 105 MB of expert output."""
    ys, perm, scores = _inputs(2048, 2560)
    shape = (1, 2048, 2560)
    ref = op.reference(ys, perm, scores, shape, mode="clone")
    got = op.metal(ys, perm, scores, shape, mode="clone")
    assert bit_identical(got, ref)
