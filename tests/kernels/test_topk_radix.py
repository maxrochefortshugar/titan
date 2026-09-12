# SPDX-License-Identifier: MIT
"""topk_radix: an exact top-K, no tolerance and no fallback.

Bar from engine/patches/round3/small-items/REPORT.md item 1: the value multiset
is bit-identical to mx.topk, and the indices are a valid selection. Ties at the
boundary value are broken arbitrarily, exactly as mx.topk breaks them
arbitrarily, so the two agree as sets whenever the K-th largest is unique and
always agree as multisets.
"""

import mlx.core as mx
import pytest

from titan.kernels import topk_radix as op


def _check(row, k):
    val, idx = op.metal(row, k)
    mx.eval(val, idx)
    ref = mx.sort(mx.topk(row.reshape(-1), k))
    got = mx.sort(val)
    mx.eval(ref, got)
    assert bool(mx.all(got == ref).item()), "value multiset differs from mx.topk"
    ids = idx.tolist()
    assert len(set(ids)) == k, "indices are not distinct"
    picked = row.reshape(-1)[idx]
    mx.eval(picked)
    assert bool(mx.all(picked == val).item()), "indices do not point at the values"


@pytest.mark.exactness
@pytest.mark.parametrize("v,k", [(4096, 1), (4096, 64), (32768, 256), (65536, 1024)])
def test_matches_mx_topk(seeded, v, k):
    row = mx.random.normal((v,)).astype(mx.float32)
    mx.eval(row)
    _check(row, k)


@pytest.mark.exactness
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_dtypes(seeded, dtype):
    row = mx.random.normal((8192,)).astype(dtype)
    mx.eval(row)
    _check(row, 128)


@pytest.mark.exactness
def test_heavy_ties_at_the_boundary(seeded):
    """Many equal values straddling the cut is where a radix select is most
    likely to over- or under-fill the output."""
    row = mx.concatenate([
        mx.zeros((6000,), dtype=mx.float32),
        mx.ones((2000,), dtype=mx.float32),
        mx.full((192,), 2.0, dtype=mx.float32),
    ])
    mx.eval(row)
    _check(row, 256)


@pytest.mark.exactness
def test_all_identical_values(seeded):
    row = mx.full((4096,), -1.5, dtype=mx.float32)
    mx.eval(row)
    _check(row, 64)


@pytest.mark.exactness
def test_negatives_and_mixed_signs(seeded):
    """The order-preserving key has to reverse the negative half; if it did
    not, an all-negative row would come back with the K smallest."""
    row = -mx.abs(mx.random.normal((8192,))).astype(mx.float32)
    mx.eval(row)
    _check(row, 64)
    mixed = (mx.random.normal((8192,)) * 100).astype(mx.float32)
    mx.eval(mixed)
    _check(mixed, 64)


@pytest.mark.exactness
def test_row_shaped_input(seeded):
    row = mx.random.normal((1, 4096)).astype(mx.float32)
    mx.eval(row)
    _check(row, 32)


def test_supports_rejects_multi_row_input(seeded):
    row = mx.random.normal((4096,)).astype(mx.float32)
    assert op.supports(op.key(row, 64))
    assert not op.supports(op.key(mx.zeros((4, 4096)), 64))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """The real lm_head row: 248,320 wide, shortlist of 2048."""
    row = mx.random.normal((248320,)).astype(mx.float32)
    mx.eval(row)
    _check(row, 2048)
