# SPDX-License-Identifier: MIT
"""qsa_gathered_attention: the padded batched arm equals the per-row arm.

Bar from engine/patches/round3/qsa-batched/REPORT.md section 2: the loop route
is bit identical to running each row on its own, the padded route within one
bf16 ULP of it. The interesting cases are the ones with a phase: when
``pads[b] % compress_ratio != 0`` each row's block grid sits at a different
offset in physical space, and an implementation that indexes blocks physically
rather than logically gets a different set of tokens for the same query. So
every batched case here mixes phases deliberately.
"""

import math

import mlx.core as mx
import pytest

from exactness import bit_identical, max_ulp
from titan.kernels import qsa_gathered_attention as op

HQ, HKV, D, DI = 8, 2, 64, 32
RATIO = 4
BUDGET = 32          # tokens, so 8 blocks


def _cfg(budget=BUDGET):
    return op.QSAConfig(HQ, HKV, D, DI, RATIO, budget)


def _case(pads, length, width=256, seed=0, cfg=None):
    """Batched tensors plus the pooled bank, in phased slot space."""
    cfg = cfg or _cfg()
    mx.random.seed(seed)
    geo = op.QSAGeometry(width, pads, RATIO)
    batch = geo.batch
    queries = mx.random.normal((batch, HQ, length, D)).astype(mx.bfloat16)
    keys = mx.random.normal((batch, HKV, width, D)).astype(mx.bfloat16)
    values = mx.random.normal((batch, HKV, width, D)).astype(mx.bfloat16)
    index_queries = mx.random.normal((batch, length, 4, DI)).astype(mx.bfloat16)
    pooled = mx.random.normal((batch, geo.slots, DI)).astype(mx.bfloat16)
    mx.eval(queries, keys, values, index_queries, pooled)
    return (queries, keys, values, index_queries, pooled), geo, cfg


@pytest.mark.exactness
@pytest.mark.parametrize("pads", [[0], [7], [0, 0], [0, 7], [3, 6, 9, 12]])
@pytest.mark.parametrize("length", [1, 4])
def test_padded_arm_within_one_ulp_of_the_loop_arm(seeded, pads, length):
    args, geo, cfg = _case(pads, length)
    ref = op.reference(*args, geo, cfg)
    got = op.metal(*args, geo, cfg)
    assert max_ulp(got, ref) <= 1, f"max {max_ulp(got, ref)} ULP for pads={pads}"


@pytest.mark.exactness
def test_single_row_takes_the_same_path_in_both(seeded):
    """B = 1 with no padding is the decode arm, and there the padded route has
    nothing to pad, so it must be bit identical."""
    args, geo, cfg = _case([0], 1)
    assert bit_identical(op.metal(*args, geo, cfg), op.reference(*args, geo, cfg))


@pytest.mark.exactness
def test_a_phased_row_is_not_its_unphased_neighbour(seeded):
    """The guard on the whole design: two rows that differ only in left padding
    select different physical columns, so a phase-blind implementation would
    return the same answer for both. If this ever starts passing trivially the
    phase logic has stopped doing anything."""
    args, geo, cfg = _case([0, 7], 1, seed=5)
    out = op.metal(*args, geo, cfg)
    mx.eval(out)
    assert not bool(mx.all(out[0] == out[1]).item())


@pytest.mark.exactness
def test_batched_rows_match_their_own_single_row_calls(seeded):
    """A row must not care what it was batched with. This is what makes batch
    composition invisible in the output."""
    args, geo, cfg = _case([0, 7, 3, 11], 1, seed=2)
    queries, keys, values, index_queries, pooled = args
    batched = op.metal(*args, geo, cfg)
    mx.eval(batched)
    for b in range(geo.batch):
        row_geo = op.QSAGeometry(geo.width, [geo.pads[b]], RATIO)
        alone = op.reference(
            queries[b:b + 1], keys[b:b + 1], values[b:b + 1],
            index_queries[b:b + 1], pooled[b:b + 1], row_geo, cfg,
        )
        mx.eval(alone)
        assert max_ulp(batched[b:b + 1], alone) <= 1


@pytest.mark.exactness
def test_below_the_budget_every_block_is_selected(seeded):
    """A row with no more complete blocks than the budget selects all of them,
    so the arm reduces to dense attention over that row's visible prefix.
    Compared against a plain SDPA over exactly those columns."""
    args, geo, cfg = _case([232], 1, width=256, seed=4)   # 24 visible tokens
    queries, keys, values, _iq, _pk = args
    got = op.metal(*args, geo, cfg)
    groups = HQ // HKV
    k = keys[:, :, 232:, :]
    v = values[:, :, 232:, :]
    q = queries.transpose(0, 2, 1, 3).reshape(1, 1, HKV, groups, D)
    kt = k[:, None].astype(mx.float32).swapaxes(-1, -2)   # [1, 1, HKV, D, T]
    attn = (q.astype(mx.float32) @ kt) / math.sqrt(D)
    want = (mx.softmax(attn, axis=-1).astype(mx.bfloat16)
            @ v[:, None]).reshape(1, 1, HQ, D)
    mx.eval(got, want)
    assert max_ulp(got, want) <= 1


def test_supports_needs_a_sparse_crossover(seeded):
    args, geo, cfg = _case([0, 7], 1)
    assert op.supports(op.key(*args, geo, cfg))
    wide = op.QSAConfig(HQ, HKV, D, DI, RATIO, 4096)
    assert not op.supports(op.key(*args, geo, wide))


def test_config_validation_rejects_a_ragged_budget(seeded):
    with pytest.raises(ValueError):
        op.QSAConfig(HQ, HKV, D, DI, 4, 30).validate()
    with pytest.raises(ValueError):
        op.QSAConfig(7, 2, D, DI, 4, 32).validate()


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """Flash-Next QSA geometry: 24 query heads over 2 K/V heads, head dim 256,
    indexer dim 128, ratio 4, budget 2048, at a 16k context."""
    cfg = op.QSAConfig(24, 2, 256, 128, 4, 2048)
    mx.random.seed(0)
    pads = [0, 7, 3, 11]
    geo = op.QSAGeometry(16384, pads, 4)
    batch, length = len(pads), 1
    args = (
        mx.random.normal((batch, 24, length, 256)).astype(mx.bfloat16),
        mx.random.normal((batch, 2, 16384, 256)).astype(mx.bfloat16),
        mx.random.normal((batch, 2, 16384, 256)).astype(mx.bfloat16),
        mx.random.normal((batch, length, 4, 128)).astype(mx.bfloat16),
        mx.random.normal((batch, geo.slots, 128)).astype(mx.bfloat16),
    )
    mx.eval(*args)
    assert max_ulp(op.metal(*args, geo, cfg), op.reference(*args, geo, cfg)) <= 1
