# SPDX-License-Identifier: MIT
"""moe_gather_int8: the one op whose fast path changes the numbers.

Bar from engine/patches/moe-int8/REPORT.md section 5: not bit-exact and not
trying to be. Quantising the activations to uint8 costs about 0.64 to 0.70% of
output RMS, measured the same on random tensors and on real checkpoint ones.
So the assertion is a relative-RMS bound, and there are two structural checks
that the error really is only the activation quantisation:

* the integer correction ``D - 128*qsum`` must stay exact, so the error must
  not grow with K (cancellation there would show up as a K-dependent floor);
* the affine bias term uses the exact fp32 group sum of the UNQUANTISED
  activations, so a weight tensor with large biases must not be worse.

The tables are the caller's, and the op refuses to select its fast path
without them: that is the opt-in that keeps an inexact kernel from being
picked by accident.
"""

import mlx.core as mx
import numpy as np
import pytest

from exactness import rel_rms
from titan.kernels import moe_gather_int8 as op
from titan.kernels.moe_gather_ws import build_tiles

BAR = 0.02   # fraction of output RMS; the report measures 0.0064 to 0.0070


def _weights(experts, n, k, seed=0):
    mx.random.seed(seed)
    dense = mx.random.normal((experts, n, k)).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=op.GROUP, bits=op.BITS)
    mx.eval(w, s, b)
    return w, s, b


def _sorted_routing(rows, experts, seed=0):
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, experts + 1)
    ids = np.sort(rng.choice(experts, size=rows, p=weights / weights.sum()))
    out = mx.array(ids.astype(np.uint32))
    mx.eval(out)
    return out


def _activations(rows, k, seed=0):
    mx.random.seed(seed + 7)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _run(rows, experts, n, k, seed=0):
    w, s, b = _weights(experts, n, k, seed)
    idx = _sorted_routing(rows, experts, seed)
    x = _activations(rows, k, seed)
    tables = op.build_tables(w, s, b)
    tiles = build_tiles(idx, experts, op.DEFAULT_CFG[0] * op.DEFAULT_CFG[2])
    ref = op.reference(x, w, s, b, idx)
    got = op.metal(x, w, s, b, idx, tables, tiles)
    return got, ref


@pytest.mark.exactness
@pytest.mark.parametrize("rows,experts,n,k", [
    (64, 8, 256, 256),
    (128, 8, 256, 512),
    (512, 32, 1280, 2560),
])
def test_relative_rms_within_the_quantisation_floor(seeded, rows, experts, n, k):
    got, ref = _run(rows, experts, n, k)
    assert rel_rms(got, ref) < BAR


@pytest.mark.exactness
def test_error_does_not_grow_with_k(seeded):
    """If the -128*qsum correction were folded into a bf16 scale instead of
    kept in the integer domain, cancellation would make the error grow with the
    reduction length. It must not."""
    errors = [rel_rms(*_run(64, 8, 256, k)) for k in (256, 1024, 4096)]
    assert all(e < BAR for e in errors)
    assert errors[-1] < 3 * errors[0] + 1e-4, errors


@pytest.mark.exactness
def test_large_biases_do_not_degrade_it(seeded):
    """The affine bias term multiplies the exact fp32 group sum of the
    unquantised activations, so it carries no activation-quantisation error."""
    mx.random.seed(3)
    dense = (mx.random.normal((8, 256, 256)) + 20.0).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=op.GROUP, bits=op.BITS)
    mx.eval(w, s, b)
    idx = _sorted_routing(64, 8)
    x = _activations(64, 256)
    tables = op.build_tables(w, s, b)
    tiles = build_tiles(idx, 8, op.DEFAULT_CFG[0] * op.DEFAULT_CFG[2])
    got = op.metal(x, w, s, b, idx, tables, tiles)
    ref = op.reference(x, w, s, b, idx)
    assert rel_rms(got, ref) < BAR


@pytest.mark.exactness
def test_qsum_matches_a_host_nibble_count(seeded):
    """The table builder is the part that must be exact: qsum is an integer in
    [0, 960] and any slip there is a silent bias on every output."""
    w, s, b = _weights(4, 128, 256)
    tables = op.build_tables(w, s, b)
    packed = np.array(w, copy=True).view(np.uint32)   # [E, N, K/8]
    nibbles = np.stack([(packed >> (4 * t)) & 0xF for t in range(8)], axis=-1)
    e, n, words, _ = nibbles.shape
    host = nibbles.reshape(e, n, words // 8, 64).sum(axis=-1)      # [E, N, G]
    got = np.array(tables.qsum, copy=True).astype(np.int64)        # [E, G, N]
    assert np.array_equal(got, host.transpose(0, 2, 1))


def test_supports_requires_the_caller_to_pass_tables(seeded):
    """The opt-in. An op that changes the numbers may not be selected just
    because the shapes happen to fit."""
    w, s, b = _weights(8, 256, 256)
    idx = _sorted_routing(64, 8)
    x = _activations(64, 256)
    tables = op.build_tables(w, s, b)
    assert not op.supports(op.key(x, w, s, b, idx))
    assert op.supports(op.key(x, w, s, b, idx, tables))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    got, ref = _run(2048 * 10 // 16, 32, 1280, 2560)
    assert rel_rms(got, ref) < BAR
