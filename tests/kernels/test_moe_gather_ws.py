# SPDX-License-Identifier: MIT
"""moe_gather_ws: bit-identical to mx.gather_qmm at every shape tested.

Bar from engine/patches/round3/gather-ws/REPORT.md section 2: 0 ULP, not 1.
Both paths dequantise to bf16, feed bf16 x bf16 tensor ops with an fp32
accumulator, and sum over the 64-wide affine groups in the same order; only the
tiling differs, and tiling does not change a sum's order here.

Routing is Zipf-like, as it is in the model: a few popular experts and a long
tail, so the ragged tile table is exercised on both sides of TM = 48 rows.
"""

import mlx.core as mx
import numpy as np
import pytest

from exactness import bit_identical, max_ulp
from titan.kernels import moe_gather_ws as op


def _weights(experts, n, k, seed=0):
    mx.random.seed(seed)
    dense = mx.random.normal((experts, n, k)).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=op.GROUP, bits=op.BITS)
    mx.eval(w, s, b)
    return w, s, b


def _sorted_routing(rows, experts, seed=0, zipf=True):
    """Sorted expert ids with a popularity skew, as the router produces."""
    rng = np.random.default_rng(seed)
    if zipf:
        weights = 1.0 / np.arange(1, experts + 1)
        weights /= weights.sum()
        ids = rng.choice(experts, size=rows, p=weights)
    else:
        ids = rng.integers(0, experts, size=rows)
    ids = np.sort(ids).astype(np.uint32)
    out = mx.array(ids)
    mx.eval(out)
    return out


def _activations(rows, k, seed=0):
    mx.random.seed(seed + 7)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    mx.eval(x)
    return x


@pytest.mark.exactness
@pytest.mark.parametrize("rows,experts,n,k", [
    (64, 8, 256, 256),
    (128, 8, 256, 512),
    (512, 32, 1280, 2560),
])
def test_bit_identical(seeded, rows, experts, n, k):
    w, s, b = _weights(experts, n, k)
    idx = _sorted_routing(rows, experts)
    x = _activations(rows, k)
    tiles = op.build_tiles(idx, experts, op.DEFAULT_CFG[0])
    ref = op.reference(x, w, s, b, idx)
    got = op.metal(x, w, s, b, idx, tiles)
    assert bit_identical(got, ref), f"max {max_ulp(got, ref)} ULP"


@pytest.mark.exactness
def test_uniform_routing_too(seeded):
    """Uniform routing puts every expert under TM rows, so every tile is a
    partial one and the row-window clamp on the last tile is exercised."""
    w, s, b = _weights(32, 256, 256)
    idx = _sorted_routing(200, 32, zipf=False)
    x = _activations(200, 256)
    tiles = op.build_tiles(idx, 32, op.DEFAULT_CFG[0])
    assert bit_identical(op.metal(x, w, s, b, idx, tiles),
                         op.reference(x, w, s, b, idx))


@pytest.mark.exactness
def test_one_expert_takes_every_row(seeded):
    """The extreme skew: one expert, many more rows than TM, so the ragged
    table has to emit several tiles for it."""
    w, s, b = _weights(8, 256, 256)
    idx = mx.zeros((256,), dtype=mx.uint32)
    mx.eval(idx)
    x = _activations(256, 256)
    tiles = op.build_tiles(idx, 8, op.DEFAULT_CFG[0])
    assert bit_identical(op.metal(x, w, s, b, idx, tiles),
                         op.reference(x, w, s, b, idx))


@pytest.mark.exactness
def test_tile_table_is_reusable_across_projections(seeded):
    """Both projections of one SwitchGLU share rhs_indices, which is why the
    table is the caller's object and not a hidden memo."""
    idx = _sorted_routing(128, 8)
    x = _activations(128, 256)
    tiles = op.build_tiles(idx, 8, op.DEFAULT_CFG[0])
    for seed in (0, 1):
        w, s, b = _weights(8, 256, 256, seed=seed)
        assert bit_identical(op.metal(x, w, s, b, idx, tiles),
                             op.reference(x, w, s, b, idx))


def test_supports_gates_on_dtype_bits_and_row_count(seeded):
    w, s, b = _weights(8, 256, 256)
    idx = _sorted_routing(64, 8)
    x = _activations(64, 256)
    assert op.supports(op.key(x, w, s, b, idx))
    assert not op.supports(op.key(x, w, s, b, idx, bits=8))
    # fewer rows than one tile: the kernel's row window cannot be placed
    small = _activations(16, 256)
    assert not op.supports(op.key(small, w, s, b, _sorted_routing(16, 8)))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """A real routed-expert GEMM: 2048 tokens at top-10 over 32 experts, into
    the 1280-wide gate_up half of a 2560-wide block."""
    experts, n, k = 32, 1280, 2560
    rows = 2048 * 10 // 16     # a slice of the chunk, to stay well under 1 GB
    w, s, b = _weights(experts, n, k)
    idx = _sorted_routing(rows, experts)
    x = _activations(rows, k)
    tiles = op.build_tiles(idx, experts, op.DEFAULT_CFG[0])
    assert bit_identical(op.metal(x, w, s, b, idx, tiles),
                         op.reference(x, w, s, b, idx))
