# SPDX-License-Identifier: MIT
"""hc_prefill: within 2 bf16 ULP of the plain chain downstream of the norm.

Bar from engine/patches/round3/hc-fuse/REPORT.md, adjusted for the reference
this library actually has. The overlay measured 0 ULP against oMLX's own fused
tail, another JIT kernel. Against a plain-MLX chain the residue is mx.sigmoid,
which is not bit-reproducible from mx.fast.metal_kernel on this build: even a
literal transcription of MLX's own Sigmoid expression lands 1 ULP away on about
0.08% of inputs. So these tests assert 2 ULP with the overwhelming majority
bit-identical, and check the pieces that ARE exact (the tail's bf16 mean) and
the pieces that deliberately are not.

Held the norm fixed, everything downstream is compared directly; end to end the
block also inherits the norm's <= 1 ULP, which the stream mean amplifies where
the four terms cancel, so that test bounds the mean and the relative error.

fp32_mean and the merged down|inject bank both move results further, and are
asserted to, so neither can quietly become the default.
"""

import mlx.core as mx
import numpy as np
import pytest

from exactness import bit_identical, rrmse, ulp_distance
from titan.kernels import grouped_rmsnorm_bf16 as gnorm
from titan.kernels import hc_prefill as op

HC, HIDDEN, LOWRANK = 4, 256, 64
WIDTH = HC * HIDDEN


def _quant(out_features, in_features, bits, seed):
    mx.random.seed(seed)
    dense = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=64, bits=bits)
    mx.eval(w, s, b)
    return op.QuantLinear(w, s, b, bits=bits, group_size=64)


def _weights(bits=4, inject=True, hidden=HIDDEN, lowrank=LOWRANK, seed=0):
    width = HC * hidden
    norm_w = (0.1 * mx.random.normal((width,))).astype(mx.bfloat16)
    mx.eval(norm_w)
    return op.HCWeights(
        norm_weight=norm_w,
        down=_quant(lowrank, width, bits, seed + 1),
        up=_quant(width, lowrank, bits, seed + 2),
        inject=_quant(HC, width, bits, seed + 3) if inject else None,
        hc_count=HC,
        hidden_size=hidden,
        eps=1e-6,
    )


def _stream(rows, hidden=HIDDEN, scale=1.0, seed=99):
    mx.random.seed(seed)
    x = (scale * mx.random.normal((1, rows, HC * hidden))).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _plain_with_kernel_norm(x, w, **kw):
    """The plain chain, given the same normed tensor the fused block computes.
    This is the comparison the report's zero-ULP column is against."""
    rows = x.shape[0] * x.shape[1]
    normed = gnorm.metal(x.reshape(rows, w.width), w.norm_weight,
                         groups=w.hc_count, eps=w.eps)
    mixed, injection = op.chain_from_normed(normed, w, **kw)
    mixed = mixed.reshape(x.shape[0], x.shape[1], w.hidden_size)
    if injection is None:
        return mixed
    return mixed, x, injection.reshape(x.shape[0], x.shape[1], w.hc_count)


@pytest.mark.exactness
@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("bits", [4, 5, 8])
def test_within_two_ulp_downstream_of_the_norm(seeded, rows, bits):
    w = _weights(bits=bits)
    x = _stream(rows)
    mixed, _, inj = op.metal(x, w)
    ref_mixed, _, ref_inj = _plain_with_kernel_norm(x, w)
    for got, ref, name in ((mixed, ref_mixed, "mixed"), (inj, ref_inj, "inject")):
        d = ulp_distance(got, ref)
        assert (d == 0).mean() > 0.99, f"{name}: only {(d == 0).mean():.4f} exact"
        assert d.mean() < 0.02, f"{name}: mean {d.mean():.4f} ULP"
        assert rrmse(got, ref) < 1e-4, f"{name}: rrmse {rrmse(got, ref):.2e}"


@pytest.mark.exactness
def test_mixer_without_injection(seeded):
    w = _weights(inject=False)
    x = _stream(16)
    got, ref = op.metal(x, w), _plain_with_kernel_norm(x, w)
    d = ulp_distance(got, ref)
    assert (d == 0).mean() > 0.99 and rrmse(got, ref) < 1e-4


@pytest.mark.exactness
@pytest.mark.parametrize("scale", [0.125, 8.0])
def test_stream_scale(seeded, scale):
    """The block is scale covariant, so eight times up or down must not widen
    the gap: if it does, an accumulator is the problem."""
    w = _weights()
    x = _stream(16, scale=scale)
    mixed, _, _ = op.metal(x, w)
    ref_mixed, _, _ = _plain_with_kernel_norm(x, w)
    d = ulp_distance(mixed, ref_mixed)
    assert (d == 0).mean() > 0.99 and rrmse(mixed, ref_mixed) < 1e-4


@pytest.mark.exactness
def test_stream_mean_is_a_sequential_bf16_sum(seeded):
    """MLX's mean over a bf16 axis carries a bf16 accumulator: it is a
    sequential bf16 sum and one divide, which is what the tail kernel does. If
    this ever stops holding, the tail's ACC32 = 0 path is no longer the
    matching one and the fp32 accumulator becomes the honest default."""
    rows = 32
    g3 = mx.random.normal((rows, HC, HIDDEN)).astype(mx.bfloat16)
    mx.eval(g3)
    acc = g3[:, 0]
    for i in range(1, HC):
        acc = (acc + g3[:, i]).astype(mx.bfloat16)
    sequential = (acc / HC).astype(mx.bfloat16)
    assert bit_identical(mx.mean(g3, axis=1).astype(mx.bfloat16), sequential)


@pytest.mark.exactness
def test_end_to_end_inherits_the_norms_deviation(seeded):
    """Against the canonical fp32 norm the block also carries that norm's own
    1 ULP, and the stream mean amplifies it wherever the four terms cancel. So
    the max ULP is meaningless here and the median and the relative error are
    what to hold: most elements are still exact, and the block as a whole moves
    by a fraction of a percent."""
    w = _weights(hidden=2560, lowrank=320, seed=20)
    x = _stream(32, hidden=2560)
    mixed, _, inj = op.metal(x, w)
    ref_mixed, _, ref_inj = op.reference(x, w)
    d = ulp_distance(mixed, ref_mixed)
    assert np.median(d) == 0
    assert (d == 0).mean() > 0.4, f"only {(d == 0).mean():.3f} exact"
    assert rrmse(mixed, ref_mixed) < 5e-2
    assert rrmse(inj, ref_inj) < 5e-2


@pytest.mark.exactness
def test_fp32_mean_moves_results_and_is_off_by_default(seeded):
    """More accurate, and further than 1 ULP from the plain bf16 reduce
    wherever the four stream terms cancel. That is why it is not the default."""
    w = _weights()
    x = _stream(64)
    default, _, _ = op.metal(x, w)
    acc32, _, _ = op.metal(x, w, fp32_mean=True)
    assert not bit_identical(acc32, default)
    assert int(ulp_distance(acc32, default).max()) > 2


@pytest.mark.exactness
def test_merged_bank_keeps_mixed(seeded):
    """The low-rank rows come out bit identical whichever way they are
    projected, so ``mixed`` is unaffected by the merge. The injection rows are
    the ones at risk: in the bank they are four rows of a much wider matmul,
    a different reduction order over K, measured at 2 to 4 ULP at the real
    K = 10240. At this test's K = 1024 the orders can still coincide, so the
    injection is only bounded here, not asserted to differ."""
    w0 = _weights()
    w = op.HCWeights(
        norm_weight=w0.norm_weight, down=w0.down, up=w0.up, inject=w0.inject,
        hc_count=HC, hidden_size=HIDDEN, eps=w0.eps,
        bank=op.build_bank(w0.down, w0.inject),
    )
    x = _stream(64)
    split_mixed, _, split_inj = op.metal(x, w)
    bank_mixed, _, bank_inj = op.metal(x, w, use_bank=True)
    assert bit_identical(bank_mixed, split_mixed)
    assert int(ulp_distance(bank_inj, split_inj).max()) <= 4


def test_supports_rejects_fp32_streams(seeded):
    w = _weights()
    x = _stream(8)
    assert op.supports(op.key(x, w))
    assert not op.supports(op.key(x.astype(mx.float32), w))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_shape(seeded):
    """The real prefill block: [1, 2048, 4 x 2560] with a 320-wide low rank."""
    w = _weights(bits=5, hidden=2560, lowrank=320, seed=10)
    x = _stream(2048, hidden=2560)
    mixed, _, inj = op.metal(x, w)
    ref_mixed, _, ref_inj = _plain_with_kernel_norm(x, w)
    for got, ref in ((mixed, ref_mixed), (inj, ref_inj)):
        d = ulp_distance(got, ref)
        assert (d == 0).mean() > 0.999, f"only {(d == 0).mean():.5f} exact"
        assert rrmse(got, ref) < 1e-4
