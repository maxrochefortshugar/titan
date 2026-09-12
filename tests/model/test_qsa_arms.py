"""Which QSA arm a forward takes, and whether the arms agree.

Three arms were unwired or misrouted before this round and each one gets the
same treatment here: prove the routing, then prove the numerics.

*Routing.* A predicate decides the arm, and a predicate that quietly stops
firing is the failure mode that costs milliseconds without changing a single
output. So every routing test asserts which function ran, by counting calls,
rather than inferring it from a timing.

*Numerics.* The gathered arms and the dense masked arm compute the same
attention over the same selected columns, in a different reduction order, so
they agree to about one bf16 ULP rather than exactly. That is the tolerance
class Titan already accepts for the batched verify attention, and it is the bar
here: one ULP against the per-row arm, which is the arm a row would have taken
had it been decoded alone.

Everything runs on one small ``Qwen4ExpAttention`` with the checkpoint's
*counts* -- indexer budget 2048, compress ratio 4, so 512 selectable blocks --
and small head dimensions, which is what makes a 12k-token cache cost a couple
of megabytes instead of a couple of gigabytes. The threshold crossings the
tests care about are properties of the counts.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from titan.adapters.mlx import kernels as adapter_kernels
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp import language as q4
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp import qsa_fast
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.config import TextConfig

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the QSA arms need a Metal device"
)

HIDDEN = 128
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 32
INDEXER_HEADS = 2
INDEXER_DIM = 32
RATIO = 4
BUDGET = 2048           # the checkpoint's, so block_topk is 512
BLOCK_TOPK = BUDGET // RATIO


def _config() -> TextConfig:
    return TextConfig(
        model_type="qwen4_exp",
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=512,
        max_position_embeddings=262144,
        hc_count=4,
        hc_lowrank=64,
        full_attention_interval=4,
        ple_layer_ids=[],
        indexer_n_heads=INDEXER_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=INDEXER_DIM,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        eos_token_id=0,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def attention():
    mx.random.seed(0)
    module = q4.Qwen4ExpAttention(_config())
    # bf16 throughout, as the checkpoint runs: the cache below is seeded rather
    # than decoded, so its dtype has to be the one the projections produce.
    module.set_dtype(mx.bfloat16)
    module.eval()
    mx.eval(module.parameters())
    return module


def _warm_cache(length: int, seed: int) -> q4.QSAKVCache:
    """A ``QSAKVCache`` holding *length* tokens, seeded rather than decoded.

    Running a real prefill to 12k tokens through one attention module would
    take longer than every test in this file put together, and the arms below
    read the cache, not the history that produced it.
    """
    mx.random.seed(seed)
    cache = q4.QSAKVCache()
    cache.update_and_fetch(
        mx.random.normal((1, KV_HEADS, length, HEAD_DIM)).astype(mx.bfloat16),
        mx.random.normal((1, KV_HEADS, length, HEAD_DIM)).astype(mx.bfloat16),
    )
    cache.update_indexer(
        mx.random.normal((1, length, INDEXER_DIM)).astype(mx.bfloat16),
        mx.arange(length, dtype=mx.int32)[None],
    )
    mx.eval(cache.keys, cache.values, cache.index_keys)
    return cache


def _rows(width: int, batch: int, seed: int = 11) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((batch, width, HIDDEN)).astype(mx.bfloat16)


def _ulp(got: mx.array, ref: mx.array) -> int:
    """bf16 bit-pattern distance, on the total order over floats."""
    mx.eval(got, ref)
    def ordinals(x):
        bits = np.array(x.astype(mx.bfloat16).view(mx.uint16), copy=True).astype(
            np.int32
        )
        return np.where(bits >= 0x8000, 0x7FFF - bits, bits)
    return int(np.abs(ordinals(got) - ordinals(ref)).max())


class _Counter:
    """Count calls to a module-level function or a class method, and put it
    back afterwards. Which arm ran is the thing under test in half this file,
    and a call count says it without depending on a timing."""

    def __init__(self, module, name):
        self.module, self.name = module, name
        self.original = getattr(module, name)
        self.calls = 0

    def __enter__(self):
        def wrapper(*args, **kwargs):
            self.calls += 1
            return self.original(*args, **kwargs)
        setattr(self.module, self.name, wrapper)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.original)


# ---------------------------------------------------------------------------
# the batched sparse arm
# ---------------------------------------------------------------------------


def _batched(caches, pads):
    joined = caches[0].to_batch([pads[0]])
    for cache, pad in zip(caches[1:], pads[1:]):
        joined.extend(cache.to_batch([pad]))
    return joined


#: The op's two routes. ``metal`` is one padded pass over the batch; the
#: reference runs each row on its own, which is what a row would get if it were
#: not batched at all, so a difference between them is a batching artefact.
def _reference_route():
    from titan.kernels import qsa_gathered_attention as op

    class _Swap:
        def __enter__(self):
            self.saved = op.OP.fast_fn
            op.OP.fast_fn = op.reference
            adapter_kernels.reset()

        def __exit__(self, *exc):
            op.OP.fast_fn = self.saved
            adapter_kernels.reset()

    return _Swap()


def _rrmse(got: mx.array, ref: mx.array) -> float:
    a = np.array(got.astype(mx.float32))
    b = np.array(ref.astype(mx.float32))
    return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))


#: One bf16 ULP, relative. bf16 carries 7 explicit mantissa bits, so adjacent
#: representable values are 2**-8 apart.
BF16_ULP = 2.0 ** -8


@pytest.mark.parametrize("lengths", [
    [12288, 12288],          # equal rows: every phase is zero
    [12288, 12285],          # a phase that is not a multiple of the ratio
    [12288, 9001, 12287],    # three rows, three phases, one of them ragged
    [8190, 8194],            # either side of the 8192 indexer buffer step
    [2052, 2064],            # just past the 2048 block-selection threshold,
                             # where 513 complete blocks meet a budget of 512
    [12288, 2052],           # one row deep in sparse territory, one at the edge
])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6])
def test_the_padded_route_equals_the_per_row_route(attention, lengths, width):
    """The batching itself changes nothing.

    Both routes are run through the whole attention module, so this covers the
    phased bank and the projections as well as the op: the same left padding,
    the same selection, the same gather, differing only in whether the batch
    was processed in one padded pass or one row at a time. The bar is
    bit-identical, which is what the padded route achieves here and what makes
    a later drift obviously a regression rather than noise.

    The crossover is lowered so the short cases take the arm at all; what is
    under test is the arm, not the routing policy, which has its own tests.
    """
    pads = [max(lengths) - n for n in lengths]
    x = _rows(width, len(lengths))
    with adapter_kernels.overridden(
        qsa_gather_min_context=BUDGET, qsa_gather_min_context_verify=BUDGET
    ):
        joined = _batched(
            [_warm_cache(n, seed=100 + i) for i, n in enumerate(lengths)], pads
        )
        with _Counter(q4.Qwen4ExpAttention, "_batched_sparse_qsa") as counter:
            padded = attention(x, mask="causal", cache=joined)
        assert counter.calls == 1, "the batched sparse arm did not run"

        with _reference_route():
            looped = attention(
                x,
                mask="causal",
                cache=_batched(
                    [_warm_cache(n, seed=100 + i) for i, n in enumerate(lengths)],
                    pads,
                ),
            )
    assert _ulp(padded, looped) == 0


@pytest.mark.parametrize("lengths", [[12288, 12288], [12288, 12285]])
@pytest.mark.parametrize("width", [1, 4])
def test_a_batched_row_matches_the_row_decoded_alone(attention, lengths, width):
    """A row must not care what it was batched with.

    Against the *singleton* arm, not against the op's own reference, so this
    also crosses the seam between the two gathered implementations: the
    singleton arm finishes on ``mx.fast.scaled_dot_product_attention`` and the
    batched op on an explicit fp32 reduction, so they differ in the last bits
    the way any two reduction orders do. The bar is relative RMS at the bf16
    epsilon class, and the control in the same units is measured alongside:
    the two arms Titan already ships at width 4, the gathered block arm and the
    dense masked one, sit at the same distance from each other.
    """
    batch = len(lengths)
    pads = [max(lengths) - n for n in lengths]
    alone = [_warm_cache(n, seed=100 + i) for i, n in enumerate(lengths)]
    joined = _batched(
        [_warm_cache(n, seed=100 + i) for i, n in enumerate(lengths)], pads
    )
    x = _rows(width, batch)
    got = attention(x, mask="causal", cache=joined)
    for row in range(batch):
        want = attention(x[row:row + 1], mask="causal", cache=alone[row])
        error = _rrmse(got[row:row + 1], want)
        assert error <= 2 * BF16_ULP, (
            f"row {row} of {lengths} at width {width}: {error:.2e}"
        )


def test_the_shipped_arms_are_the_same_distance_apart(attention):
    """The control for the bar above, so it is not a number pulled from air.

    At width 1 the gathered decode arm and the dense masked arm are
    bit-identical on these shapes. At width 4 they are not, and the distance
    between them is the tolerance class the batched arm is held to.
    """
    x = _rows(4, 1)
    got = attention(
        x, mask="causal", cache=_warm_cache(12288, seed=100), target_verify=True
    )
    with adapter_kernels.overridden(
        qsa_gather_min_context=10 ** 9, qsa_gather_min_context_verify=10 ** 9
    ):
        want = attention(
            x, mask="causal", cache=_warm_cache(12288, seed=100), target_verify=True
        )
    assert _rrmse(got, want) <= 2 * BF16_ULP

    x = _rows(1, 1)
    got = attention(x, mask="causal", cache=_warm_cache(12288, seed=100))
    with adapter_kernels.overridden(
        qsa_gather_min_context=10 ** 9, qsa_gather_min_context_verify=10 ** 9
    ):
        want = attention(x, mask="causal", cache=_warm_cache(12288, seed=100))
    assert _ulp(got, want) == 0


def test_batched_sparse_can_be_switched_off(attention):
    """Off is a supported configuration and it is the dense arm, not an error."""
    caches = [_warm_cache(12288, seed=1), _warm_cache(12285, seed=2)]
    joined = _batched(caches, [0, 3])
    x = _rows(1, 2)
    with adapter_kernels.overridden(qsa_batched_sparse=False):
        with _Counter(q4.Qwen4ExpAttention, "_batched_sparse_qsa") as counter:
            out = attention(x, mask="causal", cache=joined)
    assert counter.calls == 0
    assert out.shape == (2, 1, HIDDEN)


def test_batched_sparse_declines_a_row_below_the_crossover(attention):
    """Every row has to clear the crossover, not just the longest one."""
    caches = [_warm_cache(12288, seed=3), _warm_cache(600, seed=4)]
    joined = _batched(caches, [0, 11688])
    with _Counter(q4.Qwen4ExpAttention, "_batched_sparse_qsa") as counter:
        attention(_rows(1, 2), mask="causal", cache=joined)
    assert counter.calls == 0


def test_the_phased_bank_is_rebuilt_only_at_its_growing_end(attention):
    """Nine steps of decode, and the bank still equals a bank pooled from
    scratch. This is the incremental update's whole contract: a settled slot is
    never recomputed and an unsettled one always is."""
    caches = [_warm_cache(12288, seed=5), _warm_cache(12285, seed=6)]
    joined = _batched(caches, [0, 3])
    for step in range(9):
        attention(_rows(1, 2, seed=40 + step), mask="causal", cache=joined)
    geometry = joined.slot_geometry(RATIO)
    incremental = joined.pooled_index_slots(
        geometry,
        attention.indexer.k_layernorm,
        attention.indexer._apply_rope,
        cache_tag=attention.indexer,
    )
    fresh = adapter_kernels.qsa_pool_slots(
        joined.index_keys,
        joined.index_position_ids,
        geometry,
        0,
        geometry.slots,
        attention.indexer.k_layernorm,
        attention.indexer._apply_rope,
    )
    mx.eval(incremental, fresh)
    assert _ulp(incremental, fresh) == 0


def test_the_batched_indexer_buffer_holds_what_a_concatenate_would(attention):
    """The capacity buffer replaced an ``mx.concatenate`` of the whole history
    per layer per step. It has to hold exactly the same columns."""
    cache = _warm_cache(2048, seed=7).to_batch([0])
    expected = [cache.index_keys]
    for step in range(5):
        mx.random.seed(200 + step)
        keys = mx.random.normal((1, 1, INDEXER_DIM)).astype(mx.bfloat16)
        positions = mx.array([[2048 + step]], dtype=mx.int32)
        cache.update_indexer(keys, positions)
        expected.append(keys)
    want = mx.concatenate(expected, axis=1)
    mx.eval(want, cache.index_keys)
    assert cache.index_offset == 2053
    assert cache.index_keys.shape == want.shape
    assert bool(mx.all(cache.index_keys == want).item())


# ---------------------------------------------------------------------------
# the one-row target-verify arm
# ---------------------------------------------------------------------------


def test_a_one_row_verify_takes_the_sparse_arm(attention):
    """``return_hidden`` sets ``target_verify`` on every layer, and that used to
    push a one-row forward onto the dense masked path, which reads the whole
    cache. Attention has no intermediates to capture, so the flag should not
    reach it."""
    cache = _warm_cache(12288, seed=8)
    with _Counter(q4, "contiguous_causal_gathered_qsa_decode") as counter:
        got = attention(
            _rows(1, 1), mask="causal", cache=cache, target_verify=True
        )
    assert counter.calls == 1
    assert got.shape == (1, 1, HIDDEN)


def test_the_one_row_verify_arm_agrees_with_the_dense_one(attention):
    """Routing only: the two arms attend the same selected columns."""
    x = _rows(1, 1, seed=13)
    sparse_cache = _warm_cache(12288, seed=9)
    dense_cache = _warm_cache(12288, seed=9)
    got = attention(x, mask="causal", cache=sparse_cache, target_verify=True)
    with adapter_kernels.overridden(qsa_sparse_singleton_verify=False):
        with _Counter(q4, "contiguous_causal_gathered_qsa_decode") as counter:
            want = attention(
                x, mask="causal", cache=dense_cache, target_verify=True
            )
        assert counter.calls == 0
    assert _ulp(got, want) <= 1


# ---------------------------------------------------------------------------
# the crossover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length,gathered", [
    (2000, False),      # below the budget: selection has nothing to remove
    (3000, False),      # legal, but the dense arm is still cheaper here
    (6144, False),      # still cheaper dense at one row
    (12288, True),      # past the crossover
])
def test_the_crossover_decides_the_decode_arm(attention, length, gathered):
    cache = _warm_cache(length, seed=20)
    with _Counter(q4, "contiguous_causal_gathered_qsa_decode") as counter:
        attention(_rows(1, 1), mask="causal", cache=cache)
    assert (counter.calls == 1) is gathered


@pytest.mark.parametrize("length,gathered", [
    (3000, False),
    (6144, True),
])
def test_a_verify_block_crosses_over_earlier_than_a_single_row(
    attention, length, gathered
):
    """The dense arm's cost rises with the block width and the gathered arm's
    barely does, so a four-row block is worth gathering thousands of tokens
    before a one-row decode is."""
    cache = _warm_cache(length, seed=23)
    with _Counter(q4, "contiguous_causal_gathered_qsa") as counter:
        attention(_rows(4, 1), mask="causal", cache=cache, target_verify=True)
    assert (counter.calls == 1) is gathered


def test_the_crossover_is_clamped_up_to_the_budget(attention):
    """Setting it below the budget cannot make the arm legal earlier: below the
    budget there are fewer complete blocks than the indexer may select, so
    there is nothing to select."""
    with adapter_kernels.overridden(
        qsa_gather_min_context=0, qsa_gather_min_context_verify=0
    ):
        assert q4._gather_min_context(BUDGET) == BUDGET
        assert q4._gather_min_context(BUDGET, 4) == BUDGET
        cache = _warm_cache(2000, seed=21)
        with _Counter(q4, "contiguous_causal_gathered_qsa_decode") as counter:
            attention(_rows(1, 1), mask="causal", cache=cache)
        assert counter.calls == 0


def test_lowering_the_crossover_takes_the_gathered_arm_earlier(attention):
    """And the two arms still agree, which is what makes the crossover a cost
    decision rather than a numerics one."""
    x = _rows(1, 1, seed=14)
    early = _warm_cache(3000, seed=22)
    late = _warm_cache(3000, seed=22)
    with adapter_kernels.overridden(qsa_gather_min_context=2048):
        with _Counter(q4, "contiguous_causal_gathered_qsa_decode") as counter:
            got = attention(x, mask="causal", cache=early)
        assert counter.calls == 1
    want = attention(x, mask="causal", cache=late)
    assert _ulp(got, want) <= 1


def test_qsa_fast_still_exports_the_pool_helper():
    """``pool_completed_index_keys`` is the singleton bank's pooling and the
    phased bank's contract is stated against it; a rename would silently make
    the two banks different."""
    assert callable(qsa_fast.pool_completed_index_keys)
