"""The compiled decode path: does it compute the same thing, and does it hold.

Everything here runs on the synthetic Qwen4-Exp that
``bench/decode/host_overhead.py`` builds -- four decoder layers, hidden 128,
eight experts, 4-bit affine quantisation, a few megabytes of weights -- so the
suite is safe beside a workbench holding the real checkpoint. What the
synthetic model does not cover is listed in ``docs/architecture/FORWARD.md``.

Four kinds of test.

*Numerics.* Every compiled block is a reformulation of an eager one, so each is
run against its eager twin and held to a bar. Most of them are bit-identical
and are asserted as such. The ones that are not -- the ones that moved an
accumulation or a mask -- are held to one bf16 ULP, which is the tolerance
class Titan already accepts for the batched verify arms.

*State.* The whole point of the reformulation is that state goes in and comes
out instead of being mutated, so a test rolls a state back and checks the next
step is the step that would have happened.

*Bridging.* The compiled state has to round-trip through the vendored caches or
nothing downstream of the model keeps working, so a test takes it out and puts
it back.

*The shapeless boundary.* The module claims particular MLX errors as the reason
particular blocks are shape-specialised. A claim about an error message rots
unless something checks it, so a test checks it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from titan.adapters.mlx import compiled as C

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the compiled path needs a Metal device"
)

# bfloat16 carries eight significand bits, so one ULP is 2**-8 relative. Every
# bound below is stated against this rather than a number pulled out of the air.
BF16_ULP = 2.0**-8


def _load_bench():
    """Import the bench by path: ``bench`` is a script directory, not a package."""
    path = Path(__file__).resolve().parents[2] / "bench" / "decode" / "host_overhead.py"
    spec = importlib.util.spec_from_file_location("titan_bench_host_overhead", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return bench.SyntheticSpec(num_hidden_layers=4)


@pytest.fixture(scope="module")
def model(spec):
    return bench.build_synthetic(spec)


@pytest.fixture(scope="module")
def language_model(model):
    return model.language_model


def relative(actual: mx.array, expected: mx.array) -> float:
    """Largest elementwise difference, as a fraction of the reference's scale."""
    a = actual.astype(mx.float32)
    b = expected.astype(mx.float32)
    scale = float(mx.max(mx.abs(b)))
    return float(mx.max(mx.abs(a - b))) / max(scale, 1e-9)


def identical(actual: mx.array, expected: mx.array) -> bool:
    return bool(mx.all(actual == expected))


def prefill(language_model, length: int, vocab: int):
    """Run the eager forward to *length* tokens and hand back its caches."""
    cache = language_model.make_cache()
    ids = mx.array([[(i % (vocab - 2)) + 1 for i in range(length)]], dtype=mx.int64)
    language_model(ids, cache=cache, skip_logits=True)
    mx.eval([a for a in _cache_arrays(cache)])
    return cache


def _cache_arrays(caches):
    arrays = []
    for cache in caches:
        held = getattr(cache, "state", None)
        for value in held if isinstance(held, (list, tuple)) else (held,):
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def clone_caches(language_model, caches):
    """A deep copy of a cache list, so the two arms cannot see each other."""
    copy = language_model.make_cache()
    for fresh, held in zip(copy, caches):
        if getattr(held, "keys", None) is not None:
            fresh.keys = mx.array(held.keys)
            fresh.values = mx.array(held.values)
            fresh.offset = held.offset
            if getattr(held, "index_keys", None) is not None:
                fresh.index_keys = mx.array(held.index_keys)
                fresh.index_position_ids = mx.array(held.index_position_ids)
        elif hasattr(held, "cache"):
            fresh.state = [None if v is None else mx.array(v) for v in held.state]
    return copy


def tokens(width: int, vocab: int, start: int = 11) -> mx.array:
    return mx.array(
        [[(start + i) % (vocab - 2) + 1 for i in range(width)]], dtype=mx.int64
    )


# ---------------------------------------------------------------------------
# (a) the Gated DeltaNet decode step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 2, 4, 6])
def test_gdn_step_matches_eager(model, spec, width):
    """The GDN block, and its two states, against the module it replaces.

    The conv is the one place the two arms differ on purpose: the compiled
    block always accumulates the taps in fp32, which is what the vendored
    *decode* arm does, while the vendored *verify* arm calls ``nn.Conv1d`` and
    accumulates in bf16. At kernel size four the two are well inside a ULP and
    the fp32 one is the more accurate.
    """
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache

    gdn = model.language_model.model.layers[0].linear_attn
    _specs, arrays = C.gdn_weights(gdn)
    C.eval_weights(arrays)
    step = C.build_gdn_step(gdn)

    x = mx.random.normal((1, width, spec.hidden_size)).astype(mx.bfloat16)
    conv_state = mx.random.normal(
        (1, gdn.conv_kernel_size - 1, gdn.conv_dim)
    ).astype(mx.bfloat16)
    ssm_state = mx.random.normal(
        (1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim)
    ).astype(mx.float32)
    mx.eval(x, conv_state, ssm_state)

    y, next_conv, next_ssm = step(x, conv_state, ssm_state, arrays)

    cache = ArraysCache(size=2)
    cache[0] = mx.array(conv_state)
    cache[1] = mx.array(ssm_state)
    expected = gdn(x, mask=None, cache=cache)
    mx.eval(y, next_conv, next_ssm, expected)

    assert relative(y, expected) < BF16_ULP
    assert identical(next_conv, cache[0])
    assert relative(next_ssm, cache[1]) < BF16_ULP


def test_gdn_step_has_no_eval_inside(model):
    """The blocker FORWARD.md named first: ``mx.eval`` in the middle of a step.

    ``_causal_conv1d_decode`` evaluated its transposed conv weight on the first
    call. The transpose is a function of the weights alone, so
    :func:`conv_decode_weight` hoists it to build time and the step itself
    evaluates nothing.
    """
    gdn = model.language_model.model.layers[0].linear_attn
    _specs, arrays = C.gdn_weights(gdn)
    C.eval_weights(arrays)
    step = C.build_gdn_step(gdn, compiled=False)

    x = mx.random.normal((1, 1, gdn.hidden_size)).astype(mx.bfloat16)
    conv_state = mx.zeros((1, gdn.conv_kernel_size - 1, gdn.conv_dim), mx.bfloat16)
    ssm_state = mx.zeros(
        (1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), mx.float32
    )
    mx.eval(x)

    calls = {"eval": 0, "async_eval": 0}
    real_eval, real_async = mx.eval, mx.async_eval

    def counted_eval(*args, **kwargs):
        calls["eval"] += 1
        return real_eval(*args, **kwargs)

    def counted_async(*args, **kwargs):
        calls["async_eval"] += 1
        return real_async(*args, **kwargs)

    mx.eval, mx.async_eval = counted_eval, counted_async
    try:
        step(x, conv_state, ssm_state, arrays)
    finally:
        mx.eval, mx.async_eval = real_eval, real_async

    assert calls == {"eval": 0, "async_eval": 0}


# ---------------------------------------------------------------------------
# (b) the MoE block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 7, 8])
def test_moe_step_is_bit_identical(model, spec, width):
    """Routing, the expert gathers and the shared expert, widths 1 to 8.

    Bit-identical rather than close: the compiled block is the same ops in the
    same order. What it removes is the per-call registry lookup and the Python
    around ``SwitchGLU``, neither of which is arithmetic.
    """
    moe = model.language_model.model.layers[0].mlp
    _specs, arrays = C.moe_weights(moe)
    C.eval_weights(arrays)
    step = C.build_moe_step(moe)

    x = mx.random.normal((1, width, spec.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    actual = step(x, arrays)
    expected = moe(x, target_verify=width > 1)
    mx.eval(actual, expected)
    assert identical(actual, expected)


# ---------------------------------------------------------------------------
# (c) the hyper-connection blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 4, 8])
def test_gated_residual_is_bit_identical(model, spec, width):
    """One shapeless trace serves every width, and it changes nothing.

    The vendored module has a compiled arm already, but it is gated off
    whenever Lightning MTP is enabled, which is production, and gated to width
    one and bfloat16. This block carries neither gate: it is the compiled
    default including when MTP is on.
    """
    layer = model.language_model.model.layers[0]
    hyper = layer.attn_hyper_connection
    _specs, arrays = C.gated_residual_weights(hyper)
    C.eval_weights(arrays)
    step = C.build_gated_residual(hyper)

    hidden = mx.random.normal(
        (1, width, spec.hidden_size * spec.hc_count)
    ).astype(mx.bfloat16)
    mx.eval(hidden)
    actual = step(hidden, arrays)
    expected = hyper._forward(hidden, target_verify=width > 1)
    mx.eval(actual, expected)
    for got, want in zip(actual, expected):
        assert identical(got, want)


def test_gated_residual_takes_one_trace_for_every_width(model, spec):
    """The same compiled callable, called at eight widths, without retracing.

    There is no MLX API that reports the trace count, so this asserts the
    property that would break if the trace were width-specialised in the way
    the shape-specialised blocks are: the *first* call at a new width costs
    what a cached call costs, not what a compile costs.
    """
    hyper = model.language_model.model.layers[0].attn_hyper_connection
    _specs, arrays = C.gated_residual_weights(hyper)
    C.eval_weights(arrays)
    step = C.build_gated_residual(hyper)

    for width in range(1, 9):
        hidden = mx.random.normal(
            (1, width, spec.hidden_size * spec.hc_count)
        ).astype(mx.bfloat16)
        mx.eval(hidden)
        out = step(hidden, arrays)
        mx.eval(out[0])
        assert out[0].shape == (1, width, spec.hidden_size)


def test_final_mixer_is_bit_identical(model, spec):
    """``hyper_connection_mixer`` has no injection branch and returns one array."""
    mixer = model.language_model.model.hyper_connection_mixer
    _specs, arrays = C.gated_residual_weights(mixer)
    C.eval_weights(arrays)
    step = C.build_gated_residual(mixer)

    hidden = mx.random.normal((1, 1, spec.hidden_size * spec.hc_count)).astype(
        mx.bfloat16
    )
    mx.eval(hidden)
    actual = step(hidden, arrays)
    expected = mixer._forward(hidden)
    mx.eval(actual, expected)
    assert identical(actual, expected)


def test_hyper_inject_is_bit_identical(model, spec):
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.language import (
        _hyper_inject_ops,
    )

    hidden = mx.random.normal((1, 4, spec.hidden_size * spec.hc_count)).astype(
        mx.bfloat16
    )
    branch = mx.random.normal((1, 4, spec.hidden_size)).astype(mx.bfloat16)
    weights = mx.random.normal((1, 4, spec.hc_count)).astype(mx.bfloat16)
    mx.eval(hidden, branch, weights)
    actual = C.hyper_inject(hidden, branch, weights)
    expected = _hyper_inject_ops(hidden, branch, weights)
    mx.eval(actual, expected)
    assert identical(actual, expected)


# ---------------------------------------------------------------------------
# (d) the dense attention step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 4])
def test_attention_step_on_a_preallocated_buffer(model, spec, language_model, width):
    """The KV cache as two fixed-capacity arrays and an ``int32`` offset array.

    Nothing calls ``int()`` on the offset, nothing reallocates, and the step
    attends over the whole capacity with the causal comparison switching off
    the columns past the write. A masked column contributes an exact zero to
    the softmax, so this is the same reduction the sliced eager arm performs.
    """
    context = 600
    caches = prefill(language_model, context, spec.vocab_size)
    attn = language_model.model.layers[3].self_attn
    _specs, arrays = C.attention_weights(attn)
    C.eval_weights(arrays)
    step = C.build_attention_step(attn)

    kv = caches[3]
    capacity = C.capacity_for(context + width)
    keys = kv.keys[:, :, :context, :]
    values = kv.values[:, :, :context, :]
    k_buf = mx.slice_update(
        C.grow_kv(None, capacity, keys), keys, mx.array([0], mx.int32), axes=(2,)
    )
    v_buf = mx.slice_update(
        C.grow_kv(None, capacity, values), values, mx.array([0], mx.int32), axes=(2,)
    )
    offset = mx.array(context, mx.int32)
    mx.eval(k_buf, v_buf, offset)

    x = mx.random.normal((1, width, spec.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    y, next_k, next_v, next_offset, index_keys = step(x, k_buf, v_buf, offset, arrays)

    clone = clone_caches(language_model, caches)
    expected = attn(x, mask="causal", cache=clone[3], target_verify=width > 1)
    mx.eval(y, next_k, next_v, next_offset, expected)

    assert relative(y, expected) < BF16_ULP
    assert int(next_offset) == context + width
    assert identical(
        next_k[:, :, : context + width, :], clone[3].keys[:, :, : context + width, :]
    )
    assert index_keys is not None
    assert identical(index_keys, clone[3].index_keys[:, context:, :])


def test_capacity_growth_points():
    """Stepping by 256 to 2048 and doubling after: bounded traces, bounded waste."""
    assert C.capacity_for(0) == 256
    assert C.capacity_for(1) == 256
    assert C.capacity_for(256) == 256
    assert C.capacity_for(257) == 512
    assert C.capacity_for(2048) == 2048
    assert C.capacity_for(2049) == 4096
    assert C.capacity_for(64000) == 65536
    for tokens_held in (1, 300, 2000, 5000, 64000):
        assert C.capacity_for(tokens_held) >= tokens_held
        assert C.capacity_for(tokens_held) < 2 * max(tokens_held, 256)


def test_grow_kv_keeps_the_rows_it_held():
    buffer = mx.random.normal((1, 2, 256, 32)).astype(mx.bfloat16)
    mx.eval(buffer)
    grown = C.grow_kv(buffer, 512, buffer)
    mx.eval(grown)
    assert grown.shape == (1, 2, 512, 32)
    assert identical(grown[:, :, :256, :], buffer)
    assert bool(mx.all(grown[:, :, 256:, :] == 0))


# ---------------------------------------------------------------------------
# the whole-model decode step
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_model(model):
    built = C.build_model(model)
    C.eval_weights(built.weights)
    return built


@pytest.mark.parametrize("context", [250, 260, 520, 2040])
@pytest.mark.parametrize("width", [1, 3, 6])
def test_whole_model_step_matches_eager(
    model, spec, language_model, compiled_model, context, width
):
    """Logits and the recurrent state, across the buffer growth points.

    The contexts straddle 256, 512 and 2048, which are the points
    :func:`capacity_for` grows at and therefore the points the compiled
    attention step retraces at. Crossing one must change the timing and nothing
    else.
    """
    caches = prefill(language_model, context, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])

    ids = tokens(width, spec.vocab_size)
    logits, _hidden, next_state = compiled_model(ids, state)

    clone = clone_caches(language_model, caches)
    expected = language_model(ids, cache=clone, return_hidden=True)
    mx.eval(logits, expected.logits)

    assert relative(logits, expected.logits) < BF16_ULP

    for index, entry in enumerate(next_state.layers):
        if entry.kind != C.LINEAR:
            continue
        for slot, value in enumerate(entry.arrays):
            mx.eval(value, clone[index].state[slot])
            assert relative(value, clone[index].state[slot]) < BF16_ULP


def test_whole_model_step_agrees_on_the_argmax(
    spec, language_model, compiled_model
):
    """The property a verify block actually depends on, at every width 1..6."""
    caches = prefill(language_model, 600, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    for width in range(1, 7):
        ids = tokens(width, spec.vocab_size, start=3 * width)
        logits, _hidden, _next = compiled_model(ids, state)
        clone = clone_caches(language_model, caches)
        expected = language_model(ids, cache=clone, return_hidden=True)
        mx.eval(logits, expected.logits)
        assert identical(
            mx.argmax(logits, axis=-1), mx.argmax(expected.logits, axis=-1)
        )


def test_the_compiled_step_submits_nothing_of_its_own(
    spec, language_model, compiled_model
):
    """One graph, one submission, against one ``async_eval`` per layer eagerly.

    FORWARD.md section 5 measured the per-layer ``async_eval`` at 0.20 ms of
    host time each, 24 of them on the synthetic model and 48 on the checkpoint,
    and called it the largest single item. The compiled step has none: the
    whole model is one graph, so there is nothing to dispatch until the caller
    asks for the result. Counted rather than timed, because the machine is
    shared.
    """
    caches = prefill(language_model, 300, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    ids = tokens(1, spec.vocab_size)

    compiled_model(ids, state)  # warm the trace

    counts = {"eager": 0, "compiled": 0}
    real_async = mx.async_eval
    key = "compiled"

    def counted(*args, **kwargs):
        counts[key] += 1
        return real_async(*args, **kwargs)

    mx.async_eval = counted
    try:
        compiled_model(ids, state)
        key = "eager"
        language_model(ids, cache=clone_caches(language_model, caches))
    finally:
        mx.async_eval = real_async

    assert counts["compiled"] == 0
    assert counts["eager"] == len(language_model.model.layers)


# ---------------------------------------------------------------------------
# the state contract
# ---------------------------------------------------------------------------


def test_truncate_state_restores_the_step_that_would_have_happened(
    spec, language_model, compiled_model
):
    """Roll a verify block back and the next step is the step from before it.

    This is the property ``ModelState.truncate`` promises and the reason the
    compiled step returns its state instead of mutating one: rolling back is
    reassigning the arrays the caller already holds.
    """
    caches = prefill(language_model, 600, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    snapshot = state.recurrent_arrays()

    baseline, _hidden, _next = compiled_model(tokens(1, spec.vocab_size), state)
    mx.eval(baseline)

    _logits, _hidden, after_block = compiled_model(tokens(4, spec.vocab_size), state)
    mx.eval(_logits)
    rolled = C.truncate_state(after_block, state.length, snapshot)
    assert rolled.length == state.length

    again, _hidden, _next = compiled_model(tokens(1, spec.vocab_size), rolled)
    mx.eval(again)
    assert identical(again, baseline)


def test_rollback_speculative_state_keeps_the_accepted_prefix(
    spec, language_model, compiled_model
):
    """``accepted`` of a block of four survives; the rest is the offset moving."""
    caches = prefill(language_model, 600, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    snapshot = state.recurrent_arrays()

    _logits, _hidden, after_block = compiled_model(tokens(4, spec.vocab_size), state)
    mx.eval(_logits)
    assert after_block.length == state.length + 4

    rolled = C.rollback_speculative_state(after_block, 1, 4, snapshot)
    assert rolled.length == state.length + 2
    for entry in rolled.layers:
        if entry.kind == C.ATTENTION:
            assert int(entry.offset) == state.length + 2

    with pytest.raises(ValueError):
        C.rollback_speculative_state(after_block, 4, 4, snapshot)


def test_truncate_without_a_snapshot_is_an_error(spec, language_model, compiled_model):
    """The recurrent half cannot be rewound, and the state says so."""
    caches = prefill(language_model, 300, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    _logits, _hidden, after = compiled_model(tokens(2, spec.vocab_size), state)
    mx.eval(_logits)
    with pytest.raises(ValueError, match="no recurrent snapshot"):
        C.truncate_state(after, state.length, None)
    with pytest.raises(ValueError, match="cannot grow"):
        C.truncate_state(after, after.length + 1, {})


def test_state_round_trips_through_the_vendored_caches(
    spec, language_model, compiled_model
):
    """Out of the caches, one compiled step, back into the caches, still eager-able.

    Everything downstream of the model -- ``ModelState.truncate``,
    ``stage_snapshot``, the prefix-cache codec -- works on these caches, so the
    compiled path is only usable if what it computes lands back in them.
    """
    context = 400
    caches = prefill(language_model, context, spec.vocab_size)
    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    assert state.length == context

    ids = tokens(2, spec.vocab_size)
    logits, _hidden, next_state = compiled_model(ids, state)
    mx.eval(logits)
    C.write_layer_state(next_state, caches)

    for cache in caches:
        if getattr(cache, "keys", None) is not None:
            assert int(cache.offset) == context + 2
            assert cache.index_keys.shape[1] == context + 2

    # The eager forward carries on from what the compiled step left behind.
    following = language_model(tokens(1, spec.vocab_size, start=41), cache=caches)
    mx.eval(following.logits)
    assert following.logits.shape[-1] == spec.vocab_size


def test_model_state_truncate_still_works_after_a_compiled_step(
    spec, language_model, compiled_model
):
    """The bridge back is complete enough for ``ModelState`` to own the result."""
    from titan.adapters.mlx.state import ModelState

    context = 300
    caches = prefill(language_model, context, spec.vocab_size)
    holder = ModelState(layers=caches, length=context)
    holder.stage_snapshot(context, pinned=True)

    state = C.read_layer_state(caches)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    _logits, _hidden, after = compiled_model(tokens(3, spec.vocab_size), state)
    mx.eval(_logits)
    C.write_layer_state(after, caches)
    holder.length = context + 3

    holder.truncate(context)
    assert holder.length == context
    for cache in caches:
        if getattr(cache, "keys", None) is not None:
            assert int(cache.offset) == context


def test_grow_state_grows_only_at_the_growth_points(model):
    state = C.new_state(model, capacity=256)
    assert state.capacity == 256
    assert C.grow_state(state, 200).capacity == 256
    assert C.grow_state(state, 256).capacity == 256
    grown = C.grow_state(state, 300)
    assert grown.capacity == 512
    for entry in grown.layers:
        if entry.kind == C.ATTENTION:
            assert entry.arrays[0].shape[2] == 512
            assert entry.arrays[2].shape[1] == 512


# ---------------------------------------------------------------------------
# the QSA seam
# ---------------------------------------------------------------------------


def test_sparse_layer_needs_the_split_and_then_matches(spec, language_model):
    """Above the indexer budget the single graph is wrong and the split is right.

    The selection depends on the layer's own mixed hyper-connection output, so
    a caller above the budget cuts the graph where the selection happens,
    computes it on the eager path and passes it in padded to capacity. The
    dense single graph is kept in the same test because the gap is the point:
    it is not rounding, it is a different set of keys.
    """
    context = 2400
    caches = prefill(language_model, context, spec.vocab_size)
    layer = language_model.model.layers[3]
    attn = layer.self_attn
    assert context // attn.indexer.compress_ratio > attn.indexer.block_topk

    weights = C.layer_weights(layer)
    C.eval_weights(weights)
    mix_step, rest_step = C.build_layer_split(layer)
    whole = C.build_layer(layer)

    capacity = C.capacity_for(context + 8)
    state = C.read_layer_state(caches, capacity=capacity)
    k_buf, v_buf, index_buf, offset = state.layers[3].arrays
    mx.eval(k_buf, v_buf, index_buf, offset)

    hidden = mx.random.normal((1, 1, spec.hidden_size * spec.hc_count)).astype(
        mx.bfloat16
    )
    mx.eval(hidden)

    eager_caches = clone_caches(language_model, caches)
    expected = layer(
        hidden,
        tokens(1, spec.vocab_size),
        mask="causal",
        cache=eager_caches[3],
        position_ids=None,
    )

    selection_caches = clone_caches(language_model, caches)
    mixed, hyper_input, injection = mix_step(hidden, weights)
    sparse = attn.indexer(mixed, selection_caches[3], None)
    assert sparse is not None
    padded = C.pad_sparse_mask(sparse, capacity)
    actual, *_ = rest_step(
        mixed, hyper_input, injection, k_buf, v_buf, index_buf, offset, weights, padded
    )
    dense, *_ = whole.step(hidden, k_buf, v_buf, index_buf, offset, weights)
    mx.eval(expected, actual, dense)

    assert relative(actual, expected) < BF16_ULP
    assert relative(dense, expected) > 100 * relative(actual, expected)


def test_pad_sparse_mask_pads_with_false():
    mask = mx.ones((1, 1, 2, 10), dtype=mx.bool_)
    padded = C.pad_sparse_mask(mask, 16)
    assert padded.shape == (1, 1, 2, 16)
    assert bool(mx.all(padded[..., :10]))
    assert not bool(mx.any(padded[..., 10:]))
    with pytest.raises(ValueError):
        C.pad_sparse_mask(mask, 8)


# ---------------------------------------------------------------------------
# the shapeless boundary
# ---------------------------------------------------------------------------


def _shapeless_error(body, args) -> str:
    compiled = mx.compile(body, shapeless=True)
    with pytest.raises(ValueError) as caught:
        mx.eval(compiled(*args))
    return str(caught.value)


def test_the_named_shapeless_errors_are_the_errors_mlx_raises(model, spec):
    """The module docstring names three errors as the reason for specialising.

    A claim about an error message is a claim about a version of MLX, so it is
    asserted rather than remembered.
    """
    from titan.adapters.mlx.vendor.mlx_lm.models.gated_delta import gated_delta_kernel

    layers = model.language_model.model.layers

    # CustomKernel: the Gated DeltaNet recurrence.
    batch, heads_k, dim_k = 1, 4, 32
    heads_v, dim_v = 8, 32
    message = _shapeless_error(
        gated_delta_kernel,
        (
            mx.zeros((batch, 1, heads_k, dim_k)),
            mx.zeros((batch, 1, heads_k, dim_k)),
            mx.zeros((batch, 1, heads_v, dim_v)),
            mx.zeros((batch, 1, heads_v)),
            mx.zeros((batch, 1, heads_v)),
            mx.zeros((batch, heads_v, dim_v, dim_k)),
        ),
    )
    assert "CustomKernel cannot infer output shapes" in message

    # Slice: argpartition followed by a negative-index slice, the MoE top-k.
    def top_k(scores):
        return mx.take_along_axis(
            scores, mx.argpartition(scores, kth=-2, axis=-1)[..., -2:], axis=-1
        )

    assert "Slice cannot infer output shapes" in _shapeless_error(
        top_k, (mx.zeros((1, 1, 8)),)
    )

    # Split at explicit indices, the q/k/v split and the query/gate split.
    def split_three(x):
        first, _second, _third = mx.split(x, [128, 256], -1)
        return first + 0

    assert "Split cannot infer output shapes" in _shapeless_error(
        split_three, (mx.zeros((1, 1, 512)),)
    )

    # And the two that do infer, which is why the KV write and the expert
    # gather are not what forces the specialisation.
    buffer = mx.zeros((1, 2, 256, 32))
    written = mx.compile(
        lambda buf, row, start: mx.slice_update(buf, row, start, axes=(2,)),
        shapeless=True,
    )(buffer, mx.zeros((1, 2, 1, 32)), mx.array([0], mx.int32))
    mx.eval(written)
    assert written.shape == buffer.shape

    del layers, spec


def test_a_shapeless_block_must_not_read_its_input_shape(model, spec):
    """Why the hyper-connection block uses ``unflatten`` and not ``reshape``.

    A shapeless trace still hands the function the first call's concrete
    shapes, so a reshape spelled with them is a reshape to the first width. The
    block as written survives the width change; the same block written with a
    literal reshape does not, and this pins the difference.
    """
    hyper = model.language_model.model.layers[0].attn_hyper_connection
    _specs, arrays = C.gated_residual_weights(hyper)
    C.eval_weights(arrays)

    def naive(x):
        y = x.reshape(*x.shape[:-1], spec.hc_count, spec.hidden_size)
        return mx.mean(y, axis=-2)

    compiled = mx.compile(naive, shapeless=True)
    narrow = mx.zeros((1, 1, spec.hidden_size * spec.hc_count))
    wide = mx.zeros((1, 4, spec.hidden_size * spec.hc_count))
    mx.eval(compiled(narrow))
    with pytest.raises(ValueError, match="Cannot reshape array"):
        mx.eval(compiled(wide))

    # The spelling with a ``-1`` in it is worse than the one that raises: it
    # reshapes to a valid but wrong shape and says nothing.
    def sneaky(x):
        return mx.mean(x.reshape(*x.shape[:-1], -1, spec.hidden_size), axis=-2)

    quiet = mx.compile(sneaky, shapeless=True)
    mx.eval(quiet(narrow))
    assert quiet(wide).shape == (1, 1, spec.hidden_size)

    step = C.build_gated_residual(hyper)
    for width in (1, 4):
        hidden = mx.zeros((1, width, spec.hidden_size * spec.hc_count), mx.bfloat16)
        out = step(hidden, arrays)
        mx.eval(out[0])
        assert out[0].shape == (1, width, spec.hidden_size)


def test_a_ple_layer_is_refused_rather_than_silently_dropped(model):
    """One layer of 48 on the checkpoint reads an mmap through numpy mid-forward."""
    layer = model.language_model.model.layers[0]
    marker = object()
    layer["ple"] = marker
    try:
        with pytest.raises(ValueError, match="PLE layer cannot be compiled"):
            C.build_layer(layer)
    finally:
        del layer["ple"]


# ---------------------------------------------------------------------------
# section 5.2: the layer's own handle on its compiled step
# ---------------------------------------------------------------------------


def test_compile_step_builds_the_layer_and_holds_it(model):
    layer = model.language_model.model.layers[0]
    try:
        built = layer.compile_step()
        assert built is not None
        assert built is layer._titan_compiled
        assert built.kind in (C.LINEAR, C.ATTENTION)
    finally:
        layer._titan_compiled = None


def test_compile_step_leaves_a_ple_layer_eager(model):
    """The layer that reads an mmap mid-forward gets ``None``, not an exception.

    ``build_layer`` raises for a PLE layer, which is right for a caller that
    asked for that layer specifically and wrong for a caller walking every
    layer in the model: one layer of 48 on the checkpoint has one and it is
    supposed to stay eager.
    """
    layer = model.language_model.model.layers[0]
    layer["ple"] = object()
    try:
        assert layer.compile_step() is None
        assert layer._titan_compiled is None
    finally:
        del layer["ple"]
        layer._titan_compiled = None


def test_the_compiled_decode_layer_path_is_off_by_default():
    from titan.adapters.mlx.vendor.mlx_vlm.models import forward_paths

    assert forward_paths.enabled("compiled_decode_layer") is False


def test_a_built_layer_stays_eager_until_the_caller_holds_a_compiled_state(
    model, language_model, spec
):
    """The switch alone is not enough, and that is the point of the guard.

    The compiled step takes the state in and gives it back, so a caller still
    holding vendored caches cannot use it. With the path on and an ordinary
    cache the layer must run eagerly and return a hidden; with a
    :class:`LayerState` it must return the step's tuple.
    """
    from titan.adapters.mlx.vendor.mlx_vlm.models import forward_paths

    layer = model.language_model.model.layers[0]
    caches = prefill(language_model, 64, spec.vocab_size)
    width = spec.hidden_size * spec.hc_count
    hidden = mx.random.normal((1, 1, width)).astype(mx.bfloat16)
    built = layer.compile_step()
    try:
        with forward_paths.overridden(compiled_decode_layer=True):
            eager = layer(hidden, None, None, caches[0], None)
        assert isinstance(eager, mx.array)

        state = C.new_state(model).layers[0]
        with forward_paths.overridden(compiled_decode_layer=True):
            out = layer(hidden, None, None, state, None)
        assert isinstance(out, tuple)
        direct = built.step(hidden, *state.arrays, built.weights)
        assert len(out) == len(direct)
        for produced, expected in zip(out, direct):
            assert identical(produced, expected)
    finally:
        layer._titan_compiled = None


def test_a_single_graph_is_refused_past_the_indexer_budget(model, spec, language_model):
    """Past the budget the eager attention selects keys and a single graph does not.

    COMPILED.md section 7 calls a plausible wrong answer here the worst failure
    mode in the document and says the builder should refuse rather than trust
    the caller. It refuses at call time, because the budget is a property of
    the length and the length is not known until a step runs.
    """
    compiled_model = C.build_model(model)
    assert compiled_model.sparse_budget > 0, "the synthetic model has a QSA layer"
    state = C.new_state(model)
    state.length = compiled_model.sparse_budget + 1
    with pytest.raises(ValueError, match="past the QSA indexer budget"):
        compiled_model(tokens(1, spec.vocab_size), state)
