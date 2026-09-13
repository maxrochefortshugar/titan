"""Generation 2: the eager path's fused Metal kernels, inside the trace.

The whole claim of generation 2 is one sentence -- the compiled step runs the
same kernels the eager step runs, so it produces the same bits -- and it rests
on one property of MLX that a docstring cannot establish: ``mx.compile`` at
fixed shapes accepts an ``mx.fast.metal_kernel`` as an opaque node, and
shapeless it does not. Both halves are asserted here, with the error message
verbatim, because a claim about an MLX error is a claim about an MLX version.

Everything runs on the synthetic Qwen4-Exp **cast to bfloat16**, which is the
one substantive difference from ``test_compiled_path.py``. In float32 -- how
``bench/decode/host_overhead.py`` leaves the model -- not one fused decode
kernel is eligible: ``hc_fused`` declines a float32 norm weight by layout and
``gdn_norm_gate`` declines a float32 input by dtype. A parity test against an
eager arm with no kernels in it would pass while testing nothing, which is
exactly the hole generation 1's synthetic numbers fell into: they promised 26%
on a model whose GPU work was trivial and delivered +71% on a checkpoint whose
GPU work is not.

Five kinds of test.

*Opaqueness.* Each kernel traces at fixed shapes and raises the documented
error shapeless.

*Kernel parity.* Each traceable body is bit-identical to the eager call site it
was lifted from, at every decode width.

*Whole-step parity.* The compiled model against the eager forward, widths 1 to
6, contexts crossing 256, 512 and 2048, logits and Gated DeltaNet state both.

*Layout.* Islands and segments for a layer stack, including the checkpoint's,
from the config alone.

*The trace cache.* Warming the grid inside a budget, and what a cached call
costs against a first one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from titan.adapters.mlx import compiled as C
from titan.adapters.mlx import compiled_kernels as CK

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels need a Metal device"
)

#: The checkpoint's layer stack, from its ``config.json``. Repeated here rather
#: than read off disk so the test does not need the 70 GB checkpoint present,
#: and so a change to either is a visible diff against the other.
CHECKPOINT_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(48)
)
CHECKPOINT_PLE_LAYERS = (2,)
CHECKPOINT_INDEXER_BUDGET = 2048


def _load_bench():
    path = Path(__file__).resolve().parents[2] / "bench" / "decode" / "compiled_path.py"
    spec = importlib.util.spec_from_file_location("titan_bench_compiled_path", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return bench.SyntheticSpec(num_hidden_layers=4)


@pytest.fixture(scope="module")
def model(spec):
    """The synthetic model in bfloat16, which is the dtype the kernels need."""
    return bench.bf16_synthetic(spec)


@pytest.fixture(scope="module")
def language_model(model):
    return model.language_model


@pytest.fixture(scope="module")
def hyper_connection(language_model):
    return language_model.model.layers[0].attn_hyper_connection


def identical(actual, expected) -> bool:
    return bool(mx.all(actual == expected))


def prefill(language_model, length: int, vocab: int):
    cache = language_model.make_cache()
    done = 0
    while done < length:
        piece = min(512, length - done)
        ids = mx.array(
            [[(done + i) % (vocab - 2) + 1 for i in range(piece)]], dtype=mx.int64
        )
        language_model(ids, cache=cache, skip_logits=True)
        mx.eval(_cache_arrays(cache))
        done += piece
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
    copy = language_model.make_cache()
    for fresh, held in zip(copy, caches):
        if getattr(held, "keys", None) is not None:
            fresh.keys = mx.array(held.keys)
            fresh.values = mx.array(held.values)
            fresh.offset = held.offset
            if getattr(held, "index_keys", None) is not None:
                fresh.index_keys = mx.array(held.index_keys)
                fresh.index_position_ids = mx.array(held.index_position_ids)
        elif getattr(held, "state", None) is not None:
            fresh.state = [None if v is None else mx.array(v) for v in held.state]
    return copy


def tokens(width: int, vocab: int, start: int = 11) -> mx.array:
    return mx.array(
        [[(start + i) % (vocab - 2) + 1 for i in range(width)]], dtype=mx.int64
    )


# ---------------------------------------------------------------------------
# the synthetic model has to be a model the kernels apply to
# ---------------------------------------------------------------------------


def test_the_bfloat16_synthetic_is_a_model_the_kernels_take(hyper_connection):
    """Without this the rest of the file is a suite that asserts nothing.

    ``build_synthetic`` leaves the model in float32 and every fused decode
    kernel declines float32. If the cast in ``bf16_synthetic`` ever stops
    landing, the parity tests below would compare the ops path against the ops
    path and pass. So the eligibility itself is a test.
    """
    assert hyper_connection.hc_norm.weight.dtype == mx.bfloat16
    assert CK.hc_fused_plan(hyper_connection) is not None


def test_the_float32_synthetic_is_not(spec):
    """The other half of the same statement, and the reason the cast exists."""
    plain = bench.build_synthetic(spec)
    module = plain.language_model.model.layers[0].attn_hyper_connection
    assert module.hc_norm.weight.dtype == mx.float32
    assert CK.hc_fused_plan(module) is None


# ---------------------------------------------------------------------------
# opaqueness: fixed shapes yes, shapeless no
# ---------------------------------------------------------------------------


def _hc_inputs(hyper_connection, width):
    arrays = CK.hc_fused_arrays(hyper_connection)
    C.eval_weights(arrays)
    columns = hyper_connection.hc_count * hyper_connection.hidden_size
    x = mx.random.normal((1, width, columns)).astype(mx.bfloat16)
    mx.eval(x)
    return x, arrays


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_hc_fused_body_traces_at_fixed_shapes(hyper_connection, width):
    plan = CK.hc_fused_plan(hyper_connection)
    x, arrays = _hc_inputs(hyper_connection, width)
    step = mx.compile(lambda a, b: CK.hc_fused_body(a, b, plan))
    mixed, held, injection = step(x, arrays)
    mx.eval(mixed, held, injection)
    assert mixed.shape == (1, width, hyper_connection.hidden_size)


def test_hc_fused_body_refuses_a_shapeless_trace(hyper_connection):
    """The error that makes generation 2 per-shape, quoted.

    It is not a limitation to route around. A custom kernel's output shape is
    something the caller declares, not something MLX can infer, so a shapeless
    trace has nothing to infer it from. Per-shape tracing is the price of
    having the kernels at all.
    """
    plan = CK.hc_fused_plan(hyper_connection)
    x, arrays = _hc_inputs(hyper_connection, 4)
    error = CK.opaque_under_shapeless(lambda a, b: CK.hc_fused_body(a, b, plan), x, arrays)
    assert error is not None
    assert CK.OPAQUE_SHAPELESS_ERROR in error
    assert "Primitive::output_shapes" in error


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_hc_fused_body_is_bit_identical_to_the_eager_fused_call(
    hyper_connection, width
):
    """Same kernels, same bits. The sentence generation 2 exists to make true.

    The reference is ``hc_fused.fused_forward``, which is what
    ``Qwen4ExpGatedResidual.__call__`` takes for one to sixteen rows on a
    checkpoint-shaped input -- so this compares the compiled body against the
    thing the eager decode actually runs, not against the ops form it also has.
    """
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp import hc_fused

    plan = CK.hc_fused_plan(hyper_connection)
    x, arrays = _hc_inputs(hyper_connection, width)
    assert hc_fused.compatible(hyper_connection, x)

    got = mx.compile(lambda a, b: CK.hc_fused_body(a, b, plan))(x, arrays)
    want = hc_fused.fused_forward(hyper_connection, x)
    mx.eval(got, want)
    assert want is not None
    for produced, expected in zip(got, want):
        assert identical(produced, expected)


def test_hc_fused_body_refuses_a_row_count_the_kernels_do_not_take(hyper_connection):
    """Seventeen rows is the prefill path's shape, and it says so rather than
    quietly computing something else."""
    plan = CK.hc_fused_plan(hyper_connection)
    x, arrays = _hc_inputs(hyper_connection, CK.HC_MAX_ROWS + 1)
    with pytest.raises(ValueError, match="hc_fused takes 1"):
        CK.hc_fused_body(x, arrays, plan)


def test_gdn_norm_gate_traces_and_is_bit_identical(language_model):
    """The other kernel generation 1 gave up, against its eager call site."""
    gdn = language_model.model.layers[0].linear_attn
    plan = CK.gdn_norm_gate_plan(gdn.norm, width=4)
    assert plan is not None

    shape = (1, 4, gdn.num_v_heads, gdn.head_v_dim)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    gate = mx.random.normal(shape).astype(mx.bfloat16)
    mx.eval(x, gate)

    step = mx.compile(lambda a, b: CK.gdn_norm_gate_body(a, b, gdn.norm.weight, plan))
    got = step(x, gate)
    want = gdn.norm(x, gate)
    mx.eval(got, want)
    assert identical(got, want)


def test_gdn_norm_gate_declines_width_one_because_the_eager_site_declines_it(
    language_model,
):
    """Not because the kernel is wrong there. Because eager does not take it.

    ``Qwen4ExpRMSNormGated.__call__`` gates the fused launch on
    ``x.shape[0] * x.shape[1] > 1``. Bit-identity with eager means taking the
    kernel exactly where eager takes it, so the plan declines the same row
    count -- and if that gate ever moves, this test moves with it rather than
    the compiled path silently diverging.
    """
    gdn = language_model.model.layers[0].linear_attn
    assert CK.gdn_norm_gate_plan(gdn.norm, width=1) is None
    assert CK.gdn_norm_gate_plan(gdn.norm, width=2) is not None


def test_gdn_norm_gate_refuses_a_shapeless_trace(language_model):
    gdn = language_model.model.layers[0].linear_attn
    plan = CK.gdn_norm_gate_plan(gdn.norm, width=4)
    shape = (1, 4, gdn.num_v_heads, gdn.head_v_dim)
    x = mx.zeros(shape, mx.bfloat16)
    gate = mx.zeros(shape, mx.bfloat16)
    error = CK.opaque_under_shapeless(
        lambda a, b: CK.gdn_norm_gate_body(a, b, gdn.norm.weight, plan), x, gate
    )
    assert error is not None and CK.OPAQUE_SHAPELESS_ERROR in error


def test_the_compiled_gated_residual_falls_back_at_a_width_the_kernels_decline(
    hyper_connection,
):
    """Seventeen rows composes the ops form, at trace time, and still agrees.

    The fallback has to happen while the trace is being taken, not while it is
    running, or a compiled step's arithmetic would depend on a value. It does:
    ``hyper_input.shape`` is a Python tuple inside a per-shape trace.
    """
    step = C.build_gated_residual(hyper_connection, fused_kernels=True)
    _specs, arrays = C.gated_residual_weights(hyper_connection, fused_kernels=True)
    C.eval_weights(arrays)
    columns = hyper_connection.hc_count * hyper_connection.hidden_size

    plain = C.build_gated_residual(hyper_connection)
    _plain_specs, plain_arrays = C.gated_residual_weights(hyper_connection)
    C.eval_weights(plain_arrays)

    x = mx.random.normal((1, CK.HC_MAX_ROWS + 1, columns)).astype(mx.bfloat16)
    mx.eval(x)
    got = step(x, arrays)
    want = plain(x, plain_arrays)
    mx.eval(got, want)
    for produced, expected in zip(got, want):
        assert identical(produced, expected)


# ---------------------------------------------------------------------------
# whole-step parity against the eager forward
# ---------------------------------------------------------------------------


def _compiled_arm(model, language_model, cache, length: int, width: int):
    islands = C.island_layers(model, length=length + width)
    compiled_model = C.build_gen2(model, length=length + width)
    C.eval_weights(compiled_model.weights)
    if islands:
        copies = clone_caches(language_model, cache)
        compiled_model.eager_caches = {index: copies[index] for index in islands}
    state = C.read_layer_state(cache, eager_indices=islands)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    return compiled_model, state, islands


@pytest.mark.parametrize("context", [200, 300, 600, 2000])
@pytest.mark.parametrize("width", [1, 2, 4, 6])
def test_gen2_step_is_bit_identical_to_eager(
    model, language_model, spec, context, width
):
    """The headline. Same kernels, same bits, at every width and either side of
    the 256 and 512 capacity growth points.

    Bit-identical, not within a ULP. Generation 1 was within a ULP because it
    computed the hyper-connection block a different way; generation 2 computes
    it the same way, so anything short of equality would be a bug rather than a
    tolerance.
    """
    cache = prefill(language_model, context, spec.vocab_size)
    compiled_model, state, _islands = _compiled_arm(
        model, language_model, cache, context, width
    )
    ids = tokens(width, spec.vocab_size)

    eager_cache = clone_caches(language_model, cache)
    eager = language_model(ids, cache=eager_cache, return_hidden=True)
    logits, _hidden, _next = compiled_model(ids, state)
    mx.eval(logits, eager.logits)
    assert identical(logits, eager.logits)


@pytest.mark.parametrize("context", [600, 2100])
@pytest.mark.parametrize("width", [1, 4])
def test_gen2_gdn_state_is_bit_identical_to_eager(
    model, language_model, spec, context, width
):
    """The recurrent state directly, not just the logits it produced.

    A step can agree on its output and disagree on the state it leaves behind,
    and the state is what the next token depends on. 2100 crosses the indexer
    budget, so the sparse layer is an island there and this also checks the
    seam.
    """
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache

    cache = prefill(language_model, context, spec.vocab_size)
    compiled_model, state, _islands = _compiled_arm(
        model, language_model, cache, context, width
    )
    ids = tokens(width, spec.vocab_size)

    eager_cache = clone_caches(language_model, cache)
    language_model(ids, cache=eager_cache, skip_logits=True)
    _logits, _hidden, produced = compiled_model(ids, state)
    mx.eval([a for entry in produced.layers for a in entry.arrays])
    mx.eval(_cache_arrays(eager_cache))

    checked = 0
    for index, entry in enumerate(produced.layers):
        if entry.kind != C.LINEAR:
            continue
        assert isinstance(eager_cache[index], ArraysCache)
        conv, ssm = entry.arrays
        assert identical(conv, eager_cache[index][0])
        assert identical(ssm, eager_cache[index][1])
        checked += 1
    assert checked, "the synthetic model has no linear layer to compare"


@pytest.mark.parametrize("context", [2100, 3000])
@pytest.mark.parametrize("width", [1, 4])
def test_gen2_runs_past_the_indexer_budget_where_generation_1_refused(
    model, language_model, spec, context, width
):
    """The refusal generation 1 raised is generation 2's island.

    COMPILED.md section 7 called a dense trace above the budget the worst
    failure mode in the document: a plausible answer that attends to keys the
    eager path drops. Generation 1 closed it by refusing the length. Generation
    2 closes it by not tracing that layer, which is the difference between "64k
    is out of scope" and "64k costs one extra dispatch per sparse layer".

    Width 4 as well as width 1, because an island runs through
    ``_segmented``'s own mask construction rather than the traced causal
    comparison, and at width 1 both helpers return ``None``. Only a verify
    width exercises the mask, and a wrong mask there would be a plausible wrong
    answer of exactly the kind this test is about.
    """
    cache = prefill(language_model, context, spec.vocab_size)
    compiled_model, state, islands = _compiled_arm(
        model, language_model, cache, context, width
    )
    assert islands, "past the budget the sparse layers must be islands"
    ids = tokens(width, spec.vocab_size)
    eager_cache = clone_caches(language_model, cache)
    eager = language_model(ids, cache=eager_cache, return_hidden=True)
    logits, _hidden, _next = compiled_model(ids, state)
    mx.eval(logits, eager.logits)
    assert identical(logits, eager.logits)


def test_a_generation_1_build_still_refuses_past_the_budget(model, language_model, spec):
    """Loosening the refusal for generation 2 must not loosen it for anyone else."""
    cache = prefill(language_model, 2100, spec.vocab_size)
    compiled_model = C.build_model(model)
    C.eval_weights(compiled_model.weights)
    state = C.read_layer_state(cache)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    with pytest.raises(ValueError, match="past the QSA indexer budget"):
        compiled_model(tokens(1, spec.vocab_size), state)


# ---------------------------------------------------------------------------
# the verify-block rollback contract, unchanged from generation 1
# ---------------------------------------------------------------------------


def test_rollback_after_a_verify_block_repeats_the_step_exactly(
    model, language_model, spec
):
    """Run width 1, run a width-4 block, roll it back, run width 1 again.

    This is ``rollback_speculative_state`` and it is unchanged from generation
    1 on purpose: the attention half is an offset assignment and the recurrent
    half comes back from the snapshot the engine stages before every block.
    What generation 2 changes is what the step computes, not what the state
    contract promises, and a test that fails here would say the fused kernels
    leaked state, which they must not.
    """
    context = 600
    cache = prefill(language_model, context, spec.vocab_size)
    compiled_model, state, _islands = _compiled_arm(
        model, language_model, cache, context, 4
    )
    snapshot = state.recurrent_arrays()
    single = tokens(1, spec.vocab_size)

    first, _hidden, _after = compiled_model(single, state)
    mx.eval(first)

    block, _bhidden, blocked = compiled_model(tokens(4, spec.vocab_size), state)
    mx.eval(block)
    assert blocked.length == state.length + 4

    # The contract's arithmetic, unchanged: come back to the length before the
    # block plus ``accepted + 1``, the bonus token included.
    kept = C.rollback_speculative_state(blocked, 1, 4, snapshot)
    assert kept.length == state.length + 2

    # And a full rejection, which puts the state back where the block started
    # so the next step is the step that would have happened.
    rolled = C.truncate_state(blocked, state.length, snapshot)
    assert rolled.length == state.length
    again, _ahidden, _next = compiled_model(single, rolled)
    mx.eval(again)
    assert identical(again, first)


def test_truncate_state_still_needs_a_snapshot_for_the_recurrent_half(
    model, language_model, spec
):
    """Unchanged contract, asserted so generation 2 cannot quietly relax it."""
    cache = prefill(language_model, 600, spec.vocab_size)
    _compiled, state, _islands = _compiled_arm(model, language_model, cache, 600, 1)
    with pytest.raises(ValueError):
        C.truncate_state(state, state.length - 1)


# ---------------------------------------------------------------------------
# the layout: islands and segments
# ---------------------------------------------------------------------------


def test_the_checkpoints_layout_below_the_budget():
    """Two traced segments and one island: the PLE layer at index 2.

    Below the indexer budget the sparse layers are dense, so they stay in the
    traces, and the only thing a trace cannot hold is the disk read.
    """
    plan = C.plan_layout(
        layer_types=CHECKPOINT_LAYER_TYPES,
        ple_layer_ids=CHECKPOINT_PLE_LAYERS,
        length=600,
        indexer_budget=CHECKPOINT_INDEXER_BUDGET,
    )
    assert plan.islands == (2,)
    assert plan.segments == ((0, 2), (3, 48))
    assert plan.capacity_in_trace_key is True


def test_the_checkpoints_layout_above_the_budget():
    """Thirteen islands and twelve segments, and capacity leaves the trace key.

    The twelve sparse-attention layers join the PLE layer on the eager side,
    which cuts the stack into twelve runs of consecutive linear layers. None of
    those runs holds a KV buffer, so no trace of theirs depends on capacity --
    which is why the trace count stops growing with the context exactly where
    the context starts getting long.
    """
    plan = C.plan_layout(
        layer_types=CHECKPOINT_LAYER_TYPES,
        ple_layer_ids=CHECKPOINT_PLE_LAYERS,
        length=64000,
        indexer_budget=CHECKPOINT_INDEXER_BUDGET,
    )
    assert plan.islands == (2, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    assert plan.island_count == 13
    assert plan.segment_count == 12
    assert plan.segments[0] == (0, 2)
    assert plan.segments[1] == (4, 7)
    assert plan.segments[-1] == (44, 47)
    assert plan.capacity_in_trace_key is False
    # capacity out of the key means the grid is the widths alone
    assert plan.traces([1, 2, 3, 4, 5, 6, 7, 8], C.capacity_buckets(up_to=65536)) == 96


def test_capacity_buckets_double_above_2048():
    buckets = C.capacity_buckets(up_to=65536)
    assert buckets[:3] == [256, 512, 768]
    assert 2048 in buckets and 4096 in buckets
    assert buckets[buckets.index(2048) + 1] == 4096
    # eight buckets of 256 up to 2048, then five doublings to 65536: thirteen
    # in all for a 64k context, against 250 if the step never changed
    assert len(buckets) == 13
    assert buckets[-1] == 65536


def test_island_layers_matches_plan_layout_on_the_synthetic_model(model):
    """The live-model function and the config-only function have to agree.

    They are two implementations of one rule and the config-only one is what
    answers "how many traces does the checkpoint need" without loading 70 GB,
    so a drift between them would be a number in a document that no code
    produces.
    """
    inner = model.language_model.model
    layer_types = [
        "linear_attention" if layer.is_linear else "full_attention"
        for layer in inner.layers
    ]
    ple = [index for index, layer in enumerate(inner.layers) if "ple" in layer]
    budget = C._sparse_budget(inner)
    for length in (600, 2048, 2049, 8192):
        expected = C.plan_layout(
            layer_types=layer_types,
            ple_layer_ids=ple,
            length=length,
            indexer_budget=budget,
        )
        assert list(expected.islands) == C.island_layers(model, length=length)


# ---------------------------------------------------------------------------
# the trace cache
# ---------------------------------------------------------------------------


def test_warming_the_grid_makes_the_second_call_cheaper(model, spec):
    """A first call at a new shape pays the trace; a later one pays a dispatch.

    Asserting the total across the grid rather than each row, and no absolute
    millisecond bound at all: the machine this runs on has a workbench on it,
    and a per-row bound would be a flaky test rather than a fact about the
    cache. Summed over eight points the trace cost is the signal and the
    machine's mood is the noise.
    """
    compiled_model = C.build_gen2(model, length=600)
    C.eval_weights(compiled_model.weights)
    widths = (1, 2, 3, 4, 5, 6, 7, 8)
    report = C.warm_traces(
        compiled_model,
        widths=widths,
        capacities=(1024,),
        budget_seconds=120.0,
        vocab=spec.vocab_size,
    )
    assert report.warmed == len(widths)
    assert not report.skipped
    assert all(row["first_ms"] > 0 and row["second_ms"] > 0 for row in report.rows)
    first = sum(row["first_ms"] for row in report.rows)
    cached = sum(row["second_ms"] for row in report.rows)
    assert cached < first


def test_warming_stops_at_its_time_budget(model, spec):
    """A budget that cannot be met cuts the grid rather than overrunning it."""
    compiled_model = C.build_gen2(model, length=600)
    C.eval_weights(compiled_model.weights)
    report = C.warm_traces(
        compiled_model,
        widths=(1, 2, 3, 4, 5, 6, 7, 8),
        capacities=(256, 512, 1024, 2048),
        budget_seconds=0.0,
        vocab=spec.vocab_size,
    )
    assert report.skipped
    assert report.warmed + len(report.skipped) == 32


def test_warming_does_not_disturb_the_eager_islands(model, language_model, spec):
    """Warming runs on throwaway caches and puts the caller's back.

    An island holds a vendored cache that the functional state cannot reach, so
    a warm-up that stepped it would leave the model one token ahead of its own
    KV -- silently, and only past the indexer budget.
    """
    cache = prefill(language_model, 2100, spec.vocab_size)
    compiled_model, _state, islands = _compiled_arm(
        model, language_model, cache, 2100, 1
    )
    assert islands
    held = dict(compiled_model.eager_caches)
    offsets = {i: int(held[i].offset) for i in islands}
    C.warm_traces(
        compiled_model,
        widths=(1, 2),
        capacities=(4096,),
        budget_seconds=60.0,
        vocab=spec.vocab_size,
    )
    assert compiled_model.eager_caches == held
    for index in islands:
        assert int(compiled_model.eager_caches[index].offset) == offsets[index]


def test_the_warm_report_says_what_it_cost(model, spec):
    compiled_model = C.build_gen2(model, length=600)
    C.eval_weights(compiled_model.weights)
    report = C.warm_traces(
        compiled_model,
        widths=(1, 2),
        capacities=(1024,),
        budget_seconds=60.0,
        vocab=spec.vocab_size,
    )
    summary = report.summary()
    assert summary["warmed"] == 2
    assert summary["first_call_total_ms"] > 0
    assert summary["cached_call_median_ms"] > 0
    assert "active_growth_mb" in summary and "peak_mb" in summary
