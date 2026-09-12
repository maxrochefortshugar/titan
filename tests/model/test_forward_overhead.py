"""What the forward costs the host, and what the cost cuts did to the numbers.

Every test here runs on the synthetic Qwen4-Exp that ``bench/decode/
host_overhead.py`` builds: four decoder layers, hidden 128, eight experts,
4-bit affine quantisation, a few megabytes of weights. It is small enough to
run beside a workbench holding the real checkpoint, and it keeps every
structural property the host cost depends on. What it does not keep is listed
in ``docs/architecture/FORWARD.md``.

Two kinds of test.

*Parity.* Each forward path this workstream added is a change to how a step is
dispatched, not to what it computes, and each one is switchable. So each one
gets the same test: run the step with the path on and with it off, and hold the
difference to a stated bar. Two of the paths are bit-identical and are asserted
as such. The two that are not are held to a bf16 ULP bound and to identical
argmax, which is the property the verify block actually depends on.

*Cost.* The per-token loops are the thing being removed, so a test counts them
directly rather than timing them: how many times the per-row helper ran, how
many attention calls the step made, and how many primitives the graph carried.
Counting beats timing on a shared machine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from titan.adapters.mlx.vendor.mlx_vlm.models import forward_paths
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen3_5 import language as q35

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the forward paths need a Metal device"
)


def _load_bench():
    """Import the bench by path: ``bench`` is a script directory, not a package."""
    path = Path(__file__).resolve().parents[2] / "bench" / "decode" / "host_overhead.py"
    spec = importlib.util.spec_from_file_location("titan_bench_host_overhead", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve their annotations through sys.modules, so a module
    # loaded by path has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bf16_ordinals(x: mx.array) -> np.ndarray:
    """Map bf16 values onto a monotone integer line, so a difference is a ULP.

    Values are rounded to bf16 first: the logits come back in fp32 and the
    question is whether they differ by more than the format the model computes
    in can represent.
    """
    raw = np.array(x.astype(mx.bfloat16).astype(mx.float32)).astype(np.float32)
    bits = raw.view(np.uint32) >> 16
    magnitude = (bits & 0x7FFF).astype(np.int32)
    return np.where((bits & 0x8000) != 0, -magnitude, magnitude)


def ulp_distance(a: mx.array, b: mx.array) -> np.ndarray:
    return np.abs(_bf16_ordinals(a) - _bf16_ordinals(b))


class CallCounter:
    """Count calls to named module-level functions, restoring them after."""

    def __init__(self, module, *names: str) -> None:
        self.module = module
        self.names = names
        self.counts: dict[str, int] = {name: 0 for name in names}
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "CallCounter":
        for name in self.names:
            real = getattr(self.module, name)
            self._saved[name] = real

            def wrapper(*args, _name=name, _real=real, **kwargs):
                self.counts[_name] += 1
                return _real(*args, **kwargs)

            setattr(self.module, name, wrapper)
        return self

    def __exit__(self, *exc) -> None:
        for name, real in self._saved.items():
            setattr(self.module, name, real)


@pytest.fixture(scope="module")
def spec():
    return bench.SyntheticSpec()


@pytest.fixture(scope="module")
def quantized(spec):
    """A quantised synthetic model, as the checkpoint is."""
    return bench.build_synthetic(spec)


@pytest.fixture(scope="module")
def dense(spec):
    """A bf16 synthetic model.

    The checkpoint leaves some projections unquantised, and those are exactly
    the ones the per-row arms used to fall back on: the MoE router, the block
    injection, the GDN gating heads. A bf16 model is how the test reaches them.
    """
    from dataclasses import replace

    return bench.build_synthetic(replace(spec, quantize=False))


def primed(model, context: int = 600):
    """A fresh cache carrying *context* tokens of history."""
    language_model = model.language_model
    cache = language_model.make_cache()
    ids = mx.array([[(index % 400) + 1 for index in range(context)]], dtype=mx.int64)
    output = language_model(ids, cache=cache, skip_logits=True)
    mx.eval(bench._step_outputs(output, cache))
    return cache


def step(model, cache, width: int, rows: int = 1):
    ids = mx.array(
        [[7 + index for index in range(width)] for _ in range(rows)], dtype=mx.int64
    )
    output = model.language_model(ids, cache=cache, return_hidden=True)
    mx.eval(bench._step_outputs(output, cache))
    return output


def rewind(cache, width: int) -> None:
    for entry in cache:
        if hasattr(entry, "trim") and entry.is_trimmable():
            entry.trim(width)


def paired_logits(model, width: int, paths: dict, rows: int = 1):
    """The same step run twice from the same history, once per setting."""
    results = {}
    for label, setting in (("on", True), ("off", False)):
        cache = primed(model)
        with forward_paths.overridden(**{name: setting for name in paths}):
            results[label] = step(model, cache, width, rows=rows).logits
    return results["on"], results["off"]


# ---------------------------------------------------------------------------
# the synthetic model runs at all
# ---------------------------------------------------------------------------


def test_the_synthetic_model_prefills_decodes_and_verifies(quantized):
    cache = primed(quantized, context=64)
    decode = step(quantized, cache, 1)
    assert decode.logits.shape == (1, 1, quantized.config.text_config.vocab_size)
    assert decode.hidden_states, "decode with return_hidden must carry the hidden"
    rewind(cache, 1)
    verify = step(quantized, cache, 4)
    assert verify.logits.shape == (1, 4, quantized.config.text_config.vocab_size)
    assert verify.gdn_states, "a verify block must capture the recurrent state"


# ---------------------------------------------------------------------------
# parity: the paths that must be bit-identical
# ---------------------------------------------------------------------------


def test_cached_norm_scale_is_bit_identical(quantized):
    on, off = paired_logits(quantized, 1, {"cached_norm_scale"})
    assert mx.array_equal(on, off), "folding 1 + weight at load time changed a value"


def test_cached_norm_scale_is_bit_identical_at_verify_width(quantized):
    on, off = paired_logits(quantized, 4, {"cached_norm_scale"})
    assert mx.array_equal(on, off)


def test_batched_verify_linear_is_bit_identical_on_the_quantised_model(quantized):
    """At batch one every quantised projection already took the block-wide call.

    This is the shape production runs, so the change has to be provably free
    there before its cost anywhere else is worth discussing.
    """
    on, off = paired_logits(quantized, 4, {"batched_verify_linear"})
    assert mx.array_equal(on, off)


# ---------------------------------------------------------------------------
# parity: the paths that move the last bits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 6])
def test_batched_verify_linear_holds_its_ulp_bound_on_the_dense_model(dense, width):
    on, off = paired_logits(dense, width, {"batched_verify_linear"})
    distance = ulp_distance(on, off)
    assert np.all(mx.array(mx.argmax(on, -1) == mx.argmax(off, -1)).tolist()), (
        "the block-wide linear changed which token the verify block picks"
    )
    # Four layers of a randomly initialised model amplify the last bit; the
    # bound is what the same measurement gives on the real shapes plus room.
    assert distance.max() <= 16, f"max {distance.max()} bf16 ULP at width {width}"
    assert distance.mean() <= 0.5


@pytest.mark.parametrize("width", [2, 4, 6])
def test_batched_verify_attention_holds_its_ulp_bound(quantized, width):
    """One masked call against one call per query row.

    The two are the same reduction over the same keys, so the difference is
    rounding: a masked-off column contributes an exact zero. It is not free,
    though. Measured against a float32 reference at the real head shapes, the
    block-wide call sits about 2.5 times further out than the per-row loop,
    which is one bf16 epsilon of relative error rather than half of one. That
    is the trade this switch exists to undo.
    """
    on, off = paired_logits(quantized, width, {"batched_verify_attention"})
    distance = ulp_distance(on, off)
    assert np.all(mx.array(mx.argmax(on, -1) == mx.argmax(off, -1)).tolist())
    assert distance.mean() <= 1.0, f"mean {distance.mean()} bf16 ULP"
    assert np.mean(distance > 1) <= 0.05


def test_compiled_gated_residual_holds_its_ulp_bound(quantized):
    on, off = paired_logits(quantized, 1, {"compiled_gated_residual"})
    distance = ulp_distance(on, off)
    assert np.all(mx.array(mx.argmax(on, -1) == mx.argmax(off, -1)).tolist())
    assert distance.mean() <= 1.0


# ---------------------------------------------------------------------------
# cost: the per-token loops are gone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 6])
def test_no_linear_runs_once_per_token_at_batch_one(dense, width):
    """The bf16 projections used to take one launch per row per token."""
    cache = primed(dense)
    with CallCounter(
        q35, "_target_verify_singletons", "_target_verify_timewise"
    ) as counter:
        step(dense, cache, width)
    assert counter.counts == {
        "_target_verify_singletons": 0,
        "_target_verify_timewise": 0,
    }


@pytest.mark.parametrize("width", [2, 4])
def test_no_linear_runs_once_per_token_in_a_batch(quantized, width):
    """Batched rows take the custom verify kernel, or the block-wide call.

    A quantised projection whose input is not a whole number of 512-element
    groups cannot use the kernel. Before this change it fell back to one call
    per token; the checkpoint has two such shapes per layer.
    """
    cache = primed(quantized)
    with CallCounter(
        q35, "_target_verify_singletons", "_target_verify_timewise"
    ) as counter:
        step(quantized, cache, width, rows=1)
    assert sum(counter.counts.values()) == 0


@pytest.mark.parametrize("width", [2, 4, 6])
def test_attention_runs_once_per_layer_not_once_per_row(quantized, width):
    layers = sum(
        1
        for kind in quantized.config.text_config.layer_types
        if kind != "linear_attention"
    )
    cache = primed(quantized)
    with CallCounter(q35, "scaled_dot_product_attention") as counter:
        step(quantized, cache, width)
    calls = counter.counts["scaled_dot_product_attention"]
    assert calls == layers, f"{calls} attention calls for {layers} sparse layers"

    cache = primed(quantized)
    with forward_paths.overridden(batched_verify_attention=False):
        with CallCounter(q35, "scaled_dot_product_attention") as counter:
            step(quantized, cache, width)
    assert counter.counts["scaled_dot_product_attention"] == layers * width


def test_the_verify_block_builds_fewer_primitives_than_it_did(quantized):
    """The graph the host builds is smaller with the added paths on."""
    counts = {}
    for label, setting in (("on", True), ("off", False)):
        cache = primed(quantized)
        with forward_paths.overridden(
            **{name: setting for name in forward_paths.ADDED}
        ):
            ids = mx.array([[7, 8, 9, 10]], dtype=mx.int64)
            with bench.Trace() as trace:
                output = quantized.language_model(
                    ids, cache=cache, return_hidden=True
                )
                trace.finish(bench._step_outputs(output, cache))
            counts[label] = trace.result.ops
    assert counts["on"] < counts["off"], counts
    assert counts["on"] <= 0.95 * counts["off"], counts


# ---------------------------------------------------------------------------
# cost: a prefill chunk is not a verify block
# ---------------------------------------------------------------------------


def test_a_prefill_chunk_asking_for_hidden_states_stays_on_the_batched_arms(
    dense,
):
    """The trap ``model.py`` used to document, closed.

    Asking the vendored model for a hidden state sets ``capture_layer_ids``,
    and that used to put every linear and every attention row of the chunk on
    the per-row arms: measured at 200 seconds for a 150-token chunk against
    0.1 without. The arms are now chosen by the block's width, so a chunk gets
    chunk-shaped work and still gets its hidden state back.
    """
    language_model = dense.language_model
    cache = language_model.make_cache()
    ids = mx.array([[(index % 400) + 1 for index in range(150)]], dtype=mx.int64)
    layers = sum(
        1 for kind in dense.config.text_config.layer_types if kind != "linear_attention"
    )
    with CallCounter(
        q35,
        "_target_verify_singletons",
        "_target_verify_timewise",
        "scaled_dot_product_attention",
    ) as counter:
        output = language_model(ids, cache=cache, return_hidden=True)
        mx.eval(bench._step_outputs(output, cache))

    assert counter.counts["_target_verify_singletons"] == 0
    assert counter.counts["_target_verify_timewise"] == 0
    assert counter.counts["scaled_dot_product_attention"] <= layers
    assert output.hidden_states, "the chunk still has to return its hidden state"


def test_a_wide_block_does_not_capture_the_recurrent_intermediates(dense):
    """The capture is a verify facility and costs a state per row per layer.

    A caller that wants a rollback has to verify a block narrow enough to roll
    back. Every width the engine uses is; a prefill chunk is not, and gets
    ``None`` rather than a hundred megabytes of intermediates it will not use.
    """
    language_model = dense.language_model
    cache = language_model.make_cache()
    ids = mx.array([[(index % 400) + 1 for index in range(150)]], dtype=mx.int64)
    output = language_model(ids, cache=cache, return_hidden=True)
    mx.eval(bench._step_outputs(output, cache))
    assert not output.gdn_states

    cache = language_model.make_cache()
    narrow = language_model(
        mx.array([[1, 2, 3, 4]], dtype=mx.int64), cache=cache, return_hidden=True
    )
    mx.eval(bench._step_outputs(narrow, cache))
    assert narrow.gdn_states, "a verify block still captures what rollback needs"


def test_the_narrow_block_predicate_matches_the_documented_ceiling():
    assert not q35._narrow_verify_block(mx.zeros((1, 1, 8)))
    assert q35._narrow_verify_block(mx.zeros((1, 4, 8)))
    assert q35._narrow_verify_block(
        mx.zeros((1, q35._TARGET_VERIFY_MAX_ROWS, 8))
    )
    assert not q35._narrow_verify_block(
        mx.zeros((1, q35._TARGET_VERIFY_MAX_ROWS + 1, 8))
    )
    assert not q35._narrow_verify_block(
        mx.zeros((8, q35._TARGET_VERIFY_MAX_ROWS, 8))
    )


# ---------------------------------------------------------------------------
# the switchboard itself
# ---------------------------------------------------------------------------


def test_every_path_defaults_on_and_restores_after_an_override():
    before = forward_paths.snapshot()
    assert before == forward_paths.DEFAULTS
    with forward_paths.overridden(eager_dispatch=False):
        assert not forward_paths.enabled("eager_dispatch")
    assert forward_paths.snapshot() == before


def test_an_unknown_path_is_rejected_rather_than_ignored():
    with pytest.raises(KeyError):
        forward_paths.enabled("no_such_path")
    with pytest.raises(KeyError):
        forward_paths.set_paths(no_such_path=True)


def test_every_added_path_is_a_real_path():
    for name in forward_paths.ADDED:
        assert name in forward_paths.DEFAULTS
        assert name in forward_paths.DESCRIPTIONS
