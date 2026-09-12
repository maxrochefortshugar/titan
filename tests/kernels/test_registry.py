# SPDX-License-Identifier: MIT
"""Registry: selection, forcing, fail-open, and the shape-class memo.

Three properties matter here and none of them is about a kernel:

1. Selection happens once per op per shape class, not per call. The decode loop
   runs 48 MoE blocks per token; a supports() predicate on the hot path would
   be a cost with no purpose.
2. Forcing an op to its reference is a config change, not a code change. That
   is what makes bisecting a bad kernel possible on a running machine, and it
   is why a stale name in the config is an error rather than a no-op.
3. A fast path that raises falls back and counts. Output cannot depend on which
   implementation ran, which is what makes leaving the fallback enabled safe.
"""

import mlx.core as mx
import pytest

from titan.config.schema import KernelConfig
from titan.core.errors import ConfigError, KernelError
from titan.kernels import gdn_norm_gate
from titan.kernels.registry import (
    KernelOp,
    KernelRegistry,
    ShapeClass,
    build_registry,
    reference_only,
    shape_class,
)

EXPECTED_OPS = {
    "gdn_norm_gate", "moe_weighted_sum", "hc_prefill", "gdn_chunk_scan",
    "moe_gather_ws", "moe_gather_int8", "grouped_rmsnorm_bf16", "topk_radix",
    "ple_packed_lookup", "qsa_gathered_attention", "verify_accept",
}

# The names the engine looks ops up by, from titan/kernels/__init__.py.
EXPECTED_ALIASES = {
    "ngram_gather": "ple_packed_lookup",
    "rms_norm_grouped": "grouped_rmsnorm_bf16",
    "gdn_scan_chunked": "gdn_chunk_scan",
    "moe_gather_gate_up": "moe_gather_int8",
    "hyper_connection_block": "hc_prefill",
    "qsa_sparse_decode": "qsa_gathered_attention",
    "mtp_shortlist": "topk_radix",
}


# ---------------------------------------------------------------------------
# a toy op, so the registry's own behaviour is tested without a Metal kernel
# ---------------------------------------------------------------------------


def _toy(calls):
    def reference(x, mode="a"):
        calls.append("reference")
        return x * 2

    def fast(x, mode="a"):
        calls.append("fast")
        if mode == "boom":
            raise RuntimeError("kernel exploded")
        return x * 2

    def key(x, mode="a"):
        return shape_class(x, extra=(mode,))

    def supports(k):
        return k.shapes[0] != (3,)          # one shape class is unsupported

    return KernelOp(name="toy", reference_fn=reference, fast_fn=fast, key=key,
                    supports_key=supports)


def _registry(calls, config=None):
    r = KernelRegistry(config or KernelConfig())
    r.register(_toy(calls))
    return r


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_every_op_is_registered():
    assert set(build_registry().names()) == EXPECTED_OPS


def test_engine_facing_aliases_resolve():
    r = build_registry()
    for alias, canonical in EXPECTED_ALIASES.items():
        assert r.get(alias).name == canonical


def test_every_op_declares_its_shapes_and_provenance():
    for name in build_registry().names():
        op = build_registry().get(name)
        assert op.shapes, f"{name} declares no test shapes"
        assert op.source, f"{name} names no source report"
        assert op.exactness != "unspecified", f"{name} declares no exactness class"


def test_duplicate_registration_is_an_error():
    r = _registry([])
    with pytest.raises(ConfigError):
        r.register(_toy([]))


def test_unknown_op_names_the_registered_ones():
    with pytest.raises(ConfigError) as exc:
        build_registry().get("no_such_op")
    assert "gdn_norm_gate" in str(exc.value)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_fast_is_selected_when_supported():
    calls = []
    r = _registry(calls)
    r.resolve("toy")(mx.zeros((4,)))
    assert calls == ["fast"]


def test_unsupported_shape_class_falls_to_reference():
    calls = []
    r = _registry(calls)
    r.resolve("toy")(mx.zeros((3,)))
    assert calls == ["reference"]


def test_selection_is_memoised_per_shape_class():
    """The predicate must run once per class, however many calls follow."""
    seen = []
    op = _toy([])
    original = op.supports_key
    op.supports_key = lambda k: (seen.append(k), original(k))[1]
    r = KernelRegistry(KernelConfig())
    r.register(op)
    fn = r.resolve("toy")
    for _ in range(10):
        fn(mx.zeros((4,)))
    assert len(seen) == 1


def test_a_different_shape_class_selects_again():
    seen = []
    op = _toy([])
    original = op.supports_key
    op.supports_key = lambda k: (seen.append(k), original(k))[1]
    r = KernelRegistry(KernelConfig())
    r.register(op)
    fn = r.resolve("toy")
    fn(mx.zeros((4,)))
    fn(mx.zeros((8,)))            # different shape
    fn(mx.zeros((4,)), mode="b")  # different mode
    fn(mx.zeros((4,)))            # already seen
    assert len(seen) == 3


def test_shape_class_ignores_values_and_captures_dtype():
    a = shape_class(mx.zeros((2, 2), dtype=mx.bfloat16))
    b = shape_class(mx.ones((2, 2), dtype=mx.bfloat16))
    c = shape_class(mx.zeros((2, 2), dtype=mx.float32))
    assert a == b and a != c
    assert isinstance(a, ShapeClass)


def test_reset_drops_the_memo_not_the_registrations():
    calls = []
    r = _registry(calls)
    r.resolve("toy")(mx.zeros((4,)))
    r.reset()
    assert r.selections() == {}
    assert r.names() == ("toy",)


# ---------------------------------------------------------------------------
# forcing
# ---------------------------------------------------------------------------


def test_disabled_op_uses_the_reference():
    calls = []
    r = _registry(calls, KernelConfig(disabled=("toy",)))
    r.resolve("toy")(mx.zeros((4,)))
    assert calls == ["reference"]


def test_enabled_list_is_an_allowlist():
    calls = []
    r = _registry(calls, KernelConfig(enabled=("something_else",)))
    r.register(KernelOp(name="something_else", reference_fn=lambda x: x,
                        key=lambda x: shape_class(x)))
    r.resolve("toy")(mx.zeros((4,)))
    assert calls == ["reference"]


def test_reference_only_turns_everything_off():
    r = reference_only()
    assert set(r.config.disabled) == EXPECTED_OPS
    x = mx.random.normal((1, 4, 48, 128)).astype(mx.bfloat16)
    w = mx.ones((128,), dtype=mx.bfloat16)
    mx.eval(x, w)
    args = (x, x, w)
    kwargs = {"eps": 1e-6, "activation": "sigmoid"}
    got = r.resolve("gdn_norm_gate")(*args, **kwargs)
    want = gdn_norm_gate.reference(*args, **kwargs)
    mx.eval(got, want)
    assert bool(mx.all(got == want).item())
    assert r.selection("gdn_norm_gate", gdn_norm_gate.key(*args, **kwargs)) \
        == "reference"


def test_a_stale_disable_flag_is_an_error_not_a_no_op():
    with pytest.raises(ConfigError) as exc:
        build_registry(KernelConfig(disabled=("gdn_norm_gaet",)))
    assert "gdn_norm_gaet" in str(exc.value)


def test_an_alias_may_be_disabled_by_either_name():
    for name in ("hc_prefill", "hyper_connection_block"):
        r = build_registry(KernelConfig(disabled=(name,)))
        assert not r._fast_allowed(r.get("hc_prefill"))


# ---------------------------------------------------------------------------
# fail-open
# ---------------------------------------------------------------------------


def test_a_raising_fast_path_falls_back_and_counts():
    calls = []
    r = _registry(calls)
    out = r.resolve("toy")(mx.zeros((4,)), mode="boom")
    mx.eval(out)
    assert calls == ["fast", "reference"]
    assert r.counters()["toy.fallbacks"] == 1
    assert r.counters()["toy.calls"] == 1


def test_fail_open_off_raises_a_kernel_error():
    calls = []
    r = _registry(calls, KernelConfig(fail_open=False))
    with pytest.raises(KernelError):
        r.resolve("toy")(mx.zeros((4,)), mode="boom")


def test_an_op_without_a_fast_path_never_selects_one():
    r = KernelRegistry(KernelConfig())
    r.register(KernelOp(name="bare", reference_fn=lambda x: x,
                        key=lambda x: shape_class(x)))
    assert r.selection("bare", shape_class(mx.zeros((4,)))) == "reference"
    with pytest.raises(KernelError):
        r.get("bare").fast(mx.zeros((4,)))
