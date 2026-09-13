"""The composition root, built against fakes.

Two things are worth asserting here and neither is about an adapter. The graph
builds in dependency order from a config alone, and a component that does not
exist yet fails with its module named rather than with an ``ImportError`` from
somewhere three layers down. The second is what lets W1.7 land before the store,
the tokenizer and the scheduler do.
"""

from __future__ import annotations

import pytest

from titan.config import settings, wiring
from titan.config.schema import TitanConfig
from titan.core.errors import ConfigError

from .conftest import FULL, MINIMAL


def test_the_wiring_module_imports_nothing_heavy():
    """Importing the composition root must not pull MLX into the process.

    ``titan check-config`` runs on a laptop against a config for a machine that
    is not this one, and it goes through this module.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(wiring))
    top_level = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    imported = set()
    for node in top_level:
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif node.module:
            imported.add(node.module)
    for name in imported:
        assert not name.startswith("titan.adapters"), name
        assert not name.startswith("titan.engine"), name
        assert not name.startswith("titan.api"), name
        assert not name.startswith("titan.kernels"), name
        assert not name.startswith("mlx"), name


def test_build_runtime_with_fakes(fake_parts):
    config = TitanConfig.from_toml(MINIMAL)
    runtime = wiring.build_runtime(config, fake_parts)
    assert runtime.config is config
    assert runtime.backend is fake_parts.backend
    assert runtime.engine is fake_parts.engine
    assert runtime.scheduler is fake_parts.engine
    assert runtime.app is fake_parts.app


def test_start_warms_the_backend_once_and_starts_the_engine(fake_parts):
    runtime = wiring.build_runtime(TitanConfig.from_toml(MINIMAL), fake_parts)
    runtime.start()
    runtime.start()
    assert fake_parts.backend.warmed == 1
    assert fake_parts.engine.started == 1


def test_stop_drains_the_engine_then_the_store(fake_parts):
    runtime = wiring.build_runtime(TitanConfig.from_toml(MINIMAL), fake_parts)
    runtime.start()
    runtime.stop(5.0)
    assert fake_parts.engine.stopped == [5.0]
    assert fake_parts.store.flushed == [5.0]


def test_build_runtime_validates_first(fake_parts):
    """A config that was built without ``from_toml`` is still validated."""
    from titan.config import schema

    bad = TitanConfig(
        model=schema.ModelConfig(path="/m"),
        server=schema.ServerConfig(port=8083),
    )
    with pytest.raises(ConfigError, match="is reserved"):
        wiring.build_runtime(bad, fake_parts)


def test_a_missing_component_names_the_module_that_owes_it(fake_parts):
    """A backend that cannot say how a state becomes bytes says so by name,
    rather than failing with an ImportError three layers down."""
    from titan.config.wiring import Parts

    parts = Parts(**{**vars(fake_parts), "codec": None})
    with pytest.raises(NotImplementedError, match="titan.adapters.mlx.backend"):
        wiring.build_runtime(TitanConfig.from_toml(MINIMAL), parts)


def test_the_codec_comes_from_the_backend_and_carries_the_signature(fake_parts):
    """Only the adapter knows how a state handle turns into bytes, and the
    signature is what stops one build reading another's payloads."""

    class Backend:
        layer_layout = ("gdn", "qsa")

        def state_codec(self, signature):
            return ("codec", signature)

    config = TitanConfig.from_toml(MINIMAL)
    kind, signature = wiring.build_codec(config, Backend())
    assert kind == "codec"
    assert signature.layer_layout == ("gdn", "qsa")
    assert signature.block_tokens == 512


def test_a_backend_without_a_layer_layout_is_named(fake_parts):
    with pytest.raises(NotImplementedError, match="titan.adapters.mlx.backend"):
        wiring.build_signature(TitanConfig.from_toml(MINIMAL), fake_parts.backend)


def test_the_store_is_built_from_the_cache_section(tmp_path):
    config = TitanConfig.from_toml(
        MINIMAL + f'\n[cache]\nram_tier_mb = 8.0\nssd_dir = "{tmp_path}"\n'
    )

    class Backend:
        layer_layout = ("gdn", "qsa")

    signature = wiring.build_signature(config, Backend())
    assert signature.block_tokens == 512
    store = wiring.build_store(config, signature)
    assert store.get_block(b"\x00" * 32) is None
    assert store.flush(1.0)


def test_the_cache_is_built_from_the_cache_section(tmp_path):
    """Both knobs that were dead or unenforced now arrive at the cache."""
    config = TitanConfig.from_toml(
        MINIMAL + "\n[cache]\nfine_min_gain_tokens = 640\nmax_stall_ms = 30.0\n"
    )

    class Backend:
        layer_layout = ("gdn", "qsa")

    signature = wiring.build_signature(config, Backend())
    store = wiring.build_store(config, signature)
    try:
        codec = _StubCodec(signature)
        cache = wiring.build_cache(config, store, codec)
        assert cache.store_budget_s == pytest.approx(0.030)
        assert cache._fine_min_gain == 640
    finally:
        store.close()


class _StubCodec:
    def __init__(self, signature):
        self.signature = signature

    def export_blocks(self, state, start, end):  # pragma: no cover - unused
        raise AssertionError

    def import_blocks(self, state, start, end, payload):  # pragma: no cover
        raise AssertionError

    def export_snapshot(self, state, length):  # pragma: no cover - unused
        raise AssertionError

    def import_snapshot(self, state, length, payload):  # pragma: no cover
        raise AssertionError


def test_the_admission_config_carries_the_soft_guard():
    config = TitanConfig.from_toml(MINIMAL)
    admission = wiring.build_admission_config(config)
    assert admission.max_sequences == 8
    assert admission.block_tokens == 512
    assert admission.memory_guard_gb == pytest.approx(110.0 * 0.85)


def test_the_decode_cycle_follows_the_speculation_switch():
    from titan.engine.decode_cycle import MTPDecodeCycle, PlainDecodeCycle

    on = TitanConfig.from_toml(MINIMAL)
    off = TitanConfig.from_toml(MINIMAL + "\n[speculation]\nenabled = false\n")
    args = dict(backend=object(), tokenizer=object(), profiler=None)
    assert isinstance(wiring.build_cycle(on, **args), MTPDecodeCycle)
    assert isinstance(wiring.build_cycle(off, **args), PlainDecodeCycle)


def test_the_profiler_is_real_and_follows_the_config():
    from titan.observability.profiler import NullProfiler, RingProfiler

    on = wiring.build_profiler(TitanConfig.from_toml(MINIMAL))
    assert isinstance(on, RingProfiler)
    assert on.snapshot()["counters"] == {}
    off = wiring.build_profiler(
        TitanConfig.from_toml(MINIMAL + "\n[observability]\ncycle_profile = false\n")
    )
    assert isinstance(off, NullProfiler)


def test_the_kernel_registry_is_real_and_honours_the_config():
    config = TitanConfig.from_toml(MINIMAL + '\n[kernels]\ndisabled = ["topk_radix"]\n')
    registry = wiring.build_registry(config)
    assert "topk_radix" in registry.names()
    assert registry.config.disabled == ("topk_radix",)


def test_reference_only_turns_every_fast_path_off():
    config = TitanConfig.from_toml(MINIMAL + "\n[kernels]\nreference_only = true\n")
    registry = wiring.build_registry(config)
    assert set(registry.config.disabled) == set(registry.names())
    assert registry.config.fail_open is False


def test_a_stale_bisect_name_is_a_config_error():
    config = TitanConfig.from_toml(MINIMAL + '\n[kernels]\ndisabled = ["nosuchop"]\n')
    with pytest.raises(ConfigError, match="nosuchop"):
        wiring.build_registry(config)


def test_build_app_uses_the_api_projection_of_the_config(fake_parts):
    config = TitanConfig.from_toml(FULL)
    app = wiring.build_app(
        config,
        engine=fake_parts.engine,
        tokenizer=fake_parts.tokenizer,
        template=fake_parts.template,
    )
    deps = app.state.deps
    assert deps.config.server.port == 8085
    assert deps.config.model.served_names() == [
        "Qwen3.8-Flash-Next-oQ4e-mtp",
        "Qwen3.8-Flash-Next-oQ4e-mtp:no-think",
    ]
    assert deps.config.model.resolve(
        "Qwen3.8-Flash-Next-oQ4e-mtp:no-think"
    ).sampling.temperature == 0.0


def test_load_config_reads_a_file_and_records_its_source(write_config):
    path = write_config(MINIMAL)
    config = wiring.load_config(str(path))
    assert config.source == str(path)
    assert config.model.path == "/models/qwen"


def test_load_config_applies_overrides(write_config):
    path = write_config(MINIMAL)
    config = wiring.load_config(str(path), ("scheduler.max_seqs=3",))
    assert config.scheduler.max_seqs == 3


def test_load_config_falls_back_to_the_one_environment_variable(
    write_config, monkeypatch
):
    path = write_config(MINIMAL)
    monkeypatch.setenv(settings.TITAN_CONFIG_ENV, str(path))
    assert wiring.load_config().model.path == "/models/qwen"


def test_load_config_without_a_path_or_variable_refuses(monkeypatch):
    monkeypatch.delenv(settings.TITAN_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError, match=settings.TITAN_CONFIG_ENV):
        wiring.load_config()


def test_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config"):
        wiring.load_config(str(tmp_path / "nope.toml"))


def test_the_resolved_config_view_is_redacted():
    view = wiring.resolved_config_view(TitanConfig.from_toml(FULL))
    assert view["server"]["api_key_file"] == "***redacted***"


def test_the_guard_takes_the_per_token_cost_from_the_backend():
    """The default is a number from another checkpoint, and a guard that
    overestimates does not fail loudly: it skips the long prompt, which keeps
    its place in the queue and never runs."""

    class Backend:
        layer_layout = ("gdn", "qsa")
        state_bytes_per_token = 28_000.0

    config = TitanConfig.from_toml(MINIMAL)
    assert wiring.build_admission_config(config).state_bytes_per_token == 320_000.0
    admission = wiring.build_admission_config(config, Backend())
    assert admission.state_bytes_per_token == 28_000.0


# ---------------------------------------------------------------------------
# the depth policy, and the forward-path switchboard
# ---------------------------------------------------------------------------
#
# ROUND5 step 1 measures the converging policy against the one ROUND4 shipped,
# so both have to be reachable from a config. "The new policy is steadier" is a
# claim about two policies, and measuring one of them from a previous round's
# notes measures the machine's drift as well.


def test_the_default_adaptive_policy_is_the_expected_value_one():
    config = TitanConfig.from_toml(MINIMAL)
    assert type(wiring.build_verifier(config)).__name__ == (
        "ExpectedValueDepthController"
    )


def test_the_old_policy_is_selectable_by_name():
    config = TitanConfig.from_toml(
        MINIMAL, overrides=["speculation.depth_policy=mean_accepted"]
    )
    assert type(wiring.build_verifier(config)).__name__ == "DepthController"


def test_a_fixed_policy_ignores_the_policy_name():
    """``adaptive_depth = false`` is a pinned depth and there is no policy to
    choose between; asking for the old one must not turn adaptation back on."""
    config = TitanConfig.from_toml(
        MINIMAL,
        overrides=[
            "speculation.adaptive_depth=false",
            "speculation.depth_policy=mean_accepted",
            "speculation.mtp_depth_min=3",
            "speculation.mtp_depth_max=3",
        ],
    )
    verifier = wiring.build_verifier(config)
    assert type(verifier).__name__ == "ExpectedValueDepthController"
    assert verifier.adaptive is False
    assert verifier.next_depths(1) == [3]


def test_the_probe_duty_comes_from_the_config():
    config = TitanConfig.from_toml(
        MINIMAL,
        overrides=[
            "speculation.depth_probe_every=96",
            "speculation.depth_probe_cycles=6",
            "speculation.depth_hysteresis=0.02",
        ],
    )
    verifier = wiring.build_verifier(config)
    assert (verifier.probe_every, verifier.probe_cycles) == (96, 6)
    assert verifier.hysteresis == pytest.approx(0.02)


def test_an_unknown_forward_path_is_named_rather_than_ignored():
    config = TitanConfig.from_toml(
        MINIMAL, overrides=['kernels.forward_paths_on=["no_such_arm"]']
    )
    with pytest.raises(ConfigError, match="no_such_arm"):
        wiring.apply_forward_paths(config)


def test_a_forward_path_named_on_is_on_and_the_rest_keep_their_defaults():
    from titan.adapters.mlx.vendor.mlx_vlm.models import forward_paths

    before = forward_paths.snapshot()
    config = TitanConfig.from_toml(
        MINIMAL,
        overrides=[
            'kernels.forward_paths_on=["qsa_pooled_bank_f32"]',
            'kernels.forward_paths_off=["eager_dispatch"]',
        ],
    )
    try:
        resolved = wiring.apply_forward_paths(config)
        assert resolved["qsa_pooled_bank_f32"] is True
        assert resolved["eager_dispatch"] is False
        assert resolved["cached_norm_scale"] == before["cached_norm_scale"]
    finally:
        forward_paths.set_paths(**before)
