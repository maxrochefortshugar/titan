"""Parsing, dumping and the round trip.

The property under test is that a config survives a lap through TOML unchanged.
That matters because ``print-config`` is how an operator captures the resolved
config of a run, and a capture that cannot be fed back in is a record of
something that can never be reproduced.
"""

from __future__ import annotations

import dataclasses

import pytest

from titan.config import schema
from titan.config.schema import TitanConfig
from titan.core.errors import ConfigError

from .conftest import FULL, MINIMAL


def load(text: str = FULL, **kwargs) -> TitanConfig:
    return TitanConfig.from_toml(text, **kwargs)


def model_text(**extra) -> str:
    """MINIMAL with extra keys in the model table, since TOML forbids a second
    ``[model]`` header and half these tests want one more model key."""
    body = "".join(f"{k} = {v!r}\n".replace("'", '"') for k, v in extra.items())
    return (
        '[model]\npath = "/models/qwen"\n'
        'name = "Qwen3.8-Flash-Next-oQ4e-mtp"\n' + body
    )


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_round_trip_toml_config_dump():
    config = load(FULL, source="a.toml")
    again = load(config.dumps(), source="a.toml")
    assert again == config


def test_round_trip_survives_overrides():
    config = load(FULL, overrides=("scheduler.max_seqs=3",), source="a.toml")
    again = load(config.dumps(), source="a.toml")
    assert again.overrides == ("scheduler.max_seqs=3",)
    assert again == config


def test_round_trip_of_a_minimal_config_is_the_defaults():
    config = load(MINIMAL)
    again = load(config.dumps())
    assert again == config
    assert again.server.port == 8085
    assert again.cache.block_tokens == 512
    assert again.cache.snapshot_grid == 2048


def test_dump_is_plain_data():
    d = load(FULL).to_dict()
    assert d["server"]["port"] == 8085
    assert d["kernels"]["disabled"] == ["topk_radix"]
    assert d["model"]["aliases"]["Qwen3.8-Flash-Next-oQ4e-mtp:no-think"][
        "enable_thinking"
    ] is False


def test_print_config_redacts_the_api_key_path():
    config = load(FULL)
    text = config.dumps(redact=True)
    assert "/etc/titan/titan.key" not in text
    assert "***redacted***" in text
    assert "/etc/titan/titan.key" in config.dumps(redact=False)


def test_redaction_of_an_unset_key_adds_nothing():
    assert "redacted" not in load(MINIMAL).dumps(redact=True)


# ---------------------------------------------------------------------------
# defaults that are decisions
# ---------------------------------------------------------------------------


def test_production_sampling_defaults():
    s = schema.SamplingConfig()
    assert (s.temperature, s.top_p, s.top_k) == (0.7, 0.8, 20)
    assert schema.ModelConfig(path="/m").reasoning_effort == "medium"


def test_server_defaults():
    server = schema.ServerConfig()
    assert (server.host, server.port) == ("127.0.0.1", 8085)


def test_soft_guard_is_eighty_five_percent():
    sc = schema.SchedulerConfig()
    assert sc.memory_guard_soft_fraction == 0.85
    assert sc.memory_guard_soft_gb == pytest.approx(110.0 * 0.85)


def test_hot_ram_tier_is_four_gigabytes():
    assert schema.CacheConfig().ram_tier_gb == 4.0


def test_the_model_name_falls_back_to_the_directory():
    assert schema.ModelConfig(path="/models/qwen/").resolved_name() == "qwen"


def test_alias_lookup():
    config = load(FULL)
    base = config.alias("Qwen3.8-Flash-Next-oQ4e-mtp")
    assert base is not None and base.enable_thinking is True
    nothink = config.alias("Qwen3.8-Flash-Next-oQ4e-mtp:no-think")
    assert nothink is not None and nothink.sampling.temperature == 0.0
    assert config.alias("nope") is None


# ---------------------------------------------------------------------------
# parse errors name the key
# ---------------------------------------------------------------------------


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown key 'schedular'"):
        load(MINIMAL + "\n[schedular]\nmax_seqs = 2\n")


def test_unknown_nested_key_names_the_full_path():
    with pytest.raises(ConfigError, match="unknown key 'scheduler.max_seq'"):
        load(MINIMAL + "\n[scheduler]\nmax_seq = 2\n")


def test_unknown_key_inside_an_alias_names_the_alias():
    with pytest.raises(ConfigError, match=r"model\.aliases\.x\.thinking"):
        load(MINIMAL + '\n[model.aliases.x]\nthinking = false\n')


def test_a_missing_required_key_is_named():
    with pytest.raises(ConfigError, match="model.path is required"):
        load("[model]\nname = 'q'\n")


def test_wrong_type_names_the_key():
    with pytest.raises(ConfigError, match="server.port must be an integer"):
        load(MINIMAL + '\n[server]\nport = "8085"\n')


def test_a_bool_is_not_an_integer():
    with pytest.raises(ConfigError, match="scheduler.max_seqs must be an integer"):
        load(MINIMAL + "\n[scheduler]\nmax_seqs = true\n")


def test_a_section_that_is_not_a_table_is_rejected():
    with pytest.raises(ConfigError, match="server must be a table"):
        TitanConfig.from_mapping({"model": {"path": "/m"}, "server": 3})


def test_broken_toml_names_the_source():
    with pytest.raises(ConfigError, match="a.toml is not valid TOML"):
        load("[model\n", source="a.toml")


def test_an_alias_may_not_restate_its_own_name():
    with pytest.raises(ConfigError, match="drop the name field"):
        load(MINIMAL + '\n[model.aliases.x]\nname = "y"\n')


# ---------------------------------------------------------------------------
# validation rules, one test each
# ---------------------------------------------------------------------------


def test_empty_model_path_is_refused():
    with pytest.raises(ConfigError, match="model.path is required"):
        TitanConfig(model=schema.ModelConfig(path="")).validate()


def test_snapshot_grid_must_be_a_multiple_of_the_block_size():
    with pytest.raises(ConfigError, match="cache.snapshot_grid"):
        load(MINIMAL + "\n[cache]\nblock_tokens = 512\nsnapshot_grid = 1500\n")


def test_prefill_chunk_must_be_a_multiple_of_the_block_size():
    with pytest.raises(ConfigError, match="scheduler.prefill_chunk"):
        load(MINIMAL + "\n[cache]\nblock_tokens = 512\n\n[scheduler]\nprefill_chunk = 1000\n")


def test_the_reserved_ports_are_refused():
    for port in (8083, 8084):
        with pytest.raises(ConfigError, match="is reserved"):
            load(MINIMAL + f"\n[server]\nport = {port}\n")
    assert load(MINIMAL + "\n[server]\nport = 8085\n").server.port == 8085


def test_a_port_outside_the_range_is_refused():
    with pytest.raises(ConfigError, match="server.port must be in 1..65535"):
        load(MINIMAL + "\n[server]\nport = 70000\n")


def test_an_empty_host_is_refused():
    with pytest.raises(ConfigError, match="server.host"):
        load(MINIMAL + '\n[server]\nhost = ""\n')


def test_keepalive_mode_is_validated():
    with pytest.raises(ConfigError, match="server.sse_keepalive_mode"):
        load(MINIMAL + '\n[server]\nsse_keepalive_mode = "sometimes"\n')
    for mode in ("chunk", "comment", "off"):
        text = MINIMAL + f'\n[server]\nsse_keepalive_mode = "{mode}"\n'
        assert load(text).server.sse_keepalive_mode == mode


def test_keepalive_interval_must_be_positive():
    with pytest.raises(ConfigError, match="sse_keepalive_seconds"):
        load(MINIMAL + "\n[server]\nsse_keepalive_seconds = 0.0\n")


def test_the_memory_guard_must_leave_headroom_over_the_weights():
    with pytest.raises(ConfigError, match="leaves no headroom"):
        load(model_text(weights_gb=62.5) + "\n[scheduler]\nmemory_guard_gb = 60.0\n")


def test_the_guard_headroom_check_is_off_when_the_weight_size_is_unknown():
    config = load(MINIMAL + "\n[scheduler]\nmemory_guard_gb = 1.0\n")
    assert config.scheduler.memory_guard_gb == 1.0


def test_the_soft_fraction_must_be_a_fraction():
    for bad in ("0.0", "1.5"):
        with pytest.raises(ConfigError, match="memory_guard_soft_fraction"):
            load(MINIMAL + f"\n[scheduler]\nmemory_guard_soft_fraction = {bad}\n")


def test_queue_depth_below_max_seqs_is_refused():
    with pytest.raises(ConfigError, match="queue_depth"):
        load(MINIMAL + "\n[scheduler]\nmax_seqs = 8\nqueue_depth = 4\n")


def test_max_depth_may_not_be_below_min_depth():
    with pytest.raises(ConfigError, match="mtp_depth_max"):
        load(MINIMAL + "\n[speculation]\nmtp_depth_max = 1\nmtp_depth_min = 2\n")


def test_min_depth_must_be_at_least_one():
    with pytest.raises(ConfigError, match="mtp_depth_min"):
        load(MINIMAL + "\n[speculation]\nmtp_depth_min = 0\n")


def test_an_op_may_not_be_both_enabled_and_disabled():
    with pytest.raises(ConfigError, match="both name 'topk_radix'"):
        load(MINIMAL + '\n[kernels]\nenabled = ["topk_radix"]\ndisabled = ["topk_radix"]\n')


def test_reference_only_and_an_enabled_op_contradict_each_other():
    with pytest.raises(ConfigError, match="kernels.reference_only"):
        load(MINIMAL + '\n[kernels]\nreference_only = true\nenabled = ["topk_radix"]\n')


def test_completion_limit_must_fit_the_context():
    with pytest.raises(ConfigError, match="limits.max_tokens"):
        load(MINIMAL + "\n[limits]\nmax_tokens = 100\nmax_context = 50\n")


def test_context_limit_must_fit_the_model():
    with pytest.raises(ConfigError, match="limits.max_context"):
        load(model_text(max_context=4096) + "\n[limits]\nmax_context = 8192\n")


def test_sampling_ranges_are_checked_and_the_path_names_the_section():
    with pytest.raises(ConfigError, match=r"^sampling\.temperature"):
        load(MINIMAL + "\n[sampling]\ntemperature = 3.0\n")
    with pytest.raises(ConfigError, match=r"model\.sampling\.top_p"):
        load(MINIMAL + "\n[model.sampling]\ntop_p = 0.0\n")
    with pytest.raises(ConfigError, match=r"model\.sampling\.min_p"):
        load(MINIMAL + "\n[model.sampling]\nmin_p = 2.0\n")
    with pytest.raises(ConfigError, match=r"repetition_penalty"):
        load(MINIMAL + "\n[model.sampling]\nrepetition_penalty = 0.0\n")
    with pytest.raises(ConfigError, match=r"model\.sampling\.max_tokens"):
        load(MINIMAL + "\n[model.sampling]\nmax_tokens = 0\n")


def test_an_alias_sampling_error_names_the_alias():
    with pytest.raises(ConfigError, match=r"model\.aliases\.x\.sampling\.top_p"):
        load(MINIMAL + "\n[model.aliases.x.sampling]\ntop_p = 4.0\n")


def test_reasoning_effort_is_validated_everywhere():
    with pytest.raises(ConfigError, match="model.reasoning_effort"):
        load(model_text(reasoning_effort="turbo"))
    with pytest.raises(ConfigError, match=r"model\.aliases\.x\.reasoning_effort"):
        load(MINIMAL + '\n[model.aliases.x]\nreasoning_effort = "turbo"\n')


def test_dtype_policy_is_validated():
    with pytest.raises(ConfigError, match="model.dtype.kv"):
        load(MINIMAL + '\n[model.dtype]\nkv = "int3"\n')
    with pytest.raises(ConfigError, match="model.dtype.quantization"):
        load(MINIMAL + '\n[model.dtype]\nquantization = "guess"\n')


def test_an_alias_may_not_shadow_the_canonical_name():
    with pytest.raises(ConfigError, match="collides with the canonical model name"):
        load(MINIMAL + '\n[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp"]\n')


def test_template_kwargs_may_not_shadow_a_real_field():
    with pytest.raises(ConfigError, match="template_kwargs may not set"):
        load(MINIMAL + "\n[model.aliases.x.template_kwargs]\nenable_thinking = false\n")


def test_fine_tail_blocks_must_be_at_least_one():
    with pytest.raises(ConfigError, match="cache.fine_tail_blocks"):
        load(MINIMAL + "\n[cache]\nfine_tail_blocks = 0\n")


def test_positive_numbers_are_checked_across_the_sections():
    cases = [
        ("[cache]\nram_tier_mb = 0.0", "cache.ram_tier_mb"),
        ("[cache]\nmax_stall_ms = 0.0", "cache.max_stall_ms"),
        ("[cache]\nssd_capacity_gb = -1.0", "cache.ssd_capacity_gb"),
        ("[scheduler]\ndecode_rows_budget = 0", "scheduler.decode_rows_budget"),
        ("[speculation]\nacceptance_window = 0", "speculation.acceptance_window"),
        ("[observability]\ncycle_profile_ring = 0", "observability.cycle_profile_ring"),
        ("[observability]\ntrace_sample_every = -1", "observability.trace_sample_every"),
    ]
    for body, expected in cases:
        with pytest.raises(ConfigError, match=expected.replace(".", r"\.")):
            load(MINIMAL + "\n" + body + "\n")
    with pytest.raises(ConfigError, match=r"model\.ngram_reader_workers"):
        load(model_text(ngram_reader_workers=0))


def test_a_frozen_config_cannot_be_mutated():
    config = load(MINIMAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.server.port = 9000  # type: ignore[misc]
