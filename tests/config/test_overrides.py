"""``--set a.b=c``: the one escape hatch, and how its values are typed.

The value goes through the TOML parser rather than being taken as a string.
That is the whole reason this is worth its own test file: an override that gives
``server.port`` the string ``"8085"`` and only fails at bind time is exactly the
kind of late error the config module exists to move to startup.
"""

from __future__ import annotations

import pytest

from titan.config.schema import TitanConfig, apply_overrides, parse_override
from titan.core.errors import ConfigError

from .conftest import MINIMAL


def test_an_integer_stays_an_integer():
    assert parse_override("server.port=8085") == (("server", "port"), 8085)


def test_a_float_stays_a_float():
    path, value = parse_override("scheduler.memory_guard_gb=96.5")
    assert path == ("scheduler", "memory_guard_gb")
    assert value == 96.5


def test_a_bool_stays_a_bool():
    assert parse_override("speculation.enabled=false")[1] is False


def test_an_array_stays_an_array():
    assert parse_override('kernels.disabled=["a", "b"]')[1] == ["a", "b"]


def test_a_quoted_string_loses_its_quotes():
    assert parse_override('server.host="0.0.0.0"')[1] == "0.0.0.0"


def test_a_bare_word_is_a_string():
    assert parse_override("model.path=/models/qwen")[1] == "/models/qwen"


def test_an_empty_value_is_the_empty_string():
    assert parse_override("cache.ssd_dir=")[1] == ""


def test_a_deep_path_is_split():
    path, value = parse_override("model.aliases.x.enable_thinking=false")
    assert path == ("model", "aliases", "x", "enable_thinking")
    assert value is False


def test_an_override_without_an_equals_sign_is_refused():
    with pytest.raises(ConfigError, match="key.path=value"):
        parse_override("server.port")


def test_an_override_with_an_empty_segment_is_refused():
    with pytest.raises(ConfigError, match="empty key segment"):
        parse_override("server..port=1")


def test_apply_does_not_mutate_the_input():
    data = {"server": {"port": 8085}}
    out = apply_overrides(data, ("server.port=9000",))
    assert out["server"]["port"] == 9000
    assert data["server"]["port"] == 8085


def test_apply_creates_missing_tables():
    out = apply_overrides({}, ("a.b.c=1",))
    assert out == {"a": {"b": {"c": 1}}}


def test_descending_into_a_scalar_is_refused():
    with pytest.raises(ConfigError, match="which is not a table"):
        apply_overrides({"server": 3}, ("server.port=1",))


def test_overrides_reach_the_parsed_config_and_are_recorded():
    config = TitanConfig.from_toml(
        MINIMAL,
        overrides=("scheduler.max_seqs=2", "speculation.enabled=false"),
    )
    assert config.scheduler.max_seqs == 2
    assert config.speculation.enabled is False
    assert config.overrides == ("scheduler.max_seqs=2", "speculation.enabled=false")


def test_an_override_of_an_unknown_key_still_fails_validation():
    with pytest.raises(ConfigError, match="unknown key 'scheduler.max_seq'"):
        TitanConfig.from_toml(MINIMAL, overrides=("scheduler.max_seq=2",))


def test_an_override_that_breaks_a_rule_fails_at_startup():
    with pytest.raises(ConfigError, match="is reserved"):
        TitanConfig.from_toml(MINIMAL, overrides=("server.port=8083",))


def test_later_overrides_win():
    config = TitanConfig.from_toml(
        MINIMAL, overrides=("scheduler.max_seqs=2", "scheduler.max_seqs=5")
    )
    assert config.scheduler.max_seqs == 5


def test_an_override_can_add_an_alias():
    config = TitanConfig.from_toml(
        MINIMAL, overrides=("model.aliases.x.enable_thinking=false",)
    )
    alias = config.alias("x")
    assert alias is not None and alias.enable_thinking is False
