"""The two views of the config must not drift.

:mod:`titan.config.schema` is the definition; :mod:`titan.config.settings` is
the pydantic projection the HTTP surface reads. Two files describing the same
sections is a maintenance hazard, so the hazard is mechanised here: add a field
on one side and forget the other, and this file fails. The alternative was one
class, and one class would have to be either a dataclass the API cannot validate
requests with or a pydantic model the engine's frozen core would import.
"""

from __future__ import annotations

import dataclasses

import pytest

from titan.config import schema, settings

from .conftest import FULL


def _dataclass_fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_server_sections_agree():
    api = set(settings.ServerConfig.model_fields)
    core = _dataclass_fields(schema.ServerConfig)
    assert api <= core, api - core
    # The API does not bind the socket, so these are the two it does not read.
    assert core - api == {"request_timeout_s", "max_body_mb"}


def test_sampling_sections_agree():
    assert set(settings.SamplingConfig.model_fields) == _dataclass_fields(
        schema.SamplingConfig
    )


def test_limits_sections_agree():
    assert set(settings.LimitsConfig.model_fields) == _dataclass_fields(
        schema.LimitsConfig
    )


def test_alias_sections_agree():
    api = set(settings.AliasConfig.model_fields)
    core = _dataclass_fields(schema.AliasConfig)
    assert core - api == {"name"}, "the dataclass carries the alias name inline"
    assert api - core == set()


def test_model_sections_agree_on_what_the_api_reads():
    api = set(settings.ModelConfig.model_fields)
    core = _dataclass_fields(schema.ModelConfig)
    assert api <= core, api - core


def test_the_defaults_are_defined_once():
    assert settings.SamplingConfig().temperature == schema.DEFAULT_TEMPERATURE
    assert settings.SamplingConfig().top_p == schema.DEFAULT_TOP_P
    assert settings.SamplingConfig().top_k == schema.DEFAULT_TOP_K
    assert settings.ServerConfig().port == schema.DEFAULT_PORT
    assert (
        settings.AliasConfig().reasoning_effort == schema.DEFAULT_REASONING_EFFORT
    )


def test_the_api_view_serves_on_the_titan_port_by_default():
    """8085, not 8083 or 8084. The reserved ports belong to production."""
    assert settings.ServerConfig().port == 8085
    assert 8085 not in schema.RESERVED_PORTS


def test_projection_round_trips_through_the_api_view():
    core = schema.TitanConfig.from_toml(FULL)
    view = settings.TitanConfig.from_schema(core)
    again = view.to_schema(core)
    assert again == core


def test_the_projection_carries_the_alias_table():
    core = schema.TitanConfig.from_toml(FULL)
    view = settings.TitanConfig.from_schema(core)
    alias = view.model.resolve("Qwen3.8-Flash-Next-oQ4e-mtp:no-think")
    assert alias is not None
    assert alias.enable_thinking is False
    assert alias.reasoning_effort == "low"
    assert alias.template_kwargs == {"tool_choice": "auto"}


def test_the_projection_carries_the_api_key_path():
    core = schema.TitanConfig.from_toml(FULL)
    view = settings.TitanConfig.from_schema(core)
    assert str(view.server.api_key_file) == "/etc/titan/titan.key"


def test_an_unset_key_file_projects_to_no_auth():
    core = schema.TitanConfig.from_toml("[model]\npath = '/m'\n")
    view = settings.TitanConfig.from_schema(core)
    assert view.server.api_key_file is None
    assert view.server.read_api_key() is None


def test_settings_load_config_accepts_a_full_titan_file(tmp_path):
    """A whole config file, sections the API does not read included."""
    p = tmp_path / "titan.toml"
    p.write_text(FULL, encoding="utf-8")
    view = settings.load_config(p)
    assert view.server.port == 8085
    assert view.model.name == "Qwen3.8-Flash-Next-oQ4e-mtp"


def test_settings_load_config_takes_overrides(tmp_path):
    p = tmp_path / "titan.toml"
    p.write_text(FULL, encoding="utf-8")
    view = settings.load_config(p, ("server.port=9000",))
    assert view.server.port == 9000


def test_settings_reads_exactly_one_environment_variable():
    """The same guard the API layer's own suite applies, kept here too.

    This is the rule most likely to be broken by a well-meaning edit to the
    config package, and the config package is what this suite owns.
    """
    import ast
    import inspect

    found = [
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(ast.parse(inspect.getsource(settings)))
        if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"))
        or (isinstance(node, ast.Name) and node.id in ("environ", "getenv"))
    ]
    assert found == ["environ"]


@pytest.mark.parametrize("module_name", ["schema", "wiring"])
def test_no_other_config_module_reads_the_environment(module_name):
    import ast
    import importlib
    import inspect

    module = importlib.import_module(f"titan.config.{module_name}")
    found = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"))
        or (isinstance(node, ast.Name) and node.id in ("environ", "getenv"))
    ]
    assert found == []
