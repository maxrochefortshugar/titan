"""Configuration validation, and the one environment variable that exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from titan.config.settings import (
    TITAN_CONFIG_ENV,
    AliasConfig,
    LimitsConfig,
    ModelConfig,
    SamplingConfig,
    ServerConfig,
    TitanConfig,
    load_config,
)

BASE_TOML = """
[server]
host = "127.0.0.1"
port = 8085
api_key_file = "{key_file}"

[model]
name = "Qwen3.8-Flash-Next-oQ4e-mtp"
path = "/models/qwen"

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think"]
enable_thinking = false

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think".sampling]
temperature = 0.0
top_p = 1.0
top_k = 0

[limits]
max_tokens = 32768
max_context = 262144
"""


def test_production_sampling_defaults():
    """The numbers the model card specifies. Changing these is a decision."""
    sampling = SamplingConfig()
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (0.7, 0.8, 20)
    assert ModelConfig(name="m", path=Path("/m")).reasoning_effort == "medium"


def test_load_from_toml(tmp_path):
    key_file = tmp_path / "titan.key"
    key_file.write_text("abc123\n")
    path = tmp_path / "titan.toml"
    path.write_text(BASE_TOML.format(key_file=key_file))

    config = TitanConfig.from_file(path)
    assert config.server.port == 8085
    assert config.server.read_api_key() == "abc123"
    assert config.model.served_names() == [
        "Qwen3.8-Flash-Next-oQ4e-mtp",
        "Qwen3.8-Flash-Next-oQ4e-mtp:no-think",
    ]
    alias = config.model.resolve("Qwen3.8-Flash-Next-oQ4e-mtp:no-think")
    assert alias.enable_thinking is False
    assert alias.sampling.temperature == 0.0


def test_load_from_json(tmp_path):
    path = tmp_path / "titan.json"
    path.write_text(json.dumps({"model": {"name": "m", "path": "/m"}}))
    assert TitanConfig.from_file(path).model.name == "m"


def test_canonical_name_resolves_to_the_model_defaults():
    config = TitanConfig(model=ModelConfig(name="m", path=Path("/m"), reasoning_effort="xhigh"))
    alias = config.model.resolve("m")
    assert alias.reasoning_effort == "xhigh"
    assert alias.enable_thinking is True
    assert config.model.resolve("nope") is None


def test_unknown_key_is_rejected():
    """A typo must stop the process, not silently take a default."""
    with pytest.raises(ValidationError):
        TitanConfig.model_validate(
            {"model": {"name": "m", "path": "/m"}, "limits": {"max_tokns": 10}}
        )


def test_bad_reasoning_effort_is_rejected():
    with pytest.raises(ValidationError):
        AliasConfig(reasoning_effort="turbo")


def test_alias_may_not_shadow_the_canonical_name():
    with pytest.raises(ValidationError):
        ModelConfig(name="m", path=Path("/m"), aliases={"m": AliasConfig()})


def test_template_kwargs_may_not_shadow_a_real_field():
    with pytest.raises(ValidationError):
        AliasConfig(template_kwargs={"enable_thinking": False})


def test_completion_limit_must_fit_the_context():
    with pytest.raises(ValidationError):
        LimitsConfig(max_tokens=100, max_context=50)


def test_keepalive_mode_is_validated():
    with pytest.raises(ValidationError):
        ServerConfig(sse_keepalive_mode="sometimes")
    for mode in ("chunk", "comment", "off"):
        assert ServerConfig(sse_keepalive_mode=mode).sse_keepalive_mode == mode


def test_no_api_key_file_means_no_auth():
    assert ServerConfig().read_api_key() is None


def test_empty_api_key_file_is_an_error(tmp_path):
    key_file = tmp_path / "empty.key"
    key_file.write_text("   \n")
    with pytest.raises(ValueError, match="is empty"):
        ServerConfig(api_key_file=key_file).read_api_key()


def test_config_is_frozen():
    config = TitanConfig(model=ModelConfig(name="m", path=Path("/m")))
    with pytest.raises(ValidationError):
        config.server.port = 9000


def test_load_config_reads_the_one_environment_variable(tmp_path, monkeypatch):
    path = tmp_path / "titan.json"
    path.write_text(json.dumps({"model": {"name": "m", "path": "/m"}}))
    monkeypatch.setenv(TITAN_CONFIG_ENV, str(path))
    assert load_config().model.name == "m"


def test_load_config_without_a_path_or_variable_refuses(monkeypatch):
    """No implicit defaults: a server nobody configured serves a model nobody chose."""
    monkeypatch.delenv(TITAN_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError, match=TITAN_CONFIG_ENV):
        load_config()


def test_explicit_path_beats_the_environment(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps({"model": {"name": "chosen", "path": "/m"}}))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"model": {"name": "other", "path": "/m"}}))
    monkeypatch.setenv(TITAN_CONFIG_ENV, str(other))
    assert load_config(chosen).model.name == "chosen"


def _environment_reads(module) -> list[str]:
    """Names of environment lookups in a module's real code, ignoring prose.

    Walks the AST rather than grepping the text, so a docstring that *mentions*
    ``os.environ`` does not count as touching it.
    """
    import ast
    import inspect

    found: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            found.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
            found.append(node.id)
    return found


def test_settings_module_reads_exactly_one_environment_variable():
    """Guard the rule mechanically rather than trusting review."""
    from titan.config import settings

    assert _environment_reads(settings) == ["environ"]


def test_no_other_module_reads_the_environment():
    from titan.adapters.chat import template, tool_parser
    from titan.api import models, openai, ports, sse

    for module in (openai, models, sse, ports, template, tool_parser):
        assert _environment_reads(module) == [], module.__name__
