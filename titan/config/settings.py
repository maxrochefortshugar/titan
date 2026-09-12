"""Validated configuration for the Titan server.

One rule holds everywhere in this module: **nothing here reads the environment**
except the single optional ``TITAN_CONFIG`` path in :func:`load_config`. Sampling
defaults, the API key, the model path and the context limits all come from a file
the operator can read and diff. An inference server whose behaviour changes with
an exported variable is one that cannot be reproduced from its config, and the
sampling defaults in particular are the difference between usable and unusable
agent turns.

The API key itself is never a config *value*, only a path to a file holding it.
That keeps the secret out of anything that gets pasted into an issue.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ServerConfig",
    "SamplingConfig",
    "AliasConfig",
    "ModelConfig",
    "LimitsConfig",
    "TitanConfig",
    "load_config",
    "TITAN_CONFIG_ENV",
]

TITAN_CONFIG_ENV = "TITAN_CONFIG"
"""The only environment variable Titan reads, anywhere. Holds a config path."""

# Production sampling defaults for Qwen3.8-Flash-Next. These are the numbers the
# model card specifies for thinking mode; deviating from them (temperature 1.0 in
# particular) produces the repetition and tool-argument corruption that made the
# first week of agent runs unusable.
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.8
_DEFAULT_TOP_K = 20
_DEFAULT_REASONING_EFFORT = "medium"

_REASONING_EFFORTS = ("low", "medium", "xhigh")


class _Strict(BaseModel):
    """Reject unknown keys. A typo in a config file is an error, not a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerConfig(_Strict):
    """Where the HTTP surface binds and how it authenticates."""

    host: str = "127.0.0.1"
    port: int = Field(default=8083, ge=1, le=65535)
    api_key_file: Path | None = None
    """File holding the bearer token. ``None`` disables auth entirely."""
    sse_keepalive_seconds: float = Field(default=10.0, gt=0.0)
    """Idle gap after which a keepalive frame goes out during long prefill."""
    sse_keepalive_mode: str = "chunk"
    """``chunk`` (oMLX-compatible no-op event), ``comment`` (``: ping``) or ``off``."""

    @field_validator("sse_keepalive_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in ("chunk", "comment", "off"):
            raise ValueError(
                f"sse_keepalive_mode must be chunk, comment or off, got {v!r}"
            )
        return v

    def read_api_key(self) -> str | None:
        """Read the bearer token, or ``None`` when auth is disabled.

        Read on demand rather than cached at load: rotating the key should not
        need a restart, and the file is read once per unauthenticated request
        at most.
        """
        if self.api_key_file is None:
            return None
        key = self.api_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"api key file {self.api_key_file} is empty")
        return key


class SamplingConfig(_Strict):
    """Sampling defaults. A request may override any of these."""

    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=_DEFAULT_TOP_P, gt=0.0, le=1.0)
    top_k: int = Field(default=_DEFAULT_TOP_K, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    """Default completion budget when the request does not set one."""


class AliasConfig(_Strict):
    """A profile: one model, one set of template kwargs, one set of defaults.

    Aliases are how a client picks a behaviour without knowing anything about
    the template. ``<model>:no-think`` is the canonical example: same weights,
    ``enable_thinking=false``, and greedy-ish sampling because a non-thinking
    turn that wanders is just a slow wrong answer.
    """

    enable_thinking: bool = True
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments passed straight to the chat template."""

    @field_validator("reasoning_effort")
    @classmethod
    def _known_effort(cls, v: str) -> str:
        if v not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {_REASONING_EFFORTS}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _no_kwarg_shadowing(self) -> "AliasConfig":
        for reserved in ("enable_thinking", "reasoning_effort", "add_generation_prompt"):
            if reserved in self.template_kwargs:
                raise ValueError(
                    f"template_kwargs may not set {reserved!r}; it has its own field"
                )
        return self


class ModelConfig(_Strict):
    """The one model this process serves, plus its aliases."""

    name: str
    """Canonical id, echoed back in every response and listed by /v1/models."""
    path: Path
    """Directory holding chat_template.jinja, tokenizer.json and the weights."""
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    enable_thinking: bool = True
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT
    aliases: dict[str, AliasConfig] = Field(default_factory=dict)

    @field_validator("reasoning_effort")
    @classmethod
    def _known_effort(cls, v: str) -> str:
        if v not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {_REASONING_EFFORTS}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _alias_names_distinct(self) -> "ModelConfig":
        if self.name in self.aliases:
            raise ValueError(
                f"alias {self.name!r} collides with the canonical model name"
            )
        return self

    @property
    def base_alias(self) -> AliasConfig:
        """The canonical name's own profile, built from the model-level defaults."""
        return AliasConfig(
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            sampling=self.sampling,
        )

    def resolve(self, requested: str) -> AliasConfig | None:
        """Profile for a requested model string, or ``None`` if unknown."""
        if requested == self.name:
            return self.base_alias
        return self.aliases.get(requested)

    def served_names(self) -> list[str]:
        """Everything /v1/models advertises: the model, then its aliases."""
        return [self.name, *sorted(self.aliases)]

    @property
    def chat_template_path(self) -> Path:
        return self.path / "chat_template.jinja"


class LimitsConfig(_Strict):
    """Hard ceilings. A request asking for more is a 400, not a silent clamp."""

    max_tokens: int = Field(default=32768, gt=0)
    """Largest completion any single request may ask for."""
    max_context: int = Field(default=262144, gt=0)
    """Prompt plus completion. Checked against the rendered prompt."""

    @model_validator(mode="after")
    def _completion_fits(self) -> "LimitsConfig":
        if self.max_tokens > self.max_context:
            raise ValueError(
                f"limits.max_tokens ({self.max_tokens}) exceeds "
                f"limits.max_context ({self.max_context})"
            )
        return self


class TitanConfig(_Strict):
    """The whole configuration. Built once, at startup, and never mutated."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TitanConfig":
        return cls.model_validate(dict(data))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "TitanConfig":
        """Load from a ``.toml`` or ``.json`` file, chosen by suffix."""
        p = Path(path)
        raw = p.read_bytes()
        if p.suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
        else:
            data = tomllib.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"config at {p} is not a table")
        return cls.from_mapping(data)


def load_config(path: str | os.PathLike[str] | None = None) -> TitanConfig:
    """Load the config from ``path``, else from ``$TITAN_CONFIG``.

    This function is the *only* place in Titan that touches ``os.environ``, and
    it reads exactly one name. If neither an argument nor the variable is set,
    that is an error rather than a guess: an inference server that silently
    starts on defaults will serve a model nobody chose.
    """
    if path is None:
        env = os.environ.get(TITAN_CONFIG_ENV)
        if not env:
            raise ValueError(
                f"no config path given and {TITAN_CONFIG_ENV} is unset; "
                "Titan does not start on implicit defaults"
            )
        path = env
    return TitanConfig.from_file(path)
