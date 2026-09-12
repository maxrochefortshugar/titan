"""The pydantic front end over :mod:`titan.config.schema`.

``titan.config.schema`` holds the one definition of what a Titan configuration
is: frozen dataclasses, stdlib only, and the type the engine and the wiring
consume. This module is the view the HTTP surface reads. It covers the three
sections the API actually touches, server, model with its aliases, and limits,
and it exists for two reasons that are worth stating rather than assuming.

First, the API layer wants request-shaped behaviour from its config: resolve an
alias, merge sampling field by field, read the bearer key off disk on demand.
Those are methods on a config object, not on a parse tree.

Second, pydantic gives the HTTP surface the same validation vocabulary it uses
for request bodies, so a bad config and a bad request fail the same way in the
same tests.

Every default here is taken from :mod:`titan.config.schema` rather than typed
again, and ``tests/config/test_parity.py`` compares the two field sets so a
section added on one side and forgotten on the other fails the build.
:meth:`TitanConfig.from_schema` is how the wiring produces this view from the
canonical config, and :meth:`TitanConfig.to_schema` goes the other way.

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

import dataclasses
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from titan.config import schema
from titan.config.schema import (
    CacheConfig,
    DtypePolicy,
    KernelConfig,
    ObservabilityConfig,
    SchedulerConfig,
    SpeculationConfig,
)
from titan.config.schema import TitanConfig as CoreConfig
from titan.core.errors import ConfigError

__all__ = [
    "ServerConfig",
    "SamplingConfig",
    "AliasConfig",
    "ModelConfig",
    "LimitsConfig",
    "TitanConfig",
    "CoreConfig",
    "CacheConfig",
    "DtypePolicy",
    "KernelConfig",
    "ObservabilityConfig",
    "SchedulerConfig",
    "SpeculationConfig",
    "load_config",
    "load_core",
    "resolve_config_path",
    "TITAN_CONFIG_ENV",
]

TITAN_CONFIG_ENV = "TITAN_CONFIG"
"""The only environment variable Titan reads, anywhere. Holds a config path."""

# Production sampling defaults for Qwen3.8-Flash-Next, defined once in the
# schema module. These are the numbers the model card specifies for thinking
# mode; deviating from them (temperature 1.0 in particular) produces the
# repetition and tool-argument corruption that made the first week of agent runs
# unusable.
_DEFAULT_TEMPERATURE = schema.DEFAULT_TEMPERATURE
_DEFAULT_TOP_P = schema.DEFAULT_TOP_P
_DEFAULT_TOP_K = schema.DEFAULT_TOP_K
_DEFAULT_REASONING_EFFORT = schema.DEFAULT_REASONING_EFFORT

_REASONING_EFFORTS = schema.REASONING_EFFORTS


class _Strict(BaseModel):
    """Reject unknown keys. A typo in a config file is an error, not a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerConfig(_Strict):
    """Where the HTTP surface binds and how it authenticates."""

    host: str = "127.0.0.1"
    port: int = Field(default=schema.DEFAULT_PORT, ge=1, le=65535)
    api_key_file: Path | None = None
    """File holding the bearer token. ``None`` disables auth entirely."""
    sse_keepalive_seconds: float = Field(default=10.0, gt=0.0)
    """Idle gap after which a keepalive frame goes out during long prefill."""
    sse_keepalive_mode: str = "chunk"
    """``chunk`` (oMLX-compatible no-op event), ``comment`` (``: ping``) or ``off``."""

    @field_validator("sse_keepalive_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in schema.KEEPALIVE_MODES:
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
        for reserved in schema.RESERVED_TEMPLATE_KWARGS:
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
    """The API's view of the configuration. Built once, at startup, never mutated.

    A projection of :class:`titan.config.schema.TitanConfig`, not a second
    definition of it: :meth:`from_schema` is how the wiring builds this, and
    every field below has a counterpart in the canonical schema.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TitanConfig":
        return cls.model_validate(dict(data))

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        overrides: Sequence[str] = (),
    ) -> "TitanConfig":
        """Load from a ``.toml`` or ``.json`` file, chosen by suffix."""
        p = Path(path)
        raw = p.read_bytes()
        if p.suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
        else:
            data = tomllib.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"config at {p} is not a table")
        if overrides:
            data = schema.apply_overrides(data, overrides)
        return cls.from_mapping(data)

    # -- the bridge to the canonical schema --------------------------------
    @classmethod
    def from_schema(cls, core: CoreConfig) -> "TitanConfig":
        """Project the canonical config onto the sections the API reads."""
        return cls(
            server=ServerConfig(
                host=core.server.host,
                port=core.server.port,
                api_key_file=Path(core.server.api_key_file)
                if core.server.api_key_file
                else None,
                sse_keepalive_seconds=core.server.sse_keepalive_seconds,
                sse_keepalive_mode=core.server.sse_keepalive_mode,
            ),
            model=ModelConfig(
                name=core.model.resolved_name(),
                path=Path(core.model.path),
                sampling=_sampling_from_schema(core.model.sampling),
                enable_thinking=core.model.enable_thinking,
                reasoning_effort=core.model.reasoning_effort,
                aliases={
                    a.name: AliasConfig(
                        enable_thinking=a.enable_thinking,
                        reasoning_effort=a.reasoning_effort,
                        sampling=_sampling_from_schema(a.sampling),
                        template_kwargs=dict(a.template_kwargs),
                    )
                    for a in core.model.aliases
                },
            ),
            limits=LimitsConfig(
                max_tokens=core.limits.max_tokens,
                max_context=core.limits.max_context,
            ),
        )

    def to_schema(self, base: CoreConfig | None = None) -> CoreConfig:
        """Fold this view back into a canonical config.

        ``base`` supplies the sections the API does not own (scheduler, cache,
        speculation, kernels, observability); without one they take their
        defaults. Used by tests and by anything that starts from an API-shaped
        config and needs a whole one.
        """
        core = base or CoreConfig(model=schema.ModelConfig(path=str(self.model.path)))
        return dataclasses.replace(
            core,
            server=schema.ServerConfig(
                host=self.server.host,
                port=self.server.port,
                api_key_file=str(self.server.api_key_file)
                if self.server.api_key_file
                else "",
                request_timeout_s=core.server.request_timeout_s,
                max_body_mb=core.server.max_body_mb,
                sse_keepalive_seconds=self.server.sse_keepalive_seconds,
                sse_keepalive_mode=self.server.sse_keepalive_mode,
            ),
            model=dataclasses.replace(
                core.model,
                path=str(self.model.path),
                name=self.model.name,
                sampling=_sampling_to_schema(self.model.sampling),
                enable_thinking=self.model.enable_thinking,
                reasoning_effort=self.model.reasoning_effort,
                aliases=tuple(
                    schema.AliasConfig(
                        name=name,
                        enable_thinking=a.enable_thinking,
                        reasoning_effort=a.reasoning_effort,
                        sampling=_sampling_to_schema(a.sampling),
                        template_kwargs=dict(a.template_kwargs),
                    )
                    for name, a in sorted(self.model.aliases.items())
                ),
            ),
            limits=schema.LimitsConfig(
                max_tokens=self.limits.max_tokens,
                max_context=self.limits.max_context,
            ),
        )


def _sampling_from_schema(s: schema.SamplingConfig) -> SamplingConfig:
    return SamplingConfig(
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=s.top_k,
        min_p=s.min_p,
        repetition_penalty=s.repetition_penalty,
        max_tokens=s.max_tokens,
    )


def _sampling_to_schema(s: SamplingConfig) -> schema.SamplingConfig:
    return schema.SamplingConfig(
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=s.top_k,
        min_p=s.min_p,
        repetition_penalty=s.repetition_penalty,
        max_tokens=s.max_tokens,
    )


def resolve_config_path(path: str | os.PathLike[str] | None) -> Path:
    """``path``, else ``$TITAN_CONFIG``, else an error.

    This function is the *only* place in Titan that touches the process
    environment, and it reads exactly one name. If neither an argument nor the
    variable is set, that is an error rather than a guess: an inference server
    that silently starts on defaults will serve a model nobody chose.
    """
    if path is not None:
        return Path(path)
    found = os.environ.get(TITAN_CONFIG_ENV)
    if not found:
        raise ValueError(
            f"no config path given and {TITAN_CONFIG_ENV} is unset; "
            "Titan does not start on implicit defaults"
        )
    return Path(found)


def load_core(
    path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] = (),
) -> CoreConfig:
    """Read a config file and build the canonical config. The one parse point.

    TOML by default, JSON when the suffix says so. Both go through the same
    dataclass parse and the same validation, so a JSON config cannot pass a
    check a TOML one would fail.
    """
    resolved = resolve_config_path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {resolved}: {exc}") from None
    if resolved.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{resolved} is not valid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"config at {resolved} is not a table")
        core = CoreConfig.from_mapping(schema.apply_overrides(data, overrides))
        core = dataclasses.replace(
            core,
            overrides=core.overrides + tuple(overrides),
            source=str(resolved),
        )
        core.validate()
        return core
    return CoreConfig.from_toml(text, overrides=overrides, source=str(resolved))


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] = (),
) -> TitanConfig:
    """Load the API's config view from ``path``, else from ``$TITAN_CONFIG``.

    Goes through the canonical parse and projects the result, so this and
    :func:`titan.config.wiring.load_config` cannot disagree about what a file
    means. It stays a separate function because a check that only cares about
    the HTTP surface should not have to construct a whole engine config.
    """
    return TitanConfig.from_schema(load_core(path, overrides))
