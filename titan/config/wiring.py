"""Composition root. The only module allowed to import across layers.

Everything is constructed here, once, in a known order, and handed to whoever
needs it as a constructor argument. There is no service locator, no global
registry lookup at call time and no import-time side effect anywhere else in the
package: an adapter is used because it was passed in, which is what makes a test
able to swap any one of them for a fake.

Construction order is fixed by dependency, not by taste:

    config -> profiler -> op registry -> kernels -> model backend
           -> tokenizer, template renderer, tool parser
           -> kv store -> prefix cache
           -> drafter, verifier -> decode cycle -> scheduler -> api app

The backend's ``warmup()`` runs after the registry is populated and before the
API binds its port, so no client ever pays for Metal compilation.
"""

from __future__ import annotations

from typing import Any, Protocol

from titan.config.schema import TitanConfig

__all__ = ["Runtime", "build_runtime", "load_config"]


class Runtime(Protocol):
    """The assembled object graph. Held by the process entry point."""

    config: TitanConfig
    scheduler: Any
    profiler: Any
    registry: Any

    def start(self) -> None: ...

    def stop(self, drain_timeout_s: float) -> None: ...


def load_config(path: str, overrides: tuple[str, ...] = ()) -> TitanConfig:
    """Read the TOML file, apply ``key.path=value`` overrides, validate, return.

    Raises :class:`~titan.core.errors.ConfigError` on the first problem, with
    the offending key named. Never falls back to a default for a key the file
    got wrong: a typo must stop the process, not silently change behaviour.
    """
    raise NotImplementedError


def build_runtime(config: TitanConfig) -> Runtime:
    """Build the object graph. Loads the model; does not bind the port."""
    raise NotImplementedError
