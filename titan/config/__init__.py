"""Configuration and the single composition root.

``titan.config.schema`` is the one definition of what a Titan configuration is:
frozen dataclasses, stdlib only, every section, and the parse and validation
that turn a TOML table into one. It belongs to nobody and imports nobody.

``titan.config.settings`` is the pydantic projection the HTTP surface reads. It
covers the server, model-alias and limit sections, takes every default from the
schema module rather than restating it, and reads exactly one environment
variable, ``TITAN_CONFIG``. Nothing else in Titan reads any.

``titan.config.wiring`` is the one module in the package that may import core,
engine, adapters, api, kernels and observability together, because building the
object graph is the one job that needs to see all of them. It does every one of
those imports inside a function, so importing the wiring costs nothing and a
component that does not exist yet fails with its module named instead of taking
the whole package down.
"""

from titan.config import schema, settings, wiring

__all__ = ["schema", "settings", "wiring"]
