"""Configuration and the single composition root.

``titan.config.schema`` is pure data and belongs to nobody. ``titan.config.wiring``
is the one module in the package that may import core, engine, adapters, api,
kernels and observability together, because building the object graph is the one
job that needs to see all of them.

``titan.config.settings`` is the validated pydantic view of that data: the
server, model-alias and limit sections the HTTP surface reads at startup. It
reads exactly one environment variable, ``TITAN_CONFIG``, and nothing else in
Titan reads any.
"""

from titan.config import schema, settings, wiring

__all__ = ["schema", "settings", "wiring"]
