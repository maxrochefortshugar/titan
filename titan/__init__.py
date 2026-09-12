"""Titan: an inference engine for agentic coding on Apple Silicon.

Layering, innermost first. Imports may only point inwards.

    titan.core          domain types, ports, errors. Imports nothing but stdlib.
    titan.engine        scheduler, decode cycle, admission. Imports core only.
    titan.adapters.*    mlx backend, cache store, tokenizer. Import core.
    titan.api           HTTP surface. Imports core and engine.
    titan.kernels       op registry and kernels. Imports core (Op, OpRegistry).
    titan.observability profiler implementations. Imports core.
    titan.config        schema, validation and the single wiring point.

``titan.config.wiring`` is the only module allowed to import from more than one
of the outer layers at once.
"""

__version__ = "0.0.1"
