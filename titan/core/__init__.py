"""Core domain. Imports nothing outside the standard library.

If a future edit makes this package import mlx, fastapi, numpy or pydantic, the
dependency rule has been broken; ``tests/test_dependency_rule.py`` fails on it.
"""

from titan.core import errors, ports, types

__all__ = ["errors", "ports", "types"]
