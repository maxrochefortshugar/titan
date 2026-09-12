"""Chat template renderer.

Renders the checkpoint's own template, including the reasoning-effort setting
and the qwen3_coder tool block. Determinism is the invariant that matters: the
same messages must render to the same string every turn, or the prefix cache
misses and an agent turn drops from 68 effective tok/s to 9-15.
"""

from __future__ import annotations

__all__: list[str] = []
