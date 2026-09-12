"""Switchboard for the host-side arms of the vendored forward.

Titan addition. The vendored forward has several arms that change how much
Python work and how many kernel launches a step costs without changing what the
step computes: the per-layer ``async_eval``, the batched verify linear, the
batched verify attention, and the compiled sub-blocks. Each one needs to be
switchable so a bench can pair it against its own baseline in one process and
so a regression can be bisected without a rebuild.

The rules this follows are the registry's, not the environment's. Every path
has a name, a default, and a docstring line below; nothing here reads
``os.environ``; and flipping a path is a function call, so a test can scope it
with :func:`overridden` and get the old value back on the way out.

A path being off is always a supported configuration. Off means the stock arm,
which is the slower one, never the wrong one.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

__all__ = [
    "ADDED",
    "DEFAULTS",
    "DESCRIPTIONS",
    "disable",
    "enable",
    "enabled",
    "overridden",
    "reset",
    "set_paths",
    "snapshot",
]

#: name -> default. Every default is the fast arm; see the module docstring.
DEFAULTS: dict[str, bool] = {
    "eager_dispatch": True,
    "batched_verify_linear": True,
    "batched_verify_attention": True,
    "compiled_gated_residual": True,
    "cached_norm_scale": True,
}

#: The paths this workstream added, as opposed to the ones it inherited.
#: Turning all of these off gives the forward Titan had before it, which is
#: what the paired before/after table in FORWARD.md is measured against.
ADDED: tuple[str, ...] = (
    "batched_verify_linear",
    "batched_verify_attention",
    "compiled_gated_residual",
    "cached_norm_scale",
)

DESCRIPTIONS: dict[str, str] = {
    "eager_dispatch": (
        "async_eval the residual stream after every decoder layer when the row "
        "count is small. Scheduling only: outputs are bit-identical."
    ),
    "batched_verify_linear": (
        "Run a verify block's linears once over the whole width instead of "
        "once per token. Off restores the per-token loop."
    ),
    "batched_verify_attention": (
        "Run a verify block's sparse attention once over the whole width "
        "instead of one masked SDPA per query row."
    ),
    "compiled_gated_residual": (
        "Compile the hyper-connection gated residual and the injection back "
        "into the 4-wide residual stream."
    ),
    "cached_norm_scale": (
        "Use the RMSNorm scale folded at load time rather than rebuilding "
        "1 + weight every call."
    ),
}

_lock = threading.Lock()
_state: dict[str, bool] = dict(DEFAULTS)


def _check(name: str) -> str:
    if name not in DEFAULTS:
        raise KeyError(
            f"no such forward path {name!r}; known: {', '.join(sorted(DEFAULTS))}"
        )
    return name


def enabled(name: str) -> bool:
    """Is *name* on? The hot-path read, so it stays a dict lookup."""
    return _state[_check(name)]


def set_paths(**paths: bool) -> dict[str, bool]:
    """Set several paths at once. Returns the values they had before."""
    with _lock:
        previous = {}
        for name, value in paths.items():
            previous[_check(name)] = _state[name]
            _state[name] = bool(value)
        return previous


def enable(name: str) -> None:
    set_paths(**{name: True})


def disable(name: str) -> None:
    set_paths(**{name: False})


def reset() -> None:
    """Back to :data:`DEFAULTS`."""
    with _lock:
        _state.clear()
        _state.update(DEFAULTS)


def snapshot() -> dict[str, bool]:
    """What is on right now. For a bench header or a resolved-config dump."""
    return dict(_state)


@contextmanager
def overridden(**paths: bool) -> Iterator[dict[str, bool]]:
    """Scope a set of paths to a block, restoring the old values after."""
    previous = set_paths(**paths)
    try:
        yield snapshot()
    finally:
        set_paths(**previous)
