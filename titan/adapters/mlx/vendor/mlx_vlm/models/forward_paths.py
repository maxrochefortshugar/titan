"""Switchboard for the host-side arms of the vendored forward.

Titan addition. The vendored forward has several arms that change how much
Python work and how many kernel launches a step costs without changing what the
step computes: the per-layer ``async_eval``, the batched verify linear, the
batched verify attention, the compiled sub-blocks, and which attention arm a
QSA layer takes. Each one needs to be switchable so a bench can pair it against
its own baseline in one process and so a regression can be bisected without a
rebuild.

There are two kinds of switch here and they are kept apart. A *path* is a
boolean -- an arm is on or it is off -- and is read with :func:`enabled`. A
*route* carries a value, because the question it answers is "from which context
length" rather than "yes or no", and is read with :func:`route`. Both live in
this file because both answer the same question for a call site, and because
splitting them by type would have put half the attention switchboard in the
adapter and half in the vendored tree, which is where they were until ROUND4.

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
from typing import Any, Iterator

__all__ = [
    "ADDED",
    "DEFAULTS",
    "DESCRIPTIONS",
    "ROUTES",
    "ROUTE_DESCRIPTIONS",
    "disable",
    "enable",
    "enabled",
    "overridden",
    "reset",
    "route",
    "routes",
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
    "compiled_decode_layer": False,
    "qsa_pooled_bank_f32": False,
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
    "qsa_pooled_bank_f32": (
        "Hold the QSA pooled block bank in float32 on the cache and extend it "
        "a block at a time, instead of casting the whole bank on every decode "
        "step. Bit-identical scores; 8 MB a layer instead of 4 at 64k."
    ),
    "compiled_decode_layer": (
        "Run a decoder layer as one traced graph, state in and state out. "
        "Needs the caller to hold a compiled DecodeState; see COMPILED.md."
    ),
}

#: name -> default, for the switches that carry a value rather than a flag.
#: Moved here from ``titan/adapters/mlx/kernels.py`` in ROUND4: they are arm
#: switches for the vendored attention, so they belong with the other arm
#: switches rather than in the adapter's kernel door.
ROUTES: dict[str, Any] = {
    "qsa_sparse_singleton_verify": True,
    "qsa_batched_sparse": True,
    "qsa_gather_min_context": 8192,
    "qsa_gather_min_context_verify": 4096,
}

ROUTE_DESCRIPTIONS: dict[str, str] = {
    "qsa_sparse_singleton_verify": (
        "Route a one-row target-verify forward (the depth-0 speculative cycle, "
        "and any decode that asks for a hidden state) through the gathered "
        "sparse decode arm. Off leaves it on the dense masked path, which "
        "reads the whole cache: 2.8 ms a forward at 64k against 1.4."
    ),
    "qsa_batched_sparse": (
        "Give a BatchQSAKVCache the gathered sparse arm, through the "
        "``qsa.gathered_batched`` registry op. Off leaves batched decode and "
        "batched verify on the dense path, where every row reads the whole "
        "cache and the QSA mask is discarded outright when rows are padded."
    ),
    "qsa_gather_min_context": (
        "Context length from which a one-row forward prefers the gathered arm "
        "to the dense masked one. The vendored gate is the QSA token budget, "
        "2048, which is where selection becomes *legal* rather than where it "
        "becomes cheaper: the gathered arm reads a flat 2,051 rows whatever "
        "the context, so at 2048 it is reading more than the dense arm and "
        "paying for selection on top. Clamped up to the budget by the caller."
    ),
    "qsa_gather_min_context_verify": (
        "The same, for a block wider than one row. It is lower because the "
        "dense arm's cost rises with the width and the gathered arm's barely "
        "does: measured per QSA layer on the real-shapes synthetic, width 4 "
        "crosses over near 4096 and width 1 not until 8192."
    ),
}

_lock = threading.Lock()
_state: dict[str, bool] = dict(DEFAULTS)
_routes: dict[str, Any] = dict(ROUTES)


def _check(name: str) -> str:
    if name not in DEFAULTS:
        raise KeyError(
            f"no such forward path {name!r}; known: {', '.join(sorted(DEFAULTS))}"
        )
    return name


def _check_route(name: str) -> str:
    if name not in ROUTES:
        raise KeyError(
            f"no such route {name!r}; known: {', '.join(sorted(ROUTES))}"
        )
    return name


def enabled(name: str) -> bool:
    """Is *name* on? The hot-path read, so it stays a dict lookup."""
    return _state[_check(name)]


def route(name: str) -> Any:
    """The current value of a route. The hot-path read, so it stays a lookup."""
    return _routes[_check_route(name)]


def set_paths(**paths: Any) -> dict[str, Any]:
    """Set several paths or routes at once. Returns the values they had before.

    One setter for both kinds, because a caller pairing arms does not want to
    know which of them happens to be a flag. A name that is neither raises, so
    a typo is loud rather than silent.
    """
    with _lock:
        previous: dict[str, Any] = {}
        for name, value in paths.items():
            if name in ROUTES:
                previous[name] = _routes[name]
                _routes[name] = value
            else:
                previous[_check(name)] = _state[name]
                _state[name] = bool(value)
        return previous


def enable(name: str) -> None:
    set_paths(**{name: True})


def disable(name: str) -> None:
    set_paths(**{name: False})


def reset() -> None:
    """Back to :data:`DEFAULTS` and :data:`ROUTES`."""
    with _lock:
        _state.clear()
        _state.update(DEFAULTS)
        _routes.clear()
        _routes.update(ROUTES)


def snapshot() -> dict[str, bool]:
    """Which paths are on right now. For a bench header or a config dump."""
    return dict(_state)


def routes() -> dict[str, Any]:
    """What the routes are set to right now. Same audience as :func:`snapshot`."""
    return dict(_routes)


@contextmanager
def overridden(**paths: Any) -> Iterator[dict[str, bool]]:
    """Scope a set of paths or routes to a block, restoring the old values."""
    previous = set_paths(**paths)
    try:
        yield snapshot()
    finally:
        set_paths(**previous)
