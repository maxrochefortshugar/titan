"""Error taxonomy. Every port raises one of these, never a bare exception.

The split matters operationally: a :class:`CapacityError` is a 429 and the
request can be retried; a :class:`StateError` means the engine's own invariants
broke, so the sequence is aborted and the incident is logged with the profile
around it.
"""

from __future__ import annotations

__all__ = [
    "TitanError",
    "ConfigError",
    "CapacityError",
    "MemoryGuardError",
    "StateError",
    "SnapshotError",
    "BackendError",
    "EngineError",
    "EngineUnhealthyError",
    "KernelError",
    "TemplateError",
    "TokenizerError",
    "ParseError",
]


class TitanError(Exception):
    """Base class for everything Titan raises deliberately."""


class ConfigError(TitanError):
    """Configuration failed validation. Raised at startup, never later."""


class CapacityError(TitanError):
    """Admission refused: queue full, or context longer than the window."""


class MemoryGuardError(CapacityError):
    """Admitting this request would cross the resident-memory guard.

    The guard is a soft limit that gates admission, not a limit that serialises
    running work. Set too low it silently serialises concurrent requests, which
    cost the overlay its whole concurrency win until it was found.
    """


class StateError(TitanError):
    """A state handle was used in a way that breaks its invariants: a stale
    handle, a truncation below the last snapshot, a length mismatch."""


class SnapshotError(TitanError):
    """A recurrent-state snapshot was missing, stale or version-mismatched.

    Never fatal to the process: the caller drops the cache entry and recomputes.
    """


class BackendError(TitanError):
    """The model backend failed. The sequence dies, the process does not."""


class EngineError(TitanError):
    """The loop failed a request for a reason that is not the request's fault.

    Every call the loop thread makes into a port is wrapped, and anything that
    comes back out of one that is not a :class:`TitanError` is reported as this:
    the request dies with an error finish, the loop keeps running, and the
    message names the port call that failed. A bare exception escaping the loop
    thread is the failure mode this class exists to prevent.
    """


class EngineUnhealthyError(EngineError):
    """The engine is not serving. New work is refused rather than queued.

    Set when the backend itself looks dead (a warm-up probe that failed twice
    in a row, or a device error) or when the watchdog caught a loop step over
    its budget. ``GET /health`` reports it and admissions answer 503.
    """


class KernelError(TitanError):
    """A fast kernel failed or produced out-of-tolerance output. The registry
    falls back to the reference implementation and counts it."""


class TemplateError(TitanError):
    """The chat template could not render the request."""


class TokenizerError(TitanError):
    """Text could not be encoded or ids could not be decoded.

    Separate from :class:`TemplateError` because the two fail at different
    points and mean different things to the caller: a template error is a
    conversation the model cannot be asked, a tokenizer error is a string the
    vocabulary cannot represent or a detokenisation stream that lost its place.
    """


class ParseError(TitanError):
    """Tool-call or reasoning markup ended in an unparseable state."""
