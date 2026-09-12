"""Scheduler contracts.

One loop, one thread. The scheduler owns every
:class:`~titan.core.types.SequenceState`, decides what runs each turn of the
loop, and is the only caller of the model backend. Concurrency is inside the
loop, not around it: sequences share cycles, they do not share the interpreter.

Loop shape, one turn:

    1. drain the admission queue (bounded work)
    2. if any sequence is prefilling, run exactly one prefill chunk
    3. otherwise run one decode cycle over every decoding sequence
    4. emit events, store finished prefixes, retire finished sequences

Prefill and decode never interleave inside a turn. The overlay let them
contend, which dropped its prefill chunk to 512 tokens and cost 22% per token
on the path that was already busy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

from titan.core.types import (
    PrefillChunk,
    Request,
    SequenceId,
    SequenceState,
    StreamEnd,
    TokenEvent,
)

__all__ = ["SchedulerPolicy", "Scheduler", "LoopStats"]


@dataclass(frozen=True, slots=True)
class LoopStats:
    """What the loop did over a window. Exposed at ``GET /metrics``."""

    turns: int
    prefill_chunks: int
    decode_cycles: int
    tokens_out: int
    admitted: int
    rejected: int
    mean_rows_per_cycle: float
    queue_depth: int
    resident_gb: float


class SchedulerPolicy(Protocol):
    """The decisions the loop delegates, kept separate so they are testable.

    Every method is pure: same inputs, same answer, no clock reads and no
    device queries. Anything time- or memory-dependent arrives as an argument.
    """

    def admit(self, request: Request, live: Sequence[SequenceState], resident_gb: float) -> bool:
        """Whether to start ``request`` now.

        The memory guard lives here. It gates admission only. A guard that
        throttles work already in flight serialises concurrent requests, which
        is exactly what held the overlay at a flat 76 tok/s aggregate across
        1, 2 and 4 streams until it was raised.
        """

    def next_prefill(self, live: Sequence[SequenceState]) -> PrefillChunk | None:
        """Pick the next prefill chunk, or ``None`` if every sequence decodes."""

    def decode_batch(self, live: Sequence[SequenceState]) -> tuple[SequenceId, ...]:
        """Sequences to advance in the coming cycle, in lockstep.

        Every returned sequence gets a row in the same verify forward. Rows are
        the currency: one row reaches 300 GB/s on the expert gather, eight reach
        549, so a policy that splits a batch to keep latency even is trading
        away most of the machine.
        """

    def rows_budget(self, n_sequences: int) -> int:
        """Cap on total verify rows this cycle, from the config's row budget."""


class Scheduler(Protocol):
    """The loop itself."""

    def submit(self, request: Request) -> Iterator[TokenEvent | StreamEnd]:
        """Admit a request and return its event stream.

        Raises :class:`~titan.core.errors.CapacityError` if admission is
        refused. The iterator yields zero or more :class:`TokenEvent` and
        exactly one :class:`StreamEnd`.
        """

    def cancel(self, sequence_id: SequenceId) -> None:
        """Abort a sequence. Its state handle is closed at the next turn
        boundary, never mid-cycle."""

    def run_forever(self) -> None:
        """Own the calling thread until :meth:`shutdown`."""

    def shutdown(self, drain_timeout_s: float) -> None: ...

    def stats(self) -> LoopStats: ...
