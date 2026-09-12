"""Admission: from an API-shaped request to a scheduled sequence.

Admission is where a request stops being text and becomes work:

    render (TemplateRenderer) -> encode (Tokenizer) -> lookup (PrefixCache)
    -> guard check -> open_state (ModelBackend) -> restore -> plan_chunks

It runs on the scheduler thread, in bounded batches, because it calls
``open_state`` and touches the guard. Rendering and encoding are pure and could
move off-thread later; the ordering constraint is only that the guard decision
and the state allocation happen in the same turn, so two requests cannot both
pass a guard that only one of them fits under.

Three rules are worth stating in one place, because each one is a measured
failure somewhere else.

**The guard gates admission and nothing else.** It refuses new work. It never
shrinks a running sequence's budget, never preempts, never throttles. A guard
that throttles serialises concurrency, and that held the overlay flat at 76
tok/s aggregate across one, two and four streams.

**A request that cannot fit is skipped, not blocked on.** oMLX's
``_schedule_waiting`` breaks out of its loop at the first request that fails a
gate, so a 60k prompt arriving in front of eight short ones stops all of them
until it fits. Here the queue is walked and each entry is asked whether it fits
now; the ones that do start, the ones that do not keep their place and their
arrival order. Order is preserved, head-of-line blocking is not.

**Chunk ends land on the grid, always.** Every snapshot-grid multiple strictly
inside the uncached suffix is itself a chunk end, because a chunk that steps
over one stages no snapshot there and the store chain truncates. That is D8's
second rule and it is checked by :func:`plan_chunks` rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from titan.core.errors import CapacityError, MemoryGuardError
from titan.core.types import (
    PrefixMatch,
    Request,
    RequestId,
    SequenceId,
    SequencePhase,
    SequenceState,
)

__all__ = [
    "AdmissionConfig",
    "AdmissionPlan",
    "Admitter",
    "MemoryGuard",
    "WaitQueue",
    "PortAdmitter",
    "plan_chunks",
    "snapshot_positions",
]


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    """Everything admission needs from the config file, flattened.

    Flattened on purpose: the engine may not import ``titan.config``, and a
    dataclass of plain numbers is what makes the whole admission path testable
    without a TOML file. ``titan.config.wiring`` builds one of these.
    """

    max_sequences: int = 8
    queue_depth: int = 64
    max_context: int = 262144
    prefill_chunk_tokens: int = 2048
    block_tokens: int = 512
    snapshot_grid: int = 2048
    memory_guard_gb: float = 110.0
    weights_gb: float = 78.0
    """Resident model weights. The guard's floor, not part of a sequence."""
    state_bytes_per_token: float = 320_000.0
    """KV plus recurrent state per token of context, from the memory model."""
    state_fixed_bytes: float = 128 * 1024 * 1024
    """Per-slot cost that does not scale with length: conv windows, carriers,
    the MTP block's own state, and the staged snapshots."""

    def __post_init__(self) -> None:
        if self.prefill_chunk_tokens % self.block_tokens:
            raise ValueError("prefill chunk must be a multiple of the block size")
        if self.snapshot_grid % self.block_tokens:
            raise ValueError("snapshot grid must be a multiple of the block size")


@dataclass(frozen=True, slots=True)
class AdmissionPlan:
    """The work a newly admitted sequence implies. Produced before any device
    allocation, so it can be logged, tested and refused cheaply."""

    request: Request
    match: PrefixMatch
    chunk_ends: tuple[int, ...]
    """Ascending prefill chunk end positions, last one at ``len(prompt)``."""
    snapshot_at: tuple[int, ...]
    """Positions where a recurrent snapshot is staged. Always includes the
    prompt end: the final chunk stops there anyway, so that snapshot costs one
    write and no forward pass."""
    estimated_gb: float
    """KV plus recurrent state at the expected final length. Feeds the guard."""


class Admitter(Protocol):
    def plan(self, request: Request) -> AdmissionPlan: ...

    def start(self, plan: AdmissionPlan) -> SequenceState:
        """Allocate state, restore the cached prefix, return a PREFILLING
        sequence. Raises :class:`~titan.core.errors.MemoryGuardError` if the
        guard refuses, having allocated nothing."""


# ---------------------------------------------------------------------------
# chunk planning
# ---------------------------------------------------------------------------


def plan_chunks(
    matched: int,
    total: int,
    *,
    chunk: int = 2048,
    block: int = 512,
    grid: int = 2048,
) -> tuple[int, ...]:
    """Chunk end positions for the uncached suffix ``[matched, total)``.

    Four properties, in the order they constrain the answer:

    1. no chunk is longer than ``chunk`` tokens;
    2. no chunk steps over a snapshot-grid multiple, so every grid multiple
       strictly inside the suffix is itself a chunk end and stages a snapshot;
    3. the prompt end rounded down to the block grid is a chunk end of its own,
       because that is where the terminal snapshot has to sit: a restore point
       must be a block end, and a snapshot staged at an unaligned prompt end
       would be rounded away by the store and dropped;
    4. an end that is not one of those is clamped down to the block grid, so a
       stored block never straddles a chunk boundary.

    The last end is ``total`` whatever the grids say, which is the boundary
    that makes a follow-up turn resume where the previous one actually stopped.
    """
    if matched < 0 or total < matched:
        raise ValueError(f"bad suffix: matched={matched} total={total}")
    if chunk <= 0 or block <= 0 or grid <= 0:
        raise ValueError("chunk, block and grid must be positive")
    fine = (total // block) * block
    ends: list[int] = []
    position = matched
    while position < total:
        next_grid = (position // grid + 1) * grid
        end = min(position + chunk, next_grid, total)
        if position < fine < end:
            end = fine
        if end < total:
            aligned = (end // block) * block
            if aligned > position:
                end = aligned
        if end <= position:  # pragma: no cover - defensive
            raise ValueError("chunk planner made no progress")
        ends.append(end)
        position = end
    return tuple(ends)


def snapshot_positions(
    ends: Sequence[int],
    *,
    grid: int = 2048,
    block: int = 512,
    prompt_end: bool = True,
) -> tuple[int, ...]:
    """Which chunk ends stage a recurrent snapshot.

    The grid multiples, plus the prompt end rounded down to the block grid.
    That last one is the cheap one and the valuable one: the chunk planner
    already stops there, so it costs a write and no forward pass, and it is the
    boundary that stops a six-turn conversation recomputing 11,617 tokens it
    has already seen.

    Rounding down is what makes it legal rather than merely useful. A restore
    point has to be a block end, because the KV either covers whole blocks or
    the chain hash of every block after it changes, so a snapshot staged at an
    unaligned prompt end is a snapshot the store rounds away and drops. The
    tokens past the block floor are recomputed next turn, which is a far better
    trade than falling back to the previous grid multiple.
    """
    if not ends:
        return ()
    at = [end for end in ends if end % grid == 0]
    if prompt_end:
        fine = (ends[-1] // block) * block
        # Only a chunk end can stage a snapshot: the backend stages at the end
        # of a forward and nowhere else. The planner puts a cut at this exact
        # position, so this is a check rather than a hope.
        if fine and fine in set(ends) and fine not in at:
            at.append(fine)
    return tuple(sorted(set(at)))


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


class MemoryGuard:
    """Admission-only resident-memory gate.

    There is exactly one question this class answers: would starting this
    request take resident memory over the guard. It has no method that shrinks,
    pauses or preempts anything, and that absence is the design.
    """

    def __init__(self, config: AdmissionConfig) -> None:
        self.config = config
        self.refusals = 0

    def estimate_gb(self, request: Request) -> float:
        """State cost at the expected final length of ``request``."""
        expected = len(request.prompt_tokens) + request.stop.max_tokens
        bytes_needed = (
            expected * self.config.state_bytes_per_token
            + self.config.state_fixed_bytes
        )
        return bytes_needed / 1e9

    def fits(self, estimated_gb: float, resident_gb: float) -> bool:
        return resident_gb + estimated_gb <= self.config.memory_guard_gb

    def check(self, estimated_gb: float, resident_gb: float) -> None:
        if not self.fits(estimated_gb, resident_gb):
            self.refusals += 1
            raise MemoryGuardError(
                f"admitting this request needs {estimated_gb:.1f} GB on top of "
                f"{resident_gb:.1f} GB resident, over the "
                f"{self.config.memory_guard_gb:.0f} GB guard"
            )


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Waiting:
    request: Request
    sink: Any
    skips: int = 0


class WaitQueue:
    """FCFS with the right to skip an entry that cannot fit yet.

    ``select`` walks the queue in arrival order and returns the first entry the
    caller's predicate accepts. An entry that is passed over keeps its place,
    so ordering is still first come first served among the requests that can
    actually run, and a long prompt waiting for room does not stop the short
    ones behind it. ``skips`` is carried per entry so starvation is visible in
    the metrics rather than inferred from a latency graph.
    """

    def __init__(self, depth: int = 64) -> None:
        self.depth = depth
        self._entries: list[_Waiting] = []
        self.rejected = 0
        self.skipped = 0

    def __len__(self) -> int:
        return len(self._entries)

    def push(self, request: Request, sink: Any = None) -> None:
        if len(self._entries) >= self.depth:
            self.rejected += 1
            raise CapacityError(
                f"admission queue is full ({self.depth} waiting); retry"
            )
        self._entries.append(_Waiting(request=request, sink=sink))

    def select(self, fits: Callable[[Request], bool]) -> _Waiting | None:
        for index, entry in enumerate(self._entries):
            if fits(entry.request):
                return self._entries.pop(index)
            entry.skips += 1
            self.skipped += 1
        return None

    def remove(self, request_id: RequestId) -> _Waiting | None:
        for index, entry in enumerate(self._entries):
            if entry.request.request_id == request_id:
                return self._entries.pop(index)
        return None

    def drain(self) -> list[_Waiting]:
        """Take every waiting entry. Used on shutdown, where each one is owed
        a terminal event: a queued request whose sink is never called is a
        client that waits for a stream that will not start."""
        entries, self._entries = self._entries, []
        return entries

    def max_skips(self) -> int:
        return max((e.skips for e in self._entries), default=0)


# ---------------------------------------------------------------------------
# the admitter
# ---------------------------------------------------------------------------


_NO_MATCH = PrefixMatch(matched_tokens=0, block_hashes=(), snapshot_id=None, tier="none")


class PortAdmitter:
    """:class:`Admitter` over the ports. Owns no policy the scheduler needs.

    The prefill plan covers ``len(prompt) - 1`` tokens, not the whole prompt.
    The last prompt token is the first decode input: the engine may not read
    logits, so the only way to turn the prompt's final position into a token id
    is to run it through the verify forward, which is what the first decode
    cycle does. oMLX arrives at the same split from the other direction, by
    handing only the final prompt token to its generator.
    """

    def __init__(
        self,
        *,
        backend: Any,
        config: AdmissionConfig,
        cache: Any = None,
        clock: Any = None,
        profiler: Any = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.cache = cache
        self.clock = clock
        self.profiler = profiler
        self.guard = MemoryGuard(config)
        self._next_sequence = 1

    # -- planning ----------------------------------------------------------
    def plan(self, request: Request) -> AdmissionPlan:
        total = len(request.prompt_tokens)
        if total == 0:
            raise CapacityError("a request with no prompt tokens cannot be admitted")
        if total + request.stop.max_tokens > self.config.max_context:
            raise CapacityError(
                f"prompt of {total} tokens plus {request.stop.max_tokens} generated "
                f"exceeds the {self.config.max_context}-token context window"
            )
        prefill_end = total - 1
        match = _NO_MATCH
        if self.cache is not None and prefill_end > 0:
            match = self.cache.lookup(request.prompt_tokens[:prefill_end])
        matched = min(match.matched_tokens, prefill_end)

        ends = self._chunk_ends(matched, prefill_end)
        return AdmissionPlan(
            request=request,
            match=match,
            chunk_ends=ends,
            snapshot_at=self._snapshot_at(matched, prefill_end, ends),
            estimated_gb=self.guard.estimate_gb(request),
        )

    def _snapshot_at(
        self, matched: int, prefill_end: int, ends: Sequence[int]
    ) -> tuple[int, ...]:
        """Where snapshots are staged. The cache decides when there is one.

        The cache owns the snapshot policy because it owns the grids and it is
        the only party that knows what the write backlog costs right now; the
        local planner is what answers when there is no cache or the cache
        raises. Either answer is filtered to positions that actually end a
        chunk, because the backend can only stage a snapshot at the end of a
        forward, and a boundary staged nowhere is a boundary the store drops.
        """
        if self.cache is not None:
            try:
                asked = tuple(
                    int(b)
                    for b in self.cache.snapshot_boundaries(matched, prefill_end, False)
                )
            except Exception as exc:  # noqa: BLE001 - a cache fault is not fatal
                self._event("snapshot_plan_failed", reason=str(exc))
            else:
                usable = tuple(sorted({b for b in asked if b in set(ends)}))
                if len(usable) != len(set(asked)):
                    self._event(
                        "snapshot_boundaries_dropped",
                        asked=len(set(asked)),
                        usable=len(usable),
                    )
                return usable
        return snapshot_positions(
            ends,
            grid=self.config.snapshot_grid,
            block=self.config.block_tokens,
        )

    def _chunk_ends(self, matched: int, prefill_end: int) -> tuple[int, ...]:
        """Ask the cache for the plan, then hold it to the grid rule.

        The cache adapter owns chunk planning because it owns the grids. It does
        not own the invariant: a returned plan that steps over a snapshot-grid
        multiple would silently truncate the store chain, so it is checked here
        and replaced by the local planner if it is wrong. Checking is three
        comparisons per chunk against a 2048-token forward pass.
        """
        local = plan_chunks(
            matched,
            prefill_end,
            chunk=self.config.prefill_chunk_tokens,
            block=self.config.block_tokens,
            grid=self.config.snapshot_grid,
        )
        if self.cache is None:
            return local
        try:
            ends = tuple(int(e) for e in self.cache.plan_chunks(matched, prefill_end, False))
        except Exception as exc:  # noqa: BLE001 - a cache fault is not fatal
            self._event("chunk_plan_failed", reason=str(exc))
            return local
        if not _plan_is_valid(ends, matched, prefill_end, self.config):
            self._event("chunk_plan_rejected", ends=len(ends))
            return local
        return ends

    # -- starting ----------------------------------------------------------
    def start(self, plan: AdmissionPlan, resident_gb: float = 0.0) -> SequenceState:
        """Allocate, restore, return a PREFILLING sequence.

        The guard is checked first and nothing is allocated if it refuses, so a
        refusal costs a comparison rather than a slot.
        """
        self.guard.check(plan.estimated_gb, resident_gb)
        request = plan.request
        sequence_id = SequenceId(self._next_sequence)
        self._next_sequence += 1
        capacity = len(request.prompt_tokens) + request.stop.max_tokens
        state = self.backend.open_state(sequence_id, capacity)

        restored = 0
        if self.cache is not None and plan.match.matched_tokens > 0:
            try:
                restored = int(self.cache.restore(plan.match, state))
            except Exception as exc:  # noqa: BLE001 - D9: degrade, never lie
                self._event("prefix_restore_failed", reason=str(exc))
                restored = 0
        return SequenceState(
            sequence_id=sequence_id,
            request=request,
            phase=SequencePhase.PREFILLING,
            state=state,
            tokens=list(request.prompt_tokens),
            prompt_len=len(request.prompt_tokens),
            prefill_position=restored,
            committed=0,
            restored_from=restored,
            admitted_at=self.clock.now() if self.clock is not None else 0.0,
        )

    def _event(self, name: str, **fields: float | int | str) -> None:
        if self.profiler is not None:
            self.profiler.event(name, **fields)


def _plan_is_valid(
    ends: Sequence[int], matched: int, total: int, config: AdmissionConfig
) -> bool:
    """The grid rule, checked rather than trusted."""
    if not ends:
        return matched >= total
    if ends[-1] != total:
        return False
    position = matched
    for end in ends:
        if end <= position or end > total:
            return False
        if end - position > config.prefill_chunk_tokens:
            return False
        next_grid = (position // config.snapshot_grid + 1) * config.snapshot_grid
        if next_grid < end:
            return False
        position = end
    return True
