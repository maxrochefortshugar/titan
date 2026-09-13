"""Scheduler contracts and the loop itself.

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

## Threads and queues

Nothing here is async. The asyncio side (``titan.engine.engine``) hands work in
through one thread-safe command queue and gets events back through a per-request
sink callable, which is the only thing the loop knows about the caller. That
keeps the MLX stream owned by exactly one thread, and it keeps a decode cycle
from being able to yield between draft and verify, which is the property the
cycle exists for.

    asyncio                     scheduler thread
      submit()  --> commands -->  drain -> admit
      cancel()  --> commands -->  drain -> retire
      queue     <-- sink     <--  emit

## The seam for mixed batches

Prefill and decode are separate phases here, per D14, and the split is one
``if`` in :meth:`EngineLoop.step`. A mixed forward -- prefill rows and decode
rows in the same block -- is the obvious thing an engine that owns its forward
can do, and it is not obviously right on this model: batched prefill loses the
gathered sparse-attention arm and materialises a 134 MB mask per QSA layer at
65k. The seam is marked ``MIXED BATCH SEAM`` in ``step``; taking it means
building one call that carries both, and nothing above ``step`` would change.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

from titan.core.errors import CapacityError, TitanError
from titan.core.types import (
    FinishReason,
    PrefillChunk,
    Request,
    RequestId,
    SequenceId,
    SequencePhase,
    SequenceState,
    StreamEnd,
    TokenEvent,
)

from titan.engine.admission import AdmissionConfig, AdmissionPlan, PortAdmitter, WaitQueue
from titan.engine.decode_cycle import MonotonicClock, NullProfiler

__all__ = [
    "SchedulerPolicy",
    "Scheduler",
    "LoopStats",
    "EngineHealth",
    "FifoPolicy",
    "EngineLoop",
]

logger = logging.getLogger(__name__)

#: Returned by :meth:`EngineLoop._guarded` when the port call raised. A
#: sentinel rather than ``None`` because plenty of port calls return ``None``
#: on success and the caller has to tell the two apart.
FAILED = object()


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
    port_failures: int = 0
    failed_requests: int = 0
    healthy: bool = True
    unhealthy_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EngineHealth:
    """Whether the engine is serving, and why not when it is not.

    ``healthy`` is False only for a fault the engine cannot serve through: a
    backend that failed its warm-up probe twice running, a device error, or a
    loop step the watchdog caught over its budget. A request that dies on its
    own bad luck leaves the engine healthy and takes only itself down.
    """

    healthy: bool
    reason: str | None = None
    port_failures: int = 0
    failed_requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "reason": self.reason,
            "port_failures": self.port_failures,
            "failed_requests": self.failed_requests,
        }


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


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


class FifoPolicy:
    """First come first served, with a seat count and a row budget.

    The guard itself lives in :class:`~titan.engine.admission.MemoryGuard`; this
    policy asks it. What this class adds is the seat count and the rule that a
    request which does not fit is passed over rather than waited on, which the
    queue implements and the loop drives.
    """

    def __init__(self, *, config: AdmissionConfig, guard: Any, rows_budget: int = 32) -> None:
        self.config = config
        self.guard = guard
        self._rows_budget = rows_budget

    def admit(
        self, request: Request, live: Sequence[SequenceState], resident_gb: float
    ) -> bool:
        if len(live) >= self.config.max_sequences:
            return False
        return self.guard.fits(self.guard.estimate_gb(request), resident_gb)

    def next_prefill(self, live: Sequence[SequenceState]) -> PrefillChunk | None:
        for sequence in live:
            if sequence.phase is SequencePhase.PREFILLING:
                return PrefillChunk(
                    sequence_id=sequence.sequence_id,
                    start=sequence.prefill_position,
                    end=sequence.prefill_position,
                    emit_snapshot=False,
                    is_last=False,
                )
        return None

    def decode_batch(self, live: Sequence[SequenceState]) -> tuple[SequenceId, ...]:
        return tuple(
            s.sequence_id for s in live if s.phase is SequencePhase.DECODING
        )

    def rows_budget(self, n_sequences: int) -> int:
        return self._rows_budget


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


@dataclass
class _StoreDrain:
    """A retired sequence whose store has not finished serialising.

    The state handle stays open until the last boundary is bytes, because
    ``export_snapshot`` reads it. That is device memory held past the answer,
    so it is bounded twice: by the per-call budget, which keeps each pump
    short, and by ``deadline``, past which the remaining boundaries are
    abandoned and the handle closes regardless.
    """

    sequence_id: int
    session: Any
    state: Any
    tokens: list[int]
    covered: int
    estimated_gb: float
    deadline: float


@dataclass(frozen=True, slots=True)
class _Submit:
    request: Request
    sink: Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class _Cancel:
    request_id: RequestId


# ---------------------------------------------------------------------------
# classifying a fault
# ---------------------------------------------------------------------------


_DEVICE_MARKERS = ("metal", "command buffer", "gpu", "device error", "out of memory")


def _is_device_error(exc: BaseException) -> bool:
    """Whether a fault is the device rather than the request.

    Matched on the exception's own module and on a short list of markers,
    because the framework does not export an exception type to catch: an MLX
    Metal fault arrives as a ``RuntimeError`` with the driver's message in it.
    A false positive costs an engine that stops admitting until it is
    restarted, so the markers are narrow and the module test is exact.
    """
    module = type(exc).__module__.split(".")[0]
    if module in {"mlx", "metal"}:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _DEVICE_MARKERS)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


class EngineLoop:
    """The single engine thread. Owns the sequences and the MLX stream.

    Constructed by ``titan.config.wiring`` and driven either by
    :meth:`run_forever` on its own thread or, in tests, one :meth:`step` at a
    time from the caller's thread. Both are the same code: there is no test-only
    path through the loop.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        cycle: Any,
        config: AdmissionConfig | None = None,
        admitter: Any = None,
        policy: Any = None,
        cache: Any = None,
        clock: Any = None,
        profiler: Any = None,
        rows_budget: int = 32,
        max_admissions_per_turn: int = 4,
        store_drain_max_s: float = 2.0,
        step_budget_s: float = 120.0,
        watchdog_poll_s: float = 0.5,
        probe_failures_to_unhealthy: int = 2,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.cycle = cycle
        self.config = config or AdmissionConfig()
        self.clock = clock or MonotonicClock()
        self.profiler = profiler or NullProfiler()
        self.cache = cache
        self.admitter = admitter or PortAdmitter(
            backend=backend,
            config=self.config,
            cache=cache,
            clock=self.clock,
            profiler=self.profiler,
        )
        self.policy = policy or FifoPolicy(
            config=self.config, guard=self.admitter.guard, rows_budget=rows_budget
        )
        self.max_admissions_per_turn = max_admissions_per_turn

        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        # Set the moment ``run_forever`` claims a thread, cleared when it lets
        # go. A runner waits on it at start so that a stop arriving in the
        # window between the two can never drain the loop from a second thread.
        self._running = threading.Event()
        self._waiting = WaitQueue(depth=self.config.queue_depth)
        self._live: list[SequenceState] = []
        self._plans: dict[int, AdmissionPlan] = {}
        self._sinks: dict[str, Callable[[Any], None]] = {}
        self._pending_error: dict[int, str | None] = {}
        self._cancelled: set[str] = set()
        self._leases: dict[int, Any] = {}

        # -- failure handling -----------------------------------------------
        # Everything the loop thread calls into a port goes through
        # ``_guarded``, so a fault is a finished request rather than a dead
        # thread. What is counted here is what ``/metrics`` and ``/health``
        # report; see the failure-handling contract in docs/architecture.
        self._healthy = True
        self._unhealthy_reason: str | None = None
        self._port_failures = 0
        self._failed_requests = 0
        self._probe_failures = 0
        self.probe_failures_to_unhealthy = max(1, int(probe_failures_to_unhealthy))

        # -- the watchdog ----------------------------------------------------
        # A step that never returns cannot report itself, so the budget is
        # checked from a second thread against a wall clock. The injected clock
        # is not used: a test clock that ticks per read would make the budget a
        # count of reads rather than a duration.
        self.step_budget_s = float(step_budget_s)
        self.watchdog_poll_s = max(0.001, float(watchdog_poll_s))
        self._step_started: float | None = None
        self._watchdog_fired = False
        self._loop_thread_id: int | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self.last_stall_stack: str | None = None

        # -- the store path -------------------------------------------------
        # One session per live sequence, opened at admission. The loop tells it
        # about each boundary as prefill reaches it and pumps it once a cycle,
        # so retirement finds the tail already serialised instead of the whole
        # prompt. A cache that offers no ``begin_store`` keeps the one-shot
        # path, which is what the engine's own fakes use.
        self._store_sessions: dict[int, Any] = {}
        self._store_drains: list[_StoreDrain] = []
        self._store_budget_s = float(getattr(cache, "store_budget_s", 0.020) or 0.0)
        self._store_drain_max_s = max(float(store_drain_max_s), 0.0)
        self._store_stall_max_s = 0.0
        self._store_seconds = 0.0

        self._turns = 0
        self._prefill_chunks = 0
        self._decode_cycles = 0
        self._tokens_out = 0
        self._admitted = 0
        self._rejected = 0
        self._rows = 0

    # -- the asyncio-facing surface ---------------------------------------
    def submit(self, request: Request, sink: Callable[[Any], None]) -> None:
        """Queue a request. Thread-safe, non-blocking, never runs a forward.

        Refusal is asynchronous by construction: the queue depth is checked on
        the loop thread, so a full queue arrives as a ``StreamEnd`` carrying
        ``FinishReason.ERROR`` rather than as an exception on the caller's
        thread. The API layer turns that into a 429 the same way it would an
        exception, and the loop keeps one owner for the queue.
        """
        self._commands.put(_Submit(request=request, sink=sink))
        self._wake.set()

    def cancel(self, request_id: RequestId) -> None:
        """Abort a request. The state handle closes at the next turn boundary."""
        self._commands.put(_Cancel(request_id=request_id))
        self._wake.set()

    # -- failure handling --------------------------------------------------
    def health(self) -> EngineHealth:
        """Whether the engine is serving. Read by ``GET /health``."""
        return EngineHealth(
            healthy=self._healthy,
            reason=self._unhealthy_reason,
            port_failures=self._port_failures,
            failed_requests=self._failed_requests,
        )

    def mark_unhealthy(self, reason: str) -> None:
        """Stop admitting. The first reason wins, because it is the cause.

        Nothing here tears the loop down: a loop that is still turning can
        still cancel, still drain and still answer the requests it already
        holds, and an engine that kills its own thread is the hang this whole
        path exists to prevent.
        """
        if self._healthy:
            self._healthy = False
            self._unhealthy_reason = reason
            logger.error("engine marked unhealthy: %s", reason)
            self.profiler.event("engine_unhealthy", reason=reason)
            self._count("engine.unhealthy")

    def mark_healthy(self) -> None:
        """Clear the flag. Only an operator action should call this."""
        self._healthy = True
        self._unhealthy_reason = None
        self._probe_failures = 0

    def _count(self, name: str, amount: int = 1) -> None:
        counter = getattr(self.profiler, "count", None)
        if callable(counter):
            counter(name, amount)

    def _guarded(
        self,
        op: str,
        sequences: Sequence[SequenceState],
        fn: Callable[..., Any],
        *args: Any,
        probe: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Call one port. Return its value, or :data:`FAILED` if it raised.

        This is the whole fix in one method. Every call the loop thread makes
        out of the engine goes through it, so the blast radius of an exception
        is the sequences named in ``sequences`` and nothing else: they finish
        with an error, their state and their lease are released by the ordinary
        retirement path, and the turn carries on with everybody else.

        ``TitanError`` and a bare ``RuntimeError`` are treated identically on
        purpose. The taxonomy says what a fault means; it has never said
        whether the loop survives it, and the two rounds lost to this bug were
        both a port raising something outside the taxonomy.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the point of the method
            self._port_failed(op, sequences, exc, probe=probe)
            return FAILED

    def _port_failed(
        self,
        op: str,
        sequences: Sequence[SequenceState],
        exc: BaseException,
        *,
        probe: bool = True,
    ) -> None:
        message = f"{op} failed: {type(exc).__name__}: {exc}"
        self._port_failures += 1
        logger.error(
            "port call %s failed for %d sequence(s)",
            op,
            len(sequences),
            exc_info=exc,
        )
        self.profiler.event(
            "port_failed",
            op=op,
            kind=type(exc).__name__,
            reason=str(exc),
            sequences=len(sequences),
        )
        self._count("engine.port_failures")
        self._count(f"engine.port_failures.{op}")
        for sequence in sequences:
            self._failed_requests += 1
            self._finish(sequence, FinishReason.ERROR, error=message)
        if _is_device_error(exc):
            self.mark_unhealthy(f"device error in {op}: {exc}")
            return
        if probe:
            self._run_probe()

    def _run_probe(self) -> None:
        """Ask the backend whether it is still there.

        A warm-up is the cheapest call the port has that still touches the
        device, so it is the probe. One failure is a bad moment; two in a row
        with no success between them is a backend that is not coming back, and
        the engine stops taking work rather than failing every request it
        admits one at a time.
        """
        warm = getattr(self.backend, "warmup", None)
        if not callable(warm):
            return
        try:
            warm()
        except Exception as exc:  # noqa: BLE001 - the probe is the diagnosis
            self._probe_failures += 1
            self.profiler.event(
                "warmup_probe_failed",
                reason=str(exc),
                consecutive=self._probe_failures,
            )
            self._count("engine.probe_failures")
            if self._probe_failures >= self.probe_failures_to_unhealthy:
                self.mark_unhealthy(
                    f"warm-up probe failed {self._probe_failures} times in a row: {exc}"
                )
        else:
            self._probe_failures = 0

    def _fail_all(self, reason: str) -> None:
        """Fail every live sequence with the same error. Used when the fault
        is the loop's own rather than one request's."""
        for sequence in list(self._live):
            self._failed_requests += 1
            self._finish(sequence, FinishReason.ERROR, error=reason)

    def on_step_error(self, exc: BaseException) -> bool:
        """A step raised anyway. Fail everything it was holding and carry on.

        Nothing inside ``step`` is supposed to reach here: every port call is
        guarded and the engine's own code is not allowed to raise. It is here
        because the cost of being wrong about that is a thread that stops and
        clients that wait forever, and the cost of the belt is one try block.
        """
        message = f"engine loop step failed: {type(exc).__name__}: {exc}"
        logger.error("engine loop step raised", exc_info=exc)
        self.profiler.event("loop_step_failed", kind=type(exc).__name__, reason=str(exc))
        self._count("engine.step_failures")
        self._port_failures += 1
        self.mark_unhealthy(message)
        self._fail_all(message)
        try:
            self._retire()
        except Exception:  # noqa: BLE001 - retirement is the last thing left
            logger.exception("retirement failed after a loop step error")
        return True

    # -- the watchdog --------------------------------------------------------
    def check_step_deadline(self, now: float | None = None) -> bool:
        """Fire the watchdog if the step in flight has run past its budget.

        Called from the watchdog thread, and directly from tests, which is why
        it takes the clock as an argument rather than reading one. Fires once
        per step: a stuck step should produce one stack and one incident, not
        one every poll.
        """
        started = self._step_started
        if started is None or self.step_budget_s <= 0.0 or self._watchdog_fired:
            return False
        elapsed = (time.monotonic() if now is None else now) - started
        if elapsed < self.step_budget_s:
            return False
        self._watchdog_fired = True
        self.last_stall_stack = self._loop_stack()
        logger.error(
            "engine loop step has been running for %.1fs (budget %.1fs); "
            "loop thread stack:\n%s",
            elapsed,
            self.step_budget_s,
            self.last_stall_stack,
        )
        self.profiler.event("loop_stalled", seconds=elapsed, budget=self.step_budget_s)
        self._count("engine.loop_stalls")
        self.mark_unhealthy(
            f"loop step exceeded its {self.step_budget_s:g}s budget "
            f"(running for {elapsed:.1f}s)"
        )
        return True

    def _loop_stack(self) -> str:
        """The loop thread's stack, from the thread that noticed the stall.

        ``sys._current_frames`` rather than ``faulthandler.dump_traceback``:
        the same information, addressed to one thread, returned as a string a
        log line and a test can both hold. faulthandler writes every thread to
        a file descriptor, which is the wrong shape for both.
        """
        thread_id = self._loop_thread_id
        if thread_id is None:
            return "(the loop is not running on a thread of its own)"
        frame = sys._current_frames().get(thread_id)
        if frame is None:
            return f"(no frame for loop thread {thread_id})"
        return "".join(traceback.format_stack(frame))

    def _start_watchdog(self) -> None:
        if self._watchdog is not None or self.step_budget_s <= 0.0:
            return
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(
            target=self._watch, name="titan-engine-watchdog", daemon=True
        )
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        self._watchdog = None

    def _watch(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_poll_s):
            if self._stopping.is_set() and self._step_started is None:
                return
            self.check_step_deadline()

    # -- one turn ----------------------------------------------------------
    def step(self) -> bool:
        """Run one turn. Returns True if the turn did any work.

        The return value is what a runner uses to decide whether to sleep, and
        it is the whole scheduling contract between the loop and whoever owns
        its thread.
        """
        self._turns += 1
        # The watchdog reads this. Set before any work and cleared in the
        # finally, so the window it measures is exactly one step.
        self._step_started = time.monotonic()
        self._watchdog_fired = False
        try:
            worked = self._drain_commands()
            worked |= self._admit_ready()

            # MIXED BATCH SEAM. Prefill and decode are separate phases (D14).
            # One call that carried both would replace this branch and nothing
            # above it.
            chunk = self._next_prefill_chunk()
            if chunk is not None:
                self._run_prefill(chunk)
                worked = True
            else:
                worked |= self._run_decode()

            worked |= self._retire()
            # Last, so a boundary the prefill above just staged is serialised
            # on the same turn it was reached, and so the budget is spent on
            # what is left after the forward rather than in front of it.
            worked |= self._pump_stores()
            return worked
        finally:
            self._step_started = None

    def run_forever(self) -> None:
        """Own the calling thread until :meth:`shutdown`.

        The thread outlives every fault it can survive. That is the contract
        the asyncio side depends on: a client is waiting on a queue that only
        this thread feeds, so a thread that dies is a request that never
        finishes and a connection that never closes.
        """
        self._loop_thread_id = threading.get_ident()
        self._running.set()
        self._start_watchdog()
        try:
            while not self._stopping.is_set():
                try:
                    worked = self.step()
                except Exception as exc:  # noqa: BLE001 - the thread survives
                    worked = self.on_step_error(exc)
                if not worked:
                    # Nothing to do. Block on the wake event rather than spin,
                    # so an idle engine costs no CPU and a submit is picked up
                    # at once.
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
        finally:
            self._stop_watchdog()
            self._loop_thread_id = None
            self._running.clear()

    def wait_until_running(self, timeout_s: float = 1.0) -> bool:
        """Block until :meth:`run_forever` owns a thread, or the timeout ends.

        Answers whether it happened rather than raising: a loop that has not
        started is a slow start on some paths and a stub runner on others, and
        neither is the caller's to diagnose.
        """
        return self._running.wait(timeout=max(0.0, timeout_s))

    def shutdown(self, drain_timeout_s: float = 5.0) -> None:
        """Stop the loop and free every sequence, draining what it can.

        Bounded by ``drain_timeout_s`` against a wall clock on every path,
        including the one where a step is wedged inside a port call. When the
        loop thread does not come back in time, the sequences it holds are
        still answered -- an ``ABORT`` ``StreamEnd`` each, so no client is left
        on a queue nobody will feed -- and their state handles are left alone,
        because closing a handle under a thread that is still using it trades a
        hang for a crash.
        """
        budget = max(0.0, drain_timeout_s)
        deadline = time.monotonic() + budget
        self._stopping.set()
        self._wake.set()

        owner = self._loop_thread_id
        if owner is not None and owner != threading.get_ident():
            # The loop owns a thread of its own. Wait for it to notice rather
            # than stepping in parallel with it: two threads in one turn is a
            # worse failure than a slow shutdown.
            while self._loop_thread_id is not None and time.monotonic() < deadline:
                time.sleep(min(0.005, self.watchdog_poll_s))
            if self._loop_thread_id is not None:
                self._abandon_to_stuck_loop()
                return
        else:
            while self._live and time.monotonic() < deadline:
                try:
                    self.step()
                except Exception as exc:  # noqa: BLE001 - shutdown finishes
                    self.on_step_error(exc)
                    break

        for sequence in list(self._live):
            self._finish(sequence, FinishReason.ABORT, error="engine shutting down")
        self._retire()
        self._drain_waiting("engine shutting down")
        # Whatever a retirement deferred still owns a state handle. Give it the
        # rest of the drain window, then take the handles back.
        while self._store_drains and time.monotonic() < deadline:
            self._pump_stores()
        for entry in list(self._store_drains):
            self._guarded("store.abandon", (), entry.session.abandon, probe=False)
            self._guarded(
                "store.finish", (), entry.session.finish, entry.tokens, entry.covered,
                probe=False,
            )
            if entry.state is not None:
                self._guarded(
                    "backend.close_state", (), self.backend.close_state, entry.state,
                    probe=False,
                )
        self._store_drains.clear()

    def _abandon_to_stuck_loop(self) -> None:
        """The drain window closed with the loop thread still inside a step.

        Everything waiting on this engine is answered here and the loop thread
        is left to whatever it is doing. It is a daemon thread, so it cannot
        hold the process open, and nothing after this point touches state it
        might still be reading.
        """
        logger.error(
            "engine shutdown timed out with a loop step still running; "
            "abandoning %d live sequence(s)",
            len(self._live),
        )
        self.profiler.event("shutdown_abandoned", live=len(self._live))
        self._count("engine.shutdown_abandoned")
        self.mark_unhealthy("shut down while a loop step was still running")
        for sequence in list(self._live):
            self._emit_for(
                sequence.request.request_id,
                StreamEnd(
                    request_id=sequence.request.request_id,
                    finish_reason=FinishReason.ABORT,
                    prompt_tokens=sequence.prompt_len,
                    cached_tokens=sequence.restored_from,
                    completion_tokens=sequence.committed,
                    error="engine shutting down while this sequence was in flight",
                ),
            )
            self._sinks.pop(str(sequence.request.request_id), None)
        self._drain_waiting("engine shutting down")

    def _drain_waiting(
        self, reason: str, finish_reason: FinishReason = FinishReason.ABORT
    ) -> int:
        """Answer every queued request that never got a seat.

        Called on shutdown, and again the first turn after the engine goes
        unhealthy: a request that was queued a moment before the fault is owed
        the same answer as one that arrives a moment after it, and the only
        thing that would otherwise reach it is the loop stopping.
        """
        entries = self._waiting.drain()
        for entry in entries:
            self._emit_to(
                entry.sink,
                StreamEnd(
                    request_id=entry.request.request_id,
                    finish_reason=finish_reason,
                    prompt_tokens=len(entry.request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error=reason,
                ),
            )
        return len(entries)

    def stats(self) -> LoopStats:
        return LoopStats(
            turns=self._turns,
            prefill_chunks=self._prefill_chunks,
            decode_cycles=self._decode_cycles,
            tokens_out=self._tokens_out,
            admitted=self._admitted,
            rejected=self._rejected,
            mean_rows_per_cycle=(
                self._rows / self._decode_cycles if self._decode_cycles else 0.0
            ),
            queue_depth=len(self._waiting),
            resident_gb=self.resident_gb(),
            port_failures=self._port_failures,
            failed_requests=self._failed_requests,
            healthy=self._healthy,
            unhealthy_reason=self._unhealthy_reason,
        )

    @property
    def live(self) -> tuple[SequenceState, ...]:
        return tuple(self._live)

    # -- resident memory ---------------------------------------------------
    def resident_gb(self) -> float:
        """Modelled resident bytes: the weights plus every live sequence.

        Modelled rather than measured because the measurement that matters
        (``mx.get_active_memory`` plus the Mach footprint) is MLX-thread-only
        and lives in the adapter. A backend that offers ``resident_gb`` is asked
        instead, which is how the real one reports the truth.
        """
        probe = getattr(self.backend, "resident_gb", None)
        if callable(probe):
            # Guarded like any other port call, and with nobody to fail: the
            # answer feeds admission, so a probe that raises falls back to the
            # model rather than taking the turn down with it.
            measured = self._guarded("backend.resident_gb", (), probe, probe=False)
            if measured is not FAILED:
                return float(measured)
        total = self.config.weights_gb
        for sequence in self._live:
            plan = self._plans.get(int(sequence.sequence_id))
            if plan is not None:
                total += plan.estimated_gb
        # A draining store still holds its state handle, so the memory is
        # still resident and admission has to see it.
        total += sum(entry.estimated_gb for entry in self._store_drains)
        return total

    # -- turn stages -------------------------------------------------------
    def _drain_commands(self) -> bool:
        worked = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            worked = True
            if isinstance(command, _Submit):
                self._accept(command)
            elif isinstance(command, _Cancel):
                self._cancel_now(command.request_id)
        return worked

    def _accept(self, command: _Submit) -> None:
        if not self._healthy:
            # Refused here rather than queued, so the caller learns now. The
            # API layer turns this into a 503; see the failure-handling
            # contract in docs/architecture/ENGINE.md.
            self._rejected += 1
            self._count("engine.refused_unhealthy")
            self._emit_to(
                command.sink,
                StreamEnd(
                    request_id=command.request.request_id,
                    finish_reason=FinishReason.ERROR,
                    prompt_tokens=len(command.request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error=f"engine unhealthy: {self._unhealthy_reason}",
                ),
            )
            return
        try:
            self._waiting.push(command.request, command.sink)
        except CapacityError as exc:
            self._rejected += 1
            self._emit_to(
                command.sink,
                StreamEnd(
                    request_id=command.request.request_id,
                    finish_reason=FinishReason.ERROR,
                    prompt_tokens=len(command.request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error=str(exc),
                ),
            )

    def _cancel_now(self, request_id: RequestId) -> None:
        entry = self._waiting.remove(request_id)
        if entry is not None:
            self._emit_to(
                entry.sink,
                StreamEnd(
                    request_id=request_id,
                    finish_reason=FinishReason.ABORT,
                    prompt_tokens=len(entry.request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                ),
            )
            return
        for sequence in self._live:
            if sequence.request.request_id == request_id:
                self._finish(sequence, FinishReason.ABORT)
                return
        self._cancelled.add(str(request_id))

    def _admit_ready(self) -> bool:
        """Start as many waiting requests as fit, up to the per-turn bound.

        Bounded because admission calls ``open_state`` and the cache restore,
        and a turn that admits eight long prompts is a turn that does not
        decode. The bound is the only fairness knob the loop has and it is
        deliberately small.
        """
        if not self._healthy:
            # Nothing is admitted while the engine is unhealthy, and nothing is
            # left queued either. Done here, on the loop thread, rather than in
            # ``mark_unhealthy``, which the watchdog thread also calls: the
            # wait queue belongs to this thread and is not going to start
            # having a lock put on it for a failure path.
            drained = self._drain_waiting(
                f"engine unhealthy: {self._unhealthy_reason}", FinishReason.ERROR
            )
            self._rejected += drained
            return drained > 0

        worked = False
        for _ in range(self.max_admissions_per_turn):
            if not len(self._waiting):
                break
            resident = self.resident_gb()
            entry = self._guarded(
                "policy.admit",
                (),
                self._waiting.select,
                lambda request: self.policy.admit(request, self._live, resident),
                probe=False,
            )
            if entry is FAILED:
                # A policy that raises cannot be asked again this turn, and the
                # queue is intact: nothing was taken off it. The waiting
                # requests keep their place and the engine is marked unhealthy,
                # because a scheduler that cannot decide who runs is not
                # serving even if every sequence already admitted finishes.
                self.mark_unhealthy("the admission policy raised")
                break
            if entry is None:
                break
            if str(entry.request.request_id) in self._cancelled:
                self._cancelled.discard(str(entry.request.request_id))
                continue
            worked = True
            self._start(entry.request, entry.sink)
        return worked

    def _start(self, request: Request, sink: Callable[[Any], None]) -> None:
        try:
            plan = self.admitter.plan(request)
            sequence = self.admitter.start(plan, self.resident_gb())
        except TitanError as exc:
            # The request is refused, which is an answer rather than a fault:
            # too long, no room, no prompt. Anything else is handled below as a
            # port failure, because an admitter that raises a KeyError is a
            # broken engine and not a bad request.
            self._rejected += 1
            self._emit_to(
                sink,
                StreamEnd(
                    request_id=request.request_id,
                    finish_reason=FinishReason.ERROR,
                    prompt_tokens=len(request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error=str(exc),
                ),
            )
            return
        except Exception as exc:  # noqa: BLE001 - admission is a port call
            self._rejected += 1
            self._failed_requests += 1
            self._port_failed("admission", (), exc)
            self._emit_to(
                sink,
                StreamEnd(
                    request_id=request.request_id,
                    finish_reason=FinishReason.ERROR,
                    prompt_tokens=len(request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error=f"admission failed: {type(exc).__name__}: {exc}",
                ),
            )
            return
        self._admitted += 1
        self._live.append(sequence)
        self._plans[int(sequence.sequence_id)] = plan
        opener = getattr(self.cache, "begin_store", None)
        if callable(opener):
            session = self._guarded(
                "cache.begin_store", (), opener, probe=False
            )
            if session is not FAILED:
                self._store_sessions[int(sequence.sequence_id)] = session
        self._take_lease(sequence, plan)
        self._sinks[str(request.request_id)] = sink
        self.profiler.event(
            "admitted",
            sequence=int(sequence.sequence_id),
            prompt_tokens=sequence.prompt_len,
            cached_tokens=sequence.restored_from,
            chunks=len(plan.chunk_ends),
        )
        # The prompt-end boundary the prefill plan cannot reach. Prefill stops
        # one token short of the prompt, so its deepest snapshot is the block
        # floor of ``prompt_len - 1``. When the prompt length is itself a block
        # multiple, that floor is a whole block short of the prompt end, and
        # the first decode cycle is the only place the missing one can be
        # staged: it is the cycle that consumes the last prompt token, and it
        # lands the state exactly on the block boundary.
        #
        # Only then. A restore point has to be a block end, so the store rounds
        # every boundary down to the grid, and a snapshot staged at an
        # unaligned prompt end is one the store looks for at the rounded
        # position, does not find, and drops.
        block = self.config.block_tokens
        aligned = sequence.prompt_len > 0 and sequence.prompt_len % block == 0
        sequence.needs_prompt_end_snapshot = aligned and sequence.prompt_len not in set(
            plan.snapshot_at
        )
        if not plan.chunk_ends:
            # Everything the prefill would have covered is already in the state:
            # a one-token prompt, or a full cache hit on the prefill prefix.
            sequence.phase = SequencePhase.DECODING

    # -- cache leases ------------------------------------------------------
    def _take_lease(self, sequence: SequenceState, plan: AdmissionPlan) -> None:
        """Pin the prefix this sequence was restored from, if the cache offers
        leases. A cache without ``reserve`` keeps the old behaviour."""
        reserve = getattr(self.cache, "reserve", None)
        if not callable(reserve) or plan.match.matched_tokens <= 0:
            return
        lease = self._guarded("cache.reserve", (), reserve, plan.match, probe=False)
        if lease is not FAILED and lease is not None:
            self._leases[int(sequence.sequence_id)] = lease

    def _release_lease(self, sequence_id: int) -> None:
        """Drop a lease on every exit path, error and abort included.

        A lease that outlives its sequence pins its prefix for the life of the
        process, so this is called from retirement rather than from the happy
        path, and it is guarded because a cache that raises here must not stop
        the sequence from retiring.
        """
        lease = self._leases.pop(sequence_id, None)
        if lease is None:
            return
        release = getattr(self.cache, "release", None)
        if callable(release):
            self._guarded("cache.release", (), release, lease, probe=False)

    def _next_prefill_chunk(self) -> PrefillChunk | None:
        if not self._live:
            return None
        chunk = self._guarded(
            "schedule.next_prefill", tuple(self._live), self._pick_prefill_chunk,
            probe=False,
        )
        return None if chunk is FAILED else chunk

    def _pick_prefill_chunk(self) -> PrefillChunk | None:
        for sequence in self._live:
            if sequence.phase is not SequencePhase.PREFILLING:
                continue
            plan = self._plans[int(sequence.sequence_id)]
            start = sequence.prefill_position
            end = next((e for e in plan.chunk_ends if e > start), None)
            if end is None:
                sequence.phase = SequencePhase.DECODING
                continue
            return PrefillChunk(
                sequence_id=sequence.sequence_id,
                start=start,
                end=end,
                emit_snapshot=end in plan.snapshot_at,
                is_last=end == plan.chunk_ends[-1],
            )
        return None

    def _run_prefill(self, chunk: PrefillChunk) -> None:
        sequence = self._sequence(chunk.sequence_id)
        if sequence is None:  # pragma: no cover - retired between stages
            return
        started = self.clock.now()
        # want_logits stays False even on the last chunk. The engine may not
        # read logits, and the prompt's final token is the first decode
        # input, so the head would be run for an answer nobody consumes --
        # 48 to 95 ms per chunk of it.
        outcome = self._guarded(
            "backend.prefill",
            (sequence,),
            self.backend.prefill,
            sequence.state,
            sequence.tokens[chunk.start : chunk.end],
            want_logits=False,
            snapshot=chunk.emit_snapshot,
            # The token after the chunk. Prefill does not consume it --
            # the decode invariant leaves the last one pending -- but a
            # backend that folds the prompt into a draft head's cache
            # needs it to close the chunk's last pair. Always in range:
            # prefill plans stop one token short of the prompt.
            next_token=(
                sequence.tokens[chunk.end]
                if chunk.end < len(sequence.tokens)
                else None
            ),
            # How much of the sequence is still ahead of this chunk. Only
            # the scheduler knows, because the scheduler is what cut the
            # prompt up, and a backend priming only the prompt's tail
            # cannot tell a middle chunk from the last one without it.
            tokens_after=max(0, len(sequence.tokens) - chunk.end),
        )
        if outcome is FAILED:
            # The sequence is already finishing with an error and its state is
            # released at the end of this turn. Every other sequence keeps its
            # place in the loop, which is the whole point.
            return
        sequence.prefill_position = chunk.end
        self._prefill_chunks += 1
        if chunk.emit_snapshot:
            # Reached, not yet bytes. The pump at the end of this turn is what
            # serialises it, and it is the same turn, so the state slice is
            # the one the forward just produced.
            session = self._store_sessions.get(int(sequence.sequence_id))
            if session is not None:
                session.note_boundary(chunk.end)
        self.profiler.event(
            "prefill_chunk",
            sequence=int(sequence.sequence_id),
            start=chunk.start,
            end=chunk.end,
            snapshot=int(chunk.emit_snapshot),
            ms=(self.clock.now() - started) * 1000.0,
        )
        if chunk.is_last:
            sequence.phase = SequencePhase.DECODING
            # The prompt is in, and the backend cannot work that out on its
            # own: prefill plans stop one token short, so the last chunk's
            # ``tokens_after`` is one rather than zero. A backend watching for
            # zero never closes, which is exactly what happened to ROUND5
            # step 2's first pass.
            close = getattr(self.backend, "close_prefill", None)
            if close is not None:
                self._guarded("backend.close_prefill", (sequence,), close)
            # Resident bytes either side of the prompt, if the backend
            # sampled them. ROUND4 could not say why a kernel that fires
            # only on prefill-shaped calls costs a 64k decode, and named
            # this sample as the missing experiment.
            memory = getattr(self.backend, "prefill_memory", None)
            if memory:
                self.profiler.event(
                    "prefill_done",
                    sequence=int(sequence.sequence_id),
                    tokens=chunk.end,
                    **{str(k): float(v) for k, v in memory.items()},
                )

    def _run_decode(self) -> bool:
        if not self._live:
            # Nothing to schedule, so nothing to ask the policy. The early
            # return is what stops a policy that raises from becoming a turn
            # that always claims work: the failure would fail no sequences,
            # report progress, and the loop would spin on it at full tilt.
            return False
        ids = self._guarded(
            "policy.decode_batch", tuple(self._live), self.policy.decode_batch,
            self._live, probe=False,
        )
        if ids is FAILED:
            # Every sequence the policy was asked about is finishing with an
            # error now. Report no work: retirement is what the turn has left
            # to do and it reports for itself.
            return False
        if not ids:
            return False
        wanted = {int(i) for i in ids}
        batch = [s for s in self._live if int(s.sequence_id) in wanted]
        # One guarded call for the whole cycle, and the blast radius is the
        # batch. Everything the cycle touches -- the drafter, the n-gram index,
        # verify, the tokenizer, the codec -- raises through here, so a fault
        # in any of them fails the sequences that shared the forward and
        # nobody else. A batch of one fails one request; a batch of eight fails
        # eight, which is the price of sharing a forward and is stated in the
        # contract rather than discovered.
        result = self._guarded("cycle.run", tuple(batch), self.cycle.run, batch)
        if result is FAILED:
            return True
        self._decode_cycles += 1
        self._rows += result.profile.n_rows
        for event in result.events:
            self._tokens_out += len(event.token_ids)
            self._emit_for(event.request_id, event)
        finished = set(int(s) for s in result.finished)
        for sequence in batch:
            if int(sequence.sequence_id) in finished:
                self._finish(sequence, sequence.finish_reason or FinishReason.STOP)
        return True

    # -- the store path ----------------------------------------------------
    def _pump_stores(self) -> bool:
        """Spend the store budget: live sequences first, then the drains.

        The budget is per request per cycle. Two sequences with work queued
        each get their own, because the alternative is a shared pot that the
        first sequence in the list always empties.

        Nothing here can be interrupted once it is inside the codec, so the
        cap is enforced by not entering: a session starts a boundary only when
        the measured cost of its last one still fits.
        """
        worked = False
        budget = self._store_budget_s
        for sequence in self._live:
            session = self._store_sessions.get(int(sequence.sequence_id))
            if session is None or sequence.state is None or not session.pending:
                continue
            worked = True
            self._spend(
                session,
                sequence.tokens,
                sequence.state,
                budget=budget,
                sequence_id=int(sequence.sequence_id),
                # One boundary a cycle, whatever the estimate says, and then
                # as many more as the budget allows. A snapshot cannot be
                # serialised in halves, and the cycle that staged it is the
                # cheapest moment it will ever have: the alternative to paying
                # here is paying for all thirty-two of them at retirement,
                # which is the 15.4 s the report measured.
                force_one=True,
            )
        for entry in list(self._store_drains):
            worked = True
            self._spend(
                entry.session,
                entry.tokens,
                entry.state,
                budget=budget,
                sequence_id=entry.sequence_id,
                force_one=True,
            )
            expired = self.clock.now() >= entry.deadline
            if entry.session.pending and not expired:
                continue
            if entry.session.pending:
                self.profiler.event(
                    "store_drain_abandoned",
                    sequence=entry.sequence_id,
                    boundaries=entry.session.pending,
                )
                self._guarded(
                    "store.abandon", (), entry.session.abandon, probe=False
                )
            self._guarded(
                "store.finish", (), entry.session.finish, entry.tokens,
                entry.covered, probe=False,
            )
            self._store_drains.remove(entry)
            if entry.state is not None:
                self._guarded(
                    "backend.close_state", (), self.backend.close_state,
                    entry.state, probe=False,
                )
            self.profiler.event("store_drained", sequence=entry.sequence_id)
        return worked

    def _spend(
        self,
        session: Any,
        tokens: Sequence[int],
        state: Any,
        *,
        budget: float,
        sequence_id: int,
        force_one: bool,
    ) -> float:
        try:
            spent = float(
                session.pump(tokens, state, budget_s=budget, force_one=force_one)
            )
        except Exception as exc:  # noqa: BLE001 - a store fault never kills a turn
            # The sequence survives a store fault: what it loses is the cache
            # entry, not its answer. Counted like any other port failure, with
            # no sequence named, which is what "degrade, never lie" costs here.
            self._port_failed("cache.store", (), exc, probe=False)
            self.profiler.event("store_failed", sequence=sequence_id, reason=str(exc))
            self._guarded("store.abandon", (), session.abandon, probe=False)
            return 0.0
        self._store_seconds += spent
        self._store_stall_max_s = max(self._store_stall_max_s, spent)
        if spent > 0.0:
            self.profiler.event(
                "store_pump",
                sequence=sequence_id,
                ms=spent * 1000.0,
                pending=session.pending,
            )
        return spent

    @property
    def store_stall_max_s(self) -> float:
        """The longest one pump held the loop thread, over the process.

        The number the integration report measured at 1.50 s. It belongs to
        the loop rather than to the cache because it is loop time, and the
        cache counts its own copy for ``/metrics``.
        """
        return self._store_stall_max_s

    # -- finishing ---------------------------------------------------------
    def _finish(
        self,
        sequence: SequenceState,
        reason: FinishReason,
        *,
        error: str | None = None,
    ) -> None:
        if sequence.phase is SequencePhase.DRAINING:
            return
        sequence.finish_reason = reason
        sequence.phase = SequencePhase.DRAINING
        self._pending_error[int(sequence.sequence_id)] = error

    def _retire(self) -> bool:
        """Close out every DRAINING sequence: flush text, store, free state.

        Retirement is its own stage, at a turn boundary, because closing a state
        handle in the middle of a cycle is how a cancelled sequence takes the
        rest of the batch with it.
        """
        draining = [s for s in self._live if s.phase is SequencePhase.DRAINING]
        if not draining:
            return False
        for sequence in draining:
            sid = int(sequence.sequence_id)
            error = self._pending_error.pop(sid, None)
            reason = sequence.finish_reason or FinishReason.STOP
            tail = ""
            flushed = self._guarded(
                "tokenizer.flush", (), self.cycle.release, sequence, probe=False
            )
            if flushed is FAILED:
                self.profiler.event("detok_flush_failed", sequence=sid)
            else:
                tail = flushed
            if tail and reason is not FinishReason.ABORT:
                self._emit_for(
                    sequence.request.request_id,
                    TokenEvent(
                        request_id=sequence.request.request_id,
                        token_ids=(),
                        text=tail,
                        timestamp=self.clock.now(),
                    ),
                )
            deferred = False
            if reason is not FinishReason.ABORT and reason is not FinishReason.ERROR:
                deferred = self._store_prefix(sequence)
            else:
                self._store_sessions.pop(sid, None)
            if deferred:
                # A drain owns the state handle now and closes it when the
                # last boundary is bytes. The sequence itself is done: it
                # emits, it retires, it holds no seat.
                sequence.state = None
            elif sequence.state is not None:
                self._guarded(
                    "backend.close_state", (), self.backend.close_state,
                    sequence.state, probe=False,
                )
                sequence.state = None
            # Before the terminal event, so a client that reconnects on it
            # cannot race the prefix it just released.
            self._release_lease(sid)
            sequence.phase = SequencePhase.DONE
            self._emit_for(
                sequence.request.request_id,
                StreamEnd(
                    request_id=sequence.request.request_id,
                    finish_reason=reason,
                    prompt_tokens=sequence.prompt_len,
                    cached_tokens=sequence.restored_from,
                    completion_tokens=sequence.committed,
                    error=error,
                ),
            )
            self.profiler.event(
                "retired",
                sequence=sid,
                reason=reason.value,
                completion_tokens=sequence.committed,
            )
            self._plans.pop(sid, None)
            self._sinks.pop(str(sequence.request.request_id), None)
        self._live = [s for s in self._live if s.phase is not SequencePhase.DONE]
        return True

    def _store_prefix(self, sequence: SequenceState) -> bool:
        """Hand the finished prefix to the cache, or decline to.

        Returns whether a post-retirement drain took ownership of the state
        handle, in which case the caller must not close it.

        The state covers every token but the pending one, so what is offered is
        ``tokens[:-1]``. If the state length does not agree with that -- a stop
        truncation the backend refused, say -- nothing is stored. D9: a boundary
        that did not commit is dropped, never recorded.

        With a session this is O(the tail): every boundary but the prompt-end
        one was serialised on the cycle that reached it. Without one it is the
        old single call, which is what the engine's fakes and the benches use.
        """
        session = self._store_sessions.pop(int(sequence.sequence_id), None)
        if self.cache is None or sequence.state is None:
            return False
        covered = len(sequence.tokens) - 1
        if covered <= 0:
            return False
        try:
            # What the state actually backs, which is not always the token list.
            # A stop string that reached back into a closed block leaves the
            # state longer than the sequence, and a truncation the backend
            # refused leaves it longer still. Neither makes the earlier blocks
            # wrong: block N is the KV for the same token ids either way. So the
            # offer is trimmed to what the state covers rather than dropped,
            # which is the difference between caching a turn that ended on EOS
            # and caching none of them.
            covered = min(covered, self.backend.state_length(sequence.state))
            if covered <= 0:
                self.profiler.event(
                    "store_skipped",
                    sequence=int(sequence.sequence_id),
                    covered=covered,
                )
                return False
            plan = self._plans.get(int(sequence.sequence_id))
            offered = set(plan.snapshot_at if plan else ())
            if sequence.prompt_end_staged:
                # The one boundary prefill could not reach. The first decode
                # cycle staged it, and it is what lets the next turn of a
                # conversation resume at the end of the prompt this one sent.
                offered.add(sequence.prompt_len)
            # Only boundaries something actually staged. The old habit of
            # appending the covered length offered the cache a position no
            # snapshot sits at, which the store rounds down, fails to export
            # and counts as a truncated chain: work and a counter for nothing.
            boundaries = tuple(sorted(b for b in offered if 0 < b <= covered))
            if session is None:
                self.cache.store(sequence.tokens[:covered], sequence.state, boundaries)
                return False
            return self._finish_session(session, sequence, covered, boundaries)
        except Exception as exc:  # noqa: BLE001 - a store fault never kills a turn
            self._port_failed("cache.store", (), exc, probe=False)
            self.profiler.event(
                "store_failed", sequence=int(sequence.sequence_id), reason=str(exc)
            )
        return False

    def _finish_session(
        self,
        session: Any,
        sequence: SequenceState,
        covered: int,
        boundaries: Sequence[int],
    ) -> bool:
        """Close a session out, deferring whatever does not fit the budget.

        Everything prefill reached is already bytes, so what is normally left
        here is the prompt-end boundary the first decode cycle staged: one
        snapshot and the handful of blocks under it, not thirty-two and 5.49
        GB. If even that overruns, it goes to a drain rather than to the loop.
        """
        sid = int(sequence.sequence_id)
        tokens = list(sequence.tokens[:covered])
        for boundary in boundaries:
            session.note_boundary(min(int(boundary), covered))
        self._spend(
            session,
            tokens,
            sequence.state,
            budget=self._store_budget_s,
            sequence_id=sid,
            force_one=False,
        )
        if not session.pending:
            session.finish(tokens, covered)
            return False
        begin = getattr(session, "begin_drain", None)
        if callable(begin):
            begin()
        plan = self._plans.get(sid)
        self._store_drains.append(
            _StoreDrain(
                sequence_id=sid,
                session=session,
                state=sequence.state,
                tokens=tokens,
                covered=covered,
                estimated_gb=plan.estimated_gb if plan is not None else 0.0,
                deadline=self.clock.now() + self._store_drain_max_s,
            )
        )
        self.profiler.event(
            "store_deferred", sequence=sid, boundaries=session.pending
        )
        return True

    # -- plumbing ----------------------------------------------------------
    def _sequence(self, sequence_id: SequenceId) -> SequenceState | None:
        for sequence in self._live:
            if sequence.sequence_id == sequence_id:
                return sequence
        return None

    def _emit_for(self, request_id: RequestId, event: Any) -> None:
        self._emit_to(self._sinks.get(str(request_id)), event)

    @staticmethod
    def _emit_to(sink: Callable[[Any], None] | None, event: Any) -> None:
        if sink is None:
            return
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - a dead consumer is not a loop fault
            return
