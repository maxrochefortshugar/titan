"""Profiler construction and the shapes it reports.

Layer one and layer two of the three the package docstring describes, in one
object. Layer three, the sampled trace that does sync the device, is a counter
here and nothing more until there is a reason to pay for it.

The whole thing is a ring buffer, a counter dict and a lock. That is not
minimalism for its own sake: every method on this object is called from the
scheduler thread in the middle of a decode cycle, so the budget is a few
microseconds and anything that allocates per call, blocks, or touches the
device would show up directly in the number it is trying to measure. The lock
is uncontended in the normal case (one writer, the scheduler; one reader, an
occasional ``GET /metrics``) and a ``deque`` with a ``maxlen`` drops its oldest
record without copying.

What ``snapshot`` returns is what a benchmark quotes, so the fields are chosen
to make a decode regression attributable without a rerun: a cycle got slower
because it carried fewer rows, because acceptance fell, because a host sync
appeared, or because the n-gram reader made it wait. There is no fifth
possibility that these numbers cannot separate, which is the point.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Mapping

from titan.config.schema import ObservabilityConfig
from titan.core.ports import Clock, Profiler
from titan.core.types import CycleProfile

__all__ = ["DecodeSummary", "RingProfiler", "NullProfiler", "build_profiler"]


@dataclass(frozen=True, slots=True)
class DecodeSummary:
    """Aggregate over a window of cycles, as reported at ``GET /metrics`` and
    written by the bench harness. The fields are chosen so a regression can be
    attributed without a rerun: a decode slowdown is either fewer rows, worse
    acceptance, more host syncs, or n-gram wait."""

    cycles: int
    tokens_per_second: float
    mean_accepted_per_cycle: float
    acceptance_rate: float
    mean_rows: float
    mean_verify_ms: float
    mean_draft_ms: float
    mean_ngram_wait_ms: float
    host_syncs_per_cycle: float
    kernel_fallbacks: int


def _mean(values, count: int) -> float:
    return (values / count) if count else 0.0


class RingProfiler:
    """The real profiler. A ring of cycles, a counter table, and events.

    Events are counted rather than kept. A structured event line per admission
    and per chunk boundary is worth writing to a log; keeping every one of them
    in memory is a leak with a plausible excuse, so the ring holds the last few
    and the counters hold how many there were of each kind.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        ring: int = 4096,
        events_kept: int = 256,
        trace_sample_every: int = 0,
    ) -> None:
        self.clock = clock
        self.trace_sample_every = int(trace_sample_every)
        self._lock = threading.Lock()
        self._cycles: Deque[CycleProfile] = deque(maxlen=max(1, int(ring)))
        self._events: Deque[dict[str, Any]] = deque(maxlen=max(1, int(events_kept)))
        self._event_counts: Counter[str] = Counter()
        self._counters: Counter[str] = Counter()
        self._dropped = 0
        self._started = clock.now()

    # -- the sinks ---------------------------------------------------------
    def cycle(self, profile: CycleProfile) -> None:
        with self._lock:
            self._cycles.append(profile)
            self._counters["cycles"] += 1
            self._counters["tokens_committed"] += profile.tokens_committed
            self._counters["tokens_drafted"] += profile.tokens_drafted
            self._counters["rows"] += profile.n_rows
            self._counters["host_syncs"] += profile.host_syncs
            if profile.host_syncs != 1:
                # The invariant the profile exists to catch. Counted rather than
                # raised: a cycle that synced twice produced correct tokens, it
                # just cost twice what it should have.
                self._counters["host_sync_violations"] += 1

    def event(self, name: str, **fields: float | int | str) -> None:
        with self._lock:
            self._event_counts[name] += 1
            self._events.append({"event": name, "at": self.clock.now(), **fields})

    def span(self, name: str) -> "_Span":
        return _Span(self, name)

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    # -- readers -----------------------------------------------------------
    def summarise(self, window: int = 0) -> DecodeSummary:
        """Aggregate the last ``window`` cycles, or every one held.

        Throughput is committed tokens over the summed wall time of the cycles
        themselves, not over elapsed clock time. The difference is prefill and
        idle, and folding either into a decode rate is how a benchmark ends up
        quoting a number that moves when nothing about decoding changed.
        """
        with self._lock:
            cycles = list(self._cycles)[-window:] if window else list(self._cycles)
            fallbacks = self._counters.get("kernel_fallbacks", 0)
        count = len(cycles)
        if not count:
            return DecodeSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, fallbacks)
        committed = sum(c.tokens_committed for c in cycles)
        drafted = sum(c.tokens_drafted for c in cycles)
        accepted = sum(max(0, c.tokens_committed - c.n_sequences) for c in cycles)
        wall = sum(c.wall_ms for c in cycles) / 1000.0
        return DecodeSummary(
            cycles=count,
            tokens_per_second=(committed / wall) if wall > 0 else 0.0,
            mean_accepted_per_cycle=_mean(accepted, count),
            acceptance_rate=(accepted / drafted) if drafted else 0.0,
            mean_rows=_mean(sum(c.n_rows for c in cycles), count),
            mean_verify_ms=_mean(sum(c.verify_ms for c in cycles), count),
            mean_draft_ms=_mean(sum(c.draft_ms for c in cycles), count),
            mean_ngram_wait_ms=_mean(sum(c.ngram_wait_ms for c in cycles), count),
            host_syncs_per_cycle=_mean(sum(c.host_syncs for c in cycles), count),
            kernel_fallbacks=fallbacks,
        )

    def stage_breakdown(self, window: int = 0) -> Mapping[str, float]:
        """Mean milliseconds per cycle by stage, plus what is left over.

        ``other_ms`` is the wall time no stage claimed, which is Python: the
        commit loop, the emitter, the event plumbing and the interpreter
        between them. It is a residual on purpose. A stage that grew a timer of
        its own would shrink this number without anybody having to explain
        where the time went.
        """
        with self._lock:
            cycles = list(self._cycles)[-window:] if window else list(self._cycles)
        count = len(cycles)
        if not count:
            return {}
        stages = {
            "draft_ms": sum(c.draft_ms for c in cycles),
            "verify_ms": sum(c.verify_ms for c in cycles),
            "accept_ms": sum(c.accept_ms for c in cycles),
            "sample_ms": sum(c.sample_ms for c in cycles),
            "detok_ms": sum(c.detok_ms for c in cycles),
            "ngram_wait_ms": sum(c.ngram_wait_ms for c in cycles),
        }
        wall = sum(c.wall_ms for c in cycles)
        out = {name: total / count for name, total in stages.items()}
        # verify_ms and accept_ms are measured inside the backend's verify call,
        # so they are already part of the wall time rather than on top of it.
        out["wall_ms"] = wall / count
        out["other_ms"] = max(0.0, (wall - sum(stages.values())) / count)
        return out

    def recent_cycles(self, count: int = 20) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(asdict(c) for c in list(self._cycles)[-count:])

    def recent_events(self, count: int = 50) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(list(self._events)[-count:])

    def snapshot(self) -> Mapping[str, Any]:
        """Current counters, for ``GET /metrics`` and the bench harness."""
        with self._lock:
            counters = dict(self._counters)
            events = dict(self._event_counts)
            uptime = self.clock.now() - self._started
        summary = self.summarise()
        return {
            "uptime_s": uptime,
            "counters": counters,
            "events": events,
            "decode": asdict(summary),
            "stages": dict(self.stage_breakdown()),
        }

    def reset(self) -> None:
        """Drop the window. The bench harness calls this between arms."""
        with self._lock:
            self._cycles.clear()
            self._events.clear()
            self._event_counts.clear()
            self._counters.clear()
            self._started = self.clock.now()


class _Span:
    """Host wall time for a named stage. No device sync, ever."""

    __slots__ = ("profiler", "name", "_start")

    def __init__(self, profiler: RingProfiler, name: str) -> None:
        self.profiler = profiler
        self.name = name
        self._start = 0.0

    def __enter__(self) -> "_Span":
        self._start = self.profiler.clock.now()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed_ms = (self.profiler.clock.now() - self._start) * 1000.0
        self.profiler.count(f"span.{self.name}.calls")
        self.profiler.count(f"span.{self.name}.us", int(elapsed_ms * 1000))


class NullProfiler:
    """Accepts everything, keeps nothing. For tests and for a disabled config."""

    def cycle(self, profile: CycleProfile) -> None: ...

    def event(self, name: str, **fields: float | int | str) -> None: ...

    def count(self, name: str, amount: int = 1) -> None: ...

    def span(self, name: str) -> Any:
        return _NullSpan()

    def snapshot(self) -> Mapping[str, Any]:
        return {}


class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: object) -> None: ...


def build_profiler(config: ObservabilityConfig, clock: Clock) -> Profiler:
    """The profiler the composition root installs.

    ``cycle_profile`` off gives the null one. It is a config key rather than a
    hard-coded truth because a machine can be profiled by something else, but
    the default is on: the per-cycle profile is a product feature, and a
    regression that cannot be attributed without a rerun costs more than the
    microseconds this saves.
    """
    if not config.cycle_profile:
        return NullProfiler()
    return RingProfiler(
        clock,
        ring=config.cycle_profile_ring,
        trace_sample_every=config.trace_sample_every,
    )
