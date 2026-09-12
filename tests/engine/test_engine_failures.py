"""What the engine does when a port raises something outside the taxonomy.

The bug these tests exist for: a backend stage raising a plain ``TypeError``
killed the loop thread, and every request waiting on that loop's queue waited
forever. The contract now is that any exception from any port call on the loop
thread fails exactly the sequences that call was for, releases their state and
their lease, counts the failure, and leaves the loop turning.

Every test here has a bound. ``pytest-timeout`` is not installed in this
environment, so the bound is written into the test: an ``asyncio.timeout``
around anything awaited, and a deadline around anything waited on across a
thread. A regression must fail in seconds rather than sit in CI until somebody
notices, because sitting in CI until somebody notices is exactly what this bug
did twice.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from titan.core.errors import CapacityError, EngineUnhealthyError
from titan.core.types import FinishReason, SequenceId, StreamEnd, TokenEvent
from titan.engine.admission import AdmissionConfig
from titan.engine.decode_cycle import NullProfiler, PlainDecodeCycle
from titan.engine.engine import InlineRunner, ThreadRunner, TitanEngine
from titan.engine.scheduler import EngineLoop

from tests.engine.conftest import (
    FakeBackend,
    FakeCache,
    FakeClock,
    FakeTokenizer,
    make_request,
)

pytestmark = pytest.mark.anyio

#: Every awaited bound in this module. Long enough that a loaded machine does
#: not flake, short enough that a hang is a failed test rather than a stalled
#: run.
BOUND_S = 5.0


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# fakes that break
# ---------------------------------------------------------------------------


class PrefillRaises(FakeBackend):
    """A backend whose prefill raises whatever the test asks for.

    ``only`` limits the fault to one state handle, which is how a test shows
    that a failing sequence takes only itself down.
    """

    def __init__(self, exc: BaseException | None = None, *, only: int | None = None, **kw):
        super().__init__(**kw)
        self.exc = exc or RuntimeError("prefill exploded")
        self.only = only
        self.attempts = 0

    def prefill(self, state, tokens, **kwargs):
        self.attempts += 1
        if self.only is None or int(state) == self.only:
            raise self.exc
        return super().prefill(state, tokens, **kwargs)


class PrefillSignatureDrift(FakeBackend):
    """The exact shape of the reported bug: the port grew a keyword argument
    and this backend did not. Python raises before the body runs."""

    def prefill(self, state, tokens, *, want_logits=False, snapshot=False):
        raise AssertionError("unreachable: the call fails on the signature")


class VerifyRaises(FakeBackend):
    """A backend whose verify raises. Everything the decode cycle does hangs
    off this call, so it stands in for the whole cycle: draft, verify, sample,
    detokenise."""

    def __init__(self, exc: BaseException | None = None, *, after: int = 0, **kw):
        super().__init__(**kw)
        self.exc = exc or RuntimeError("verify exploded")
        self.after = after
        self.calls = 0

    def verify(self, states, drafts, sampling):
        self.calls += 1
        if self.calls > self.after:
            raise self.exc
        return super().verify(states, drafts, sampling)


class SleepyPrefill(FakeBackend):
    """Prefill that does not come back until the test lets it. The watchdog's
    subject: a step that runs past its budget inside a port call."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.entered = threading.Event()
        self.release = threading.Event()

    def prefill(self, state, tokens, **kwargs):
        self.entered.set()
        self.release.wait(timeout=BOUND_S * 4)
        return super().prefill(state, tokens, **kwargs)


class WarmupRaises(FakeBackend):
    """A backend that is not coming back: the probe fails every time."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.warmups = 0

    def prefill(self, state, tokens, **kwargs):
        raise RuntimeError("the forward failed")

    def warmup(self) -> None:
        self.warmups += 1
        raise RuntimeError("the device is gone")


class StoreRaises(FakeCache):
    """A prefix cache whose store fails. The request keeps its answer; what it
    loses is the cache entry."""

    def store(self, tokens, state, boundaries):
        raise RuntimeError("the codec exploded")


class TokenizerRaises(FakeTokenizer):
    """Incremental detokenisation that fails, which happens inside the decode
    cycle and so fails the batch that shared the forward."""

    def decode_incremental(self, seq, ids):
        raise KeyError("no piece for that id")


class FlushRaises(FakeTokenizer):
    """Only the retirement flush fails. The answer is already streamed, so the
    sequence still retires with its real finish reason."""

    def flush_incremental(self, seq):
        raise RuntimeError("the detokeniser died on the way out")


class OneSeatPolicy:
    """Decodes one sequence a turn, so a batch is a proper subset of the live
    set and "the batch fails, nobody else does" has something to mean."""

    def __init__(self, *, config: AdmissionConfig) -> None:
        self.config = config

    def admit(self, request, live, resident_gb):
        return len(live) < self.config.max_sequences

    def decode_batch(self, live):
        for sequence in live:
            if sequence.phase.name == "DECODING":
                return (sequence.sequence_id,)
        return ()

    def rows_budget(self, n_sequences: int) -> int:
        return 32


class DeadRunner:
    """A runner that never turns and admits it. Stands in for a loop thread
    that died where nothing in-process can repair it."""

    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def alive(self) -> bool:
        return False

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        self.started = False


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def build_loop(*, backend=None, tokenizer=None, cache=None, policy=None, **kw):
    backend = backend or FakeBackend()
    tokenizer = tokenizer or FakeTokenizer({})
    clock = FakeClock()
    config = kw.pop("config", None) or AdmissionConfig()
    profiler = kw.pop("profiler", None) or NullProfiler()
    cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)
    loop = EngineLoop(
        backend=backend,
        tokenizer=tokenizer,
        cycle=cycle,
        config=config,
        cache=cache,
        clock=clock,
        profiler=profiler,
        policy=policy(config=config) if policy is not None else None,
        **kw,
    )
    return loop, backend, tokenizer, profiler


def build_engine(**kw):
    loop, backend, tokenizer, profiler = build_loop(**kw)
    return TitanEngine(loop, runner=InlineRunner(loop)), loop, backend, profiler


async def collect(engine, request, *, bound: float = BOUND_S):
    """Drain one stream, or fail the test rather than hang."""
    events = []
    async with asyncio.timeout(bound):
        async for event in engine.generate(request):
            events.append(event)
    return events


def drive(loop: EngineLoop, turns: int = 12) -> None:
    """Step a loop a bounded number of times. Never ``while not done``."""
    for _ in range(turns):
        loop.step()


def sink_collector():
    events: list = []
    return events, events.append


def wait_for(predicate, *, bound: float = BOUND_S, what: str = "the condition") -> None:
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out after {bound:g}s waiting for {what}")


def only_end(events) -> StreamEnd:
    ends = [e for e in events if isinstance(e, StreamEnd)]
    assert len(ends) == 1, f"expected exactly one StreamEnd, got {len(ends)}"
    return ends[0]


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


async def test_a_prefill_that_raises_runtime_error_finishes_the_request():
    engine, loop, backend, profiler = build_engine(
        backend=PrefillRaises(RuntimeError("prefill exploded"))
    )
    events = await collect(engine, make_request((11, 12, 13), max_tokens=6))

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    # The original type and the original text both survive: a message that says
    # "internal error" is a message that costs somebody an afternoon.
    assert "RuntimeError" in end.error
    assert "prefill exploded" in end.error
    assert "backend.prefill" in end.error
    assert backend.open_handles == set()
    assert loop.live == ()
    assert loop.health().healthy is True
    assert loop.stats().failed_requests == 1
    assert profiler.counters["engine.port_failures"] == 1


async def test_the_reported_signature_drift_finishes_rather_than_hangs():
    """The regression, in the shape it was reported in.

    A fake backend whose ``prefill`` predates the ``next_token`` keyword. Before
    the fix this test did not fail; it never returned.
    """
    engine, _loop, _backend, _profiler = build_engine(backend=PrefillSignatureDrift())
    events = await collect(engine, make_request((11, 12, 13), max_tokens=6))

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert "TypeError" in end.error
    assert "next_token" in end.error


async def test_a_failing_prefill_releases_the_cache_lease():
    cache = FakeCache(matched=2048, config=AdmissionConfig())
    engine, _loop, _backend, _profiler = build_engine(
        backend=PrefillRaises(), cache=cache
    )
    events = await collect(engine, make_request(tuple(range(3000)), max_tokens=4))

    assert only_end(events).finish_reason is FinishReason.ERROR
    assert len(cache.leases) == 1
    assert cache.released == cache.leases


async def test_a_failing_prefill_leaves_the_other_request_alone():
    """One sequence dies, the other finishes normally. The blast radius is the
    whole point of the fix."""
    backend = PrefillRaises(only=1)  # the first state handle the backend opens
    engine, _loop, _backend, _profiler = build_engine(backend=backend)

    doomed, healthy = await asyncio.gather(
        collect(engine, make_request((11, 12, 13), request_id="a", max_tokens=5)),
        collect(engine, make_request((21, 22, 23), request_id="b", max_tokens=5)),
    )
    assert only_end(doomed).finish_reason is FinishReason.ERROR
    end = only_end(healthy)
    assert end.finish_reason is FinishReason.LENGTH
    assert end.completion_tokens == 5


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("verify exploded"),
        TypeError("verify() got an unexpected keyword argument 'sampling'"),
        KeyError("sequence"),
    ],
    ids=["runtime", "type", "key"],
)
async def test_a_decode_failure_finishes_the_request_with_an_error(exc):
    engine, loop, backend, _profiler = build_engine(
        backend=VerifyRaises(exc, after=1)
    )
    events = await collect(engine, make_request((11, 12, 13), max_tokens=8))

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert type(exc).__name__ in end.error
    assert "cycle.run" in end.error
    # Whatever it managed to say before the fault is still the client's.
    assert end.completion_tokens >= 1
    assert backend.open_handles == set()
    assert loop.live == ()


async def test_a_batched_decode_failure_fails_only_that_batch():
    """One forward, one batch, one blast radius.

    ``OneSeatPolicy`` decodes a single sequence a turn, so the sequence whose
    cycle raises is a strict subset of the live set and the other one keeps its
    seat.
    """
    backend = VerifyRaises(RuntimeError("the forward failed"), after=2)
    engine, loop, _backend, _profiler = build_engine(
        backend=backend, policy=OneSeatPolicy
    )
    first = asyncio.create_task(
        collect(engine, make_request((11, 12, 13), request_id="a", max_tokens=4))
    )
    async with asyncio.timeout(BOUND_S):
        await first
    # The first sequence took the fault. The engine is still serving, which is
    # what a second request that finishes cleanly proves.
    backend.exc = RuntimeError("no longer raised")
    backend.after = 10_000
    second = await collect(
        engine, make_request((21, 22, 23), request_id="b", max_tokens=4)
    )
    assert only_end(first.result()).finish_reason is FinishReason.ERROR
    assert only_end(second).finish_reason is FinishReason.LENGTH
    assert loop.health().healthy is True


async def test_a_raising_tokenizer_fails_the_batch_and_not_the_loop():
    engine, loop, backend, _profiler = build_engine(tokenizer=TokenizerRaises({}))
    events = await collect(engine, make_request((11, 12, 13), max_tokens=6))

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert "KeyError" in end.error
    assert backend.open_handles == set()
    assert loop.health().healthy is True


async def test_a_flush_that_raises_does_not_swallow_the_finish_reason():
    """Retirement's detokeniser flush is guarded and never changes the answer."""
    engine, _loop, backend, profiler = build_engine(tokenizer=FlushRaises({}))
    events = await collect(engine, make_request((11, 12, 13), max_tokens=4))

    end = only_end(events)
    assert end.finish_reason is FinishReason.LENGTH
    assert end.completion_tokens == 4
    assert backend.open_handles == set()
    assert any(name == "detok_flush_failed" for name, _ in profiler.events)


# ---------------------------------------------------------------------------
# the cache store
# ---------------------------------------------------------------------------


async def test_a_raising_cache_store_keeps_the_answer():
    """Degrade, never lie: the prefix is not cached and the request still ends
    on its real finish reason rather than on an error."""
    cache = StoreRaises(matched=0, config=AdmissionConfig())
    engine, loop, backend, profiler = build_engine(cache=cache)
    events = await collect(engine, make_request((11, 12, 13), max_tokens=4))

    end = only_end(events)
    assert end.finish_reason is FinishReason.LENGTH
    assert end.completion_tokens == 4
    assert backend.open_handles == set()
    assert loop.stats().port_failures >= 1
    assert loop.stats().failed_requests == 0
    assert loop.health().healthy is True


# ---------------------------------------------------------------------------
# health, the probe, and the device
# ---------------------------------------------------------------------------


def test_two_failed_warmup_probes_mark_the_engine_unhealthy():
    loop, backend, _tokenizer, _profiler = build_loop(backend=WarmupRaises())
    first, sink_a = sink_collector()
    second, sink_b = sink_collector()

    loop.submit(make_request((11, 12, 13), request_id="a"), sink_a)
    drive(loop)
    assert only_end(first).finish_reason is FinishReason.ERROR
    # One fault, one probe, and the engine is still nominally serving.
    assert backend.warmups == 1

    loop.submit(make_request((21, 22, 23), request_id="b"), sink_b)
    drive(loop)
    assert only_end(second).finish_reason is FinishReason.ERROR
    assert backend.warmups == 2
    health = loop.health()
    assert health.healthy is False
    assert "warm-up probe failed 2 times" in health.reason


def test_a_device_error_marks_the_engine_unhealthy_at_once():
    """No second chance for the device. One Metal fault is the diagnosis."""
    backend = PrefillRaises(RuntimeError("Metal command buffer execution failed"))
    loop, _backend, _tokenizer, _profiler = build_loop(backend=backend)
    events, sink = sink_collector()

    loop.submit(make_request((11, 12, 13)), sink)
    drive(loop)

    assert only_end(events).finish_reason is FinishReason.ERROR
    health = loop.health()
    assert health.healthy is False
    assert "device error" in health.reason


def test_an_unhealthy_engine_refuses_new_work_rather_than_queueing_it():
    loop, _backend, _tokenizer, _profiler = build_loop()
    loop.mark_unhealthy("the device is gone")
    events, sink = sink_collector()

    loop.submit(make_request((11, 12, 13)), sink)
    drive(loop, turns=4)

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert "engine unhealthy" in end.error
    assert "the device is gone" in end.error
    assert loop.live == ()


def test_a_request_already_queued_when_the_engine_falls_over_is_answered():
    """The one a naive refusal path forgets: queued before the fault, owed an
    answer after it."""
    config = AdmissionConfig(max_sequences=1)
    loop, _backend, _tokenizer, _profiler = build_loop(config=config)
    running, sink_a = sink_collector()
    queued, sink_b = sink_collector()

    loop.submit(make_request((11, 12, 13), request_id="a", max_tokens=64), sink_a)
    loop.submit(make_request((21, 22, 23), request_id="b", max_tokens=64), sink_b)
    drive(loop, turns=3)
    assert queued == []  # still waiting for a seat

    loop.mark_unhealthy("the device is gone")
    drive(loop, turns=2)

    end = only_end(queued)
    assert end.finish_reason is FinishReason.ERROR
    assert "engine unhealthy" in end.error


def test_marking_unhealthy_keeps_the_first_reason():
    loop, _backend, _tokenizer, _profiler = build_loop()
    loop.mark_unhealthy("the cause")
    loop.mark_unhealthy("the consequence")
    assert loop.health().reason == "the cause"
    loop.mark_healthy()
    assert loop.health().healthy is True


def test_a_step_that_raises_anyway_fails_everybody_and_keeps_the_loop():
    """The belt behind the braces. Nothing in ``step`` is supposed to reach
    ``on_step_error``; the cost of being wrong is a dead thread."""
    loop, backend, _tokenizer, _profiler = build_loop()
    events, sink = sink_collector()
    loop.submit(make_request((11, 12, 13), max_tokens=64), sink)
    drive(loop, turns=3)
    assert loop.live != ()

    assert loop.on_step_error(RuntimeError("something nobody guarded")) is True

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert "something nobody guarded" in end.error
    assert loop.health().healthy is False
    assert backend.open_handles == set()


# ---------------------------------------------------------------------------
# the watchdog
# ---------------------------------------------------------------------------


def test_the_watchdog_fires_once_when_a_step_runs_past_its_budget():
    loop, _backend, _tokenizer, profiler = build_loop(step_budget_s=10.0)
    loop._step_started = 1000.0  # a step in flight, as the watchdog reads it
    loop._loop_thread_id = threading.get_ident()

    assert loop.check_step_deadline(now=1005.0) is False
    assert loop.health().healthy is True
    assert loop.check_step_deadline(now=1011.0) is True
    # Once per step: a wedged loop produces one incident, not one per poll.
    assert loop.check_step_deadline(now=1099.0) is False

    health = loop.health()
    assert health.healthy is False
    assert "budget" in health.reason
    assert loop.last_stall_stack is not None
    assert any(name == "loop_stalled" for name, _ in profiler.events)
    assert profiler.counters["engine.loop_stalls"] == 1


def test_a_zero_budget_disables_the_watchdog():
    loop, _backend, _tokenizer, _profiler = build_loop(step_budget_s=0.0)
    loop._step_started = 1000.0
    assert loop.check_step_deadline(now=1e9) is False
    assert loop.health().healthy is True


def test_the_watchdog_thread_catches_a_sleeping_backend():
    """The end-to-end version: a real thread, a real budget, a fake that does
    not come back. Asserts on the captured stack, which is the artefact an
    operator actually reads."""
    backend = SleepyPrefill()
    loop, _backend, _tokenizer, _profiler = build_loop(
        backend=backend, step_budget_s=0.05, watchdog_poll_s=0.01
    )
    events, sink = sink_collector()
    runner = ThreadRunner(loop)
    runner.start()
    try:
        loop.submit(make_request((11, 12, 13)), sink)
        assert backend.entered.wait(timeout=BOUND_S)
        wait_for(
            lambda: not loop.health().healthy,
            what="the watchdog to mark the engine unhealthy",
        )
        assert "budget" in loop.health().reason
        stack = loop.last_stall_stack
        assert stack is not None
        # The stack names the port call the loop is stuck in, which is the
        # difference between a useful incident and "the engine was slow".
        assert "prefill" in stack
    finally:
        backend.release.set()
        runner.stop(drain_timeout_s=BOUND_S)


# ---------------------------------------------------------------------------
# shutdown and cancellation while a sequence is failing
# ---------------------------------------------------------------------------


def test_shutdown_answers_a_sequence_that_is_mid_failure():
    backend = SleepyPrefill()
    loop, _backend, _tokenizer, _profiler = build_loop(
        backend=backend, step_budget_s=0.05, watchdog_poll_s=0.01
    )
    events, sink = sink_collector()
    runner = ThreadRunner(loop)
    runner.start()
    try:
        loop.submit(make_request((11, 12, 13)), sink)
        assert backend.entered.wait(timeout=BOUND_S)
        started = time.monotonic()
        # The loop thread is wedged inside prefill. Stop must still return
        # inside its budget and must still answer the client.
        runner.stop(drain_timeout_s=0.2)
        assert time.monotonic() - started < BOUND_S
        end = only_end(events)
        assert end.finish_reason is FinishReason.ABORT
        assert end.error
    finally:
        backend.release.set()


def test_shutdown_answers_a_request_that_never_got_a_seat():
    config = AdmissionConfig(max_sequences=1)
    loop, _backend, _tokenizer, _profiler = build_loop(config=config)
    running, sink_a = sink_collector()
    queued, sink_b = sink_collector()
    loop.submit(make_request((11, 12, 13), request_id="a", max_tokens=10_000), sink_a)
    loop.submit(make_request((21, 22, 23), request_id="b", max_tokens=10_000), sink_b)
    drive(loop, turns=3)
    assert queued == []

    loop.shutdown(drain_timeout_s=0.0)

    assert only_end(queued).finish_reason is FinishReason.ABORT
    assert only_end(running).finish_reason is FinishReason.ABORT


async def test_stopping_the_engine_ends_streams_that_are_still_open():
    """``runtime.stop`` under the inline runner. The task that drives the loop
    is cancelled, so anything still live has to be answered by the shutdown
    rather than by the next step, which is never coming."""
    engine, _loop, backend, _profiler = build_engine()
    request = make_request((11, 12, 13), max_tokens=10_000)
    stream = engine.generate(request)
    async with asyncio.timeout(BOUND_S):
        first = await stream.__anext__()
    assert isinstance(first, TokenEvent)

    engine.stop(drain_timeout_s=0.2)

    async with asyncio.timeout(BOUND_S):
        rest = [event async for event in stream]
    assert only_end(rest).finish_reason is FinishReason.ABORT
    assert backend.open_handles == set()


async def test_cancelling_the_consumer_during_a_failing_sequence():
    """A cancel that lands while the loop is failing the sequence still
    cancels: the bridge must not turn a disconnect into a completed stream."""
    engine, loop, backend, _profiler = build_engine(
        backend=VerifyRaises(RuntimeError("boom"))
    )
    request = make_request((11, 12, 13), max_tokens=10_000)

    async def consume():
        async for _event in engine.generate(request):
            pass

    # Cancelled while the loop is on its way to failing the same sequence, so
    # the two land in the same handful of ticks. Whichever wins, the consumer
    # must see a cancellation and the engine must let go of the handle.
    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    async with asyncio.timeout(BOUND_S):
        with pytest.raises(asyncio.CancelledError):
            await task
    drive(loop, turns=6)
    assert backend.open_handles == set()
    assert loop.live == ()


# ---------------------------------------------------------------------------
# the asyncio bridge
# ---------------------------------------------------------------------------


async def test_the_bridge_ends_the_stream_when_the_loop_is_gone():
    """The last line of defence. Nothing feeds the queue and nothing ever
    will, so the bridge answers for the loop instead of waiting on it."""
    loop, _backend, _tokenizer, _profiler = build_loop()
    engine = TitanEngine(loop, runner=DeadRunner(), liveness_poll_s=0.01)

    events = await collect(engine, make_request((11, 12, 13)))

    end = only_end(events)
    assert end.finish_reason is FinishReason.ERROR
    assert "the engine loop stopped" in end.error


async def test_generating_against_an_unhealthy_loop_raises_before_the_first_yield():
    """The bridge checks health too. The API layer checks it first, but the
    engine can fall over in the gap, and a raise before the first yield is
    still a status code rather than an error frame."""
    loop, _backend, _tokenizer, _profiler = build_loop()
    loop.mark_unhealthy("the device is gone")
    engine = TitanEngine(loop, runner=InlineRunner(loop), liveness_poll_s=0.01)

    with pytest.raises(EngineUnhealthyError) as caught:
        async with asyncio.timeout(BOUND_S):
            async for _event in engine.generate(make_request((11, 12, 13))):
                pass
    assert "the device is gone" in str(caught.value)
    engine.stop(drain_timeout_s=0.1)


async def test_a_submit_that_raises_reaches_the_caller_as_an_engine_error():
    """Before the first yield there is still a status code to set, so this one
    is allowed to raise rather than becoming a StreamEnd."""

    class RefusingLoop:
        def submit(self, request, sink):
            raise CapacityError("the command queue is full")

        def cancel(self, request_id):
            return None

        def health(self):
            return None

    engine = TitanEngine.__new__(TitanEngine)
    engine.loop = RefusingLoop()
    engine.runner = DeadRunner()
    engine.queue_maxsize = 0
    engine.liveness_poll_s = 0.01
    engine._started = True

    with pytest.raises(Exception) as caught:
        async with asyncio.timeout(BOUND_S):
            async for _event in engine.generate(make_request((11, 12, 13))):
                pass
    assert "the command queue is full" in str(caught.value)


async def test_the_loop_thread_survives_a_fault_and_serves_the_next_request():
    """Production shape: a real thread, a fault, and a request afterwards.

    This is the assertion the whole change is for. Before it, the second
    request never returned, because the thread that would have served it died
    on the first.
    """
    backend = PrefillRaises(RuntimeError("prefill exploded"))
    loop, _backend, _tokenizer, _profiler = build_loop(backend=backend)
    engine = TitanEngine(loop, runner=ThreadRunner(loop), liveness_poll_s=0.05)
    engine.start()
    try:
        doomed = await collect(engine, make_request((11, 12, 13), request_id="a"))
        assert only_end(doomed).finish_reason is FinishReason.ERROR

        backend.exc = None
        backend.only = -1  # no handle matches, so prefill works again
        good = await collect(
            engine, make_request((21, 22, 23), request_id="b", max_tokens=4)
        )
        end = only_end(good)
        assert end.finish_reason is FinishReason.LENGTH
        assert end.completion_tokens == 4
    finally:
        engine.stop(drain_timeout_s=BOUND_S)
