"""The store path through the loop: when bytes are made, and on whose time.

This is the regression test for the retirement the integration report caught.
A 65k-token request wrote 32 GDN snapshots, 5.49 GB, in 15.4 s, all of it in
one call on the scheduler loop thread at retirement, and the request that
followed decoded at 25 tok/s instead of 55 while the store drained.

The loop here is the real :class:`EngineLoop` over the real
:class:`BlockPrefixCache`, with a codec that charges a manual clock for every
export. Nothing sleeps and nothing races: the clock only moves when a
serialisation happens, so "what did this call cost the loop thread" is exact.

Four properties:

1. a boundary becomes bytes on the prefill turn that reaches it, not later;
2. no single call is longer than one boundary plus the budget, whatever the
   backlog;
3. a retirement that cannot finish hands the state to a drain, which finishes
   it over later turns and then closes the handle;
4. a 65k-token request retires in bounded loop-thread time.
"""

from __future__ import annotations

import pytest

from titan.adapters.cache.format import CacheSignature
from titan.adapters.cache.prefix import BlockPrefixCache
from titan.adapters.cache.store import TwoTierStateStore
from titan.core.errors import StateError
from titan.core.types import FinishReason, StateHandle, StreamEnd, TokenEvent
from titan.engine.admission import AdmissionConfig
from titan.engine.decode_cycle import NullProfiler, PlainDecodeCycle
from titan.engine.scheduler import EngineLoop

from tests.engine.conftest import FakeBackend, FakeTokenizer, make_request

BLOCK = 512
GRID = 2048
CAP_S = 0.050
SNAPSHOT_S = 0.480
"""What one boundary costs the loop thread on the real model: 15.4 s over the
32 snapshots the report measured."""


class ManualClock:
    """Time moves when the codec says it does, and at no other moment."""

    def __init__(self, tick: float = 0.0) -> None:
        self.t = 0.0
        self.tick = tick

    def now(self) -> float:
        self.t += self.tick
        return self.t

    def __call__(self) -> float:
        return self.now()

    def advance(self, seconds: float) -> None:
        self.t += seconds


class HandleCodec:
    """A :class:`StateCodec` over the engine's fake backend.

    It reaches through the backend rather than around it, so the same rule the
    mlx codec lives under holds here: an export only succeeds where the
    backend actually staged a snapshot, and it fails on a closed handle. That
    is what makes the drain test meaningful -- if the loop closed the state
    early, the drain's exports would raise instead of quietly succeeding.
    """

    def __init__(self, backend: FakeBackend, clock: ManualClock, *, snapshot_s: float):
        self._backend = backend
        self._clock = clock
        self._snapshot_s = snapshot_s
        self.signature = CacheSignature(
            model_name="fake",
            layer_layout=("gdn", "qsa"),
            block_tokens=BLOCK,
            snapshot_dtype="fp32",
        )
        self.snapshot_exports: list[int] = []
        self.block_exports: list[tuple[int, int]] = []

    def export_blocks(self, state: StateHandle, start: int, end: int) -> bytes:
        tokens = self._backend.state_tokens(state)
        if len(tokens) < end:
            raise StateError(f"state covers {len(tokens)}, not {end}")
        self.block_exports.append((start, end))
        return bytes(str(tokens[start:end]), "utf-8")

    def import_blocks(
        self, state: StateHandle, start: int, end: int, payload: bytes
    ) -> None:  # pragma: no cover - these tests never restore
        raise AssertionError("the store path does not read")

    def export_snapshot(self, state: StateHandle, length: int) -> bytes:
        blob = self._backend.export_snapshot(state, length)
        self.snapshot_exports.append(length)
        self._clock.advance(self._snapshot_s)
        return blob

    def import_snapshot(
        self, state: StateHandle, length: int, payload: bytes
    ) -> None:  # pragma: no cover - these tests never restore
        raise AssertionError("the store path does not read")


def build(
    *,
    prompt_len: int,
    snapshot_s: float = SNAPSHOT_S,
    cap_s: float = CAP_S,
    max_tokens: int = 2,
    drain_max_s: float = 60.0,
):
    """A loop over the real cache, sized for a prompt of ``prompt_len``."""
    clock = ManualClock()
    backend = FakeBackend(max_context=prompt_len + 4096)
    codec = HandleCodec(backend, clock, snapshot_s=snapshot_s)
    store = TwoTierStateStore(codec.signature, hot_budget_bytes=64 * 1024**2)
    cache = BlockPrefixCache(
        store,
        codec,
        block_tokens=BLOCK,
        snapshot_grid=GRID,
        chunk_tokens=GRID,
        contended_chunk_tokens=BLOCK,
        store_budget_s=cap_s,
        clock=clock,
    )
    config = AdmissionConfig(
        prefill_chunk_tokens=GRID,
        block_tokens=BLOCK,
        snapshot_grid=GRID,
        max_context=prompt_len + 4096,
        memory_guard_gb=10_000.0,
        weights_gb=0.0,
    )
    tokenizer = FakeTokenizer()
    loop = EngineLoop(
        backend=backend,
        tokenizer=tokenizer,
        cycle=PlainDecodeCycle(
            backend=backend, tokenizer=tokenizer, profiler=NullProfiler()
        ),
        config=config,
        cache=cache,
        clock=clock,
        store_drain_max_s=drain_max_s,
    )
    return loop, backend, cache, codec, clock, store


class Sink:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)

    @property
    def end(self) -> StreamEnd | None:
        return next((e for e in self.events if isinstance(e, StreamEnd)), None)


def prompt_of(length: int) -> list[int]:
    return [(index * 7 + 3) % 60 + 1 for index in range(length)]


def run(loop: EngineLoop, sink: Sink, *, limit: int = 400) -> list[float]:
    """Step until the request ends, recording what each turn cost the store."""
    costs: list[float] = []
    for _ in range(limit):
        spent_before = loop._store_seconds
        loop.step()
        costs.append(loop._store_seconds - spent_before)
        if sink.end is not None and not loop._store_drains:
            break
    return costs


# -- 1. boundaries become bytes during prefill ------------------------------


def test_a_boundary_is_serialised_on_the_turn_that_reaches_it():
    prompt_len = 8 * GRID + 100
    loop, _backend, cache, codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        # The first turn admits and runs a chunk; every turn after it runs
        # one more, and each grid multiple is bytes by the end of its turn.
        seen: list[int] = []
        for _ in range(9):
            loop.step()
            seen.append(len(codec.snapshot_exports))
        assert seen[0] == 1
        assert seen == sorted(seen)
        assert seen[-1] >= 8
        # One a turn, never a burst.
        assert max(b - a for a, b in zip(seen, seen[1:])) <= 1
        assert cache.counters.snapshots_written >= 8
    finally:
        store.close()


def test_retirement_serialises_the_tail_and_not_the_prompt():
    """The whole point: retirement is O(tail), not O(prompt).

    Eight grid boundaries during prefill, and at most the prompt-end one left
    for the turn the sequence retires on.
    """
    prompt_len = 8 * GRID
    loop, backend, cache, codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        retirement_exports = None
        for _ in range(200):
            before = len(codec.snapshot_exports)
            loop.step()
            if sink.end is not None:
                # The turn the sequence retired on, drain included.
                retirement_exports = len(codec.snapshot_exports) - before
                break
        assert retirement_exports is not None
        assert retirement_exports <= 1, "retirement is the tail, not the prompt"
        assert len(codec.snapshot_exports) >= 8
    finally:
        store.close()


# -- 2. the cap -------------------------------------------------------------


def test_no_single_turn_costs_more_than_one_boundary_plus_the_budget():
    """A snapshot cannot be cut in half, so one is the floor of a call.

    What the cap buys is that a call is never *more* than that: the thirty-two
    the report measured in one call become one a turn, and the sequence
    sharing the loop gets a decode cycle between each of them.
    """
    prompt_len = 12 * GRID
    loop, _backend, cache, _codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        costs = run(loop, sink)
        assert max(costs) <= SNAPSHOT_S + CAP_S
        assert cache.counters.max_store_stall_s <= SNAPSHOT_S + CAP_S
        # Twelve boundaries at 480 ms is 5.8 s of work, and no turn saw more
        # than half a second of it.
        assert cache.counters.store_seconds > 10 * SNAPSHOT_S
    finally:
        store.close()


def test_a_cheap_codec_chains_boundaries_up_to_the_budget_and_stops():
    """With a unit under the cap, the budget is what ends the call."""
    prompt_len = 20 * GRID
    loop, _backend, cache, codec, _clock, store = build(
        prompt_len=prompt_len, snapshot_s=0.008
    )
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        costs = run(loop, sink)
        assert max(costs) <= CAP_S
        assert cache.counters.max_store_stall_s <= CAP_S
        assert len(codec.snapshot_exports) >= 20
    finally:
        store.close()


# -- 3. the post-retirement drain -------------------------------------------


def test_a_deferred_store_finishes_after_retirement_and_then_closes_the_state():
    """A retirement that cannot finish hands the handle to a drain.

    The drain owns the state until the last boundary is bytes, which is why
    the handle is still open while it runs: ``export_snapshot`` reads it, and
    a codec asked for a closed handle raises.
    """
    prompt_len = 4 * GRID
    loop, backend, cache, codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        for _ in range(200):
            loop.step()
            live = loop.live
            if live and live[0].committed >= 1:
                break
        # Retire by hand, and stop before the pump, which is the only moment
        # a drain is observable from outside a turn: a whole ``step`` creates
        # one and finishes it.
        sequence = loop.live[0]
        loop._finish(sequence, FinishReason.STOP)
        loop._retire()
        assert loop._store_drains, "the prompt-end boundary should have deferred"
        entry = loop._store_drains[0]
        assert int(entry.state) in backend.open_handles
        assert entry.session.pending >= 1
        exports = len(codec.snapshot_exports)

        for _ in range(20):
            loop.step()
            if not loop._store_drains:
                break
        assert not loop._store_drains
        assert len(codec.snapshot_exports) > exports
        assert cache.counters.drain_snapshots >= 1
        assert cache.counters.boundaries_abandoned == 0
        assert int(entry.state) not in backend.open_handles
        assert sink.end is not None
    finally:
        store.close()


def test_a_drain_that_overruns_its_deadline_is_abandoned_and_the_state_closes():
    """The drain is bounded twice: per call, and in total."""
    prompt_len = 4 * GRID
    loop, backend, cache, _codec, _clock, store = build(
        prompt_len=prompt_len, drain_max_s=0.0
    )
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        for _ in range(200):
            loop.step()
            if sink.end is not None and not loop._store_drains:
                break
        assert not loop._store_drains
        assert not backend.open_handles
    finally:
        store.close()


def test_shutdown_finishes_the_drains_it_can_and_frees_every_handle():
    prompt_len = 4 * GRID
    loop, backend, _cache, _codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        for _ in range(6):
            loop.step()
        loop.shutdown(drain_timeout_s=30.0)
        assert not loop._store_drains
        assert not backend.open_handles
    finally:
        store.close()


# -- 4. the sixty-five thousand token request -------------------------------


@pytest.mark.parametrize("prompt_len", [65 * 1024])
def test_a_65k_request_retires_in_bounded_loop_thread_time(prompt_len):
    """The measured case from the report, with its measured per-snapshot cost.

    Old shape: one call at retirement, 32 snapshots, 15.4 s. New shape: the
    same 32 snapshots and the same total work, spread one to a turn, with no
    turn over half a second and retirement itself under one boundary.
    """
    loop, _backend, cache, codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        costs = run(loop, sink, limit=600)
        assert sink.end is not None
        assert len(codec.snapshot_exports) >= 32
        # No turn cost the loop more than one boundary plus its budget, where
        # the old shape cost one turn 15.4 s.
        assert max(costs) <= SNAPSHOT_S + CAP_S
        # And the work is spread: at least thirty turns did some of it.
        assert sum(1 for cost in costs if cost > 0.0) >= 30
        assert cache.counters.snapshots_written >= 32
        assert cache.counters.chain_truncations == 0
    finally:
        store.close()


def test_the_stall_is_visible_in_the_stats_the_metrics_endpoint_dumps():
    prompt_len = 4 * GRID
    loop, _backend, cache, _codec, _clock, store = build(prompt_len=prompt_len)
    try:
        sink = Sink()
        loop.submit(make_request(prompt_of(prompt_len), max_tokens=2), sink)
        run(loop, sink)
        values = cache.stats()
        assert values["prefix.max_store_stall_s"] > 0.0
        assert values["prefix.store_seconds"] > 0.0
        assert values["prefix.store_pumps"] > 0.0
        assert "store.queue_depth" in values
        assert "store.pending_peak_bytes" in values
        assert loop.store_stall_max_s > 0.0
    finally:
        store.close()


def test_a_cache_without_sessions_keeps_the_one_shot_path():
    """The fallback the engine's own fakes and the benches take."""
    from tests.engine.conftest import FakeCache

    config = AdmissionConfig(prefill_chunk_tokens=GRID, block_tokens=BLOCK)
    cache = FakeCache(config=config)
    backend = FakeBackend()
    tokenizer = FakeTokenizer()
    loop = EngineLoop(
        backend=backend,
        tokenizer=tokenizer,
        cycle=PlainDecodeCycle(
            backend=backend, tokenizer=tokenizer, profiler=NullProfiler()
        ),
        config=config,
        cache=cache,
    )
    sink = Sink()
    loop.submit(make_request([5, 6, 7], max_tokens=2), sink)
    for _ in range(20):
        loop.step()
        if sink.end is not None:
            break
    assert cache.stores, "a cache with no begin_store still gets its one call"
    assert not loop._store_drains
    assert not backend.open_handles


def tokens_of(sink: Sink) -> list[int]:
    out: list[int] = []
    for event in sink.events:
        if isinstance(event, TokenEvent):
            out.extend(event.token_ids)
    return out


def test_the_answer_is_unchanged_by_the_store_path():
    """Serialising during prefill must not touch the token stream."""
    prompt_len = 2 * GRID
    prompt = prompt_of(prompt_len)
    loop, backend, _cache, _codec, _clock, store = build(
        prompt_len=prompt_len, max_tokens=6
    )
    try:
        sink = Sink()
        loop.submit(make_request(prompt, max_tokens=6), sink)
        run(loop, sink)
        assert tokens_of(sink) == backend.greedy_continuation(prompt, 6)
    finally:
        store.close()
