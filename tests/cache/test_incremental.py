"""The incremental store: when a boundary becomes bytes, and what it costs.

The integration report measured the old shape on a real 65k-token request:
32 snapshots, 5.49 GB and 15.4 s of serialisation, all of it at retirement and
all of it on the scheduler loop thread, while the next request decoded at
25 tok/s instead of 55. ``store.max_stall_s`` read 1.50 s against a configured
50 ms cap, because the cap bounded one put and a retirement made thirty-two.

What is asserted here is the shape of the fix rather than the seconds:

* a boundary is serialised on the cycle it is reached, not at retirement;
* one pump never spends more than the cap, however much work is queued;
* what does not fit is deferred, and deferring loses nothing;
* the write queue is bounded in bytes and says what it dropped.

The clock is manual and the slow codec charges it a fixed amount per export,
so every timing assertion here is exact rather than a race with the machine.
"""

from __future__ import annotations

import time

import pytest

from titan.adapters.cache.prefix import BlockPrefixCache
from titan.adapters.cache.store import TwoTierStateStore

from tests.cache.fakes import FakeCodec, FakeState, signature, tokens_for

BLOCK = 8
GRID = 32
CAP_S = 0.050


class ManualClock:
    """Time only moves when a test or a slow codec says it does."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class SlowCodec(FakeCodec):
    """A codec that charges the clock for every byte it produces.

    Snapshots dominate: a recurrent snapshot is 110 MiB against a KV block's
    few hundred KB, so the block cost is set an order of magnitude below the
    snapshot cost rather than to zero.
    """

    def __init__(self, sig, clock: ManualClock, *, snapshot_s: float, block_s: float = 0.0):
        super().__init__(sig)
        self._clock = clock
        self.snapshot_s = snapshot_s
        self.block_s = block_s
        self.snapshot_calls: list[tuple[int, int]] = []
        """``(length asked for, tokens the state covered)`` per export."""

    def export_snapshot(self, state: FakeState, length: int) -> bytes:
        blob = super().export_snapshot(state, length)
        self.snapshot_calls.append((length, state.length))
        self._clock.advance(self.snapshot_s)
        return blob

    def export_blocks(self, state: FakeState, start: int, end: int) -> bytes:
        payload = super().export_blocks(state, start, end)
        self._clock.advance(self.block_s)
        return payload


def build(clock=None, codec=None, **kwargs):
    sig = signature(block_tokens=BLOCK)
    store = TwoTierStateStore(sig, hot_budget_bytes=8 * 1024**2)
    codec = FakeCodec(sig) if codec is None else codec
    cache = BlockPrefixCache(
        store,
        codec,
        block_tokens=BLOCK,
        snapshot_grid=GRID,
        chunk_tokens=GRID,
        contended_chunk_tokens=BLOCK,
        fine_min_gain_tokens=0,
        store_budget_s=CAP_S,
        clock=clock or time.monotonic,
        **kwargs,
    )
    return cache, codec, store


def prefill_with_session(cache, codec, tokens, *, budget_s=None):
    """One prompt, driven the way the scheduler drives it.

    Chunk, stage, note, pump: the pump is what the loop does at the end of the
    turn, so a boundary reached on turn three is bytes on turn three.
    """
    state = FakeState()
    session = cache.begin_store()
    ends = cache.plan_chunks(0, len(tokens), False)
    snaps = set(cache.snapshot_boundaries(0, len(tokens), False))
    position = 0
    spends: list[float] = []
    for end in ends:
        state.prefill(tokens[position:end], snapshot=end in snaps)
        position = end
        if end in snaps:
            session.note_boundary(end)
        spends.append(session.pump(tokens, state, budget_s=budget_s))
    return session, state, spends


# -- boundaries become bytes when they are reached -------------------------


def test_each_boundary_is_serialised_on_the_chunk_that_reaches_it():
    cache, codec, store = build()
    try:
        tokens = tokens_for(96)
        session, _state, spends = prefill_with_session(cache, codec, tokens)
        # Three chunks, three boundaries, and work on every one of them: the
        # old shape did nothing here and everything at retirement.
        assert len(spends) == 3
        assert all(spend >= 0.0 for spend in spends)
        assert cache.counters.snapshots_written == 3
        assert cache.counters.store_pumps == 3
        assert session.pending == 0
    finally:
        store.close()


def test_serialisation_happens_at_the_boundary_not_at_the_end():
    clock = ManualClock()
    sig = signature(block_tokens=BLOCK)
    codec = SlowCodec(sig, clock, snapshot_s=0.001)
    cache, codec, store = build(clock=clock, codec=codec)
    try:
        tokens = tokens_for(96)
        session, _state, _ = prefill_with_session(cache, codec, tokens)
        lengths = [length for length, _covered in codec.snapshot_calls]
        assert lengths == [32, 64, 96]
        # The state was the length of the boundary at every export, which is
        # only true if the export ran on the chunk that produced it.
        assert [covered for _length, covered in codec.snapshot_calls] == lengths
        assert session.pending == 0
    finally:
        store.close()


def test_retirement_is_the_tail_and_not_the_prompt():
    """What is left for the last call after an incremental prefill."""
    cache, codec, store = build()
    try:
        tokens = tokens_for(96)
        session, state, _ = prefill_with_session(cache, codec, tokens)
        exports_before = codec.exports
        # The prompt-end boundary the prefill could not reach, the only thing
        # a retirement normally has to serialise.
        state.prefill(tokens_for(8, seed=3), snapshot=True)
        session.note_boundary(104)
        session.pump(tokens + tokens_for(8, seed=3), state)
        assert codec.exports - exports_before == 1
    finally:
        store.close()


# -- the cap ----------------------------------------------------------------


def test_one_pump_never_spends_more_than_the_cap():
    """Twenty boundaries queued at once against a 50 ms cap and an 8 ms codec.

    The gate is ``spent + estimate > budget``: a boundary is entered only when
    the cost of the last one says it will fit. Nothing can interrupt an
    ``export_snapshot`` once it starts, so not starting it is the only lever
    the loop has.
    """
    clock = ManualClock()
    sig = signature(block_tokens=BLOCK)
    codec = SlowCodec(sig, clock, snapshot_s=0.008, block_s=0.0002)
    cache, codec, store = build(clock=clock, codec=codec)
    try:
        tokens = tokens_for(20 * GRID)
        state = FakeState()
        session = cache.begin_store()
        for index in range(1, 21):
            state.prefill(tokens[(index - 1) * GRID : index * GRID], snapshot=True)
            session.note_boundary(index * GRID)

        spends = []
        while session.pending:
            spends.append(session.pump(tokens, state))
        assert max(spends) <= CAP_S
        assert len(spends) > 1, "a 160 ms burst has to cross more than one cycle"
        assert cache.counters.max_store_stall_s <= CAP_S
        assert cache.counters.snapshots_deferred >= 1
        assert cache.counters.snapshots_written == 20
    finally:
        store.close()


def test_deferred_work_finishes_and_the_prefix_is_whole():
    """Deferring costs cycles, never bytes: the chain is the same either way."""
    clock = ManualClock()
    sig = signature(block_tokens=BLOCK)
    codec = SlowCodec(sig, clock, snapshot_s=0.030)
    cache, codec, store = build(clock=clock, codec=codec)
    try:
        tokens = tokens_for(6 * GRID)
        state = FakeState()
        session = cache.begin_store()
        for index in range(1, 7):
            state.prefill(tokens[(index - 1) * GRID : index * GRID], snapshot=True)
            session.note_boundary(index * GRID)
        assert session.pending == 6
        while session.pending:
            session.pump(tokens, state)
        session.finish(tokens, len(tokens))

        match = cache.lookup(tokens + [7])
        assert match.matched_tokens == 6 * GRID
        restored = FakeState()
        assert cache.restore(match, restored) == 6 * GRID
        assert cache.counters.chain_truncations == 0
    finally:
        store.close()


def test_a_drain_makes_progress_even_when_the_estimate_says_no():
    """``force_one`` is what stops a post-retirement drain from stalling."""
    clock = ManualClock()
    sig = signature(block_tokens=BLOCK)
    codec = SlowCodec(sig, clock, snapshot_s=0.500)  # ten times the cap
    cache, codec, store = build(clock=clock, codec=codec)
    try:
        tokens = tokens_for(2 * GRID)
        state = FakeState()
        session = cache.begin_store()
        for index in (1, 2):
            state.prefill(tokens[(index - 1) * GRID : index * GRID], snapshot=True)
            session.note_boundary(index * GRID)
        # One unit costs ten caps, so the budget alone would never start one
        # after the first, and the state handle would never close.
        session.pump(tokens, state)
        assert session.pending == 1
        session.begin_drain()
        session.pump(tokens, state, force_one=True)
        assert session.pending == 0
        assert cache.counters.drain_snapshots == 1
    finally:
        store.close()


def test_abandoning_a_drain_counts_what_it_dropped():
    cache, _codec, store = build()
    try:
        session = cache.begin_store()
        session.note_boundary(GRID)
        session.note_boundary(2 * GRID)
        session.abandon()
        assert session.pending == 0
        assert cache.counters.boundaries_abandoned == 2
    finally:
        store.close()


# -- the write queue --------------------------------------------------------


def test_the_write_queue_is_bounded_in_bytes_and_says_what_it_dropped(tmp_path):
    """Over budget the queue sheds. It never waits, and it never grows.

    5.49 GB of pending writes is the other half of the report's retirement:
    every queued snapshot is also in the hot tier, so a backlog is counted
    twice until the disk catches up.
    """
    store = TwoTierStateStore(
        signature(),
        ssd_dir=str(tmp_path),
        hot_budget_bytes=1024**2,
        pending_budget_bytes=2500,
        max_stall_s=CAP_S,
    )
    try:
        store.set_writer_paused(True)
        for index in range(8):
            store.put_snapshot(f"s{index}", bytes([index]) * 1000)
        assert store.pending_bytes() <= 2500
        assert store.queue_depth() <= 3
        assert store.stats.writes_dropped >= 5
        assert store.stats.queue_evictions >= 1
        assert store.stats.dropped_bytes >= 5000
        assert store.stats.max_stall_s < CAP_S
        assert store.stats.stall_cap_exceeded == 0
        # Durability is what a drop costs. The bytes are still servable.
        assert store.get_snapshot("s0") == b"\x00" * 1000
        assert store.get_snapshot("s7") == bytes([7]) * 1000
        metrics = store.metrics()
        assert metrics["store.queue_depth"] == float(store.queue_depth())
        assert metrics["store.pending_bytes"] == float(store.pending_bytes())
        assert metrics["store.pending_peak_bytes"] > 0
        store.set_writer_paused(False)
        assert store.flush(2.0)
    finally:
        store.close()


def test_a_pinned_write_is_never_the_victim(tmp_path):
    """A pin means a live lease is matching on it. Drop something else."""
    store = TwoTierStateStore(
        signature(),
        ssd_dir=str(tmp_path),
        hot_budget_bytes=1024**2,
        pending_budget_bytes=2100,
    )
    try:
        store.set_writer_paused(True)
        store.put_snapshot("pinned", b"p" * 1000)
        store.pin_snapshot("pinned")
        store.put_snapshot("loose", b"l" * 1000)
        store.put_snapshot("newest", b"n" * 1000)
        store.set_writer_paused(False)
        assert store.flush(2.0)
        assert store.tier_of("pinned") == "ram"
        assert store.stats.queue_evictions == 1
        # The pinned record is the one that reached the disk index.
        assert store.contains_snapshot("pinned")
    finally:
        store.close()


# -- the cost model ---------------------------------------------------------


def test_the_loop_thread_cost_of_one_boundary_is_recorded():
    """The number CACHE.md's cost model is sized against.

    Tiny arrays, real clock: what this measures is the fixed overhead of a
    boundary, the part that does not scale with the payload. The payload term
    is what the real-model measurement in CACHE.md supplies.
    """
    cache, _codec, store = build()
    try:
        tokens = tokens_for(64 * BLOCK)
        state = FakeState()
        session = cache.begin_store()
        state.prefill(tokens, snapshot=True)
        session.note_boundary(len(tokens))
        spent = session.pump(tokens, state, budget_s=float("inf"))
        per_snapshot_ms = spent * 1000.0
        assert cache.counters.snapshots_written == 1
        assert cache.counters.blocks_written == 64
        # Generous: this is 64 numpy slices and one integer on a shared CI
        # box. The assertion is that the fixed cost is milliseconds and not
        # the 480 ms per snapshot the real model spends copying 171 MB.
        assert per_snapshot_ms < 250.0
        assert cache.counters.store_seconds == pytest.approx(spent, rel=1e-6)
    finally:
        store.close()
