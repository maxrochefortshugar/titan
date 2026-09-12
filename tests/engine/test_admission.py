"""Admission: the guard, the queue, and chunk planning.

Three separate claims, all cheap to check and each one a measured failure
somewhere else: the guard gates admission and never running work, a request
that cannot fit is skipped rather than blocked on, and every snapshot-grid
multiple inside a prefill suffix ends a chunk.
"""

from __future__ import annotations

import pytest

from titan.core.errors import CapacityError, MemoryGuardError
from titan.core.types import SequencePhase
from titan.engine.admission import (
    AdmissionConfig,
    MemoryGuard,
    PortAdmitter,
    WaitQueue,
    plan_chunks,
    snapshot_positions,
)

from tests.engine.conftest import FakeCache, FakeClock, make_request


# ---------------------------------------------------------------------------
# chunk planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matched,total,expected",
    [
        (0, 600, (512, 600)),
        (0, 2048, (2048,)),
        (0, 2600, (2048, 2560, 2600)),
        (0, 5000, (2048, 4096, 4608, 5000)),
        (100, 5000, (2048, 4096, 4608, 5000)),
        (2100, 5000, (4096, 4608, 5000)),
        (4096, 4096, ()),
        (0, 4096, (2048, 4096)),
    ],
)
def test_chunks_stop_on_the_grid(matched, total, expected):
    """Grid multiples end a chunk, and so does the block floor of the prompt
    end: that is where the terminal snapshot has to sit, because a restore
    point is a block end and the store rounds an unaligned one away."""
    assert plan_chunks(matched, total) == expected


@pytest.mark.parametrize("total", [1, 7, 511, 512, 513, 2047, 2049, 6000, 65536])
@pytest.mark.parametrize("matched", [0, 1, 511, 512, 2048, 3000])
def test_every_grid_multiple_inside_the_suffix_ends_a_chunk(matched, total):
    """D8's second rule: step over a grid multiple and nothing stages a
    snapshot there, so the store chain truncates at the previous boundary."""
    if matched > total:
        pytest.skip("no suffix")
    ends = plan_chunks(matched, total)
    assert list(ends) == sorted(set(ends))
    if ends:
        assert ends[-1] == total
    position = matched
    for end in ends:
        assert 0 < end - position <= 2048
        position = end
    for multiple in range(2048, total, 2048):
        if multiple > matched:
            assert multiple in ends, f"grid multiple {multiple} is not a chunk end"


def test_ends_that_are_not_the_last_sit_on_the_block_grid():
    ends = plan_chunks(0, 5000, chunk=1024, block=512, grid=2048)
    assert ends == (1024, 2048, 3072, 4096, 4608, 5000)
    for end in ends[:-1]:
        assert end % 512 == 0


def test_snapshots_land_on_the_grid_and_at_the_block_floor_of_the_prompt_end():
    ends = plan_chunks(0, 5000)
    assert snapshot_positions(ends) == (2048, 4096, 4608)
    assert snapshot_positions(plan_chunks(0, 600)) == (512,)
    assert snapshot_positions(()) == ()


def test_an_unaligned_last_end_never_becomes_a_snapshot_position():
    """Only a chunk end can stage one, and only a block end can be restored
    from, so a planner that offered neither gets nothing rather than a boundary
    the store would drop."""
    assert snapshot_positions((600,)) == ()
    assert snapshot_positions((512, 600)) == (512,)
    assert snapshot_positions((2048,)) == (2048,)


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def guard_config(**kwargs) -> AdmissionConfig:
    base = dict(
        memory_guard_gb=110.0,
        weights_gb=78.0,
        state_bytes_per_token=1e6,
        state_fixed_bytes=0.0,
        max_sequences=4,
    )
    base.update(kwargs)
    return AdmissionConfig(**base)


def test_the_guard_refuses_what_does_not_fit():
    guard = MemoryGuard(guard_config())
    request = make_request(tuple(range(1000)), max_tokens=1000)
    assert guard.estimate_gb(request) == pytest.approx(2.0)
    assert guard.fits(2.0, 78.0)
    assert not guard.fits(2.0, 109.0)
    with pytest.raises(MemoryGuardError):
        guard.check(20.0, 100.0)
    assert guard.refusals == 1


def test_the_guard_has_no_way_to_touch_running_work():
    """The absence is the design. A guard that throttles serialises
    concurrency, which held the overlay at a flat 76 tok/s across 1, 2 and 4
    streams."""
    guard = MemoryGuard(guard_config())
    for forbidden in ("preempt", "throttle", "evict", "shrink", "pause"):
        assert not hasattr(guard, forbidden)


def test_the_guard_gates_admission_and_running_sequences_keep_going(backend, clock):
    config = guard_config(state_bytes_per_token=1e7)
    admitter = PortAdmitter(backend=backend, config=config, clock=clock)
    first = admitter.plan(make_request(tuple(range(1000)), request_id="a"))
    started = admitter.start(first, resident_gb=78.0)
    assert started.phase is SequencePhase.PREFILLING
    # The next one does not fit, and refusing it changes nothing about the one
    # already running.
    with pytest.raises(MemoryGuardError):
        admitter.start(admitter.plan(make_request(tuple(range(1000)), request_id="b")), 109.0)
    assert backend.state_length(started.state) == 0
    assert int(started.state) in backend.open_handles


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------


def test_a_request_that_cannot_fit_is_skipped_not_blocked_on():
    """oMLX breaks out of its admission loop at the first blocked request, so a
    long prompt in front holds up everything behind it. This does not."""
    queue = WaitQueue(depth=8)
    big = make_request(tuple(range(100)), request_id="big")
    small_a = make_request((1, 2), request_id="small-a")
    small_b = make_request((1, 2), request_id="small-b")
    for request in (big, small_a, small_b):
        queue.push(request)

    fits = lambda request: len(request.prompt_tokens) < 10
    assert queue.select(fits).request.request_id == "small-a"
    assert queue.select(fits).request.request_id == "small-b"
    assert len(queue) == 1
    assert queue.max_skips() == 2
    # And it starts the moment it fits, still in arrival order.
    assert queue.select(lambda _r: True).request.request_id == "big"


def test_order_is_preserved_among_requests_that_fit():
    queue = WaitQueue(depth=8)
    for index in range(4):
        queue.push(make_request((1,), request_id=f"r{index}"))
    seen = [queue.select(lambda _r: True).request.request_id for _ in range(4)]
    assert seen == ["r0", "r1", "r2", "r3"]


def test_a_full_queue_refuses_rather_than_grows():
    queue = WaitQueue(depth=2)
    queue.push(make_request((1,), request_id="a"))
    queue.push(make_request((1,), request_id="b"))
    with pytest.raises(CapacityError):
        queue.push(make_request((1,), request_id="c"))
    assert queue.rejected == 1


def test_nothing_selected_when_nothing_fits():
    queue = WaitQueue()
    queue.push(make_request((1,), request_id="a"))
    assert queue.select(lambda _r: False) is None
    assert len(queue) == 1


# ---------------------------------------------------------------------------
# the admitter
# ---------------------------------------------------------------------------


def test_the_plan_covers_the_prompt_but_the_last_token(backend, clock):
    """The final prompt token is the first decode input, so prefill stops one
    short of the prompt: the engine cannot read logits, and the only way to turn
    that position into a token id is the verify forward."""
    config = AdmissionConfig()
    admitter = PortAdmitter(backend=backend, config=config, clock=clock)
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert plan.chunk_ends[-1] == 4999
    assert plan.snapshot_at == (2048, 4096, 4608)


def test_a_cache_hit_shortens_the_plan_and_is_reported(backend, clock):
    config = AdmissionConfig()
    cache = FakeCache(matched=2048, config=config)
    admitter = PortAdmitter(backend=backend, config=config, cache=cache, clock=clock)
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert plan.match.matched_tokens == 2048
    assert plan.chunk_ends == (4096, 4608, 4999)
    sequence = admitter.start(plan)
    assert sequence.restored_from == 2048
    assert sequence.prefill_position == 2048
    assert cache.plans == [(2048, 4999)]


def test_a_cache_plan_that_breaks_the_grid_rule_is_replaced(backend, clock):
    """The cache owns chunk planning; it does not own the invariant."""

    class BadCache(FakeCache):
        def plan_chunks(self, matched, total, contended):
            return (total,)  # one giant chunk, straight over every grid point

    config = AdmissionConfig()
    admitter = PortAdmitter(backend=backend, config=config, cache=BadCache(), clock=clock)
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert plan.chunk_ends == (2048, 4096, 4608, 4999)


def test_a_prompt_longer_than_the_window_is_refused(backend, clock):
    config = AdmissionConfig(max_context=1024)
    admitter = PortAdmitter(backend=backend, config=config, clock=clock)
    with pytest.raises(CapacityError):
        admitter.plan(make_request(tuple(range(2000)), max_tokens=10))
    with pytest.raises(CapacityError):
        admitter.plan(make_request((), max_tokens=10))


def test_a_failed_restore_degrades_to_a_cold_prefill(backend, clock):
    """D9: a cache that cannot deliver is a slower turn, never a wrong one."""

    class BrokenCache(FakeCache):
        def restore(self, match, state):
            raise RuntimeError("ssd read failed")

    config = AdmissionConfig()
    cache = BrokenCache(matched=2048, config=config)
    admitter = PortAdmitter(backend=backend, config=config, cache=cache, clock=clock)
    sequence = admitter.start(admitter.plan(make_request(tuple(range(5000)))))
    assert sequence.restored_from == 0
    assert sequence.prefill_position == 0


# ---------------------------------------------------------------------------
# where the snapshot boundaries come from
# ---------------------------------------------------------------------------


def test_the_cache_owns_the_snapshot_policy(backend, clock):
    """The cache knows the grids and the write backlog, so it decides. The
    admitter asks it with the same suffix it planned chunks for."""
    config = AdmissionConfig()
    cache = FakeCache(matched=2048, config=config)
    admitter = PortAdmitter(backend=backend, config=config, cache=cache, clock=clock)
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert cache.boundary_calls == [(2048, 4999, False)]
    assert plan.snapshot_at == (4096, 4608)


def test_a_boundary_that_ends_no_chunk_is_dropped(backend, clock):
    """The backend stages a snapshot at the end of a forward and nowhere else,
    so a boundary in the middle of a chunk is one the store would drop."""

    class OffGridCache(FakeCache):
        def snapshot_boundaries(self, matched, total, contended=False):
            return (2048, 3000, 4608)

    config = AdmissionConfig()
    admitter = PortAdmitter(
        backend=backend, config=config, cache=OffGridCache(), clock=clock
    )
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert plan.snapshot_at == (2048, 4608)


def test_a_cache_that_raises_falls_back_to_the_local_policy(backend, clock):
    class AngryCache(FakeCache):
        def snapshot_boundaries(self, matched, total, contended=False):
            raise RuntimeError("index is rebuilding")

    config = AdmissionConfig()
    admitter = PortAdmitter(
        backend=backend, config=config, cache=AngryCache(), clock=clock
    )
    plan = admitter.plan(make_request(tuple(range(5000)), max_tokens=10))
    assert plan.snapshot_at == (2048, 4096, 4608)


@pytest.mark.parametrize("total", [600, 2049, 5000, 12345, 65536])
def test_every_snapshot_position_ends_a_chunk_and_sits_on_the_block_grid(total):
    ends = plan_chunks(0, total)
    at = snapshot_positions(ends)
    for position in at:
        assert position in ends
        assert position % 512 == 0
