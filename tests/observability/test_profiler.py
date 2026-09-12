"""The profiler: a ring, a counter table, and the summary a bench quotes.

The claims worth pinning are about what the numbers mean rather than about the
data structure. Throughput is committed tokens over the cycles' own wall time,
so idle and prefill cannot inflate it. Acceptance is drafts that survived over
drafts proposed, so a cycle with no drafter does not read as perfect. The stage
breakdown accounts for the whole cycle, with the unclaimed remainder named as
what it is rather than quietly dropped.
"""

from __future__ import annotations

from titan.config.schema import ObservabilityConfig
from titan.core.types import CycleProfile
from titan.observability.profiler import NullProfiler, RingProfiler, build_profiler


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        self.t += 0.001
        return self.t


def profile(**kwargs) -> CycleProfile:
    base = dict(
        cycle=1,
        n_sequences=1,
        n_rows=4,
        draft_ms=1.0,
        verify_ms=8.0,
        accept_ms=0.5,
        sample_ms=0.0,
        detok_ms=0.2,
        ngram_wait_ms=0.1,
        host_syncs=1,
        tokens_committed=3,
        tokens_drafted=3,
        wall_ms=11.0,
    )
    base.update(kwargs)
    return CycleProfile(**base)


def test_throughput_counts_the_cycles_own_time_and_nothing_else():
    """Elapsed clock time includes prefill and idle. Folding either into a
    decode rate is how a number moves when nothing about decoding changed."""
    p = RingProfiler(Clock())
    for _ in range(10):
        p.cycle(profile(wall_ms=10.0, tokens_committed=2))
    summary = p.summarise()
    assert summary.cycles == 10
    assert summary.tokens_per_second == 200.0


def test_acceptance_is_drafts_that_survived_over_drafts_proposed():
    p = RingProfiler(Clock())
    p.cycle(profile(tokens_drafted=3, tokens_committed=3, n_sequences=1))
    p.cycle(profile(tokens_drafted=3, tokens_committed=1, n_sequences=1))
    summary = p.summarise()
    assert summary.mean_accepted_per_cycle == 1.0
    assert summary.acceptance_rate == 2 / 6


def test_a_cycle_with_no_drafter_does_not_read_as_perfect_acceptance():
    p = RingProfiler(Clock())
    p.cycle(profile(tokens_drafted=0, tokens_committed=1))
    assert p.summarise().acceptance_rate == 0.0


def test_the_ring_forgets_rather_than_grows():
    p = RingProfiler(Clock(), ring=8)
    for index in range(100):
        p.cycle(profile(cycle=index))
    assert p.summarise().cycles == 8
    assert p.snapshot()["counters"]["cycles"] == 100


def test_a_second_host_sync_is_counted_not_raised():
    """A cycle that synced twice produced the right tokens. It just cost twice
    what it should have, which is a counter, not an exception."""
    p = RingProfiler(Clock())
    p.cycle(profile(host_syncs=2))
    assert p.snapshot()["counters"]["host_sync_violations"] == 1


def test_the_stage_breakdown_accounts_for_the_whole_cycle():
    p = RingProfiler(Clock())
    p.cycle(profile())
    stages = p.stage_breakdown()
    assert stages["verify_ms"] == 8.0
    named = sum(
        stages[k]
        for k in ("draft_ms", "verify_ms", "accept_ms", "sample_ms", "detok_ms", "ngram_wait_ms")
    )
    assert stages["other_ms"] == stages["wall_ms"] - named


def test_events_are_counted_and_the_last_few_kept():
    p = RingProfiler(Clock(), events_kept=3)
    for index in range(10):
        p.event("prefill_chunk", end=index)
    p.event("admitted", sequence=1)
    snapshot = p.snapshot()
    assert snapshot["events"] == {"prefill_chunk": 10, "admitted": 1}
    assert len(p.recent_events()) == 3


def test_a_span_measures_host_time_and_syncs_nothing():
    p = RingProfiler(Clock())
    with p.span("restore"):
        pass
    counters = p.snapshot()["counters"]
    assert counters["span.restore.calls"] == 1


def test_reset_drops_the_window_between_bench_arms():
    p = RingProfiler(Clock())
    p.cycle(profile())
    p.reset()
    assert p.summarise().cycles == 0
    assert p.snapshot()["counters"] == {}


def test_the_null_profiler_accepts_everything_and_keeps_nothing():
    p = build_profiler(ObservabilityConfig(cycle_profile=False), Clock())
    assert isinstance(p, NullProfiler)
    p.cycle(profile())
    p.event("admitted")
    with p.span("x"):
        pass
    assert p.snapshot() == {}


def test_position_acceptance_is_survival_conditional_on_reaching():
    """Four depth-3 cycles accepting 3, 2, 1 and 0 drafts.

    Every chain reached all three positions, so the denominators are all four
    and the curve is the survival counts: three of four kept the first draft,
    two the second, one the third.
    """
    p = RingProfiler(Clock())
    for accepted in (3, 2, 1, 0):
        p.cycle(profile(tokens_drafted=3, tokens_committed=accepted + 1))
    assert p.position_acceptance() == (0.75, 0.5, 0.25)


def test_position_acceptance_denominator_is_chains_that_got_there():
    """A depth-1 cycle says nothing about position 1, so it is not counted."""
    p = RingProfiler(Clock())
    p.cycle(profile(tokens_drafted=1, tokens_committed=2))
    p.cycle(profile(tokens_drafted=3, tokens_committed=3))
    # Position 0: both chains reached it, both kept it.
    # Position 1: one chain reached it and kept it.
    # Position 2: one chain reached it and lost it.
    assert p.position_acceptance() == (1.0, 1.0, 0.0)


def test_position_acceptance_skips_multi_sequence_cycles():
    """Two chains and one committed count cannot be split back out."""
    p = RingProfiler(Clock())
    p.cycle(profile(n_sequences=2, tokens_drafted=6, tokens_committed=4))
    assert p.position_acceptance() == ()


def test_position_acceptance_is_empty_without_a_drafter():
    p = RingProfiler(Clock())
    p.cycle(profile(tokens_drafted=0, tokens_committed=1))
    assert p.position_acceptance() == ()
    assert p.accepted_histogram() == {"0": 1}


def test_accepted_histogram_counts_cycles_by_accepted_run():
    p = RingProfiler(Clock())
    for accepted in (0, 1, 1, 3):
        p.cycle(profile(tokens_drafted=3, tokens_committed=accepted + 1))
    assert p.accepted_histogram() == {"0": 1, "1": 2, "3": 1}


def test_snapshot_carries_the_acceptance_curve():
    p = RingProfiler(Clock())
    p.cycle(profile(tokens_drafted=2, tokens_committed=2))
    body = p.snapshot()
    assert body["position_acceptance"] == [1.0, 0.0]
    assert body["accepted_histogram"] == {"1": 1}


def test_a_window_reads_back_only_the_last_n_cycles():
    """A sweep measures several contexts in one process, so the window is how
    one request's cycles are read back without resetting anything."""
    p = RingProfiler(Clock())
    for _ in range(5):
        p.cycle(profile(tokens_drafted=3, tokens_committed=4, wall_ms=10.0))
    for _ in range(3):
        p.cycle(profile(tokens_drafted=3, tokens_committed=1, wall_ms=20.0))
    body = p.snapshot(3)
    assert body["window"] == 3
    assert body["decode"]["cycles"] == 3
    assert body["decode"]["mean_accepted_per_cycle"] == 0.0
    assert body["position_acceptance"] == [0.0, 0.0, 0.0]
    # The counters stay cumulative: that is what makes the window computable.
    assert body["counters"]["cycles"] == 8
