# SPDX-License-Identifier: MIT
"""The expected-value depth policy: the cost model, the acceptance curve, the choice.

Everything here is arithmetic on host data, so the tests state the numbers
rather than sampling them. The three questions worth asking of a policy that
decides how many GPU rows to spend are whether it prices a width it has never
run, whether it believes a curve rather than an average, and whether it can
choose to spend nothing at all -- which at 64k, with an accepted median of 1,
is the choice that matters.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from titan.core.types import CycleProfile, SequenceId, VerifyOutcome
from titan.engine.decode_cycle import (
    SEED_CYCLE_COST_MS,
    CycleCostModel,
    ExpectedValueDepthController,
    PositionAcceptance,
)


class FlatAcceptance(PositionAcceptance):
    """Every position accepted with the same probability. For the arithmetic."""

    def __init__(self, p: float, max_depth: int = 8) -> None:
        super().__init__(max_depth=max_depth)
        self.p = float(p)

    def probability(self, position: int) -> float:
        return 1.0 if position < 1 else self.p


def profile(width: int, wall_ms: float, sequences: int = 1) -> CycleProfile:
    return CycleProfile(
        cycle=1,
        n_sequences=sequences,
        n_rows=width * sequences,
        draft_ms=0.0,
        verify_ms=0.0,
        accept_ms=0.0,
        sample_ms=0.0,
        detok_ms=0.0,
        ngram_wait_ms=0.0,
        host_syncs=1,
        tokens_committed=1,
        tokens_drafted=0,
        wall_ms=wall_ms,
    )


# -- the cost model ---------------------------------------------------------


def test_seed_fit_reproduces_the_measured_widths():
    cost = CycleCostModel()
    for width, measured in SEED_CYCLE_COST_MS.items():
        assert cost.ms(width) == pytest.approx(measured, abs=0.7)


def test_the_model_prices_widths_it_has_never_seen():
    cost = CycleCostModel()
    intercept, slope = cost.fit
    assert intercept > 0 and slope > 0
    assert cost.ms(3) == pytest.approx(intercept + 3 * slope)
    assert cost.ms(8) > cost.ms(6) > cost.ms(4)


def test_observations_move_the_fit():
    cost = CycleCostModel(half_life=8.0)
    before = cost.ms(4)
    for _ in range(200):
        cost.observe(1, 8.0)
        cost.observe(4, 14.0)
    after = cost.ms(4)
    assert after < before
    assert after == pytest.approx(14.0, abs=1.5)


def test_an_outlier_cycle_does_not_reprice_the_machine():
    cost = CycleCostModel()
    before = cost.fit
    cost.observe(2, 4000.0)
    assert cost.fit == before
    assert cost.observations == 0


def test_a_degenerate_fit_falls_back_to_the_seed():
    cost = CycleCostModel(half_life=1.0, seed_weight=0.0)
    for _ in range(50):
        cost.observe(1, 30.0)
        cost.observe(4, 5.0)
    intercept, slope = cost.fit
    assert slope > 0


# -- the acceptance curve ---------------------------------------------------


def test_acceptance_is_conditional_on_reaching_the_position():
    curve = PositionAcceptance(max_depth=4, half_life=1e6, prior=0.5, prior_weight=1.0)
    for _ in range(200):
        # Four drafted, none accepted: position one is refuted every time and
        # positions two to four are never reached, so they learn nothing.
        curve.observe(0, 4)
    assert curve.probability(1) < 0.02
    assert curve.probability(3) == pytest.approx(0.5)


def test_a_full_chain_teaches_every_position():
    curve = PositionAcceptance(max_depth=4, half_life=1e6, prior=0.5, prior_weight=1.0)
    for _ in range(200):
        curve.observe(4, 4)
    for position in range(1, 5):
        assert curve.probability(position) > 0.98
    assert curve.expected_committed(4) == pytest.approx(5.0, abs=0.1)


def test_expected_committed_is_the_running_product():
    curve = FlatAcceptance(0.5, max_depth=4)
    assert curve.expected_committed(0) == pytest.approx(1.0)
    assert curve.expected_committed(1) == pytest.approx(1.5)
    assert curve.expected_committed(3) == pytest.approx(1 + 0.5 + 0.25 + 0.125)


def test_a_decaying_curve_forgets_the_prompt_it_used_to_be_in():
    curve = PositionAcceptance(max_depth=4, half_life=8.0, prior=0.5, prior_weight=1.0)
    for _ in range(100):
        curve.observe(4, 4)
    assert curve.probability(1) > 0.9
    for _ in range(100):
        curve.observe(0, 4)
    assert curve.probability(1) < 0.1


# -- the choice -------------------------------------------------------------


def controller(**kwargs) -> ExpectedValueDepthController:
    kwargs.setdefault("max_depth", 8)
    kwargs.setdefault("rows_budget", 64)
    return ExpectedValueDepthController(**kwargs)


@pytest.mark.parametrize(
    "acceptance,expected",
    [
        (0.2, 0),  # the 64k regime: a drafted row is a wasted row
        (0.5, 1),
        (0.8, 3),
        (0.95, 8),  # nothing beats drafting more when almost everything sticks
    ],
)
def test_ev_picks_the_depth_the_arithmetic_says(acceptance, expected):
    policy = controller()
    assert policy.best_depth(FlatAcceptance(acceptance), ceiling=8) == expected


def test_the_choice_follows_the_cost_table_and_not_only_acceptance():
    """The same acceptance on a cheaper marginal row buys a deeper chain."""
    cheap = CycleCostModel(seed={1: 16.0, 2: 16.5, 4: 17.5, 6: 18.5})
    policy = controller(cost=cheap)
    assert policy.best_depth(FlatAcceptance(0.5), ceiling=8) == 4
    assert controller().best_depth(FlatAcceptance(0.5), ceiling=8) == 1


def test_the_row_budget_is_a_ceiling_the_policy_cannot_argue_with():
    policy = controller(rows_budget=8)
    depths = policy.plan_depth(4, 3.0, 8)
    assert depths == [1, 1, 1, 1]
    assert policy.plan_depth(8, 3.0, 8) == [0] * 8


def test_a_fixed_policy_ignores_the_measurement():
    policy = controller(max_depth=3, min_depth=1, adaptive=False)
    for _ in range(50):
        policy.record(
            profile(4, 30.0),
            [VerifyOutcome(SequenceId(1), accepted=(), bonus=7, n_drafted=3)],
        )
    assert policy.next_depths(1) == [3]


def test_depth_falls_when_nothing_is_accepted():
    policy = controller(max_depth=6)
    # The prior is 0.7 a position, which the seed cost table prices at two.
    assert policy.next_depths(1) == [2]
    for _ in range(200):
        policy.record(
            profile(7, 40.0),
            [VerifyOutcome(SequenceId(1), accepted=(), bonus=7, n_drafted=6)],
        )
    assert policy.next_depths_for([1]) == [0]


def test_two_sequences_get_two_depths():
    """A 64k session and a short one share a process, not an acceptance curve."""
    policy = controller(max_depth=6)
    for _ in range(200):
        policy.record(
            profile(7, 40.0),
            [
                VerifyOutcome(SequenceId(1), accepted=(1, 2, 3, 4, 5, 6), bonus=7, n_drafted=6),
                VerifyOutcome(SequenceId(2), accepted=(), bonus=7, n_drafted=6),
            ],
        )
    generous, stingy = policy.next_depths_for([1, 2])
    assert generous == 6
    assert stingy == 0


def test_the_policy_learns_the_cost_of_the_widths_it_ran():
    policy = controller()
    for _ in range(400):
        policy.record(profile(4, 12.0), [])
        policy.record(profile(1, 9.0), [])
    assert policy.cost.ms(4) < SEED_CYCLE_COST_MS[4]
    assert policy.cost.observations == 800


def test_a_batched_cycle_is_not_folded_into_the_batch_of_one_table():
    policy = controller()
    for _ in range(50):
        policy.record(profile(2, 40.0, sequences=4), [])
    assert policy.cost.observations == 0


def test_the_estimator_still_reports_a_mean_for_anything_reading_it():
    policy = controller(max_depth=4)
    policy.record(
        profile(5, 30.0),
        [VerifyOutcome(SequenceId(1), accepted=(1, 2), bonus=3, n_drafted=4)],
    )
    assert policy.estimator.mean_accepted == pytest.approx(2.0)
    assert policy.estimator.acceptance_rate == pytest.approx(0.5)


def test_a_bad_depth_range_is_refused():
    with pytest.raises(ValueError):
        ExpectedValueDepthController(max_depth=1, min_depth=4)


# -- convergence ------------------------------------------------------------
#
# ROUND4 section 1a measured the identical reference configuration at 93.5 and
# 79.0 tok/s an hour apart, entirely because the policy settled on mean rows of
# 3.60 once and 2.97 the other time. These tests are the fake-driven version of
# that: one simulated machine and one simulated text, two starting points, and
# the requirement that the policy ends up in the same place. They are the unit
# half of ROUND5 step 1; the real-model half is the four-repeat spread in
# bench/decode/ROUND5.md.


class FakeWorkload:
    """A machine and a text the policy can be run against deterministically.

    ``truth`` is the real per-position acceptance probability, which the policy
    is never told. ``intercept`` and ``slope`` are the real cycle cost. Nothing
    is sampled: a chain of depth *k* accepts the number of positions whose
    running survival product is still above a rotating threshold, which gives
    the right long-run frequencies without a random number generator and
    without a seed anyone has to trust.
    """

    def __init__(
        self,
        truth: Sequence[float],
        *,
        intercept: float = 14.0,
        slope: float = 3.0,
    ) -> None:
        self.truth = [float(p) for p in truth]
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.step = 0

    def cost_ms(self, width: int) -> float:
        return self.intercept + self.slope * max(1, int(width))

    def accepted(self, depth: int) -> int:
        """How many of ``depth`` drafts stick this cycle."""
        self.step += 1
        # A fixed low-discrepancy sweep through [0, 1): position i is accepted
        # when its own threshold falls under its true probability, so over any
        # window the frequency at position i converges to truth[i - 1].
        n = 0
        for position in range(1, int(depth) + 1):
            threshold = ((self.step * 0.6180339887 + position * 0.31) % 1.0)
            if threshold >= self.truth[min(position, len(self.truth)) - 1]:
                break
            n += 1
        return n

    def run(self, policy: ExpectedValueDepthController, cycles: int) -> list[int]:
        """Drive the policy for ``cycles`` cycles. Returns the depths it ran."""
        depths: list[int] = []
        for _ in range(cycles):
            depth = policy.next_depths_for([1])[0]
            depths.append(depth)
            n_accepted = self.accepted(depth)
            width = depth + 1
            outcomes = (
                [
                    VerifyOutcome(
                        SequenceId(1),
                        accepted=tuple(range(n_accepted)),
                        bonus=7,
                        n_drafted=depth,
                    )
                ]
                if depth > 0
                else []
            )
            policy.record(profile(width, self.cost_ms(width)), outcomes)
        return depths


def converged(policy, workload, cycles=800):
    """The depth the policy is running once it has stopped moving."""
    depths = workload.run(policy, cycles)
    tail = [d for d in depths[-200:]]
    return max(set(tail), key=tail.count), tail


def steer(policy, high: bool) -> None:
    """Put the policy at a starting point before it sees the real workload.

    ``high`` feeds it a run of fully accepted deep chains, which is the resting
    point ROUND4 recorded at mean rows 3.60; the other feeds it a run of total
    rejections, which is the one it recorded at 2.97. Neither is the workload
    the policy is then asked to converge on.
    """
    for _ in range(120):
        if high:
            outcome = VerifyOutcome(
                SequenceId(1), accepted=(1, 2, 3, 4, 5, 6), bonus=7, n_drafted=6
            )
            policy.record(profile(7, 20.0), [outcome])
        else:
            outcome = VerifyOutcome(SequenceId(1), accepted=(), bonus=7, n_drafted=1)
            policy.record(profile(2, 60.0), [outcome])


def test_the_policy_converges_to_one_depth_from_a_low_start_and_a_high_one():
    truth = [0.85, 0.75, 0.62, 0.45, 0.3, 0.2]
    low = controller(max_depth=6, window=64)
    high = controller(max_depth=6, window=64)
    fresh = controller(max_depth=6, window=64)
    steer(low, high=False)
    steer(high, high=True)
    from_low, _ = converged(low, FakeWorkload(truth))
    from_high, _ = converged(high, FakeWorkload(truth))
    from_fresh, _ = converged(fresh, FakeWorkload(truth))
    assert from_low == from_high == from_fresh


def test_convergence_happens_inside_one_short_request():
    """A resting point reached after the request has finished is not a fix.

    Four 600-token prompts at three committed tokens a cycle is about eight
    hundred cycles, so a policy that needs more than a couple of hundred to
    agree with itself would still report ROUND4's spread.
    """
    truth = [0.85, 0.75, 0.62, 0.45, 0.3, 0.2]
    low = controller(max_depth=6, window=64)
    high = controller(max_depth=6, window=64)
    steer(low, high=False)
    steer(high, high=True)
    from_low = FakeWorkload(truth).run(low, 400)[-200:]
    from_high = FakeWorkload(truth).run(high, 400)[-200:]
    assert max(set(from_low), key=from_low.count) == max(
        set(from_high), key=from_high.count
    )


def test_the_two_starting_points_disagreed_before_the_workload_ran():
    """The steer is a real steer: without it the test above proves nothing."""
    low = controller(max_depth=6, window=64)
    high = controller(max_depth=6, window=64)
    steer(low, high=False)
    steer(high, high=True)
    assert low.next_depths_for([1])[0] != high.next_depths_for([1])[0]


def test_the_running_depth_stops_moving_once_it_has_converged():
    workload = FakeWorkload([0.85, 0.75, 0.62, 0.45, 0.3, 0.2])
    policy = controller(max_depth=6, window=64)
    depths = workload.run(policy, 900)
    tail = depths[-200:]
    incumbent = max(set(tail), key=tail.count)
    # Everything that is not the incumbent is a probe, and a probe is one
    # decision in ``probe_every``, both directions, so at most two in that many.
    off = sum(1 for d in tail if d != incumbent)
    duty = policy.probe_cycles / policy.probe_every
    assert off <= len(tail) * duty * 1.5 + policy.probe_cycles


def test_the_probe_reaches_positions_the_incumbent_never_drafts():
    """The depth-conditioning bug, stated as a test.

    A policy parked at depth zero drafts nothing, so no chain ever reaches
    position one, so the acceptance curve there is the prior for the life of
    the process however generous the text has become. Holding still is the one
    move that cannot resolve this. The costly cost table below is what puts the
    policy at zero; the text underneath it accepts nine drafts in ten.
    """
    expensive = {1: 10.0, 2: 400.0, 4: 1000.0, 6: 1600.0}
    generous = [0.9] * 6

    frozen = controller(max_depth=6, window=64, probe_every=0)
    frozen.cost = CycleCostModel(seed=expensive)
    depths = FakeWorkload(generous).run(frozen, 300)
    assert set(depths) == {0}
    assert frozen.acceptance_for(1).evidence(1) == 0.0

    probing = controller(max_depth=6, window=64)
    probing.cost = CycleCostModel(seed=expensive)
    FakeWorkload(generous).run(probing, 300)
    curve = probing.acceptance_for(1)
    assert curve.evidence(1) > 0.0
    assert curve.probability(1) > 0.7


def test_hysteresis_holds_a_depth_against_a_candidate_that_barely_wins():
    policy = controller(max_depth=6, hysteresis=0.5, dwell=0)
    ceiling = 6
    first = policy.stable_depth(1, FlatAcceptance(0.5), ceiling)
    # A curve whose argmax is elsewhere, but only just: the margin refuses it.
    nudged = FlatAcceptance(0.52)
    for _ in range(20):
        held = policy.stable_depth(1, nudged, ceiling)
    assert held == first


def test_a_candidate_that_wins_by_more_than_the_margin_takes_the_depth():
    policy = controller(max_depth=6, hysteresis=0.05, dwell=2, probe_every=0)
    ceiling = 6
    first = policy.stable_depth(1, FlatAcceptance(0.2), ceiling)
    for _ in range(20):
        held = policy.stable_depth(1, FlatAcceptance(0.97), ceiling)
    assert held > first


def test_dwell_is_counted_in_decisions_and_not_in_cycles():
    policy = controller(max_depth=6, hysteresis=0.0, dwell=5, probe_every=0)
    policy.stable_depth(1, FlatAcceptance(0.2), 6)
    generous = FlatAcceptance(0.97)
    seen = [policy.stable_depth(1, generous, 6) for _ in range(5)]
    assert seen[:4] == [seen[0]] * 4
    assert seen[-1] != seen[0]


# -- the cost model's own resting point -------------------------------------


def test_one_width_forever_does_not_bend_the_price_of_the_others():
    """The pooled fit's failure mode, as a regression test.

    A policy that settles on depth 2 feeds the cost model width 3 and nothing
    else. Under the pooled least-squares fit the seed points decayed away with
    every observation and the determinant went to zero, so the price of width 5
    -- the number that decides whether depth 4 is worth trying -- moved with
    the *count* of width-3 cycles rather than with the machine.
    """
    cost = CycleCostModel()
    early = None
    for i in range(4000):
        cost.observe(3, 21.0)
        if i == 40:
            early = cost.ms(5)
    late = cost.ms(5)
    assert early is not None
    assert late == pytest.approx(early, rel=0.02)
    assert cost.ms(3) == pytest.approx(21.0, abs=0.1)


def test_a_width_below_the_sample_floor_is_priced_off_the_line():
    cost = CycleCostModel(min_samples=8)
    for _ in range(3):
        cost.observe(2, 1.0)
    assert not cost.measured(2)
    assert cost.ms(2) > 5.0
    for _ in range(6):
        cost.observe(2, 1.0)
    assert cost.measured(2)
    assert cost.ms(2) == pytest.approx(1.0, abs=0.05)


def test_a_width_that_stopped_being_run_decays_back_to_the_line():
    cost = CycleCostModel(half_life=8.0, min_samples=4)
    for _ in range(20):
        cost.observe(6, 2.0)
    assert cost.measured(6)
    for _ in range(400):
        cost.observe(1, 16.0)
    assert not cost.measured(6)
    assert cost.ms(6) > 5.0
