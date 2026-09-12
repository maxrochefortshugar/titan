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
