"""The decode cycle: the piece the overlay could not fix from outside.

One cycle advances every decoding sequence by at least one token:

    draft   -> Drafter.propose for all sequences, no host sync
    prefetch-> NgramReader.prefetch for the draft candidates' rows
    verify  -> ModelBackend.verify, one padded row block, one host sync
    commit  -> per sequence, accepted + bonus; rollback is a truncation
    emit    -> incremental detokenisation and events
    profile -> one CycleProfile, always

Four properties are designed in here rather than bolted on:

1. **Lockstep batching.** All sequences share one verify forward at a common
   row width. Ragged widths are padded, not split. The overlay's batched MTP
   lost to plain batching because it split.
2. **Replay-free rollback.** Rejected drafts are undone by truncating state to
   the accepted length, using the snapshot the verify pass staged for the 36
   recurrent layers. Nothing is recomputed.
3. **In-graph acceptance.** The comparison between drafted ids and target
   argmax happens on device; only a small integer vector crosses to the host,
   once per cycle. ``CycleProfile.host_syncs`` must read 1.
4. **N-gram prefetch.** Row ids for the draft candidates are queued before the
   verify forward, so the SSD read overlaps the GPU work instead of adding
   about 2.3 ms of host wait to each forward.

## The row layout, stated once

A decoding sequence always has exactly one token the backend has not consumed
yet: the token the previous cycle produced last. Call it the pending token. So
during decode the invariant is

    backend.state_length(state) == len(sequence.tokens) - 1

and one verify row block for a sequence is

    (pending, draft_1, ..., draft_k)

of width k+1. The forward consumes all k+1 positions, the acceptance reduction
keeps 1 + n of them where n is the number of confirmed drafts, and the bonus
token the forward produced at the first rejection point becomes the next
cycle's pending token. Committing ``drafts[:n] + (bonus,)`` is exactly what
plain greedy decoding would have produced, which is the lossless-speculation
invariant the parity tests check.

The plain M1 loop is the same cycle at k = 0: one row per sequence, no drafter,
one host sync, argmax taken in the graph. It is written as its own class rather
than as a branch, because M1 parity is measured against it and a branch inside
the speculative path is a branch that can drift.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from titan.core.errors import StateError, TitanError
from titan.core.types import (
    CycleProfile,
    DraftCandidate,
    FinishReason,
    SamplingParams,
    SequenceId,
    SequenceState,
    StopCondition,
    TokenEvent,
    VerifyOutcome,
)

__all__ = [
    "CycleResult",
    "DecodeCycle",
    "AcceptancePolicy",
    "GreedyAcceptance",
    "SampledAcceptance",
    "AcceptanceEstimator",
    "DepthController",
    "CycleCostModel",
    "PositionAcceptance",
    "ExpectedValueDepthController",
    "SEED_CYCLE_COST_MS",
    "TextEmitter",
    "EmitResult",
    "NullProfiler",
    "MonotonicClock",
    "PlainDecodeCycle",
    "MTPDecodeCycle",
    "EMPTY_PROFILE",
]


# ---------------------------------------------------------------------------
# small utilities the engine owns rather than imports
# ---------------------------------------------------------------------------


class MonotonicClock:
    """Default :class:`titan.core.ports.Clock`. Injected everywhere else."""

    def now(self) -> float:
        return time.monotonic()


class NullProfiler:
    """A profiler that keeps the last cycle and counts events, nothing more.

    The real one is ``titan.observability.profiler``; the engine takes the port
    and this stand-in exists so a cycle can be constructed without one.
    """

    def __init__(self) -> None:
        self.cycles: list[CycleProfile] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.counters: dict[str, int] = {}

    def cycle(self, profile: CycleProfile) -> None:
        self.cycles.append(profile)

    def event(self, name: str, **fields: float | int | str) -> None:
        self.events.append((name, dict(fields)))

    def count(self, name: str, amount: int = 1) -> None:
        # Here rather than only on the real profiler because the loop counts
        # its failures through this port, and a failure that is only counted
        # in production is a failure no test can assert on.
        self.counters[name] = self.counters.get(name, 0) + amount

    def span(self, name: str) -> "_NullSpan":
        return _NullSpan()

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "cycles": len(self.cycles),
            "events": len(self.events),
            **self.counters,
        }


class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


EMPTY_PROFILE = CycleProfile(
    cycle=0,
    n_sequences=0,
    n_rows=0,
    draft_ms=0.0,
    verify_ms=0.0,
    accept_ms=0.0,
    sample_ms=0.0,
    detok_ms=0.0,
    ngram_wait_ms=0.0,
    host_syncs=0,
    tokens_committed=0,
    tokens_drafted=0,
    wall_ms=0.0,
)


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Everything one cycle produced. Pure data; the caller does the emitting."""

    outcomes: tuple[VerifyOutcome, ...]
    events: tuple[TokenEvent, ...]
    finished: tuple[SequenceId, ...]
    profile: CycleProfile


class AcceptancePolicy(Protocol):
    """How a drafted token is judged against the target distribution.

    Greedy acceptance (id equality against the target argmax) is the only mode
    used at parity milestones, and it is exact by construction. Sampled
    acceptance uses the standard probability-ratio rule, which preserves the
    target distribution but not any particular sample, so it is never compared
    token-for-token against oMLX.
    """

    @property
    def is_exact(self) -> bool: ...

    def name(self) -> str: ...


class GreedyAcceptance:
    """Id equality against the target argmax. Exact, and the parity currency."""

    @property
    def is_exact(self) -> bool:
        return True

    def name(self) -> str:
        return "greedy"


class SampledAcceptance:
    """Probability-ratio acceptance with in-graph residual sampling.

    Distribution-preserving but not sample-preserving, so it is never compared
    token for token with a recorded reference. The arithmetic lives in the
    ``verify_accept`` op, which computes the residual for every position so the
    cycle still crosses to the host exactly once.
    """

    @property
    def is_exact(self) -> bool:
        return False

    def name(self) -> str:
        return "sampled"


def policy_for(sampling: SamplingParams) -> AcceptancePolicy:
    return GreedyAcceptance() if sampling.is_greedy else SampledAcceptance()


class DecodeCycle(Protocol):
    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        """Advance ``batch`` by one cycle.

        Preconditions: every sequence is DECODING, holds an open state handle,
        and its state length equals ``prompt_len + committed`` (which is
        ``len(tokens) - 1``, the pending token being the one the backend has
        not consumed).

        Postconditions: for each sequence, state length grew by exactly
        ``outcome.n_committed``; ``committed`` grew by the same; the sequence is
        finished if a stop condition matched inside the accepted run, in which
        case tokens after the stop are discarded and the state is truncated to
        the stop position. Discarding them matters beyond tidiness: a copy or
        draft block accepted past a stop token corrupts the recurrent state that
        the prefix cache is about to store.
        """


# ---------------------------------------------------------------------------
# acceptance history and the depth controller
# ---------------------------------------------------------------------------


class AcceptanceEstimator:
    """Rolling mean of accepted drafts per drafted chain.

    A window rather than an EWMA because the number the depth policy wants is
    "how many of the last N drafts stuck", and a window answers that without a
    time constant nobody can name. The measured decode budget says three
    accepted tokens per cycle is 111 tok/s and that the accepted median at 64k
    is 1, so the estimator has to fall as fast as the context grows.
    """

    def __init__(self, window: int = 64, initial: float = 1.0) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._initial = float(initial)
        self._samples: list[int] = []
        self._drafted: list[int] = []

    def observe(self, outcomes: Sequence[VerifyOutcome]) -> None:
        for outcome in outcomes:
            if outcome.n_drafted <= 0:
                continue
            self._samples.append(len(outcome.accepted))
            self._drafted.append(outcome.n_drafted)
        overflow = len(self._samples) - self.window
        if overflow > 0:
            del self._samples[:overflow]
            del self._drafted[:overflow]

    @property
    def mean_accepted(self) -> float:
        """Mean accepted drafts per chain. The number depth is planned from."""
        if not self._samples:
            return self._initial
        return sum(self._samples) / len(self._samples)

    @property
    def acceptance_rate(self) -> float:
        """Accepted over drafted. Reported, not used for depth: a chain that
        drafts 8 and accepts 1 has a fine rate and a terrible cycle."""
        drafted = sum(self._drafted)
        if drafted == 0:
            return 0.0
        return sum(self._samples) / drafted

    @property
    def n_samples(self) -> int:
        return len(self._samples)


class DepthController:
    """Adaptive draft depth. Satisfies :class:`titan.core.ports.Verifier`.

    Pure: the same inputs give the same depths, no clock and no device query.
    The rule is one line of arithmetic and two clamps, and it is deliberately
    the whole policy:

        depth = round(mean_accepted) + 1, clamped to [min, max] and to the
        row budget divided across the batch.

    The +1 is what makes it self-correcting. A cycle that accepts everything it
    drafted has no evidence about the depth above it, so the policy spends one
    speculative row to find out; a cycle that accepts nothing falls to the
    floor within a window. The overlay's fixed 3 could not do either, and at 64k
    where the accepted median is 1 it paid for two rejected rows every cycle.
    """

    def __init__(
        self,
        *,
        max_depth: int = 3,
        min_depth: int = 0,
        adaptive: bool = True,
        window: int = 64,
        rows_budget: int = 32,
    ) -> None:
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("need 0 <= min_depth <= max_depth")
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.adaptive = adaptive
        self.rows_budget = rows_budget
        self.estimator = AcceptanceEstimator(window=window, initial=float(max_depth))

    # -- the Verifier port -------------------------------------------------
    def plan_depth(
        self,
        n_sequences: int,
        recent_acceptance: float,
        rows_budget: int,
    ) -> list[int]:
        if n_sequences <= 0:
            return []
        if not self.adaptive:
            depth = self.max_depth
        else:
            depth = int(round(recent_acceptance)) + 1
        # Every sequence spends one row on its pending token, so the budget for
        # drafts is what is left after the batch has paid for itself.
        per_sequence = max(0, rows_budget // n_sequences - 1)
        depth = min(depth, self.max_depth, per_sequence)
        depth = max(depth, self.min_depth)
        return [depth] * n_sequences

    def record(self, profile: CycleProfile, outcomes: Sequence[VerifyOutcome]) -> None:
        self.estimator.observe(outcomes)

    # -- convenience for the cycle ----------------------------------------
    def next_depths(self, n_sequences: int) -> list[int]:
        return self.plan_depth(
            n_sequences, self.estimator.mean_accepted, self.rows_budget
        )


# ---------------------------------------------------------------------------
# the expected-value depth policy
# ---------------------------------------------------------------------------


SEED_CYCLE_COST_MS: Mapping[int, float] = {1: 16.2, 2: 19.6, 4: 27.6, 6: 36.2}
"""Measured cycle wall time against verify row width, batch of one.

The integration report's numbers, and the only thing the policy knows before it
has run a cycle of its own. They are a seed and not a constant: the same table
on a warm machine, a cold one, or at 64k of context is a different table, which
is the whole reason the model is refitted at runtime.
"""


class CycleCostModel:
    """Cycle milliseconds against verify row width: a table where it has
    evidence, a line where it does not.

    ROUND4 section 1a measured the same configuration at 93.5 and 79.0 tok/s an
    hour apart, and the column that moved with it was the depth the policy
    chose. This class was half the reason. It used to pool every observation
    into one weighted least-squares fit, and a policy that settles on one depth
    only ever feeds it one width. Two things follow from that and both are bad.
    The mass at a single ``x`` drives the regression's determinant to zero, so
    the fit flips between "a line through one point and a decaying seed" and
    the seed fallback, and which of the two a run is in depends on how many
    cycles it has run rather than on the machine. And the seed points, which
    are the only other widths in the sum, decay with every observation, so an
    hour-old process prices width 6 off nothing at all.

    So: one decayed mean per width, and the line is fitted over those *means*,
    one point per width, rather than over the raw observations. A width the
    policy has run a lot contributes exactly as much to the line's shape as one
    it ran twice, which is what stops the incumbent depth from bending the
    price of every alternative toward itself. Widths with at least
    ``min_weight`` of accumulated evidence are quoted from their own mean and
    never from the line; the line exists only to price the widths the policy
    has not run, which is the question a depth policy has to ask every cycle.

    Observations are still decayed rather than windowed. Thermal drift alone
    was measured at 10 to 16% on this machine, so a cost table that averages an
    hour of cycles is describing a machine that is no longer there.
    """

    def __init__(
        self,
        *,
        seed: Mapping[int, float] = SEED_CYCLE_COST_MS,
        half_life: float = 512.0,
        seed_weight: float = 8.0,
        min_samples: int = 8,
        min_weight: float = 1.0,
        own_half_life: float = 16.0,
    ) -> None:
        if half_life <= 0 or own_half_life <= 0:
            raise ValueError("half_life and own_half_life must be positive")
        if min_samples < 1 or min_weight <= 0:
            raise ValueError("min_samples and min_weight must be positive")
        self._decay = 0.5 ** (1.0 / half_life)
        self._own_decay = 0.5 ** (1.0 / own_half_life)
        self._seed = dict(seed)
        self._seed_weight = float(seed_weight)
        self._min_samples = int(min_samples)
        self._min_weight = float(min_weight)
        # A width's price and a width's recency are two different questions and
        # they need two different clocks.
        #
        # The price is an average over that width's *own* last ~``own_half_life``
        # cycles, decayed in its own observations. A width the policy stopped
        # running and then probed sixteen times has a price set by those sixteen
        # probes, not by the four hundred cycles it ran an hour ago. Under a
        # single clock shared with every other width, a stale price at a rarely
        # run width takes thousands of cycles to wash out, and while it is
        # washing out it is what pins the policy to the depth that produced it.
        # That is a resting point, and it is the one that traps a policy at
        # depth zero: nothing but a probe ever runs width two again.
        #
        # Recency is a weight decayed in *every* observation, and it decides
        # only whether the width has been seen lately enough to quote at all.
        self._own_weight: dict[int, float] = {}
        self._own_total: dict[int, float] = {}
        self._weight: dict[int, float] = {}
        self._count: dict[int, int] = {}
        self.observations = 0

    # -- evidence ----------------------------------------------------------
    def observe(self, width: int, wall_ms: float) -> None:
        """Record one cycle. Outliers are dropped, not smoothed.

        A cycle that ran ten times its predicted cost is a prefill sharing the
        thread, an SSD stall or a snapshot store, and folding it into the mean
        would price every future width off one event that had nothing to do
        with the row count.
        """
        if width < 1 or wall_ms <= 0.0:
            return
        predicted = self.ms(width)
        if wall_ms > 4.0 * predicted:
            return
        for key in list(self._weight):
            self._weight[key] *= self._decay
        index = int(width)
        self._weight[index] = self._weight.get(index, 0.0) + 1.0
        self._own_weight[index] = self._own_weight.get(index, 0.0) * self._own_decay + 1.0
        self._own_total[index] = (
            self._own_total.get(index, 0.0) * self._own_decay + float(wall_ms)
        )
        self._count[index] = self._count.get(index, 0) + 1
        self.observations += 1

    def weight(self, width: int) -> float:
        """Accumulated, decayed evidence at this width. Zero means the line."""
        return self._weight.get(int(width), 0.0)

    def discount(self, width: int, factor: float) -> None:
        """Forget most of what is known about one width, on purpose.

        Called when the policy starts a probe run at a width it has not been
        running. Whatever price that width carries was measured under a regime
        the policy has not observed since -- a different context length, a
        different thermal state, a different kernel set -- and averaging eight
        fresh cycles against four hundred stale ones is how a wrong price
        survives the measurement sent to correct it.
        """
        index = int(width)
        if index not in self._own_weight or not 0.0 <= factor <= 1.0:
            return
        self._own_weight[index] *= factor
        self._own_total[index] *= factor

    def measured(self, width: int) -> bool:
        """Two gates, and they answer different questions.

        ``min_samples`` is the sample count the brief asks for: a mean of two
        cycles is not a price. ``min_weight`` is recency: a width measured
        two hundred times an hour ago and never since has decayed to nothing,
        and quoting its stale mean would be worse than the line, which at
        least tracks the widths the machine is still running.
        """
        index = int(width)
        return (
            self._count.get(index, 0) >= self._min_samples
            and self._weight.get(index, 0.0) >= self._min_weight
        )

    def mean(self, width: int) -> float | None:
        """The width's own decayed mean, or ``None`` below the sample floor."""
        index = int(width)
        if not self.measured(index):
            return None
        return self._own_total[index] / self._own_weight[index]

    # -- the line ----------------------------------------------------------
    def _points(self) -> list[tuple[float, float, float]]:
        """``(width, ms, weight)`` for the fit: one point per width.

        A measured width contributes its own mean at unit weight. A seed width
        the policy has not measured contributes the seed value at
        ``seed_weight``, and a seed width it *has* measured contributes
        nothing, because the measurement is the better answer and the seed
        would only drag the line back toward the machine it was taken on.
        """
        points: list[tuple[float, float, float]] = []
        for width in sorted(self._weight):
            own = self.mean(width)
            if own is not None:
                points.append((float(width), own, 1.0))
        measured = {int(p[0]) for p in points}
        for width, ms in self._seed.items():
            if int(width) not in measured and self._seed_weight > 0.0:
                points.append((float(width), float(ms), self._seed_weight))
        return points

    @property
    def fit(self) -> tuple[float, float]:
        """``(intercept, slope)`` in milliseconds."""
        points = self._points()
        if len(points) < 2:
            return self._seed_fit()
        n = sum(w for _, _, w in points)
        sx = sum(w * x for x, _, w in points)
        sy = sum(w * y for _, y, w in points)
        sxx = sum(w * x * x for x, _, w in points)
        sxy = sum(w * x * y for x, y, w in points)
        denominator = n * sxx - sx * sx
        if denominator <= 1e-9:
            return self._seed_fit()
        slope = (n * sxy - sx * sy) / denominator
        intercept = (sy - slope * sx) / n
        if slope <= 0.0 or intercept <= 0.0:
            # A non-positive slope says wider cycles are free, which they are
            # not; it says the observations are degenerate. The seed is a worse
            # description of this machine and a better description of physics.
            return self._seed_fit()
        return intercept, slope

    def _seed_fit(self) -> tuple[float, float]:
        widths = list(self._seed)
        if len(widths) < 2:
            return (SEED_CYCLE_COST_MS[1], 4.0)
        n = float(len(widths))
        sx = sum(widths)
        sy = sum(self._seed.values())
        sxx = sum(w * w for w in widths)
        sxy = sum(w * self._seed[w] for w in widths)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        return (sy - slope * sx) / n, slope

    def ms(self, width: int) -> float:
        """The price of a cycle at this width. Its own mean, or the line."""
        index = max(1, int(width))
        own = self.mean(index)
        if own is not None:
            return max(1e-3, own)
        intercept, slope = self.fit
        return max(1e-3, intercept + slope * index)


class PositionAcceptance:
    """P(draft *i* is accepted | the chain reached position *i*).

    Per position, not one number, because the whole reason a depth policy can
    beat a fixed depth is that positions are not alike: vLLM's DSpark
    measurement is that the seventh drafted token survives under 10% of the
    time against over 70% for the first. A single mean cannot express that, and
    a policy that plans depth from one is planning against an average of a
    curve it could have measured.

    Conditional on reaching the position, so the expected committed tokens is a
    running product and positions past the first rejection are not counted as
    failures. A chain of eight that dies at two says nothing about position
    five and this estimator says nothing about it either.
    """

    def __init__(
        self,
        *,
        max_depth: int = 8,
        half_life: float = 64.0,
        prior: float = 0.7,
        prior_weight: float = 2.0,
    ) -> None:
        self.max_depth = int(max_depth)
        self._decay = 0.5 ** (1.0 / max(1e-9, half_life))
        self._prior = float(prior)
        self._prior_weight = float(prior_weight)
        self._hits = [prior * prior_weight] * (self.max_depth + 1)
        self._trials = [prior_weight] * (self.max_depth + 1)

    def observe(self, n_accepted: int, n_drafted: int) -> None:
        if n_drafted <= 0:
            return
        reached = min(n_accepted + 1, n_drafted)
        for position in range(1, reached + 1):
            if position > self.max_depth:
                break
            self._hits[position] *= self._decay
            self._trials[position] *= self._decay
            self._trials[position] += 1.0
            if position <= n_accepted:
                self._hits[position] += 1.0

    def probability(self, position: int) -> float:
        if position < 1:
            return 1.0
        index = min(position, self.max_depth)
        trials = self._trials[index]
        if trials <= 0.0:
            return self._prior
        return min(1.0, max(0.0, self._hits[index] / trials))

    def evidence(self, position: int) -> float:
        """Trials at this position beyond the prior's own weight.

        Zero means the position has never been drafted, so its probability is
        the prior and not a measurement. The depth policy needs to know the
        difference: a position it has never reached is the one thing a depth
        policy cannot learn about by holding still, and it is what the probe
        in :class:`ExpectedValueDepthController` exists to reach.
        """
        if position < 1:
            return 0.0
        index = min(int(position), self.max_depth)
        return max(0.0, self._trials[index] - self._prior_weight)

    def discount(self, position: int, factor: float) -> None:
        """Forget most of the evidence at one position. The probe's companion.

        A position the policy stopped drafting keeps whatever it last believed,
        because nothing decays a counter nobody touches. That is correct while
        the belief is fresh and wrong the moment the text moves on, and the
        policy cannot tell the difference without drafting there again. So when
        it does draft there again, it discounts first.
        """
        if position < 1 or not 0.0 <= factor <= 1.0:
            return
        index = min(int(position), self.max_depth)
        self._hits[index] *= factor
        self._trials[index] *= factor

    def measured_depth(self) -> int:
        """The deepest position with evidence of its own. Zero if none."""
        deepest = 0
        for position in range(1, self.max_depth + 1):
            if self.evidence(position) > 0.0:
                deepest = position
        return deepest

    def expected_committed(self, depth: int) -> float:
        """Tokens a cycle at this depth commits, drafts plus the bonus."""
        total = 1.0
        survival = 1.0
        for position in range(1, int(depth) + 1):
            survival *= self.probability(position)
            total += survival
        return total

    @property
    def mean_accepted(self) -> float:
        return self.expected_committed(self.max_depth) - 1.0


class ExpectedValueDepthController:
    """Depth by expected committed tokens per millisecond of cycle.

    The rule the fixed and the ``round(mean) + 1`` policies both approximate,
    written out:

        k* = argmax_k  E[committed | k] / cycle_ms(k + 1)

    with ``E`` from :class:`PositionAcceptance` and ``cycle_ms`` from
    :class:`CycleCostModel`. Both sides are measured on the running machine and
    neither reads the drafter's own confidence, which is the difference between
    this and the confidence-gated depth the overlay measured at -6%. TapOut's
    survey is that gates keyed on the draft's certainty lose (AdaEDL at 0.93x,
    SpecDec++ at 0.99x); the DSpark result is that a profiled cost table wins.
    The usual effect here is to draft *shallower*, which is the opposite
    intervention to the one that failed.

    Estimates are per sequence. A 64k session and a 200-token one share a
    process, a machine and a cost table, and they do not share an acceptance
    curve; averaging them gives both the wrong depth. The cost table is shared
    because it describes the machine, and the acceptance curve is not because
    it describes the text.

    ## Why the argmax alone is not the policy

    ROUND4 section 1a measured the identical reference configuration at 93.5
    and 79.0 tok/s an hour apart, with mean rows of 3.60 and 2.97. That is an
    18% spread, larger than every kernel effect the same round went on to
    measure, and it is not noise: it is a control loop with more than one
    resting point. Three mechanisms put it there, and the policy answers all
    three.

    *The estimator learns cycle cost from cycles the policy itself shortened.*
    A run that settles on depth 2 only ever feeds :class:`CycleCostModel`
    width 3, and the pooled regression it used to keep would then price widths
    4 and 5 off a determinant driven to zero by its own mass. That is fixed in
    the cost model, which now keeps one decayed mean per width and fits the
    line over the means rather than over the observations.

    *The acceptance estimate is conditioned on the depth.*
    ``PositionAcceptance`` can only learn about position *i* from a chain that
    drafted *i* tokens, so at depth 2 positions 3 and above hold the prior
    forever, whatever the text is actually doing. Holding still is the one
    thing that cannot resolve this, so the policy does not hold still: every
    ``probe_every`` decisions it spends one cycle at the depth above the
    incumbent and one at the depth below, and both neighbours get real
    evidence at both estimators. A probe is a row, not a rollback, so it costs
    a fraction of a cycle and it changes nothing about what the model says.

    *Nothing damped the switch.* An argmax over two values a per-cent apart
    flips on noise, and each flip changes which width the cost model is fed,
    which changes the argmax. A candidate now has to beat the incumbent by
    ``hysteresis`` in expected tokens per millisecond, and the incumbent has to
    have held for ``dwell`` decisions before it can be displaced at all.
    """

    def __init__(
        self,
        *,
        max_depth: int = 3,
        min_depth: int = 0,
        adaptive: bool = True,
        window: int = 64,
        rows_budget: int = 32,
        cost: CycleCostModel | None = None,
        max_tracked: int = 64,
        hysteresis: float = 0.06,
        dwell: int = 8,
        probe_every: int = 48,
        probe_cycles: int = 8,
        probe_discount: float = 0.3,
    ) -> None:
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("need 0 <= min_depth <= max_depth")
        if hysteresis < 0.0:
            raise ValueError("hysteresis must not be negative")
        if dwell < 0 or probe_every < 0 or probe_cycles < 0:
            raise ValueError("dwell, probe_every and probe_cycles must not be negative")
        if not 0.0 <= probe_discount <= 1.0:
            raise ValueError("probe_discount must be a fraction")
        self.max_depth = int(max_depth)
        self.min_depth = int(min_depth)
        self.adaptive = bool(adaptive)
        self.rows_budget = int(rows_budget)
        self.window = int(window)
        self.cost = cost or CycleCostModel()
        self.max_tracked = int(max_tracked)
        self.hysteresis = float(hysteresis)
        self.dwell = int(dwell)
        self.probe_every = int(probe_every)
        self.probe_cycles = int(probe_cycles)
        self.probe_discount = float(probe_discount)
        # Kept so anything reading the old controller's rolling mean still
        # finds one, and so the two policies can be compared on one number.
        self.estimator = AcceptanceEstimator(window=window, initial=float(max_depth))
        self.shared = self._new_acceptance()
        self._by_sequence: dict[int, PositionAcceptance] = {}
        self.chosen: list[int] = []
        # Per decision-track state. The shared track is keyed ``None`` so that
        # ``plan_depth`` and ``next_depths_for`` do not fight over one
        # incumbent while addressing different estimators.
        self._incumbent: dict[Any, int] = {}
        self._held: dict[Any, int] = {}
        self._decisions: dict[Any, int] = {}
        # track -> (depth being probed, decisions left in the run)
        self._probe: dict[Any, tuple[int, int]] = {}
        self.probes = 0
        self.probe_runs = 0

    def _new_acceptance(self) -> PositionAcceptance:
        return PositionAcceptance(
            max_depth=max(1, self.max_depth), half_life=float(self.window)
        )

    def acceptance_for(self, sequence_id: int) -> PositionAcceptance:
        estimate = self._by_sequence.get(int(sequence_id))
        if estimate is None:
            estimate = self._new_acceptance()
            self._by_sequence[int(sequence_id)] = estimate
            if len(self._by_sequence) > self.max_tracked:
                self._by_sequence.pop(next(iter(self._by_sequence)))
        return estimate

    # -- the policy --------------------------------------------------------
    def best_depth(self, acceptance: PositionAcceptance, ceiling: int) -> int:
        """The depth that maximises committed tokens per millisecond."""
        floor = min(self.min_depth, ceiling)
        if ceiling <= floor:
            return floor
        best = floor
        best_value = acceptance.expected_committed(floor) / self.cost.ms(floor + 1)
        for depth in range(floor + 1, ceiling + 1):
            value = acceptance.expected_committed(depth) / self.cost.ms(depth + 1)
            if value > best_value:
                best, best_value = depth, value
        return best

    def _ceiling(self, n_sequences: int, rows_budget: int) -> int:
        per_sequence = max(0, rows_budget // max(1, n_sequences) - 1)
        return max(self.min_depth, min(self.max_depth, per_sequence))

    # -- convergence: hysteresis, dwell, and the probe ---------------------
    def value_of(self, acceptance: PositionAcceptance, depth: int) -> float:
        """Expected committed tokens per millisecond at this depth."""
        return acceptance.expected_committed(depth) / self.cost.ms(depth + 1)

    def _begin_probe(
        self,
        track: Any,
        acceptance: PositionAcceptance,
        held: int,
        probe: int,
    ) -> None:
        """Start a probe run, discounting what is believed about its depth."""
        self._probe[track] = (probe, max(0, self.probe_cycles - 1))
        self.probes += 1
        self.probe_runs += 1
        factor = self.probe_discount
        self.cost.discount(probe + 1, factor)
        for position in range(min(held, probe) + 1, max(held, probe) + 1):
            acceptance.discount(position, factor)

    def stable_depth(
        self, track: Any, acceptance: PositionAcceptance, ceiling: int
    ) -> int:
        """The depth this track will actually run, incumbent included.

        The first decision on a track is the plain argmax: there is nothing to
        be hysteretic about yet, and a policy that needed ``dwell`` decisions
        before it could choose anything would spend its first cycles at a depth
        no measurement chose. Afterwards the incumbent keeps the depth unless a
        candidate is worth ``hysteresis`` more, and only once the incumbent has
        held for ``dwell`` decisions.
        """
        floor = min(self.min_depth, ceiling)
        held = self._incumbent.get(track)
        decisions = self._decisions.get(track, 0)
        self._decisions[track] = decisions + 1

        if held is None:
            chosen = self.best_depth(acceptance, ceiling)
            self._incumbent[track] = chosen
            self._held[track] = 0
            return chosen

        held = max(floor, min(held, ceiling))
        # The probe comes before the comparison, and it does not disturb the
        # incumbent: it is a measurement, not a decision. Alternating up and
        # down means neither neighbour is the one the policy never prices.
        #
        # A probe is a *run* of cycles, not one cycle, and that is the whole
        # difference between a probe that works and one that does not. Both
        # estimators average over their own last few dozen observations at a
        # position, so a single cycle at the neighbouring depth moves the
        # estimate by about one part in ninety, and a policy in a wrong resting
        # point needs a hundred and forty such cycles to climb out of it. Eight
        # in a row, after discounting what was believed about that depth
        # before, is a measurement the policy can act on the same minute.
        running = self._probe.get(track)
        if running is not None:
            depth, left = running
            if left > 0 and floor <= depth <= ceiling:
                self._probe[track] = (depth, left - 1)
                self.probes += 1
                return depth
            self._probe.pop(track, None)
        if self.probe_every > 0 and self.probe_cycles > 0 and (
            decisions % self.probe_every == 0
        ):
            direction = 1 if (decisions // self.probe_every) % 2 else -1
            for candidate_probe in (held + direction, held - direction):
                # Flipping when the first neighbour is out of range is not
                # tidiness. Depth zero is an absorbing state -- a cycle that
                # drafts nothing teaches the acceptance curve nothing and the
                # cost model only width one -- and it is exactly the state
                # where half the probes would otherwise be thrown away.
                if floor <= candidate_probe <= ceiling and candidate_probe != held:
                    self._begin_probe(track, acceptance, held, candidate_probe)
                    return candidate_probe

        self._held[track] = self._held.get(track, 0) + 1
        candidate = self.best_depth(acceptance, ceiling)
        if candidate == held:
            return held
        if self._held[track] < self.dwell:
            return held
        incumbent_value = self.value_of(acceptance, held)
        if self.value_of(acceptance, candidate) <= incumbent_value * (
            1.0 + self.hysteresis
        ):
            return held
        self._incumbent[track] = candidate
        self._held[track] = 0
        return candidate

    # -- the Verifier port -------------------------------------------------
    def plan_depth(
        self,
        n_sequences: int,
        recent_acceptance: float,
        rows_budget: int,
    ) -> list[int]:
        if n_sequences <= 0:
            return []
        ceiling = self._ceiling(n_sequences, rows_budget)
        if not self.adaptive:
            return [min(self.max_depth, ceiling)] * n_sequences
        return [self.stable_depth(None, self.shared, ceiling)] * n_sequences

    def record(self, profile: CycleProfile, outcomes: Sequence[VerifyOutcome]) -> None:
        self.estimator.observe(outcomes)
        for outcome in outcomes:
            if outcome.n_drafted <= 0:
                continue
            self.shared.observe(len(outcome.accepted), outcome.n_drafted)
            self.acceptance_for(int(outcome.sequence_id)).observe(
                len(outcome.accepted), outcome.n_drafted
            )
        # One sequence at a time is the only composition the seed table
        # describes, and it is the one the cost question is asked about. A
        # cycle of four sequences is priced by the same line, which is an
        # approximation the policy states rather than hides.
        if profile.n_sequences == 1 and profile.n_rows >= 1 and profile.wall_ms > 0:
            self.cost.observe(int(profile.n_rows), float(profile.wall_ms))

    # -- what the cycle calls ---------------------------------------------
    def next_depths(self, n_sequences: int) -> list[int]:
        return self.plan_depth(
            n_sequences, self.estimator.mean_accepted, self.rows_budget
        )

    def next_depths_for(self, sequence_ids: Sequence[int]) -> list[int]:
        """Per-sequence depths. Used when the cycle can name its sequences."""
        n = len(sequence_ids)
        if n == 0:
            return []
        ceiling = self._ceiling(n, self.rows_budget)
        if not self.adaptive:
            depths = [min(self.max_depth, ceiling)] * n
        else:
            depths = [
                self.stable_depth(int(s), self.acceptance_for(int(s)), ceiling)
                for s in sequence_ids
            ]
        self.chosen = depths
        return depths


# ---------------------------------------------------------------------------
# stop conditions and streaming text
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmitResult:
    """What one committed run of tokens turned into for one sequence.

    ``keep`` is the number of tokens from the run that survive: a stop string
    or an EOS id inside the run discards the rest, and the caller truncates the
    backend state to match. ``text`` is safe to send: partial UTF-8 and any
    suffix that could still grow into a stop string are held back.
    """

    text: str
    keep: int
    finish_reason: FinishReason | None


class TextEmitter:
    """Streaming detokenisation plus stop detection for one sequence.

    Two things have to be true at once and neither is free. The client may
    never see a character of a stop string, so text is held back while a
    suffix could still become one. And the state the prefix cache is about to
    store may not contain tokens past the stop, so the emitter reports how many
    tokens of the run survive rather than only how much text to send.

    Token boundaries are tracked by feeding the tokenizer one id at a time and
    recording the length of the accumulated text after each. That is host-side
    string work at decode rates, and it is the only way to map a character
    offset (where the stop string starts) back to a token count (what the state
    has to be truncated to).
    """

    def __init__(
        self,
        tokenizer: Any,
        sequence_id: SequenceId,
        stop: StopCondition,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_id = sequence_id
        self.stop = stop
        self.text = ""
        self.emitted = 0
        self.tokens_seen = 0
        self._token_ends: list[int] = []
        self._holdback = max((len(s) for s in stop.stop_strings), default=1) - 1
        self._done = False

    # -- helpers -----------------------------------------------------------
    def _tokens_within(self, chars: int) -> int:
        """Tokens whose text ends at or before ``chars``."""
        return bisect_right(self._token_ends, chars)

    def _find_stop(self, search_from: int) -> tuple[int, int] | None:
        """Earliest (start, end) of a stop string at or after ``search_from``."""
        best: tuple[int, int] | None = None
        for needle in self.stop.stop_strings:
            if not needle:
                continue
            index = self.text.find(needle, search_from)
            if index < 0:
                continue
            if best is None or index < best[0]:
                best = (index, index + len(needle))
        return best

    def _safe_prefix(self) -> int:
        """Longest prefix of ``text`` that no stop string could still extend."""
        if not self.stop.stop_strings or self._holdback <= 0:
            return len(self.text)
        limit = len(self.text)
        tail_start = max(self.emitted, limit - self._holdback)
        for needle in self.stop.stop_strings:
            for size in range(min(len(needle) - 1, limit - tail_start), 0, -1):
                if self.text.endswith(needle[:size]):
                    limit = min(limit, len(self.text) - size)
                    break
        return max(self.emitted, limit)

    # -- the one call the cycle makes -------------------------------------
    def push(self, token_ids: Sequence[int], budget_left: int) -> EmitResult:
        """Append a committed run and decide what leaves the engine.

        ``budget_left`` is ``max_tokens`` minus the tokens already committed, so
        a cycle that accepts three tokens against a budget of one keeps one and
        the state is truncated to match. That is what makes ``max_tokens`` count
        accepted tokens rather than cycles.
        """
        if self._done:
            return EmitResult("", 0, None)
        search_from = max(0, self.emitted - self._holdback)
        base = self.tokens_seen
        keep = 0
        finish: FinishReason | None = None
        for index, token_id in enumerate(token_ids):
            if index >= budget_left:
                finish = FinishReason.LENGTH
                break
            if token_id in self.stop.eos_token_ids:
                # The EOS id itself is never text and never counts as output.
                finish = FinishReason.STOP
                break
            piece = self.tokenizer.decode_incremental(self.sequence_id, [token_id])
            self.text += piece
            self._token_ends.append(len(self.text))
            self.tokens_seen += 1
            keep += 1
            hit = self._find_stop(search_from)
            if hit is not None:
                start, _end = hit
                # Token counts are absolute over the whole stream; ``keep`` is
                # relative to this run. A stop string that began in an earlier
                # run therefore keeps nothing of this one.
                surviving = self._tokens_within(start)
                keep = max(0, surviving - base)
                self.text = self.text[:start]
                del self._token_ends[surviving:]
                self.tokens_seen = surviving
                finish = FinishReason.STOP
                break
        if finish is not None:
            self._done = True
            cut = len(self.text)
        else:
            cut = self._safe_prefix()
        text = self.text[self.emitted : cut]
        self.emitted = cut
        return EmitResult(text=text, keep=keep, finish_reason=finish)

    def flush(self) -> str:
        """Release anything held back. Called once, when the sequence ends."""
        tail = self.tokenizer.flush_incremental(self.sequence_id)
        if tail and not self._done:
            self.text += tail
        out = self.text[self.emitted :]
        self.emitted = len(self.text)
        return out


# ---------------------------------------------------------------------------
# the cycles
# ---------------------------------------------------------------------------


class _BaseCycle:
    """Commit and emit, shared by the plain loop and the MTP cycle.

    Everything that decides *what* runs is in the subclasses. Everything that
    decides what the result means -- which tokens are committed, where a stop
    lands, what the state has to be truncated to, what the client sees -- is
    here, once, so the two paths cannot produce different token streams for the
    same backend. The lossless-speculation test is exactly the assertion that
    they do not.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        clock: Any = None,
        profiler: Any = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.clock = clock or MonotonicClock()
        self.profiler = profiler or NullProfiler()
        self._emitters: dict[int, TextEmitter] = {}
        self._cycle_index = 0

    # -- emitter lifecycle -------------------------------------------------
    def emitter_for(self, sequence: SequenceState) -> TextEmitter:
        emitter = self._emitters.get(int(sequence.sequence_id))
        if emitter is None:
            emitter = TextEmitter(
                self.tokenizer, sequence.sequence_id, sequence.request.stop
            )
            self._emitters[int(sequence.sequence_id)] = emitter
        return emitter

    def release(self, sequence: SequenceState) -> str:
        """Drop a sequence's detokenisation stream and return its tail."""
        emitter = self._emitters.pop(int(sequence.sequence_id), None)
        if emitter is None:
            return self.tokenizer.flush_incremental(sequence.sequence_id)
        return emitter.flush()

    # -- verify rows -------------------------------------------------------
    @staticmethod
    def verify_candidates(
        batch: Sequence[SequenceState],
        proposals: Mapping[int, tuple[int, ...]],
    ) -> list[DraftCandidate]:
        """Build the row block: pending token first, then the drafted chain.

        The drafter proposes a continuation; it does not know about the pending
        token, and it must not, because the pending token is scheduler state.
        Prepending it here is the only place the row layout is decided.
        """
        rows: list[DraftCandidate] = []
        for sequence in batch:
            pending = sequence.tokens[-1]
            drafted = proposals.get(int(sequence.sequence_id), ())
            rows.append(
                DraftCandidate(
                    sequence_id=sequence.sequence_id,
                    tokens=(pending, *drafted),
                    source="mtp" if drafted else "none",
                )
            )
        return rows

    @staticmethod
    def draft_budget(sequence: SequenceState) -> int:
        """How many drafts this sequence may spend without overrunning its budget.

        The verify forward always produces a bonus token, so a sequence with one
        token of budget left must draft nothing: the bonus alone finishes it.
        Clamping here rather than discarding the overrun afterwards is what
        keeps the state consistent -- a token committed past ``max_tokens`` has
        to be truncated away, and a truncation inside the verify block lands
        below the snapshot the block staged, which the backend is right to
        refuse. oMLX clamps in the same place and for the same reason.
        """
        stop = sequence.request.stop
        budget = stop.max_tokens - sequence.committed - 1
        if stop.max_total_tokens is not None:
            budget = min(budget, stop.max_total_tokens - len(sequence.tokens) - 1)
        return max(0, budget)

    # -- commit ------------------------------------------------------------
    def commit(
        self,
        batch: Sequence[SequenceState],
        outcomes: Sequence[VerifyOutcome],
        proposals: Mapping[int, tuple[int, ...]],
    ) -> tuple[list[TokenEvent], list[SequenceId], int]:
        """Apply one cycle's outcomes: tokens, text, stops, truncation."""
        by_id = {int(s.sequence_id): s for s in batch}
        events: list[TokenEvent] = []
        finished: list[SequenceId] = []
        committed_total = 0
        now = self.clock.now()
        for outcome in outcomes:
            sequence = by_id[int(outcome.sequence_id)]
            drafted = proposals.get(int(sequence.sequence_id), ())
            n_accepted = len(outcome.accepted)
            if n_accepted > len(drafted):
                raise StateError(
                    f"verify accepted {n_accepted} of {len(drafted)} drafted tokens"
                )
            run = (*drafted[:n_accepted], outcome.bonus)
            committed_total += len(run)

            emitter = self.emitter_for(sequence)
            budget_left = sequence.request.stop.max_tokens - sequence.committed
            result = emitter.push(run, budget_left)
            kept = run[: result.keep]
            before = len(sequence.tokens)
            sequence.tokens.extend(kept)
            # A stop string can span cycles: its first half may have been
            # committed a cycle ago, held back as text but already in the token
            # list. The emitter's absolute survivor count is the authority on
            # what is left, so the trim can reach back past this cycle. Those
            # tokens were never emitted as text, so nothing the client saw is
            # retracted; what changes is the prefix the cache is offered.
            surviving = sequence.prompt_len + emitter.tokens_seen
            if len(sequence.tokens) > surviving:
                del sequence.tokens[surviving:]
            sequence.committed = len(sequence.tokens) - sequence.prompt_len
            sequence.text_emitted += len(result.text)
            if sequence.first_token_at is None and kept:
                sequence.first_token_at = now
            if result.text or kept:
                events.append(
                    TokenEvent(
                        request_id=sequence.request.request_id,
                        token_ids=tuple(kept),
                        text=result.text,
                        timestamp=now,
                    )
                )

            finish = result.finish_reason
            if finish is None and sequence.committed >= sequence.request.stop.max_tokens:
                finish = FinishReason.LENGTH
            if finish is None:
                total = sequence.request.stop.max_total_tokens
                if total is not None and len(sequence.tokens) >= total:
                    finish = FinishReason.LENGTH
            if len(sequence.tokens) != before + len(run):
                self._truncate_to_kept(sequence)
            if finish is not None:
                sequence.finish_reason = finish
                finished.append(sequence.sequence_id)
        return events, finished, committed_total

    def stage_prompt_end(self, sequence: SequenceState) -> None:
        """Stage the recurrent snapshot at the end of the prompt, once.

        Prefill covers ``prompt_len - 1`` tokens, because the last prompt token
        is the first decode input, so the deepest boundary prefill can stage is
        the block floor of that. The end of the prompt itself exists only after
        the first decode cycle has consumed the pending token, and it is the
        boundary that decides whether the next turn of a conversation resumes
        where this one started or a block earlier. Staging it costs one copy of
        the recurrent state and no forward pass.

        The check is on the state length rather than on a cycle counter,
        because it is the state that has to be at the boundary: a first cycle
        that committed two tokens has already passed it, and staging there
        would key a snapshot to a length it does not describe.
        """
        if not sequence.needs_prompt_end_snapshot or sequence.state is None:
            return
        stage = getattr(self.backend, "stage_snapshot", None)
        if stage is None:
            return
        if self.backend.state_length(sequence.state) != sequence.prompt_len:
            return
        try:
            stage(sequence.state, sequence.prompt_len)
        except TitanError as exc:
            self.profiler.event(
                "prompt_end_snapshot_failed",
                sequence=int(sequence.sequence_id),
                length=sequence.prompt_len,
                reason=str(exc),
            )
            return
        sequence.needs_prompt_end_snapshot = False
        sequence.prompt_end_staged = True
        self.profiler.event(
            "prompt_end_snapshot",
            sequence=int(sequence.sequence_id),
            length=sequence.prompt_len,
        )

    def first_cycle_depth(self, sequence: SequenceState) -> int:
        """Drafts this sequence may spend on the cycle that ends its prompt.

        Zero, once, for the cycle that consumes the last prompt token, and only
        when the prefill plan could not reach the prompt end itself. A verify
        block wider than one column lands the state past that boundary, and the
        boundary is then unreachable without a forward pass to put it back.

        The condition matters as much as the clamp. Most prompts already have a
        snapshot at the block floor of their end, staged by the last prefill
        chunk, and paying a cycle of speculation to stage a second one at the
        same rounded position buys nothing. The scheduler compares the plan
        against the block grid and sets the flag only when the two disagree,
        which is a prompt whose length is a block multiple, or one whose fine
        cut the cache declined.
        """
        if not sequence.needs_prompt_end_snapshot:
            return -1
        return 0 if sequence.committed == 0 else -1

    def _truncate_to_kept(self, sequence: SequenceState) -> None:
        """Undo the tokens a stop discarded, without a forward pass.

        The backend has consumed every token but the pending one, so the target
        length is ``len(tokens) - 1``. If the backend refuses -- the truncation
        point is below the snapshot the verify staged -- the sequence is still
        correct on the wire; what is lost is the right to store this prefix. The
        scheduler checks the state length against the token list before it
        stores, so a refused truncation drops the store rather than recording a
        length the cache cannot restore.
        """
        target = len(sequence.tokens) - 1
        if self.backend.state_length(sequence.state) <= target:
            return
        try:
            self.backend.truncate_state(sequence.state, target)
        except TitanError as exc:
            self.profiler.event(
                "truncate_refused",
                sequence=int(sequence.sequence_id),
                target=target,
                reason=str(exc),
            )

    # -- profile -----------------------------------------------------------
    def finish_profile(
        self,
        profile: CycleProfile,
        *,
        n_sequences: int,
        draft_ms: float,
        detok_ms: float,
        ngram_wait_ms: float,
        wall_ms: float,
        tokens_committed: int,
        tokens_drafted: int,
    ) -> CycleProfile:
        self._cycle_index += 1
        return replace(
            profile,
            cycle=self._cycle_index,
            n_sequences=n_sequences,
            draft_ms=draft_ms,
            detok_ms=detok_ms,
            ngram_wait_ms=ngram_wait_ms,
            wall_ms=wall_ms,
            tokens_committed=tokens_committed,
            tokens_drafted=tokens_drafted,
        )


class PlainDecodeCycle(_BaseCycle):
    """M1: one token per sequence per cycle, greedy, no speculation.

    The row block is one column wide, so the verify forward is a plain decode
    forward and the acceptance reduction degenerates to an argmax. It goes
    through ``verify`` rather than ``decode`` for one reason: the core may not
    read logits, and ``verify`` is the only port method that turns logits into
    token ids without handing the engine a device array. That also makes this
    loop and the MTP cycle share their sampling, which is what the parity test
    between them is worth anything for.
    """

    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        if not batch:
            return CycleResult((), (), (), EMPTY_PROFILE)
        started = self.clock.now()
        proposals: dict[int, tuple[int, ...]] = {}
        rows = self.verify_candidates(batch, proposals)
        states = [s.state for s in batch]
        sampling = [s.request.sampling for s in batch]
        outcomes, profile = self.backend.verify(states, rows, sampling)

        detok_start = self.clock.now()
        events, finished, committed = self.commit(batch, outcomes, proposals)
        detok_ms = (self.clock.now() - detok_start) * 1000.0
        for sequence in batch:
            self.stage_prompt_end(sequence)
        wall_ms = (self.clock.now() - started) * 1000.0
        profile = self.finish_profile(
            profile,
            n_sequences=len(batch),
            draft_ms=0.0,
            detok_ms=detok_ms,
            ngram_wait_ms=0.0,
            wall_ms=wall_ms,
            tokens_committed=committed,
            tokens_drafted=0,
        )
        self.profiler.cycle(profile)
        return CycleResult(tuple(outcomes), tuple(events), tuple(finished), profile)


class MTPDecodeCycle(_BaseCycle):
    """M4: draft a chain of depth k, verify k+1 rows, accept in the graph.

    Sequence of one cycle, and the order is load-bearing:

    1. the depth controller hands out a depth per sequence from the rolling
       acceptance estimate;
    2. the drafter proposes, without syncing -- the chain is dispatched and
       left on the queue;
    3. the n-gram rows the draft implies are queued for prefetch, so the SSD
       read overlaps the verify forward rather than adding 2.3 ms in front of
       it;
    4. one verify forward over the padded row block, one host sync, acceptance
       and the bonus token computed in the graph by ``verify_accept``;
    5. commit, with rollback already done inside verify (the port's contract:
       state grows by exactly ``len(accepted) + 1``, never by the block width);
    6. one CycleProfile, always, with ``host_syncs`` straight from the backend.

    ### The batched-verify seam

    This class is already written for a batch: it plans a depth per sequence,
    builds one row per sequence, and makes exactly one ``verify`` call for the
    whole batch. What is not here yet is a common width. ``plan_depth`` returns
    a uniform depth today, so the block is rectangular by construction and the
    backend's padding never engages. Lockstep across ragged depths is a local
    change in two places and nowhere else: ``plan_depth`` may return different
    depths per sequence, and the backend pads the short rows. Nothing in
    ``commit`` knows the width -- it reads ``len(outcome.accepted)`` and the
    sequence's own proposal -- which is what keeps the seam local.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        drafter: Any = None,
        verifier: Any = None,
        ngram: Any = None,
        clock: Any = None,
        profiler: Any = None,
        max_depth: int = 3,
        rows_budget: int = 32,
        overlap_draft: bool = False,
    ) -> None:
        super().__init__(
            backend=backend, tokenizer=tokenizer, clock=clock, profiler=profiler
        )
        self.drafter = drafter
        self.ngram = ngram
        self.overlap_draft = bool(overlap_draft) and hasattr(drafter, "dispatch")
        """Dispatch the next cycle's chain at the end of this one.

        The chain's GPU work is then enqueued behind this cycle's verify rather
        than in front of the next one's, so the wait for it overlaps the
        commit, the detokenisation and the next cycle's depth planning instead
        of stalling in the middle of the next cycle.

        It changes nothing about what is drafted: the fold needs the tokens
        this cycle committed, which the commit has just produced, so the chain
        dispatched here is the same chain the next cycle would have built.
        """
        self._pending: Any = None
        self.overlap_hits = 0
        self.overlap_misses = 0
        self.controller = verifier or DepthController(
            max_depth=min(max_depth, getattr(backend, "draft_depth_max", max_depth)),
            rows_budget=rows_budget,
        )

    # -- steps -------------------------------------------------------------
    def _plan_depths(self, batch: Sequence[SequenceState]) -> list[int]:
        """Ask the controller for a depth per sequence.

        Two shapes of controller, one call site. The ``Verifier`` port only
        promises ``plan_depth(n_sequences, ...)``, which cannot express a
        per-sequence answer; a controller that keeps a per-sequence acceptance
        curve offers ``next_depths_for`` as well, and the difference matters
        the moment a 64k session shares the batch with a short one.
        """
        per_sequence = getattr(self.controller, "next_depths_for", None)
        if per_sequence is not None:
            return list(per_sequence([int(s.sequence_id) for s in batch]))
        return list(self.controller.next_depths(len(batch)))

    def _depths_for(self, batch: Sequence[SequenceState]) -> list[int]:
        depths = []
        for sequence, depth in zip(batch, self._plan_depths(batch)):
            depth = min(depth, self.draft_budget(sequence))
            cap = self.first_cycle_depth(sequence)
            if cap >= 0:
                depth = min(depth, cap)
            depths.append(depth)
        return depths

    def _propose(
        self, batch: Sequence[SequenceState]
    ) -> tuple[dict[int, tuple[int, ...]], int]:
        if self.drafter is None:
            return {}, 0
        states = [s.state for s in batch]
        contexts = [s.tokens for s in batch]

        pending, self._pending = self._pending, None
        if pending is not None and pending.matches(states, contexts):
            # The chain this batch needs is already on the queue.
            self.overlap_hits += 1
            candidates = self.drafter.read(pending)
            depths = [len(c.tokens) for c in candidates]
        else:
            if pending is not None:
                # Dispatched a cycle ago for a batch that no longer exists: a
                # sequence finished, one was admitted, or a stop truncated a
                # context. Dropped rather than read. The fold it did is still
                # sound -- it was of committed tokens -- so the only cost is
                # this cycle's draft, and the drafter answers the re-proposal
                # with nothing because there is nothing left uncommitted.
                self.overlap_misses += 1
            depths = self._depths_for(batch)
            if not any(depths):
                return {}, 0
            candidates = self.drafter.propose(states, contexts, depths)
        proposals: dict[int, tuple[int, ...]] = {}
        drafted = 0
        # Keyed by the batch's own order rather than by the candidate's
        # sequence_id: the drafter is handed states and contexts, so it has no
        # authoritative id to return, and trusting one it invented is how a
        # draft ends up on the wrong sequence.
        for sequence, candidate, depth in zip(batch, candidates, depths):
            tokens = tuple(candidate.tokens)[:depth]
            if tokens:
                proposals[int(sequence.sequence_id)] = tokens
                drafted += len(tokens)
        return proposals, drafted

    def _prefetch(self, proposals: Mapping[int, tuple[int, ...]]) -> None:
        """Queue the n-gram rows the draft needs, before the verify forward."""
        if self.ngram is None or not proposals:
            return
        rows: list[int] = []
        for tokens in proposals.values():
            rows.extend(int(t) for t in tokens)
        self.ngram.prefetch(rows)

    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        if not batch:
            return CycleResult((), (), (), EMPTY_PROFILE)
        started = self.clock.now()

        draft_start = self.clock.now()
        if drafts:
            proposals = {int(d.sequence_id): tuple(d.tokens) for d in drafts}
            drafted = sum(len(t) for t in proposals.values())
        else:
            proposals, drafted = self._propose(batch)
        draft_ms = (self.clock.now() - draft_start) * 1000.0

        ngram_start = self.clock.now()
        self._prefetch(proposals)
        ngram_wait_ms = (self.clock.now() - ngram_start) * 1000.0

        rows = self.verify_candidates(batch, proposals)
        states = [s.state for s in batch]
        sampling = [s.request.sampling for s in batch]
        outcomes, profile = self.backend.verify(states, rows, sampling)

        detok_start = self.clock.now()
        events, finished, committed = self.commit(batch, outcomes, proposals)
        detok_ms = (self.clock.now() - detok_start) * 1000.0
        for sequence in batch:
            self.stage_prompt_end(sequence)

        if self.drafter is not None and hasattr(self.drafter, "observe"):
            self.drafter.observe(outcomes)

        if self.overlap_draft:
            self._dispatch_next(batch, finished)

        wall_ms = (self.clock.now() - started) * 1000.0
        profile = self.finish_profile(
            profile,
            n_sequences=len(batch),
            draft_ms=draft_ms,
            detok_ms=detok_ms,
            ngram_wait_ms=ngram_wait_ms,
            wall_ms=wall_ms,
            tokens_committed=committed,
            tokens_drafted=drafted,
        )
        # After the profile is finished, not before: the depth policy prices
        # widths against the cycle's own wall time, and the backend's half of
        # the profile does not have one. A policy fitted to the verify time
        # alone would price away the 15.8 ms of host graph build that is most
        # of what a narrow cycle costs.
        self.controller.record(profile, outcomes)
        self.profiler.cycle(profile)
        return CycleResult(tuple(outcomes), tuple(events), tuple(finished), profile)

    def _dispatch_next(
        self,
        batch: Sequence[SequenceState],
        finished: Sequence[Any],
    ) -> None:
        """Enqueue the next cycle's chain, for the sequences that have one.

        Runs after commit, which is the earliest point the fold's inputs exist,
        and before the profile, which is the latest point that still leaves the
        GPU something to do while the host finishes the cycle.

        A sequence that finished this cycle is left out, and if that empties
        the batch nothing is dispatched. Anything else that moves the batch
        between here and the next cycle is caught by ``_Dispatch.matches``
        rather than predicted here: the scheduler admits sequences on its own
        turn and this class does not get a say.
        """
        done = {int(getattr(f, "sequence_id", f)) for f in finished}
        live = [s for s in batch if int(s.sequence_id) not in done]
        if not live:
            return
        depths = self._depths_for(live)
        if not any(depths):
            return
        self._pending = self.drafter.dispatch(
            [s.state for s in live], [s.tokens for s in live], depths
        )
