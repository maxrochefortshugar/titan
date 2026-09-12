"""The lossless-speculation invariant.

Speculation is allowed to change how fast tokens arrive and nothing else. For
every acceptance pattern -- every draft accepted, none accepted, rejected at
each position in the chain, and random mixtures of all three -- the MTP cycle
must emit the same token stream as the plain greedy loop, on the same backend,
from the same prompt.

This is the M4a acceptance criterion (``tests/parity/test_m4_chain.py`` will
say the same thing against the real model), and it is checked here
exhaustively, because a speculative decoder that is right 99.9% of the time is
a corrupted decoder that is hard to catch.
"""

from __future__ import annotations

import random

import pytest

from titan.core.types import SequencePhase, SequenceState
from titan.engine.decode_cycle import (
    AcceptanceEstimator,
    DepthController,
    MTPDecodeCycle,
    PlainDecodeCycle,
)

from tests.engine.conftest import (
    FakeBackend,
    FakeTokenizer,
    ScriptedDrafter,
    make_request,
)
from tests.engine.test_greedy_loop import decoding_sequence


def run_plain(prompt, count, *, vocab=64):
    backend = FakeBackend(vocab=vocab)
    request = make_request(prompt, max_tokens=count)
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer())
    while sequence.finish_reason is None:
        cycle.run([sequence])
    return backend, sequence


def run_mtp(prompt, count, policy, *, depth=3, vocab=64):
    backend = FakeBackend(vocab=vocab)
    request = make_request(prompt, max_tokens=count)
    sequence = decoding_sequence(backend, request)
    drafter = ScriptedDrafter(policy)
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(),
        drafter=drafter,
        verifier=DepthController(max_depth=depth, min_depth=depth, adaptive=False),
    )
    profiles = []
    while sequence.finish_reason is None:
        profiles.append(cycle.run([sequence]).profile)
    return backend, sequence, profiles


# ---------------------------------------------------------------------------
# draft policies, one per acceptance pattern
# ---------------------------------------------------------------------------


def perfect(backend):
    """Every drafted token is the one the target would have produced."""

    def policy(context, depth):
        tokens = list(context)
        out = []
        for _ in range(depth):
            token = backend.argmax_after(tokens)
            tokens.append(token)
            out.append(token)
        return out

    return policy


def wrong_at(backend, position: int):
    """Correct up to ``position``, wrong there, arbitrary after."""

    def policy(context, depth):
        chain = perfect(backend)(context, depth)
        if position < len(chain):
            chain[position] = (chain[position] + 1) % backend.vocab_size
        return chain

    return policy


def always_wrong(backend):
    return wrong_at(backend, 0)


def random_draft(backend, seed: int):
    rng = random.Random(seed)

    def policy(context, depth):
        chain = perfect(backend)(context, depth)
        return [
            token if rng.random() < 0.5 else rng.randrange(backend.vocab_size)
            for token in chain
        ]

    return policy


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------


PROMPT = (11, 12, 13)
COUNT = 32


@pytest.mark.parametrize("depth", list(range(1, 9)))
def test_perfect_drafts_produce_the_greedy_stream(depth):
    _plain_backend, plain = run_plain(PROMPT, COUNT)
    backend, sequence, profiles = run_mtp(
        PROMPT, COUNT, perfect(FakeBackend()), depth=depth
    )
    assert sequence.tokens == plain.tokens
    # A perfect chain of depth k commits k+1 tokens a cycle, so the whole run
    # takes a fraction of the cycles the plain loop needs.
    assert len(profiles) <= -(-COUNT // (depth + 1)) + 1


@pytest.mark.parametrize("depth", list(range(1, 9)))
def test_zero_acceptance_produces_the_greedy_stream(depth):
    _plain_backend, plain = run_plain(PROMPT, COUNT)
    _backend, sequence, profiles = run_mtp(
        PROMPT, COUNT, always_wrong(FakeBackend()), depth=depth
    )
    assert sequence.tokens == plain.tokens
    assert len(profiles) == COUNT  # one token a cycle, exactly like the plain loop


@pytest.mark.parametrize("depth", list(range(2, 9)))
@pytest.mark.parametrize("position", list(range(0, 8)))
def test_rejection_at_every_position_produces_the_greedy_stream(depth, position):
    if position >= depth:
        pytest.skip("the rejection point has to be inside the chain")
    _plain_backend, plain = run_plain(PROMPT, COUNT)
    _backend, sequence, _profiles = run_mtp(
        PROMPT, COUNT, wrong_at(FakeBackend(), position), depth=depth
    )
    assert sequence.tokens == plain.tokens


@pytest.mark.parametrize("seed", list(range(24)))
def test_random_drafts_produce_the_greedy_stream(seed):
    """The exhaustive arm: random draft/target pairs over random depths."""
    rng = random.Random(seed)
    depth = rng.randrange(1, 9)
    prompt = tuple(rng.randrange(64) for _ in range(rng.randrange(1, 6)))
    count = rng.randrange(5, 40)
    _plain_backend, plain = run_plain(prompt, count)
    _backend, sequence, _profiles = run_mtp(
        prompt, count, random_draft(FakeBackend(), seed), depth=depth
    )
    assert sequence.tokens == plain.tokens
    assert sequence.committed == plain.committed


def test_the_state_is_identical_to_never_having_drafted():
    """Replay-free rollback, checked as state equality rather than as timing.

    After a run with rejections, the backend's token history must be exactly
    what the plain loop left behind. A rollback that replayed, double-committed
    or kept a rejected token would show up here as a different state.
    """
    plain_backend, plain = run_plain(PROMPT, COUNT)
    mtp_backend, sequence, _profiles = run_mtp(
        PROMPT, COUNT, random_draft(FakeBackend(), 99), depth=4
    )
    plain_state = plain_backend.state_tokens(plain.state)
    mtp_state = mtp_backend.state_tokens(sequence.state)
    assert mtp_state == plain_state
    assert mtp_state == sequence.tokens[:-1]


def test_every_cycle_syncs_the_host_exactly_once():
    _backend, _sequence, profiles = run_mtp(
        PROMPT, COUNT, random_draft(FakeBackend(), 5), depth=3
    )
    assert profiles
    assert all(profile.host_syncs == 1 for profile in profiles)
    assert profiles[0].n_rows == 4  # pending token plus a chain of three
    # The tail cycles draft less, because the budget clamp will not let a chain
    # commit past max_tokens.
    assert all(1 <= profile.n_rows <= 4 for profile in profiles)


def test_the_profile_counts_what_the_cycle_did():
    _backend, sequence, profiles = run_mtp(
        PROMPT, COUNT, perfect(FakeBackend()), depth=3
    )
    assert sum(p.tokens_committed for p in profiles) >= sequence.committed
    assert profiles[0].tokens_drafted == 3
    assert all(p.tokens_drafted <= 3 for p in profiles)
    assert all(p.n_sequences == 1 for p in profiles)


# ---------------------------------------------------------------------------
# adaptive depth
# ---------------------------------------------------------------------------


def test_depth_falls_when_nothing_is_accepted():
    backend = FakeBackend()
    request = make_request(PROMPT, max_tokens=1000)
    sequence = decoding_sequence(backend, request)
    drafter = ScriptedDrafter(always_wrong(FakeBackend()))
    controller = DepthController(max_depth=6, min_depth=1, window=8)
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(),
        drafter=drafter,
        verifier=controller,
    )
    for _ in range(30):
        cycle.run([sequence])
    assert drafter.depths[0] == [6]
    assert drafter.depths[-1] == [1]
    assert controller.estimator.mean_accepted == 0.0


def test_depth_rises_when_everything_is_accepted():
    backend = FakeBackend()
    request = make_request(PROMPT, max_tokens=1000)
    sequence = decoding_sequence(backend, request)
    drafter = ScriptedDrafter(perfect(FakeBackend()))
    controller = DepthController(max_depth=6, min_depth=1, window=8)
    controller.estimator = AcceptanceEstimator(window=8, initial=0.0)
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(),
        drafter=drafter,
        verifier=controller,
    )
    for _ in range(30):
        cycle.run([sequence])
    assert drafter.depths[0] == [1]
    assert drafter.depths[-1] == [6]


def test_the_row_budget_caps_depth_before_the_ceiling_does():
    controller = DepthController(max_depth=8, min_depth=0, rows_budget=8)
    # Eight rows across four sequences is two rows each: one pending token and
    # one drafted token.
    assert controller.plan_depth(4, 7.0, 8) == [1, 1, 1, 1]
    assert controller.plan_depth(8, 7.0, 8) == [0] * 8


def test_the_estimator_is_a_window_not_a_lifetime():
    estimator = AcceptanceEstimator(window=4, initial=3.0)
    assert estimator.mean_accepted == 3.0
    from titan.core.types import VerifyOutcome

    for accepted in (3, 3, 3, 3, 0, 0, 0, 0):
        estimator.observe(
            [
                VerifyOutcome(
                    sequence_id=1,
                    accepted=tuple(range(accepted)),
                    bonus=1,
                    n_drafted=3,
                )
            ]
        )
    assert estimator.mean_accepted == 0.0
    assert estimator.n_samples == 4
