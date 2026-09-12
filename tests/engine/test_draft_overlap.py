# SPDX-License-Identifier: MIT
"""Dispatching the next cycle's draft chain at the end of this one.

The claim the overlap makes is narrow and the tests here are the claim: the
chain dispatched at the end of cycle N is the chain cycle N+1 would have built,
so the output does not move; and a dispatch whose batch has changed underneath
it is dropped rather than read onto the wrong sequence, which is the only way
this could go silently wrong.

The drafter is a fake that records the two halves separately, so a test can see
which cycle each dispatch belonged to.
"""

from __future__ import annotations

from typing import Sequence

from titan.core.types import DraftCandidate
from titan.engine.decode_cycle import MTPDecodeCycle

from .conftest import FakeBackend, FakeClock, FakeTokenizer, ScriptedDrafter, make_request
from .test_greedy_loop import decoding_sequence


class SplitDrafter(ScriptedDrafter):
    """A ``ScriptedDrafter`` with the dispatch/read seam the overlap needs."""

    class Pending:
        def __init__(self, states, contexts, candidates) -> None:
            self.states = tuple(int(h) for h in states)
            self.lengths = tuple(len(c) for c in contexts)
            self.candidates = candidates

        def matches(self, states, contexts) -> bool:
            return self.states == tuple(
                int(h) for h in states
            ) and self.lengths == tuple(len(c) for c in contexts)

    def __init__(self, policy) -> None:
        super().__init__(policy)
        self.dispatched: list[tuple[int, ...]] = []
        self.reads = 0

    def dispatch(self, states, contexts, depth) -> "SplitDrafter.Pending":
        self.dispatched.append(tuple(len(c) for c in contexts))
        candidates = super().propose(states, contexts, depth)
        return self.Pending(states, contexts, candidates)

    def read(self, pending) -> list[DraftCandidate]:
        self.reads += 1
        return pending.candidates


def perfect(context: Sequence[int], depth: int) -> list[int]:
    """The backend's own continuation, so every draft is accepted."""
    return [context[-1] + 1 + i for i in range(depth)]


def build(drafter, *, overlap: bool):
    backend = FakeBackend()
    return (
        MTPDecodeCycle(
            backend=backend,
            tokenizer=FakeTokenizer({}),
            drafter=drafter,
            clock=FakeClock(),
            overlap_draft=overlap,
        ),
        backend,
    )


def sequences(backend, prompt=(1, 2, 3), n=1, max_tokens=64, first_id=1):
    return [
        decoding_sequence(
            backend, make_request(prompt, max_tokens=max_tokens), first_id + index
        )
        for index in range(n)
    ]


# ---------------------------------------------------------------------------


def test_overlap_is_off_unless_asked_for():
    drafter = SplitDrafter(perfect)
    cycle, _backend = build(drafter, overlap=False)
    assert cycle.overlap_draft is False


def test_a_drafter_without_the_seam_cannot_turn_it_on():
    """The flag is a request, not an assertion: an older drafter still works."""
    drafter = ScriptedDrafter(perfect)
    cycle, _backend = build(drafter, overlap=True)
    assert cycle.overlap_draft is False


def test_the_second_cycle_reads_the_chain_the_first_dispatched():
    drafter = SplitDrafter(perfect)
    cycle, backend = build(drafter, overlap=True)
    batch = sequences(backend)

    cycle.run(batch)
    assert drafter.reads == 0
    assert len(drafter.dispatched) == 1

    cycle.run(batch)
    assert drafter.reads == 1
    assert cycle.overlap_hits == 1
    assert cycle.overlap_misses == 0


def test_the_overlap_does_not_change_what_is_committed():
    """The same prompt, the same drafter, with and without. Same tokens."""
    out = []
    for overlap in (False, True):
        drafter = SplitDrafter(perfect)
        cycle, backend = build(drafter, overlap=overlap)
        batch = sequences(backend)
        for _ in range(6):
            cycle.run(batch)
        out.append(tuple(batch[0].tokens))
    assert out[0] == out[1]


def test_a_dispatch_whose_batch_moved_is_dropped_not_read():
    """A second sequence joins between the dispatch and the cycle that would
    have read it. Reading it would put sequence one's draft on sequence two."""
    drafter = SplitDrafter(perfect)
    cycle, backend = build(drafter, overlap=True)
    one = sequences(backend, n=1)
    cycle.run(one)
    assert len(drafter.dispatched) == 1

    two = one + sequences(backend, prompt=(9, 9, 9), n=1, first_id=2)
    cycle.run(two)
    assert cycle.overlap_misses == 1
    assert cycle.overlap_hits == 0


def test_nothing_is_dispatched_for_a_batch_that_all_finished():
    drafter = SplitDrafter(perfect)
    cycle, backend = build(drafter, overlap=True)
    batch = sequences(backend, max_tokens=1)
    result = cycle.run(batch)
    assert result.finished
    assert drafter.dispatched == []
