# SPDX-License-Identifier: MIT
"""Row accounting in ``MLXModelBackend.verify``, against a fake model.

The engine hands the backend whole rows: ``verify_candidates`` prepends the
pending token, so a candidate carrying ``(pending, d1, d2)`` is a row of width
three with two drafts in it. Everything the backend reports back is measured
from that layout, and getting it wrong is not a crash but a quiet
off-by-one: the pending token counted as a draft, an accepted run shifted one
token early, and a drafted count that never matches what the drafter proposed.

The fake model here returns logits chosen so that the target's argmax at each
column is known, which is what lets a test say exactly how many drafts should
survive. No weights, no GPU work beyond a few small arrays.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from titan.adapters.mlx.backend import MLXModelBackend
from titan.core.types import DraftCandidate, SamplingParams, SequenceId

VOCAB = 16


class FakeState:
    """What the backend's handle table holds. Only length and truncation."""

    def __init__(self) -> None:
        self.length = 10
        self.rows = 1
        self.mtp_hidden = None
        self.truncations: list[int] = []
        self.snapshots: list[int] = []

    def truncate(self, length: int) -> None:
        if length > self.length:
            raise AssertionError("verify must never grow a state by truncating")
        self.truncations.append(length)
        self.length = length

    def stage_snapshot(self, length=None, *, pinned: bool = False) -> None:
        self.snapshots.append(self.length if length is None else length)

    def prune_snapshots(self, keep: int) -> int:
        dropped = [n for n in self.snapshots if n != keep]
        self.snapshots = [n for n in self.snapshots if n == keep]
        self.pruned = getattr(self, "pruned", 0) + len(dropped)
        return len(dropped)


class FakeModel:
    """Returns a fixed argmax per column, per row."""

    draft_depth_max = 3
    n_layers = 4
    vocab_size = VOCAB
    max_context = 1024

    def __init__(self, targets: list[list[int]]):
        self.targets = targets
        self.rows_seen: list[list[int]] = []
        self.rollbacks: list[tuple[list[int], int]] = []
        self.calls = 0
        self.snapshots_asked: list[bool] = []

    def new_state(self) -> FakeState:
        return FakeState()

    def verify(self, rows, state, snapshot=False, want_hidden=True):
        from titan.adapters.mlx.model import VerifyResult

        self.rows_seen = [list(r) for r in rows]
        self.snapshots_asked.append(bool(snapshot))
        width = len(rows[0])
        state.length += width
        logits = mx.zeros((len(rows), width, VOCAB))
        # One target row per forward, in call order, so two sequences in one
        # cycle can be told apart.
        picks = []
        for _row_index in range(len(rows)):
            picks.append(self.targets[self.calls % len(self.targets)][:width])
            self.calls += 1
        # One-hot the intended argmax at every column.
        index = mx.array(picks, dtype=mx.int32)[:, :, None]
        return VerifyResult(
            logits=mx.put_along_axis(
                logits, index, mx.ones((len(rows), width, 1)), axis=-1
            ),
            gdn_states=["captured"],
        )

    def rollback_verify(self, state, gdn_states, accepted, width) -> None:
        self.rollbacks.append((list(accepted), width))
        kept = max(accepted) + 1
        state.length -= width - kept


def _backend(targets: list[list[int]]) -> tuple[MLXModelBackend, FakeModel]:
    model = FakeModel(targets)
    return MLXModelBackend(model), model


def _verify(backend: MLXModelBackend, handle, tokens: tuple[int, ...]):
    draft = DraftCandidate(sequence_id=SequenceId(1), tokens=tokens, source="mtp")
    outcomes, profile = backend.verify([handle], [draft], [SamplingParams()])
    return outcomes[0], profile


def test_the_pending_token_is_not_a_draft():
    """Width three, two drafts. A row of width one drafts nothing at all."""
    backend, model = _backend([[9, 9, 9]])
    handle = backend.open_state(SequenceId(1), 64)
    outcome, _ = _verify(backend, handle, (5,))
    assert outcome.n_drafted == 0
    assert outcome.accepted == ()
    assert model.rows_seen == [[5]]


def test_every_draft_accepted_reports_the_whole_chain():
    # The target's argmax after the pending token is 7 then 8, which is exactly
    # what was drafted, so both survive and the bonus is the third column.
    backend, _model = _backend([[7, 8, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    outcome, _ = _verify(backend, handle, (5, 7, 8))
    assert outcome.n_drafted == 2
    assert outcome.accepted == (7, 8)
    assert outcome.bonus == 3
    assert outcome.n_committed == 3


def test_a_rejection_keeps_the_prefix_and_takes_the_bonus_from_there():
    # The target wants 7 then 4; the draft said 7 then 8, so one draft sticks
    # and the bonus is the target's own token at the rejection point.
    backend, _model = _backend([[7, 4, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    outcome, _ = _verify(backend, handle, (5, 7, 8))
    assert outcome.n_drafted == 2
    assert outcome.accepted == (7,)
    assert outcome.bonus == 4


def test_the_first_draft_rejected_commits_only_the_bonus():
    backend, _model = _backend([[1, 4, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    outcome, _ = _verify(backend, handle, (5, 7, 8))
    assert outcome.accepted == ()
    assert outcome.bonus == 1
    assert outcome.n_committed == 1


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
def test_the_state_grows_by_exactly_what_was_committed(depth):
    """The port's second invariant: on return the state covers
    ``len(accepted) + 1`` more tokens than it did on entry, whatever the block
    width was and whatever the drafter proposed."""
    backend, _model = _backend([[7] * (depth + 1)])
    handle = backend.open_state(SequenceId(1), 64)
    state = backend._state(handle)
    entry = state.length
    outcome, _ = _verify(backend, handle, (5, *([7] * depth)))
    assert outcome.n_drafted == depth
    assert backend.state_length(handle) == entry + len(outcome.accepted) + 1


def test_short_rows_are_padded_to_the_block_width():
    """Padding is width, never content. The extra columns sit past the first
    rejection, so they are discarded with the rest of the rejected suffix and
    cannot change the row's own output."""
    from titan.adapters.mlx.backend import _pad_draft

    short = DraftCandidate(sequence_id=SequenceId(2), tokens=(5,), source="none")
    assert _pad_draft(short, 3) == [5, 5, 5]
    full = DraftCandidate(sequence_id=SequenceId(1), tokens=(5, 7, 8), source="mtp")
    assert _pad_draft(full, 3) == [5, 7, 8]


def test_one_host_sync_per_cycle():
    backend, _model = _backend([[7, 8, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    _outcome, profile = _verify(backend, handle, (5, 7, 8))
    assert profile.host_syncs == 1
    assert profile.tokens_drafted == 2


# ---------------------------------------------------------------------------
# staged snapshots
# ---------------------------------------------------------------------------


def test_rollback_copies_do_not_accumulate():
    """A verify stages a recurrent snapshot before every forward, and one is
    around 110 MiB. Six hundred tokens of generation would be 66 GB of copies
    nothing can use again, so every cycle drops the ones it superseded."""
    from titan.adapters.mlx.state import ModelState

    state = ModelState(layers=[], length=0)
    for length in range(0, 5):
        state.length = length
        state.stage_snapshot()
        state.prune_snapshots(length)
        assert set(state.snapshots) == {length}


def test_a_pinned_boundary_survives_every_prune():
    """The plan's boundaries are what the prefix cache is asked to serialise at
    retirement, so they outlive every rollback copy between here and there."""
    from titan.adapters.mlx.state import ModelState

    state = ModelState(layers=[], length=2048)
    state.stage_snapshot(pinned=True)
    for length in (2049, 2050, 2051):
        state.length = length
        state.stage_snapshot()
        state.prune_snapshots(length)
    assert sorted(state.snapshots) == [2048, 2051]
    assert state.snapshots[2048].pinned
    assert not state.snapshots[2051].pinned


def test_staging_again_at_a_pinned_length_keeps_the_pin():
    from titan.adapters.mlx.state import ModelState

    state = ModelState(layers=[], length=512)
    state.stage_snapshot(pinned=True)
    state.stage_snapshot()
    assert state.snapshots[512].pinned


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_a_block_that_accepted_everything_rolls_back_nothing():
    """The forward left the state exactly where it belongs. The cheapest
    correct rollback of nothing is not doing one."""
    backend, model = _backend([[7, 8, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    _verify(backend, handle, (5, 7, 8))
    assert model.rollbacks == []


def test_a_rejection_rolls_the_block_back_once_for_the_whole_batch():
    backend, model = _backend([[7, 4, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    _verify(backend, handle, (5, 7, 8))
    assert model.rollbacks == [([1], 3)]


def test_a_one_column_block_stages_no_snapshot():
    """A snapshot is a full copy of the recurrent state, around 110 MiB. There
    is nothing inside a one-column block to roll back to, so paying for one on
    every plain decode cycle buys nothing."""
    backend, model = _backend([[9]])
    handle = backend.open_state(SequenceId(1), 64)
    _verify(backend, handle, (5,))
    assert model.snapshots_asked == [False]


def test_a_speculative_block_stages_one():
    backend, model = _backend([[7, 8, 3]])
    handle = backend.open_state(SequenceId(1), 64)
    _verify(backend, handle, (5, 7, 8))
    assert model.snapshots_asked == [True]


def test_two_sequences_in_one_cycle_each_get_their_own_answer():
    """Lockstep batched verify is not finished, and the fallback has to be
    correct rather than fast: each sequence's row is dispatched against its own
    state, and one host sync still covers the whole cycle."""
    backend, model = _backend([[11, 0, 0], [12, 0, 0]])
    handles = [backend.open_state(SequenceId(i), 64) for i in (1, 2)]
    drafts = [
        DraftCandidate(sequence_id=SequenceId(1), tokens=(5,), source="none"),
        DraftCandidate(sequence_id=SequenceId(2), tokens=(6,), source="none"),
    ]
    outcomes, profile = backend.verify(handles, drafts, [SamplingParams()] * 2)
    assert [o.bonus for o in outcomes] == [11, 12]
    assert profile.host_syncs == 1
    assert profile.n_sequences == 2
