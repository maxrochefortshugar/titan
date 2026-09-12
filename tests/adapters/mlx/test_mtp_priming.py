# SPDX-License-Identifier: MIT
"""Folding the prompt into the MTP head during prefill.

The head predicts token *t+2* from the trunk's hidden at position *t* and the
embedding of token *t+1*, so the whole question here is an indexing one: does
the chunk loop hand the head every pair the prompt contains, exactly once, in
order, across chunk boundaries and across the boundary between the prompt and
the pending token that prefill deliberately does not consume. Get that wrong by
one and the head's cache is a mirror of a sequence that never existed, which
costs acceptance silently -- the drafter's numerics cannot reach the output, so
nothing else in the system will notice.

The head is faked and records what it was fed. The real thing here is
``TitanQwenFlashNext.prefill``.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from titan.adapters.mlx.model import TitanQwenFlashNext
from titan.adapters.mlx.state import ModelState

HIDDEN = 4
HC = 2


class FakeHeadCache:
    def __init__(self) -> None:
        self.offset = 0


class FakeMTPModule:
    """Records every fold: the ids it was given and the hidden width."""

    def __init__(self) -> None:
        self.folds: list[tuple[list[int], int]] = []

    def __call__(self, hidden, ids, embed, cache, position_ids=None):
        ids_list = [int(v) for v in mx.array(ids).reshape(-1).tolist()]
        assert hidden.shape[1] == len(ids_list), (
            "a fold pairs one hidden state with one token id"
        )
        self.folds.append((ids_list, hidden.shape[1]))
        for entry in cache:
            entry.offset += len(ids_list)
        return hidden, hidden

    @property
    def folded_ids(self) -> list[int]:
        out: list[int] = []
        for ids, _ in self.folds:
            out.extend(ids)
        return out


class FakeTrunk:
    """A language model that returns one hidden row per input token."""

    def __init__(self, *, with_head: bool = True) -> None:
        self.mtp = FakeMTPModule() if with_head else None
        self.model = SimpleNamespace(embed_tokens=lambda ids: ids)
        self.calls: list[dict] = []

    def get_mtp_module(self):
        return self.mtp

    def make_cache(self):
        return [FakeHeadCache()]

    def make_mtp_cache(self):
        return [FakeHeadCache()]

    def __call__(self, piece, cache=None, return_hidden=False, skip_logits=True):
        self.calls.append(
            {"length": piece.shape[1], "return_hidden": bool(return_hidden)}
        )
        hidden = (
            mx.zeros((1, piece.shape[1], HC * HIDDEN)) if return_hidden else None
        )
        return SimpleNamespace(
            logits=mx.zeros((1, piece.shape[1], 8)),
            hidden_states=[hidden] if hidden is not None else None,
        )

    def prefetch_ple(self, *args, **kwargs):
        return None


def make(with_head: bool = True, chunk: int = 4):
    trunk = FakeTrunk(with_head=with_head)
    args = SimpleNamespace(
        hc_count=HC,
        hidden_size=HIDDEN,
        vocab_size=8,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        max_position_embeddings=1024,
    )
    model = TitanQwenFlashNext(
        SimpleNamespace(
            language_model=trunk, config=SimpleNamespace(text_config=args)
        ),
        prefill_chunk=chunk,
    )
    state = ModelState(layers=[FakeHeadCache()], mtp_layers=[FakeHeadCache()])
    return trunk, model, state


# ---------------------------------------------------------------------------


def test_priming_is_off_by_default_and_asks_for_no_hidden():
    trunk, model, state = make()
    model.prefill([1, 2, 3, 4, 5], state, want_logits=False)
    assert trunk.mtp.folds == []
    assert not any(call["return_hidden"] for call in trunk.calls)


def test_one_chunk_folds_every_pair_the_prompt_contains():
    """Nine tokens consumed, the tenth pending: nine pairs, ids 2 through 10.

    Pair *t* is ``(hidden[t], token[t+1])``, so the ids the head sees are the
    prompt shifted by one, and the last of them is the pending token that
    prefill itself never runs.
    """
    trunk, model, state = make(chunk=64)
    model.prefill(list(range(1, 10)), state, prime_mtp=True, next_token=10)
    assert trunk.mtp.folded_ids == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert state.mtp_layers[0].offset == 9


def test_pairs_run_across_chunk_boundaries_without_a_gap_or_a_repeat():
    """Three chunks of four, and the seam is where an off-by-one would live."""
    trunk, model, state = make(chunk=4)
    model.prefill(list(range(1, 13)), state, prime_mtp=True, next_token=13)
    assert [ids for ids, _ in trunk.mtp.folds] == [
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
    ]
    assert state.mtp_layers[0].offset == 12


def test_without_the_pending_token_the_last_pair_is_left_open():
    """The head ends one entry short rather than folding a token it invented."""
    trunk, model, state = make(chunk=64)
    model.prefill([1, 2, 3, 4], state, prime_mtp=True)
    assert trunk.mtp.folded_ids == [2, 3, 4]
    assert state.mtp_layers[0].offset == 3


def test_a_single_token_prefill_with_a_pending_token_folds_one_pair():
    trunk, model, state = make(chunk=64)
    model.prefill([1], state, prime_mtp=True, next_token=2)
    assert trunk.mtp.folded_ids == [2]


def test_a_single_token_prefill_without_one_folds_nothing():
    trunk, model, state = make(chunk=64)
    model.prefill([1], state, prime_mtp=True)
    assert trunk.mtp.folds == []


def test_priming_a_model_without_a_head_is_a_no_op():
    trunk, model, state = make(with_head=False)
    model.prefill([1, 2, 3], state, prime_mtp=True, next_token=4)
    assert not any(call["return_hidden"] for call in trunk.calls)


def test_priming_does_not_leave_the_chunk_hidden_on_the_state():
    """A 2048-wide hidden block is 40 MB and the drafter never reads it: what
    it drafts from is the verify's hidden, which is a few columns wide."""
    _trunk, model, state = make(chunk=4)
    model.prefill(list(range(1, 13)), state, prime_mtp=True, next_token=13)
    assert state.mtp_hidden is None


def test_want_hidden_still_returns_the_last_chunks_hidden_when_primed():
    """The two are independent: one is a fold, the other is a return value."""
    _trunk, model, state = make(chunk=4)
    result = model.prefill(
        list(range(1, 9)), state, prime_mtp=True, next_token=9, want_hidden=True
    )
    assert result.hidden is not None
    assert state.mtp_hidden is not None


# -- the window -------------------------------------------------------------
#
# Priming the whole of a 64k prompt gives the head a 64k KV cache and it
# re-attends over all of it once per drafted token. The window is the answer to
# that, and what it has to get right is the same indexing question as above
# asked from the other end of the sequence: the pairs it keeps are the last
# ``window`` of them, counted from the end of the *sequence*, not the end of
# whatever chunk the caller happens to be running.


def test_a_window_wider_than_the_prompt_primes_all_of_it():
    trunk, model, state = make(chunk=64)
    model.prefill(
        list(range(1, 10)), state, prime_mtp=True, prime_window=1000, next_token=10
    )
    assert trunk.mtp.folded_ids == [2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_a_window_keeps_exactly_the_last_pairs_across_a_chunk_seam():
    """Twelve pairs, a window of five: ids 9 through 13, and the seam is inside
    the window rather than on it, which is the case a chunk-local window gets
    wrong."""
    trunk, model, state = make(chunk=4)
    model.prefill(
        list(range(1, 13)), state, prime_mtp=True, prime_window=5, next_token=13
    )
    assert trunk.mtp.folded_ids == [9, 10, 11, 12, 13]
    assert state.mtp_layers[0].offset == 5


def test_a_chunk_entirely_outside_the_window_asks_the_trunk_for_no_hidden():
    """The saving is not only the fold. A chunk nothing will be folded from
    does not need its hidden states, and asking for them is what puts the
    vendored capture path on a 2048-wide chunk."""
    trunk, model, state = make(chunk=4)
    model.prefill(
        list(range(1, 13)), state, prime_mtp=True, prime_window=5, next_token=13
    )
    assert [call["return_hidden"] for call in trunk.calls] == [False, True, True]


def test_the_window_counts_from_the_end_of_the_sequence_not_of_the_chunk():
    """The caller chunked the prompt, so only the caller knows the sequence is
    longer than what it just handed over. With ten tokens still to come, a
    window of five reaches none of this chunk."""
    trunk, model, state = make(chunk=4)
    model.prefill(
        list(range(1, 13)),
        state,
        prime_mtp=True,
        prime_window=5,
        prime_after=10,
        next_token=13,
    )
    assert trunk.mtp.folds == []
    assert not any(call["return_hidden"] for call in trunk.calls)


def test_a_window_that_starts_mid_chunk_takes_that_chunks_tail_only():
    trunk, model, state = make(chunk=8)
    model.prefill(
        list(range(1, 9)), state, prime_mtp=True, prime_window=3, next_token=9
    )
    assert trunk.mtp.folded_ids == [7, 8, 9]
