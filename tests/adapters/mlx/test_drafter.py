# SPDX-License-Identifier: MIT
"""The MTP chain, against a fake head.

The head is faked and the drafter is real, including the real
``TitanQwenFlashNext.mtp_step`` and ``mtp_lift``, because the things that can
go wrong here are all in the plumbing rather than in the arithmetic: which
hidden state step two is fed, which cache it appends to, how many times the
host is asked for an answer, and whether the persistent head KV still mirrors
the committed sequence after a chain nobody accepted.

The fake head is deterministic and, deliberately, *distinguishable*: its
pre-mixer streams are not a multiple of the lift of its mixer output, so a
chain that feeds the wrong one produces different tokens and a test can say
which form ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from titan.adapters.mlx import drafter as drafter_module
from titan.adapters.mlx.drafter import MTPDrafter
from titan.adapters.mlx.model import TitanQwenFlashNext

HC = 2
HIDDEN = 4
VOCAB = 8


class FakeCache:
    """One head KV layer: a length and the ids it consumed, nothing else."""

    def __init__(self) -> None:
        self.offset = 0
        self.consumed: list[list[int]] = []

    def append(self, ids: mx.array) -> None:
        self.offset += int(ids.shape[1])
        self.consumed.append([int(v) for v in ids.reshape(-1).tolist()])

    def extract(self, idx: int) -> "FakeCache":
        clone = FakeCache()
        clone.offset = self.offset
        clone.consumed = [list(row) for row in self.consumed]
        return clone


class FakeEmbed:
    def __call__(self, ids: mx.array) -> mx.array:
        base = ids.astype(mx.float32).reshape(*ids.shape, 1)
        return mx.broadcast_to(base, (*ids.shape, HIDDEN)) * 0.25


class FakeLMHead:
    """Logits with a controllable peak, so a test can drive the p_min gate.

    ``sharpness`` is popped per call when the test has queued values: a large
    value puts almost all the mass on the argmax, and zero makes the
    distribution uniform, which is the shape a probability floor exists to
    catch.
    """

    def __init__(self) -> None:
        self.sharpness: list[float] = []
        self.default_sharpness = 12.0

    def __call__(self, hidden: mx.array) -> mx.array:
        peak = int(abs(float(mx.sum(hidden).item()))) % VOCAB
        scale = self.sharpness.pop(0) if self.sharpness else self.default_sharpness
        onehot = mx.zeros((VOCAB,), dtype=mx.float32)
        onehot[peak] = 1.0
        return mx.broadcast_to(
            (onehot * scale).reshape(1, 1, VOCAB),
            (hidden.shape[0], hidden.shape[1], VOCAB),
        )


class FakeMTPModule:
    """A one-layer head that records what it was fed and by whom."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def __call__(self, hidden, ids, embed, cache):
        ids = ids if isinstance(ids, mx.array) else mx.array(ids)
        self.calls.append(
            SimpleNamespace(hidden=hidden, ids=ids, cache=cache)
        )
        for layer in cache:
            layer.append(ids)
        collapsed = hidden.reshape(*hidden.shape[:-1], HC, HIDDEN).sum(axis=-2)
        mixed = collapsed + embed(ids)
        # Not a multiple of ``mtp_lift(mixed)``: the two chain forms have to be
        # tellable apart from the outside.
        streams = mx.concatenate([mixed * 3.0, mixed * 1.5], axis=-1)
        return mixed, streams


class FakeLanguageModel:
    def __init__(self) -> None:
        self.mtp = FakeMTPModule()
        self.lm_head = FakeLMHead()
        self.model = SimpleNamespace(embed_tokens=FakeEmbed())
        self.made_caches = 0

    def get_mtp_module(self):
        return self.mtp

    def make_mtp_cache(self):
        self.made_caches += 1
        return [FakeCache()]


class FakeState:
    def __init__(self) -> None:
        self.mtp_layers: list[FakeCache] = [FakeCache()]
        self.mtp_hidden: mx.array | None = None
        self.length = 0


class FakeBackend:
    def __init__(self, model: TitanQwenFlashNext) -> None:
        self.model = model
        self.states: dict[int, FakeState] = {}

    def draft_state(self, handle):
        return self.states.setdefault(int(handle), FakeState())


@pytest.fixture()
def rig():
    language = FakeLanguageModel()
    args = SimpleNamespace(
        hc_count=HC,
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        max_position_embeddings=1024,
    )
    model = TitanQwenFlashNext(
        SimpleNamespace(
            language_model=language, config=SimpleNamespace(text_config=args)
        )
    )
    backend = FakeBackend(model)
    return SimpleNamespace(language=language, model=model, backend=backend)


def hidden_block(width: int) -> mx.array:
    """A stand-in for what a verify of ``width`` columns leaves on the state."""
    values = mx.arange(width * HC * HIDDEN, dtype=mx.float32) * 0.1 + 1.0
    return values.reshape(1, width, HC * HIDDEN)


def seed(drafter: MTPDrafter, backend: FakeBackend, context: list[int]) -> None:
    """Run the cycle a sequence spends without a hidden state to draft from.

    Prefill never asks the backbone for a hidden state, so the first proposal
    of a sequence has nothing to fold and says so. Every test starts past that
    point, which is also where the real server spends exactly one cycle.
    """
    drafter.propose([1], [context], [3])
    backend.draft_state(1).mtp_hidden = hidden_block(1)


# ---------------------------------------------------------------------------


def test_first_cycle_drafts_nothing_and_touches_no_cache(rig):
    drafter = MTPDrafter(rig.backend)
    out = drafter.propose([1], [[5, 6, 7]], [3])
    assert out[0].tokens == ()
    assert rig.backend.draft_state(1).mtp_layers[0].offset == 0
    assert rig.language.mtp.calls == []


@pytest.mark.parametrize("p_min", [0.0, 0.01])
def test_draft_count_equals_requested_depth(rig, p_min):
    drafter = MTPDrafter(rig.backend, p_min=p_min)
    seed(drafter, rig.backend, [5, 6, 7])
    for depth in range(1, 9):
        rig.backend.draft_state(1).mtp_hidden = hidden_block(1)
        out = drafter.propose([1], [[5, 6, 7] + [9] * depth], [depth])
        assert len(out[0].tokens) == depth
        # Logprobs only exist where something reads them: a floor of zero does
        # not pay for a softmax over the vocabulary on every chain step.
        expected = depth if p_min else 0
        assert len(out[0].draft_logprobs) == expected


def test_fold_covers_every_token_the_cycle_committed(rig):
    """The head consumes the accepted run, not just the pending token."""
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    state = rig.backend.draft_state(1)
    state.mtp_hidden = hidden_block(4)
    drafter.propose([1], [[5, 6, 7, 11, 12, 13]], [1])
    fold = rig.language.mtp.calls[0]
    assert fold.ids.tolist() == [[11, 12, 13]]
    assert fold.hidden.shape == (1, 3, HC * HIDDEN)
    # One head KV entry per committed token, and nothing more.
    assert state.mtp_layers[0].offset == 3


def test_chain_re_enters_on_the_head_output_not_the_backbone(rig):
    drafter = MTPDrafter(rig.backend, chain="head_output")
    seed(drafter, rig.backend, [5, 6, 7])
    drafter.propose([1], [[5, 6, 7, 11]], [3])
    calls = rig.language.mtp.calls
    assert len(calls) == 3
    backbone = rig.backend.draft_state(1).mtp_hidden
    for step in (1, 2):
        assert not mx.array_equal(calls[step].hidden, backbone)
        assert calls[step].hidden.shape == (1, 1, HC * HIDDEN)
    # Step two is fed the lift of step one's mixer output, which is this
    # architecture's post-final-norm hidden state.
    expected = rig.model.mtp_lift(
        _mixed_of(calls[0], rig)[:, -1:, :]
    )
    assert mx.allclose(calls[1].hidden, expected)


def test_omlx_chain_re_enters_on_the_pre_mixer_streams(rig):
    drafter = MTPDrafter(rig.backend, chain="omlx")
    seed(drafter, rig.backend, [5, 6, 7])
    drafter.propose([1], [[5, 6, 7, 11]], [2])
    calls = rig.language.mtp.calls
    streams = _streams_of(calls[0], rig)
    assert mx.allclose(calls[1].hidden, streams[:, -1:, :])


def test_the_two_chain_forms_disagree_after_the_first_draft(rig):
    """If they agreed the switch would be measuring nothing."""
    first = MTPDrafter(rig.backend, chain="head_output")
    seed(first, rig.backend, [5, 6, 7])
    head_out = first.propose([1], [[5, 6, 7, 11]], [4])[0].tokens

    rig.language.mtp.calls.clear()
    rig.backend.states.clear()
    second = MTPDrafter(rig.backend, chain="omlx")
    seed(second, rig.backend, [5, 6, 7])
    omlx_out = second.propose([1], [[5, 6, 7, 11]], [4])[0].tokens

    assert head_out[0] == omlx_out[0]
    assert head_out != omlx_out


def test_one_host_sync_per_proposal(rig, monkeypatch):
    syncs = {"n": 0}
    original = drafter_module._sync_and_read

    def counted(values):
        syncs["n"] += 1
        return original(values)

    monkeypatch.setattr(drafter_module, "_sync_and_read", counted)
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    syncs["n"] = 0
    drafter.propose([1], [[5, 6, 7, 11]], [6])
    assert syncs["n"] == 1
    assert drafter.host_syncs == 1


def test_one_host_sync_for_the_whole_batch(rig, monkeypatch):
    syncs = {"n": 0}
    original = drafter_module._sync_and_read
    monkeypatch.setattr(
        drafter_module,
        "_sync_and_read",
        lambda values: (syncs.__setitem__("n", syncs["n"] + 1), original(values))[1],
    )
    drafter = MTPDrafter(rig.backend)
    for handle in (1, 2, 3):
        drafter.propose([handle], [[5, 6, 7]], [3])
        rig.backend.draft_state(handle).mtp_hidden = hidden_block(1)
    syncs["n"] = 0
    out = drafter.propose([1, 2, 3], [[5, 6, 7, 11]] * 3, [3, 3, 3])
    assert syncs["n"] == 1
    assert all(len(candidate.tokens) == 3 for candidate in out)


def test_head_cache_untouched_by_a_rejected_chain(rig):
    """The chain runs on a clone, so nothing it drafted is history."""
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    state = rig.backend.draft_state(1)
    persistent = state.mtp_layers[0]

    drafter.propose([1], [[5, 6, 7, 11]], [5])
    assert persistent.offset == 1
    assert persistent.consumed == [[11]]

    # Nothing was accepted: the next cycle commits one bonus token and folds
    # only that. The head cache grows by one, not by the five it drafted.
    state.mtp_hidden = hidden_block(6)
    drafter.propose([1], [[5, 6, 7, 11, 12]], [5])
    assert persistent.offset == 2
    assert persistent.consumed == [[11], [12]]
    assert rig.language.made_caches == 0


def test_head_cache_grows_by_the_accepted_run(rig):
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    state = rig.backend.draft_state(1)
    state.mtp_hidden = hidden_block(4)
    drafter.propose([1], [[5, 6, 7, 11]], [3])
    state.mtp_hidden = hidden_block(4)
    drafter.propose([1], [[5, 6, 7, 11, 12, 13, 14]], [3])
    assert state.mtp_layers[0].offset == 4
    assert state.mtp_layers[0].consumed == [[11], [12, 13, 14]]


def test_p_min_truncates_the_chain(rig):
    drafter = MTPDrafter(rig.backend, p_min=0.5)
    seed(drafter, rig.backend, [5, 6, 7])
    # Steps one and two are confident, step three is uniform over eight ids,
    # which is a top probability of 0.125.
    rig.language.lm_head.sharpness = [12.0, 12.0, 0.0, 12.0]
    out = drafter.propose([1], [[5, 6, 7, 11]], [4])
    assert len(out[0].tokens) == 2
    assert drafter.gated_steps == 1


def test_p_min_of_zero_keeps_everything(rig):
    drafter = MTPDrafter(rig.backend, p_min=0.0)
    seed(drafter, rig.backend, [5, 6, 7])
    rig.language.lm_head.sharpness = [12.0, 0.0, 0.0, 0.0]
    out = drafter.propose([1], [[5, 6, 7, 11]], [4])
    assert len(out[0].tokens) == 4
    assert drafter.gated_steps == 0


def test_zero_depth_drafts_nothing(rig):
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    out = drafter.propose([1], [[5, 6, 7, 11]], [0])
    assert out[0].tokens == ()
    assert out[0].source == "none"
    assert rig.language.mtp.calls == []


def test_a_misaligned_head_cache_is_dropped_rather_than_fed_a_gap(rig):
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    drafter.propose([1], [[5, 6, 7, 11]], [2])
    state = rig.backend.draft_state(1)
    # What ``ModelState.truncate`` does on the stop path: it trims the head by
    # the trunk's delta, which the head never advanced by.
    state.mtp_layers[0].offset = 0
    state.mtp_hidden = hidden_block(2)
    out = drafter.propose([1], [[5, 6, 7, 11, 12]], [2])
    assert out[0].tokens == ()
    assert rig.language.made_caches == 1
    # And the cycle after the reset drafts again.
    state.mtp_hidden = hidden_block(2)
    out = drafter.propose([1], [[5, 6, 7, 11, 12, 13]], [2])
    assert len(out[0].tokens) == 2


def test_a_committed_run_longer_than_the_hidden_block_resets(rig):
    drafter = MTPDrafter(rig.backend)
    seed(drafter, rig.backend, [5, 6, 7])
    state = rig.backend.draft_state(1)
    state.mtp_hidden = hidden_block(2)
    out = drafter.propose([1], [[5, 6, 7] + [9] * 9], [2])
    assert out[0].tokens == ()
    assert drafter.cache_resets == 1


def test_unknown_chain_form_is_refused(rig):
    with pytest.raises(ValueError, match="unknown MTP chain form"):
        MTPDrafter(rig.backend, chain="eagle")


# -- helpers ---------------------------------------------------------------


def _replay(call, rig):
    hidden = call.hidden
    collapsed = hidden.reshape(*hidden.shape[:-1], HC, HIDDEN).sum(axis=-2)
    return collapsed + FakeEmbed()(call.ids)


def _mixed_of(call, rig):
    return _replay(call, rig)


def _streams_of(call, rig):
    mixed = _replay(call, rig)
    return mx.concatenate([mixed * 3.0, mixed * 1.5], axis=-1)
