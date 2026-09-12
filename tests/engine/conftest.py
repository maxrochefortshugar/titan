"""Fakes for the engine: a backend, a tokenizer, a cache, a drafter, a clock.

Every port the engine touches has a fake here, and none of them import mlx or
load anything. That is the point of the dependency rule: the scheduler, the
admission path and both decode cycles run at full speed with no model and no
GPU, so the invariants that matter -- lossless speculation, stop handling,
replay-free rollback, the guard -- are checked exhaustively rather than sampled.

The fake backend is a real model in the only sense the engine cares about: it
has a deterministic next-token function, it keeps per-sequence state, it
refuses a truncation below its last snapshot, and its ``verify`` implements the
row layout the port describes -- ``(pending, draft_1, ..., draft_k)`` in, the
accepted prefix plus one bonus token out.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import pytest

from titan.core.errors import StateError
from titan.core.types import (
    BlockHash,
    CycleProfile,
    DraftCandidate,
    PrefixMatch,
    Request,
    RequestId,
    SamplingParams,
    SequenceId,
    StateHandle,
    StopCondition,
    VerifyOutcome,
)
from titan.engine.admission import AdmissionConfig, plan_chunks


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonic and deterministic: every read advances by a fixed tick."""

    def __init__(self, tick: float = 0.001) -> None:
        self.tick = tick
        self.t = 0.0

    def now(self) -> float:
        self.t += self.tick
        return self.t


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------


def default_next_token(context: Sequence[int], vocab: int) -> int:
    """A deterministic stand-in for the model's argmax.

    It depends on the last four tokens and on the position, so a token stream
    is sensitive to exactly the things a real one is: what came before it, and
    where it is. Any bug that commits a token twice, drops one, or replays a
    rejected draft changes the stream immediately.
    """
    tail = list(context[-4:])
    value = 5 + 7 * len(context)
    for index, token in enumerate(tail):
        value += (index + 1) * 31 * (token + 1)
    return value % vocab


@dataclass
class _FakeState:
    tokens: list[int] = field(default_factory=list)
    snapshots: set[int] = field(default_factory=set)
    closed: bool = False


class FakeBackend:
    """A deterministic ``ModelBackend``. No arrays, no device, no randomness."""

    def __init__(
        self,
        *,
        vocab: int = 64,
        next_token: Callable[[Sequence[int], int], int] | None = None,
        draft_depth_max: int = 8,
        max_context: int = 4096,
    ) -> None:
        self.vocab_size = vocab
        self.n_layers = 4
        self.max_context = max_context
        self.draft_depth_max = draft_depth_max
        self._next = next_token or default_next_token
        self._states: dict[int, _FakeState] = {}
        self._handles = itertools.count(1)
        self._cycle = itertools.count(1)
        self.open_handles: set[int] = set()
        self.verify_calls: list[int] = []
        self.prefill_calls: list[tuple[int, int, bool]] = []
        self.truncations: list[tuple[int, int]] = []

    # -- the oracle the tests compare against ------------------------------
    def argmax_after(self, context: Sequence[int]) -> int:
        return self._next(context, self.vocab_size)

    def greedy_continuation(self, prompt: Sequence[int], count: int) -> list[int]:
        """What plain greedy decoding produces. The reference token stream."""
        tokens = list(prompt)
        out: list[int] = []
        for _ in range(count):
            token = self.argmax_after(tokens)
            tokens.append(token)
            out.append(token)
        return out

    # -- state lifecycle ---------------------------------------------------
    def open_state(self, seq: SequenceId, capacity_hint: int) -> StateHandle:
        handle = next(self._handles)
        self._states[handle] = _FakeState()
        self.open_handles.add(handle)
        return StateHandle(handle)

    def close_state(self, state: StateHandle) -> None:
        entry = self._states.get(int(state))
        if entry is not None:
            entry.closed = True
        self.open_handles.discard(int(state))

    def state_length(self, state: StateHandle) -> int:
        return len(self._state(state).tokens)

    def state_tokens(self, state: StateHandle) -> list[int]:
        return list(self._state(state).tokens)

    def truncate_state(self, state: StateHandle, length: int) -> None:
        entry = self._state(state)
        if length > len(entry.tokens):
            raise StateError("cannot grow a state by truncating it")
        if length < len(entry.tokens) and length not in entry.snapshots:
            raise StateError(
                f"no recurrent snapshot at {length}; staged: {sorted(entry.snapshots)}"
            )
        self.truncations.append((int(state), length))
        del entry.tokens[length:]
        entry.snapshots = {s for s in entry.snapshots if s <= length}

    def _state(self, state: StateHandle) -> _FakeState:
        entry = self._states.get(int(state))
        if entry is None or entry.closed:
            raise StateError(f"no such state handle: {int(state)}")
        return entry

    # -- forward -----------------------------------------------------------
    def prefill(
        self,
        state: StateHandle,
        tokens: Sequence[int],
        *,
        want_logits: bool = False,
        snapshot: bool = False,
    ) -> None:
        entry = self._state(state)
        entry.tokens.extend(tokens)
        self.prefill_calls.append((int(state), len(tokens), snapshot))
        if snapshot:
            entry.snapshots.add(len(entry.tokens))
        return None

    def decode(self, states: Sequence[StateHandle], tokens: Sequence[int]) -> None:
        raise AssertionError("the engine goes through verify, never decode")

    def verify(
        self,
        states: Sequence[StateHandle],
        drafts: Sequence[DraftCandidate],
        sampling: Sequence[SamplingParams],
    ) -> tuple[list[VerifyOutcome], CycleProfile]:
        """One padded row block, greedy acceptance, replay-free rollback.

        The row is ``(pending, draft_1, ..., draft_k)``. Every row is consumed,
        the accepted prefix is kept, and the state is left covering exactly
        ``1 + n_accepted`` more tokens than it did on entry, which is the port's
        replay-free rollback invariant stated as code.
        """
        width = max(len(d.tokens) for d in drafts)
        self.verify_calls.append(width)
        outcomes: list[VerifyOutcome] = []
        for state, draft in zip(states, drafts):
            entry = self._state(state)
            entry.snapshots.add(len(entry.tokens))  # staged before the forward
            row = list(draft.tokens)
            pending, drafted = row[0], row[1:]
            context = list(entry.tokens) + [pending]
            accepted: list[int] = []
            for proposal in drafted:
                target = self.argmax_after(context)
                if target != proposal:
                    break
                accepted.append(proposal)
                context.append(proposal)
            bonus = self.argmax_after(context)
            entry.tokens = context
            outcomes.append(
                VerifyOutcome(
                    sequence_id=draft.sequence_id,
                    accepted=tuple(accepted),
                    bonus=bonus,
                    n_drafted=len(drafted),
                )
            )
        profile = CycleProfile(
            cycle=next(self._cycle),
            n_sequences=len(drafts),
            n_rows=len(drafts) * width,
            draft_ms=0.0,
            verify_ms=1.0,
            accept_ms=0.1,
            sample_ms=0.0,
            detok_ms=0.0,
            ngram_wait_ms=0.0,
            host_syncs=1,
            tokens_committed=sum(o.n_committed for o in outcomes),
            tokens_drafted=sum(o.n_drafted for o in outcomes),
            wall_ms=1.5,
        )
        return outcomes, profile

    # -- snapshots ---------------------------------------------------------
    def export_snapshot(self, state: StateHandle, length: int) -> bytes:
        entry = self._state(state)
        if length not in entry.snapshots:
            raise StateError(f"nothing staged at {length}")
        return b"snapshot:" + str(length).encode()

    def import_snapshot(self, state: StateHandle, length: int, blob: bytes) -> None:
        entry = self._state(state)
        entry.snapshots.add(length)

    def warmup(self) -> None:
        return None


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Ids to text through a table, with a per-sequence streaming stream.

    ``pieces`` maps a token id to the text it contributes. Anything absent
    renders as ``<id>``, which makes an unexpected token loud in an assertion
    instead of invisible.
    """

    def __init__(
        self,
        pieces: Mapping[int, str] | None = None,
        eos_token_ids: Iterable[int] = (0,),
        vocab_size: int = 64,
    ) -> None:
        self.pieces = dict(pieces or {})
        self.eos_token_ids = frozenset(eos_token_ids)
        self.vocab_size = vocab_size
        self.streams: dict[int, list[int]] = {}
        self.flushed: list[int] = []

    def piece(self, token_id: int) -> str:
        return self.pieces.get(token_id, f"<{token_id}>")

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.piece(i) for i in ids)

    def decode_incremental(self, seq: SequenceId, ids: Sequence[int]) -> str:
        self.streams.setdefault(int(seq), []).extend(ids)
        return "".join(self.piece(i) for i in ids)

    def flush_incremental(self, seq: SequenceId) -> str:
        self.streams.pop(int(seq), None)
        self.flushed.append(int(seq))
        return ""


# ---------------------------------------------------------------------------
# prefix cache
# ---------------------------------------------------------------------------


class FakeCache:
    """A prefix cache that hits on a length the test sets, and records stores."""

    def __init__(self, *, matched: int = 0, config: AdmissionConfig | None = None) -> None:
        self.matched = matched
        self.config = config or AdmissionConfig()
        self.stores: list[tuple[int, tuple[int, ...]]] = []
        self.restored: list[int] = []
        self.plans: list[tuple[int, int]] = []

    def lookup(self, tokens: Sequence[int]) -> PrefixMatch:
        matched = min(self.matched, len(tokens))
        return PrefixMatch(
            matched_tokens=matched,
            block_hashes=(),
            snapshot_id="s" if matched else None,
            tier="ram" if matched else "none",
        )

    def restore(self, match: PrefixMatch, state: StateHandle) -> int:
        self.restored.append(match.matched_tokens)
        return match.matched_tokens

    def store(
        self, tokens: Sequence[int], state: StateHandle, boundaries: Sequence[int]
    ) -> None:
        self.stores.append((len(tokens), tuple(boundaries)))

    def plan_chunks(self, matched: int, total: int, contended: bool) -> tuple[int, ...]:
        self.plans.append((matched, total))
        return plan_chunks(
            matched,
            total,
            chunk=self.config.prefill_chunk_tokens,
            block=self.config.block_tokens,
            grid=self.config.snapshot_grid,
        )

    def stats(self) -> Mapping[str, float]:
        return {"stores": float(len(self.stores))}


# ---------------------------------------------------------------------------
# drafter
# ---------------------------------------------------------------------------


class ScriptedDrafter:
    """Proposes whatever the test tells it to.

    ``policy(context, depth, backend)`` returns the chain for one sequence, so a
    test can ask for a perfect draft, a draft that is wrong at position two, or
    a draft drawn at random -- which is how the lossless-speculation invariant
    is checked over the whole acceptance space rather than at its ends.
    """

    name = "scripted"

    def __init__(self, policy: Callable[[Sequence[int], int], Sequence[int]]) -> None:
        self.policy = policy
        self.observed: list[VerifyOutcome] = []
        self.depths: list[list[int]] = []

    def propose(
        self,
        states: Sequence[StateHandle],
        contexts: Sequence[Sequence[int]],
        depth: Sequence[int],
    ) -> list[DraftCandidate]:
        self.depths.append(list(depth))
        return [
            DraftCandidate(
                sequence_id=SequenceId(index + 1),
                tokens=tuple(self.policy(context, d)),
                source="mtp",
            )
            for index, (context, d) in enumerate(zip(contexts, depth))
        ]

    def observe(self, outcomes: Sequence[VerifyOutcome]) -> None:
        self.observed.extend(outcomes)


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


def make_request(
    prompt: Sequence[int] = (11, 12, 13),
    *,
    request_id: str = "req-1",
    max_tokens: int = 16,
    stop_strings: Sequence[str] = (),
    eos: Iterable[int] = (),
    max_total_tokens: int | None = None,
) -> Request:
    return Request(
        request_id=RequestId(request_id),
        prompt_tokens=tuple(prompt),
        sampling=SamplingParams(temperature=0.0),
        stop=StopCondition(
            eos_token_ids=frozenset(eos),
            stop_strings=tuple(stop_strings),
            max_tokens=max_tokens,
            max_total_tokens=max_total_tokens,
        ),
        stream=True,
    )


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()
