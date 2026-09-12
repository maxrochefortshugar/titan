"""``ModelBackend`` over the vendored Qwen3.8-Flash-Next code.

The port (``titan/core/ports.py``) speaks in opaque ``StateHandle`` integers;
this module keeps the table that maps them to :class:`ModelState` objects and
forwards to :mod:`titan.adapters.mlx.model`.  The split is deliberate: the model
module is usable on its own (the parity harness drives it directly), and the
backend adds only handle bookkeeping, batching and the acceptance reduction.

The one piece of real work here is :meth:`verify`.  Acceptance is computed in
the graph -- compare drafted ids against the target argmax, cumulative product
along the draft axis, sum -- so a cycle crosses to the host once, carrying an
integer per sequence rather than a logits block.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import mlx.core as mx

from titan.core.types import (
    CycleProfile,
    DraftCandidate,
    SamplingParams,
    SequenceId,
    StateHandle,
    VerifyOutcome,
)

from .model import TitanQwenFlashNext
from .state import PHASE_VERIFY, ModelState, StateError


@dataclass
class _Logits:
    """A logits block still on the GPU. The core never reads it."""

    array: mx.array

    @property
    def rows(self) -> int:
        return self.array.shape[0]

    @property
    def vocab(self) -> int:
        return self.array.shape[-1]


class MLXModelBackend:
    """One process, one model, one instance."""

    def __init__(self, model: TitanQwenFlashNext):
        self.model = model
        self._states: dict[int, ModelState] = {}
        self._sequences: dict[int, SequenceId] = {}
        self._handles = itertools.count(1)
        self._cycle = itertools.count(1)

    # -- properties --------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return self.model.n_layers

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def max_context(self) -> int:
        return self.model.max_context

    @property
    def draft_depth_max(self) -> int:
        return self.model.draft_depth_max

    # -- state lifecycle ---------------------------------------------------
    def open_state(self, seq: SequenceId, capacity_hint: int) -> StateHandle:
        handle = next(self._handles)
        self._states[handle] = self.model.new_state()
        self._sequences[handle] = seq
        return StateHandle(handle)

    def close_state(self, state: StateHandle) -> None:
        self._states.pop(int(state), None)
        self._sequences.pop(int(state), None)

    def state_length(self, state: StateHandle) -> int:
        return self._state(state).length

    def truncate_state(self, state: StateHandle, length: int) -> None:
        self._state(state).truncate(length)

    def _state(self, handle: StateHandle) -> ModelState:
        try:
            return self._states[int(handle)]
        except KeyError:
            raise StateError(f"no such state handle: {int(handle)}") from None

    # -- forward -----------------------------------------------------------
    def prefill(
        self,
        state: StateHandle,
        tokens: Sequence[int],
        *,
        want_logits: bool = False,
        snapshot: bool = False,
    ) -> _Logits | None:
        model_state = self._state(state)
        result = self.model.prefill(
            tokens,
            model_state,
            want_logits=want_logits,
            want_hidden=self.draft_depth_max > 0,
        )
        if snapshot:
            model_state.stage_snapshot()
        return None if result.logits is None else _Logits(result.logits)

    def decode(
        self,
        states: Sequence[StateHandle],
        tokens: Sequence[int],
    ) -> _Logits:
        model_states = [self._state(s) for s in states]
        if len(model_states) == 1:
            logits = self.model.decode(
                [tokens[0]], model_states[0], want_hidden=self.draft_depth_max > 0
            )
            return _Logits(logits)
        batch = ModelState.batch(model_states)
        logits = self.model.decode(
            list(tokens), batch, want_hidden=self.draft_depth_max > 0
        )
        self._scatter(batch, model_states)
        return _Logits(logits)

    def verify(
        self,
        states: Sequence[StateHandle],
        drafts: Sequence[DraftCandidate],
        sampling: Sequence[SamplingParams],
    ) -> tuple[list[VerifyOutcome], CycleProfile]:
        """One padded row block for the whole batch, one host sync."""
        start = time.perf_counter()
        model_states = [self._state(s) for s in states]
        width = max(len(d.tokens) for d in drafts) + 1
        rows = [_pad_draft(d, width) for d in drafts]

        target = (
            model_states[0]
            if len(model_states) == 1
            else ModelState.batch(model_states, phase=PHASE_VERIFY)
        )
        verify_start = time.perf_counter()
        logits = self.model.verify(rows, target, snapshot=True)
        verify_ms = (time.perf_counter() - verify_start) * 1000.0

        accept_start = time.perf_counter()
        drafted = mx.array([list(r[1:]) for r in rows], dtype=mx.int32)
        argmax = mx.argmax(logits, axis=-1).astype(mx.int32)
        agree = (argmax[:, :-1] == drafted).astype(mx.int32)
        accepted_counts = mx.cumprod(agree, axis=1).sum(axis=1)
        bonus_index = mx.minimum(accepted_counts, width - 1)
        bonus = mx.take_along_axis(argmax, bonus_index[:, None], axis=1)[:, 0]
        # The single host sync of the cycle: two small integer vectors cross,
        # never the logits.
        mx.eval(accepted_counts, bonus)
        counts = accepted_counts.tolist()
        bonuses = bonus.tolist()
        accept_ms = (time.perf_counter() - accept_start) * 1000.0

        outcomes: list[VerifyOutcome] = []
        for draft, model_state, n, extra in zip(
            drafts, model_states, counts, bonuses
        ):
            n = min(int(n), len(draft.tokens))
            committed = n + 1
            entry_length = model_state.length - width
            model_state.truncate(entry_length)
            model_state.length = entry_length + committed
            outcomes.append(
                VerifyOutcome(
                    sequence_id=draft.sequence_id,
                    accepted=tuple(draft.tokens[:n]),
                    bonus=int(extra),
                    n_drafted=len(draft.tokens),
                )
            )
        if target is not model_states[0]:
            self._scatter(target, model_states)

        wall_ms = (time.perf_counter() - start) * 1000.0
        profile = CycleProfile(
            cycle=next(self._cycle),
            n_sequences=len(model_states),
            n_rows=len(rows) * width,
            draft_ms=0.0,
            verify_ms=verify_ms,
            accept_ms=accept_ms,
            sample_ms=0.0,
            detok_ms=0.0,
            ngram_wait_ms=0.0,
            host_syncs=1,
            tokens_committed=sum(o.n_committed for o in outcomes),
            tokens_drafted=sum(o.n_drafted for o in outcomes),
            wall_ms=wall_ms,
        )
        return outcomes, profile

    # -- snapshots ---------------------------------------------------------
    def export_snapshot(self, state: StateHandle, length: int) -> bytes:
        return self._state(state).export_snapshot(length)

    def import_snapshot(self, state: StateHandle, length: int, blob: bytes) -> None:
        self._state(state).import_snapshot(length, blob)

    def warmup(self) -> None:
        self.model.warmup()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _scatter(batch: ModelState, singles: Sequence[ModelState]) -> None:
        """Return a batched state's rows to their per-sequence states.

        The batched caches are the live arrays; each single keeps a filtered
        view, which is why rows are handed back rather than copied.
        """
        for row, single in enumerate(singles):
            single.length = batch.length
            if batch.mtp_hidden is not None:
                single.mtp_hidden = batch.mtp_hidden[row : row + 1]


def _pad_draft(draft: DraftCandidate, width: int) -> list[int]:
    """One verify row: the sequence's own next token plus the drafted chain.

    Short drafts are padded by repeating the last token.  Padding cannot change
    a row's own output -- the extra columns sit past the first rejection and are
    discarded with the rest of the rejected suffix.
    """
    row = list(draft.tokens)
    if len(row) > width:
        raise StateError("draft longer than the block width")
    while len(row) < width:
        row.append(row[-1] if row else 0)
    return row
