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

    @property
    def layer_layout(self) -> tuple[str, ...]:
        """Per-layer cache kind, for the cache signature.

        The store comes after the backend in the composition order precisely so
        that it can name this: a payload written under one layer layout and read
        back under another is silent nonsense, so the layout goes into every
        block digest and every file header.
        """
        return self.model.layer_layout

    @property
    def state_bytes_per_token(self) -> float:
        """What one more token of context costs the guard. From the shapes."""
        return self.model.state_bytes_per_token

    def state_codec(self, signature: Any) -> Any:
        """The :class:`~titan.adapters.cache.codec.StateCodec` for this model.

        Built here rather than by the wiring because only the adapter knows how
        a state handle turns into bytes, and only the backend knows which
        ``ModelState`` a handle resolves to.
        """
        from .codec import MLXStateCodec  # noqa: PLC0415 - keeps import cost local

        # The codec is handed the handle table, not a state. The cache calls it
        # with whatever the engine gave the cache, and the engine deals in
        # opaque handles, so resolving one is the backend's job and nobody
        # else's.
        return MLXStateCodec(signature, self._state)

    def resident_gb(self) -> float:
        """What the loop's guard reads. Active device memory plus the weights.

        Measured rather than modelled, and measured here because
        ``mx.get_active_memory`` is only meaningful on the thread that owns the
        MLX stream, which is the scheduler thread that calls this.
        """
        return mx.get_active_memory() / 1e9

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
        # No hidden state from a prefill chunk, ever. Asking for it turns on the
        # vendored ``target_verify`` path for the whole chunk, and that path
        # runs one kernel launch per token: it exists for a verify block a few
        # columns wide and it is a disaster over 2048. A 150-token final chunk
        # measured 200 seconds this way against 0.1 for the same chunk without.
        # The drafter's seed hidden state comes from the verify forward, which
        # is small and is where that path belongs.
        result = self.model.prefill(
            tokens,
            model_state,
            want_logits=want_logits,
            want_hidden=False,
        )
        if snapshot:
            # A plan boundary: the cache will be asked to serialise it when the
            # sequence retires, so it outlives every rollback copy.
            model_state.stage_snapshot(pinned=True)
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
        # A candidate's tokens are the whole row: the engine prepends the
        # pending token in ``_BaseCycle.verify_candidates``, so the row is
        # (pending, d1, ..., dk) and the block width is the longest row as it
        # stands. Adding one here would verify a column of padding and count it
        # as a drafted token.
        width = max(len(d.tokens) for d in drafts)
        rows = [_pad_draft(d, width) for d in drafts]

        # One forward per sequence, one host sync for the batch.
        #
        # Lockstep batched verify is the shape this method is written for and
        # it is not finished: ``ModelState.batch`` builds joined caches, the
        # forward advances those, and nothing puts the rows back into the
        # per-sequence states afterwards, so a second stream reads a state that
        # never moved. That is W4.2, and it is a change in the state module
        # rather than here. Until it lands, the batch is dispatched as separate
        # forwards, which is correct, gives up the row-count win the expert
        # gather pays for (300 GB/s at one row against 549 at eight), and keeps
        # the acceptance reduction and the single host sync exactly as they
        # will be: the graph for every row is built before anything is
        # evaluated, so the cycle still crosses to the host once.
        verify_start = time.perf_counter()
        # A one-column block has nothing to roll back to, so it stages no
        # snapshot. Staging one is a full copy of the recurrent state, around
        # 110 MiB, and paying that on every decode cycle to protect a rollback
        # that cannot happen is most of a decode step.
        results = [
            self.model.verify([row], state, snapshot=width > 1)
            for row, state in zip(rows, model_states)
        ]
        verify_ms = (time.perf_counter() - verify_start) * 1000.0

        accept_start = time.perf_counter()
        accept_parts = []
        bonus_parts = []
        for row, result in zip(rows, results):
            argmax = mx.argmax(result.logits, axis=-1).astype(mx.int32)
            if width > 1:
                drafted = mx.array([row[1:]], dtype=mx.int32)
                agree = (argmax[:, :-1] == drafted).astype(mx.int32)
                counted = mx.cumprod(agree, axis=1).sum(axis=1)
            else:
                # Depth zero: one column, nothing drafted, the argmax is it.
                counted = mx.zeros((1,), dtype=mx.int32)
            index = mx.minimum(counted, width - 1)
            accept_parts.append(counted)
            bonus_parts.append(mx.take_along_axis(argmax, index[:, None], axis=1)[:, 0])
        accept_vector = mx.concatenate(accept_parts)
        bonus = mx.concatenate(bonus_parts)
        # The single host sync of the cycle: two small integer vectors cross,
        # never the logits.
        mx.eval(accept_vector, bonus)
        counts = accept_vector.tolist()
        bonuses = bonus.tolist()
        accept_ms = (time.perf_counter() - accept_start) * 1000.0

        outcomes: list[VerifyOutcome] = []
        accepted_counts: list[int] = []
        for draft, n in zip(drafts, counts):
            # Column 0 of the row is the pending token, which was never a
            # draft. The drafted chain is everything after it, so a row of
            # width w carries w - 1 drafts and the accepted run is the slice
            # that starts at column 1.
            n_drafted = len(draft.tokens) - 1
            accepted_counts.append(min(int(n), n_drafted))

        # Rollback, once for the whole block, and only when something was
        # rejected. The forward left every state covering ``width`` more
        # tokens; a row that committed all of them is already where it belongs,
        # and the cheapest correct rollback of nothing is not doing one.
        for state, result, n in zip(model_states, results, accepted_counts):
            if n + 1 < width:
                self.model.rollback_verify(state, result.gdn_states, [n], width)

        for draft, model_state, n, extra in zip(
            drafts, model_states, accepted_counts, bonuses
        ):
            committed = n + 1
            entry_length = model_state.length - width
            model_state.length = entry_length + committed
            model_state.prune_snapshots(entry_length)
            outcomes.append(
                VerifyOutcome(
                    sequence_id=draft.sequence_id,
                    accepted=tuple(draft.tokens[1 : n + 1]),
                    bonus=int(extra),
                    n_drafted=len(draft.tokens) - 1,
                )
            )

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
    def stage_snapshot(self, state: StateHandle, length: int) -> None:
        model_state = self._state(state)
        if model_state.length != length:
            raise StateError(
                f"cannot stage a snapshot at {length}: the state covers "
                f"{model_state.length} tokens"
            )
        model_state.stage_snapshot(length, pinned=True)

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
