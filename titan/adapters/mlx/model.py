"""The forward API: four calls, explicit state, no hidden turn-to-turn memory.

    prefill(tokens, state, chunk)  -> PrefillResult   (logits optional)
    decode(tokens, state)          -> logits [B, V]
    verify(tokens, state)          -> logits [B, M, V]
    mtp_draft(hidden, tokens, state) -> logits [B, D, V]

Every one takes the :class:`~titan.adapters.mlx.state.ModelState` it reads and
writes.  Nothing is stashed on the model between calls.

Exactness with respect to the vendored originals
------------------------------------------------

*Exact.*  ``prefill`` and ``decode`` reproduce the vendored
``LanguageModel.__call__`` for the same inputs, up to the op-registry
substitutions that ``titan/adapters/mlx/VENDORED.md`` lists as bit-identical.
Chunking a prompt is exact by construction: the caches carry the offset and the
QSA indexer state across chunk boundaries, which is what the model does under
oMLX too.

*Exact, conditionally.*  ``verify`` reproduces the target model's greedy
continuation for the accepted prefix.  Padding a batch changes nothing about a
row's own output as long as the padding is on the left and each row's window is
gathered separately; the padded batched-QSA arm is the one place where the
result can move by a single bf16 ULP against the unbatched reference (round3/
qsa-batched measured exactly 0 at L=4 and <= 1 ULP at L=1).  Pass
``route="loop"`` for the bit-identical, slower reference.

*Approximate.*  ``mtp_draft`` is a draft: its logits are not the target model's
and are never emitted directly.  Only ``verify`` decides what is committed, so
the drafter's numerics cannot change output, only the acceptance rate.

The ops flagged approximate in VENDORED.md (the chunked GDN scan, the int8
expert gather, the grouped bf16 norm) move prefill numerics slightly.  Note the
warning from ``engine/patches/ple-fix/REPORT.md``: on this model, 36 recurrent
GDN layers amplify a one-ULP change into a large divergence over a few thousand
tokens, so bitwise agreement with a reference server is not a valid acceptance
test for any numeric change here.  The parity harness exists to measure that
divergence, not to assert it away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import mlx.core as mx

from .state import PHASE_DECODE, PHASE_PREFILL, PHASE_VERIFY, ModelState, StateError

logger = logging.getLogger(__name__)

DEFAULT_PREFILL_CHUNK = 2048


@dataclass
class PrefillResult:
    """What a prefill chunk produced."""

    logits: Optional[mx.array]
    hidden: Optional[mx.array]
    length: int


class TitanQwenFlashNext:
    """Titan's handle on the vendored Qwen3.8-Flash-Next model.

    Holds the weights and nothing else.  Two sequences share one instance and
    interfere only through the GPU.
    """

    def __init__(self, model, *, prefill_chunk: int = DEFAULT_PREFILL_CHUNK):
        self.model = model
        self.language_model = model.language_model
        self.args = model.config.text_config
        self.prefill_chunk = int(prefill_chunk)

    # -- properties the scheduler asks for ---------------------------------
    @property
    def n_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    @property
    def max_context(self) -> int:
        return self.args.max_position_embeddings

    @property
    def draft_depth_max(self) -> int:
        if self.language_model.get_mtp_module() is None:
            return 0
        return int(getattr(self.language_model, "_titan_mtp_depth", 1))

    def new_state(self) -> ModelState:
        return ModelState.new(self.model)

    # -- forward -----------------------------------------------------------
    def prefill(
        self,
        tokens: Sequence[int] | mx.array,
        state: ModelState,
        chunk: Optional[int] = None,
        *,
        want_logits: bool = False,
        want_hidden: bool = False,
        snapshot_every: Optional[int] = None,
    ) -> PrefillResult:
        """Run *tokens* into *state*, one chunk at a time.

        Chunking is the memory contract, not an optimisation: a 2048-token chunk
        is what the guard sizes for.  ``want_logits`` is honoured for the final
        position of the final chunk only; skipping the head elsewhere is worth
        48-95 ms per chunk on this model.
        """
        if state.is_batched:
            raise StateError("prefill runs one sequence per turn")
        ids = _as_batch(tokens)
        total = ids.shape[1]
        if total == 0:
            return PrefillResult(None, None, state.length)
        chunk = int(chunk or self.prefill_chunk)
        state.phase = PHASE_PREFILL

        logits = hidden = None
        start = 0
        while start < total:
            stop = min(start + chunk, total)
            piece = ids[:, start:stop]
            last = stop >= total
            self._prefetch_ple(ids, start, stop, chunk)
            want_head = want_logits and last
            output = self.language_model(
                piece,
                cache=state.layers,
                return_hidden=want_hidden and last,
                # Skipping the head is worth 48-95 ms on a 2048-token chunk;
                # only the final position of the final chunk is ever sampled.
                skip_logits=not want_head,
            )
            state.length += piece.shape[1]
            if last:
                logits = output.logits[:, -1:, :] if want_head else None
                hidden = _first_hidden(output) if want_hidden else None
            if snapshot_every and state.length % snapshot_every == 0:
                state.stage_snapshot()
            start = stop
        if hidden is not None:
            state.mtp_hidden = hidden
        return PrefillResult(logits=logits, hidden=hidden, length=state.length)

    def decode(
        self,
        tokens: Sequence[int] | mx.array,
        state: ModelState,
        *,
        want_hidden: bool = False,
    ) -> mx.array:
        """One token per row. Returns ``[B, vocab]``. Does not sync."""
        ids = _as_rows(tokens)
        if ids.shape[1] != 1:
            raise StateError("decode takes exactly one token per row")
        if ids.shape[0] != state.rows:
            raise StateError(
                f"{ids.shape[0]} tokens for {state.rows} rows of state"
            )
        state.phase = PHASE_DECODE
        output = self.language_model(
            ids,
            cache=state.layers,
            return_hidden=want_hidden,
        )
        state.length += 1
        if want_hidden:
            state.mtp_hidden = _first_hidden(output)
        return output.logits[:, -1, :]

    def verify(
        self,
        tokens: Sequence[Sequence[int]] | mx.array,
        state: ModelState,
        *,
        snapshot: bool = True,
        want_hidden: bool = True,
    ) -> mx.array:
        """Target forward over a drafted block. Returns ``[B, M, vocab]``.

        A verify is the only call that can be rolled back, so it stages the
        recurrent snapshot first: the 36 GDN layers have no inverse, and a
        rejected draft must leave no trace.
        """
        ids = _as_rows(tokens)
        if ids.shape[0] != state.rows:
            raise StateError(
                f"{ids.shape[0]} draft rows for {state.rows} rows of state"
            )
        if snapshot:
            state.stage_snapshot()
        state.phase = PHASE_VERIFY
        output = self.language_model(
            ids,
            cache=state.layers,
            target_verify=True,
            return_hidden=want_hidden,
        )
        state.length += ids.shape[1]
        if want_hidden:
            state.mtp_hidden = _first_hidden(output)
        return output.logits

    def mtp_draft(
        self,
        hidden: Optional[mx.array],
        tokens: Sequence[int] | mx.array,
        state: ModelState,
        *,
        depth: int = 1,
    ) -> mx.array:
        """Draft ``depth`` tokens from the MTP head. Returns ``[B, depth, V]``.

        ``hidden`` is the residual stream the target left behind; passing
        ``None`` uses the one the last forward stored on *state*.  The head runs
        against its own KV (``state.mtp_layers``), so drafting never touches the
        target's caches and a rejected draft costs nothing to undo.
        """
        if self.language_model.get_mtp_module() is None:
            raise StateError("this model was loaded without the MTP head")
        source = state.mtp_hidden if hidden is None else hidden
        if source is None:
            raise StateError(
                "no hidden state to draft from: run prefill or decode with "
                "want_hidden=True first"
            )
        ids = _as_rows(tokens)
        logits = self.language_model.mtp_forward(
            source,
            ids,
            state.mtp_layers,
            logits_keep=depth,
        )
        return logits

    # -- helpers -----------------------------------------------------------
    def _prefetch_ple(self, ids: mx.array, start: int, stop: int, chunk: int) -> None:
        """Start the next chunk's n-gram row reads during this chunk.

        The packed table lives on SSD; the gather-ahead is what keeps a cold
        chunk at 456 ms instead of 822 (engine/patches/ple-fix).
        """
        next_stop = min(stop + chunk, ids.shape[1])
        if next_stop <= stop:
            return
        try:
            self.language_model.prefetch_ple(ids[:, stop:next_stop], ids[:, start:stop])
        except Exception as exc:  # noqa: BLE001 - prefetch is best effort
            logger.debug("PLE gather-ahead skipped: %s", exc)

    def warmup(self, *, chunk: Optional[int] = None) -> None:
        """Compile every kernel shape the engine will use, once."""
        state = self.new_state()
        tokens = [self.args.eos_token_id if isinstance(self.args.eos_token_id, int)
                  else (self.args.eos_token_id or [0])[0]] * 8
        self.prefill(tokens, state, chunk or self.prefill_chunk, want_logits=True,
                     want_hidden=self.draft_depth_max > 0)
        self.decode([tokens[0]], state)
        if self.draft_depth_max:
            self.mtp_draft(None, [tokens[0]], state)
        mx.eval(*[v for v in (state.mtp_hidden,) if v is not None])


def _as_batch(tokens) -> mx.array:
    if isinstance(tokens, mx.array):
        return tokens if tokens.ndim == 2 else tokens[None]
    return mx.array([list(tokens)], dtype=mx.int64)


def _as_rows(tokens) -> mx.array:
    if isinstance(tokens, mx.array):
        return tokens if tokens.ndim == 2 else tokens[:, None]
    rows = list(tokens)
    if rows and isinstance(rows[0], (list, tuple)):
        return mx.array([list(r) for r in rows], dtype=mx.int64)
    return mx.array([[int(t)] for t in rows], dtype=mx.int64)


def _first_hidden(output) -> Optional[mx.array]:
    states = getattr(output, "hidden_states", None)
    return states[0] if states else None
