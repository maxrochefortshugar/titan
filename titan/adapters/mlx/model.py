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


@dataclass
class VerifyResult:
    """What a verify forward produced, including what a rollback needs.

    ``gdn_states`` is the per-layer intermediate the Gated DeltaNet arm keeps
    while it runs a block, and it is the whole reason rollback needs no replay:
    the recurrent state at any position inside the block can be rebuilt from
    it. The vendored ``rollback_speculative_cache`` is what consumes it, and it
    is only populated when the forward ran with the capture on.
    """

    logits: mx.array
    gdn_states: Optional[list]


@dataclass
class MTPStep:
    """What one call into the MTP head produced. See :meth:`mtp_step`."""

    logits: mx.array
    mixed: mx.array
    streams: mx.array


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

    @property
    def layer_layout(self) -> tuple[str, ...]:
        """The per-layer cache kind, in layer order.

        Three kinds, because three shapes of bytes: ``qsa`` for the sparse
        attention layers that carry sliceable KV, ``gdn`` for the recurrent
        layers that carry two slots, and ``gdn+ple`` for the recurrent layers
        that also carry the n-gram token history and therefore four. The cache
        signature hashes this, so a build that moves one layer makes every
        stored byte unreachable rather than making it wrong.
        """
        ple = set(getattr(self.args, "ple_layer_ids", ()) or ())
        layout: list[str] = []
        for index, kind in enumerate(self.args.layer_types):
            if kind == "linear_attention":
                layout.append("gdn+ple" if index in ple else "gdn")
            else:
                layout.append("qsa")
        return tuple(layout)

    @property
    def state_bytes_per_token(self) -> float:
        """Device bytes one more token of context costs, from the shapes.

        Only the 12 sparse-attention layers scale with context: two tensors of
        ``kv_heads x head_dim`` each, plus the indexer's raw key and its
        position id. The 36 recurrent layers do not, whatever the context
        length, which is the entire reason this model is worth serving at 64k.

        Computed rather than assumed. The guard's own default is an order of
        magnitude above this on this checkpoint, and a guard that overestimates
        does not fail loudly: it skips the long prompt, keeps its place in the
        queue, and serves everything behind it forever.
        """
        args = self.args
        kv_bytes = 2 if args.num_key_value_heads else 2
        qsa_layers = sum(1 for k in args.layer_types if k != "linear_attention")
        per_layer = (
            2 * args.num_key_value_heads * args.head_dim * kv_bytes
            + args.indexer_kv_heads * args.indexer_head_dim * 2
            + 4
        )
        return float(qsa_layers * per_layer)

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

        ``want_hidden`` used to be a trap.  Asking the vendored model for
        hidden states sets ``capture_layer_ids``, which used to turn on its
        ``target_verify`` arms whatever the block's width, and those arms run
        one kernel launch per row: they are built for a verify block a few
        columns wide.  Over a prefill chunk that was three orders of magnitude
        slower, measured at 200 seconds for a 150-token chunk against 0.1
        without.  The arms are now chosen by width rather than by the capture
        (``_narrow_verify_block``, and ``docs/architecture/FORWARD.md``
        section 3), so a chunk gets chunk-shaped work and its hidden state.
        A wide chunk does not carry the recurrent intermediates, which are a
        verify-block facility and cost one state per recurrent layer per row;
        rollback is unaffected because every width the engine verifies at is
        narrow.
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
            # Evaluate the chunk before starting the next one. MLX is lazy, so
            # without this the whole prompt is one unevaluated graph: it holds
            # every intermediate live at once, the guard reads an active-memory
            # figure that has not happened yet, and the entire prefill is billed
            # to whatever forces the first evaluation, which is the first decode
            # cycle. A 65k prompt showed up as a single 90-second decode cycle.
            # The work is the same work; this is where it belongs.
            mx.eval(_cache_arrays(state))
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
        snapshot: bool = False,
        want_hidden: bool = True,
    ) -> VerifyResult:
        """Target forward over a drafted block. Logits are ``[B, M, vocab]``.

        The forward captures the Gated DeltaNet intermediates, which is what
        makes rollback replay-free: the recurrent state at any position inside
        the block can be rebuilt from them, so a rejected draft costs a rebuild
        rather than a second forward. ``capture_layer_ids`` is what turns the
        capture on in the vendored code, and asking for the hidden state is
        what sets it, so a verify that wants rollback wants the hidden state.

        ``snapshot`` is the older, coarser path: a full copy of the recurrent
        state before the forward, restored wholesale. It costs around 110 MiB
        of copies per cycle, so it is off by default and the caller turns it on
        only when it has something to roll back to that the capture cannot
        rebuild.
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
            return_hidden=True,
        )
        state.length += ids.shape[1]
        if want_hidden:
            state.mtp_hidden = _first_hidden(output)
        return VerifyResult(
            logits=output.logits, gdn_states=getattr(output, "gdn_states", None)
        )

    def rollback_verify(
        self, state: ModelState, gdn_states: Optional[list], accepted, width: int
    ) -> None:
        """Undo the rejected tail of a verify block, without a forward pass.

        The 12 attention layers trim by an offset move. The 36 recurrent layers
        are rebuilt at the accepted position from the intermediates the forward
        captured, which is the vendored model's own
        ``rollback_speculative_cache``. Nothing is recomputed and nothing is
        replayed, which is what the whole speculative path is worth.
        """
        rollback = getattr(self.language_model, "rollback_speculative_cache", None)
        if rollback is None or not gdn_states:
            raise StateError(
                "this build cannot roll back a verify block: the forward "
                "captured no recurrent intermediates"
            )
        rollback(state.layers, gdn_states, accepted, width)

    def mtp_step(
        self,
        hidden: mx.array,
        tokens: Sequence[int] | mx.array,
        cache: list,
    ) -> "MTPStep":
        """One MTP head call, returning both hidden states the chain can reuse.

        ``mtp_forward`` throws away everything the head produced except the
        logits, and a chained drafter needs one of two hidden states to
        continue from, so this runs the head module directly and keeps both.

        * ``mixed`` is the head's output after ``hyper_connection_mixer``,
          which is this architecture's final norm and the tensor the logits are
          read from. Shape ``[B, T, hidden_size]``.
        * ``streams`` is the head layer's output before that mixer, one
          residual stream per hyper connection. Shape
          ``[B, T, hc_count * hidden_size]``, which is the shape
          ``fuse_inputs`` takes directly, and it is what the backbone leaves on
          ``state.mtp_hidden``.

        Logits are for the last position only: a fold over the whole committed
        run needs every position's KV in the head cache, but only the last
        position predicts anything the drafter has not already committed.
        """
        mtp = self.language_model.get_mtp_module()
        if mtp is None:
            raise StateError("this model was loaded without the MTP head")
        ids = _as_rows(tokens)
        embed = self.language_model.model.embed_tokens
        mixed, streams = mtp(hidden, ids, embed, cache)
        source = mixed[:, -1:, :]
        if self.args.tie_word_embeddings:
            logits = embed.as_linear(source)
        else:
            logits = self.language_model.lm_head(source)
        return MTPStep(logits=logits, mixed=mixed, streams=streams)

    def mtp_lift(self, mixed: mx.array) -> mx.array:
        """Lift a post-mixer hidden ``[B, T, H]`` into the head's stream width.

        The trunk does exactly this to the token embedding on the way in
        (``mx.tile(hidden_states, (1, 1, hc_count))``, ``language.py:2748``), so
        it is the architecture's own way of turning one vector into the hyper
        connection streams. ``fuse_inputs`` then applies ``pre_fc_norm_hidden``
        to the result, which is the re-normalisation the vLLM chain relies on.
        """
        return mx.tile(mixed, (1, 1, self.args.hc_count))

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


def _cache_arrays(state: ModelState) -> list:
    """Every live array behind a state's caches, for a forced evaluation."""
    arrays: list = []
    for cache in list(state.layers) + list(state.mtp_layers):
        held = getattr(cache, "state", None)
        if held is None:
            continue
        for value in held if isinstance(held, (list, tuple)) else (held,):
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def _first_hidden(output) -> Optional[mx.array]:
    states = getattr(output, "hidden_states", None)
    return states[0] if states else None
