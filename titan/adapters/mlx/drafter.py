"""``MTPDrafter``: the Lightning MTP head driven as a chain, one sync a cycle.

The head is a single sparse-attention layer that predicts token *t+2* from the
backbone's residual stream at position *t* and the embedding of token *t+1*.
That is the whole contract, and everything here follows from it.

## What one proposal does

    fold    -> one head call over every token the last cycle committed,
               against the persistent head cache, logits for the last position
    chain   -> depth-1 further calls, each re-entering the head on its own
               output hidden, against a clone of the head cache
    gate    -> drop the tail of the chain past the first draft whose
               probability is below ``p_min``
    read    -> one ``mx.async_eval`` and one ``.tolist()`` for the whole batch

The fold is the piece that is easy to get wrong. It is tempting to feed the
head one position -- the pending token and the hidden beside it -- and take the
draft from that. Then the head's KV is missing every token the cycle accepted,
its positions drift from the sequence's by the accepted count every cycle, and
acceptance decays for a reason nothing in the profile names. So the fold covers
the whole committed run: the backbone left a hidden state for every column of
the verify block, the accepted prefix of that block is exactly the run, and
folding all of it appends one head KV entry per committed token. The head cache
stays a committed-only mirror of the sequence, which is also what makes the
rollback question disappear -- the drafts never touched it.

## The two chain forms

Both are exact in the sense that matters: a draft is only ever a proposal, and
``verify`` decides what is committed, so neither form can change output. They
differ only in acceptance, so both are here and both are measured.

``head_output`` (the default) is vLLM's form, verified in `qwen3_next_mtp.py`:
each further step is fed the head's own output hidden *after* the head's final
norm, and the head re-applies ``pre_fc_norm_hidden`` to it on the way back in,
so the drafter behaves as a recursive invocation of the target. In this
architecture the final norm is ``hyper_connection_mixer``, so the fed-back
tensor is the mixer output, lifted back to the hyper-connection width the same
way the trunk lifts a token embedding.

``omlx`` is what oMLX does (`batch_generator.py:2422`): the step re-enters on
``head_hidden``, the head layer's output *before* the mixer, which is already
at stream width and needs no lift. EAGLE 3.1 attributes long-context acceptance
collapse to exactly this choice, which is why it is a switch and not a comment.

## Cost

One head call per drafted token, each one row wide. The head is one layer out
of 49, so a draft step is roughly 2% of a verify row in weights touched, and
the measured marginal cost of a chain step is the verify row it adds, not the
draft. That is why the p_min gate is worth having even though it cannot save
the draft compute: it saves the verify column, which is the expensive half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import mlx.core as mx

from titan.core.types import DraftCandidate, SequenceId, StateHandle, VerifyOutcome

__all__ = ["MTPDrafter", "CHAIN_FORMS"]

CHAIN_FORMS = ("head_output", "omlx")
"""The two ways step ``i+1`` is fed. See the module docstring."""


def _sync_and_read(values: mx.array) -> list[float]:
    """The one host sync of a proposal. Named so a test can count it."""
    mx.async_eval(values)
    return values.tolist()


@dataclass
class _Track:
    """What the drafter remembers about one sequence between cycles."""

    tokens_seen: int = 0
    """Length of the token list at the end of the last proposal."""
    fed: int = 0
    """Tokens folded into the head cache so far. The head's expected offset."""
    last_cycle: int = 0
    resets: int = 0


@dataclass
class _Plan:
    """One sequence's slot in the flat read-back."""

    index: int
    depth: int
    offset: int = 0


class MTPDrafter:
    """``Drafter`` over the Lightning MTP head. Chain of depth 1 to 8.

    Correct for batch sizes above one by proposing per sequence: the cycle
    dispatches one verify per sequence today, and the head is one row wide, so
    there is nothing to be gained by batching the head that the backbone is not
    already giving up. What is shared across the batch is the read-back: every
    sequence's chain lands in one array and crosses to the host once.
    """

    name = "mtp"

    def __init__(
        self,
        backend: Any,
        *,
        chain: str = "head_output",
        p_min: float = 0.0,
        max_depth: int = 8,
        shortlist: Any = None,
        profiler: Any = None,
    ) -> None:
        if chain not in CHAIN_FORMS:
            raise ValueError(
                f"unknown MTP chain form {chain!r}, expected one of {CHAIN_FORMS}"
            )
        self.backend = backend
        self.model = backend.model
        self.chain = chain
        self.p_min = float(p_min)
        self.gated = self.p_min > 0.0
        """Whether the read-back carries a probability beside every draft id.

        A floor of zero is the common case and a softmax over 248,320 logits
        per chain step is not free, so the probability is computed only when
        something is going to read it. The read-back stride follows."""
        self.max_depth = int(max_depth)
        # The registry's top-k shortlist lane, when there is one. Absent today
        # and not blocked on: greedy argmax is what the verify compares against,
        # so a shortlist can only ever be a second candidate lane beside it.
        self.shortlist = shortlist
        self.profiler = profiler
        self.host_syncs = 0
        self.gated_steps = 0
        self.cache_resets = 0
        self._tracks: dict[int, _Track] = {}
        self._cycle = 0

    # -- the Drafter port --------------------------------------------------
    def propose(
        self,
        states: Sequence[StateHandle],
        contexts: Sequence[Sequence[int]],
        depth: Sequence[int],
    ) -> list[DraftCandidate]:
        self._cycle += 1
        plans: list[_Plan] = []
        parts: list[mx.array] = []
        candidates: list[DraftCandidate] = [
            DraftCandidate(sequence_id=SequenceId(int(h)), tokens=(), source="none")
            for h in states
        ]
        stride = 2 if self.gated else 1
        for index, (handle, context, want) in enumerate(zip(states, contexts, depth)):
            want = min(int(want), self.max_depth)
            chain = self._chain_for(handle, context, want)
            if chain is None:
                continue
            plans.append(
                _Plan(index=index, depth=len(chain) // stride, offset=len(parts))
            )
            parts.extend(chain)
        if not parts:
            return candidates

        flat = mx.concatenate([p.reshape(1) for p in parts]).astype(mx.float32)
        self.host_syncs += 1
        read = _sync_and_read(flat)

        for plan in plans:
            tokens: list[int] = []
            logprobs: list[float] = []
            for step in range(plan.depth):
                token = read[plan.offset + stride * step]
                if not self.gated:
                    tokens.append(int(token))
                    continue
                probability = read[plan.offset + stride * step + 1]
                if probability < self.p_min:
                    self.gated_steps += 1
                    break
                tokens.append(int(token))
                logprobs.append(math.log(max(probability, 1e-9)))
            candidates[plan.index] = DraftCandidate(
                sequence_id=SequenceId(int(states[plan.index])),
                tokens=tuple(tokens),
                source="mtp" if tokens else "none",
                draft_logprobs=tuple(logprobs),
            )
        return candidates

    def observe(self, outcomes: Sequence[VerifyOutcome]) -> None:
        """Nothing to learn here. Depth is the controller's question.

        The drafter deliberately keeps no acceptance history of its own. The
        rejected confidence gate lived inside the overlay's drafter and could
        never be measured against batch composition, which is why the policy
        that decides how many rows to spend lives in the engine instead.
        """
        return None

    # -- one sequence ------------------------------------------------------
    def _chain_for(
        self,
        handle: StateHandle,
        context: Sequence[int],
        depth: int,
    ) -> Optional[list[mx.array]]:
        """Dispatch one sequence's chain. Returns ``[id, p, id, p, ...]``.

        ``None`` means this sequence drafts nothing this cycle, which the cycle
        handles by verifying one row for it. That is the honest answer on the
        first cycle of a sequence (the backbone has left no hidden state yet,
        because prefill never asks for one) and after anything that put the
        head cache out of step with the sequence.
        """
        state = self.backend.draft_state(handle)
        track = self._tracks.get(int(handle))
        if track is None:
            track = _Track()
            self._tracks[int(handle)] = track
            self._prune_tracks()
        track.last_cycle = self._cycle

        committed = len(context) - track.tokens_seen
        track.tokens_seen = len(context)
        hidden = state.mtp_hidden
        if depth <= 0 or hidden is None or committed <= 0:
            return None
        if committed > hidden.shape[1] or committed > len(context):
            # The state moved by something other than a verify this drafter
            # saw: a restored prefix, a stop truncation, a cycle it sat out.
            # The head cache is now a mirror of a sequence that no longer
            # exists, so it is dropped rather than fed a gap.
            self._reset_head(state, track)
            return None
        if not self._aligned(state, track):
            self._reset_head(state, track)
            return None

        # The fold. Column j of the verify block held the hidden state the
        # backbone produced for the token at that column, and the committed run
        # is the accepted prefix of that block plus the bonus, so hidden column
        # j pairs with committed token j+1 exactly.
        fold_hidden = hidden[:, :committed, :]
        fold_ids = [list(context[-committed:])]
        step = self.model.mtp_step(fold_hidden, fold_ids, state.mtp_layers)
        track.fed += committed

        out: list[mx.array] = []
        token = self._emit(step, out)
        if depth > 1:
            cache = self._clone_head(state)
            for _ in range(depth - 1):
                nxt = self._next_hidden(step)
                step = self.model.mtp_step(nxt, token.reshape(1, 1), cache)
                token = self._emit(step, out)
        return out

    def _emit(self, step: Any, out: list) -> mx.array:
        """Queue one step's draft id, and its probability when it is read."""
        token = mx.argmax(step.logits[:, -1, :], axis=-1).astype(mx.int32)
        out.append(token.reshape(1))
        if self.gated:
            out.append(self._probability(step.logits))
        return token

    def _next_hidden(self, step: Any) -> mx.array:
        """What the next chain step is fed. The one difference between forms."""
        if self.chain == "omlx":
            return step.streams[:, -1:, :]
        return self.model.mtp_lift(step.mixed[:, -1:, :])

    @staticmethod
    def _probability(logits: mx.array) -> mx.array:
        """Top-token probability of one draft step, on device."""
        return mx.max(mx.softmax(logits[:, -1, :].astype(mx.float32), axis=-1)).reshape(
            1
        )

    # -- the head cache ----------------------------------------------------
    def _aligned(self, state: Any, track: _Track) -> bool:
        """Is the head cache still a mirror of the committed sequence?

        The head appends one entry per committed token and nothing else appends
        to it, so its offset and the drafter's own count of folded tokens have
        to agree. They can disagree for one reason that is nobody's bug:
        ``ModelState.truncate`` trims the head caches by the *trunk's* delta on
        the stop path, because it has no way to know the head advanced by a
        different amount. A mismatch is cheap to detect and cheap to fix, and
        it is worth detecting because the failure it prevents is silent.
        """
        for cache in state.mtp_layers:
            if int(getattr(cache, "offset", track.fed)) != track.fed:
                return False
        return True

    def _reset_head(self, state: Any, track: _Track) -> None:
        """Start the head's KV again from empty. Costs acceptance, not output."""
        fresh = self.model.language_model.make_mtp_cache()
        state.mtp_layers[:] = fresh
        track.fed = 0
        track.resets += 1
        self.cache_resets += 1
        if self.profiler is not None:
            self.profiler.event("mtp_head_cache_reset", fed=track.fed)

    def _clone_head(self, state: Any) -> list:
        """A throwaway copy of the head KV for the speculative tail.

        Steps 2..k are drafts about drafts: nothing they append is committed,
        and the next cycle's fold would read their KV as history. Copying is
        cheap here in a way it is not for the trunk -- one layer, two kv heads,
        one row -- and it is what keeps the persistent cache committed-only
        without a trim on every rejection.
        """
        clone: list = []
        for cache in state.mtp_layers:
            extract = getattr(cache, "extract", None)
            if extract is None:
                raise TypeError(
                    f"{type(cache).__name__} cannot be cloned for the draft chain"
                )
            clone.append(extract(0))
        return clone

    def _prune_tracks(self) -> None:
        """Forget sequences that have not proposed in a long time."""
        if len(self._tracks) <= 256:
            return
        cutoff = self._cycle - 1024
        for key in [k for k, t in self._tracks.items() if t.last_cycle < cutoff]:
            del self._tracks[key]
