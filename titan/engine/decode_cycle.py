"""The decode cycle: the piece the overlay could not fix from outside.

One cycle advances every decoding sequence by at least one token:

    draft   -> Drafter.propose for all sequences, no host sync
    prefetch-> NgramReader.prefetch for the draft candidates' rows
    verify  -> ModelBackend.verify, one padded row block, one host sync
    commit  -> per sequence, accepted + bonus; rollback is a truncation
    emit    -> incremental detokenisation and events
    profile -> one CycleProfile, always

Four properties are designed in here rather than bolted on:

1. **Lockstep batching.** All sequences share one verify forward at a common
   row width. Ragged widths are padded, not split. The overlay's batched MTP
   lost to plain batching because it split.
2. **Replay-free rollback.** Rejected drafts are undone by truncating state to
   the accepted length, using the snapshot the verify pass staged for the 36
   recurrent layers. Nothing is recomputed.
3. **In-graph acceptance.** The comparison between drafted ids and target
   argmax happens on device; only a small integer vector crosses to the host,
   once per cycle. ``CycleProfile.host_syncs`` must read 1.
4. **N-gram prefetch.** Row ids for the draft candidates are queued before the
   verify forward, so the SSD read overlaps the GPU work instead of adding
   about 2.3 ms of host wait to each forward.

## The row layout, stated once

A decoding sequence always has exactly one token the backend has not consumed
yet: the token the previous cycle produced last. Call it the pending token. So
during decode the invariant is

    backend.state_length(state) == len(sequence.tokens) - 1

and one verify row block for a sequence is

    (pending, draft_1, ..., draft_k)

of width k+1. The forward consumes all k+1 positions, the acceptance reduction
keeps 1 + n of them where n is the number of confirmed drafts, and the bonus
token the forward produced at the first rejection point becomes the next
cycle's pending token. Committing ``drafts[:n] + (bonus,)`` is exactly what
plain greedy decoding would have produced, which is the lossless-speculation
invariant the parity tests check.

The plain M1 loop is the same cycle at k = 0: one row per sequence, no drafter,
one host sync, argmax taken in the graph. It is written as its own class rather
than as a branch, because M1 parity is measured against it and a branch inside
the speculative path is a branch that can drift.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from titan.core.errors import StateError, TitanError
from titan.core.types import (
    CycleProfile,
    DraftCandidate,
    FinishReason,
    SamplingParams,
    SequenceId,
    SequenceState,
    StopCondition,
    TokenEvent,
    VerifyOutcome,
)

__all__ = [
    "CycleResult",
    "DecodeCycle",
    "AcceptancePolicy",
    "GreedyAcceptance",
    "SampledAcceptance",
    "AcceptanceEstimator",
    "DepthController",
    "TextEmitter",
    "EmitResult",
    "NullProfiler",
    "MonotonicClock",
    "PlainDecodeCycle",
    "MTPDecodeCycle",
    "EMPTY_PROFILE",
]


# ---------------------------------------------------------------------------
# small utilities the engine owns rather than imports
# ---------------------------------------------------------------------------


class MonotonicClock:
    """Default :class:`titan.core.ports.Clock`. Injected everywhere else."""

    def now(self) -> float:
        return time.monotonic()


class NullProfiler:
    """A profiler that keeps the last cycle and counts events, nothing more.

    The real one is ``titan.observability.profiler``; the engine takes the port
    and this stand-in exists so a cycle can be constructed without one.
    """

    def __init__(self) -> None:
        self.cycles: list[CycleProfile] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def cycle(self, profile: CycleProfile) -> None:
        self.cycles.append(profile)

    def event(self, name: str, **fields: float | int | str) -> None:
        self.events.append((name, dict(fields)))

    def span(self, name: str) -> "_NullSpan":
        return _NullSpan()

    def snapshot(self) -> Mapping[str, Any]:
        return {"cycles": len(self.cycles), "events": len(self.events)}


class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


EMPTY_PROFILE = CycleProfile(
    cycle=0,
    n_sequences=0,
    n_rows=0,
    draft_ms=0.0,
    verify_ms=0.0,
    accept_ms=0.0,
    sample_ms=0.0,
    detok_ms=0.0,
    ngram_wait_ms=0.0,
    host_syncs=0,
    tokens_committed=0,
    tokens_drafted=0,
    wall_ms=0.0,
)


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Everything one cycle produced. Pure data; the caller does the emitting."""

    outcomes: tuple[VerifyOutcome, ...]
    events: tuple[TokenEvent, ...]
    finished: tuple[SequenceId, ...]
    profile: CycleProfile


class AcceptancePolicy(Protocol):
    """How a drafted token is judged against the target distribution.

    Greedy acceptance (id equality against the target argmax) is the only mode
    used at parity milestones, and it is exact by construction. Sampled
    acceptance uses the standard probability-ratio rule, which preserves the
    target distribution but not any particular sample, so it is never compared
    token-for-token against oMLX.
    """

    @property
    def is_exact(self) -> bool: ...

    def name(self) -> str: ...


class GreedyAcceptance:
    """Id equality against the target argmax. Exact, and the parity currency."""

    @property
    def is_exact(self) -> bool:
        return True

    def name(self) -> str:
        return "greedy"


class SampledAcceptance:
    """Probability-ratio acceptance with in-graph residual sampling.

    Distribution-preserving but not sample-preserving, so it is never compared
    token for token with a recorded reference. The arithmetic lives in the
    ``verify_accept`` op, which computes the residual for every position so the
    cycle still crosses to the host exactly once.
    """

    @property
    def is_exact(self) -> bool:
        return False

    def name(self) -> str:
        return "sampled"


def policy_for(sampling: SamplingParams) -> AcceptancePolicy:
    return GreedyAcceptance() if sampling.is_greedy else SampledAcceptance()


class DecodeCycle(Protocol):
    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        """Advance ``batch`` by one cycle.

        Preconditions: every sequence is DECODING, holds an open state handle,
        and its state length equals ``prompt_len + committed`` (which is
        ``len(tokens) - 1``, the pending token being the one the backend has
        not consumed).

        Postconditions: for each sequence, state length grew by exactly
        ``outcome.n_committed``; ``committed`` grew by the same; the sequence is
        finished if a stop condition matched inside the accepted run, in which
        case tokens after the stop are discarded and the state is truncated to
        the stop position. Discarding them matters beyond tidiness: a copy or
        draft block accepted past a stop token corrupts the recurrent state that
        the prefix cache is about to store.
        """


# ---------------------------------------------------------------------------
# acceptance history and the depth controller
# ---------------------------------------------------------------------------


class AcceptanceEstimator:
    """Rolling mean of accepted drafts per drafted chain.

    A window rather than an EWMA because the number the depth policy wants is
    "how many of the last N drafts stuck", and a window answers that without a
    time constant nobody can name. The measured decode budget says three
    accepted tokens per cycle is 111 tok/s and that the accepted median at 64k
    is 1, so the estimator has to fall as fast as the context grows.
    """

    def __init__(self, window: int = 64, initial: float = 1.0) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._initial = float(initial)
        self._samples: list[int] = []
        self._drafted: list[int] = []

    def observe(self, outcomes: Sequence[VerifyOutcome]) -> None:
        for outcome in outcomes:
            if outcome.n_drafted <= 0:
                continue
            self._samples.append(len(outcome.accepted))
            self._drafted.append(outcome.n_drafted)
        overflow = len(self._samples) - self.window
        if overflow > 0:
            del self._samples[:overflow]
            del self._drafted[:overflow]

    @property
    def mean_accepted(self) -> float:
        """Mean accepted drafts per chain. The number depth is planned from."""
        if not self._samples:
            return self._initial
        return sum(self._samples) / len(self._samples)

    @property
    def acceptance_rate(self) -> float:
        """Accepted over drafted. Reported, not used for depth: a chain that
        drafts 8 and accepts 1 has a fine rate and a terrible cycle."""
        drafted = sum(self._drafted)
        if drafted == 0:
            return 0.0
        return sum(self._samples) / drafted

    @property
    def n_samples(self) -> int:
        return len(self._samples)


class DepthController:
    """Adaptive draft depth. Satisfies :class:`titan.core.ports.Verifier`.

    Pure: the same inputs give the same depths, no clock and no device query.
    The rule is one line of arithmetic and two clamps, and it is deliberately
    the whole policy:

        depth = round(mean_accepted) + 1, clamped to [min, max] and to the
        row budget divided across the batch.

    The +1 is what makes it self-correcting. A cycle that accepts everything it
    drafted has no evidence about the depth above it, so the policy spends one
    speculative row to find out; a cycle that accepts nothing falls to the
    floor within a window. The overlay's fixed 3 could not do either, and at 64k
    where the accepted median is 1 it paid for two rejected rows every cycle.
    """

    def __init__(
        self,
        *,
        max_depth: int = 3,
        min_depth: int = 0,
        adaptive: bool = True,
        window: int = 64,
        rows_budget: int = 32,
    ) -> None:
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("need 0 <= min_depth <= max_depth")
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.adaptive = adaptive
        self.rows_budget = rows_budget
        self.estimator = AcceptanceEstimator(window=window, initial=float(max_depth))

    # -- the Verifier port -------------------------------------------------
    def plan_depth(
        self,
        n_sequences: int,
        recent_acceptance: float,
        rows_budget: int,
    ) -> list[int]:
        if n_sequences <= 0:
            return []
        if not self.adaptive:
            depth = self.max_depth
        else:
            depth = int(round(recent_acceptance)) + 1
        # Every sequence spends one row on its pending token, so the budget for
        # drafts is what is left after the batch has paid for itself.
        per_sequence = max(0, rows_budget // n_sequences - 1)
        depth = min(depth, self.max_depth, per_sequence)
        depth = max(depth, self.min_depth)
        return [depth] * n_sequences

    def record(self, profile: CycleProfile, outcomes: Sequence[VerifyOutcome]) -> None:
        self.estimator.observe(outcomes)

    # -- convenience for the cycle ----------------------------------------
    def next_depths(self, n_sequences: int) -> list[int]:
        return self.plan_depth(
            n_sequences, self.estimator.mean_accepted, self.rows_budget
        )


# ---------------------------------------------------------------------------
# stop conditions and streaming text
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmitResult:
    """What one committed run of tokens turned into for one sequence.

    ``keep`` is the number of tokens from the run that survive: a stop string
    or an EOS id inside the run discards the rest, and the caller truncates the
    backend state to match. ``text`` is safe to send: partial UTF-8 and any
    suffix that could still grow into a stop string are held back.
    """

    text: str
    keep: int
    finish_reason: FinishReason | None


class TextEmitter:
    """Streaming detokenisation plus stop detection for one sequence.

    Two things have to be true at once and neither is free. The client may
    never see a character of a stop string, so text is held back while a
    suffix could still become one. And the state the prefix cache is about to
    store may not contain tokens past the stop, so the emitter reports how many
    tokens of the run survive rather than only how much text to send.

    Token boundaries are tracked by feeding the tokenizer one id at a time and
    recording the length of the accumulated text after each. That is host-side
    string work at decode rates, and it is the only way to map a character
    offset (where the stop string starts) back to a token count (what the state
    has to be truncated to).
    """

    def __init__(
        self,
        tokenizer: Any,
        sequence_id: SequenceId,
        stop: StopCondition,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_id = sequence_id
        self.stop = stop
        self.text = ""
        self.emitted = 0
        self.tokens_seen = 0
        self._token_ends: list[int] = []
        self._holdback = max((len(s) for s in stop.stop_strings), default=1) - 1
        self._done = False

    # -- helpers -----------------------------------------------------------
    def _tokens_within(self, chars: int) -> int:
        """Tokens whose text ends at or before ``chars``."""
        return bisect_right(self._token_ends, chars)

    def _find_stop(self, search_from: int) -> tuple[int, int] | None:
        """Earliest (start, end) of a stop string at or after ``search_from``."""
        best: tuple[int, int] | None = None
        for needle in self.stop.stop_strings:
            if not needle:
                continue
            index = self.text.find(needle, search_from)
            if index < 0:
                continue
            if best is None or index < best[0]:
                best = (index, index + len(needle))
        return best

    def _safe_prefix(self) -> int:
        """Longest prefix of ``text`` that no stop string could still extend."""
        if not self.stop.stop_strings or self._holdback <= 0:
            return len(self.text)
        limit = len(self.text)
        tail_start = max(self.emitted, limit - self._holdback)
        for needle in self.stop.stop_strings:
            for size in range(min(len(needle) - 1, limit - tail_start), 0, -1):
                if self.text.endswith(needle[:size]):
                    limit = min(limit, len(self.text) - size)
                    break
        return max(self.emitted, limit)

    # -- the one call the cycle makes -------------------------------------
    def push(self, token_ids: Sequence[int], budget_left: int) -> EmitResult:
        """Append a committed run and decide what leaves the engine.

        ``budget_left`` is ``max_tokens`` minus the tokens already committed, so
        a cycle that accepts three tokens against a budget of one keeps one and
        the state is truncated to match. That is what makes ``max_tokens`` count
        accepted tokens rather than cycles.
        """
        if self._done:
            return EmitResult("", 0, None)
        search_from = max(0, self.emitted - self._holdback)
        base = self.tokens_seen
        keep = 0
        finish: FinishReason | None = None
        for index, token_id in enumerate(token_ids):
            if index >= budget_left:
                finish = FinishReason.LENGTH
                break
            if token_id in self.stop.eos_token_ids:
                # The EOS id itself is never text and never counts as output.
                finish = FinishReason.STOP
                break
            piece = self.tokenizer.decode_incremental(self.sequence_id, [token_id])
            self.text += piece
            self._token_ends.append(len(self.text))
            self.tokens_seen += 1
            keep += 1
            hit = self._find_stop(search_from)
            if hit is not None:
                start, _end = hit
                # Token counts are absolute over the whole stream; ``keep`` is
                # relative to this run. A stop string that began in an earlier
                # run therefore keeps nothing of this one.
                surviving = self._tokens_within(start)
                keep = max(0, surviving - base)
                self.text = self.text[:start]
                del self._token_ends[surviving:]
                self.tokens_seen = surviving
                finish = FinishReason.STOP
                break
        if finish is not None:
            self._done = True
            cut = len(self.text)
        else:
            cut = self._safe_prefix()
        text = self.text[self.emitted : cut]
        self.emitted = cut
        return EmitResult(text=text, keep=keep, finish_reason=finish)

    def flush(self) -> str:
        """Release anything held back. Called once, when the sequence ends."""
        tail = self.tokenizer.flush_incremental(self.sequence_id)
        if tail and not self._done:
            self.text += tail
        out = self.text[self.emitted :]
        self.emitted = len(self.text)
        return out


# ---------------------------------------------------------------------------
# the cycles
# ---------------------------------------------------------------------------


class _BaseCycle:
    """Commit and emit, shared by the plain loop and the MTP cycle.

    Everything that decides *what* runs is in the subclasses. Everything that
    decides what the result means -- which tokens are committed, where a stop
    lands, what the state has to be truncated to, what the client sees -- is
    here, once, so the two paths cannot produce different token streams for the
    same backend. The lossless-speculation test is exactly the assertion that
    they do not.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        clock: Any = None,
        profiler: Any = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.clock = clock or MonotonicClock()
        self.profiler = profiler or NullProfiler()
        self._emitters: dict[int, TextEmitter] = {}
        self._cycle_index = 0

    # -- emitter lifecycle -------------------------------------------------
    def emitter_for(self, sequence: SequenceState) -> TextEmitter:
        emitter = self._emitters.get(int(sequence.sequence_id))
        if emitter is None:
            emitter = TextEmitter(
                self.tokenizer, sequence.sequence_id, sequence.request.stop
            )
            self._emitters[int(sequence.sequence_id)] = emitter
        return emitter

    def release(self, sequence: SequenceState) -> str:
        """Drop a sequence's detokenisation stream and return its tail."""
        emitter = self._emitters.pop(int(sequence.sequence_id), None)
        if emitter is None:
            return self.tokenizer.flush_incremental(sequence.sequence_id)
        return emitter.flush()

    # -- verify rows -------------------------------------------------------
    @staticmethod
    def verify_candidates(
        batch: Sequence[SequenceState],
        proposals: Mapping[int, tuple[int, ...]],
    ) -> list[DraftCandidate]:
        """Build the row block: pending token first, then the drafted chain.

        The drafter proposes a continuation; it does not know about the pending
        token, and it must not, because the pending token is scheduler state.
        Prepending it here is the only place the row layout is decided.
        """
        rows: list[DraftCandidate] = []
        for sequence in batch:
            pending = sequence.tokens[-1]
            drafted = proposals.get(int(sequence.sequence_id), ())
            rows.append(
                DraftCandidate(
                    sequence_id=sequence.sequence_id,
                    tokens=(pending, *drafted),
                    source="mtp" if drafted else "none",
                )
            )
        return rows

    @staticmethod
    def draft_budget(sequence: SequenceState) -> int:
        """How many drafts this sequence may spend without overrunning its budget.

        The verify forward always produces a bonus token, so a sequence with one
        token of budget left must draft nothing: the bonus alone finishes it.
        Clamping here rather than discarding the overrun afterwards is what
        keeps the state consistent -- a token committed past ``max_tokens`` has
        to be truncated away, and a truncation inside the verify block lands
        below the snapshot the block staged, which the backend is right to
        refuse. oMLX clamps in the same place and for the same reason.
        """
        stop = sequence.request.stop
        budget = stop.max_tokens - sequence.committed - 1
        if stop.max_total_tokens is not None:
            budget = min(budget, stop.max_total_tokens - len(sequence.tokens) - 1)
        return max(0, budget)

    # -- commit ------------------------------------------------------------
    def commit(
        self,
        batch: Sequence[SequenceState],
        outcomes: Sequence[VerifyOutcome],
        proposals: Mapping[int, tuple[int, ...]],
    ) -> tuple[list[TokenEvent], list[SequenceId], int]:
        """Apply one cycle's outcomes: tokens, text, stops, truncation."""
        by_id = {int(s.sequence_id): s for s in batch}
        events: list[TokenEvent] = []
        finished: list[SequenceId] = []
        committed_total = 0
        now = self.clock.now()
        for outcome in outcomes:
            sequence = by_id[int(outcome.sequence_id)]
            drafted = proposals.get(int(sequence.sequence_id), ())
            n_accepted = len(outcome.accepted)
            if n_accepted > len(drafted):
                raise StateError(
                    f"verify accepted {n_accepted} of {len(drafted)} drafted tokens"
                )
            run = (*drafted[:n_accepted], outcome.bonus)
            committed_total += len(run)

            emitter = self.emitter_for(sequence)
            budget_left = sequence.request.stop.max_tokens - sequence.committed
            result = emitter.push(run, budget_left)
            kept = run[: result.keep]
            before = len(sequence.tokens)
            sequence.tokens.extend(kept)
            # A stop string can span cycles: its first half may have been
            # committed a cycle ago, held back as text but already in the token
            # list. The emitter's absolute survivor count is the authority on
            # what is left, so the trim can reach back past this cycle. Those
            # tokens were never emitted as text, so nothing the client saw is
            # retracted; what changes is the prefix the cache is offered.
            surviving = sequence.prompt_len + emitter.tokens_seen
            if len(sequence.tokens) > surviving:
                del sequence.tokens[surviving:]
            sequence.committed = len(sequence.tokens) - sequence.prompt_len
            sequence.text_emitted += len(result.text)
            if sequence.first_token_at is None and kept:
                sequence.first_token_at = now
            if result.text or kept:
                events.append(
                    TokenEvent(
                        request_id=sequence.request.request_id,
                        token_ids=tuple(kept),
                        text=result.text,
                        timestamp=now,
                    )
                )

            finish = result.finish_reason
            if finish is None and sequence.committed >= sequence.request.stop.max_tokens:
                finish = FinishReason.LENGTH
            if finish is None:
                total = sequence.request.stop.max_total_tokens
                if total is not None and len(sequence.tokens) >= total:
                    finish = FinishReason.LENGTH
            if len(sequence.tokens) != before + len(run):
                self._truncate_to_kept(sequence)
            if finish is not None:
                sequence.finish_reason = finish
                finished.append(sequence.sequence_id)
        return events, finished, committed_total

    def _truncate_to_kept(self, sequence: SequenceState) -> None:
        """Undo the tokens a stop discarded, without a forward pass.

        The backend has consumed every token but the pending one, so the target
        length is ``len(tokens) - 1``. If the backend refuses -- the truncation
        point is below the snapshot the verify staged -- the sequence is still
        correct on the wire; what is lost is the right to store this prefix. The
        scheduler checks the state length against the token list before it
        stores, so a refused truncation drops the store rather than recording a
        length the cache cannot restore.
        """
        target = len(sequence.tokens) - 1
        if self.backend.state_length(sequence.state) <= target:
            return
        try:
            self.backend.truncate_state(sequence.state, target)
        except TitanError as exc:
            self.profiler.event(
                "truncate_refused",
                sequence=int(sequence.sequence_id),
                target=target,
                reason=str(exc),
            )

    # -- profile -----------------------------------------------------------
    def finish_profile(
        self,
        profile: CycleProfile,
        *,
        n_sequences: int,
        draft_ms: float,
        detok_ms: float,
        ngram_wait_ms: float,
        wall_ms: float,
        tokens_committed: int,
        tokens_drafted: int,
    ) -> CycleProfile:
        self._cycle_index += 1
        return replace(
            profile,
            cycle=self._cycle_index,
            n_sequences=n_sequences,
            draft_ms=draft_ms,
            detok_ms=detok_ms,
            ngram_wait_ms=ngram_wait_ms,
            wall_ms=wall_ms,
            tokens_committed=tokens_committed,
            tokens_drafted=tokens_drafted,
        )


class PlainDecodeCycle(_BaseCycle):
    """M1: one token per sequence per cycle, greedy, no speculation.

    The row block is one column wide, so the verify forward is a plain decode
    forward and the acceptance reduction degenerates to an argmax. It goes
    through ``verify`` rather than ``decode`` for one reason: the core may not
    read logits, and ``verify`` is the only port method that turns logits into
    token ids without handing the engine a device array. That also makes this
    loop and the MTP cycle share their sampling, which is what the parity test
    between them is worth anything for.
    """

    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        if not batch:
            return CycleResult((), (), (), EMPTY_PROFILE)
        started = self.clock.now()
        proposals: dict[int, tuple[int, ...]] = {}
        rows = self.verify_candidates(batch, proposals)
        states = [s.state for s in batch]
        sampling = [s.request.sampling for s in batch]
        outcomes, profile = self.backend.verify(states, rows, sampling)

        detok_start = self.clock.now()
        events, finished, committed = self.commit(batch, outcomes, proposals)
        detok_ms = (self.clock.now() - detok_start) * 1000.0
        wall_ms = (self.clock.now() - started) * 1000.0
        profile = self.finish_profile(
            profile,
            n_sequences=len(batch),
            draft_ms=0.0,
            detok_ms=detok_ms,
            ngram_wait_ms=0.0,
            wall_ms=wall_ms,
            tokens_committed=committed,
            tokens_drafted=0,
        )
        self.profiler.cycle(profile)
        return CycleResult(tuple(outcomes), tuple(events), tuple(finished), profile)


class MTPDecodeCycle(_BaseCycle):
    """M4: draft a chain of depth k, verify k+1 rows, accept in the graph.

    Sequence of one cycle, and the order is load-bearing:

    1. the depth controller hands out a depth per sequence from the rolling
       acceptance estimate;
    2. the drafter proposes, without syncing -- the chain is dispatched and
       left on the queue;
    3. the n-gram rows the draft implies are queued for prefetch, so the SSD
       read overlaps the verify forward rather than adding 2.3 ms in front of
       it;
    4. one verify forward over the padded row block, one host sync, acceptance
       and the bonus token computed in the graph by ``verify_accept``;
    5. commit, with rollback already done inside verify (the port's contract:
       state grows by exactly ``len(accepted) + 1``, never by the block width);
    6. one CycleProfile, always, with ``host_syncs`` straight from the backend.

    ### The batched-verify seam

    This class is already written for a batch: it plans a depth per sequence,
    builds one row per sequence, and makes exactly one ``verify`` call for the
    whole batch. What is not here yet is a common width. ``plan_depth`` returns
    a uniform depth today, so the block is rectangular by construction and the
    backend's padding never engages. Lockstep across ragged depths is a local
    change in two places and nowhere else: ``plan_depth`` may return different
    depths per sequence, and the backend pads the short rows. Nothing in
    ``commit`` knows the width -- it reads ``len(outcome.accepted)`` and the
    sequence's own proposal -- which is what keeps the seam local.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        drafter: Any = None,
        verifier: Any = None,
        ngram: Any = None,
        clock: Any = None,
        profiler: Any = None,
        max_depth: int = 3,
        rows_budget: int = 32,
    ) -> None:
        super().__init__(
            backend=backend, tokenizer=tokenizer, clock=clock, profiler=profiler
        )
        self.drafter = drafter
        self.ngram = ngram
        self.controller = verifier or DepthController(
            max_depth=min(max_depth, getattr(backend, "draft_depth_max", max_depth)),
            rows_budget=rows_budget,
        )

    # -- steps -------------------------------------------------------------
    def _propose(
        self, batch: Sequence[SequenceState]
    ) -> tuple[dict[int, tuple[int, ...]], int]:
        if self.drafter is None:
            return {}, 0
        depths = [
            min(depth, self.draft_budget(sequence))
            for sequence, depth in zip(batch, self.controller.next_depths(len(batch)))
        ]
        if not any(depths):
            return {}, 0
        states = [s.state for s in batch]
        contexts = [s.tokens for s in batch]
        candidates = self.drafter.propose(states, contexts, depths)
        proposals: dict[int, tuple[int, ...]] = {}
        drafted = 0
        # Keyed by the batch's own order rather than by the candidate's
        # sequence_id: the drafter is handed states and contexts, so it has no
        # authoritative id to return, and trusting one it invented is how a
        # draft ends up on the wrong sequence.
        for sequence, candidate, depth in zip(batch, candidates, depths):
            tokens = tuple(candidate.tokens)[:depth]
            if tokens:
                proposals[int(sequence.sequence_id)] = tokens
                drafted += len(tokens)
        return proposals, drafted

    def _prefetch(self, proposals: Mapping[int, tuple[int, ...]]) -> None:
        """Queue the n-gram rows the draft needs, before the verify forward."""
        if self.ngram is None or not proposals:
            return
        rows: list[int] = []
        for tokens in proposals.values():
            rows.extend(int(t) for t in tokens)
        self.ngram.prefetch(rows)

    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate] = (),
    ) -> CycleResult:
        if not batch:
            return CycleResult((), (), (), EMPTY_PROFILE)
        started = self.clock.now()

        draft_start = self.clock.now()
        if drafts:
            proposals = {int(d.sequence_id): tuple(d.tokens) for d in drafts}
            drafted = sum(len(t) for t in proposals.values())
        else:
            proposals, drafted = self._propose(batch)
        draft_ms = (self.clock.now() - draft_start) * 1000.0

        ngram_start = self.clock.now()
        self._prefetch(proposals)
        ngram_wait_ms = (self.clock.now() - ngram_start) * 1000.0

        rows = self.verify_candidates(batch, proposals)
        states = [s.state for s in batch]
        sampling = [s.request.sampling for s in batch]
        outcomes, profile = self.backend.verify(states, rows, sampling)

        detok_start = self.clock.now()
        events, finished, committed = self.commit(batch, outcomes, proposals)
        detok_ms = (self.clock.now() - detok_start) * 1000.0

        self.controller.record(profile, outcomes)
        if self.drafter is not None and hasattr(self.drafter, "observe"):
            self.drafter.observe(outcomes)

        wall_ms = (self.clock.now() - started) * 1000.0
        profile = self.finish_profile(
            profile,
            n_sequences=len(batch),
            draft_ms=draft_ms,
            detok_ms=detok_ms,
            ngram_wait_ms=ngram_wait_ms,
            wall_ms=wall_ms,
            tokens_committed=committed,
            tokens_drafted=drafted,
        )
        self.profiler.cycle(profile)
        return CycleResult(tuple(outcomes), tuple(events), tuple(finished), profile)
