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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from titan.core.types import (
    CycleProfile,
    DraftCandidate,
    SequenceId,
    SequenceState,
    TokenEvent,
    VerifyOutcome,
)

__all__ = ["CycleResult", "DecodeCycle", "AcceptancePolicy"]


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


class DecodeCycle(Protocol):
    def run(
        self,
        batch: Sequence[SequenceState],
        drafts: Sequence[DraftCandidate],
    ) -> CycleResult:
        """Advance ``batch`` by one cycle.

        Preconditions: every sequence is DECODING, holds an open state handle,
        and its state length equals ``prompt_len + committed``.

        Postconditions: for each sequence, state length grew by exactly
        ``outcome.n_committed``; ``committed`` grew by the same; the sequence is
        finished if a stop condition matched inside the accepted run, in which
        case tokens after the stop are discarded and the state is truncated to
        the stop position. Discarding them matters beyond tidiness: a copy or
        draft block accepted past a stop token corrupts the recurrent state that
        the prefix cache is about to store.
        """
