"""Admission: from an API-shaped request to a scheduled sequence.

Admission is where a request stops being text and becomes work:

    render (TemplateRenderer) -> encode (Tokenizer) -> lookup (PrefixCache)
    -> guard check -> open_state (ModelBackend) -> restore -> plan_chunks

It runs on the scheduler thread, in bounded batches, because it calls
``open_state`` and touches the guard. Rendering and encoding are pure and could
move off-thread later; the ordering constraint is only that the guard decision
and the state allocation happen in the same turn, so two requests cannot both
pass a guard that only one of them fits under.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from titan.core.types import PrefixMatch, Request, SequenceState

__all__ = ["AdmissionPlan", "Admitter"]


@dataclass(frozen=True, slots=True)
class AdmissionPlan:
    """The work a newly admitted sequence implies. Produced before any device
    allocation, so it can be logged, tested and refused cheaply."""

    request: Request
    match: PrefixMatch
    chunk_ends: tuple[int, ...]
    """Ascending prefill chunk end positions, last one at ``len(prompt)``."""
    snapshot_at: tuple[int, ...]
    """Positions where a recurrent snapshot is staged. Always includes the
    prompt end: the final chunk stops there anyway, so that snapshot costs one
    write and no forward pass."""
    estimated_gb: float
    """KV plus recurrent state at the expected final length. Feeds the guard."""


class Admitter(Protocol):
    def plan(self, request: Request) -> AdmissionPlan: ...

    def start(self, plan: AdmissionPlan) -> SequenceState:
        """Allocate state, restore the cached prefix, return a PREFILLING
        sequence. Raises :class:`~titan.core.errors.MemoryGuardError` if the
        guard refuses, having allocated nothing."""
