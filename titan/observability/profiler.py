"""Profiler construction and the shapes it reports."""

from __future__ import annotations

from dataclasses import dataclass

from titan.config.schema import ObservabilityConfig
from titan.core.ports import Clock, Profiler

__all__ = ["DecodeSummary", "build_profiler"]


@dataclass(frozen=True, slots=True)
class DecodeSummary:
    """Aggregate over a window of cycles, as reported at ``GET /metrics`` and
    written by the bench harness. The fields are chosen so a regression can be
    attributed without a rerun: a decode slowdown is either fewer rows, worse
    acceptance, more host syncs, or n-gram wait."""

    cycles: int
    tokens_per_second: float
    mean_accepted_per_cycle: float
    acceptance_rate: float
    mean_rows: float
    mean_verify_ms: float
    mean_draft_ms: float
    mean_ngram_wait_ms: float
    host_syncs_per_cycle: float
    kernel_fallbacks: int


def build_profiler(config: ObservabilityConfig, clock: Clock) -> Profiler:
    raise NotImplementedError
