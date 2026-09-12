"""Prefix cache policy: two grids, and the rules that keep them consistent.

Block size 512, snapshot grid 2048, plus a snapshot at every prompt end.

The three rules that make it work, each of which the overlay learned the hard
way:

1. Every snapshot-grid multiple that falls strictly inside a prefill suffix must
   itself end a chunk. Round 4a clamped straight to a fine target, stepped over
   26624 without landing on it, so nothing staged a snapshot there and the store
   chain truncated at the previous grid point.
2. Emission is restricted to the grid plus the cuts the planner itself chose.
   Otherwise a 512-token block size emits a 110 MiB snapshot at every chunk end
   whenever the scheduler shortens a chunk, four times the stock snapshot rate
   on the contended path.
3. A boundary whose snapshot did not commit is dropped from the chain rather
   than recorded. A lookup must never return a length it cannot restore.

Cost model, fitted on measured turns and kept in the tests as a regression
check: a turn costs ``K + r*cached + sum over chunks of (F + T*(a + b*ctx)) + S``
per snapshot, with K = 53 ms, r = 2.77e-3 ms per cached token, F = 73 ms per
chunk, a = 0.4488 ms per token, b = 7.63e-6, and S about 210 ms per snapshot on
a quiet writer. Fine boundaries win while S stays under about 400 ms, which is
why the store reports its backlog and the planner reads it.
"""

from __future__ import annotations

__all__: list[str] = []
