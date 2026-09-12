"""Packed n-gram table reader.

Layout: one contiguous row per n-gram entry, so a lookup is one pread instead of
three. Measured against the stock three-tensor layout, a 2048-token prefill
chunk went from 93,046 page reads and about 963 ms to 31,855 and 456 ms, and
decode gained 18 to 28% because the lookup runs per decode token too.

Scheduling is the part that belongs to Titan rather than to the overlay. The
decode cycle prefetches rows for the draft candidates before it issues the
verify forward, so the SSD read overlaps GPU work. A blocking read on the
critical path costs about 2.3 ms per forward, roughly 9% of a decode step.

A resident copy of this table is banned on a 128 GB machine. It is 32 GB on top
of a 78 GB model, and the attempt produced two UI freezes and a kernel panic.
"""

from __future__ import annotations

__all__: list[str] = []
