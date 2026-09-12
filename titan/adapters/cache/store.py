"""Two-tier block and snapshot store.

Hot RAM tier at 4 GB, measured to hold the same hit rate as 16 GB with far less
memory pressure, in front of an SSD tier. Writes go through a background thread;
the scheduler thread never waits longer than ``cache.max_stall_ms``.

The stall bound is a lesson, not a preference. The overlay's writer could hold
the inference thread for up to two seconds waiting on a pending-bytes budget,
and that turned a follow-up turn immediately after a large store into a 2.69 s
turn where the model itself needed 1.7. Under backpressure Titan drops the
optional fine snapshot instead of waiting for the queue.
"""

from __future__ import annotations

__all__: list[str] = []
