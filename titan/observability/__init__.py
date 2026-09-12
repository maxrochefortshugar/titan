"""Observability. Built in, not patched on.

Three layers, each cheaper than the one below it:

1. **Counters and the cycle profile.** Always on. Every decode cycle emits one
   :class:`~titan.core.types.CycleProfile` into a ring buffer, with no extra
   host sync, because every field is either host wall time or a counter the
   cycle already had. ``GET /metrics`` reads the ring.
2. **Structured events.** Admission, prefix hit or miss with lengths, chunk
   boundaries, snapshot writes and their backlog, guard refusals, kernel
   fallbacks. One line each, machine readable, no sampling.
3. **Sampled traces.** Off by default. Every Nth cycle, a detailed per-stage
   breakdown that does sync the device. This is the arm that answers "where did
   this cycle go", and it is the only one allowed to cost anything.

The rule that keeps layer 1 honest: if a number needs a device readback, it
belongs in layer 3.
"""

from titan.observability import profiler

__all__ = ["profiler"]
