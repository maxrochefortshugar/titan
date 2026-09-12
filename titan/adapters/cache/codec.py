"""The one seam between the cache and device memory.

The cache is a bytes business. It hashes token ids, moves byte strings between
RAM and SSD, and counts things. It never sees an mlx array, which is what lets
its whole test suite run on numpy in milliseconds and what keeps the writer
thread away from the GPU.

Turning a slice of a sequence's state into bytes is the model adapter's job,
and :class:`StateCodec` is the four methods the cache needs from it. The
implementation for Qwen3.8-Flash-Next wraps
:class:`titan.adapters.mlx.state.ModelState`; the tests substitute a numpy
double.

Two rules bind every implementation, both learned from the overlay:

1. Every method here runs on the caller's thread, which is the scheduler
   thread, and every one of them may touch mlx. The cache calls them before it
   hands bytes to the writer, so no mlx call ever happens on a background
   thread. Serialising on a worker thread is a Metal command-buffer race, not a
   performance choice.
2. Import must ``mx.eval`` what it builds. Arrays constructed from a buffer
   that came off disk stay attached to that buffer until they are evaluated,
   and a state holding file-backed arrays across an eviction is a use after
   free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from titan.adapters.cache.format import CacheSignature

__all__ = ["StateCodec"]


@runtime_checkable
class StateCodec(Protocol):
    """Serialise and restore parts of one sequence's backend state.

    ``state`` is whatever the backend's handle resolves to. The cache treats it
    as opaque and passes it straight back.
    """

    @property
    def signature(self) -> CacheSignature:
        """Model name, layer layout, block size and snapshot dtype.

        Hashed into every block name and written into every file header. If
        this changes, every stored byte in the tier becomes unreachable, which
        is the intent: a cache from another build is not a cache.
        """

    def export_blocks(self, state: object, start: int, end: int) -> bytes:
        """Attention KV for the half-open token range ``[start, end)``.

        Called on the scheduler thread after a prompt finishes prefilling.
        Recurrent layers are not part of this: they cannot be sliced and they
        travel as snapshots instead.
        """

    def import_blocks(self, state: object, start: int, end: int, payload: bytes) -> None:
        """Append the KV for ``[start, end)`` to ``state``.

        Blocks arrive in ascending order and each one starts where the previous
        ended, so an implementation may append rather than scatter. Must
        ``mx.eval`` before returning.
        """

    def export_snapshot(self, state: object, length: int) -> bytes:
        """Serialise the recurrent snapshot staged at ``length`` tokens.

        Raises if nothing was staged there. The cache treats that as a boundary
        that did not commit and drops it from the chain rather than recording a
        length it could not restore.
        """

    def import_snapshot(self, state: object, length: int, payload: bytes) -> None:
        """Restore recurrent state so ``state`` is consistent with ``length``
        tokens. The KV blocks are already in place. Must ``mx.eval``."""
