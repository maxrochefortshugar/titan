# SPDX-License-Identifier: MIT
"""``NgramReader`` port: the packed n-gram table, streamed from SSD.

The table is 51B parameters in one row-contiguous file. The reader over it lives
in :mod:`titan.kernels.ple_packed_lookup`, which owns the manifest, the memory
map and the bit-exact dequantise; this module is the adapter that turns it into
the port the decode cycle uses, and it adds exactly one thing: prefetch that
returns immediately.

That is the whole point of the port. Reading a row on the critical path cost
about 2.3 ms per forward in the overlay, roughly 9% of a decode step, and it is
host and SSD work with no reason to be there. The kernel's own ``prefetch``
waits for its worker pool because the kernel is called synchronously. Here the
page touches are submitted and the futures are kept, so :meth:`prefetch` hands
back control while the SSD works and :meth:`gather` waits only for pages that
are still in flight. Call ``prefetch`` for the rows the next verify forward will
need, then do the drafting work, then ``gather``.

**There is no resident mode and there will not be one.** Faulting the planes
into device arrays panicked the kernel on the 128 GB machine. Every option that
would do it is refused at open time rather than guarded at read time, because a
guard that can be flipped is an option. The only mapping is the read-only
memory map of the rows file, and only requested rows are ever copied out of it.

Ownership: one reader owns one memory map, one file descriptor for the prefetch
preads and one thread pool. Close it, or use it as a context manager.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from titan.core.errors import ConfigError
from titan.kernels.ple_packed_lookup import (
    PAGE,
    LayerEntry,
    PackedRowTable,
    ReaderPool,
    load_manifest,
    reference,
)

__all__ = [
    "PackedNgramReader",
    "open_packed_table",
    "packed_table",
    "DEFAULT_WORKERS",
    "PAGES_PER_TASK",
]

DEFAULT_WORKERS = 32
"""Scattered single-page preads are filesystem-overhead bound: this workload
measured 1.14 GB/s against 13.6 GB/s sequential, so filling the SSD queue with
parallel readers is worth several times serialised page faults."""

PAGES_PER_TASK = 64
"""Pages per pool task. One task per page turns a 2048-token chunk into
thousands of submissions and the scheduling starts to cost more than the read."""


def _refuse_resident(**options: Any) -> None:
    """Reject anything that would ask for a RAM-resident table.

    Named and separate so the refusal is greppable. The overlay's resident mode
    read the three quantised planes into device arrays; on 128 GB that panicked
    the kernel, and the fix is not a smaller resident set, it is no resident
    set.
    """
    for name, value in options.items():
        if value:
            raise ConfigError(
                f"{name} is not supported: the packed n-gram table is streamed "
                "from SSD, never made resident. A resident copy of this table "
                "panicked the kernel on a 128 GB machine."
            )


class PackedNgramReader:
    """One packed layer, with non-blocking prefetch in front of it."""

    def __init__(
        self,
        table: PackedRowTable,
        *,
        workers: int = DEFAULT_WORKERS,
        pages_per_task: int = PAGES_PER_TASK,
    ) -> None:
        self._table = table
        self._pool = ReaderPool(workers=workers, name="ngram-io")
        self._pages_per_task = max(1, int(pages_per_task))
        self._fd = os.open(str(table.path), os.O_RDONLY)
        self._stride = table.entry.row_stride
        pages = 1 + (table.path.stat().st_size - 1) // PAGE
        self._done = np.zeros(pages, dtype=np.uint8)
        self._inflight: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._stats = {
            "rows_requested": 0.0,
            "rows_hit": 0.0,
            "rows_missed": 0.0,
            "pages_read": 0.0,
            "bytes_read": 0.0,
            "prefetch_calls": 0.0,
            "gather_calls": 0.0,
            "wait_ms": 0.0,
        }

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close(wait=True)
        self._table.close()
        try:
            os.close(self._fd)
        except OSError:  # pragma: no cover - already closed
            pass

    def __enter__(self) -> "PackedNgramReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- shape -------------------------------------------------------------
    @property
    def entry(self) -> LayerEntry:
        return self._table.entry

    @property
    def rows(self) -> int:
        return self._table.entry.rows

    @property
    def dims(self) -> int:
        return self._table.entry.dims

    @property
    def table(self) -> PackedRowTable:
        """The kernel-level table. The registry op takes this, not the reader."""
        return self._table

    # -- port --------------------------------------------------------------
    def prefetch(self, rows: Iterable[int]) -> None:
        """Queue the pages ``rows`` will need. Returns without waiting.

        Duplicate ids are cheap: a page already read, or already queued by an
        earlier call, is skipped under the lock before anything is submitted.
        """
        ids = self._host_rows(rows)
        if ids.size == 0:
            return
        self._stats["prefetch_calls"] += 1
        fresh = self._fresh_pages(self._pages_for(ids))
        if fresh.size == 0:
            return
        batches = [
            fresh[i : i + self._pages_per_task].tolist()
            for i in range(0, fresh.size, self._pages_per_task)
        ]
        for batch in batches:
            future = self._pool.submit(self._read_pages, batch)
            with self._lock:
                for page in batch:
                    self._inflight[page] = future

    def gather(self, rows: Sequence[int]) -> Any:
        """The rows as a device array, blocking only on pages still in flight.

        Bit-identical to the kernel's own reference path by calling it: the same
        packed nibbles, scales and biases go into the same ``mx.dequantize``.
        """
        ids = self._host_rows(rows)
        self._stats["gather_calls"] += 1
        self._stats["rows_requested"] += float(ids.size)
        if ids.size:
            self._wait_for(self._pages_for(ids), n_rows=int(ids.size))
        return reference(self._table, ids)

    def stats(self) -> Mapping[str, float]:
        out = dict(self._stats)
        gathers = out["gather_calls"] or 1.0
        out["mean_wait_ms"] = out["wait_ms"] / gathers
        rows = out["rows_requested"] or 1.0
        out["hit_rate"] = out["rows_hit"] / rows
        return out

    # -- internals ---------------------------------------------------------
    def _host_rows(self, rows: Iterable[int]) -> np.ndarray:
        """Row ids on the host, bounds-checked. Offsets are host work."""
        if hasattr(rows, "dtype") and not isinstance(rows, np.ndarray):
            # an mx.array: evaluating it is unavoidable, the preads need ints
            import mlx.core as mx

            if isinstance(rows, mx.array):
                mx.eval(rows)
                rows = np.array(rows.reshape(-1), copy=False)
        ids = np.asarray(rows, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self._table.entry.rows):
            raise IndexError(
                f"n-gram row id outside [0, {self._table.entry.rows})"
            )
        return ids

    def _pages_for(self, ids: np.ndarray) -> np.ndarray:
        offsets = ids * self._stride
        return np.unique(
            np.concatenate((offsets // PAGE, (offsets + self._stride - 1) // PAGE))
        )

    def _fresh_pages(self, pages: np.ndarray) -> np.ndarray:
        with self._lock:
            candidates = pages[self._done[pages] == 0]
            if candidates.size == 0:
                return candidates
            inflight = self._inflight
            return np.array(
                [p for p in candidates.tolist() if p not in inflight], dtype=np.int64
            )

    def _read_pages(self, pages: list[int]) -> None:
        """Touch one batch of pages so the mapping finds them warm."""
        read = 0
        for page in pages:
            offset = int(page) * PAGE
            remaining = PAGE
            while remaining > 0:
                chunk = os.pread(self._fd, remaining, offset + (PAGE - remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            read += PAGE - remaining
        with self._lock:
            for page in pages:
                self._done[page] = 1
                self._inflight.pop(page, None)
            self._stats["pages_read"] += float(len(pages))
            self._stats["bytes_read"] += float(read)

    def _wait_for(self, pages: np.ndarray, *, n_rows: int) -> None:
        """Block until every page of the request has been read, if it can be.

        A hit is a request the caller prefetched, whether or not the read has
        landed yet: that is the distinction the decode cycle can act on. A miss
        means at least one page was never queued, so the memory map will fault
        it in during assemble, on the critical path, which is what
        :meth:`prefetch` exists to avoid.
        """
        with self._lock:
            outstanding = {
                self._inflight[p] for p in pages.tolist() if p in self._inflight
            }
            cold = int(
                sum(
                    1
                    for p in pages.tolist()
                    if not self._done[p] and p not in self._inflight
                )
            )
        if cold:
            self._stats["rows_missed"] += float(n_rows)
        else:
            self._stats["rows_hit"] += float(n_rows)
        if not outstanding:
            return
        started = time.perf_counter()
        for future in outstanding:
            future.result()
        self._stats["wait_ms"] += (time.perf_counter() - started) * 1e3


def _select_layer(
    layers: Mapping[int, LayerEntry], layer: int | None
) -> tuple[int, LayerEntry]:
    if not layers:
        raise ConfigError("packed n-gram manifest lists no layers")
    if layer is None:
        if len(layers) > 1:
            raise ConfigError(
                "packed n-gram manifest holds "
                f"{sorted(layers)}; pass layer= to pick one"
            )
        only = next(iter(layers))
        return only, layers[only]
    if layer not in layers:
        raise ConfigError(f"no packed n-gram layer {layer}; have {sorted(layers)}")
    return layer, layers[layer]


def open_packed_table(
    directory: str | Path,
    *,
    layer: int | None = None,
    workers: int = DEFAULT_WORKERS,
    pages_per_task: int = PAGES_PER_TASK,
    resident: bool = False,
    preload: bool = False,
    mode: str = "mmap",
) -> PackedNgramReader:
    """Open a ``ple-packed`` directory for streaming reads.

    ``resident`` and ``preload`` exist only to be refused, and ``mode`` accepts
    nothing but ``mmap``. See :func:`_refuse_resident`.
    """
    _refuse_resident(resident=resident, preload=preload)
    if mode != "mmap":
        raise ConfigError(f"unsupported n-gram table mode {mode!r}; only 'mmap'")
    path, layers = load_manifest(directory)
    _, entry = _select_layer(layers, layer)
    return PackedNgramReader(
        PackedRowTable(path, entry), workers=workers, pages_per_task=pages_per_task
    )


def packed_table(prefix: str, model_path: str | Path) -> PackedRowTable | None:
    """The kernel-level table for a layer prefix, or ``None`` if none is built.

    ``titan.adapters.mlx.kernels.ple_table`` calls this and hands the result
    straight to the ``ple.packed_rows_lookup`` op, so it returns the kernel's
    :class:`PackedRowTable` and not a reader. The reader is for the decode
    cycle, which wants prefetch; the op is called synchronously from inside the
    model and wants the mapping.
    """
    from titan.adapters.mlx.checkpoint import ple_manifest

    manifest = ple_manifest(Path(model_path))
    if not manifest:
        return None
    directory = Path(model_path) / "ple-packed"
    for entry in (manifest.get("layers") or {}).values():
        entry_prefix = entry.get("prefix")
        if entry_prefix and prefix.endswith(str(entry_prefix).split(".", 2)[-1]):
            return PackedRowTable(directory, LayerEntry.from_manifest(entry))
    return None
