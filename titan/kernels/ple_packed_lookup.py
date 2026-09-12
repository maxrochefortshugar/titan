# SPDX-License-Identifier: MIT
"""Packed n-gram table reader: rows layout, pooled page reads, device dequantise.

Ported from ``engine/patches/ple-fix/patch.py`` (the ``rows`` mode only) and
``engine/patches/round3/small-items/ple_workers.py``.

The n-gram table is 51B parameters on SSD. Gathering each row from three
separate quantised tensors preads whole 16 KB pages out of each: about 93,000
page reads and 1.5 GB of SSD traffic for the 3.3 MB a 2048-token chunk actually
needs. The repack tool writes a row-contiguous copy, so one row is one
contiguous slice of one file: the page count drops to 31,855, exactly the
predicted 3x, and the bytes drop with it.

Two things are read from the pack: a ``manifest.json`` describing each layer
(row count, dims, bits, group size, the byte split between weights, scales and
biases, the row stride, the shard offsets the live embedding must match), and
``rows.bin``, ``rows`` x ``row_stride`` bytes of packed rows.

**There is no resident mode.** The overlay had one, faulting the three 32 GB
planes into device arrays; on a 128 GB machine that panicked the kernel, and it
is not ported. This module streams from SSD, always.

Exactness: bit-identical. The same packed nibbles, scales and biases go into
the same ``mx.dequantize(..., mode="affine")``. The accelerated path differs
from the reference only in *when* the pages are read, not in what is computed:
it touches the pages a row set will need through a caller-owned worker pool
before the mapping is indexed, so the scattered single-page preads overlap.
Worker count is the only knob over the read path and it matters: the same
workload measured 1.14 GB/s against 13.6 GB/s sequential on this SSD, and 32
parallel readers are worth several times serialised page faults.

State: the memory map, the seen-page bitmap and the thread pool live in
:class:`PackedRowTable` and :class:`ReaderPool`, both constructed and owned by
the caller. Nothing is cached at module level, and the table is a context
manager so the file descriptor has an owner.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import numpy as np

from titan.kernels.registry import KernelOp, ShapeClass

__all__ = ["LayerEntry", "OP", "PackedRowTable", "ReaderPool", "key",
           "load_manifest", "metal", "reference", "supports"]

PAGE = os.sysconf("SC_PAGE_SIZE")
DEFAULT_WORKERS = 16


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerEntry:
    """One layer's slice of the pack manifest."""

    rows: int
    dims: int
    bits: int
    group_size: int
    weight_bytes: int
    scale_bytes: int
    row_stride: int
    shard_offsets: tuple[int, ...]
    rows_file: str

    @classmethod
    def from_manifest(cls, entry: dict) -> "LayerEntry":
        layouts = entry["layouts"]
        if "rows" not in layouts:
            raise ValueError("packed layer has no rows layout; re-run the repack")
        return cls(
            rows=int(entry["rows"]),
            dims=int(entry["dims"]),
            bits=int(entry["bits"]),
            group_size=int(entry["group_size"]),
            weight_bytes=int(entry["weight_bytes"]),
            scale_bytes=int(entry["scale_bytes"]),
            row_stride=int(entry["row_stride"]),
            shard_offsets=tuple(entry["shard_offsets"]),
            rows_file=layouts["rows"]["file"],
        )


def load_manifest(directory) -> tuple[Path, dict[int, LayerEntry]]:
    """Read ``manifest.json`` and check every rows file it names is present.

    Only the rows layout is read. A pack that also carries planar files (the
    resident mode's layout) is fine; those files are ignored.
    """
    directory = Path(directory).expanduser()
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no packed n-gram table at {directory}")
    raw = json.loads(path.read_text())
    layers: dict[int, LayerEntry] = {}
    for name, entry in raw["layers"].items():
        layer = LayerEntry.from_manifest(entry)
        if not (directory / layer.rows_file).is_file():
            raise FileNotFoundError(f"packed rows file missing: {layer.rows_file}")
        layers[int(name)] = layer
    return directory, layers


# ---------------------------------------------------------------------------
# caller-owned reader pool
# ---------------------------------------------------------------------------


class ReaderPool:
    """A pool of page-touching threads. The caller constructs and closes it.

    The count is the only knob over the read path, so it is a constructor
    argument and not a module constant: a scattered single-page pread workload
    is dominated by filesystem overhead, and parallel readers are how the SSD's
    queue gets filled.
    """

    def __init__(self, workers: int = DEFAULT_WORKERS, name: str = "ple-io") -> None:
        self.workers = max(1, min(int(workers), 512))
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix=name
        )

    def map(self, fn, items: Iterable[Any]) -> None:
        """Run ``fn`` over ``items`` and wait. Exceptions propagate."""
        list(self._pool.map(fn, items))

    def submit(self, fn, *args) -> Future:
        return self._pool.submit(fn, *args)

    def close(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "ReaderPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# caller-owned table
# ---------------------------------------------------------------------------


class PackedRowTable:
    """One layer of the pack, streamed from SSD in the row-contiguous layout."""

    def __init__(self, directory, entry: LayerEntry) -> None:
        self.entry = entry
        self.path = Path(directory) / entry.rows_file
        self._file = self.path.open("rb")
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r").reshape(
            entry.rows, entry.row_stride
        )
        self._seen = bytearray(1 + (self.path.stat().st_size - 1) // PAGE)
        self.pages_read = 0

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._map = None
        try:
            self._file.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

    def __enter__(self) -> "PackedRowTable":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reads -------------------------------------------------------------
    def prefetch(self, rows: np.ndarray, pool: ReaderPool) -> int:
        """Touch every page the row set spans that has not been touched yet.

        One page per row now, rather than one per row per tensor. Returns the
        number of pages actually read.
        """
        offsets = rows.astype(np.int64) * self.entry.row_stride
        pages = np.unique(np.concatenate(
            (offsets // PAGE, (offsets + self.entry.row_stride - 1) // PAGE)
        ))
        seen = np.frombuffer(self._seen, dtype=np.uint8)
        fresh = pages[seen[pages] == 0]
        if fresh.size == 0:
            return 0
        fd = self._file.fileno()

        def touch(page: int) -> None:
            offset = int(page) * PAGE
            remaining = PAGE
            while remaining > 0:
                chunk = os.pread(fd, remaining, offset + (PAGE - remaining))
                if not chunk:
                    break
                remaining -= len(chunk)

        pool.map(touch, (int(p) for p in fresh.tolist()))
        for page in fresh.tolist():
            self._seen[page] = 1
        self.pages_read += int(fresh.size)
        return int(fresh.size)

    def assemble(self, rows: np.ndarray):
        """Copy the requested rows out of the mapping as three packed planes."""
        raw = np.array(self._map[np.asarray(rows, dtype=np.intp)], copy=True)
        wb, sb = self.entry.weight_bytes, self.entry.scale_bytes
        return (raw[:, :wb].copy().view("<u4"),
                raw[:, wb:wb + sb].copy().view("<u2"),
                raw[:, wb + sb:].copy().view("<u2"))

    def dequantize(self, planes) -> mx.array:
        """Upload the packed planes and dequantise on the device."""
        w, s, b = planes
        return mx.dequantize(
            mx.array(w),
            mx.array(s).view(mx.bfloat16),
            mx.array(b).view(mx.bfloat16),
            group_size=self.entry.group_size,
            bits=self.entry.bits,
            mode="affine",
        )


def _host_rows(indices) -> np.ndarray:
    """Row ids on the host. Syncs the device when given an mx.array, which is
    unavoidable: the pread offsets are host work."""
    if isinstance(indices, mx.array):
        mx.eval(indices)
        return np.array(indices.reshape(-1), copy=False).astype(np.int64)
    return np.asarray(indices, dtype=np.int64).reshape(-1)


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(table: PackedRowTable, indices, pool: ReaderPool | None = None):
    """Row lookup with no page prefetch: index the mapping and let the kernel
    fault each page in on demand. Returns dequantised bf16 rows [N, dims]."""
    rows = _host_rows(indices)
    if rows.size == 0:
        return mx.zeros((0, table.entry.dims), dtype=mx.bfloat16)
    return table.dequantize(table.assemble(rows))


def metal(table: PackedRowTable, indices, pool: ReaderPool | None = None):
    """Same lookup, with the pages touched through the caller's worker pool
    first. Identical signature and bit-identical output to :func:`reference`.

    Named ``metal`` for consistency with the rest of the library, though the
    acceleration is host I/O; the dequantise is on the device in both.
    """
    rows = _host_rows(indices)
    if rows.size == 0:
        return mx.zeros((0, table.entry.dims), dtype=mx.bfloat16)
    if pool is not None and rows.size > 8:
        table.prefetch(rows, pool)
    return table.dequantize(table.assemble(rows))


def key(table: PackedRowTable, indices, pool: ReaderPool | None = None) -> ShapeClass:
    n = int(indices.size) if hasattr(indices, "size") else len(indices)
    return ShapeClass(
        shapes=((n,),),
        dtypes=(mx.uint32,),
        device="cpu",   # the read is host work; the dequantise is not selected on
        extra=(table.entry.dims, table.entry.bits, table.entry.group_size,
               pool is not None),
    )


def supports(k: ShapeClass) -> bool:
    # the pooled path needs a pool and enough rows to be worth a round of
    # scheduling; below that the prefetch costs more than the faults it avoids
    return bool(k.extra) and bool(k.extra[3]) and k.shapes[0][0] > 8


OP = KernelOp(
    name="ple_packed_lookup",
    aliases=("ngram_gather",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=0.0,
    shapes=(
        {"rows": 16, "dims": 160},
        {"rows": 512, "dims": 160},
        {"rows": 31855, "dims": 160},
    ),
    exactness="bit-identical",
    source="engine/patches/ple-fix/REPORT.md section 4.1, "
           "engine/patches/round3/small-items/REPORT.md item 2",
)
