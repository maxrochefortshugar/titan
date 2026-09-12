#!/usr/bin/env python3
"""oMLX patch: a RAM row cache in front of the packed PLE n-gram table.

The deployed reader (``kernels/ple-fix/patch.py``, ``PackedPLETable`` in
``rows`` mode) turns every PLE lookup into one contiguous 100-byte slice of
``layer1.rows.bin``, a 32 GB file on SSD.  That is 3x fewer pages than stock
oMLX but still ~2.3 ms of host time per decode forward and ~450 ms per
2048-token prefill chunk on cold ids (AUDIT-2026-09-12.md section A, section D
item 6).

N-gram row ids over natural text are extremely skewed: the bigram heads hit the
same rows over and over inside one document, and the 16 heads of one token
re-request rows the previous chunk already read.  So keep the hot rows in RAM.

Cache design: **16-way set-associative, exact LRU inside each set**.
Why not a global LRU list: maintaining recency order for a 20-million-entry
table means a linked list and a Python-level touch per row, which is exactly the
per-row loop the hot path cannot afford.  Why not CLOCK: CLOCK needs a hand
sweep whose length is data dependent, again a loop.  A set-associative table
does the whole probe as four numpy ops on the batch (hash, one (n, ways) gather,
one compare, one argmax), and victim selection is one argsort over the (few
hundred) sets a batch actually touches.  Recency inside a set is a real
timestamp, so behaviour equals true LRU whenever a set holds fewer than 16 live
hot rows; ``bench.py`` measures the gap against exact LRU on the real id stream
(it is under a tenth of a point).

Byte budget: ``OMLX_PLE_LRU_GB`` (default 2, hard max 4) counts everything, the
row payload and the tag and timestamp arrays, so the number an operator sets is
the number the process grows by.  Per entry that is 100 + 4 + 4 = 108 bytes.

Bit exactness: a cached row is the same 100 bytes the mmap would have returned,
handed to the same ``mx.dequantize``.  The cache changes where the bytes come
from and nothing else.  ``test_exact.py`` checks the three planes and the
dequantized bf16 output byte for byte against the uncached reader.

Miss reads, ``OMLX_PLE_LRU_READER``: ``pread`` (default) fetches the 100 bytes
of a missing row and nothing else, so neither the process nor the OS page cache
grows by the 16 KB page around it; ``nocache`` is the same with F_NOCACHE, for
cold measurements and for a machine with no spare RAM at all; ``mmap``
reproduces the deployed page-prefetch-then-gather path for A/B.

What is replaced: ``PackedPLETable.assemble_host`` (ple-fix/patch.py:178-191)
and, in pread/nocache mode, ``_prefetch_pages`` (:150-176).  Nothing in
ple-fix/patch.py is edited; the class is subclassed and the module attribute
rebound, so ``packed_call`` (:282-305) keeps calling ``self._packed.assemble_host``
and gets the cached one.

Enable with ``OMLX_PLE_LRU=1`` on top of ``OMLX_PLE_PACKED=1`` and
``OMLX_PLE_PACKED_MODE=rows``.  Call :func:`install` at import time (it
rebinds the table class the base patch instantiates) or after the model is
loaded (it also promotes tables that already exist); both are idempotent.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BASE_PATCH = Path("~/inference-server/kernels/ple-fix/patch.py").expanduser()
EMPTY = np.uint32(0xFFFFFFFF)
MIX64 = np.uint64(0x9E3779B97F4A7C15)
_PAGE = os.sysconf("SC_PAGE_SIZE")
F_NOCACHE = 48


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _env_flag("OMLX_PLE_LRU")


def budget_bytes() -> int:
    gb = float(os.environ.get("OMLX_PLE_LRU_GB", "2"))
    gb = max(0.0, min(gb, 4.0))          # hard max: 70 GB model + 100 GB guard
    return int(gb * (1 << 30))


def find_base_module():
    """Return the deployed packed reader if something already imported it.

    prod/bootstrap.py loads ple-fix/patch.py with its own
    ``spec_from_file_location``, under whatever module name it likes, so the
    only reliable handle is the file path.  Rebinding the class on a second,
    private copy of the module would silently do nothing, which is the one
    failure mode worth hunting for.
    """
    import sys
    target = BASE_PATCH.resolve()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not hasattr(module, "PackedPLETable"):
            continue
        try:
            if Path(path).resolve() == target:
                return module
        except OSError:
            continue
    return None


def load_base_module():
    """The deployed packed reader: the live copy if there is one, else ours."""
    found = find_base_module()
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("omlx_ple_packed_base", BASE_PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    sys.modules.setdefault("omlx_ple_packed_base", module)
    return module


# --------------------------------------------------------------------- cache
class SetAssocRowCache:
    """Fixed-byte-budget row cache. All batch operations are vectorised."""

    def __init__(self, capacity_bytes: int, stride: int, ways: int = 16):
        self.stride = int(stride)
        self.ways = int(ways)
        per_entry = self.stride + 8
        entries = max(self.ways, int(capacity_bytes) // per_entry)
        sets = max(1, 1 << int(math.floor(math.log2(max(1, entries // self.ways)))))
        self.n_sets = sets
        self.n_entries = sets * self.ways
        self.mask = np.uint64(sets - 1)
        self.keys = np.full((sets, self.ways), EMPTY, dtype=np.uint32)
        self.stamp = np.zeros((sets, self.ways), dtype=np.uint32)
        self.store = np.empty((self.n_entries, self.stride), dtype=np.uint8)
        self.clock = np.uint32(0)
        self.bytes_reserved = (self.store.nbytes + self.keys.nbytes
                               + self.stamp.nbytes)
        # counters
        self.lookups = 0
        self.rows_requested = 0
        self.rows_unique = 0
        self.hits = 0
        self.misses = 0
        self.inserts = 0
        self.overflow = 0
        self.live = 0

    def _sets(self, ids: np.ndarray) -> np.ndarray:
        h = (ids.astype(np.uint64) * MIX64) >> np.uint64(32)
        return (h & self.mask).astype(np.intp)

    def get(self, ids: np.ndarray):
        """ids: unique int64 row ids. Returns (sets, slots, hit_mask)."""
        sets = self._sets(ids)
        match = self.keys[sets] == ids.astype(np.uint32)[:, None]
        hit = match.any(axis=1)
        way = match.argmax(axis=1)
        slots = sets * self.ways + way
        return sets, slots, hit

    def touch(self, slots: np.ndarray) -> None:
        self.clock = np.uint32((int(self.clock) + 1) & 0xFFFFFFFF)
        self.stamp.reshape(-1)[slots] = self.clock

    def put(self, ids: np.ndarray, sets: np.ndarray, rows: np.ndarray) -> None:
        """Insert unique missing ids. Victim per set: smallest timestamp (LRU)."""
        n = ids.size
        if n == 0:
            return
        order = np.argsort(sets, kind="stable")
        s_sorted = sets[order]
        head = np.empty(n, dtype=bool)
        head[0] = True
        np.not_equal(s_sorted[1:], s_sorted[:-1], out=head[1:])
        positions = np.arange(n)
        group_start = np.maximum.accumulate(np.where(head, positions, 0))
        rank = positions - group_start
        keep = rank < self.ways
        self.overflow += int(n - keep.sum())

        uniq_sets = s_sorted[head]
        age_order = np.argsort(self.stamp[uniq_sets], axis=1, kind="stable")
        gidx = np.cumsum(head) - 1
        victim_way = age_order[gidx[keep], rank[keep]]
        target_sets = s_sorted[keep]
        slots = target_sets * self.ways + victim_way
        src = order[keep]

        flat_keys = self.keys.reshape(-1)
        self.live += int((flat_keys[slots] == EMPTY).sum())
        flat_keys[slots] = ids[src].astype(np.uint32)
        self.store[slots] = rows[src]
        self.stamp.reshape(-1)[slots] = self.clock
        self.inserts += int(slots.size)

    def clear(self):
        """Drop every entry, keep the buffers. Used to re-run a cold pass."""
        self.keys[:] = EMPTY
        self.stamp[:] = 0
        self.live = 0

    def stats(self):
        total = self.hits + self.misses
        return {
            "ways": self.ways,
            "sets": self.n_sets,
            "entries": self.n_entries,
            "bytes_reserved": self.bytes_reserved,
            "gb_reserved": self.bytes_reserved / 1e9,
            "live_entries": self.live,
            "fill": self.live / self.n_entries if self.n_entries else 0.0,
            "lookups": self.lookups,
            "rows_requested": self.rows_requested,
            "rows_unique_after_dedup": self.rows_unique,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "inserts": self.inserts,
            "evictions": max(0, self.inserts - self.live),
            "set_overflow_rows": self.overflow,
        }


# ------------------------------------------------------------- cached table
class CachedPLETable:
    """Mixin that overrides ``assemble_host`` on the deployed rows reader.

    ``install`` builds a subclass of whatever ``PackedPLETable`` the base module
    exposes, so nothing in ple-fix/patch.py is edited or copied.
    """

    # ---- set up ------------------------------------------------------
    def attach_cache(self, capacity_bytes=None, ways=16, reader=None,
                     bypass=False):
        if getattr(self, "_lru", None) is not None:
            return self._lru
        self._lru_bypass = bool(bypass)
        if self.mode != "rows":
            raise ValueError("the row cache only makes sense in rows mode")
        cap = budget_bytes() if capacity_bytes is None else int(capacity_bytes)
        self._lru = SetAssocRowCache(cap, self.stride, ways=ways)
        self._lru_lock = threading.Lock()
        self._lru_reader = reader or os.environ.get("OMLX_PLE_LRU_READER", "pread")
        if self._lru_reader not in {"pread", "nocache", "mmap"}:
            raise ValueError(f"unknown OMLX_PLE_LRU_READER {self._lru_reader!r}")
        self._lru_nocache_fd = None
        self._lru_pool = None
        self.miss_read_seconds = 0.0
        self.miss_rows_read = 0
        self.cache_copy_seconds = 0.0
        logger.info("PLE LRU: %.2f GB, %d entries, %d sets x %d ways",
                    self._lru.bytes_reserved / 1e9, self._lru.n_entries,
                    self._lru.n_sets, self._lru.ways)
        return self._lru

    # ---- miss reads --------------------------------------------------
    def _read_fd(self):
        """A private fd for the miss path. ``nocache`` keeps the OS page cache
        out of it entirely (F_NOCACHE, fcntl.h:48), which is what the cold
        measurement wants and what a machine with no spare RAM wants."""
        if self._lru_nocache_fd is None:
            import fcntl
            from concurrent.futures import ThreadPoolExecutor
            fd = os.open(str(self._rows_path), os.O_RDONLY)
            if self._lru_reader == "nocache":
                fcntl.fcntl(fd, F_NOCACHE, 1)
            self._lru_nocache_fd = fd
            self._lru_pool = ThreadPoolExecutor(max_workers=48,
                                                thread_name_prefix="ple-lru-io")
        return self._lru_nocache_fd

    def _read_rows(self, ids: np.ndarray) -> np.ndarray:
        """Fetch ``ids`` (unique, sorted) out of layer1.rows.bin.

        ``pread`` (default) and ``nocache`` read the 100 bytes of each missing
        row and nothing else, so neither the process resident set nor the OS
        page cache grows by the 16 KB page the row happens to sit in.  ``mmap``
        reproduces the deployed reader exactly (page prefetch through the
        48-worker pool, then a vectorised gather out of the mapping) and is
        there for A/B against it.
        """
        t0 = time.perf_counter()
        stride = self.stride
        if self._lru_reader == "mmap":
            if ids.size > 8:          # base reader's threshold, patch.py:180
                self._prefetch_pages(ids)
            out = np.array(self._rows_mm[ids.astype(np.intp)], copy=True)
        else:
            fd = self._read_fd()
            out = np.empty((ids.size, stride), dtype=np.uint8)
            offsets = ids.astype(np.int64) * stride

            def fetch(i):
                offset = int(offsets[i])
                data = os.pread(fd, stride, offset)
                while len(data) < stride:
                    more = os.pread(fd, stride - len(data), offset + len(data))
                    if not more:
                        break
                    data += more
                out[i] = np.frombuffer(data, dtype=np.uint8)

            if ids.size >= 4:
                chunk = max(1, min(256, int(ids.size // 48) or 1))
                list(self._lru_pool.map(fetch, range(ids.size), chunksize=chunk))
            else:
                for i in range(ids.size):
                    fetch(i)
        self.miss_read_seconds += time.perf_counter() - t0
        self.miss_rows_read += int(ids.size)
        return out

    # ---- the hot path ------------------------------------------------
    def assemble_host(self, host: np.ndarray):
        cache = self._lru
        host = np.asarray(host, dtype=np.int64).reshape(-1)
        if self._lru_bypass:
            # measurement path: the deployed algorithm, this module's reader,
            # no dedup and no cache, so a bench can price the reader alone
            full = self._read_rows(host)
            wb, sb = self.weight_bytes, self.scale_bytes
            return (full[:, :wb].copy().view("<u4"),
                    full[:, wb:wb + sb].copy().view("<u2"),
                    full[:, wb + sb:].copy().view("<u2"))
        with self._lru_lock:
            t0 = time.perf_counter()
            uniq, inverse = np.unique(host, return_inverse=True)
            sets, slots, hit = cache.get(uniq)

            raw = np.empty((uniq.size, self.stride), dtype=np.uint8)
            hit_slots = slots[hit]
            raw[hit] = cache.store[hit_slots]
            cache.touch(hit_slots)

            n_miss = int(uniq.size - hit.sum())
            if n_miss:
                miss = ~hit
                miss_ids = uniq[miss]
                rows = self._read_rows(miss_ids)
                raw[miss] = rows
                cache.put(miss_ids, sets[miss], rows)

            cache.lookups += 1
            cache.rows_requested += int(host.size)
            cache.rows_unique += int(uniq.size)
            cache.hits += int(uniq.size - n_miss)
            cache.misses += n_miss

            # np.unique returns sorted rows; inverse restores the caller's order
            full = raw[inverse]
            wb, sb = self.weight_bytes, self.scale_bytes
            planes = (full[:, :wb].copy().view("<u4"),
                      full[:, wb:wb + sb].copy().view("<u2"),
                      full[:, wb + sb:].copy().view("<u2"))
            self.cache_copy_seconds += time.perf_counter() - t0
            return planes

    # ---- static hot set ---------------------------------------------
    def preload(self, ids: np.ndarray, batch: int = 1 << 16) -> int:
        """Fault a static hot set in. Same path as a miss, no output built."""
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        cache = self._lru
        for start in range(0, ids.size, batch):
            piece = ids[start:start + batch]
            sets, _, hit = cache.get(piece)
            miss = ~hit
            if not miss.any():
                continue
            miss_ids = piece[miss]
            cache.touch(np.empty(0, dtype=np.intp))
            cache.put(miss_ids, sets[miss], self._read_rows(miss_ids))
        return cache.live

    def cache_stats(self):
        base = self._lru.stats()
        base.update({
            "miss_read_seconds": self.miss_read_seconds,
            "miss_rows_read": self.miss_rows_read,
            "miss_bytes_read": self.miss_rows_read * self.stride,
            "host_path_seconds": self.cache_copy_seconds,
            "reader": self._lru_reader,
        })
        return base

    def reset_counters(self):
        lru = self._lru
        for name in ("lookups", "rows_requested", "rows_unique", "hits",
                     "misses", "inserts", "overflow"):
            setattr(lru, name, 0)
        self.miss_read_seconds = 0.0
        self.miss_rows_read = 0
        self.cache_copy_seconds = 0.0


def make_cached_class(base_cls):
    name = "Cached" + base_cls.__name__
    cls = type(name, (CachedPLETable, base_cls), {})
    cls._omlx_ple_lru = True
    return cls


# ------------------------------------------------------------------ install
_STATE: dict = {"base": None, "cls": None, "tables": {}}


def install(model=None, model_path=None, *, force=False, capacity_bytes=None,
            ways=16, reader=None, base_module=None) -> bool:
    """Wrap the deployed packed reader with the row cache. Idempotent.

    Returns False and leaves the deployed path untouched when the env toggle is
    off, when the base patch is missing, or when the table is not in rows mode.
    Safe at import time (rebinds the class the base patch instantiates) and
    after model load (promotes tables that already exist).
    """
    if not force and not enabled():
        return False
    base = base_module or _STATE["base"] or find_base_module()
    if base is None:
        try:
            base = load_base_module()
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("PLE LRU: cannot load %s (%s); staying on the "
                           "uncached reader", BASE_PATCH, exc)
            return False
        _STATE["base"] = base

    if _STATE["cls"] is None:
        original = getattr(base, "PackedPLETable", None)
        if original is None:
            logger.warning("PLE LRU: base patch has no PackedPLETable")
            return False
        _STATE["cls"] = make_cached_class(original)
        _STATE["original_cls"] = original
    cached_cls = _STATE["cls"]

    # 1. future tables the base patch builds come out cached
    if getattr(base.PackedPLETable, "_omlx_ple_lru", False) is not True:
        base.PackedPLETable = cached_cls

    # 2. the base patch itself, if a live model was handed to us
    if model is not None:
        base.apply_ple_packed_patch(model, model_path)

    # 3. promote tables that already exist (created before this call)
    promoted = 0
    for index, table in base.packed_tables().items():
        if getattr(table, "mode", None) != "rows":
            continue
        if not isinstance(table, CachedPLETable):
            table.__class__ = cached_cls
            table._lru = None
        table.attach_cache(capacity_bytes=capacity_bytes, ways=ways, reader=reader)
        _STATE["tables"][index] = table
        promoted += 1
    if promoted:
        logger.info("PLE LRU: cache attached to %d packed table(s)", promoted)
    return True


def uninstall() -> bool:
    """Put the class binding back and drop the cache buffers."""
    base = _STATE["base"]
    if base is None:
        return False
    if _STATE.get("original_cls") is not None:
        base.PackedPLETable = _STATE["original_cls"]
    for table in list(_STATE["tables"].values()):
        lru = getattr(table, "_lru", None)
        table._lru = None
        del lru
        if getattr(table, "_lru_nocache_fd", None) is not None:
            try:
                os.close(table._lru_nocache_fd)
            except OSError:
                pass
            table._lru_nocache_fd = None
        table.__class__ = _STATE["original_cls"]
    _STATE["tables"].clear()
    return True


def tables():
    return dict(_STATE["tables"])


if __name__ == "__main__":
    import json
    base = load_base_module()
    print(json.dumps({
        "base_patch": str(BASE_PATCH),
        "budget_gb": budget_bytes() / (1 << 30),
        "enabled": enabled(),
        "base_has_PackedPLETable": hasattr(base, "PackedPLETable"),
    }, indent=1))
