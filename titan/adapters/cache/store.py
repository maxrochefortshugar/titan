"""Two-tier block and snapshot store.

Hot RAM tier at 4 GB, measured to hold the same hit rate as 16 GB with far less
memory pressure, in front of an SSD tier. Writes go through a background
thread; the scheduler thread never waits longer than ``cache.max_stall_ms``.

The stall bound is a lesson, not a preference. The overlay's writer could hold
the inference thread for up to two seconds waiting on a pending-bytes budget,
and that turned a follow-up turn immediately after a large store into a 2.69 s
turn where the model itself needed 1.7. Under backpressure Titan drops the
optional fine snapshot instead of waiting for the queue.

How the two tiers divide the work:

* The hot tier is an ordered dict of byte strings under a byte budget, evicted
  least recently used first. Pinned entries are skipped: a block a running
  request is about to restore from is not a candidate no matter how old it is.
  Accounting is in bytes because entries differ by three orders of magnitude, a
  512-token KV block against a 110 MiB recurrent snapshot, and counting entries
  would let twelve snapshots masquerade as a small cache.
* The SSD tier is one file per record under ``<dir>/<model>/<xx>/<key>``, a
  256-way fan-out so no directory holds a million entries. Writes are
  write-behind on a single daemon thread: the caller hands over bytes and
  returns, the thread writes a temporary file next to the target and renames it
  into place, so a reader never sees a half-written record. The tier keeps a
  size-capped LRU index and deletes oldest-first once the configured capacity
  is passed.

Three properties the rest of the engine relies on:

* Nothing here imports mlx. Payloads arrive as bytes, already serialised by the
  caller on the scheduler thread, so the writer thread never touches the GPU.
* Nothing here raises into the engine. A missing file, a corrupt record, a
  record from another build, a full disk: every one of them returns ``None`` or
  drops a write, counts itself, and lets the caller recompute.
* ``pending_bytes`` is honest and cheap to read, because the chunk planner
  reads it every turn to decide whether one more snapshot is affordable.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from titan.core.types import BlockHash

from titan.adapters.cache.format import (
    CacheSignature,
    RecordError,
    decode_record,
    encode_record,
)

__all__ = ["StoreStats", "TwoTierStateStore"]

_BLOCK_SUFFIX = ".tkv"
_SNAPSHOT_SUFFIX = ".tsnap"
_TEMP_PREFIX = ".writing-"


def _key_for_block(key: BlockHash) -> str:
    return "b:" + bytes(key).hex()


def _key_for_snapshot(snapshot_id: str) -> str:
    return "s:" + snapshot_id


@dataclass
class StoreStats:
    """Counters. Read by ``/metrics`` and by the tests, never branched on."""

    hot_hits: int = 0
    ssd_hits: int = 0
    misses: int = 0
    puts: int = 0
    duplicate_puts: int = 0
    """A put of a key the tier already holds. Two requests sharing a prefix
    take this path, and it is why sharing does not double-store."""
    bytes_written: int = 0
    writes_dropped: int = 0
    """Writes abandoned because the queue was full for longer than the stall
    bound. The record stays in RAM; only its durability is lost."""
    write_errors: int = 0
    hot_evictions: int = 0
    ssd_evictions: int = 0
    corrupt_skipped: int = 0
    signature_rejected: int = 0
    max_stall_s: float = 0.0

    def as_mapping(self) -> dict[str, float]:
        return {f"store.{name}": float(value) for name, value in vars(self).items()}


@dataclass
class _HotEntry:
    payload: bytes
    kind: str
    tokens: int
    durable: bool = False


@dataclass
class _DiskEntry:
    path: str
    size: int
    last_access: float


@dataclass
class _PendingWrite:
    path: str
    record: bytes
    cache_key: str
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.record)


class TwoTierStateStore:
    """A :class:`titan.core.ports.KVStateStore` over RAM and one directory.

    ``ssd_dir`` may be empty, in which case the store is RAM only and every
    write is a no-op past the hot tier. That is the configuration the engine
    tests use and a legitimate production choice on a machine whose disk is
    busy with the n-gram table.
    """

    def __init__(
        self,
        signature: CacheSignature,
        *,
        ssd_dir: str = "",
        hot_budget_bytes: int = 4 * 1024**3,
        ssd_capacity_bytes: int = 200 * 1024**3,
        pending_budget_bytes: int = 512 * 1024**2,
        max_stall_s: float = 0.050,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._signature = signature
        self._hot_budget = int(hot_budget_bytes)
        self._ssd_capacity = int(ssd_capacity_bytes)
        self._pending_budget = int(pending_budget_bytes)
        self._max_stall_s = float(max_stall_s)
        self._clock = clock
        self.stats = StoreStats()

        self._lock = threading.RLock()
        self._hot: "OrderedDict[str, _HotEntry]" = OrderedDict()
        self._hot_bytes = 0
        self._pins: dict[str, int] = {}

        self._queue: "deque[_PendingWrite]" = deque()
        self._pending = 0
        self._inflight: set[str] = set()
        self._cond = threading.Condition(self._lock)
        self._stopping = False
        self._paused = False
        """Test hook. A paused writer keeps accepting work until the byte
        budget fills, which is how the backpressure path is exercised without
        a slow disk."""

        self._disk: "OrderedDict[str, _DiskEntry]" = OrderedDict()
        self._disk_bytes = 0

        self._root = ""
        self._writer: Optional[threading.Thread] = None
        if ssd_dir:
            self._root = os.path.join(ssd_dir, signature.slug())
            os.makedirs(self._root, exist_ok=True)
            self._scan_existing()
            self._writer = threading.Thread(
                target=self._writer_loop,
                name="titan-store-writer",
                daemon=True,
            )
            self._writer.start()

    # -- port surface ------------------------------------------------------
    def get_block(self, key: BlockHash) -> bytes | None:
        """Blocking read. Called on the prefill path, off the decode loop."""
        return self._get(_key_for_block(key))

    def put_block(self, key: BlockHash, payload: bytes) -> None:
        """Queue a write. Returns as soon as the payload is owned by the store."""
        self._put(_key_for_block(key), payload, kind="block", tokens=0)

    def get_snapshot(self, snapshot_id: str) -> bytes | None:
        return self._get(_key_for_snapshot(snapshot_id))

    def put_snapshot(self, snapshot_id: str, payload: bytes) -> None:
        self._put(_key_for_snapshot(snapshot_id), payload, kind="snapshot", tokens=0)

    def pending_bytes(self) -> int:
        """Bytes queued but not yet durable, including the write in flight."""
        with self._lock:
            return self._pending

    def flush(self, timeout_s: float) -> bool:
        """Drain the write queue. Used at shutdown and in tests."""
        deadline = self._clock() + timeout_s
        with self._cond:
            while self._pending > 0 and not self._stopping:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return self._pending == 0

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Stop the writer. Idempotent; safe to call without an SSD tier."""
        with self._cond:
            self._stopping = True
            self._paused = False
            self._cond.notify_all()
        if self._writer is not None:
            self._writer.join(timeout=5.0)
            self._writer = None

    def set_writer_paused(self, paused: bool) -> None:
        """Test hook: hold the writer so the queue fills predictably."""
        with self._cond:
            self._paused = paused
            self._cond.notify_all()

    # -- pinning -----------------------------------------------------------
    def pin_block(self, key: BlockHash) -> None:
        """Protect one block from hot-tier eviction while a request needs it."""
        self._pin(_key_for_block(key), 1)

    def unpin_block(self, key: BlockHash) -> None:
        self._pin(_key_for_block(key), -1)

    def pin_snapshot(self, snapshot_id: str) -> None:
        self._pin(_key_for_snapshot(snapshot_id), 1)

    def unpin_snapshot(self, snapshot_id: str) -> None:
        self._pin(_key_for_snapshot(snapshot_id), -1)

    def _pin(self, key: str, delta: int) -> None:
        with self._lock:
            count = self._pins.get(key, 0) + delta
            if count > 0:
                self._pins[key] = count
            else:
                self._pins.pop(key, None)

    # -- observability -----------------------------------------------------
    def hot_bytes(self) -> int:
        with self._lock:
            return self._hot_bytes

    def disk_bytes(self) -> int:
        with self._lock:
            return self._disk_bytes

    def contains(self, key: BlockHash) -> bool:
        """Cheap existence check used by lookup, which must not read payloads.

        Walking a 60-block prefix should not pull 60 payloads off disk before
        deciding how long the match is. The restore that follows pulls only the
        blocks the match actually resolved to.
        """
        return self._contains(_key_for_block(key))

    def contains_snapshot(self, snapshot_id: str) -> bool:
        return self._contains(_key_for_snapshot(snapshot_id))

    def tier_of(self, snapshot_id: str) -> str:
        with self._lock:
            if _key_for_snapshot(snapshot_id) in self._hot:
                return "ram"
            return "ssd"

    def metrics(self) -> Mapping[str, float]:
        values = self.stats.as_mapping()
        values["store.hot_bytes"] = float(self.hot_bytes())
        values["store.disk_bytes"] = float(self.disk_bytes())
        values["store.pending_bytes"] = float(self.pending_bytes())
        return values

    # -- internals ---------------------------------------------------------
    def _contains(self, key: str) -> bool:
        with self._lock:
            if key in self._hot or key in self._inflight:
                return True
            return key in self._disk

    def _get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._hot.get(key)
            if entry is not None:
                self._hot.move_to_end(key)
                self.stats.hot_hits += 1
                return entry.payload
            disk = self._disk.get(key)
            path = None if disk is None else disk.path
        if path is None:
            with self._lock:
                self.stats.misses += 1
            return None

        payload = self._read_file(path, key)
        if payload is None:
            with self._lock:
                self._forget_disk(key)
                self.stats.misses += 1
            return None
        with self._lock:
            self.stats.ssd_hits += 1
            disk = self._disk.get(key)
            if disk is not None:
                disk.last_access = self._clock()
                self._disk.move_to_end(key)
            self._hot_put(key, _HotEntry(payload, kind="block", tokens=0, durable=True))
        return payload

    def _read_file(self, path: str, key: str) -> bytes | None:
        """Read one record whole. Never mmap: a state must not hold a file.

        Returning a plain ``bytes`` means the arrays the codec builds from it
        own their memory once it evaluates them, so evicting the file cannot
        pull device memory out from under a running sequence.
        """
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError:
            return None
        try:
            _, payload = decode_record(raw, signature=self._signature, expect_key=key)
        except RecordError as exc:
            with self._lock:
                if "signature" in str(exc):
                    self.stats.signature_rejected += 1
                else:
                    self.stats.corrupt_skipped += 1
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        return payload

    def _put(self, key: str, payload: bytes, *, kind: str, tokens: int) -> None:
        started = self._clock()
        with self._lock:
            if key in self._hot or key in self._inflight or key in self._disk:
                self.stats.duplicate_puts += 1
                if key in self._hot:
                    self._hot.move_to_end(key)
                return
            self.stats.puts += 1
            self._hot_put(key, _HotEntry(payload, kind=kind, tokens=tokens))
            if self._root == "":
                return

        record = encode_record(
            kind=kind,
            key=key,
            signature=self._signature,
            tokens=tokens,
            payload=payload,
        )
        self._enqueue(key, record, started)

    def _enqueue(self, key: str, record: bytes, started: float) -> None:
        path = self._path_for(key)
        pending = _PendingWrite(path=path, record=record, cache_key=key)
        deadline = started + self._max_stall_s
        with self._cond:
            while (
                self._pending + pending.size > self._pending_budget
                and self._pending > 0
                and not self._stopping
            ):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self.stats.writes_dropped += 1
                    self.stats.max_stall_s = max(
                        self.stats.max_stall_s, self._clock() - started
                    )
                    return
                self._cond.wait(remaining)
            if self._stopping:
                self.stats.writes_dropped += 1
                return
            self._queue.append(pending)
            self._inflight.add(key)
            self._pending += pending.size
            self.stats.max_stall_s = max(
                self.stats.max_stall_s, self._clock() - started
            )
            self._cond.notify_all()

    def _path_for(self, key: str) -> str:
        kind, _, name = key.partition(":")
        suffix = _BLOCK_SUFFIX if kind == "b" else _SNAPSHOT_SUFFIX
        stem = name if kind == "b" else name.replace(os.sep, "-")
        shard = (stem[-2:] if kind != "b" else stem[:2]) or "00"
        directory = os.path.join(self._root, kind, shard)
        return os.path.join(directory, stem + suffix)

    def _writer_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait(0.5)
                while self._paused and not self._stopping:
                    self._cond.wait(0.05)
                if self._stopping and not self._queue:
                    return
                if not self._queue:
                    continue
                pending = self._queue.popleft()
            self._write_one(pending)

    def _write_one(self, pending: _PendingWrite) -> None:
        ok = False
        directory = os.path.dirname(pending.path)
        temporary = os.path.join(
            directory, _TEMP_PREFIX + uuid.uuid4().hex + os.path.basename(pending.path)
        )
        try:
            os.makedirs(directory, exist_ok=True)
            with open(temporary, "wb") as handle:
                handle.write(pending.record)
            os.replace(temporary, pending.path)
            ok = True
        except OSError:
            with self._lock:
                self.stats.write_errors += 1
            try:
                os.unlink(temporary)
            except OSError:
                pass
        with self._cond:
            self._pending -= pending.size
            self._inflight.discard(pending.cache_key)
            if ok:
                self.stats.bytes_written += pending.size
                self._disk[pending.cache_key] = _DiskEntry(
                    path=pending.path,
                    size=pending.size,
                    last_access=self._clock(),
                )
                self._disk_bytes += pending.size
                entry = self._hot.get(pending.cache_key)
                if entry is not None:
                    entry.durable = True
                self._evict_disk_locked()
            self._cond.notify_all()

    def _hot_put(self, key: str, entry: _HotEntry) -> None:
        self._hot[key] = entry
        self._hot_bytes += len(entry.payload)
        self._evict_hot_locked()

    def _evict_hot_locked(self) -> None:
        if self._hot_bytes <= self._hot_budget:
            return
        for key in list(self._hot.keys()):
            if self._hot_bytes <= self._hot_budget:
                return
            if self._pins.get(key):
                continue
            entry = self._hot.pop(key)
            self._hot_bytes -= len(entry.payload)
            self.stats.hot_evictions += 1

    def _evict_disk_locked(self) -> None:
        if self._disk_bytes <= self._ssd_capacity:
            return
        for key in list(self._disk.keys()):
            if self._disk_bytes <= self._ssd_capacity:
                return
            if self._pins.get(key):
                continue
            entry = self._disk.pop(key)
            self._disk_bytes -= entry.size
            self.stats.ssd_evictions += 1
            try:
                os.unlink(entry.path)
            except OSError:
                pass

    def _forget_disk(self, key: str) -> None:
        entry = self._disk.pop(key, None)
        if entry is not None:
            self._disk_bytes -= entry.size

    def _scan_existing(self) -> None:
        """Rebuild the disk index from the directory, oldest first.

        One process owns the directory, so there is no lock and no journal: the
        index is whatever the filesystem holds. Files whose header does not
        match the current signature are left alone rather than deleted, because
        the same disk may serve a second model.
        """
        found: list[tuple[float, str, _DiskEntry]] = []
        for kind in ("b", "s"):
            base = os.path.join(self._root, kind)
            if not os.path.isdir(base):
                continue
            for shard in os.listdir(base):
                shard_dir = os.path.join(base, shard)
                if not os.path.isdir(shard_dir):
                    continue
                for name in os.listdir(shard_dir):
                    if name.startswith(_TEMP_PREFIX):
                        try:
                            os.unlink(os.path.join(shard_dir, name))
                        except OSError:
                            pass
                        continue
                    path = os.path.join(shard_dir, name)
                    stem = name
                    for suffix in (_BLOCK_SUFFIX, _SNAPSHOT_SUFFIX):
                        if stem.endswith(suffix):
                            stem = stem[: -len(suffix)]
                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue
                    key = f"{kind}:{stem}"
                    found.append(
                        (
                            stat.st_mtime,
                            key,
                            _DiskEntry(path=path, size=stat.st_size, last_access=stat.st_mtime),
                        )
                    )
        for _, key, entry in sorted(found, key=lambda item: item[0]):
            self._disk[key] = entry
            self._disk_bytes += entry.size
        self._evict_disk_locked()
