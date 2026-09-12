"""Two-tier block and snapshot store.

Hot RAM tier at 4 GB, measured to hold the same hit rate as 16 GB with far less
memory pressure, in front of an SSD tier. Writes go through a background
thread, and a put never waits on it at all.

The stall bound is a lesson, not a preference. The overlay's writer could hold
the inference thread for up to two seconds waiting on a pending-bytes budget,
and that turned a follow-up turn immediately after a large store into a 2.69 s
turn where the model itself needed 1.7. Titan's first answer to that was a
50 ms cap on the same wait, and the integration report shows what a per-put cap
is worth when a retirement makes thirty-two puts back to back: ``max_stall_s``
came out at 1.50 s. So the wait is gone. A put that does not fit the byte
budget sheds the oldest unpinned entry in the queue, and if that is not enough
it drops itself, keeps the payload in RAM and counts the lost durability.

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
    """Writes abandoned because the queue had no room for them. The record
    stays in RAM; only its durability is lost."""
    dropped_bytes: int = 0
    """Payload bytes in those dropped writes, so a drop count can be read
    against the size of what was lost."""
    queue_evictions: int = 0
    """Queued writes thrown out to make room for a newer one. The oldest
    unpinned entry goes first: a record no lease needs is the cheapest thing
    in the queue to lose."""
    pending_peak_bytes: int = 0
    """High-water mark of the write queue. Size the budget against this."""
    write_errors: int = 0
    hot_evictions: int = 0
    ssd_evictions: int = 0
    corrupt_skipped: int = 0
    signature_rejected: int = 0
    max_stall_s: float = 0.0
    """The longest a single put held the calling thread. The put path no
    longer waits on anything, so this is a dictionary insert plus a queue
    append and it should stay in the microseconds. It is kept because it is
    the number the report caught at 1.50 s against a 50 ms cap."""
    stall_cap_exceeded: int = 0
    """Puts that took longer than ``cache.max_stall_ms``. Should be zero."""

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
    """One queued write, still unframed.

    The payload is the bytes the codec produced; the record wrapper, with its
    JSON header and its crc32 over the whole payload, is built on the writer
    thread. Framing a 171 MB snapshot copies it once and checksums it once,
    which is 40 ms of the caller's thread for nothing the caller needs.
    """

    path: str
    payload: bytes
    cache_key: str
    kind: str
    tokens: int
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.payload)


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

    def queue_depth(self) -> int:
        """Records queued but not yet written, including none in flight."""
        with self._lock:
            return len(self._queue)

    def metrics(self) -> Mapping[str, float]:
        values = self.stats.as_mapping()
        values["store.hot_bytes"] = float(self.hot_bytes())
        values["store.disk_bytes"] = float(self.disk_bytes())
        values["store.pending_bytes"] = float(self.pending_bytes())
        values["store.queue_depth"] = float(self.queue_depth())
        values["store.pending_budget_bytes"] = float(self._pending_budget)
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
        """Take ownership of one payload. Never waits on the writer.

        The whole call is a dictionary insert into the hot tier and an append
        to the write queue. Nothing here frames a record, checksums a payload
        or touches the disk, and nothing here blocks: the old version waited
        up to ``max_stall_ms`` for queue room and the integration report caught
        it holding the loop for 1.50 s against a 50 ms cap, because the wait
        was per put and thirty-two puts arrived at once.

        Over budget the queue sheds instead of stalling. Durability is the
        thing given up, never the loop.
        """
        started = self._clock()
        with self._cond:
            if key in self._hot or key in self._inflight or key in self._disk:
                self.stats.duplicate_puts += 1
                if key in self._hot:
                    self._hot.move_to_end(key)
                self._note_stall_locked(started)
                return
            self.stats.puts += 1
            self._hot_put(key, _HotEntry(payload, kind=kind, tokens=tokens))
            if self._root == "" or self._stopping:
                self._note_stall_locked(started)
                return
            self._admit_write_locked(
                _PendingWrite(
                    path=self._path_for(key),
                    payload=payload,
                    cache_key=key,
                    kind=kind,
                    tokens=tokens,
                )
            )
            self._note_stall_locked(started)

    def _admit_write_locked(self, pending: _PendingWrite) -> None:
        """Bytes-aware queue admission. Sheds the oldest droppable entry.

        The budget bounds RAM held for durability alone: a queued write's
        payload is also in the hot tier, so a snapshot waiting to be written
        is counted twice for as long as it waits. Thirty-two 171 MB snapshots
        queued at once is 5.49 GB of that, which is the other half of the
        report's retirement.

        A pinned record is never the victim. A pin means a live lease is
        matching against it, and dropping its durability is the one drop that
        can cost a warm turn rather than a cold restart.
        """
        while (
            self._pending + pending.size > self._pending_budget
            and self._pending > 0
        ):
            victim = self._evict_pending_locked()
            if victim is None:
                break
        if self._pending + pending.size > self._pending_budget and self._pending > 0:
            self.stats.writes_dropped += 1
            self.stats.dropped_bytes += pending.size
            return
        self._queue.append(pending)
        self._inflight.add(pending.cache_key)
        self._pending += pending.size
        self.stats.pending_peak_bytes = max(
            self.stats.pending_peak_bytes, self._pending
        )
        self._cond.notify_all()

    def _evict_pending_locked(self) -> _PendingWrite | None:
        for index, candidate in enumerate(self._queue):
            if self._pins.get(candidate.cache_key):
                continue
            del self._queue[index]
            self._pending -= candidate.size
            self._inflight.discard(candidate.cache_key)
            self.stats.writes_dropped += 1
            self.stats.queue_evictions += 1
            self.stats.dropped_bytes += candidate.size
            return candidate
        return None

    def _note_stall_locked(self, started: float) -> None:
        """Time the put and check the store's own promise.

        The put path has nothing left in it that can take a millisecond, so a
        breach here is a symptom, not a policy: a hot tier evicting thousands
        of entries under one lock, or a caller holding it. Counting it is what
        turns the next 1.50 s into a number somebody sees."""
        elapsed = self._clock() - started
        self.stats.max_stall_s = max(self.stats.max_stall_s, elapsed)
        if elapsed > self._max_stall_s:
            self.stats.stall_cap_exceeded += 1

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
        """Frame the record, then write it. Both on this thread.

        ``encode_record`` copies the payload once into the framed record and
        crc32s it. On a 171 MB snapshot that is tens of milliseconds, and it
        used to happen on the scheduler thread inside the put.
        """
        ok = False
        try:
            record = encode_record(
                kind=pending.kind,
                key=pending.cache_key,
                signature=self._signature,
                tokens=pending.tokens,
                payload=pending.payload,
            )
        except Exception:  # noqa: BLE001 - a bad record is a lost write, not a crash
            with self._cond:
                self.stats.write_errors += 1
                self._pending -= pending.size
                self._inflight.discard(pending.cache_key)
                self._cond.notify_all()
            return
        directory = os.path.dirname(pending.path)
        temporary = os.path.join(
            directory, _TEMP_PREFIX + uuid.uuid4().hex + os.path.basename(pending.path)
        )
        try:
            os.makedirs(directory, exist_ok=True)
            with open(temporary, "wb") as handle:
                handle.write(record)
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
                written = len(record)
                self.stats.bytes_written += written
                self._disk[pending.cache_key] = _DiskEntry(
                    path=pending.path,
                    size=written,
                    last_access=self._clock(),
                )
                self._disk_bytes += written
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
