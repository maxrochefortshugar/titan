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

## The asymmetry the whole design turns on

Attention KV is positional and sliceable, so it is stored per 512-token block
and reassembled block by block. Gated DeltaNet state is recurrent and has no
inverse, so it can only be resumed at a point where a snapshot was taken. A
match therefore has two lengths: how far the block chain agrees, and how far
back from there the newest snapshot sits. The second one is the answer, and the
difference between the two is recompute the engine is told about precisely
rather than discovering at prefill time.

Keeping the two grids separate is worth about half a second on a warm turn. On
the six-turn probe the overlay's single 2048 grid recomputed 11617 tokens; the
512 tail with a snapshot at each fine boundary recomputes 8545, and the warm
median went from 2.59 s to 2.10 s.

## Call sequence

The engine drives one turn like this, and nothing here is optional:

```
match  = cache.lookup(tokens)             # or cache.match, the same call
lease  = cache.reserve(match)             # pins the blocks against eviction
n      = cache.restore(match, state)      # bytes in, n <= match.matched_tokens
ends   = cache.plan_chunks(n, len(tokens), contended)
snaps  = cache.snapshot_boundaries(n, len(tokens), contended)
writer = cache.begin_store()              # incremental, one per sequence
for end in ends:                          # one chunk per scheduler turn
    backend.prefill(state, tokens[pos:end],
                    want_logits=(end == len(tokens)),
                    snapshot=(end in snaps))
    if end in snaps:
        writer.note_boundary(end)
    writer.pump(tokens, state, force_one=True)
...decode, pumping each cycle...
cache.store(full_tokens, state, snaps)    # or cache.commit, the same call
cache.release(lease)
```

The session is what keeps the store off the retirement turn. Without one,
``store`` serialises every boundary in a single call: 32 snapshots and 5.49 GB
took 15.4 s of loop thread on the measured 65k request, and the request behind
it decoded at 25 tok/s instead of 55 for the duration.

``restore`` may return fewer tokens than the match promised if a block went
missing between the lookup and the read. The planner takes its return value,
never the match's, which is why ``n`` and not ``match.matched_tokens`` is what
feeds ``plan_chunks``.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from titan.core.types import BlockHash, PrefixMatch, StateHandle

from titan.adapters.cache.codec import StateCodec
from titan.adapters.cache.format import chain_hash, snapshot_id_for
from titan.adapters.cache.store import TwoTierStateStore

logger = logging.getLogger(__name__)

__all__ = ["PrefixLease", "PrefixStats", "PrefixWriteSession", "BlockPrefixCache"]


@dataclass
class _BlockEntry:
    """One 512-token block in the chain index."""

    hash: bytes
    index: int
    """Block number from the start of the sequence. ``end`` is derived."""
    parent: Optional[bytes]
    refs: int = 0
    last_access: float = 0.0


@dataclass
class _SnapshotEntry:
    """A recurrent snapshot, named by the block whose end it sits at."""

    snapshot_id: str
    length: int
    block_hash: bytes
    refs: int = 0


@dataclass(frozen=True, slots=True)
class PrefixLease:
    """A claim on a matched prefix, held for as long as a request needs it.

    Two requests that arrive on the same prefix take two leases over the same
    blocks. The blocks are stored once, pinned while either lease lives, and
    become evictable again when the second one is released. Without this a
    long-running request can watch its own prefix get evicted underneath it by
    the request that followed it.
    """

    lease_id: int
    block_hashes: tuple[BlockHash, ...]
    snapshot_id: str | None
    tokens: int


@dataclass
class PrefixStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    matched_tokens: int = 0
    recompute_tokens: int = 0
    kv_only_blocks: int = 0
    """Blocks whose KV matched but which sat past the newest usable snapshot.
    A number that climbs here means the snapshot policy is too coarse."""
    restored_tokens: int = 0
    restore_degraded: int = 0
    restore_aborted: int = 0
    blocks_written: int = 0
    blocks_deduped: int = 0
    snapshots_written: int = 0
    snapshots_deduped: int = 0
    snapshots_failed: int = 0
    snapshot_seconds: float = 0.0
    fine_snapshots: int = 0
    """Snapshots written at a prompt end rather than on the coarse grid. These
    are the ones that pay for the fine tail."""
    chain_truncations: int = 0
    leases_open: int = 0
    store_seconds: float = 0.0
    """Loop-thread seconds spent serialising, summed over every pump."""
    max_store_stall_s: float = 0.0
    """The longest a single pump held the loop thread. This is the number the
    integration report measured at 1.50 s against a 50 ms cap, and the one the
    budget exists to hold down."""
    store_pumps: int = 0
    snapshots_deferred: int = 0
    """Pumps that stopped with work still queued because the next boundary did
    not fit the remaining budget. Deferred, not dropped: the work happens on a
    later cycle or in the post-retirement drain."""
    drain_snapshots: int = 0
    """Boundaries serialised after the sequence retired."""
    boundaries_abandoned: int = 0
    """Boundaries dropped because the drain ran out of its own budget. The
    chain truncates to the deepest snapshot that did commit."""

    def as_mapping(self) -> dict[str, float]:
        return {f"prefix.{name}": float(value) for name, value in vars(self).items()}


class PrefixWriteSession:
    """One sequence's store, spread over the cycles the sequence lives for.

    The engine used to hand the cache a finished prompt and a list of
    boundaries, and the cache serialised all of them in one call on the
    scheduler loop thread. On a 65k-token request that is 32 snapshots and
    5.49 GB, and the integration report timed it at 15.4 s, during which the
    next request decoded at 25 tok/s instead of 55.

    A session turns that burst into a stream. Each boundary is serialised when
    it is reached, or on a later cycle if the loop is busy, and retirement is
    left with the tail nobody had time for rather than the whole prompt.

    What runs where, per boundary:

    * loop thread: one ``export_snapshot`` and one ``export_blocks`` per new
      block, which is an ``mx.eval`` of the staged slice and a copy of it into
      ``bytes``, plus the sha256 chain hash of the block's token ids;
    * writer thread: the record framing and its crc32, and the file write.

    The chain hash stays on the loop thread on purpose. It is sha256 over 512
    token ids, about 20 microseconds a block against tens of milliseconds for
    the copy, and moving it would mean the index that ``lookup`` reads is
    behind the bytes the store already holds.

    Not thread safe. One session belongs to one sequence and is only ever
    touched by the loop thread that owns it.
    """

    def __init__(
        self,
        cache: "BlockPrefixCache",
        *,
        budget_s: float,
        clock: Callable[[], float],
    ) -> None:
        self._cache = cache
        self._budget_s = float(budget_s)
        self._clock = clock
        self._queued: list[int] = []
        self._served: set[int] = set()
        self._chain: list[bytes] = []
        self._next_block = 0
        self._deepest = 0
        self._draining = False
        self._unit_s = 0.0
        """Exponential mean of what one boundary costs the loop thread. The
        budget is spent against this rather than against a fixed count,
        because a boundary is 40 ms on a 4k prompt and 400 on a 64k one."""

    # -- properties --------------------------------------------------------
    @property
    def pending(self) -> int:
        """Boundaries reached but not yet serialised."""
        return len(self._queued)

    @property
    def deepest_committed(self) -> int:
        return self._deepest

    @property
    def unit_estimate_s(self) -> float:
        return self._unit_s

    # -- driving -----------------------------------------------------------
    def note_boundary(self, length: int) -> None:
        """Record that the backend staged a snapshot at ``length``.

        Cheap by design: the scheduler calls it from the prefill stage, and
        anything expensive here would be the burst again, one chunk at a time.
        """
        block = self._cache.block_tokens
        rounded = (int(length) // block) * block
        if rounded <= 0 or rounded in self._served or rounded in self._queued:
            return
        self._queued.append(rounded)
        self._queued.sort()

    def begin_drain(self) -> None:
        """Mark the sequence retired. Only changes what the pumps are counted
        as, so a drain that is doing real work is visible in ``/metrics``."""
        self._draining = True

    def pump(
        self,
        tokens: Sequence[int],
        state: StateHandle,
        *,
        budget_s: float | None = None,
        force_one: bool = False,
    ) -> float:
        """Serialise what fits in the budget. Returns seconds spent.

        The gate is ``spent + estimate > budget``, so a boundary is only
        started when the measured cost of the last one says it will fit. A
        pump that would overrun stops instead and leaves the rest queued,
        which is the whole enforcement of the cap: the loop thread cannot be
        preempted once inside ``export_snapshot``, so the only lever is not
        entering it.

        ``force_one`` runs one boundary whatever the estimate says. The
        post-retirement drain sets it, because a drain that defers forever is
        a state handle that never closes.
        """
        budget = self._budget_s if budget_s is None else float(budget_s)
        started = self._clock()
        spent = 0.0
        did_one = False
        while self._queued:
            if not (force_one and not did_one) and spent + self._unit_s > budget:
                self._cache.counters.snapshots_deferred += 1
                break
            length = self._queued[0]
            if length > len(tokens):
                # The boundary is past what the caller can name yet. It will
                # be serialisable on a later pump, or dropped at finish.
                break
            unit_started = self._clock()
            self._cache._serialise_boundary(self, tokens, state, length)
            elapsed = self._clock() - unit_started
            self._unit_s = elapsed if self._unit_s == 0.0 else (
                0.5 * self._unit_s + 0.5 * elapsed
            )
            self._queued.pop(0)
            self._served.add(length)
            if self._draining:
                self._cache.counters.drain_snapshots += 1
            did_one = True
            spent = self._clock() - started
        spent = self._clock() - started
        counters = self._cache.counters
        counters.store_pumps += 1
        counters.store_seconds += spent
        counters.max_store_stall_s = max(counters.max_store_stall_s, spent)
        return spent

    def finish(self, tokens: Sequence[int], total: int) -> None:
        """Close the chain out. Counts a truncation if it is short.

        Called once the last boundary is serialised, whether that happened on
        a prefill cycle or in the drain.
        """
        block = self._cache.block_tokens
        full_blocks = min(len(tokens), total) // block
        if full_blocks and self._deepest // block < full_blocks:
            self._cache.counters.chain_truncations += 1

    def abandon(self) -> None:
        """Give up on the boundaries still queued and count them."""
        if self._queued:
            self._cache.counters.boundaries_abandoned += len(self._queued)
            self._queued.clear()


class BlockPrefixCache:
    """A :class:`titan.core.ports.PrefixCache` over a two-tier byte store.

    Everything is keyed by content. A block is named by the sha256 chain of its
    predecessor, the compatibility signature and its own token ids, so two
    conversations that share a system prompt share its blocks without either of
    them knowing, and a build that changes the layer layout cannot read a byte
    the previous one wrote.

    The index lives in RAM and is authoritative about what the cache believes;
    the store is authoritative about what actually exists. Lookup asks both,
    which is what makes an eviction under the index harmless rather than a
    dangling match.
    """

    def __init__(
        self,
        store: TwoTierStateStore,
        codec: StateCodec,
        *,
        block_tokens: int = 512,
        snapshot_grid: int = 2048,
        snapshot_at_prompt_end: bool = True,
        chunk_tokens: int = 2048,
        contended_chunk_tokens: int = 512,
        fine_min_gain_tokens: int = 384,
        fine_max_pending_bytes: int = 192 * 1024**2,
        store_budget_s: float = 0.020,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        if snapshot_grid % block_tokens:
            raise ValueError("snapshot_grid must be a multiple of block_tokens")
        if chunk_tokens % block_tokens or contended_chunk_tokens % block_tokens:
            raise ValueError("prefill chunks must be a multiple of block_tokens")
        self._store = store
        self._codec = codec
        self._block = int(block_tokens)
        self._grid = int(snapshot_grid)
        self._prompt_end_snapshot = bool(snapshot_at_prompt_end)
        self._chunk = int(chunk_tokens)
        self._contended_chunk = int(contended_chunk_tokens)
        self._fine_min_gain = int(fine_min_gain_tokens)
        self._fine_max_pending = int(fine_max_pending_bytes)
        self._store_budget_s = max(float(store_budget_s), 0.0)
        self._clock = clock
        self._signature_digest = codec.signature.digest()

        self._lock = threading.RLock()
        self._index: dict[bytes, _BlockEntry] = {}
        self._snapshots: dict[bytes, _SnapshotEntry] = {}
        self._leases: dict[int, PrefixLease] = {}
        self._lease_ids = itertools.count(1)
        self._snapshot_ids: dict[str, _SnapshotEntry] = {}
        self.counters = PrefixStats()

    # -- properties --------------------------------------------------------
    @property
    def block_tokens(self) -> int:
        return self._block

    @property
    def snapshot_grid(self) -> int:
        return self._grid

    @property
    def store_budget_s(self) -> float:
        """Loop-thread seconds one sequence's store may spend per cycle.

        Read by the scheduler, which owns the loop and does the pumping. It
        lives here because ``cache.store_budget_ms`` is a cache setting and
        the engine may not import the config.
        """
        return self._store_budget_s

    # -- lookup ------------------------------------------------------------
    def lookup(self, tokens: Sequence[int]) -> PrefixMatch:
        """Longest restorable prefix of ``tokens``.

        Two walks. Forward over full blocks while the chain hash is known and
        the bytes still exist, which gives the KV frontier. Then backward from
        that frontier to the newest block end that also has a recurrent
        snapshot, which gives the answer. The gap between the two is counted,
        not returned: KV without a snapshot is not a restorable state.

        The match never covers the whole prompt. Prefill needs at least one
        token to produce logits from, so the deepest candidate block end is
        ``len(tokens) - 1`` rounded down to the grid.
        """
        with self._lock:
            self.counters.lookups += 1
            total = len(tokens)
            usable_blocks = max(total - 1, 0) // self._block

            chain: list[bytes] = []
            parent: Optional[bytes] = None
            now = self._clock()
            for index in range(usable_blocks):
                start = index * self._block
                digest = chain_hash(
                    parent,
                    tokens[start : start + self._block],
                    self._signature_digest,
                )
                entry = self._index.get(digest)
                if entry is None or not self._store.contains(BlockHash(digest)):
                    break
                entry.last_access = now
                chain.append(digest)
                parent = digest

            for depth in range(len(chain), 0, -1):
                digest = chain[depth - 1]
                snapshot = self._snapshots.get(digest)
                if snapshot is None:
                    continue
                if not self._store.contains_snapshot(snapshot.snapshot_id):
                    continue
                matched = depth * self._block
                self.counters.hits += 1
                self.counters.matched_tokens += matched
                self.counters.recompute_tokens += total - matched
                self.counters.kv_only_blocks += len(chain) - depth
                return PrefixMatch(
                    matched_tokens=matched,
                    block_hashes=tuple(BlockHash(h) for h in chain[:depth]),
                    snapshot_id=snapshot.snapshot_id,
                    tier=self._store.tier_of(snapshot.snapshot_id),
                )

            self.counters.misses += 1
            self.counters.kv_only_blocks += len(chain)
            self.counters.recompute_tokens += total
            return PrefixMatch(
                matched_tokens=0,
                block_hashes=(),
                snapshot_id=None,
                tier="none",
            )

    match = lookup
    """The engine's name for :meth:`lookup`. Same call, same semantics."""

    def recompute_tail(self, match: PrefixMatch, tokens: Sequence[int]) -> int:
        """Tokens the engine must still forward after restoring ``match``."""
        return max(len(tokens) - match.matched_tokens, 0)

    # -- reference counting ------------------------------------------------
    def reserve(self, match: PrefixMatch) -> PrefixLease:
        """Pin everything ``match`` names until the lease is released."""
        with self._lock:
            lease = PrefixLease(
                lease_id=next(self._lease_ids),
                block_hashes=match.block_hashes,
                snapshot_id=match.snapshot_id,
                tokens=match.matched_tokens,
            )
            for digest in match.block_hashes:
                entry = self._index.get(bytes(digest))
                if entry is not None:
                    entry.refs += 1
                self._store.pin_block(digest)
            if match.snapshot_id is not None:
                snapshot = self._snapshot_by_id(match.snapshot_id)
                if snapshot is not None:
                    snapshot.refs += 1
                self._store.pin_snapshot(match.snapshot_id)
            self._leases[lease.lease_id] = lease
            self.counters.leases_open = len(self._leases)
            return lease

    def release(self, lease: PrefixLease) -> None:
        """Drop a lease. The last one to go makes its blocks evictable again.

        Idempotent: releasing a lease twice is a no-op rather than a refcount
        that goes negative and pins a prefix forever.
        """
        with self._lock:
            if self._leases.pop(lease.lease_id, None) is None:
                return
            for digest in lease.block_hashes:
                entry = self._index.get(bytes(digest))
                if entry is not None and entry.refs > 0:
                    entry.refs -= 1
                self._store.unpin_block(digest)
            if lease.snapshot_id is not None:
                snapshot = self._snapshot_by_id(lease.snapshot_id)
                if snapshot is not None and snapshot.refs > 0:
                    snapshot.refs -= 1
                self._store.unpin_snapshot(lease.snapshot_id)
            self.counters.leases_open = len(self._leases)

    def ref_count(self, digest: BlockHash) -> int:
        with self._lock:
            entry = self._index.get(bytes(digest))
            return 0 if entry is None else entry.refs

    # -- restore -----------------------------------------------------------
    def restore(self, match: PrefixMatch, state: StateHandle) -> int:
        """Populate ``state`` from a match. Returns tokens actually restored.

        Every byte is fetched before a single one is imported. A block that
        went missing between the lookup and the read shortens the restore to
        the newest snapshot the surviving blocks reach, and that shortening
        happens while the state is still empty, so the engine gets a smaller
        number and prefills more rather than getting a state that is partly
        someone else's.
        """
        if match.matched_tokens <= 0:
            return 0
        with self._lock:
            payloads: list[bytes] = []
            for digest in match.block_hashes:
                payload = self._store.get_block(digest)
                if payload is None:
                    break
                payloads.append(payload)

            depth = len(payloads)
            snapshot_blob: bytes | None = None
            snapshot_length = 0
            while depth > 0:
                digest = bytes(match.block_hashes[depth - 1])
                snapshot = self._snapshots.get(digest)
                if snapshot is not None:
                    snapshot_blob = self._store.get_snapshot(snapshot.snapshot_id)
                    if snapshot_blob is not None:
                        snapshot_length = snapshot.length
                        break
                depth -= 1

            if depth == 0 or snapshot_blob is None:
                self.counters.restore_degraded += 1
                return 0
            if snapshot_length != match.matched_tokens:
                self.counters.restore_degraded += 1

            try:
                for index in range(depth):
                    start = index * self._block
                    self._codec.import_blocks(
                        state, start, start + self._block, payloads[index]
                    )
                self._codec.import_snapshot(state, snapshot_length, snapshot_blob)
            except Exception as exc:  # noqa: BLE001 - degrade, never raise
                logger.warning("restore aborted at %d: %s", snapshot_length, exc)
                self.counters.restore_aborted += 1
                return 0

            self.counters.restored_tokens += snapshot_length
            return snapshot_length

    # -- store -------------------------------------------------------------
    def begin_store(self, *, budget_s: float | None = None) -> PrefixWriteSession:
        """Open an incremental store for one sequence.

        The engine opens one at admission, tells it about each boundary as
        prefill reaches it, and pumps it once per cycle. What is left at
        retirement is the tail, not the prompt.
        """
        return PrefixWriteSession(
            self,
            budget_s=self._store_budget_s if budget_s is None else budget_s,
            clock=self._clock,
        )

    def store(
        self,
        tokens: Sequence[int],
        state: StateHandle,
        boundaries: Sequence[int],
    ) -> None:
        """Persist ``tokens`` with resumable points at ``boundaries``.

        The one-shot form, and now a special case of the incremental one: a
        session with no budget, told about every boundary at once. A caller
        that can afford to block the thread it is on -- a test, a bench, an
        engine with no session -- gets exactly the old behaviour.

        Boundaries are rounded down to the block grid, because a restore point
        has to be a block end: the KV either covers whole blocks or the chain
        hash of everything after it changes. A boundary at the prompt end of
        25043 tokens is therefore recorded at 24576, and the 467 tokens after
        it are recomputed next turn.

        Snapshots are written first and blocks second, and blocks are written
        only up to the deepest snapshot that committed. That ordering is the
        third rule: a block chain that reaches further than any snapshot is
        unusable on the next turn, and recording it would let a lookup report a
        length it cannot restore.
        """
        total = len(tokens)
        cap = (total // self._block) * self._block
        if cap == 0:
            return
        session = self.begin_store(budget_s=float("inf"))
        for value in boundaries:
            session.note_boundary(min(int(value), cap))
        session.pump(tokens, state, budget_s=float("inf"))
        session.finish(tokens, total)

    # -- the incremental path ----------------------------------------------
    def _serialise_boundary(
        self,
        session: PrefixWriteSession,
        tokens: Sequence[int],
        state: StateHandle,
        length: int,
    ) -> bool:
        """One boundary: its snapshot, then the blocks it makes reachable.

        This is the only method the loop thread spends real time in, and every
        millisecond of it is a codec call. Ordering is the store's third rule
        stated per boundary rather than per prompt: the snapshot first, its
        blocks second, and no block published past a snapshot that did not
        commit.
        """
        with self._lock:
            blocks = length // self._block
            if blocks == 0 or blocks * self._block > len(tokens):
                return False
            chain = self._chain_upto(session, tokens, blocks)
            digest = chain[blocks - 1]
            snapshot_id = snapshot_id_for(digest, length)
            if digest in self._snapshots and self._store.contains_snapshot(snapshot_id):
                self.counters.snapshots_deduped += 1
            else:
                started = self._clock()
                try:
                    blob = self._codec.export_snapshot(state, length)
                except Exception as exc:  # noqa: BLE001 - a boundary, not a turn
                    # Counted and logged. A dropped boundary is the difference
                    # between a warm next turn and a cold one, so a silent
                    # count leaves nobody able to say which of the three
                    # reasons it was.
                    logger.warning("snapshot at %d did not commit: %s", length, exc)
                    self.counters.snapshots_failed += 1
                    return False
                self._store.put_snapshot(snapshot_id, blob)
                self.counters.snapshots_written += 1
                if length % self._grid:
                    self.counters.fine_snapshots += 1
                self.counters.snapshot_seconds += self._clock() - started
                entry = _SnapshotEntry(
                    snapshot_id=snapshot_id,
                    length=length,
                    block_hash=digest,
                )
                self._snapshots[digest] = entry
                self._snapshot_ids[snapshot_id] = entry
            if not self._write_blocks(session, chain, state, blocks):
                return False
            session._deepest = max(session._deepest, length)
            return True

    def _write_blocks(
        self,
        session: PrefixWriteSession,
        chain: Sequence[bytes],
        state: StateHandle,
        upto: int,
    ) -> bool:
        """Export and index blocks ``[session._next_block, upto)``.

        Each block is exported once per process, whichever boundary first
        reaches it, so a 65k prompt with 32 boundaries still exports its 128
        blocks 128 times and not 4096.
        """
        now = self._clock()
        parent = chain[session._next_block - 1] if session._next_block else None
        for index in range(session._next_block, upto):
            digest = chain[index]
            start = index * self._block
            if digest in self._index and self._store.contains(BlockHash(digest)):
                self._index[digest].last_access = now
                self.counters.blocks_deduped += 1
                parent = digest
                session._next_block = index + 1
                continue
            try:
                payload = self._codec.export_blocks(state, start, start + self._block)
            except Exception as exc:  # noqa: BLE001 - the chain, not a turn
                logger.warning(
                    "block [%d, %d) did not export: %s",
                    start,
                    start + self._block,
                    exc,
                )
                self.counters.chain_truncations += 1
                return False
            self._store.put_block(BlockHash(digest), payload)
            self._index[digest] = _BlockEntry(
                hash=digest,
                index=index,
                parent=parent,
                last_access=now,
            )
            self.counters.blocks_written += 1
            parent = digest
            session._next_block = index + 1
        return True

    def _chain_upto(
        self,
        session: PrefixWriteSession,
        tokens: Sequence[int],
        blocks: int,
    ) -> list[bytes]:
        """Extend the session's chain hashes to ``blocks``, reusing the rest.

        A block's digest depends only on the tokens before it, so a session
        that hashed 48 blocks at one boundary hashes 4 more at the next rather
        than all 52 again.
        """
        chain = session._chain
        parent = chain[-1] if chain else None
        for index in range(len(chain), blocks):
            start = index * self._block
            parent = chain_hash(
                parent,
                tokens[start : start + self._block],
                self._signature_digest,
            )
            chain.append(parent)
        return chain

    commit = store
    """The engine's name for :meth:`store`. Same call, same semantics."""

    # -- planning ----------------------------------------------------------
    def snapshot_boundaries(
        self,
        matched: int,
        total: int,
        contended: bool = False,
    ) -> tuple[int, ...]:
        """Where the backend should stage a recurrent snapshot this prompt.

        The coarse grid, always, because a long prompt that is never resumed
        from its middle still needs somewhere to resume from. Plus one cut at
        the prompt end rounded down to the block grid, which is the whole point
        of the fine tail: without it a follow-up turn resumes at the previous
        2048 multiple and recomputes up to 2047 tokens it has already seen.

        Chunk ends that are neither are deliberately excluded. A snapshot is
        110 MiB, and emitting one at every chunk end quadruples the snapshot
        rate exactly on the contended path that shortened the chunks.
        """
        if total <= matched:
            return ()
        first = ((matched // self._grid) + 1) * self._grid
        points = {g for g in range(first, total + 1, self._grid)}
        fine = (total // self._block) * self._block
        if self._prompt_end_snapshot and fine > matched:
            if self._fine_cut_allowed(matched, total, fine):
                points.add(fine)
        return tuple(sorted(points))

    def plan_chunks(
        self,
        matched: int,
        total: int,
        contended: bool,
    ) -> tuple[int, ...]:
        """Chunk end positions for the uncached suffix.

        Every point that wants a snapshot ends a chunk, because the backend can
        only stage one at the end of a chunk. That is the whole of rule one:
        the round-4a planner clamped straight to its fine target, ran a chunk
        from 26112 to 27648 that stepped over the grid multiple at 26624, and
        the store found nothing staged there and truncated the chain back to
        26112.
        """
        if total <= matched:
            return ()
        chunk = self._contended_chunk if contended else self._chunk
        cuts = {b for b in self.snapshot_boundaries(matched, total, contended) if b < total}
        ends: list[int] = []
        position = matched
        while position < total:
            end = min(position + chunk, total)
            inside = [c for c in cuts if position < c < end]
            if inside:
                end = min(inside)
            ends.append(end)
            position = end
        return tuple(ends)

    def _fine_cut_allowed(self, matched: int, total: int, fine: int) -> bool:
        """Three gates, in the order they are cheapest to check.

        The extra cut costs one chunk launch and one snapshot, about 283 ms
        together on a quiet writer, and buys back the tokens between the last
        coarse point and the prompt end. Under 384 tokens of gain that trade is
        negative, and while the writer still holds a large backlog the snapshot
        stops being 210 ms and starts being the turn-2 stall.
        """
        coarse = (total // self._grid) * self._grid
        if coarse <= matched:
            coarse = matched
        if fine <= coarse:
            return False
        if fine - coarse < self._fine_min_gain:
            return False
        if self._store.pending_bytes() > self._fine_max_pending:
            return False
        return True

    # -- observability -----------------------------------------------------
    def stats_snapshot(self) -> PrefixStats:
        with self._lock:
            return PrefixStats(**vars(self.counters))

    def stats(self) -> Mapping[str, float]:  # type: ignore[override]
        """Hit rate, restored tokens, recomputed tokens, evictions.

        Flat ``name -> float``, prefixed by layer, so ``/metrics`` can dump it
        without knowing anything about the cache.
        """
        with self._lock:
            values = self.stats_snapshot().as_mapping()
        values.update(self._store.metrics())
        lookups = values["prefix.lookups"]
        values["prefix.hit_rate"] = (
            values["prefix.hits"] / lookups if lookups else 0.0
        )
        snapshots = values["prefix.snapshots_written"]
        values["prefix.snapshot_seconds_mean"] = (
            values["prefix.snapshot_seconds"] / snapshots if snapshots else 0.0
        )
        return values

    def prune(self) -> int:
        """Drop index entries whose bytes are gone from both tiers.

        Not required for correctness, since lookup checks the store on every
        block, but an index that outlives its payloads is a slow memory leak on
        a long-running process.
        """
        with self._lock:
            dead = [
                digest
                for digest, entry in self._index.items()
                if entry.refs == 0 and not self._store.contains(BlockHash(digest))
            ]
            for digest in dead:
                self._index.pop(digest, None)
                stale = self._snapshots.pop(digest, None)
                if stale is not None:
                    self._snapshot_ids.pop(stale.snapshot_id, None)
            return len(dead)

    # -- internals ---------------------------------------------------------
    def _snapshot_by_id(self, snapshot_id: str) -> Optional[_SnapshotEntry]:
        return self._snapshot_ids.get(snapshot_id)
