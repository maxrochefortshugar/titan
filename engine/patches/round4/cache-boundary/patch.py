# SPDX-License-Identifier: Apache-2.0
"""Fine-grained prefix-cache store boundaries for the split-GDN layout.

Background
----------
``config.paged_cache_block_size`` does three jobs at once on this model:

1. it is the granularity at which a warm prefix can be matched and restored
   (``cache/paged_cache.py:1035`` ``get_computed_blocks`` floors to whole
   blocks),
2. it is the grid on which prefill chunks are cut and GDN boundary snapshots
   are emitted (``scheduler.py:3686/3830/3973/5498/5602/5735`` via
   ``prefill_boundaries.clamp_prefill_chunk_to_boundary`` and
   ``should_emit_prefill_boundary``), and
3. it is the grid on which ``store_cache`` demands a committed recurrent
   sidecar (``cache/prefix_cache.py:1285-1330``).

``Scheduler._enlarge_block_size_for_arrays_cache`` (``scheduler.py:2848-2901``)
pins it to 2048 for this model, so a warm prefix always floors to a 2048
multiple and every follow-up turn re-prefills the trailing partial block.

Round 3 lowered the block size to 512 and left the snapshots on the 2048 grid,
so every store truncated at the first block. Round 4a added the fine tail cut
and the off-grid carve-out below; four of its five stores worked and one did
not. The failing store (26112 restored, 1807-token suffix, boundary 27648)
had a coarse boundary *inside* the suffix: the tail clamp jumped straight
from 26112 to the fine target 27648, so the chunk crossed 26624 = 13 x 2048
without ending there and no snapshot was staged at 26624. The block ending at
26624 is coarse-aligned, so the strict gate in ``commit()`` below (correctly)
refused it, ``store_cache`` truncated the chain, and the next turn restored
26112 and recomputed 3245 tokens. Fix: the tail cut now stops at the next
coarse boundary first and takes the fine cut afterwards.

What this patch does
--------------------
Restore already tolerates blocks without a recurrent checkpoint: the split-GDN
endpoint search at ``cache/prefix_cache.py:3348-3400`` walks back from the last
block to the newest one whose sidecar loads, and truncates the chain there. So
a block that carries only its (fully sliceable) QSA KV is safe to keep; it
simply cannot serve as a restore endpoint. This patch makes the store agree
with that, and puts checkpoints exactly where they are wanted:

* prefill chunks stay 2048 wide for the body of a prompt. In the tail, the
  chunk stops at the next coarse boundary when the suffix crosses one, and is
  then cut once more so it ends at the highest ``fine`` multiple at or below
  the prompt end. That emits one extra boundary snapshot per prompt;
* the paged block size drops to ``fine`` so matching, storing and restoring can
  address that boundary;
* a block whose end is not on the coarse grid may be stored without a committed
  sidecar instead of truncating the chain. A missing sidecar on the coarse grid
  is still an anomaly and still truncates, as in stock;
* decode-time boundary captures and MTP commit alignment stay on the coarse
  grid, so lowering the block size does not multiply 110 MiB snapshots by four
  during generation.

The extra snapshot is not free. ``_on_prefill_boundary_snapshot``
(``scheduler.py:6761-6770``) calls ``BoundarySnapshotSSDStore.save`` on the
inference thread; save extracts and evaluates the recurrent state, copies
115.66 MB to host bytes, and can block up to 2 s against the 512 MB pending
budget (``cache/boundary_snapshot_store.py:296-312``) when the writer is still
draining an earlier store. Measured on the workbench, one extra tail snapshot
plus its extra chunk launch costs about 0.29 s with an idle writer and about
0.97 s directly after a cold-prefill store. Three gates keep that cost off the
critical path when it cannot pay for itself: a minimum gain, an optional
minimum remainder, and a writer-backlog check.

Install at IMPORT time, before the engine starts. It rebinds module-level
functions and wraps three ``Scheduler`` methods plus one
``BlockAwarePrefixCache`` method; it touches no model instance.

Bootstrap hook:
  OMLX_ROUND2_IMPORT_PATCHES=".../round4/cache-boundary/patch.py:install"

Env:
  OMLX_CACHE_FINE_TAIL=512            fine store grid in tokens; 0/unset disables
  OMLX_CACHE_COARSE_CHUNK=2048        prefill chunk width kept for the prompt body
  OMLX_CACHE_FINE_TAIL_MIN_GAIN=384   skip the fine cut when it would advance the
                                      stored boundary by fewer tokens than this;
                                      384 tokens is the ~0.28 s the extra chunk
                                      and snapshot cost at 0.65 ms per token
  OMLX_CACHE_FINE_TAIL_MIN_REMAINDER=0
                                      skip the fine cut when fewer than this many
                                      tokens follow the fine boundary. Off by
                                      default: splitting a chunk costs one fixed
                                      chunk launch whatever the split ratio, so a
                                      small remainder is not the expensive case
                                      (see REPORT.md section 4)
  OMLX_CACHE_FINE_TAIL_MAX_PENDING_MB=192
                                      skip the fine cut while the boundary
                                      snapshot writer already holds at least this
                                      many MB unwritten; 0 disables the check
"""

from __future__ import annotations

import logging
import os
import time
import weakref

logger = logging.getLogger(__name__)

_INSTALLED = False
_FINE = 0
_COARSE = 0
_MIN_GAIN = 0
_MIN_REMAINDER = 0
_MAX_PENDING_BYTES = 0
_SCHEDULER_REF = None
# Fine boundaries this patch deliberately cut a chunk at, so the emission
# predicate can allow a snapshot there and nowhere else off the coarse grid.
_FINE_TARGETS: set[int] = set()
_FINE_TARGETS_MAX = 64
_STATS = {
    "fine_cuts": 0,
    "skipped_min_gain": 0,
    "skipped_min_remainder": 0,
    "skipped_pending": 0,
    "coarse_first": 0,
    "kept_without_checkpoint": 0,
    "coarse_rejects": 0,
    "snapshot_calls": 0,
    "snapshot_ms_total": 0.0,
    "snapshot_ms_max": 0.0,
}


def _parse(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or 0)
    except ValueError:
        return default


def config() -> dict:
    """Return the installed configuration. Empty when not installed."""
    if not _INSTALLED:
        return {}
    return {
        "fine": _FINE,
        "coarse": _COARSE,
        "min_gain": _MIN_GAIN,
        "min_remainder": _MIN_REMAINDER,
        "max_pending_bytes": _MAX_PENDING_BYTES,
    }


def stats() -> dict:
    """Counters for the workbench run. Cheap, no locks."""
    return dict(_STATS)


def _note_fine_target(token_count: int) -> None:
    """Remember a chosen fine boundary; bounded, no lock (set ops are atomic)."""
    if len(_FINE_TARGETS) >= _FINE_TARGETS_MAX:
        _FINE_TARGETS.clear()
    _FINE_TARGETS.add(int(token_count))


def _pending_snapshot_bytes() -> int:
    """Bytes the boundary snapshot writer still holds. -1 when unknown."""
    if _SCHEDULER_REF is None:
        return -1
    sched = _SCHEDULER_REF()
    if sched is None:
        return -1
    store = getattr(sched, "_boundary_snapshot_store", None)
    if store is None:
        return -1
    try:
        return int(store.pending_bytes)
    except Exception:  # noqa: BLE001
        return -1


def install() -> bool:
    """Idempotent. Returns False and leaves the stock path on any precondition failure."""
    global _INSTALLED, _FINE, _COARSE, _MIN_GAIN, _MIN_REMAINDER, _MAX_PENDING_BYTES
    if _INSTALLED:
        return True

    fine = _parse("OMLX_CACHE_FINE_TAIL", 0)
    coarse = _parse("OMLX_CACHE_COARSE_CHUNK", 2048)
    min_gain = max(0, _parse("OMLX_CACHE_FINE_TAIL_MIN_GAIN", 384))
    min_remainder = max(0, _parse("OMLX_CACHE_FINE_TAIL_MIN_REMAINDER", 0))
    max_pending_mb = max(0, _parse("OMLX_CACHE_FINE_TAIL_MAX_PENDING_MB", 192))
    if fine <= 0:
        return False
    if coarse <= 0 or fine >= coarse or coarse % fine != 0:
        logger.warning(
            "cache-boundary: coarse chunk %d must be a proper multiple of fine %d; skipped",
            coarse,
            fine,
        )
        return False

    try:
        import omlx.scheduler as sch
        from omlx.cache.prefix_cache import BlockAwarePrefixCache
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache-boundary: omlx not importable: %s", exc)
        return False

    required = (
        (sch, "clamp_prefill_chunk_to_boundary"),
        (sch, "should_emit_prefill_boundary"),
        (sch.Scheduler, "_enlarge_block_size_for_arrays_cache"),
        (sch.Scheduler, "_maybe_capture_boundary_snapshot"),
        (sch.Scheduler, "_on_prefill_boundary_snapshot"),
        (sch.Scheduler, "_enable_mtp_boundary_alignment"),
        (BlockAwarePrefixCache, "_commit_split_gdn_checkpoint"),
    )
    for owner, name in required:
        if not hasattr(owner, name):
            logger.warning("cache-boundary: %s has no %s; skipped", owner, name)
            return False

    max_pending_bytes = max_pending_mb * 1024 * 1024

    # ------------------------------------------------------------------ 1
    # Prefill chunking: coarse for the body, at most one extra fine cut in
    # the tail, and never a chunk that steps over a coarse boundary.
    stock_clamp = sch.clamp_prefill_chunk_to_boundary

    def clamp(chunk_tokens: int, *, cache_tokens: int, block_size: int) -> int:
        if block_size <= 0 or chunk_tokens <= 0:
            return max(1, chunk_tokens)
        if chunk_tokens >= coarse:
            # Body of the prompt: unchanged 2048-token forwards, cut on the
            # coarse grid.
            return stock_clamp(chunk_tokens, cache_tokens=cache_tokens, block_size=coarse)

        end = cache_tokens + chunk_tokens
        next_coarse = ((cache_tokens // coarse) + 1) * coarse
        if next_coarse < end:
            # THE ROUND-4a BUG. A block that ends on the coarse grid must have
            # a committed checkpoint or the store truncates the whole chain
            # (see commit() below), and a checkpoint exists only where a chunk
            # ended. Stop here; the next call handles the rest of the tail.
            _STATS["coarse_first"] += 1
            return next_coarse - cache_tokens

        last_fine = (end // fine) * fine
        if last_fine <= cache_tokens or last_fine == end:
            # Nothing to cut, or the chunk already ends on the fine grid.
            return chunk_tokens
        if last_fine == next_coarse:
            # The stock cut is already the fine cut. Free.
            return last_fine - cache_tokens

        # An extra cut here costs one chunk launch plus one 110 MiB boundary
        # snapshot on the inference thread. Only take it when it buys enough
        # cached tokens on the next turn, and not while the snapshot writer is
        # still draining an earlier store.
        last_coarse = (end // coarse) * coarse
        gain = last_fine - max(cache_tokens, last_coarse)
        if gain < min_gain:
            _STATS["skipped_min_gain"] += 1
            return chunk_tokens
        if min_remainder and (end - last_fine) < min_remainder:
            _STATS["skipped_min_remainder"] += 1
            return chunk_tokens
        if max_pending_bytes:
            pending = _pending_snapshot_bytes()
            if pending >= max_pending_bytes:
                _STATS["skipped_pending"] += 1
                logger.debug(
                    "cache-boundary: skipping the fine cut at %d, snapshot writer "
                    "holds %.0f MB",
                    last_fine,
                    pending / (1024 * 1024),
                )
                return chunk_tokens
        _STATS["fine_cuts"] += 1
        _note_fine_target(last_fine)
        return last_fine - cache_tokens

    clamp.__wrapped__ = stock_clamp  # type: ignore[attr-defined]
    sch.clamp_prefill_chunk_to_boundary = clamp

    # ------------------------------------------------------------------ 1b
    # Emission stays on the coarse grid plus the fine cuts this patch chose.
    # Without this, a block size of 512 would emit a 110 MiB snapshot at every
    # chunk end whenever the scheduler drops the prefill step to 512 tokens
    # under decode contention (scheduler.py:1542-1550, 5177-5197): four times
    # the stock snapshot rate on exactly the path that is already struggling.
    stock_emit = sch.should_emit_prefill_boundary

    def emit(*, total_tokens: int, block_size: int, last_emitted_tokens: int) -> bool:
        if not stock_emit(
            total_tokens=total_tokens,
            block_size=block_size,
            last_emitted_tokens=last_emitted_tokens,
        ):
            return False
        if total_tokens % coarse == 0:
            return True
        return total_tokens in _FINE_TARGETS

    emit.__wrapped__ = stock_emit  # type: ignore[attr-defined]
    sch.should_emit_prefill_boundary = emit

    # ------------------------------------------------------------------ 2
    # Block size: chosen post-load from the cache layout. Let the stock logic
    # run, then lower it. Matching, storing and restoring all read
    # config.paged_cache_block_size and all want the fine value. This is also
    # where the scheduler instance is captured, for the writer-backlog check.
    stock_enlarge = sch.Scheduler._enlarge_block_size_for_arrays_cache

    def enlarge(self) -> None:
        global _SCHEDULER_REF
        stock_enlarge(self)
        try:
            _SCHEDULER_REF = weakref.ref(self)
        except TypeError:
            _SCHEDULER_REF = None
        current = int(getattr(self.config, "paged_cache_block_size", 0) or 0)
        if current <= 0 or current % fine != 0 or current == fine:
            return
        logger.info(
            "cache-boundary: paged cache block_size %d -> %d (prefill chunks stay at %d, "
            "min_gain=%d min_remainder=%d max_pending=%d MB)",
            current,
            fine,
            coarse,
            min_gain,
            min_remainder,
            max_pending_mb,
        )
        self.config.paged_cache_block_size = fine

    enlarge.__wrapped__ = stock_enlarge  # type: ignore[attr-defined]
    sch.Scheduler._enlarge_block_size_for_arrays_cache = enlarge

    # ------------------------------------------------------------------ 3
    # Decode-time captures stay on the coarse grid. Without this the 110 MiB
    # snapshot would be taken every 512 generated tokens instead of every
    # 2048, against a 512 MB pending-write budget.
    stock_capture = sch.Scheduler._maybe_capture_boundary_snapshot

    def capture(self, request, uid):
        total = int(getattr(request, "num_tokens", 0) or 0)
        if total <= 0 or total % coarse != 0:
            return None
        return stock_capture(self, request, uid)

    capture.__wrapped__ = stock_capture  # type: ignore[attr-defined]
    sch.Scheduler._maybe_capture_boundary_snapshot = capture

    # ------------------------------------------------------------------ 3b
    # Time the snapshot emission. It runs on the inference thread and the
    # cost model has no direct measurement of it, only the 0.29 s difference
    # between two probe turns. One INFO line per snapshot, about 13 per cold
    # prefill and one per warm turn.
    stock_on_boundary = sch.Scheduler._on_prefill_boundary_snapshot

    def on_boundary(self, request_id, snapshot_cache, token_count, *, source="prefill"):
        t0 = time.perf_counter()
        pending_before = _pending_snapshot_bytes()
        try:
            return stock_on_boundary(
                self, request_id, snapshot_cache, token_count, source=source
            )
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            _STATS["snapshot_calls"] += 1
            _STATS["snapshot_ms_total"] += dt
            _STATS["snapshot_ms_max"] = max(_STATS["snapshot_ms_max"], dt)
            logger.info(
                "cache-boundary: snapshot at %d took %.0f ms (writer held %.0f MB, "
                "source=%s)",
                int(token_count),
                dt,
                max(0, pending_before) / (1024 * 1024),
                source,
            )

    on_boundary.__wrapped__ = stock_on_boundary  # type: ignore[attr-defined]
    sch.Scheduler._on_prefill_boundary_snapshot = on_boundary

    # ------------------------------------------------------------------ 4
    # MTP commit alignment follows the capture grid, not the block size.
    def align(self) -> None:
        try:
            self.model._omlx_mtp_commit_align = coarse
        except Exception:  # noqa: BLE001
            pass

    align.__wrapped__ = sch.Scheduler._enable_mtp_boundary_alignment  # type: ignore[attr-defined]
    sch.Scheduler._enable_mtp_boundary_alignment = align

    # ------------------------------------------------------------------ 5
    # A block whose end is off the coarse grid was never going to have a
    # staged snapshot; keeping its KV costs nothing and the restore endpoint
    # search walks back past it. A coarse-aligned block without one is a real
    # anomaly: keep the stock truncation, and say which boundary it was, which
    # is what the stock log line (block id only) does not.
    stock_commit = BlockAwarePrefixCache._commit_split_gdn_checkpoint

    def commit(
        self,
        boundary_snapshots,
        token_count,
        block_hash,
        layer_cache_types,
        layer_meta_states,
    ) -> bool:
        committed = stock_commit(
            self,
            boundary_snapshots,
            token_count,
            block_hash,
            layer_cache_types,
            layer_meta_states,
        )
        if committed:
            return True
        tc = int(token_count)
        if tc % coarse == 0:
            _STATS["coarse_rejects"] += 1
            logger.warning(
                "cache-boundary: no staged snapshot at coarse boundary %d "
                "(coarse=%d fine=%d); the store will truncate here",
                tc,
                coarse,
                fine,
            )
            return False
        _STATS["kept_without_checkpoint"] += 1
        logger.debug(
            "cache-boundary: block ending at %d kept without a recurrent "
            "checkpoint (restore walks back to the newest endpoint)",
            tc,
        )
        return True

    commit.__wrapped__ = stock_commit  # type: ignore[attr-defined]
    BlockAwarePrefixCache._commit_split_gdn_checkpoint = commit

    _INSTALLED = True
    _FINE, _COARSE = fine, coarse
    _MIN_GAIN, _MIN_REMAINDER = min_gain, min_remainder
    _MAX_PENDING_BYTES = max_pending_bytes
    logger.info(
        "cache-boundary: fine store grid %d, coarse chunk %d, min_gain %d",
        fine,
        coarse,
        min_gain,
    )
    return True


def uninstall() -> bool:
    """Restore the stock functions. For tests."""
    global _INSTALLED, _FINE, _COARSE, _MIN_GAIN, _MIN_REMAINDER
    global _MAX_PENDING_BYTES, _SCHEDULER_REF
    if not _INSTALLED:
        return False
    import omlx.scheduler as sch
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    def _unwrap(owner, name) -> None:
        current = getattr(owner, name)
        stock = getattr(current, "__wrapped__", None)
        if stock is not None:
            setattr(owner, name, stock)

    _unwrap(sch, "clamp_prefill_chunk_to_boundary")
    _unwrap(sch, "should_emit_prefill_boundary")
    _unwrap(sch.Scheduler, "_enlarge_block_size_for_arrays_cache")
    _unwrap(sch.Scheduler, "_maybe_capture_boundary_snapshot")
    _unwrap(sch.Scheduler, "_on_prefill_boundary_snapshot")
    _unwrap(sch.Scheduler, "_enable_mtp_boundary_alignment")
    _unwrap(BlockAwarePrefixCache, "_commit_split_gdn_checkpoint")
    _INSTALLED = False
    _FINE = _COARSE = _MIN_GAIN = _MIN_REMAINDER = _MAX_PENDING_BYTES = 0
    _SCHEDULER_REF = None
    _FINE_TARGETS.clear()
    return True
