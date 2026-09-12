# SPDX-License-Identifier: Apache-2.0
"""Finer prefix-cache flooring without finer prefill chunks.

oMLX uses one number, ``config.paged_cache_block_size``, for two unrelated
jobs. It is the granularity at which a warm prefix can be restored, and it is
the width at which prefill forwards are cut so a GDN boundary snapshot can be
taken (``scheduler.py:3686, 3830, 5498, 5602`` via
``prefill_boundaries.clamp_prefill_chunk_to_boundary`` and
``should_emit_prefill_boundary``). ``Scheduler._enlarge_block_size_for_arrays_cache``
(``scheduler.py:2850-2901``) pins it to 2048 for this model because 2048 is the
chunk width the kernels want. The cost is paid on every warm turn: a cached
prefix floors to a 2048 multiple, so a follow-up turn re-prefills the trailing
partial block even when that text has not changed. Measured on this box's
traffic, the discarded tail is a median of 1155 tokens per store.

This patch separates the two. Prefill still runs 2048-token forwards for the
body of a prompt. Only the trailing region shorter than one full chunk is cut
once more, at the largest ``fine`` multiple it contains, which emits one extra
boundary snapshot there. The paged block size drops to ``fine`` so the prefix
cache can actually keep that boundary. Blocks between snapshots carry no GDN
checkpoint; restore already handles that by walking back to the newest block
that has one (``cache/prefix_cache.py:3311-3330``).

Cost: one extra forward launch and one extra 110 MiB snapshot per prompt, and
4x the paged block count. Not the 13.8 GiB of snapshots a uniform 512-token
block size over a 64k prompt would need.

Install at IMPORT time, before the engine starts: it rebinds module-level
functions and wraps one Scheduler method, and touches no model instance.
Bootstrap hook: OMLX_ROUND2_IMPORT_PATCHES=".../cache-reuse/patch.py:install"

Env:
  OMLX_CACHE_FINE_BOUNDARY=512   fine boundary in tokens, 0/unset disables
  OMLX_CACHE_COARSE_CHUNK=2048   chunk width kept for the body of a prompt
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def _parse(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or 0)
    except ValueError:
        return 0


def install() -> bool:
    """Idempotent. Returns False and leaves the stock path on any precondition failure."""
    global _INSTALLED
    if _INSTALLED:
        return True

    fine = _parse("OMLX_CACHE_FINE_BOUNDARY", 0)
    coarse = _parse("OMLX_CACHE_COARSE_CHUNK", 2048)
    if fine <= 0:
        return False
    if coarse <= 0 or coarse % fine != 0 or fine == coarse:
        logger.warning(
            "cache-reuse: coarse chunk %d must be a proper multiple of fine %d; skipped",
            coarse, fine,
        )
        return False

    try:
        import omlx.scheduler as sch
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache-reuse: omlx.scheduler not importable: %s", exc)
        return False

    for name in ("clamp_prefill_chunk_to_boundary", "should_emit_prefill_boundary",
                 "Scheduler"):
        if not hasattr(sch, name):
            logger.warning("cache-reuse: omlx.scheduler has no %s; skipped", name)
            return False

    stock_clamp = sch.clamp_prefill_chunk_to_boundary

    def clamp(chunk_tokens: int, *, cache_tokens: int, block_size: int) -> int:
        """Cut at coarse boundaries for the body, once at a fine boundary in the tail."""
        if block_size <= 0 or chunk_tokens <= 0:
            return max(1, chunk_tokens)
        if chunk_tokens >= coarse:
            # Body of the prompt: unchanged 2048-token forwards.
            return stock_clamp(
                chunk_tokens, cache_tokens=cache_tokens, block_size=coarse
            )
        # Tail. ``chunk_tokens`` is the whole remainder here, since the caller
        # passes min(prefill_step_size, remaining) and the step size is coarse.
        # Stop once, at the last fine boundary the remainder contains, so the
        # store keeps floor(L / fine) * fine tokens instead of
        # floor(L / coarse) * coarse. An adaptive step size can land
        # cache_tokens off a fine boundary; deriving the target from the
        # absolute end keeps that case correct.
        last_fine = ((cache_tokens + chunk_tokens) // fine) * fine
        if last_fine > cache_tokens:
            return last_fine - cache_tokens
        return chunk_tokens

    clamp.__wrapped__ = stock_clamp  # type: ignore[attr-defined]
    sch.clamp_prefill_chunk_to_boundary = clamp

    # The block size is chosen post-load from the model's cache layout. Let the
    # stock logic run, then lower it: emission and the store's alignment gate
    # (``scheduler.py:7281-7285``) both read config.paged_cache_block_size, and
    # both want the fine value.
    stock_enlarge = sch.Scheduler._enlarge_block_size_for_arrays_cache

    def enlarge(self) -> None:
        stock_enlarge(self)
        current = int(getattr(self.config, "paged_cache_block_size", 0) or 0)
        if current <= 0 or current % fine != 0 or current == fine:
            return
        logger.info(
            "cache-reuse: paged cache block_size %d -> %d (prefill chunks stay at %d)",
            current, fine, coarse,
        )
        self.config.paged_cache_block_size = fine

    enlarge.__wrapped__ = stock_enlarge  # type: ignore[attr-defined]
    sch.Scheduler._enlarge_block_size_for_arrays_cache = enlarge

    _INSTALLED = True
    logger.info("cache-reuse: fine boundary %d, coarse chunk %d", fine, coarse)
    return True


def uninstall() -> bool:
    """Restore the stock functions. For tests."""
    global _INSTALLED
    if not _INSTALLED:
        return False
    import omlx.scheduler as sch

    sch.clamp_prefill_chunk_to_boundary = (
        sch.clamp_prefill_chunk_to_boundary.__wrapped__
    )
    sch.Scheduler._enlarge_block_size_for_arrays_cache = (
        sch.Scheduler._enlarge_block_size_for_arrays_cache.__wrapped__
    )
    _INSTALLED = False
    return True
