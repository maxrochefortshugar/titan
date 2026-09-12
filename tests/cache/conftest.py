"""Shared rigging: a cache with a tiny block size, and one turn of the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from titan.adapters.cache.prefix import BlockPrefixCache
from titan.adapters.cache.store import TwoTierStateStore

from tests.cache.fakes import FakeCodec, FakeState, signature

BLOCK = 8
GRID = 32


def build_cache(tmp_path=None, **kwargs) -> tuple[BlockPrefixCache, FakeCodec, TwoTierStateStore]:
    """A cache on an 8-token block and a 32-token grid, RAM only by default."""
    block = kwargs.pop("block_tokens", BLOCK)
    sig = signature(block_tokens=block)
    store_kwargs = kwargs.pop("store_kwargs", {})
    store = TwoTierStateStore(
        sig,
        ssd_dir="" if tmp_path is None else str(tmp_path),
        hot_budget_bytes=store_kwargs.pop("hot_budget_bytes", 8 * 1024**2),
        **store_kwargs,
    )
    codec = FakeCodec(sig)
    defaults = dict(
        block_tokens=block,
        snapshot_grid=GRID,
        chunk_tokens=GRID,
        contended_chunk_tokens=block,
        fine_min_gain_tokens=0,
    )
    defaults.update(kwargs)
    cache = BlockPrefixCache(store, codec, **defaults)
    return cache, codec, store


@pytest.fixture
def cache_bundle():
    cache, codec, store = build_cache()
    yield cache, codec, store
    store.close()


@dataclass
class TurnResult:
    matched: int
    chunk_ends: tuple[int, ...]
    snapshots: tuple[int, ...]
    state: FakeState
    match_tokens: int


def run_turn(
    cache: BlockPrefixCache,
    tokens: Sequence[int],
    *,
    contended: bool = False,
    state: FakeState | None = None,
    keep_lease: bool = False,
) -> TurnResult:
    """The exact call sequence the engine is meant to use, start to finish."""
    state = FakeState() if state is None else state
    match = cache.match(tokens)
    lease = cache.reserve(match)
    restored = cache.restore(match, state)
    ends = cache.plan_chunks(restored, len(tokens), contended)
    snapshots = cache.snapshot_boundaries(restored, len(tokens), contended)
    position = restored
    wanted = set(snapshots)
    for end in ends:
        state.prefill(tokens[position:end], snapshot=end in wanted)
        position = end
    cache.commit(tokens, state, snapshots)
    if not keep_lease:
        cache.release(lease)
    return TurnResult(
        matched=restored,
        chunk_ends=ends,
        snapshots=snapshots,
        state=state,
        match_tokens=match.matched_tokens,
    )
