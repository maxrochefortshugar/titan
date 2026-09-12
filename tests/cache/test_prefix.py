"""Matching, the two grids, chunk planning, sharing and every degrade path."""

from __future__ import annotations

import pytest

from titan.core.types import BlockHash
from titan.adapters.cache.prefix import BlockPrefixCache
from titan.adapters.cache.store import TwoTierStateStore

from tests.cache.conftest import BLOCK, GRID, build_cache, run_turn
from tests.cache.fakes import FakeCodec, FakeState, fold, signature, tokens_for


# -- matching ------------------------------------------------------------


def test_cold_lookup_is_a_miss(cache_bundle):
    cache, _, _ = cache_bundle
    match = cache.lookup(tokens_for(100))
    assert match.matched_tokens == 0
    assert match.block_hashes == ()
    assert match.snapshot_id is None
    assert match.tier == "none"


def test_second_turn_matches_the_stored_prompt_end(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(100)
    first = run_turn(cache, tokens)
    assert first.matched == 0
    assert first.snapshots[-1] == 96  # 100 rounded down to the 8-token grid

    follow_up = tokens + tokens_for(20, seed=1)
    second = run_turn(cache, follow_up)
    assert second.matched == 96
    # Restored, then prefilled the 24-token tail: the same state as recomputing.
    assert second.state.recurrent == fold(0, follow_up)
    assert second.state.snapshots[96] == fold(0, tokens[:96])
    assert cache.counters.hits == 1


def test_partial_match_stops_where_the_prompts_diverge(cache_bundle):
    cache, _, _ = cache_bundle
    shared = tokens_for(80)
    run_turn(cache, shared + tokens_for(40, seed=2))
    other = shared[:37] + [shared[37] + 1] + tokens_for(60, seed=3)
    match = cache.lookup(other)
    assert match.matched_tokens == 32  # block 5 covers token 37, so four blocks
    assert len(match.block_hashes) == 4


def test_match_never_covers_the_whole_prompt(cache_bundle):
    """Prefill needs a token to produce logits from, so the last block is out.

    The prompt is 64 tokens and 64 is both a block end and a snapshot point,
    but resubmitting it verbatim resolves to 32, the newest snapshot at or
    before the deepest block the match is allowed to use.
    """
    cache, _, _ = cache_bundle
    tokens = tokens_for(64)
    run_turn(cache, tokens)
    match = cache.lookup(tokens)
    assert match.matched_tokens == 32
    assert match.matched_tokens < len(tokens)


def test_match_falls_back_to_the_newest_snapshot_behind_the_kv_frontier(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(64)
    state = FakeState()
    state.prefill(tokens[:16], snapshot=True)
    state.prefill(tokens[16:48])  # KV runs on, no snapshot staged
    cache.store(tokens[:48], state, [16, 48])

    match = cache.lookup(tokens)
    assert match.matched_tokens == 16
    assert len(match.block_hashes) == 2
    assert cache.counters.snapshots_failed == 1
    # The chain truncated at the snapshot, so nothing past it was recorded.
    assert cache.counters.kv_only_blocks == 0
    assert cache.counters.chain_truncations == 1


def test_restored_recurrent_state_equals_recomputation(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(200)
    run_turn(cache, tokens)
    follow_up = tokens + tokens_for(30, seed=4)
    result = run_turn(cache, follow_up)
    assert result.matched == 200
    assert result.state.recurrent == fold(0, follow_up)
    assert result.state.kv.shape[0] == len(follow_up)


# -- planning ------------------------------------------------------------


def production_cache():
    return build_cache(block_tokens=512, snapshot_grid=2048, chunk_tokens=2048,
                       contended_chunk_tokens=512, fine_min_gain_tokens=384)


def test_every_grid_multiple_inside_the_suffix_ends_a_chunk():
    """The round-4a regression: a chunk that steps over 26624 and never lands."""
    cache, _, store = production_cache()
    try:
        ends = cache.plan_chunks(26112, 27919, contended=False)
        assert 26624 in ends
        assert ends == (26624, 27648, 27919)
        assert cache.snapshot_boundaries(26112, 27919) == (26624, 27648)
    finally:
        store.close()


def test_contended_chunks_do_not_multiply_snapshots():
    cache, _, store = production_cache()
    try:
        quiet = cache.snapshot_boundaries(24576, 27919, contended=False)
        busy = cache.snapshot_boundaries(24576, 27919, contended=True)
        assert quiet == busy == (26624, 27648)
        ends = cache.plan_chunks(24576, 27919, contended=True)
        assert len(ends) > len(busy)  # many chunks, still two snapshots
        assert set(busy) <= set(ends)
    finally:
        store.close()


def test_fine_cut_is_skipped_when_the_gain_is_too_small():
    cache, _, store = production_cache()
    try:
        # 24576 is a grid point; the prompt ends 200 tokens later, so the fine
        # cut would buy nothing and costs a chunk plus a snapshot.
        assert cache.snapshot_boundaries(20480, 24776) == (22528, 24576)
        assert cache.plan_chunks(20480, 24776, False) == (22528, 24576, 24776)
    finally:
        store.close()


def test_fine_cut_is_skipped_while_the_writer_is_backed_up(tmp_path):
    cache, _, store = build_cache(
        tmp_path,
        block_tokens=512,
        snapshot_grid=2048,
        chunk_tokens=2048,
        contended_chunk_tokens=512,
        fine_min_gain_tokens=384,
        fine_max_pending_bytes=1024,
        store_kwargs=dict(pending_budget_bytes=1024 * 1024, max_stall_s=0.01),
    )
    try:
        assert cache.snapshot_boundaries(26112, 27919) == (26624, 27648)
        store.set_writer_paused(True)
        store.put_block(BlockHash(b"\x01" * 32), b"x" * 4096)
        assert store.pending_bytes() > 1024
        assert cache.snapshot_boundaries(26112, 27919) == (26624,)
        store.set_writer_paused(False)
        store.flush(2.0)
    finally:
        store.close()


def test_plan_is_empty_when_nothing_is_left_to_prefill():
    cache, _, store = production_cache()
    try:
        assert cache.plan_chunks(1024, 1024, False) == ()
        assert cache.snapshot_boundaries(1024, 1024) == ()
    finally:
        store.close()


def test_grid_must_be_a_multiple_of_the_block():
    store = TwoTierStateStore(signature())
    try:
        with pytest.raises(ValueError, match="multiple"):
            BlockPrefixCache(
                store, FakeCodec(signature()), block_tokens=512, snapshot_grid=1000
            )
        with pytest.raises(ValueError, match="multiple"):
            BlockPrefixCache(
                store,
                FakeCodec(signature()),
                block_tokens=512,
                snapshot_grid=2048,
                chunk_tokens=1000,
            )
    finally:
        store.close()


# -- sharing and reference counting --------------------------------------


def test_a_shared_prefix_is_stored_once(cache_bundle):
    cache, codec, _ = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    written = cache.counters.blocks_written
    exports = codec.exports

    run_turn(cache, tokens + tokens_for(16, seed=5))
    assert cache.counters.blocks_deduped >= written
    # Only the new tail was serialised.
    assert codec.exports - exports < written


def test_two_leases_on_one_prefix_hold_it_until_the_last_release():
    cache, _, store = build_cache(store_kwargs=dict(hot_budget_bytes=6000))
    try:
        tokens = tokens_for(96)
        run_turn(cache, tokens)
        match = cache.lookup(tokens + [7])
        assert match.matched_tokens > 0

        first = cache.reserve(match)
        second = cache.reserve(match)
        head = match.block_hashes[0]
        assert cache.ref_count(head) == 2

        # A flood of unrelated traffic cannot evict a pinned block.
        for index in range(30):
            store.put_block(BlockHash(bytes([index]) * 32), b"x" * 512)
        assert store.get_block(head) is not None

        cache.release(first)
        assert cache.ref_count(head) == 1
        for index in range(30, 60):
            store.put_block(BlockHash(bytes([index]) * 32), b"x" * 512)
        assert store.get_block(head) is not None

        cache.release(second)
        assert cache.ref_count(head) == 0
        for index in range(60, 90):
            store.put_block(BlockHash(bytes([index]) * 32), b"x" * 512)
        assert store.get_block(head) is None
    finally:
        store.close()


def test_releasing_twice_is_a_no_op(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    match = cache.lookup(tokens + [1])
    lease = cache.reserve(match)
    cache.release(lease)
    cache.release(lease)
    assert cache.ref_count(match.block_hashes[0]) == 0
    assert cache.counters.leases_open == 0


def test_two_concurrent_requests_on_one_prefix_both_restore(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)

    left_tokens = tokens + tokens_for(12, seed=6)
    right_tokens = tokens + tokens_for(12, seed=7)
    left_match = cache.lookup(left_tokens)
    right_match = cache.lookup(right_tokens)
    left_lease = cache.reserve(left_match)
    right_lease = cache.reserve(right_match)

    left_state, right_state = FakeState(), FakeState()
    assert cache.restore(left_match, left_state) == 96
    assert cache.restore(right_match, right_state) == 96
    assert left_state.recurrent == right_state.recurrent == fold(0, tokens[:96])

    cache.release(left_lease)
    cache.release(right_lease)
    assert cache.counters.leases_open == 0


# -- degrade paths -------------------------------------------------------


def test_restore_shortens_when_a_block_went_missing(cache_bundle):
    cache, _, store = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    follow_up = tokens + tokens_for(8, seed=8)
    match = cache.lookup(follow_up)
    assert match.matched_tokens == 96

    # Drop the fifth block out from under the match.
    victim = "b:" + bytes(match.block_hashes[4]).hex()
    store._hot.pop(victim, None)
    store._disk.pop(victim, None)

    state = FakeState()
    restored = cache.restore(match, state)
    assert restored == 32
    assert state.recurrent == fold(0, tokens[:32])
    assert cache.counters.restore_degraded == 1


def test_restore_returns_zero_when_the_snapshot_is_gone(cache_bundle):
    cache, _, store = cache_bundle
    tokens = tokens_for(64)
    run_turn(cache, tokens)
    match = cache.lookup(tokens + [3])
    for key in list(store._hot):
        if key.startswith("s:"):
            store._hot.pop(key)
    state = FakeState()
    assert cache.restore(match, state) == 0
    assert state.length == 0
    assert cache.counters.restore_degraded >= 1


def test_a_boundary_whose_snapshot_failed_is_dropped(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(64)
    state = FakeState()
    state.prefill(tokens[:32], snapshot=True)
    state.prefill(tokens[32:64])
    cache.store(tokens, state, [32, 64])
    assert cache.counters.snapshots_written == 1
    assert cache.counters.snapshots_failed == 1
    assert cache.lookup(tokens + [1]).matched_tokens == 32


def test_a_codec_that_raises_on_import_does_not_reach_the_engine(cache_bundle):
    cache, codec, _ = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    match = cache.lookup(tokens + [2])

    def boom(*args, **kwargs):
        raise RuntimeError("device fell over")

    codec.import_blocks = boom  # type: ignore[method-assign]
    assert cache.restore(match, FakeState()) == 0
    assert cache.counters.restore_aborted == 1


def test_lookup_ignores_an_index_entry_whose_bytes_are_gone(cache_bundle):
    cache, _, store = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    store._hot.clear()
    store._disk.clear()
    assert cache.lookup(tokens + [4]).matched_tokens == 0
    assert cache.prune() > 0
    assert cache.lookup(tokens + [4]).matched_tokens == 0


# -- observability -------------------------------------------------------


def test_stats_carry_both_layers(cache_bundle):
    cache, _, _ = cache_bundle
    tokens = tokens_for(96)
    run_turn(cache, tokens)
    run_turn(cache, tokens + tokens_for(16, seed=9))
    values = cache.stats()
    assert values["prefix.hits"] == 1
    assert values["prefix.lookups"] == 2
    assert values["prefix.hit_rate"] == 0.5
    assert values["prefix.matched_tokens"] == 96
    assert values["prefix.recompute_tokens"] == 96 + 16
    assert values["prefix.snapshots_written"] >= 3
    assert values["prefix.snapshot_seconds"] >= 0.0
    assert values["store.hot_bytes"] > 0
    assert values["store.pending_bytes"] == 0


def test_both_adapters_satisfy_their_ports(cache_bundle):
    """The ports are runtime checkable, so this is a real structural check."""
    from titan.core.ports import KVStateStore, PrefixCache

    cache, _, store = cache_bundle
    assert isinstance(store, KVStateStore)
    assert isinstance(cache, PrefixCache)
