"""The six-turn probe from the cache-boundary report, as a regression test.

The measured arms are in ``engine/patches/round4/cache-boundary/REPORT.md``.
Under the overlay's single 2048 grid the six turns recomputed 11617 tokens they
had already processed; with the block grid at 512 and a snapshot at each prompt
end they recompute 8545, and the warm median went from 2.59 s to 2.10 s.

This test asserts the token counts, not the seconds. The counts are what the
policy controls and they are exact; the seconds belong to the benchmark.
"""

from __future__ import annotations

import pytest

from tests.cache.conftest import build_cache, run_turn
from tests.cache.fakes import fold, tokens_for

PROMPTS = (25043, 26481, 27919, 29357, 30795, 32233)

FINE_CACHED = (0, 24576, 26112, 27648, 29184, 30720)
"""What the 512 grid with a prompt-end snapshot restores per turn."""

COARSE_CACHED = (0, 24576, 24576, 26624, 28672, 30720)
"""What a snapshot grid welded to the block grid restores. The overlay."""


@pytest.fixture(scope="module")
def conversation() -> list[list[int]]:
    """Six turns, each extending the last, as a chat with a growing history."""
    longest = tokens_for(PROMPTS[-1], seed=11)
    return [longest[:length] for length in PROMPTS]


def fine_cache(tmp_path):
    return build_cache(
        tmp_path,
        block_tokens=512,
        snapshot_grid=2048,
        chunk_tokens=2048,
        contended_chunk_tokens=512,
        fine_min_gain_tokens=384,
    )


def coarse_cache(tmp_path):
    """The overlay's shape: one grid, no snapshot at the prompt end."""
    return build_cache(
        tmp_path,
        block_tokens=2048,
        snapshot_grid=2048,
        chunk_tokens=2048,
        contended_chunk_tokens=2048,
        snapshot_at_prompt_end=False,
    )


def replay(cache, conversation):
    return [run_turn(cache, tokens) for tokens in conversation]


def test_fine_grid_resumes_at_the_previous_prompt_end(tmp_path, conversation):
    cache, _, store = fine_cache(tmp_path)
    try:
        turns = replay(cache, conversation)
        assert tuple(turn.matched for turn in turns) == FINE_CACHED
        for tokens, turn in zip(conversation, turns):
            assert turn.state.recurrent == fold(0, tokens)
            assert turn.state.kv.shape[0] == len(tokens)
    finally:
        store.close()


def test_fine_grid_recomputes_8545_warm_tokens(tmp_path, conversation):
    cache, _, store = fine_cache(tmp_path)
    try:
        turns = replay(cache, conversation)
        warm = sum(
            len(tokens) - turn.matched
            for tokens, turn in list(zip(conversation, turns))[1:]
        )
        assert warm == 8545
    finally:
        store.close()


def test_coarse_grid_recomputes_11617_warm_tokens(tmp_path, conversation):
    """The number the fine grid has to beat, reproduced from the same replay."""
    cache, _, store = coarse_cache(tmp_path)
    try:
        turns = replay(cache, conversation)
        assert tuple(turn.matched for turn in turns) == COARSE_CACHED
        warm = sum(
            len(tokens) - turn.matched
            for tokens, turn in list(zip(conversation, turns))[1:]
        )
        assert warm == 11617
    finally:
        store.close()


def test_every_coarse_multiple_inside_a_suffix_still_ends_a_chunk(
    tmp_path, conversation
):
    """Turn 3 is the one that broke: its suffix steps over 26624."""
    cache, _, store = fine_cache(tmp_path)
    try:
        turns = replay(cache, conversation)
        assert 26624 in turns[2].chunk_ends
        assert turns[2].chunk_ends == (26624, 27648, 27919)
        assert turns[2].snapshots == (26624, 27648)
        assert cache.counters.snapshots_failed == 0
        assert cache.counters.chain_truncations == 0
    finally:
        store.close()


def test_the_fine_tail_costs_one_extra_snapshot_per_warm_turn(
    tmp_path, conversation
):
    fine, _, fine_store = fine_cache(tmp_path / "fine")
    coarse, _, coarse_store = coarse_cache(tmp_path / "coarse")
    try:
        replay(fine, conversation)
        replay(coarse, conversation)
        extra = (
            fine.counters.snapshots_written - coarse.counters.snapshots_written
        )
        # Four, not five: turn 5's prompt end rounds onto 30720, which is a
        # grid multiple already, so that turn pays nothing extra.
        assert fine.counters.fine_snapshots == 4
        assert extra == 4
    finally:
        fine_store.close()
        coarse_store.close()


def test_the_fine_grid_stores_more_blocks_but_not_more_bytes_per_token(
    tmp_path, conversation
):
    """Block count is free; resolution is not. 48 of 512 cost what 12 of 2048 do."""
    fine, fine_codec, fine_store = fine_cache(tmp_path / "fine")
    coarse, coarse_codec, coarse_store = coarse_cache(tmp_path / "coarse")
    try:
        replay(fine, conversation)
        replay(coarse, conversation)
        assert fine.counters.blocks_written > coarse.counters.blocks_written
        assert fine_store.flush(10.0) and coarse_store.flush(10.0)
        # Four times the records and under a tenth more bytes: the difference
        # is per-record headers and four extra snapshots, not payload.
        ratio = fine_store.stats.bytes_written / coarse_store.stats.bytes_written
        assert 1.0 < ratio < 1.10
    finally:
        fine_store.close()
        coarse_store.close()
