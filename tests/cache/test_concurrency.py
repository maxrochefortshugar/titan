"""Two requests on one prefix, on two threads, with the writer draining.

The engine is single threaded and drives the cache from the scheduler, so this
is not the normal path. It is here because the store's writer is real and the
lock ordering between the hot tier, the pending queue and the index is the kind
of thing that is fine until a flush lands mid-eviction.
"""

from __future__ import annotations

import threading

from tests.cache.conftest import build_cache, run_turn
from tests.cache.fakes import FakeState, fold, tokens_for


def test_concurrent_share_and_release(tmp_path):
    cache, _, store = build_cache(tmp_path)
    try:
        shared = tokens_for(240, seed=21)
        run_turn(cache, shared)
        assert store.flush(5.0)

        results: dict[int, int] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(index: int) -> None:
            try:
                tokens = shared + tokens_for(24, seed=100 + index)
                match = cache.match(tokens)
                barrier.wait(timeout=5)
                lease = cache.reserve(match)
                state = FakeState()
                restored = cache.restore(match, state)
                state.prefill(tokens[restored:], snapshot=True)
                cache.commit(tokens, state, [len(tokens)])
                cache.release(lease)
                results[index] = restored
                assert state.recurrent == fold(0, tokens)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors
        assert set(results.values()) == {240}
        assert cache.counters.leases_open == 0
        # One prefix, four readers, and the shared blocks were written once.
        assert cache.counters.blocks_deduped >= 4 * 30
        assert store.flush(5.0)
    finally:
        store.close()


def test_release_frees_the_last_reference_only(tmp_path):
    cache, _, store = build_cache(tmp_path)
    try:
        tokens = tokens_for(96, seed=22)
        run_turn(cache, tokens)
        match = cache.match(tokens + [5])
        leases = [cache.reserve(match) for _ in range(3)]
        head = match.block_hashes[0]
        assert cache.ref_count(head) == 3
        for index, lease in enumerate(leases):
            cache.release(lease)
            assert cache.ref_count(head) == 2 - index
        assert cache.counters.leases_open == 0
    finally:
        store.close()
