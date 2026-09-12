"""The codec seam against real mlx arrays, at a size that fits in a cache line.

Two properties are worth a device for. Serialisation happens on the thread that
called the cache, never on the writer, because mlx is not safe to touch from a
background thread; and a restored array is evaluated, so it does not stay
attached to whatever buffer it was built from.
"""

from __future__ import annotations

import io
import threading

import mlx.core as mx
import numpy as np

from titan.adapters.cache.prefix import BlockPrefixCache
from titan.adapters.cache.store import TwoTierStateStore

from tests.cache.fakes import signature

BLOCK = 4


class MlxState:
    """A slice of a sequence: KV rows on device, one recurrent vector."""

    def __init__(self) -> None:
        self.kv = mx.zeros((0, 2), dtype=mx.float32)
        self.recurrent = mx.zeros((2,), dtype=mx.float32)
        self.length = 0
        self.snapshots: dict[int, mx.array] = {}

    def prefill(self, tokens, *, snapshot: bool = False) -> None:
        rows = mx.array(
            [[float(token), float(token) * 0.5] for token in tokens], dtype=mx.float32
        )
        self.kv = mx.concatenate([self.kv, rows], axis=0)
        for row in rows:
            self.recurrent = self.recurrent * 1.5 + row
        self.length += len(tokens)
        mx.eval(self.kv, self.recurrent)
        if snapshot:
            self.snapshots[self.length] = mx.array(self.recurrent)


class MlxCodec:
    """Serialise on the caller's thread, evaluate on the way back in."""

    def __init__(self, sig) -> None:
        self._signature = sig
        self.export_threads: set[int] = set()

    @property
    def signature(self):
        return self._signature

    def _dump(self, array: mx.array) -> bytes:
        self.export_threads.add(threading.get_ident())
        buffer = io.BytesIO()
        np.save(buffer, np.array(array, copy=True), allow_pickle=False)
        return buffer.getvalue()

    def _load(self, payload: bytes) -> mx.array:
        array = mx.array(np.load(io.BytesIO(payload), allow_pickle=False))
        mx.eval(array)
        return array

    def export_blocks(self, state: MlxState, start: int, end: int) -> bytes:
        return self._dump(state.kv[start:end])

    def import_blocks(self, state: MlxState, start: int, end: int, payload: bytes) -> None:
        state.kv = mx.concatenate([state.kv, self._load(payload)], axis=0)
        mx.eval(state.kv)

    def export_snapshot(self, state: MlxState, length: int) -> bytes:
        return self._dump(state.snapshots[length])

    def import_snapshot(self, state: MlxState, length: int, payload: bytes) -> None:
        state.recurrent = self._load(payload)
        state.length = length
        state.snapshots = {length: state.recurrent}


def test_mlx_state_round_trips_through_the_cache(tmp_path):
    sig = signature(block_tokens=BLOCK)
    store = TwoTierStateStore(sig, ssd_dir=str(tmp_path), hot_budget_bytes=1 << 20)
    codec = MlxCodec(sig)
    cache = BlockPrefixCache(
        store,
        codec,
        block_tokens=BLOCK,
        snapshot_grid=8,
        chunk_tokens=8,
        contended_chunk_tokens=4,
        fine_min_gain_tokens=0,
    )
    try:
        tokens = list(range(1, 21))
        cold = MlxState()
        cold.prefill(tokens[:8], snapshot=True)
        cold.prefill(tokens[8:16], snapshot=True)
        cold.prefill(tokens[16:])
        cache.store(tokens, cold, [8, 16])
        assert store.flush(5.0)

        follow_up = tokens + [99, 98]
        match = cache.lookup(follow_up)
        assert match.matched_tokens == 16

        warm = MlxState()
        assert cache.restore(match, warm) == 16
        assert np.allclose(np.array(warm.kv), np.array(cold.kv[:16]))
        assert np.allclose(
            np.array(warm.recurrent), np.array(cold.snapshots[16])
        )

        warm.prefill(follow_up[16:])
        reference = MlxState()
        reference.prefill(follow_up)
        assert np.allclose(np.array(warm.recurrent), np.array(reference.recurrent))
    finally:
        store.close()

    assert codec.export_threads == {threading.get_ident()}
