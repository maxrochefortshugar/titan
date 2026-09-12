"""A model-shaped state that is small enough to keep a thousand of.

The cache only needs three things to be true of a backend state, and all three
can be had from numpy in a few lines:

* attention KV is per token and sliceable, so a restored prefix is exactly the
  rows the same tokens would have produced;
* recurrent state is a fold over every token seen, so it cannot be sliced and
  can only come back from a snapshot;
* a snapshot exists only where one was staged.

``FakeState`` is those three. The fold is deliberately order dependent, so a
test that restores from the wrong point produces a different number rather than
a plausible one, and every assertion about "restored equals recomputed" is a
real assertion.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from titan.adapters.cache.format import CacheSignature

_MASK = (1 << 61) - 1


def fold(seed: int, tokens: Sequence[int]) -> int:
    """The recurrence. No inverse, which is the whole point."""
    value = seed
    for token in tokens:
        value = (value * 1000003 + token + 12345) & _MASK
    return value


def kv_rows(tokens: Sequence[int], width: int = 4) -> np.ndarray:
    """Positional, sliceable, one row per token."""
    ids = np.asarray(tokens, dtype=np.int64).reshape(-1, 1)
    offsets = np.arange(width, dtype=np.int64).reshape(1, -1)
    return (ids * 7 + offsets).astype(np.int32)


@dataclass
class FakeState:
    """One sequence's state: KV rows, a recurrent fold, staged snapshots."""

    width: int = 4
    kv: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    recurrent: int = 0
    length: int = 0
    snapshots: dict[int, int] = field(default_factory=dict)

    def prefill(self, tokens: Sequence[int], *, snapshot: bool = False) -> None:
        """Advance the state over ``tokens``, optionally staging a snapshot."""
        self.kv = np.concatenate([self.kv, kv_rows(tokens, self.width)], axis=0)
        self.recurrent = fold(self.recurrent, tokens)
        self.length += len(tokens)
        if snapshot:
            self.stage_snapshot()

    def stage_snapshot(self, length: int | None = None) -> None:
        length = self.length if length is None else length
        if length != self.length:
            raise ValueError("a snapshot can only be staged at the current length")
        self.snapshots[length] = self.recurrent


class FakeCodec:
    """A :class:`titan.adapters.cache.codec.StateCodec` over :class:`FakeState`.

    Serialisation is numpy's own format, which makes a corrupted payload look
    like a corrupted payload rather than like a shorter one.
    """

    def __init__(self, signature: CacheSignature) -> None:
        self._signature = signature
        self.exports = 0
        self.imports = 0

    @property
    def signature(self) -> CacheSignature:
        return self._signature

    def export_blocks(self, state: FakeState, start: int, end: int) -> bytes:
        if end > state.kv.shape[0]:
            raise ValueError(f"state holds {state.kv.shape[0]} rows, asked for {end}")
        self.exports += 1
        buffer = io.BytesIO()
        np.save(buffer, state.kv[start:end], allow_pickle=False)
        return buffer.getvalue()

    def import_blocks(
        self, state: FakeState, start: int, end: int, payload: bytes
    ) -> None:
        rows = np.load(io.BytesIO(payload), allow_pickle=False)
        if state.kv.shape[0] != start:
            raise ValueError("blocks must arrive in order")
        self.imports += 1
        state.kv = np.concatenate([state.kv, rows], axis=0)

    def export_snapshot(self, state: FakeState, length: int) -> bytes:
        if length not in state.snapshots:
            raise KeyError(f"no snapshot staged at {length}")
        return int(state.snapshots[length]).to_bytes(16, "little")

    def import_snapshot(self, state: FakeState, length: int, payload: bytes) -> None:
        state.recurrent = int.from_bytes(payload, "little")
        state.length = length
        state.snapshots = {length: state.recurrent}


def signature(**overrides) -> CacheSignature:
    fields = {
        "model_name": "qwen4_exp-test",
        "layer_layout": ("gdn", "gdn", "qsa"),
        "block_tokens": 8,
        "snapshot_dtype": "fp32",
    }
    fields.update(overrides)
    return CacheSignature(**fields)  # type: ignore[arg-type]


def tokens_for(count: int, *, seed: int = 0) -> list[int]:
    """Deterministic ids. Prompt ``n+1`` extends prompt ``n``, as a turn does."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=count, dtype=np.int64).tolist()
