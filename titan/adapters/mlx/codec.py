"""The state codec: :class:`ModelState` to bytes and back.

This is the one place in Titan where device memory becomes a byte string. The
cache above it hashes token ids and moves byte strings between RAM and SSD and
never sees an mlx array; the backend below it owns arrays and never sees a
file. Everything that knows both lives here.

Two rules bind every method, and both are the cache's, not this module's.

Serialise on the caller's thread. All four methods run on the scheduler thread,
which is the thread that owns the MLX stream. The cache hands its writer bytes
and nothing else. Serialising on a worker is a Metal command-buffer race rather
than a performance choice.

Evaluate on import. An array built from a buffer that came off disk stays
attached to that buffer until ``mx.eval`` runs, and a state holding file-backed
arrays across an eviction is a use after free. Every array this module builds
from a payload is evaluated before the payload goes out of scope.

## What is in a block and what is in a snapshot

The model is a hybrid, and the split between the two payload kinds is the same
split the whole cache design turns on. The 12 sparse-attention layers hold
positional KV: token 700's keys do not depend on how the sequence arrived at
token 700, so they can be cut at any block boundary and reassembled. Those are
blocks. The 36 Gated DeltaNet layers hold a fold over every token processed,
which has no inverse and cannot be sliced. Those are snapshots, and one exists
only where the engine staged it.

A block payload therefore carries, per attention layer, four arrays: the keys
and values for the token range, the indexer's raw keys, and its position ids.
The pooled and rotated block bank the indexer also keeps is deliberately not
serialised, because the vendored cache rebuilds it exactly from those raw
tensors on restore, and writing it would double the payload to store something
derived.

## Restore is applied all at once

``import_blocks`` stages its arrays on the state rather than appending them to
the live caches, and ``import_snapshot`` applies the whole set. That is what
makes a failed restore leave the state empty instead of half filled: the cache
catches the failure, counts it and prefills from zero, and a state carrying
three of eight blocks would be a state that quietly answers from the wrong
context. The staged arrays are evaluated as they arrive, so the payload buffer
is free to go the moment ``import_blocks`` returns.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import mlx.core as mx

from titan.adapters.cache.format import CacheSignature

from .payload import pack_arrays, unpack_arrays
from .state import ModelState, Snapshot, StateError
from .vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache

__all__ = ["MLXStateCodec"]

_KIND_BLOCKS = "blocks"
_KIND_SNAPSHOT = "snapshot"


# ---------------------------------------------------------------------------
# attention layers
# ---------------------------------------------------------------------------


def _attention_slots(cache: Any) -> tuple[Optional[mx.array], ...]:
    """The four persisted tensors of one QSA cache, or empties.

    ``state`` is the vendored cache's own accessor and it already trims the
    keys and values to the offset, so what comes out is exactly the tokens the
    cache covers rather than its capacity.
    """
    state = cache.state
    if state is None:
        return (None, None, None, None)
    keys, values, index_keys, index_positions = state
    return (keys, values, index_keys, index_positions)


def _slice_tokens(array: Optional[mx.array], axis: int, start: int, end: int):
    if array is None:
        return None
    length = array.shape[axis]
    if start >= length:
        return None
    stop = min(end, length)
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


class MLXStateCodec:
    """:class:`titan.adapters.cache.codec.StateCodec` over :class:`ModelState`.

    Stateless apart from the signature, so one instance serves every sequence.
    """

    def __init__(self, signature: CacheSignature, resolve: Any = None):
        self._signature = signature
        self._resolve = resolve

    def _state(self, state: object) -> ModelState:
        """Turn whatever the cache handed back into a state.

        The cache calls these methods with the same object the engine gave it,
        and the engine deals in opaque ``StateHandle`` integers, so the codec
        needs the backend's table to get from one to the other. A
        :class:`ModelState` is accepted directly, which is what the tests hand
        it and what a caller holding the state already has.
        """
        if isinstance(state, ModelState):
            return state
        if self._resolve is not None:
            resolved = self._resolve(state)
            if isinstance(resolved, ModelState):
                return resolved
        raise StateError(
            f"the mlx codec needs a ModelState or a live handle, got "
            f"{type(state).__name__}"
        )

    @property
    def signature(self) -> CacheSignature:
        return self._signature

    # -- blocks ------------------------------------------------------------
    def export_blocks(self, state: object, start: int, end: int) -> bytes:
        """Attention KV for the half-open token range, one payload."""
        model_state = self._state(state)
        arrays: dict[str, mx.array] = {}
        layers: list[int] = []
        for index, cache in enumerate(model_state.layers):
            if isinstance(cache, ArraysCache):
                continue
            keys, values, index_keys, index_positions = _attention_slots(cache)
            if keys is None:
                raise StateError(
                    f"layer {index} holds no attention KV to export at [{start}, {end})"
                )
            covered = keys.shape[2]
            if covered < end:
                raise StateError(
                    f"layer {index} covers {covered} tokens, not the {end} the "
                    "block asked for"
                )
            layers.append(index)
            arrays[f"l{index}.keys"] = _slice_tokens(keys, 2, start, end)
            arrays[f"l{index}.values"] = _slice_tokens(values, 2, start, end)
            piece = _slice_tokens(index_keys, 1, start, end)
            if piece is not None:
                arrays[f"l{index}.index_keys"] = piece
            piece = _slice_tokens(index_positions, -1, start, end)
            if piece is not None:
                arrays[f"l{index}.index_positions"] = piece
        if not layers:
            raise StateError("this state has no attention layers to export")
        return pack_arrays(
            {"kind": _KIND_BLOCKS, "start": start, "end": end, "layers": layers},
            {name: value for name, value in arrays.items() if value is not None},
        )

    def import_blocks(
        self, state: object, start: int, end: int, payload: bytes
    ) -> None:
        """Stage the KV for one block. Applied by :meth:`import_snapshot`.

        Staged rather than appended because a restore that fails halfway has to
        leave the state empty: the cache counts the failure and prefills from
        zero, and a state carrying half its blocks would answer from a context
        nobody asked for. The arrays are evaluated here, so the payload buffer
        is free as soon as this returns.
        """
        model_state = self._state(state)
        header, arrays = unpack_arrays(payload)
        if header.get("kind") != _KIND_BLOCKS:
            raise StateError(f"expected a block payload, got {header.get('kind')!r}")
        if int(header.get("start", -1)) != start or int(header.get("end", -1)) != end:
            raise StateError(
                f"block payload covers [{header.get('start')}, {header.get('end')}), "
                f"not the [{start}, {end}) it was asked for"
            )
        pending = getattr(model_state, "pending_blocks", None)
        if start == 0 or pending is None:
            # A restore always begins at zero. Starting one drops whatever a
            # previous attempt left staged, which is the only place stale
            # staging can survive a failure.
            pending = []
            model_state.pending_blocks = pending
        if pending and pending[-1][1] != start:
            raise StateError(
                f"block [{start}, {end}) does not follow the staged prefix ending "
                f"at {pending[-1][1]}"
            )
        pending.append((start, end, arrays))

    # -- snapshots ---------------------------------------------------------
    def export_snapshot(self, state: object, length: int) -> bytes:
        """Serialise the recurrent snapshot staged at ``length`` tokens."""
        model_state = self._state(state)
        snapshot = model_state.snapshots.get(length)
        if snapshot is None:
            raise StateError(
                f"no recurrent snapshot staged at length {length} "
                f"(staged: {sorted(model_state.snapshots)})"
            )
        return pack_arrays(
            {"kind": _KIND_SNAPSHOT, "length": length}, snapshot.arrays
        )

    def import_snapshot(self, state: object, length: int, payload: bytes) -> None:
        """Restore recurrent state and apply the staged blocks, all at once."""
        model_state = self._state(state)
        header, arrays = unpack_arrays(payload)
        if header.get("kind") != _KIND_SNAPSHOT:
            raise StateError(f"expected a snapshot payload, got {header.get('kind')!r}")
        stored = int(header.get("length", -1))
        if stored != length:
            raise StateError(f"snapshot covers {stored} tokens, not {length}")

        pending = getattr(model_state, "pending_blocks", None) or []
        covered = pending[-1][1] if pending else 0
        if covered < length:
            raise StateError(
                f"snapshot at {length} needs {length} tokens of KV, only "
                f"{covered} were staged"
            )
        _apply_blocks(model_state, pending, length)
        model_state.pending_blocks = []
        model_state.snapshots = {length: _snapshot_from(length, arrays)}
        model_state.restore_snapshot(length)
        model_state.length = length


def _apply_blocks(
    model_state: ModelState, pending: Sequence[tuple[int, int, dict]], length: int
) -> None:
    """Join the staged blocks per layer and install them, trimmed to ``length``.

    One concatenate per layer per restore rather than one per block: at 512
    tokens a 64k prefix is 128 blocks, and appending to a live cache each time
    would copy the whole prefix 128 times for no reason.
    """
    for index, cache in enumerate(model_state.layers):
        if isinstance(cache, ArraysCache):
            continue
        keys = _join(pending, f"l{index}.keys", axis=2)
        values = _join(pending, f"l{index}.values", axis=2)
        if keys is None or values is None:
            raise StateError(f"restored blocks carry no KV for layer {index}")
        index_keys = _join(pending, f"l{index}.index_keys", axis=1)
        index_positions = _join(pending, f"l{index}.index_positions", axis=-1)
        keys = _trim(keys, 2, length)
        values = _trim(values, 2, length)
        index_keys = _trim(index_keys, 1, length)
        index_positions = _trim(index_positions, -1, length)
        cache.state = (keys, values, index_keys, index_positions)
        mx.eval([a for a in (keys, values, index_keys, index_positions) if a is not None])


def _join(pending: Sequence[tuple[int, int, dict]], name: str, *, axis: int):
    pieces = [arrays[name] for _s, _e, arrays in pending if name in arrays]
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return mx.concatenate(pieces, axis=axis)


def _trim(array, axis: int, length: int):
    if array is None or array.shape[axis] <= length:
        return array
    index = [slice(None)] * array.ndim
    index[axis] = slice(0, length)
    return array[tuple(index)]


def _snapshot_from(length: int, arrays: Mapping[str, mx.array]):
    return Snapshot(length=length, arrays=dict(arrays))


