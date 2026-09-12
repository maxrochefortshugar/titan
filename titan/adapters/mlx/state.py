"""``ModelState``: every device array a sequence owns, in one object.

The forward functions in :mod:`titan.adapters.mlx.model` take a state and return
a state.  Nothing about a sequence lives in a module attribute, a process global
or a closure, which is what lets the scheduler and the prefix cache own
placement, batching and rollback rather than discovering them.

What a state holds, for Qwen3.8-Flash-Next:

* ``layers`` -- one cache per decoder layer.  The 12 full-attention layers carry
  a ``QSAKVCache`` (keys, values, and the indexer's pooled keys and positions).
  The 36 linear-attention layers carry an ``ArraysCache``: the Gated DeltaNet
  recurrent state and the depthwise conv window, plus, on the PLE layer, the
  n-gram token history.
* ``mtp_layers`` -- the Lightning MTP head's own KV, one cache per MTP layer.
* ``mtp_hidden`` -- the residual stream the MTP block drafts from.
* ``snapshots`` -- staged recurrent-state copies, keyed by token length.

Two asymmetries drive the whole design and are worth stating plainly.  The QSA
layers are positional: truncating them is an offset move.  The GDN layers are
recurrent: the update has no inverse, so rolling back means restoring a
snapshot taken before the tokens you are discarding.  That is why
:meth:`stage_snapshot` exists and why :meth:`truncate` refuses to go below the
last staged point instead of silently recomputing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np

from .vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache
from .vendor.mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
)

SNAPSHOT_MAGIC = b"TITANSNAP"
SNAPSHOT_VERSION = 1

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_VERIFY = "verify"


class StateError(RuntimeError):
    """A state operation the backend refuses rather than approximates."""


@dataclass
class Snapshot:
    """A recurrent-state copy staged at a token length."""

    length: int
    arrays: dict[str, mx.array]


@dataclass
class ModelState:
    """Per-sequence device state. One row unless :meth:`batch` built it."""

    layers: list[Any]
    mtp_layers: list[Any] = field(default_factory=list)
    length: int = 0
    mtp_hidden: Optional[mx.array] = None
    snapshots: dict[int, Snapshot] = field(default_factory=dict)
    rows: int = 1
    phase: str = PHASE_PREFILL
    # Left padding per row, in tokens. ``None`` for an unbatched state.
    left_padding: Optional[Sequence[int]] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def new(cls, model) -> "ModelState":
        """Empty state for one sequence of *model*."""
        language_model = getattr(model, "language_model", model)
        return cls(
            layers=language_model.make_cache(),
            mtp_layers=(
                language_model.make_mtp_cache()
                if language_model.get_mtp_module() is not None
                else []
            ),
        )

    # -- introspection -----------------------------------------------------
    @property
    def is_batched(self) -> bool:
        return self.rows > 1 or self.left_padding is not None

    def recurrent_caches(self) -> list[ArraysCache]:
        return [c for c in self.layers if isinstance(c, ArraysCache)]

    def attention_caches(self) -> list[Any]:
        return [c for c in self.layers if not isinstance(c, ArraysCache)]

    # -- slicing and batching ---------------------------------------------
    def slice(self, rows: Sequence[int]) -> "ModelState":
        """A view of a subset of this state's batch rows.

        Used when a sequence leaves a decode batch: the caches filter in place
        on the surviving rows, so the arrays are not copied per row.
        """
        if not self.is_batched:
            raise StateError("slice() needs a batched state")
        indices = mx.array(list(rows), dtype=mx.int32)
        for cache in self.layers + self.mtp_layers:
            filter_rows = getattr(cache, "filter", None)
            if filter_rows is None:
                raise StateError(f"{type(cache).__name__} cannot be sliced")
            filter_rows(indices)
        hidden = None if self.mtp_hidden is None else self.mtp_hidden[list(rows)]
        padding = (
            None
            if self.left_padding is None
            else [self.left_padding[r] for r in rows]
        )
        return ModelState(
            layers=self.layers,
            mtp_layers=self.mtp_layers,
            length=self.length,
            mtp_hidden=hidden,
            rows=len(rows),
            phase=self.phase,
            left_padding=padding,
        )

    @classmethod
    def batch(
        cls,
        states: Sequence["ModelState"],
        *,
        phase: str = PHASE_DECODE,
    ) -> "ModelState":
        """Join single-sequence states into one left-padded batch.

        Left padding, not right: every row's newest token must sit at the same
        column so one decode step advances the whole block, and the QSA gathered
        arm reads a contiguous window per row.  ``phase`` records what the batch
        was built for -- the batched sparse-QSA arm is only correct for the
        decode and verify layouts, and the caller must not reuse a decode batch
        for a prefill chunk.
        """
        if not states:
            raise StateError("cannot batch zero states")
        if phase not in (PHASE_DECODE, PHASE_VERIFY):
            raise StateError(
                f"batching is for decode and verify, not {phase!r}: prefill runs "
                "one sequence per turn"
            )
        if any(s.is_batched for s in states):
            raise StateError("cannot batch an already batched state")
        if len({len(s.layers) for s in states}) != 1:
            raise StateError("states come from different models")

        width = max(s.length for s in states)
        pads = [width - s.length for s in states]

        joined: list[Any] = []
        for layer_index in range(len(states[0].layers)):
            caches = [s.layers[layer_index] for s in states]
            joined.append(_join_layer(caches, pads))

        joined_mtp: list[Any] = []
        for layer_index in range(len(states[0].mtp_layers)):
            caches = [s.mtp_layers[layer_index] for s in states]
            joined_mtp.append(_join_layer(caches, pads))

        hidden = None
        if all(s.mtp_hidden is not None for s in states):
            hidden = mx.concatenate([s.mtp_hidden for s in states], axis=0)

        return cls(
            layers=joined,
            mtp_layers=joined_mtp,
            length=width,
            mtp_hidden=hidden,
            rows=len(states),
            phase=phase,
            left_padding=pads,
        )

    # -- snapshots ---------------------------------------------------------
    def stage_snapshot(self, length: Optional[int] = None) -> Snapshot:
        """Copy the recurrent state as it stands, keyed by token length.

        Staged, not written.  The store decides whether it reaches disk; the
        state only guarantees that :meth:`truncate` back to this length is exact.
        """
        length = self.length if length is None else length
        arrays: dict[str, mx.array] = {}
        for index, cache in enumerate(self.layers):
            if not isinstance(cache, ArraysCache):
                continue
            for slot, value in enumerate(cache.state or ()):
                if value is not None:
                    arrays[f"layer{index}.slot{slot}"] = mx.array(value)
        snapshot = Snapshot(length=length, arrays=arrays)
        self.snapshots[length] = snapshot
        return snapshot

    def restore_snapshot(self, length: int) -> None:
        """Put the recurrent state back to a staged length."""
        snapshot = self.snapshots.get(length)
        if snapshot is None:
            raise StateError(f"no recurrent snapshot staged at length {length}")
        for index, cache in enumerate(self.layers):
            if not isinstance(cache, ArraysCache):
                continue
            slots = list(cache.state or ())
            for slot in range(len(slots)):
                key = f"layer{index}.slot{slot}"
                slots[slot] = snapshot.arrays.get(key)
            cache.state = tuple(slots)

    def export_snapshot(self, length: int) -> bytes:
        """Serialise a staged snapshot. Opaque to everything but this module."""
        snapshot = self.snapshots.get(length)
        if snapshot is None:
            raise StateError(f"no recurrent snapshot staged at length {length}")
        buffer = io.BytesIO()
        payload = {
            name: np.array(value, copy=False)
            for name, value in snapshot.arrays.items()
        }
        np.savez(buffer, __length__=np.array([length]), **payload)
        return SNAPSHOT_MAGIC + bytes([SNAPSHOT_VERSION]) + buffer.getvalue()

    def import_snapshot(self, length: int, blob: bytes) -> None:
        """Restore recurrent state from :meth:`export_snapshot` bytes."""
        head = len(SNAPSHOT_MAGIC)
        if blob[:head] != SNAPSHOT_MAGIC:
            raise StateError("not a Titan recurrent snapshot")
        version = blob[head]
        if version != SNAPSHOT_VERSION:
            raise StateError(
                f"snapshot version {version} was written by another build"
            )
        with np.load(io.BytesIO(blob[head + 1 :])) as data:
            stored = int(data["__length__"][0])
            if stored != length:
                raise StateError(
                    f"snapshot covers {stored} tokens, not {length}"
                )
            arrays = {
                name: mx.array(data[name])
                for name in data.files
                if name != "__length__"
            }
        self.snapshots[length] = Snapshot(length=length, arrays=arrays)
        self.restore_snapshot(length)
        self.length = length

    # -- rollback ----------------------------------------------------------
    def truncate(self, length: int) -> None:
        """Roll back to exactly *length* tokens, without recomputation.

        The attention layers move their offset.  The recurrent layers cannot be
        rewound, so this restores the snapshot staged at *length*; if there is
        none, the state says so rather than quietly recomputing.
        """
        if length > self.length:
            raise StateError(f"cannot grow a state from {self.length} to {length}")
        if length == self.length:
            return
        if length not in self.snapshots:
            raise StateError(
                f"no recurrent snapshot at {length} (staged: "
                f"{sorted(self.snapshots)}); truncating below the last snapshot "
                "is an error, not a slow path"
            )
        for cache in self.layers + self.mtp_layers:
            if isinstance(cache, ArraysCache):
                continue
            trim = getattr(cache, "trim", None)
            if trim is None:
                raise StateError(f"{type(cache).__name__} cannot be truncated")
            trim(self.length - length)
        self.restore_snapshot(length)
        self.length = length
        for staged in [n for n in self.snapshots if n > length]:
            del self.snapshots[staged]

    def drop_snapshots_before(self, length: int) -> None:
        for staged in [n for n in self.snapshots if n < length]:
            del self.snapshots[staged]


def _join_layer(caches: Sequence[Any], pads: Sequence[int]) -> Any:
    """Join one layer's caches across rows, left-padded."""
    first = caches[0]
    if isinstance(first, QSAKVCache):
        joined = first.to_batch([pads[0]])
        for cache, pad in zip(caches[1:], pads[1:]):
            joined.extend(cache.to_batch([pad]))
        return joined
    if isinstance(first, BatchQSAKVCache):
        joined = first
        for cache in caches[1:]:
            joined.extend(cache)
        return joined
    if isinstance(first, ArraysCache):
        joined = ArraysCache(size=len(first.cache))
        slots: list[Optional[mx.array]] = []
        for slot in range(len(first.cache)):
            values = [c.cache[slot] for c in caches]
            if any(v is None for v in values):
                slots.append(None)
                continue
            slots.append(mx.concatenate(list(values), axis=0))
        joined.cache = slots
        joined.left_padding = mx.array(list(pads))
        return joined
    raise StateError(f"cannot batch {type(first).__name__}")
