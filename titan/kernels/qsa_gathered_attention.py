# SPDX-License-Identifier: MIT
"""Sparse gathered QSA attention: single-sequence and batched arms.

Ported from ``engine/patches/round3/qsa-batched/batched_qsa.py`` and the
single-sequence arm ``round3/qsa-verify`` routes M = 1 verify rows onto.

QSA scores compressed blocks of the key history with a small indexer, keeps the
``token_budget / compress_ratio`` best complete blocks, gathers their token rows
plus the incomplete tail, and attends only those. Main attention is therefore
bounded by ``token_budget + compress_ratio - 1`` instead of scanning a dense
mask over the whole cache.

Geometry. Batched rows are LEFT padded and right aligned: physical width
``width``, row ``b`` carries ``pads[b]`` dead slots at the front, so its logical
length is ``width - pads[b]`` and logical token ``t`` of row ``b`` lives at
physical column ``pads[b] + t``. Blocks are groups of ``compress_ratio``
*logical* tokens, so each row's block grid has its own phase in physical space
whenever ``pads[b] % compress_ratio != 0``. Everything below indexes blocks in
phased slot space and converts to physical columns only at the final gather,
which is exactly what makes the batched output equal the per-row output.

The pooled index-key bank is in phased slot space too: slot ``j`` of row ``b``
covers physical columns ``ratio*j + phase_b`` through
``ratio*j + phase_b + ratio - 1``, so row ``b``'s logical block ``k`` lives at
slot ``k + offset_b`` where ``offset_b = pads[b] // ratio``.

Two implementations, both plain MLX call sequences (this op has no Metal source
of its own; the acceleration is the shape of the call sequence, which the
registry explicitly allows):

``reference``
    one row at a time. Every row is sliced out of the batched views, so no
    phase arithmetic is needed at all, and each row runs the single-sequence
    arm. Exact by construction, B launches per stage.
``metal``
    selection and one gathered attention over a padded ``[B, L, selected]``
    index set. One launch per stage regardless of B.

Exactness: the loop arm is bit identical to running each row on its own; the
padded arm is within one bf16 ULP of it, because the padded softmax reduces
over a fixed-width span with masked lanes rather than each row's own width.
A single-row call (B = 1, pads = [0]) takes the same path in both.

State: :class:`QSAGeometry` and :class:`QSAConfig` are value objects the caller
builds. The pooled bank is the caller's array. Nothing is memoised here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["OP", "QSAConfig", "QSAGeometry", "indexer_scores", "key", "metal",
           "pool_slots", "reference", "supports"]


# ---------------------------------------------------------------------------
# caller-owned value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QSAConfig:
    """Head geometry and the sparsity budget."""

    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    indexer_head_dim: int
    compress_ratio: int
    token_budget: int

    @property
    def block_budget(self) -> int:
        return self.token_budget // self.compress_ratio

    def validate(self) -> None:
        if self.compress_ratio <= 0 or self.token_budget <= 0:
            raise ValueError("QSA ratio and budget must be positive")
        if self.token_budget % self.compress_ratio:
            raise ValueError("QSA token budget must contain complete blocks")
        if self.num_query_heads % self.num_key_value_heads:
            raise ValueError("QSA query heads must divide over K/V heads")


class QSAGeometry:
    """Physical width, per-row left padding, and the derived slot bookkeeping."""

    __slots__ = ("width", "pads", "lengths", "offsets", "phases", "batch", "ratio")

    def __init__(self, width: int, pads: Sequence[int], compress_ratio: int):
        self.width = int(width)
        self.ratio = int(compress_ratio)
        self.pads = [int(p) for p in pads]
        self.lengths = [self.width - p for p in self.pads]
        self.offsets = [p // self.ratio for p in self.pads]
        self.phases = [p % self.ratio for p in self.pads]
        self.batch = len(self.pads)
        if any(n <= 0 for n in self.lengths):
            raise ValueError("QSA rows must have positive logical length")

    @property
    def block_counts(self) -> list[int]:
        """Complete blocks per row, in that row's own logical block space."""
        return [n // self.ratio for n in self.lengths]

    @property
    def slots(self) -> int:
        """Slots the phased bank must hold."""
        return max(c + o for c, o in zip(self.block_counts, self.offsets))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def indexer_scores(index_queries: mx.array, pooled_keys: mx.array,
                   head_dim: int) -> mx.array:
    """fp32 QSA block scores. ``index_queries``: [B, L, Hi, Di] -> [B, L, S].

    The query-token and index-head axes are flattened so MLX emits one fp32
    GEMM for the chunk instead of a broadcast batch of tiny matmuls. Each dot
    product and the head reduction are unchanged by that.
    """
    batch, query_tokens, query_heads, _ = index_queries.shape
    scores = (
        index_queries.astype(mx.float32).reshape(
            batch, query_tokens * query_heads, head_dim
        )
        @ pooled_keys.astype(mx.float32).swapaxes(-1, -2)
    ).reshape(batch, query_tokens, query_heads, pooled_keys.shape[1])
    return mx.sum(mx.maximum(scores, 0), axis=-2) / math.sqrt(head_dim)


def pool_slots(
    index_keys: mx.array,
    index_positions: mx.array,
    geo: QSAGeometry,
    slot_start: int,
    slot_count: int,
    index_key_norm: Callable[[mx.array], mx.array],
    apply_index_rope: Callable[[mx.array, mx.array], mx.array],
) -> mx.array:
    """Pool bank slots ``[slot_start, slot_start + slot_count)`` for every row.

    Slot ``j`` of row ``b`` reads physical columns ``ratio*j + phase_b + r``.
    The caller owns the bank this writes into; this returns the fresh slots.
    """
    ratio = geo.ratio
    batch = geo.batch
    dim = index_keys.shape[-1]
    phases = mx.array(geo.phases, dtype=mx.int32)
    span = mx.arange(slot_count * ratio, dtype=mx.int32)
    columns = mx.minimum(
        ratio * slot_start + phases[:, None] + span[None, :], geo.width - 1
    )
    gathered = mx.take_along_axis(
        index_keys[:, : geo.width], columns[..., None], axis=1
    ).reshape(batch, slot_count, ratio, dim)
    pooled = mx.mean(gathered.astype(mx.float32), axis=-2).astype(index_keys.dtype)
    pooled = index_key_norm(pooled)
    starts = columns[:, ::ratio]
    if index_positions.ndim == 3:
        positions = mx.take_along_axis(
            index_positions[..., : geo.width], starts[None], axis=2
        )
    else:
        positions = mx.take_along_axis(
            index_positions[:, : geo.width], starts, axis=1
        )
    return apply_index_rope(pooled[:, None], positions)[:, 0]


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def _single_row(queries, keys, values, index_queries, pooled_keys, cfg: QSAConfig):
    """One row, logical space: keys/values already sliced to the row's own
    length and ``pooled_keys`` to the row's own complete blocks."""
    ratio = cfg.compress_ratio
    budget = cfg.block_budget
    _b, _h, length, _d = queries.shape
    key_tokens = int(keys.shape[2])
    blocks = int(pooled_keys.shape[1])

    q_abs = key_tokens - length + mx.arange(length, dtype=mx.int32)[None, :]
    complete = (q_abs + 1) // ratio                                  # [1, L]

    scores = indexer_scores(index_queries, pooled_keys, cfg.indexer_head_dim)
    columns = mx.arange(blocks, dtype=mx.int32)[None, None, :]
    scores = mx.where(columns < complete[..., None], scores,
                      mx.finfo(scores.dtype).min)

    ranked = mx.argpartition(scores, kth=-budget, axis=-1)[..., -budget:].astype(
        mx.int32
    )
    canonical = mx.broadcast_to(
        mx.arange(budget, dtype=mx.int32), (1, length, budget)
    )
    selected = mx.where((complete <= budget)[..., None], canonical, ranked)
    # Restore chronological order so the fp32 reduction matches block order.
    selected = mx.sort(selected, axis=-1)
    selected_count = mx.minimum(complete, budget)

    tokens = (
        selected[..., None] * ratio + mx.arange(ratio, dtype=mx.int32)
    ).reshape(1, length, budget * ratio)
    token_valid = mx.broadcast_to(
        mx.arange(budget, dtype=mx.int32)[None, None, :, None]
        < selected_count[..., None, None],
        (1, length, budget, ratio),
    ).reshape(1, length, budget * ratio)

    # The zero to ratio-1 visible tokens after the last complete block.
    tail = complete[..., None] * ratio + mx.arange(ratio - 1, dtype=mx.int32)
    tail_valid = tail <= q_abs[..., None]
    tokens = mx.concatenate((tokens, tail), axis=-1)
    token_valid = mx.concatenate((token_valid, tail_valid), axis=-1)
    return _gathered_sdpa(queries, keys, values, tokens, token_valid, cfg)


def _gathered_sdpa(queries, keys, values, columns, token_valid, cfg: QSAConfig):
    """fp32 attention over the gathered columns. ``columns``: [B, L, span]."""
    batch, _h, length, _d = queries.shape
    span = int(columns.shape[-1])
    hkv, hq, dim = cfg.num_key_value_heads, cfg.num_query_heads, cfg.head_dim
    columns = mx.where(token_valid, columns, 0).astype(mx.int32)
    gather_index = mx.broadcast_to(
        columns.reshape(batch, 1, length * span, 1), (batch, hkv, length * span, dim)
    )
    selected_keys = mx.take_along_axis(keys, gather_index, axis=2).reshape(
        batch, hkv, length, span, dim
    ).transpose(0, 2, 1, 3, 4)
    selected_values = mx.take_along_axis(values, gather_index, axis=2).reshape(
        batch, hkv, length, span, dim
    ).transpose(0, 2, 1, 3, 4)

    groups = hq // hkv
    grouped = queries.transpose(0, 2, 1, 3).reshape(batch, length, hkv, groups, dim)
    attn = (
        grouped.astype(mx.float32)
        @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
    ) / math.sqrt(dim)
    attn = mx.where(token_valid[:, :, None, None, :], attn, mx.finfo(attn.dtype).min)
    probabilities = mx.softmax(attn, axis=-1).astype(queries.dtype)
    output = probabilities @ selected_values
    return output.reshape(batch, length, hq, dim)


def reference(queries, keys, values, index_queries, pooled_index_keys,
              geo: QSAGeometry, cfg: QSAConfig):
    """One row at a time. Returns [B, L, Hq, D].

    ``queries``: [B, Hq, L, D]; ``keys``/``values``: [B, Hkv, width, D] already
    updated for this step; ``index_queries``: [B, L, Hi, Di], normalised and
    rotated; ``pooled_index_keys``: [B, slots, Di] in phased slot space.

    Row slices stay views: materialising a long row's K/V would cost more than
    the attention it feeds.
    """
    cfg.validate()
    counts = geo.block_counts
    rows = []
    for b in range(geo.batch):
        pad, off = geo.pads[b], geo.offsets[b]
        rows.append(_single_row(
            queries[b:b + 1],
            keys[b:b + 1, :, pad:geo.width, :],
            values[b:b + 1, :, pad:geo.width, :],
            index_queries[b:b + 1],
            pooled_index_keys[b:b + 1, off:off + counts[b]],
            cfg,
        ))
    return mx.concatenate(rows, axis=0)


def metal(queries, keys, values, index_queries, pooled_index_keys,
          geo: QSAGeometry, cfg: QSAConfig):
    """One padded pass over the whole batch. Same signature as :func:`reference`.

    Selection runs in the bank's phased slot space, so a picked slot ``j`` of
    row ``b`` maps straight onto physical columns ``ratio*j + phase_b + r`` with
    no per-row reindexing.
    """
    cfg.validate()
    batch, query_heads, length, dim = queries.shape
    if query_heads != cfg.num_query_heads or dim != cfg.head_dim:
        raise ValueError("QSA queries do not match the configured geometry")
    if keys.shape != values.shape or keys.shape[0] != batch:
        raise ValueError("QSA K/V must be matching batched arrays")
    if keys.shape[2] != geo.width:
        raise ValueError("QSA K/V width does not match the cache geometry")
    ratio = cfg.compress_ratio
    budget = cfg.block_budget
    slots = int(pooled_index_keys.shape[1])

    lengths = mx.array(geo.lengths, dtype=mx.int32)[:, None]
    offsets = mx.array(geo.offsets, dtype=mx.int32)[:, None]
    phases = mx.array(geo.phases, dtype=mx.int32)[:, None, None]
    q_abs = lengths - length + mx.arange(length, dtype=mx.int32)[None, :]
    complete = (q_abs + 1) // ratio                                  # [B, L]

    scores = indexer_scores(index_queries, pooled_index_keys, cfg.indexer_head_dim)
    columns = mx.arange(slots, dtype=mx.int32)[None, None, :]
    valid_slots = (columns >= offsets[..., None]) & (
        columns < (offsets + complete)[..., None]
    )
    scores = mx.where(valid_slots, scores, mx.finfo(scores.dtype).min)

    ranked = mx.argpartition(scores, kth=-budget, axis=-1)[..., -budget:].astype(
        mx.int32
    )
    canonical = offsets[..., None] + mx.arange(budget, dtype=mx.int32)
    selected = mx.where((complete <= budget)[..., None], canonical, ranked)
    selected = mx.sort(selected, axis=-1)
    selected_count = mx.minimum(complete, budget)

    tokens = (
        selected[..., None] * ratio + mx.arange(ratio, dtype=mx.int32)
    ).reshape(batch, length, budget * ratio)
    token_valid = mx.broadcast_to(
        mx.arange(budget, dtype=mx.int32)[None, None, :, None]
        < selected_count[..., None, None],
        (batch, length, budget, ratio),
    ).reshape(batch, length, budget * ratio)

    tail = (offsets + complete)[..., None] * ratio + mx.arange(
        ratio - 1, dtype=mx.int32
    )
    tail_valid = (
        complete[..., None] * ratio + mx.arange(ratio - 1, dtype=mx.int32)
    ) <= q_abs[..., None]
    tokens = mx.concatenate((tokens, tail), axis=-1)
    token_valid = mx.concatenate((token_valid, tail_valid), axis=-1)

    # Slot space -> physical columns; masked-off slots read column 0.
    return _gathered_sdpa(queries, keys, values, tokens + phases, token_valid, cfg)


def key(queries, keys, values, index_queries, pooled_index_keys,
        geo: QSAGeometry, cfg: QSAConfig) -> ShapeClass:
    return shape_class(
        queries, keys, values, index_queries, pooled_index_keys,
        extra=(geo.batch, cfg.compress_ratio, cfg.token_budget,
               cfg.num_query_heads, cfg.num_key_value_heads,
               tuple(p % cfg.compress_ratio for p in geo.pads)),
    )


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) != 5 or len(k.extra) != 6:
        return False
    _batch, ratio, budget, hq, hkv, _phases = k.extra
    if ratio <= 0 or budget % ratio or hq % hkv:
        return False
    qs, ks, vs, iqs, pks = k.shapes
    if len(qs) != 4 or len(ks) != 4 or ks != vs or len(iqs) != 4 or len(pks) != 3:
        return False
    # a sparse crossover has to exist, otherwise selection removes nothing
    return pks[1] > budget // ratio


OP = KernelOp(
    name="qsa_gathered_attention",
    aliases=("qsa_sparse_decode",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=1.0,
    shapes=(
        {"B": 1, "L": 1, "Hq": 24, "Hkv": 2, "D": 256, "Di": 128,
         "ratio": 4, "budget": 64},
        {"B": 4, "L": 1, "Hq": 24, "Hkv": 2, "D": 256, "Di": 128,
         "ratio": 4, "budget": 64},
        {"B": 4, "L": 4, "Hq": 24, "Hkv": 2, "D": 256, "Di": 128,
         "ratio": 4, "budget": 2048},
    ),
    exactness="loop arm bit identical per row; padded arm within 1 bf16 ULP",
    source="engine/patches/round3/qsa-batched/REPORT.md section 2, "
           "engine/patches/round3/qsa-verify/REPORT.md",
)
