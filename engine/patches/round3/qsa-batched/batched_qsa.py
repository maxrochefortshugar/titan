# SPDX-License-Identifier: Apache-2.0
"""Sparse gathered QSA for batched decode and batched verify (BatchQSAKVCache).

The stock qwen4_exp arms are batch-one only: every gathered predicate tests
``type(cache) is QSAKVCache``, so a batch join (``QSAKVCache.to_batch`` ->
``BatchQSAKVCache``) drops all three sparse arms and decode goes dense over the
whole cache. This module supplies the missing arm.

Geometry of a ``BatchQSAKVCache``. Rows are LEFT padded and right aligned:
physical width ``T = kv_cache._idx``, row ``b`` carries ``pad_b`` dead slots at
the front, so its logical length is ``n_b = T - pad_b`` and logical token ``t``
of row ``b`` lives at physical column ``pad_b + t``. QSA blocks are groups of
``compress_ratio`` LOGICAL tokens, so the block grid of each row has its own
phase in physical space whenever ``pad_b % compress_ratio != 0``. Everything
below indexes blocks logically and converts to physical columns only at the
final gather, which is what makes the batched output equal the per-sequence
single-row output exactly.

Two routes, same result:

* ``looped_gathered_qsa``  (route a) slices each row out of the batched views
  and calls the stock single-sequence kernels. Simple, exact by construction,
  B small launches per layer.
* ``padded_gathered_qsa``  (route b) runs selection and one gathered attention
  over a padded ``[B, L, selected]`` index set. One launch per stage regardless
  of B.

Both take and return batched tensors, so projections, RoPE and the output
projection stay batched in the caller.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import mlx.core as mx

_QSA = None


def _qsa_module():
    """Import the vendored qsa_fast lazily so this file can be read standalone."""
    global _QSA
    if _QSA is None:
        import importlib

        _QSA = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")
    return _QSA


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------


def gather_min_context(token_budget: int) -> int:
    """Context from which a gathered arm is allowed to engage.

    The stock gate is ``cache.offset + L > token_budget`` (2048 here), which is
    the point at which selection first removes work, not the point at which it
    starts paying. Measured on this M5 Max the gathered decode arm only beats
    dense masked SDPA past roughly 8k, so the default is 8192 and
    ``OMLX_QSA_GATHER_MIN_CTX`` overrides it. Values below ``token_budget`` are
    clamped up: below that the arm is not merely slow, it has nothing to select.
    """
    raw = os.environ.get("OMLX_QSA_GATHER_MIN_CTX", "").strip()
    if raw:
        try:
            return max(int(token_budget), int(raw))
        except ValueError:
            pass
    return max(int(token_budget), 8192)


def batched_route() -> str:
    raw = os.environ.get("OMLX_QSA_BATCHED_ROUTE", "").strip().lower()
    return raw if raw in {"loop", "padded"} else "padded"


# --------------------------------------------------------------------------
# batched cache geometry
# --------------------------------------------------------------------------


def host_left_padding(cache) -> Optional[list[int]]:
    """Host-side left padding, memoized on the cache by array identity.

    ``left_padding`` is an ``mx.array``; reading it syncs. It is rebound only by
    prepare/filter/finalize/extend, so caching on identity costs one sync per
    batch composition rather than one per layer per step.
    """
    padding = getattr(cache, "left_padding", None)
    if not isinstance(padding, mx.array) or padding.ndim == 0:
        return None
    cached = getattr(cache, "_qsab_pad_host", None)
    if cached is not None and cached[0] is padding:
        return cached[1]
    pads = [int(p) for p in padding.tolist()]
    cache._qsab_pad_host = (padding, pads)
    return pads


class BatchGeometry:
    """Physical width, per-row padding and per-row logical lengths."""

    __slots__ = ("width", "pads", "lengths", "batch")

    def __init__(self, width: int, pads: Sequence[int]):
        self.width = int(width)
        self.pads = [int(p) for p in pads]
        self.lengths = [self.width - p for p in self.pads]
        self.batch = len(self.pads)


def batch_geometry(cache) -> Optional[BatchGeometry]:
    pads = host_left_padding(cache)
    if pads is None:
        return None
    kv = getattr(cache, "kv_cache", None)
    width = getattr(kv, "_idx", None)
    if width is None:
        return None
    geo = BatchGeometry(int(width), pads)
    if any(n <= 0 for n in geo.lengths):
        return None
    return geo


# --------------------------------------------------------------------------
# incremental pooled block bank
# --------------------------------------------------------------------------


class PooledBank:
    """Completed-block bank in PHASED block space.

    Slot ``j`` of row ``b`` holds the block covering physical index columns
    ``[ratio*j + phase_b, ratio*j + phase_b + ratio)``, where
    ``phase_b = pad_b % ratio`` and ``off_b = pad_b // ratio``. Row ``b``'s
    logical block ``k`` therefore lives at slot ``k + off_b``, and the slot a
    step has to write lands within a two or three slot window that is the same
    for every row. That makes the update an in-place slice assignment instead
    of a full-bank scatter: ``mx.put_along_axis`` copies the whole bank (33 MB
    at 130k, per layer, per step) while a slice write costs what it writes.

    ``BatchQSAKVCache`` keeps no bank at all today, so the stock batched path
    re-pools the entire indexer history on every step (language.py:1189 falls
    through to ``pool_completed_index_keys``). This replaces that with O(1)
    work per step.
    """

    __slots__ = (
        "keys", "counts", "capacity", "ratio", "tag", "offsets", "phases", "width",
    )

    step = 4096  # slots

    def __init__(self, geo: "BatchGeometry", ratio: int, tag: Any):
        self.keys = None
        self.capacity = 0
        self.ratio = ratio
        self.tag = tag
        self.offsets = [p // ratio for p in geo.pads]
        self.phases = [p % ratio for p in geo.pads]
        self.counts = [0] * geo.batch
        self.width = 0

    def valid_for(self, geo: BatchGeometry, ratio: int, tag: Any) -> bool:
        if self.keys is None or self.ratio != ratio or self.tag is not tag:
            return False
        if len(self.counts) != geo.batch:
            return False
        if self.offsets != [p // ratio for p in geo.pads]:
            return False
        return all(c <= n // ratio for c, n in zip(self.counts, geo.lengths))

    def grow(self, needed: int, sample: mx.array, batch: int) -> None:
        if needed <= self.capacity:
            return
        capacity = max(
            ((needed + self.step - 1) // self.step) * self.step, 2 * self.capacity
        )
        fresh = mx.zeros((batch, capacity, sample.shape[-1]), dtype=sample.dtype)
        if self.keys is not None and self.capacity:
            fresh[:, : self.capacity] = self.keys[:, : self.capacity]
        self.keys = fresh
        self.capacity = capacity


def _write_mask(slots, pads, offsets, previous_width, width, ratio):
    """Which bank slots this step completed, per row, without a host round trip."""
    lower = offsets + (previous_width - pads) // ratio
    upper = offsets + (width - pads) // ratio
    return ((slots >= lower) & (slots < upper))[..., None]


def _pool_slots(
    index_keys: mx.array,
    index_positions: mx.array,
    phases: mx.array,
    span: mx.array,
    slot_start: int,
    slot_count: int,
    ratio: int,
    index_key_norm,
    apply_index_rope,
    width: int,
) -> mx.array:
    """Pool bank slots ``[slot_start, slot_start + slot_count)`` for every row.

    Slot ``j`` of row ``b`` reads physical columns ``ratio*j + phase_b + r``.
    """
    batch = phases.shape[0]
    dim = index_keys.shape[-1]
    columns = mx.minimum(
        ratio * slot_start + phases[:, None] + span[None, :], width - 1
    )
    gathered = mx.take_along_axis(
        index_keys[:, :width], columns[..., None], axis=1
    ).reshape(batch, slot_count, ratio, dim)
    pooled = mx.mean(gathered.astype(mx.float32), axis=-2).astype(index_keys.dtype)
    pooled = index_key_norm(pooled)
    starts = columns[:, ::ratio]
    if index_positions.ndim == 3:
        positions = mx.take_along_axis(
            index_positions[..., :width], starts[None], axis=2
        )
    else:
        positions = mx.take_along_axis(index_positions[:, :width], starts, axis=1)
    return apply_index_rope(pooled[:, None], positions)[:, 0]


def batched_pooled_index_keys(
    cache,
    geo: BatchGeometry,
    *,
    compress_ratio: int,
    index_key_norm,
    apply_index_rope,
    cache_tag: Any = None,
) -> tuple[mx.array, PooledBank]:
    """Return the ``[B, S, D]`` bank and its slot bookkeeping."""
    index_keys = cache.index_keys
    index_positions = cache.index_position_ids
    if index_keys is None or index_positions is None:
        raise ValueError("batched QSA requires indexer state on the cache")
    width = int(cache.index_offset)
    if width != geo.width:
        raise ValueError("batched QSA indexer and KV widths disagree")

    bank = getattr(cache, "_qsab_bank", None)
    if bank is None or not bank.valid_for(geo, compress_ratio, cache_tag):
        bank = PooledBank(geo, compress_ratio, cache_tag)
        cache._qsab_bank = bank
    counts = [n // compress_ratio for n in geo.lengths]
    if counts == bank.counts:
        return bank.keys[:, : max(c + o for c, o in zip(counts, bank.offsets))], bank

    # Small index constants are rebuilt every layer every step otherwise, and a
    # host to device transfer costs about as much as the work it feeds.
    constants = getattr(cache, "_qsab_constants", None)
    if constants is None or constants["batch"] != geo.batch:
        constants = {
            "batch": geo.batch,
            "phases": mx.array(bank.phases, dtype=mx.int32),
            "pads": mx.array(geo.pads, dtype=mx.int32)[:, None],
            "offsets": mx.array(bank.offsets, dtype=mx.int32)[:, None],
            "spans": {},
            "ramps": {},
        }
        mx.eval(
            constants["phases"], constants["pads"], constants["offsets"]
        )
        cache._qsab_constants = constants
    phases = constants["phases"]

    slot_stop = max(c + o for c, o in zip(counts, bank.offsets))
    slot_start = min(p + o for p, o in zip(bank.counts, bank.offsets))
    bank.grow(slot_stop, index_keys, geo.batch)
    span = slot_stop - slot_start
    if span not in constants["spans"]:
        constants["spans"][span] = mx.arange(
            span * compress_ratio, dtype=mx.int32
        )
        constants["ramps"][span] = mx.arange(span, dtype=mx.int32)[None, :]
    fresh = _pool_slots(
        index_keys,
        index_positions,
        phases,
        constants["spans"][span],
        slot_start,
        span,
        compress_ratio,
        index_key_norm,
        apply_index_rope,
        width,
    )
    # Only the slots this step actually completed may overwrite the bank.
    # Bounds come from the cached pad vector and two host scalars, so no small
    # host to device transfer happens on the steady-state path.
    slots = slot_start + constants["ramps"][span]
    written = _write_mask(
        slots,
        constants["pads"],
        constants["offsets"],
        bank.width,
        width,
        compress_ratio,
    )
    bank.keys[:, slot_start:slot_stop] = mx.where(
        written, fresh.astype(bank.keys.dtype), bank.keys[:, slot_start:slot_stop]
    )
    bank.counts = counts
    bank.width = width
    return bank.keys[:, :slot_stop], bank


# --------------------------------------------------------------------------
# route (b): one padded gathered attention over the whole batch
# --------------------------------------------------------------------------


def padded_gathered_qsa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    pooled_index_keys: mx.array,
    bank: PooledBank,
    geo: BatchGeometry,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
) -> mx.array:
    """Exact gathered QSA for ``B`` rows x ``L`` query tokens in one pass.

    ``queries``/``keys``/``values`` are the batched cache views ``[B, H, *, D]``
    with ``keys``/``values`` already updated for this step (physical width
    ``geo.width``). ``index_queries`` is ``[B, L, Hi, Di]``, normalized and
    RoPE-rotated. Returns ``[B, L, Hq, D]``.

    Selection runs in the bank's phased slot space, so a picked slot ``j`` of
    row ``b`` maps straight onto physical columns ``ratio*j + phase_b + r``
    with no per-row reindexing.
    """
    qsa = _qsa_module()
    batch, query_heads, length, dim = queries.shape
    if query_heads != num_query_heads or dim != head_dim:
        raise ValueError("batched QSA queries do not match the configured geometry")
    if keys.shape != values.shape or keys.shape[0] != batch:
        raise ValueError("batched QSA K/V must be matching batched arrays")
    if keys.shape[2] != geo.width:
        raise ValueError("batched QSA K/V width does not match the cache geometry")
    if num_query_heads % num_key_value_heads:
        raise ValueError("batched QSA query heads must divide over K/V heads")
    ratio = compress_ratio
    budget = token_budget // ratio
    slots = int(pooled_index_keys.shape[1])
    if slots <= budget:
        raise ValueError("batched QSA requires a sparse block crossover")

    lengths = mx.array(geo.lengths, dtype=mx.int32)[:, None]
    offsets = mx.array(bank.offsets, dtype=mx.int32)[:, None]
    phases = mx.array(bank.phases, dtype=mx.int32)[:, None, None]
    # Absolute logical index of every query row.
    q_abs = lengths - length + mx.arange(length, dtype=mx.int32)[None, :]
    complete = (q_abs + 1) // ratio                               # [B, L]

    scores = qsa._portable_indexer_scores(
        index_queries, pooled_index_keys, indexer_head_dim
    )                                                              # [B, L, S]
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
    # Restore chronological order so the FP32 reduction matches the reference.
    selected = mx.sort(selected, axis=-1)
    selected_count = mx.minimum(complete, budget)                  # [B, L]

    tokens = (
        selected[..., None] * ratio + mx.arange(ratio, dtype=mx.int32)
    ).reshape(batch, length, budget * ratio)
    token_valid = mx.broadcast_to(
        mx.arange(budget, dtype=mx.int32)[None, None, :, None]
        < selected_count[..., None, None],
        (batch, length, budget, ratio),
    ).reshape(batch, length, budget * ratio)

    # The zero to ratio-1 visible tokens after the last complete block sit in
    # the slot immediately past it.
    tail = (offsets + complete)[..., None] * ratio + mx.arange(
        ratio - 1, dtype=mx.int32
    )
    tail_valid = (
        complete[..., None] * ratio + mx.arange(ratio - 1, dtype=mx.int32)
    ) <= q_abs[..., None]
    tokens = mx.concatenate((tokens, tail), axis=-1)
    token_valid = mx.concatenate((token_valid, tail_valid), axis=-1)
    span = tokens.shape[-1]

    # Slot space -> physical columns; masked-off slots read column 0.
    columns = mx.where(token_valid, tokens + phases, 0).astype(mx.int32)

    gather_index = mx.broadcast_to(
        columns.reshape(batch, 1, length * span, 1),
        (batch, num_key_value_heads, length * span, head_dim),
    )
    selected_keys = mx.take_along_axis(keys, gather_index, axis=2).reshape(
        batch, num_key_value_heads, length, span, head_dim
    ).transpose(0, 2, 1, 3, 4)
    selected_values = mx.take_along_axis(values, gather_index, axis=2).reshape(
        batch, num_key_value_heads, length, span, head_dim
    ).transpose(0, 2, 1, 3, 4)

    groups = num_query_heads // num_key_value_heads
    grouped = queries.transpose(0, 2, 1, 3).reshape(
        batch, length, num_key_value_heads, groups, head_dim
    )
    attn = (
        grouped.astype(mx.float32)
        @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
    ) / math.sqrt(head_dim)
    attn = mx.where(
        token_valid[:, :, None, None, :], attn, mx.finfo(attn.dtype).min
    )
    probabilities = mx.softmax(attn, axis=-1).astype(queries.dtype)
    output = probabilities @ selected_values
    return output.reshape(batch, length, num_query_heads, head_dim)


# --------------------------------------------------------------------------
# route (a): per-row slices through the stock single-sequence kernels
# --------------------------------------------------------------------------


def looped_gathered_qsa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    pooled_index_keys: mx.array,
    bank: PooledBank,
    geo: BatchGeometry,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
    index_keys: Optional[mx.array] = None,
    index_position_ids: Optional[mx.array] = None,
) -> mx.array:
    """Same contract as :func:`padded_gathered_qsa`, one stock call per row.

    The row slices stay views: materializing a 130k row's K/V would cost more
    than the attention it feeds.
    """
    qsa = _qsa_module()
    batch, _, length, _ = queries.shape
    rows = []
    for b in range(batch):
        pad = geo.pads[b]
        off = bank.offsets[b]
        row_keys = keys[b : b + 1, :, pad : geo.width, :]
        row_values = values[b : b + 1, :, pad : geo.width, :]
        row_pooled = pooled_index_keys[b : b + 1, off : off + bank.counts[b]]
        if length == 1:
            rows.append(
                qsa.contiguous_causal_gathered_qsa_decode(
                    queries[b : b + 1],
                    row_keys,
                    row_values,
                    index_queries[b : b + 1],
                    row_pooled,
                    num_query_heads=num_query_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    indexer_head_dim=indexer_head_dim,
                    compress_ratio=compress_ratio,
                    token_budget=token_budget,
                )
            )
            continue
        if index_keys is None or index_position_ids is None:
            raise ValueError("multi-row loop route needs the raw indexer state")
        if index_position_ids.ndim == 3:
            row_positions = index_position_ids[:, b : b + 1, pad : geo.width]
        else:
            row_positions = index_position_ids[b : b + 1, pad : geo.width]
        rows.append(
            qsa.contiguous_causal_gathered_qsa(
                queries[b : b + 1],
                row_keys,
                row_values,
                index_queries[b : b + 1],
                index_keys[b : b + 1, pad : geo.width],
                row_positions,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                indexer_head_dim=indexer_head_dim,
                compress_ratio=compress_ratio,
                token_budget=token_budget,
                index_key_norm=None,
                apply_index_rope=None,
                pooled_index_keys=row_pooled,
            )
        )
    return mx.concatenate(rows, axis=0)


def gathered_qsa_batched(*args, route: Optional[str] = None, **kwargs) -> mx.array:
    """Dispatch to the configured route. Both are numerically interchangeable."""
    chosen = route or batched_route()
    if chosen == "loop":
        return looped_gathered_qsa(*args, **kwargs)
    kwargs.pop("index_keys", None)
    kwargs.pop("index_position_ids", None)
    return padded_gathered_qsa(*args, **kwargs)


__all__ = [
    "BatchGeometry",
    "batch_geometry",
    "batched_pooled_index_keys",
    "batched_route",
    "gather_min_context",
    "gathered_qsa_batched",
    "host_left_padding",
    "looped_gathered_qsa",
    "padded_gathered_qsa",
]
