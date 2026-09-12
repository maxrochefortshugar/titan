# SPDX-License-Identifier: MIT
"""``verify_accept``: acceptance and the bonus token, computed in the graph.

This is the op the decode cycle's single host sync is made of. Given the target
model's logits over one verify row block and the tokens that were drafted, it
returns **one small integer array**: the number of accepted drafts per row
followed by the bonus token per row. Nothing else crosses to the host, and it
crosses once.

    counts_and_bonus = verify_accept(logits, drafted)      # int32 [2R]
    host = counts_and_bonus.tolist()                       # the only sync

Why it is an op rather than three lines in the backend. The three lines are
``argmax``, ``cumprod``, ``sum``, and each is a launch over a block whose last
dimension is 248,320 wide; at depth 3 and eight rows that is a real amount of
bandwidth spent to produce sixteen integers. Behind the registry, the fused
Metal version replaces all of it with one launch without the cycle knowing.
Today the fast path is the reference, so selection is a no-op and the numbers
cannot move.

## Greedy

Compare each drafted id against the target's argmax at the position that
predicted it, take the cumulative product along the draft axis so acceptance
stops at the first mismatch, and sum. The bonus is the target's argmax at the
first rejection point, which is exactly the token plain greedy decoding would
have produced there. Committing ``drafted[:n] + (bonus,)`` is therefore
identical to greedy decoding, for every acceptance pattern -- the lossless
property, true by construction rather than by measurement.

## Stochastic

Exact rejection sampling (Leviathan et al.), also single-sync. A draft is
accepted when ``u <= p_target(x) / q_draft(x)``; the first rejection draws from
the normalised residual ``max(p - q, 0)``, and a fully accepted chain draws its
bonus from the target distribution at the last position. The residual is
computed for **every** position, not only the one that turned out to reject,
because a residual computed after the host learned where the rejection was
would need a second sync. Sampling is inverse-CDF against a uniform supplied by
the caller, so the whole thing is deterministic given its inputs and can be
tested against numpy.

## Shapes

``logits`` is ``[R, k+1, V]`` and ``drafted`` is ``[R, k]``: R rows in lockstep,
k drafted tokens each, plus the one extra column that produces the bonus. The
reference already handles R > 1, which is most of why lockstep batched verify
is a local change in the backend rather than a change here.
"""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["OP", "key", "reference", "fast", "supports", "split_host"]

_EPS = 1e-30


def reference(
    logits,
    drafted,
    *,
    accept_uniforms=None,
    residual_uniforms=None,
    draft_probs=None,
    temperature: float = 1.0,
):
    """Acceptance counts and bonus tokens as one ``int32`` array of length ``2R``.

    Greedy when ``accept_uniforms`` is ``None``. Otherwise exact rejection
    sampling, which additionally needs ``draft_probs`` (the drafter's
    distribution over the same positions) and ``residual_uniforms`` (one uniform
    per row per position, including the bonus position).

    Every branch here is host-side and depends only on shapes, so the graph this
    builds is the same graph every cycle and nothing in it reads a value.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [rows, width, vocab], got {logits.shape}")
    rows, width, vocab = logits.shape
    drafted = drafted.reshape(rows, width - 1).astype(mx.int32)

    if accept_uniforms is None:
        counts, bonus = _greedy(logits, drafted)
    else:
        if draft_probs is None or residual_uniforms is None:
            raise ValueError(
                "stochastic acceptance needs draft_probs and residual_uniforms"
            )
        counts, bonus = _stochastic(
            logits,
            drafted,
            accept_uniforms=accept_uniforms,
            residual_uniforms=residual_uniforms,
            draft_probs=draft_probs,
            temperature=temperature,
        )
    # One array, one crossing. Concatenated rather than returned as a pair
    # because two arrays are two syncs however carefully they are evaluated.
    return mx.concatenate([counts, bonus]).astype(mx.int32)


def _greedy(logits, drafted):
    rows, width, _vocab = logits.shape
    targets = mx.argmax(logits, axis=-1).astype(mx.int32)
    agree = (targets[:, : width - 1] == drafted).astype(mx.int32)
    counts = mx.cumprod(agree, axis=1).sum(axis=1).astype(mx.int32)
    bonus = mx.take_along_axis(targets, counts[:, None], axis=1)[:, 0]
    return counts, bonus


def _stochastic(
    logits,
    drafted,
    *,
    accept_uniforms,
    residual_uniforms,
    draft_probs,
    temperature: float,
):
    rows, width, vocab = logits.shape
    k = width - 1
    p = mx.softmax(logits.astype(mx.float32) / max(temperature, _EPS), axis=-1)
    q = draft_probs.astype(mx.float32).reshape(rows, k, vocab)

    index = drafted[:, :, None]
    p_drafted = mx.take_along_axis(p[:, :k], index, axis=-1)[:, :, 0]
    q_drafted = mx.take_along_axis(q, index, axis=-1)[:, :, 0]
    ratio = p_drafted / mx.maximum(q_drafted, _EPS)
    accepted = (accept_uniforms.reshape(rows, k) <= ratio).astype(mx.int32)
    counts = mx.cumprod(accepted, axis=1).sum(axis=1).astype(mx.int32)

    # Residual for every position, so the sync stays single: the host learns
    # where the chain stopped and which token to take in the same readback.
    residual = mx.maximum(p[:, :k] - q, 0.0)
    residual = residual / mx.maximum(
        residual.sum(axis=-1, keepdims=True), _EPS
    )
    candidates = mx.concatenate([residual, p[:, k : k + 1]], axis=1)
    cdf = mx.cumsum(candidates, axis=-1)
    draws = residual_uniforms.reshape(rows, width, 1).astype(mx.float32)
    sampled = (cdf < draws).sum(axis=-1).astype(mx.int32)
    sampled = mx.minimum(sampled, vocab - 1)
    bonus = mx.take_along_axis(sampled, counts[:, None], axis=1)[:, 0]
    return counts, bonus


def fast(logits, drafted, **kwargs):
    """The accelerated path.

    It is the reference today. The fused ``mx.fast.metal_kernel`` that turns the
    argmax, the cumprod and the gather into one launch is a later change, and
    the seam exists now so that landing it is a registry selection rather than
    an edit to the decode cycle. Keeping the two identical also keeps the
    exactness test honest while the kernel does not exist: it asserts the pair
    agrees, and it will keep asserting it when they stop being the same code.
    """
    return reference(logits, drafted, **kwargs)


def split_host(values: Sequence[int], rows: int) -> tuple[list[int], list[int]]:
    """Unpack the readback: ``(counts, bonus_tokens)``.

    The backend calls this on the list the single ``.tolist()`` produced. It is
    here rather than in the backend so the packing order is defined in one file.
    """
    if len(values) != 2 * rows:
        raise ValueError(f"expected {2 * rows} values for {rows} rows, got {len(values)}")
    return list(values[:rows]), list(values[rows:])


def key(logits, drafted, **kwargs) -> ShapeClass:
    mode = "greedy" if kwargs.get("accept_uniforms") is None else "sampled"
    return shape_class(logits, drafted, extra=(mode,))


def supports(sc: ShapeClass) -> bool:
    if len(sc.shapes) != 2:
        return False
    logits, drafted = sc.shapes
    if len(logits) != 3 or len(drafted) != 2:
        return False
    rows, width, _vocab = logits
    return drafted[0] == rows and drafted[1] == width - 1 and width >= 1


OP = KernelOp(
    name="verify_accept",
    reference_fn=reference,
    fast_fn=fast,
    key=key,
    supports_key=supports,
    tolerance=0.0,
    shapes=(
        {"R": 1, "K": 0, "V": 4096},
        {"R": 1, "K": 3, "V": 4096},
        {"R": 8, "K": 3, "V": 32768},
        {"R": 8, "K": 8, "V": 32768},
    ),
    exactness="bit-identical: integer comparisons and a cumulative product",
    source="docs/architecture/DECISIONS.md D6; oMLX batch_generator.py:2984-3049",
)
