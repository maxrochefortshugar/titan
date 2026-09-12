# SPDX-License-Identifier: MIT
"""Exactness for ``verify_accept``.

The op is the decode cycle's single host sync, so what is tested is not a
tolerance but a claim: for every acceptance pattern, committing the accepted
drafts plus the bonus token is exactly what the target model would have emitted
on its own. Greedy is checked against a numpy transcription of that sentence;
the stochastic path is checked against a numpy transcription of Leviathan
rejection sampling with the same uniforms, so both sides are deterministic.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from titan.config.schema import KernelConfig
from titan.kernels import verify_accept as op
from titan.kernels.registry import KernelRegistry

pytestmark = pytest.mark.exactness


# ---------------------------------------------------------------------------
# numpy references
# ---------------------------------------------------------------------------


def numpy_greedy(logits: np.ndarray, drafted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, width, _vocab = logits.shape
    targets = logits.argmax(-1)
    counts = np.zeros(rows, dtype=np.int32)
    bonus = np.zeros(rows, dtype=np.int32)
    for r in range(rows):
        n = 0
        while n < width - 1 and targets[r, n] == drafted[r, n]:
            n += 1
        counts[r] = n
        bonus[r] = targets[r, n]
    return counts, bonus


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x.astype(np.float32) / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def numpy_sampled(
    logits: np.ndarray,
    drafted: np.ndarray,
    accept_u: np.ndarray,
    residual_u: np.ndarray,
    draft_probs: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, width, vocab = logits.shape
    k = width - 1
    p = _softmax(logits, temperature)
    q = draft_probs.astype(np.float32)
    counts = np.zeros(rows, dtype=np.int32)
    bonus = np.zeros(rows, dtype=np.int32)
    for r in range(rows):
        n = 0
        while n < k:
            token = drafted[r, n]
            ratio = p[r, n, token] / max(q[r, n, token], 1e-30)
            if accept_u[r, n] <= ratio:
                n += 1
            else:
                break
        counts[r] = n
        if n == k:
            distribution = p[r, k]
        else:
            residual = np.maximum(p[r, n] - q[r, n], 0.0)
            distribution = residual / max(residual.sum(), 1e-30)
        cdf = np.cumsum(distribution.astype(np.float32))
        bonus[r] = min(int((cdf < residual_u[r, n]).sum()), vocab - 1)
    return counts, bonus


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_logits(rows: int, width: int, vocab: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(rows, width, vocab)).astype(np.float32) * 4.0


def run(logits: np.ndarray, drafted: np.ndarray, **kwargs) -> tuple[list[int], list[int]]:
    out = op.reference(mx.array(logits), mx.array(drafted), **kwargs)
    mx.eval(out)
    return op.split_host(out.tolist(), logits.shape[0])


# ---------------------------------------------------------------------------
# greedy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", list(range(1, 9)))
def test_greedy_matches_numpy_over_random_drafts(k: int) -> None:
    rng = np.random.default_rng(1000 + k)
    vocab = 64
    for _ in range(24):
        rows = int(rng.integers(1, 5))
        logits = make_logits(rows, k + 1, vocab, rng)
        targets = logits.argmax(-1)
        # A mixture of right and wrong drafts, so every acceptance length in
        # 0..k shows up across the sweep rather than only the extremes.
        drafted = np.where(
            rng.random((rows, k)) < 0.6,
            targets[:, :k],
            rng.integers(0, vocab, size=(rows, k)),
        ).astype(np.int32)
        counts, bonus = run(logits, drafted)
        want_counts, want_bonus = numpy_greedy(logits, drafted)
        assert counts == want_counts.tolist()
        assert bonus == want_bonus.tolist()


@pytest.mark.parametrize("k", list(range(1, 9)))
def test_full_accept(k: int) -> None:
    rng = np.random.default_rng(7)
    vocab = 32
    logits = make_logits(2, k + 1, vocab, rng)
    drafted = logits.argmax(-1)[:, :k].astype(np.int32)
    counts, bonus = run(logits, drafted)
    assert counts == [k, k]
    assert bonus == logits.argmax(-1)[:, k].tolist()


@pytest.mark.parametrize("k", list(range(1, 9)))
def test_zero_accept(k: int) -> None:
    rng = np.random.default_rng(8)
    vocab = 32
    logits = make_logits(3, k + 1, vocab, rng)
    targets = logits.argmax(-1)
    drafted = ((targets[:, :k] + 1) % vocab).astype(np.int32)
    counts, bonus = run(logits, drafted)
    assert counts == [0, 0, 0]
    assert bonus == targets[:, 0].tolist()


@pytest.mark.parametrize("k", list(range(2, 9)))
@pytest.mark.parametrize("cut", [1, 2])
def test_mid_rejection(k: int, cut: int) -> None:
    """Reject at position ``cut``: everything before it is committed, the bonus
    is the target's own token there, and the drafts after it are discarded."""
    if cut >= k:
        pytest.skip("the rejection point has to be inside the chain")
    rng = np.random.default_rng(100 * k + cut)
    vocab = 48
    logits = make_logits(1, k + 1, vocab, rng)
    targets = logits.argmax(-1)
    drafted = targets[:, :k].copy()
    drafted[0, cut] = (targets[0, cut] + 1) % vocab
    counts, bonus = run(logits, drafted.astype(np.int32))
    assert counts == [cut]
    assert bonus == [int(targets[0, cut])]


def test_depth_zero_is_a_plain_decode_step() -> None:
    """k = 0 is the M1 loop: no drafts, no acceptance, argmax of one row."""
    rng = np.random.default_rng(11)
    logits = make_logits(4, 1, 128, rng)
    counts, bonus = run(logits, np.zeros((4, 0), dtype=np.int32))
    assert counts == [0, 0, 0, 0]
    assert bonus == logits.argmax(-1)[:, 0].tolist()


def test_rows_are_independent() -> None:
    """Batch composition may not change a row's own answer."""
    rng = np.random.default_rng(12)
    k, vocab = 4, 40
    logits = make_logits(6, k + 1, vocab, rng)
    targets = logits.argmax(-1)
    drafted = np.where(
        rng.random((6, k)) < 0.5, targets[:, :k], (targets[:, :k] + 3) % vocab
    ).astype(np.int32)
    batched_counts, batched_bonus = run(logits, drafted)
    for row in range(6):
        counts, bonus = run(logits[row : row + 1], drafted[row : row + 1])
        assert counts == [batched_counts[row]]
        assert bonus == [batched_bonus[row]]


# ---------------------------------------------------------------------------
# stochastic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", list(range(1, 9)))
def test_sampled_matches_numpy(k: int) -> None:
    rng = np.random.default_rng(2000 + k)
    rows, vocab, temperature = 3, 24, 0.8
    logits = make_logits(rows, k + 1, vocab, rng)
    draft_logits = make_logits(rows, k, vocab, rng)
    draft_probs = _softmax(draft_logits, 1.0)
    drafted = np.array(
        [
            [int(rng.choice(vocab, p=draft_probs[r, i])) for i in range(k)]
            for r in range(rows)
        ],
        dtype=np.int32,
    )
    accept_u = rng.random((rows, k)).astype(np.float32)
    residual_u = rng.random((rows, k + 1)).astype(np.float32)

    counts, bonus = run(
        logits,
        drafted,
        accept_uniforms=mx.array(accept_u),
        residual_uniforms=mx.array(residual_u),
        draft_probs=mx.array(draft_probs),
        temperature=temperature,
    )
    want_counts, want_bonus = numpy_sampled(
        logits, drafted, accept_u, residual_u, draft_probs, temperature
    )
    assert counts == want_counts.tolist()
    assert bonus == want_bonus.tolist()


def test_sampled_accepts_everything_when_the_draft_is_the_target() -> None:
    """q == p makes every ratio one, so any uniform below it accepts the whole
    chain: the ratio rule degenerates to the identity it should degenerate to."""
    rng = np.random.default_rng(31)
    rows, k, vocab = 2, 5, 20
    logits = make_logits(rows, k + 1, vocab, rng)
    probs = _softmax(logits[:, :k], 1.0)
    drafted = np.array(
        [[int(rng.choice(vocab, p=probs[r, i])) for i in range(k)] for r in range(rows)],
        dtype=np.int32,
    )
    counts, _bonus = run(
        logits,
        drafted,
        accept_uniforms=mx.full((rows, k), 0.99),
        residual_uniforms=mx.array(rng.random((rows, k + 1)).astype(np.float32)),
        draft_probs=mx.array(probs),
        temperature=1.0,
    )
    assert counts == [k, k]


def test_sampled_needs_its_inputs() -> None:
    with pytest.raises(ValueError):
        op.reference(
            mx.zeros((1, 2, 4)),
            mx.zeros((1, 1), dtype=mx.int32),
            accept_uniforms=mx.zeros((1, 1)),
        )


# ---------------------------------------------------------------------------
# the op and the registry
# ---------------------------------------------------------------------------


def test_fast_equals_reference() -> None:
    rng = np.random.default_rng(41)
    logits = mx.array(make_logits(4, 4, 64, rng))
    drafted = mx.array(rng.integers(0, 64, size=(4, 3)).astype(np.int32))
    assert op.fast(logits, drafted).tolist() == op.reference(logits, drafted).tolist()


def test_registry_dispatches_the_op() -> None:
    """The registry entry is one line in ``build_registry``; the op works behind
    it either way, which is what this checks."""
    registry = KernelRegistry(KernelConfig())
    registry.register(op.OP)
    rng = np.random.default_rng(42)
    logits = mx.array(make_logits(2, 3, 32, rng))
    drafted = mx.array(rng.integers(0, 32, size=(2, 2)).astype(np.int32))
    call = registry.resolve("verify_accept")
    assert call(logits, drafted).shape == (4,)
    assert registry.counters()["verify_accept.calls"] == 1
    assert "verify_accept.fallbacks" not in registry.counters()


def test_shape_guard_rejects_a_mismatched_block() -> None:
    with pytest.raises(ValueError):
        op.reference(mx.zeros((2, 4)), mx.zeros((2, 1), dtype=mx.int32))
    assert not op.supports(op.key(mx.zeros((2, 3, 8)), mx.zeros((2, 1), dtype=mx.int32)))


def test_split_host_is_the_packing_order() -> None:
    assert op.split_host([3, 1, 77, 88], 2) == ([3, 1], [77, 88])
    with pytest.raises(ValueError):
        op.split_host([1, 2, 3], 2)
