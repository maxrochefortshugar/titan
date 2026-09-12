# SPDX-License-Identifier: Apache-2.0
"""Shared synthetic harness: no model, no safetensors, no server."""

from __future__ import annotations

import importlib
import math
import statistics
import sys
import time

RES = "/Applications/oMLX.app/Contents/Resources"
sys.path.insert(0, RES + "/Python/framework-mlx-base/lib/python3.11/site-packages")
sys.path.insert(0, RES)

import mlx.core as mx  # noqa: E402
import mlx_vlm  # noqa: E402
import mlx_vlm.models  # noqa: E402

VENDOR = RES + "/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm"
if VENDOR not in mlx_vlm.__path__:
    mlx_vlm.__path__.append(VENDOR)
    mlx_vlm.models.__path__.append(VENDOR + "/models")

lang = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

QSAKVCache = lang.QSAKVCache
BatchQSAKVCache = lang.BatchQSAKVCache

# Qwen3.8-Flash-Next QSA geometry.
H_Q, H_KV, D = 24, 2, 256
ID, IN_H = 128, 4
RATIO, BUDGET = 4, 2048
BLOCK_BUDGET = BUDGET // RATIO
LAYERS = 12


class Norm:
    """Stand-in for the checkpoint k/q layernorm (weightless RMS)."""

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def rope(x, position_ids):
    """Position-sensitive interleaved rotary, so phase errors cannot hide.

    ``x`` is ``[B, 1, S, ID]`` and ``position_ids`` is ``[B, S]`` (or a 3-plane
    block whose first plane carries the text positions).
    """
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    half = x.shape[-1] // 2
    inv = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    angle = position_ids.astype(mx.float32)[:, None, :, None] * inv
    cos, sin = mx.cos(angle), mx.sin(angle)
    even = x[..., 0::2].astype(mx.float32)
    odd = x[..., 1::2].astype(mx.float32)
    out = mx.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
    return mx.flatten(out, start_axis=-2, end_axis=-1).astype(x.dtype)


def make_row(length, seed):
    """A warm single-sequence QSA cache plus its raw indexer history."""
    mx.random.seed(seed)
    cache = QSAKVCache()
    keys = mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, length, ID)).astype(mx.bfloat16)
    positions = mx.arange(length, dtype=mx.int32).reshape(1, length)
    cache.state = (keys, values, index_keys, positions)
    mx.eval(keys, values, index_keys, positions)
    return cache


def step_tensors(batch, length, seed):
    """One decode/verify step: queries and the new K/V and indexer rows."""
    mx.random.seed(seed)
    return dict(
        queries=mx.random.normal((batch, H_Q, length, D)).astype(mx.bfloat16),
        keys=mx.random.normal((batch, H_KV, length, D)).astype(mx.bfloat16),
        values=mx.random.normal((batch, H_KV, length, D)).astype(mx.bfloat16),
        index_queries=mx.random.normal((batch, length, IN_H, ID)).astype(mx.bfloat16),
        index_keys=mx.random.normal((batch, length, ID)).astype(mx.bfloat16),
    )


def med_ms(fn, reps=15, chain=3):
    """Median wall time per call, warm, chained to amortize submission."""
    out = fn()
    mx.eval(out)
    mx.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(chain):
            out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / chain)
    return statistics.median(samples)


def ulp_bf16(reference):
    """One bf16 ULP at the magnitude of ``reference``."""
    peak = float(mx.max(mx.abs(reference.astype(mx.float32))))
    if peak == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(peak)) - 7)
