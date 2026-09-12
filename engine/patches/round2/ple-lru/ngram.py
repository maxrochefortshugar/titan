#!/usr/bin/env python3
"""Numpy reproduction of the qwen4_exp PLE n-gram row indices.

Mirrors, op for op, ``Qwen4ExpNGramEmbedding._ngram_indices`` and
``_shift_right_ignore_eos`` in
``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py``
(``LANG:2603-2656``), plus ``_build_layer_multipliers`` (``LANG:1884-1896``)
and ``_find_nth_prime_after`` (``LANG:1909-1915``).

Point of this file: build a realistic id stream for the row-cache benchmark
without loading the 70 GB model.  ``test_ngram.py`` checks it against the
vendored mlx code itself.

Config for Qwen3.8-Flash-Next (config.json, text_config):
    vocab_size 248320, ngram_size 3, heads_per_ngram 8 -> 16 heads,
    ngram_vocab_size_base 20000000, make_ngram_vocab_size_divisible_by 128,
    seed 1234 (config.py:55 default), eos 248044, ple_layer_ids [2] so the
    PLE lives on decoder layer 1 with ple_layer_index 0.
"""
from __future__ import annotations

import math

import numpy as np

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return np.array(out, dtype=np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class NGramIndexer:
    """Row indices for one PLE layer, pure numpy, batch size 1."""

    def __init__(
        self,
        vocab_size=248320,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        divisible_by=128,
        seed=1234,
        eos_token_id=248044,
        ple_layer_index=0,
    ):
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = self.context_len * heads_per_ngram
        self.eos_token_id = eos_token_id

        sizes, offsets, total = [], [], 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.sizes = np.array(sizes, dtype=np.int64)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.total_vocab_size = total
        self.padded_vocab_size = math.ceil(total / divisible_by) * divisible_by
        self.multipliers = build_layer_multipliers(
            vocab_size, ngram_size, ple_layer_index, seed
        )

    # LANG:2603-2623
    def _shift_right_ignore_eos(self, tokens: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return tokens
        seq_len = tokens.shape[0]
        positions = np.arange(seq_len, dtype=np.int64)
        eos_positions = np.where(tokens == self.eos_token_id, positions, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions)
        previous_eos = np.concatenate(
            [np.array([-1], dtype=np.int64), previous_eos_inclusive[:-1]]
        )
        segment_start = previous_eos + 1
        position_in_segment = positions - segment_start
        source_positions = positions - shift
        shifted = tokens[np.maximum(source_positions, 0)]
        valid = (position_in_segment >= shift) & (source_positions >= 0)
        return np.where(valid, shifted, self.eos_token_id)

    # LANG:2634-2656
    def ngram_indices(self, token_history: np.ndarray, length: int) -> np.ndarray:
        token_history = token_history.astype(np.int64)
        shifted = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[position] * self.multipliers[position])
            sizes = self.sizes[start:end]
            offsets = self.offsets[start:end]
            blocks.append((mixed[:, None] % sizes[None]) + offsets[None])
        return np.concatenate(blocks, axis=-1)[-length:]

    def stream(self, tokens: np.ndarray, chunk: int = 2048):
        """Yield (chunk_tokens, row_ids[n, 16]) exactly as chunked prefill does.

        The two-token carry is ``cache[3]`` in ``Qwen4ExpNGramEmbedding.__call__``
        (``LANG:2665-2672``); the first chunk starts from eos padding
        (``_previous_context``, ``LANG:2626-2632``).
        """
        tokens = tokens.astype(np.int64)
        context = np.full(self.context_len, self.eos_token_id, dtype=np.int64)
        for start in range(0, tokens.size, chunk):
            piece = tokens[start : start + chunk]
            history = np.concatenate([context, piece])
            yield piece, self.ngram_indices(history, piece.size)
            context = history[-self.context_len :].copy()
