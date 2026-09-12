# SPDX-License-Identifier: Apache-2.0
"""Where the gathered arm starts paying: single-sequence crossover sweep.

The stock gate engages the gathered arm at ``cache.offset + L > token_budget``
(2048). This prices the two alternatives per QSA layer at 2k, 4k, 8k and 16k
for L=1 and L=4, and sets the default for OMLX_QSA_GATHER_MIN_CTX.

The honest dense fallback is the indexer's bool-mask construction plus masked
SDPA, so both are measured; ``dense causal`` is the optimistic lower bound that
ignores the mask entirely.

Run only while ~/inference-server/staging/GPU_FREE exists.
"""

from __future__ import annotations

import math

from common import (
    BLOCK_BUDGET,
    BUDGET,
    D,
    H_KV,
    H_Q,
    ID,
    IN_H,
    LAYERS,
    Norm,
    RATIO,
    med_ms,
    mx,
    qsa,
    rope,
)


def indexer_mask(index_queries, pooled, length, key_tokens):
    """language.py:1204-1250, the mask the dense fallback has to build."""
    batch = index_queries.shape[0]
    max_blocks = pooled.shape[1]
    scores = index_queries.astype(mx.float32).transpose(0, 2, 1, 3) @ pooled[
        :, None
    ].astype(mx.float32).transpose(0, 1, 3, 2)
    scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(ID)
    query_ends = key_tokens - length + mx.arange(length) + 1
    complete = query_ends // RATIO
    valid = mx.arange(max_blocks)[None, None, :] < complete[None, :, None]
    scores = mx.where(valid, scores, -mx.inf)
    picked = mx.argpartition(scores, kth=-BLOCK_BUDGET, axis=-1)[..., -BLOCK_BUDGET:]
    hits = mx.put_along_axis(
        mx.zeros((batch, length, max_blocks), dtype=mx.bool_),
        picked,
        mx.array(True),
        axis=-1,
    )
    tokens = mx.repeat(hits, RATIO, axis=-1)
    complete_len = max_blocks * RATIO
    if complete_len < key_tokens:
        tokens = mx.concatenate(
            [tokens, mx.zeros((batch, length, key_tokens - complete_len), dtype=mx.bool_)],
            axis=-1,
        )
    columns = mx.arange(key_tokens)
    tail_start = complete * RATIO
    tail = (columns[None, None, :] >= tail_start[None, :, None]) & (
        columns[None, None, :] < query_ends[None, :, None]
    )
    return (tokens | tail)[:, None]


def run(key_tokens, length):
    mx.random.seed(5)
    queries = mx.random.normal((1, H_Q, length, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, H_KV, key_tokens, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, H_KV, key_tokens, D)).astype(mx.bfloat16)
    index_queries = mx.random.normal((1, length, IN_H, ID)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, key_tokens, ID)).astype(mx.bfloat16)
    positions = mx.arange(key_tokens, dtype=mx.int32).reshape(1, key_tokens)
    pooled = qsa.pool_completed_index_keys(
        index_keys,
        positions,
        compress_ratio=RATIO,
        index_key_norm=Norm(),
        apply_index_rope=rope,
    )
    mx.eval(queries, keys, values, index_queries, index_keys, positions, pooled)

    def sparse():
        if length == 1:
            return qsa.contiguous_causal_gathered_qsa_decode(
                queries,
                keys,
                values,
                index_queries,
                pooled,
                num_query_heads=H_Q,
                num_key_value_heads=H_KV,
                head_dim=D,
                indexer_head_dim=ID,
                compress_ratio=RATIO,
                token_budget=BUDGET,
            )
        return qsa.contiguous_causal_gathered_qsa(
            queries,
            keys,
            values,
            index_queries,
            index_keys,
            positions,
            num_query_heads=H_Q,
            num_key_value_heads=H_KV,
            head_dim=D,
            indexer_head_dim=ID,
            compress_ratio=RATIO,
            token_budget=BUDGET,
            index_key_norm=Norm(),
            apply_index_rope=rope,
            pooled_index_keys=pooled,
        )

    def dense_causal():
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=D**-0.5, mask="causal"
        )

    def mask_build():
        return indexer_mask(index_queries, pooled, length, key_tokens)

    prebuilt = indexer_mask(index_queries, pooled, length, key_tokens)
    mx.eval(prebuilt)

    def dense_masked():
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=D**-0.5, mask=prebuilt
        )

    def dense_full():
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=D**-0.5,
            mask=indexer_mask(index_queries, pooled, length, key_tokens),
        )

    result = {
        "sparse": med_ms(sparse),
        "dense_causal": med_ms(dense_causal),
        "mask": med_ms(mask_build),
        "dense_masked": med_ms(dense_masked),
        "dense_full": med_ms(dense_full),
    }
    mx.clear_cache()
    return result


def main():
    # One throwaway configuration: the first timed shape otherwise carries the
    # allocator and kernel-cache warmup and reads 2x high.
    run(4100, 1)
    run(4100, 4)
    print(
        f"{'ctx':>7} {'L':>2} | {'sparse':>8} {'dense+mask':>10} {'(mask':>7} "
        f"{'sdpa)':>7} {'dense causal':>12} | {'winner':>10} {'fwd sparse':>10} "
        f"{'fwd dense':>9}"
    )
    # 2052 rather than 2048: below 513 complete blocks the gathered arm has
    # nothing to select and the stock gate cannot fire either.
    for key_tokens in (2052, 4096, 8192, 16384):
        for length in (1, 4):
            r = run(key_tokens + length, length)
            full = r["dense_full"]
            winner = "sparse" if r["sparse"] < full else "DENSE"
            print(
                f"{key_tokens:>7} {length:>2} | {r['sparse']:>8.3f} {full:>10.3f} "
                f"{r['mask']:>7.3f} {r['dense_masked']:>7.3f} "
                f"{r['dense_causal']:>12.3f} | {winner:>10} "
                f"{r['sparse'] * LAYERS:>10.2f} {full * LAYERS:>9.2f}"
            )
    print(f"\npeak GPU MB: {mx.get_peak_memory() / 1e6:.0f}")


if __name__ == "__main__":
    main()
