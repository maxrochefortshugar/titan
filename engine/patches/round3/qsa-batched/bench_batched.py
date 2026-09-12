# SPDX-License-Identifier: Apache-2.0
"""Batched decode/verify: today's dense path against the sparse gathered arm.

Per QSA layer, at B = 1, 2, 4 and contexts 65k and 130k. Everything is
synthetic; no model is loaded.

Today's batched path costs, per layer per step:
  repool  the whole indexer history, because BatchQSAKVCache has no pooled
          block bank and language.py:1189 falls through to
          pool_completed_index_keys over all of index_keys
  mask    the QSA bool mask, built and then discarded for left-padded decode
          (language.py:1628 keeps neither the string mask nor the QSA mask)
  sdpa    dense attention over the whole padded cache

The sparse arm costs an incremental one-block-per-row pooling plus the gathered
attention.

Run only while ~/inference-server/staging/GPU_FREE exists.
"""

from __future__ import annotations

import argparse
import gc

import batched_qsa as bq
from bench_crossover import indexer_mask
from common import (
    BUDGET,
    D,
    H_KV,
    H_Q,
    ID,
    IN_H,
    LAYERS,
    Norm,
    RATIO,
    lang,
    med_ms,
    mx,
    qsa,
    rope,
)

TAG = object()


def build(batch, context, spread=97):
    """A warm BatchQSAKVCache with mixed row lengths, already stepped."""
    width = context
    lengths = [context - i * spread for i in range(batch)]
    pads = [width - n for n in lengths]
    cache = lang.BatchQSAKVCache(pads)
    mx.random.seed(9)
    keys = mx.random.normal((batch, H_KV, width, D)).astype(mx.bfloat16)
    values = mx.random.normal((batch, H_KV, width, D)).astype(mx.bfloat16)
    index_keys = mx.random.normal((batch, width, ID)).astype(mx.bfloat16)
    positions = mx.broadcast_to(
        mx.arange(width, dtype=mx.int32)[None], (batch, width)
    )
    cache.state = (
        (
            keys,
            values,
            mx.array(lengths, dtype=mx.int32),
            mx.array(pads, dtype=mx.int32),
        ),
        index_keys,
        positions,
    )
    cache.kv_cache._idx = width
    mx.eval(keys, values, index_keys, positions)
    return cache, keys, values, lengths


def run(batch, context, length):
    cache, keys, values, lengths = build(batch, context)
    geo = bq.batch_geometry(cache)
    index_queries = mx.random.normal((batch, length, IN_H, ID)).astype(mx.bfloat16)
    queries = mx.random.normal((batch, H_Q, length, D)).astype(mx.bfloat16)
    pooled, bank = bq.batched_pooled_index_keys(
        cache,
        geo,
        compress_ratio=RATIO,
        index_key_norm=Norm(),
        apply_index_rope=rope,
        cache_tag=TAG,
    )
    mx.eval(pooled, index_queries, queries)

    kwargs = dict(
        num_query_heads=H_Q,
        num_key_value_heads=H_KV,
        head_dim=D,
        indexer_head_dim=ID,
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )

    def padded():
        return bq.padded_gathered_qsa(
            queries, keys, values, index_queries, pooled, bank, geo, **kwargs
        )

    def loop():
        return bq.looped_gathered_qsa(
            queries,
            keys,
            values,
            index_queries,
            pooled,
            bank,
            geo,
            index_keys=cache.index_keys,
            index_position_ids=cache.index_position_ids,
            **kwargs,
        )

    # Steady state: rewind one slot per row and let the bank catch up, which is
    # exactly what a decode step costs it.
    previous = [c - 1 for c in bank.counts]

    def pool_incremental():
        bank.counts = list(previous)
        fresh, _ = bq.batched_pooled_index_keys(
            cache,
            geo,
            compress_ratio=RATIO,
            index_key_norm=Norm(),
            apply_index_rope=rope,
            cache_tag=TAG,
        )
        return fresh

    def repool_full():
        return qsa.pool_completed_index_keys(
            cache.index_keys,
            cache.index_position_ids,
            compress_ratio=RATIO,
            index_key_norm=Norm(),
            apply_index_rope=rope,
        )

    def mask_build():
        return indexer_mask(index_queries, pooled, length, geo.width)

    def dense_sdpa():
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=D**-0.5, mask="causal" if length > 1 else None
        )

    result = {
        "padded": med_ms(padded),
        "loop": med_ms(loop),
        "pool_inc": med_ms(pool_incremental),
        "repool": med_ms(repool_full),
        "mask": med_ms(mask_build),
        "sdpa": med_ms(dense_sdpa),
    }
    del cache, keys, values, index_queries, queries, pooled, bank
    gc.collect()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    # Throwaway configuration: the first timed shape otherwise absorbs the
    # allocator and kernel-cache warmup.
    run(2, 8192, 1)
    run(2, 8192, 4)
    print(
        f"{'ctx':>7} {'B':>2} {'L':>2} | {'dense total':>11} "
        f"{'(repool':>8} {'mask':>6} {'sdpa)':>6} | {'sparse pad':>10} "
        f"{'sparse loop':>11} {'(pool':>6} | {'fwd dense':>9} {'fwd sparse':>10} "
        f"{'speedup':>7}"
    )
    for context in args.contexts:
        for batch in args.batches:
            for length in args.lengths:
                r = run(batch, context, length)
                dense = r["repool"] + r["mask"] + r["sdpa"]
                sparse = r["pool_inc"] + min(r["padded"], r["loop"])
                print(
                    f"{context:>7} {batch:>2} {length:>2} | {dense:>11.3f} "
                    f"{r['repool']:>8.3f} {r['mask']:>6.3f} {r['sdpa']:>6.3f} | "
                    f"{r['padded']:>10.3f} {r['loop']:>11.3f} "
                    f"{r['pool_inc']:>6.3f} | {dense * LAYERS:>9.2f} "
                    f"{sparse * LAYERS:>10.2f} {dense / sparse:>7.2f}x"
                )
            print()
    print(f"peak GPU MB: {mx.get_peak_memory() / 1e6:.0f}")


if __name__ == "__main__":
    main()
