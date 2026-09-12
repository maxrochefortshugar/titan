# SPDX-License-Identifier: Apache-2.0
"""Exactness of the batched sparse gathered QSA arm.

For every config we build B warm single-sequence ``QSAKVCache`` rows of mixed
length, merge them into the production ``BatchQSAKVCache`` (which left pads and
right aligns), run one decode/verify step, and compare three ways:

  [A] padded route (b) against the stock single-sequence kernels run per row.
  [B] loop route (a) against the same per-row reference.
  [C] padded route against dense SDPA over the row's whole cache masked to
      exactly the tokens QSA selected, with the selection reproduced
      independently.

Usage: test_exact_batched.py [--big]   (--big adds the 65k rows, ~1.2 GB)
"""

from __future__ import annotations

import argparse
import sys

from common import (  # noqa: E402
    BUDGET,
    BLOCK_BUDGET,
    BatchQSAKVCache,
    D,
    H_KV,
    H_Q,
    ID,
    IN_H,
    Norm,
    RATIO,
    make_row,
    mx,
    qsa,
    rope,
    step_tensors,
    ulp_bf16,
)

import batched_qsa as bq  # noqa: E402

TAG = object()


def prepare(lengths, length, seed):
    """Merge warm rows, apply one step to the batch, return everything needed."""
    rows = [make_row(n, 1000 + i) for i, n in enumerate(lengths)]
    batch = BatchQSAKVCache.merge(rows)
    step = step_tensors(len(lengths), length, seed)
    positions = mx.stack(
        [mx.arange(n, n + length, dtype=mx.int32) for n in lengths]
    )
    keys, values = batch.update_and_fetch(step["keys"], step["values"])
    batch.update_indexer(step["index_keys"], positions)

    normed = Norm()(step["index_queries"]).transpose(0, 2, 1, 3)
    index_queries = rope(normed, positions).transpose(0, 2, 1, 3)
    mx.eval(keys, values, index_queries)
    return batch, keys, values, index_queries, step, positions


def reference_rows(lengths, length, seed, step, positions, index_queries):
    """Per-row stock single-sequence sparse output, the definition of correct."""
    outputs = []
    caches = []
    for i, n in enumerate(lengths):
        cache = make_row(n, 1000 + i)
        keys, values = cache.update_and_fetch(
            step["keys"][i : i + 1], step["values"][i : i + 1]
        )
        cache.update_indexer(step["index_keys"][i : i + 1], positions[i : i + 1])
        pooled = cache.pooled_indexer_keys(RATIO, Norm(), rope, cache_tag=TAG)
        row_q = mx.contiguous(step["queries"][i : i + 1])
        row_iq = mx.contiguous(index_queries[i : i + 1])
        if length == 1:
            out = qsa.contiguous_causal_gathered_qsa_decode(
                row_q,
                keys,
                values,
                row_iq,
                pooled,
                num_query_heads=H_Q,
                num_key_value_heads=H_KV,
                head_dim=D,
                indexer_head_dim=ID,
                compress_ratio=RATIO,
                token_budget=BUDGET,
            )
        else:
            out = qsa.contiguous_causal_gathered_qsa(
                row_q,
                keys,
                values,
                row_iq,
                cache.index_keys,
                cache.index_position_ids,
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
        mx.eval(out)
        outputs.append(out)
        caches.append((cache, keys, values, pooled))
    return mx.concatenate(outputs, axis=0), caches


def dense_masked_row(cache, keys, values, pooled, query, index_query, n_total, length):
    """Dense SDPA masked to exactly the blocks QSA picks, selection redone here."""
    max_blocks = pooled.shape[1]
    rows = []
    for j in range(length):
        q_abs = n_total - length + j
        scores = qsa._portable_indexer_scores(index_query[:, j : j + 1], pooled, ID)
        complete = (q_abs + 1) // RATIO
        valid = mx.arange(max_blocks)[None, None, :] < complete
        scores = mx.where(valid, scores, mx.finfo(scores.dtype).min)
        if max_blocks > BLOCK_BUDGET:
            picked = mx.argpartition(scores, kth=-BLOCK_BUDGET, axis=-1)[
                ..., -BLOCK_BUDGET:
            ]
        else:
            picked = mx.broadcast_to(
                mx.arange(max_blocks, dtype=mx.int32)[None, None],
                (1, 1, max_blocks),
            )
        keep = min(int(complete), BLOCK_BUDGET)
        tokens = set()
        for block in mx.sort(picked.astype(mx.int32), axis=-1).reshape(-1).tolist()[
            :keep
        ]:
            tokens.update(range(block * RATIO, block * RATIO + RATIO))
        tokens.update(range(int(complete) * RATIO, q_abs + 1))
        marker = mx.zeros((n_total,))
        marker = marker.at[mx.array(sorted(tokens), dtype=mx.int32)].add(1.0)
        rows.append(marker > 0.5)
    mask = mx.stack(rows)[None, None]
    out = mx.fast.scaled_dot_product_attention(
        query, keys, values, scale=D**-0.5, mask=mask
    )
    return out.transpose(0, 2, 1, 3)


def run(lengths, length, seed=7, dense=True):
    batch, keys, values, index_queries, step, positions = prepare(
        lengths, length, seed
    )
    geo = bq.batch_geometry(batch)
    pooled, bank = bq.batched_pooled_index_keys(
        batch,
        geo,
        compress_ratio=RATIO,
        index_key_norm=Norm(),
        apply_index_rope=rope,
        cache_tag=TAG,
    )
    mx.eval(pooled)

    kwargs = dict(
        num_query_heads=H_Q,
        num_key_value_heads=H_KV,
        head_dim=D,
        indexer_head_dim=ID,
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )
    padded = bq.padded_gathered_qsa(
        step["queries"], keys, values, index_queries, pooled, bank, geo, **kwargs
    )
    looped = bq.looped_gathered_qsa(
        step["queries"],
        keys,
        values,
        index_queries,
        pooled,
        bank,
        geo,
        index_keys=batch.index_keys,
        index_position_ids=batch.index_position_ids,
        **kwargs,
    )
    mx.eval(padded, looped)

    reference, caches = reference_rows(
        lengths, length, seed, step, positions, index_queries
    )
    ulp = ulp_bf16(reference)

    def worst(candidate):
        return float(
            mx.max(
                mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
            )
        )

    err_a, err_b = worst(padded), worst(looped)

    err_c = 0.0
    err_c_ref = 0.0
    if dense:
        for i, n in enumerate(lengths):
            cache, row_keys, row_values, row_pooled = caches[i]
            ref = dense_masked_row(
                cache,
                row_keys,
                row_values,
                row_pooled,
                mx.contiguous(step["queries"][i : i + 1]),
                mx.contiguous(index_queries[i : i + 1]),
                n + length,
                length,
            )
            mx.eval(ref)
            err_c = max(
                err_c,
                float(
                    mx.max(
                        mx.abs(
                            padded[i : i + 1].astype(mx.float32)
                            - ref.astype(mx.float32)
                        )
                    )
                ),
            )
            # Same dense reference against the single-sequence arm: whatever
            # gap remains here is the stock arm's, not batching's.
            err_c_ref = max(
                err_c_ref,
                float(
                    mx.max(
                        mx.abs(
                            reference[i : i + 1].astype(mx.float32)
                            - ref.astype(mx.float32)
                        )
                    )
                ),
            )
    return err_a, err_b, err_c, err_c_ref, ulp, geo.pads


def multistep(lengths, steps, length=1, seed=21):
    """Walk several decode steps so the incremental bank update is exercised.

    A single step always rebuilds the bank from scratch. Only consecutive steps
    hit the slice-update path, and only after four of them has every row's
    block phase wrapped.
    """
    rows = [make_row(n, 1000 + i) for i, n in enumerate(lengths)]
    batched = BatchQSAKVCache.merge(rows)
    singles = [make_row(n, 1000 + i) for i, n in enumerate(lengths)]
    cursor = list(lengths)
    worst = 0.0
    scale = 0.0
    kwargs = dict(
        num_query_heads=H_Q,
        num_key_value_heads=H_KV,
        head_dim=D,
        indexer_head_dim=ID,
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )
    for step_index in range(steps):
        step = step_tensors(len(lengths), length, seed + step_index)
        positions = mx.stack(
            [mx.arange(n, n + length, dtype=mx.int32) for n in cursor]
        )
        normed = Norm()(step["index_queries"]).transpose(0, 2, 1, 3)
        index_queries = rope(normed, positions).transpose(0, 2, 1, 3)

        keys, values = batched.update_and_fetch(step["keys"], step["values"])
        batched.update_indexer(step["index_keys"], positions)
        geo = bq.batch_geometry(batched)
        pooled, bank = bq.batched_pooled_index_keys(
            batched,
            geo,
            compress_ratio=RATIO,
            index_key_norm=Norm(),
            apply_index_rope=rope,
            cache_tag=TAG,
        )
        out = bq.padded_gathered_qsa(
            step["queries"], keys, values, index_queries, pooled, bank, geo, **kwargs
        )
        mx.eval(out)

        reference = []
        for i, cache in enumerate(singles):
            row_keys, row_values = cache.update_and_fetch(
                step["keys"][i : i + 1], step["values"][i : i + 1]
            )
            cache.update_indexer(step["index_keys"][i : i + 1], positions[i : i + 1])
            row_pooled = cache.pooled_indexer_keys(RATIO, Norm(), rope, cache_tag=TAG)
            reference.append(
                qsa.contiguous_causal_gathered_qsa_decode(
                    mx.contiguous(step["queries"][i : i + 1]),
                    row_keys,
                    row_values,
                    mx.contiguous(index_queries[i : i + 1]),
                    row_pooled,
                    **kwargs,
                )
            )
        reference = mx.concatenate(reference, axis=0)
        mx.eval(reference)
        worst = max(
            worst,
            float(
                mx.max(mx.abs(out.astype(mx.float32) - reference.astype(mx.float32)))
            ),
        )
        scale = max(scale, ulp_bf16(reference))
        cursor = [n + length for n in cursor]
    return worst, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", action="store_true", help="include the 65k rows")
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()

    configs = [
        ([4096, 16384], 1),
        ([4096, 16384], 4),
        ([4093, 16382], 1),
        ([4093, 16382], 4),
        ([4096, 8192, 16384, 16381], 1),
        ([4096, 8192, 16384, 16381], 4),
    ]
    if args.big:
        configs += [
            ([16384, 65536], 1),
            ([16384, 65536], 4),
            ([4096, 16384, 65536, 65533], 1),
            ([4096, 16384, 65536, 65533], 4),
        ]

    print(
        f"{'lengths':>34} {'L':>2} | {'[A] padded':>11} {'[B] loop':>10} "
        f"{'[C] dense':>10} {'stock/dense':>11} | {'1 bf16 ULP':>10} {'pads':>18}"
    )
    failures = 0
    for lengths, length in configs:
        err_a, err_b, err_c, err_c_ref, ulp, pads = run(
            lengths, length, dense=not args.no_dense
        )
        # The bar that matters: the batched arms reproduce the per-sequence
        # single-row sparse output. [C] is a second opinion whose own error
        # floor (stock/dense) is printed beside it.
        bad = "" if max(err_a, err_b) <= ulp and err_c <= max(ulp, err_c_ref) else "   FAIL"
        failures += bool(bad)
        label = ",".join(str(n) for n in lengths)
        print(
            f"{label:>34} {length:>2} | {err_a:>11.3e} {err_b:>10.3e} "
            f"{err_c:>10.3e} {err_c_ref:>11.3e} | {ulp:>10.3e} {str(pads):>18}{bad}"
        )
        mx.clear_cache()
    print("\nincremental bank over consecutive decode steps")
    print(f"{'lengths':>34} {'steps':>6} | {'max abs':>10} {'1 bf16 ULP':>11}  verdict")
    for lengths, steps in (([4096, 16384], 9), ([4093, 16382, 12290, 16381], 9)):
        err, ulp = multistep(lengths, steps)
        ok = err <= ulp
        failures += not ok
        label = ",".join(str(n) for n in lengths)
        print(
            f"{label:>34} {steps:>6} | {err:>10.3e} {ulp:>11.3e}  "
            f"{'ok' if ok else 'FAIL'}"
        )
        mx.clear_cache()

    print(f"\npeak GPU MB: {mx.get_peak_memory() / 1e6:.0f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
