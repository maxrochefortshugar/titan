#!/usr/bin/env python3
"""Structure of the real PLE id stream, and hit rates, with no disk I/O.

Everything here runs off the numpy reproduction of the model's n-gram hashing
(ngram.py, checked bit-exact against the vendored mlx code by test_ngram.py),
so it needs neither the model nor the 32 GB table.
"""
from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch as lru          # noqa: E402
from corpus import build     # noqa: E402
from ngram import NGramIndexer  # noqa: E402

CHUNK = 2048
STRIDE = 100


def interleaved_stream(tokens, prose_len, total, piece=8192, seed=0):
    """Alternate prose and code documents the way a served mix would."""
    prose, code = tokens[:prose_len], tokens[prose_len:]
    rng = np.random.default_rng(seed)
    out, cursor = [], 0
    p = c = 0
    while cursor < total:
        src, off = (prose, p) if len(out) % 2 == 0 else (code, c)
        take = min(piece, src.size - off, total - cursor)
        out.append(src[off:off + take])
        if len(out) % 2 == 1:
            p += take
        else:
            c += take
        cursor += take
        if p >= prose.size or c >= code.size:
            break
    return np.concatenate(out)[:total]


def chunk_ids(idx, tokens, chunk=CHUNK):
    return [ids.reshape(-1) for _, ids in idx.stream(tokens, chunk=chunk)]


def exact_lru_hit_rate(batches, capacity_rows):
    """True global LRU over the same per-batch unique id sequence."""
    cache = OrderedDict()
    hits = misses = 0
    for batch in batches:
        for row in np.unique(batch).tolist():
            if row in cache:
                cache.move_to_end(row)
                hits += 1
            else:
                misses += 1
                cache[row] = None
                if len(cache) > capacity_rows:
                    cache.popitem(last=False)
    return hits / (hits + misses)


def setassoc_hit_rate(batches, capacity_bytes, ways=16):
    cache = lru.SetAssocRowCache(capacity_bytes, STRIDE, ways=ways)
    rows = np.zeros((1 << 16, STRIDE), dtype=np.uint8)
    for batch in batches:
        uniq = np.unique(batch)
        sets, slots, hit = cache.get(uniq)
        cache.touch(slots[hit])
        miss = ~hit
        n = int(miss.sum())
        cache.hits += int(uniq.size - n)
        cache.misses += n
        if n:
            cache.put(uniq[miss], sets[miss], rows[:n] if n <= rows.shape[0]
                      else np.zeros((n, STRIDE), np.uint8))
    return cache


def main():
    info = build()
    tokens = info["tokens"]
    prefill_tokens = int(1e9) if False else 300_000
    stream = interleaved_stream(tokens, info["prose_len"], prefill_tokens + 60_000)
    idx = NGramIndexer()

    t0 = time.perf_counter()
    prefill = stream[:prefill_tokens]
    decode = stream[prefill_tokens:]
    batches = chunk_ids(idx, prefill)
    hash_seconds = time.perf_counter() - t0

    flat = np.concatenate(batches)
    uniq_all, counts = np.unique(flat, return_counts=True)
    per_chunk_unique = np.array([np.unique(b).size for b in batches])

    out = {
        "corpus": info["meta"],
        "prefill_tokens": int(prefill.size),
        "chunks": len(batches),
        "lookups": int(flat.size),
        "lookups_per_token": 16,
        "distinct_rows_touched": int(uniq_all.size),
        "distinct_rows_gb": uniq_all.size * STRIDE / 1e9,
        "mean_unique_per_2048_chunk": float(per_chunk_unique.mean()),
        "intra_chunk_dedup_factor": float(CHUNK * 16 / per_chunk_unique.mean()),
        "top1_row_count": int(counts.max()),
        "share_of_lookups_in_top_1pct_rows": float(
            np.sort(counts)[::-1][: max(1, uniq_all.size // 100)].sum() / counts.sum()),
        "hash_seconds_for_stream": hash_seconds,
    }

    # bigram vs trigram heads: heads 0-7 are bigram, 8-15 trigram (LANG:2641-2653)
    per_head = np.concatenate([b.reshape(-1, 16) for b in batches], axis=0)
    bg = per_head[:, :8].reshape(-1)
    tg = per_head[:, 8:].reshape(-1)
    out["distinct_bigram_rows"] = int(np.unique(bg).size)
    out["distinct_trigram_rows"] = int(np.unique(tg).size)
    out["bigram_lookups"] = int(bg.size)
    out["trigram_lookups"] = int(tg.size)
    out["bigram_reuse_factor"] = float(bg.size / np.unique(bg).size)
    out["trigram_reuse_factor"] = float(tg.size / np.unique(tg).size)

    # hit rate versus capacity, set-associative and exact LRU
    table = []
    for gb in (0.25, 0.5, 1.0, 2.0, 4.0):
        cap = int(gb * (1 << 30))
        cache = setassoc_hit_rate(batches, cap)
        stats = cache.stats()
        table.append({
            "gb": gb, "entries": stats["entries"], "hit_rate": stats["hit_rate"],
            "evictions": stats["evictions"], "fill": stats["fill"],
            "set_overflow_rows": stats["set_overflow_rows"],
        })
    out["setassoc_hit_rate_vs_capacity"] = table

    lru_table = []
    for gb in (0.25, 2.0):
        cap_rows = int(gb * (1 << 30)) // (STRIDE + 8)
        t = time.perf_counter()
        lru_table.append({"gb": gb, "capacity_rows": cap_rows,
                          "exact_lru_hit_rate": exact_lru_hit_rate(batches, cap_rows),
                          "seconds": time.perf_counter() - t})
    out["exact_lru_reference"] = lru_table

    for ways in (4, 8, 16, 32):
        cache = setassoc_hit_rate(batches, int(0.25 * (1 << 30)), ways=ways)
        out.setdefault("associativity_sweep_at_0.25gb", []).append(
            {"ways": ways, "hit_rate": cache.stats()["hit_rate"]})

    # ---- static hot set: train on the first 70%, evaluate on the last 30%
    split = int(len(batches) * 0.7)
    train = np.concatenate(batches[:split])
    evaluate = np.concatenate(batches[split:])
    trows, tcounts = np.unique(train, return_counts=True)
    order = np.argsort(tcounts)[::-1]
    hot_curve = []
    for k in (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
              2_000_000, 5_000_000, 10_000_000, 20_000_000):
        if k > trows.size:
            k_eff = trows.size
        else:
            k_eff = k
        hot = np.sort(trows[order[:k_eff]])
        covered = float(np.isin(evaluate, hot).mean())
        hot_curve.append({"rows_preloaded": int(k_eff),
                          "gb_of_rows": k_eff * STRIDE / 1e9,
                          "gb_if_capacity_reserved": k * (STRIDE + 8) / 1e9,
                          "static_hit_rate_on_held_out": covered})
        if k_eff < k:
            break
    out["static_hot_set"] = {
        "train_lookups": int(train.size), "eval_lookups": int(evaluate.size),
        "train_distinct_rows": int(trows.size), "curve": hot_curve,
    }
    # and what the adaptive cache gets on the same held-out slice
    warm = lru.SetAssocRowCache(int(2 * (1 << 30)), STRIDE)
    zeros = np.zeros((1 << 17, STRIDE), np.uint8)
    for phase, group in (("train", batches[:split]), ("eval", batches[split:])):
        if phase == "eval":
            warm.hits = warm.misses = 0
        for batch in group:
            uniq = np.unique(batch)
            sets, slots, hit = warm.get(uniq)
            warm.touch(slots[hit])
            miss = ~hit
            n = int(miss.sum())
            warm.hits += int(uniq.size - n)
            warm.misses += n
            if n:
                warm.put(uniq[miss], sets[miss],
                         zeros[:n] if n <= zeros.shape[0] else np.zeros((n, STRIDE), np.uint8))
    out["static_hot_set"]["adaptive_2gb_hit_rate_on_same_eval"] = warm.stats()["hit_rate"]

    # cross-domain: does a hot set trained on prose transfer to code?
    prose_only = interleaved_stream(tokens[:info["prose_len"]], info["prose_len"],
                                    150_000, piece=10 ** 9)
    code_only = tokens[info["prose_len"]:info["prose_len"] + 150_000]
    pb = np.concatenate(chunk_ids(idx, prose_only))
    cb = np.concatenate(chunk_ids(idx, code_only))
    cross = []
    for name, train_rows, eval_rows in (("prose->code", pb, cb),
                                        ("code->prose", cb, pb),
                                        ("prose->prose_heldout", pb[:len(pb) // 2], pb[len(pb) // 2:]),
                                        ("code->code_heldout", cb[:len(cb) // 2], cb[len(cb) // 2:])):
        rows, counts = np.unique(train_rows, return_counts=True)
        order2 = np.argsort(counts)[::-1]
        row = {"pair": name, "train_lookups": int(train_rows.size)}
        for k in (100_000, 1_000_000):
            k_eff = min(k, rows.size)
            hot = np.sort(rows[order2[:k_eff]])
            row[f"hit_rate_top_{k}"] = float(np.isin(eval_rows, hot).mean())
            row[f"gb_top_{k}"] = k_eff * STRIDE / 1e9
        cross.append(row)
    out["static_hot_set"]["cross_domain"] = cross

    # ---- decode: 16 lookups per step, cache already warm from prefill
    dec_batches = chunk_ids(idx, decode[:4096], chunk=1)
    warm2 = lru.SetAssocRowCache(int(2 * (1 << 30)), STRIDE)
    zeros16 = np.zeros((16, STRIDE), np.uint8)
    for batch in batches:
        uniq = np.unique(batch)
        sets, slots, hit = warm2.get(uniq)
        warm2.touch(slots[hit])
        miss = ~hit
        n = int(miss.sum())
        if n:
            warm2.put(uniq[miss], sets[miss], np.zeros((n, STRIDE), np.uint8))
    warm2.hits = warm2.misses = 0
    for batch in dec_batches:
        uniq = np.unique(batch)
        sets, slots, hit = warm2.get(uniq)
        warm2.touch(slots[hit])
        miss = ~hit
        n = int(miss.sum())
        warm2.hits += int(uniq.size - n)
        warm2.misses += n
        if n:
            warm2.put(uniq[miss], sets[miss], zeros16[:n])
    out["decode"] = {
        "steps": len(dec_batches),
        "lookups_per_step": 16,
        "hit_rate_after_300k_prefill": warm2.stats()["hit_rate"],
    }
    print(json.dumps(out, indent=1))
    (HERE / "analyze.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
