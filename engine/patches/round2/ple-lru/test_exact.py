#!/usr/bin/env python3
"""Bit-exactness of the cached PLE row reader against the uncached one.

Both readers are built on the same real ``layer1.rows.bin``.  For every batch
we compare the three packed planes byte for byte and the dequantized bf16
output bit for bit (viewed as uint16 so a NaN or a -0 cannot hide).  Batches
repeat so that the second and third pass answer out of the cache, and one batch
is small enough (<= 8 rows) to take the reader's no-prefetch branch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OMLX_PLE_LRU", "1")

from stub_io_pool import install_stub  # noqa: E402

install_stub()

import patch as lru  # noqa: E402
from ngram import NGramIndexer  # noqa: E402

MODEL = os.environ.get(
    "PROFILE_MODEL",
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")


def build_tables(capacity_bytes):
    base = lru.load_base_module()
    directory, manifest = base.load_manifest(MODEL)
    entry = manifest["layers"]["1"]
    plain = base.PackedPLETable(directory, entry, "rows")
    cached_cls = lru.make_cached_class(base.PackedPLETable)
    cached = base.PackedPLETable(directory, entry, "rows")
    cached.__class__ = cached_cls
    cached._lru = None
    cached.attach_cache(capacity_bytes=capacity_bytes)
    return plain, cached, entry


def bits(array: mx.array) -> np.ndarray:
    return np.asarray(array.view(mx.uint16))


def main():
    rng = np.random.default_rng(7)
    idx = NGramIndexer()
    # a realistic batch: 256 tokens of a repetitive sentence -> 4096 lookups
    text_tokens = np.tile(rng.integers(0, 40000, size=23, dtype=np.int64), 12)[:256]
    realistic = idx.ngram_indices(
        np.concatenate([np.full(2, 248044, dtype=np.int64), text_tokens]),
        text_tokens.size).reshape(-1)

    batches = {
        "random_4096": np.sort(rng.integers(0, 320001536, size=4096, dtype=np.int64)),
        "random_unsorted_2048": rng.integers(0, 320001536, size=2048, dtype=np.int64),
        "with_duplicates": np.repeat(
            rng.integers(0, 320001536, size=64, dtype=np.int64), 32),
        "ngram_realistic_4096": realistic,
        "decode_16": idx.ngram_indices(
            np.concatenate([np.full(2, 248044, dtype=np.int64),
                            rng.integers(0, 248320, size=1, dtype=np.int64)]), 1
        ).reshape(-1),
        "boundary_rows": np.array([0, 1, 319, 320001535, 320001534,
                                   160000000, 160000000], dtype=np.int64),
    }

    # a tiny cache so that eviction runs during the test
    plain, cached, entry = build_tables(capacity_bytes=512 << 10)
    results = []
    ok = True
    for name, host in batches.items():
        for attempt in range(3):
            want = plain.assemble_host(host)
            got = cached.assemble_host(host)
            planes_equal = all(np.array_equal(a, b) for a, b in zip(want, got))
            dq_want = plain.dequantize_host(want)
            dq_got = cached.dequantize_host(got)
            mx.eval(dq_want, dq_got)
            bit_equal = bool(np.array_equal(bits(dq_want), bits(dq_got)))
            scaled_want = dq_want.astype(mx.bfloat16) * entry["weight_scale"]
            scaled_got = dq_got.astype(mx.bfloat16) * entry["weight_scale"]
            mx.eval(scaled_want, scaled_got)
            scaled_equal = bool(np.array_equal(bits(scaled_want), bits(scaled_got)))
            diff = np.asarray(
                (dq_want.astype(mx.float32) - dq_got.astype(mx.float32)))
            ok &= planes_equal and bit_equal and scaled_equal
            results.append({
                "batch": name, "rows": int(host.size), "pass": attempt,
                "planes_byte_identical": planes_equal,
                "dequantized_bit_identical": bit_equal,
                "after_weight_scale_bit_identical": scaled_equal,
                "max_abs_diff": float(np.abs(diff).max()) if diff.size else 0.0,
                "ulp_diff": 0,
            })

    stats = cached.cache_stats()
    out = {
        "model": MODEL,
        "cache_bytes_reserved": stats["bytes_reserved"],
        "cache_entries": stats["entries"],
        "evictions_exercised": stats["evictions"] > 0,
        "cache_stats": stats,
        "results": results,
        "all_bit_exact": bool(ok),
    }
    print(json.dumps(out, indent=1))
    (HERE / "test_exact.json").write_text(json.dumps(out, indent=1))
    plain.close()
    cached.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
