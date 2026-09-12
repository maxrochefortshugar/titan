#!/usr/bin/env python3
"""Timing of the PLE row cache against the real 32 GB layer1.rows.bin.

Only the rows the id stream touches are ever read.  The system page cache is
never dropped (antisocial next to a 78 GB production model), so the three
regimes are separated like this:

  A. cold, paired, F_NOCACHE on both sides.  With the OS cache bypassed the two
     readers see the same SSD every time, so the difference is the row cache
     and nothing else.
  B. page-cache warm, with the deployed mmap reader: pass 1 cold, pass 2 warm,
     then the cached reader over the same ids, LRU cold then LRU warm.
  C. decode, 16 lookups per step, after a realistic prefill warm-up.

Configurations run pread/nocache first and mmap last so that the resident-set
figures are not polluted by the 32 GB mapping.
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stub_io_pool import install_stub  # noqa: E402

install_stub()
os.environ.setdefault("OMLX_PLE_LRU", "1")

import mlx.core as mx             # noqa: E402
import patch as lru               # noqa: E402
from analyze import chunk_ids, interleaved_stream  # noqa: E402
from corpus import build          # noqa: E402
from ngram import NGramIndexer    # noqa: E402

MODEL = os.environ.get(
    "PROFILE_MODEL",
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")
SEG = int(os.environ.get("BENCH_CHUNKS", "12"))
WARM_CHUNKS = int(os.environ.get("BENCH_WARM_CHUNKS", "40"))
DECODE_STEPS = int(os.environ.get("BENCH_DECODE_STEPS", "512"))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def vm():
    import subprocess
    text = __import__("subprocess").run(["vm_stat"], capture_output=True,
                                        text=True).stdout
    got = {}
    for line in text.splitlines():
        for key, name in (("Pages free", "free_gb"), ("Pages inactive", "inactive_gb"),
                          ("Pages active", "active_gb")):
            if line.startswith(key):
                got[name] = round(int(line.split()[-1].strip(".")) * 16384 / 1e9, 2)
    return got


def timed_pass(table, batches):
    times = []
    for batch in batches:
        t0 = time.perf_counter()
        planes = table.assemble_host(batch)
        times.append((time.perf_counter() - t0) * 1e3)
        del planes
    return times


def summarize(times, extra=None):
    out = {
        "n": len(times),
        "ms_median": round(statistics.median(times), 4),
        "ms_mean": round(sum(times) / len(times), 4),
        "ms_min": round(min(times), 4),
        "ms_max": round(max(times), 4),
        "ms_p90": round(float(np.percentile(times, 90)), 4),
    }
    if extra:
        out.update(extra)
    return out


def delta(before, after):
    dh = after["hits"] - before["hits"]
    dm = after["misses"] - before["misses"]
    return {"hit_rate": round(dh / (dh + dm), 4) if dh + dm else 0.0,
            "rows_from_ssd": after["miss_rows_read"] - before["miss_rows_read"]}


def make(base, directory, entry, capacity_bytes, reader, bypass=False):
    table = base.PackedPLETable(directory, entry, "rows")
    if reader == "deployed":
        return table
    table.__class__ = lru.make_cached_class(base.PackedPLETable)
    table._lru = None
    table.attach_cache(capacity_bytes=capacity_bytes, reader=reader, bypass=bypass)
    return table


ZERO = {"hits": 0, "misses": 0, "miss_rows_read": 0}


def main():
    gb = float(os.environ.get("OMLX_PLE_LRU_GB", "2"))
    cap = int(gb * (1 << 30))
    info = build()
    idx = NGramIndexer()
    stream = interleaved_stream(info["tokens"], info["prose_len"], 300_000)
    batches = chunk_ids(idx, stream)
    seg_a = batches[0:SEG]
    seg_b = batches[SEG:2 * SEG]
    warm = batches[2 * SEG:2 * SEG + WARM_CHUNKS]
    dec_start = (2 * SEG + WARM_CHUNKS) * 2048
    dec = chunk_ids(idx, stream[dec_start:dec_start + DECODE_STEPS], chunk=1)

    base = lru.load_base_module()
    directory, entry = (lambda d, m: (d, m["layers"]["1"]))(*base.load_manifest(MODEL))

    out = {
        "model": MODEL, "capacity_gb_requested": gb, "segment_chunks": SEG,
        "warm_chunks": WARM_CHUNKS, "decode_steps": DECODE_STEPS,
        "lookups_per_chunk": 2048 * 16, "lookups_per_decode_step": 16,
        "vm_before": vm(), "rss_start_gb": round(rss_gb(), 3),
        "A_cold_paired_nocache": {}, "B_pagecache": {}, "C_decode": {}, "memory": {},
    }

    # ---------------- A. cold, paired, F_NOCACHE on every side.
    # Three readers over the same ids, interleaved over REPS rounds so that
    # SSD drift hits all three equally; the cache is cleared before each of its
    # cold rounds.  F_NOCACHE means no round is ever page-cache warm.
    reps = int(os.environ.get("BENCH_COLD_REPS", "3"))
    a_raw = make(base, directory, entry, 0, "nocache", bypass=True)
    a_dedup = make(base, directory, entry, 0, "nocache")
    a_cached = make(base, directory, entry, cap, "nocache")
    acc = {"no_cache_no_dedup": [], "dedup_only": [], "cache_cold": []}
    for _ in range(reps):
        acc["no_cache_no_dedup"] += timed_pass(a_raw, seg_a)
        a_dedup._lru.clear()
        acc["dedup_only"] += timed_pass(a_dedup, seg_a)
        a_cached._lru.clear()
        before = a_cached.cache_stats()
        acc["cache_cold"] += timed_pass(a_cached, seg_a)
        cold_delta = delta(before, a_cached.cache_stats())
    for name, times in acc.items():
        out["A_cold_paired_nocache"][name] = summarize(times)
    out["A_cold_paired_nocache"]["cache_cold"].update(cold_delta)
    out["A_cold_paired_nocache"]["dedup_only"]["rows_from_ssd_per_chunk"] = round(
        a_dedup.cache_stats()["miss_rows_read"] / (reps * len(seg_a)))
    out["A_cold_paired_nocache"]["no_cache_no_dedup"]["rows_from_ssd_per_chunk"] = 2048 * 16
    before = a_cached.cache_stats()
    out["A_cold_paired_nocache"]["cache_lru_warm"] = summarize(
        timed_pass(a_cached, seg_a), delta(before, a_cached.cache_stats()))
    out["A_cold_paired_nocache"]["reps"] = reps
    out["memory"]["rss_after_A_gb"] = round(rss_gb(), 3)

    # ---------------- C. decode, warmed by a realistic prefill (pread reader)
    c_cached = make(base, directory, entry, cap, "pread")
    warm_times = timed_pass(c_cached, warm)
    warm_stats = c_cached.cache_stats()
    out["C_decode"]["prefill_warmup"] = summarize(
        warm_times, {"chunks": len(warm), "tokens": len(warm) * 2048,
                     "hit_rate": round(warm_stats["hit_rate"], 4)})
    before = c_cached.cache_stats()
    out["C_decode"]["cached_after_prefill"] = summarize(
        timed_pass(c_cached, dec), delta(before, c_cached.cache_stats()))
    before = c_cached.cache_stats()
    out["C_decode"]["cached_lru_warm"] = summarize(
        timed_pass(c_cached, dec), delta(before, c_cached.cache_stats()))
    c_plain = make(base, directory, entry, 0, "nocache", bypass=True)
    out["C_decode"]["no_cache_cold_nocache"] = summarize(timed_pass(c_plain, dec))
    out["C_decode"]["no_cache_cold_nocache_rep2"] = summarize(timed_pass(c_plain, dec))
    out["memory"]["rss_after_C_gb"] = round(rss_gb(), 3)
    out["memory"]["cache_reserved_gb"] = round(c_cached.cache_stats()["gb_reserved"], 3)
    out["memory"]["cache_fill_after_warmup"] = round(
        c_cached.cache_stats()["fill"], 4)

    # ---------------- B. the deployed mmap reader and the OS page cache
    b_plain = make(base, directory, entry, cap, "deployed")
    out["B_pagecache"]["deployed_mmap_cold"] = summarize(
        timed_pass(b_plain, seg_b),
        {"pages_read": b_plain.pages_read,
         "page_bytes_gb": round(b_plain.pages_read * 16384 / 1e9, 2)})
    out["B_pagecache"]["deployed_mmap_pagecache_warm"] = summarize(
        timed_pass(b_plain, seg_b))
    b_cached = make(base, directory, entry, cap, "pread")
    out["B_pagecache"]["cached_lru_cold_pagecache_warm"] = summarize(
        timed_pass(b_cached, seg_b), delta(ZERO, b_cached.cache_stats()))
    before = b_cached.cache_stats()
    out["B_pagecache"]["cached_lru_warm"] = summarize(
        timed_pass(b_cached, seg_b), delta(before, b_cached.cache_stats()))
    dec_b = chunk_ids(idx, stream[2 * SEG * 2048:2 * SEG * 2048 + DECODE_STEPS], chunk=1)
    out["B_pagecache"]["decode_deployed_mmap_cold"] = summarize(timed_pass(b_plain, dec_b))
    out["B_pagecache"]["decode_deployed_mmap_pagecache_warm"] = summarize(
        timed_pass(b_plain, dec_b))
    out["memory"]["rss_after_B_mmap_gb"] = round(rss_gb(), 3)

    # ---------------- the rest of the lookup call: upload + dequantize
    for name, ids in (("chunk", seg_a[0]), ("decode_step", dec[0])):
        planes = c_cached.assemble_host(ids)
        times = []
        for _ in range(15):
            mx.synchronize()
            t = time.perf_counter()
            values = c_cached.dequantize_host(planes).astype(mx.bfloat16) * entry["weight_scale"]
            mx.eval(values)
            mx.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        out[f"upload_and_dequantize_{name}_ms"] = summarize(times)

    out["memory"]["rss_peak_gb"] = round(rss_gb(), 3)
    out["vm_after"] = vm()
    print(json.dumps(out, indent=1))
    (HERE / "bench.json").write_text(json.dumps(out, indent=1))
    for table in (a_raw, a_dedup, a_cached, c_cached, c_plain, b_plain, b_cached):
        table.close()


if __name__ == "__main__":
    main()
