#!/usr/bin/env python3
"""Cold-read cost of one 2048-token n-gram chunk against the reader thread count.

Replays exactly what ``PackedPLETable._prefetch_pages`` does
(``kernels/ple-fix/patch.py:150``): 32,768 n-gram row ids per 2048-token chunk
(16 lookups per token), de-duplicated to a page set, one 16 KB ``os.pread`` per
page through a thread pool, against the real ``layer1.rows.bin``.

Cold by construction: every repetition draws a fresh set of uniformly random
rows out of 320,001,536, and by default the file descriptor carries
``F_NOCACHE`` so nothing is served from, or left in, the page cache.  Resident
set stays a few MB (the reads land in a per-thread scratch buffer; nothing is
kept).  ``--cached`` drops F_NOCACHE for one comparison run, which does leave
~522 MB of reclaimable page cache per repetition.

No GPU, no model, no mlx.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

F_NOCACHE = 48
PAGE = os.sysconf("SC_PAGE_SIZE")
DEFAULT_BIN = os.path.expanduser(
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/"
    "ple-packed/layer1.rows.bin"
)
ROWS = 320_001_536
STRIDE = 100
ROWS_PER_CHUNK = 32_768          # 16 n-gram lookups per token x 2048 tokens


def page_set(rng, rows_per_chunk):
    ids = rng.integers(0, ROWS, size=rows_per_chunk, dtype=np.int64)
    off = ids * STRIDE
    pages = np.unique(np.concatenate((off // PAGE, (off + STRIDE - 1) // PAGE)))
    return pages


def touch_all(fd, pages, nworkers, pool):
    def touch(page):
        offset = int(page) * PAGE
        remaining = PAGE
        while remaining > 0:
            chunk = os.pread(fd, remaining, offset + (PAGE - remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    if nworkers == 1:
        for p in pages:
            touch(int(p))
    else:
        list(pool.map(touch, (int(p) for p in pages.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--workers", default="1,4,8,16,32,48,64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=ROWS_PER_CHUNK)
    ap.add_argument("--cached", action="store_true",
                    help="drop F_NOCACHE (leaves ~522 MB of page cache per rep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    size = os.path.getsize(a.bin)
    print(f"{a.bin}\n{size/1e9:.1f} GB, {a.rows} rows per chunk, "
          f"{'cached' if a.cached else 'F_NOCACHE'}, {a.reps} reps per setting")
    rng = np.random.default_rng(a.seed)

    fd = os.open(a.bin, os.O_RDONLY)
    if not a.cached:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    results = []
    try:
        print(f"\n{'workers':>8} {'pages':>8} {'MB':>7} {'median ms':>10} "
              f"{'min ms':>8} {'max ms':>8} {'GB/s':>7} {'us/page':>8} "
              f"{'vs 48':>7}")
        rows = {}
        for n in [int(s) for s in a.workers.split(",")]:
            pool = ThreadPoolExecutor(max_workers=n) if n > 1 else None
            ts, npages = [], 0
            try:
                for _ in range(a.reps):
                    pages = page_set(rng, a.rows)
                    npages = int(pages.size)
                    t0 = time.perf_counter()
                    touch_all(fd, pages, n, pool)
                    ts.append(time.perf_counter() - t0)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            med = st.median(ts)
            mb = npages * PAGE / 1e6
            rows[n] = med
            results.append({"workers": n, "pages": npages, "mb": mb,
                            "median_ms": med * 1e3, "all_ms": [t * 1e3 for t in ts]})
            print(f"{n:>8} {npages:>8} {mb:>7.1f} {med*1e3:>10.1f} "
                  f"{min(ts)*1e3:>8.1f} {max(ts)*1e3:>8.1f} "
                  f"{mb/1e3/med:>7.2f} {med*1e6/npages:>8.1f} "
                  f"{(rows.get(48, med)/med):>6.2f}x")
    finally:
        os.close(fd)

    best = min(results, key=lambda r: r["median_ms"])
    print(f"\nbest: {best['workers']} workers, {best['median_ms']:.1f} ms per "
          f"2048-token chunk")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"cached": a.cached, "rows_per_chunk": a.rows,
                       "results": results}, fh, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
