# ple-lru: a RAM row cache in front of the packed PLE n-gram table

Round 2, 2026-09-12. Audit reference: section A (PLE table lookup, 6 ms warm /
456 ms cold per chunk, ~2.3 ms per decode forward) and section D item 6.

## What was built

`patch.py` subclasses the deployed reader and replaces
`PackedPLETable.assemble_host` (`~/inference-server/kernels/ple-fix/patch.py:178-191`)
and, in the default read mode, `_prefetch_pages` (`:150-176`). Nothing in
ple-fix is edited: `install()` rebinds the module attribute the deployed patch
instantiates and promotes tables that already exist, so `packed_call` (`:282-305`)
keeps calling `self._packed.assemble_host` and gets the cached one.

Policy: **16-way set-associative, exact LRU inside each set**. A global LRU list
needs a Python touch per row, the per-row loop the hot path cannot afford, and
CLOCK needs a hand sweep of data-dependent length. The set-associative probe is
four numpy ops on the whole batch (hash, one `(n, 16)` gather, compare, argmax)
and victim choice is one `argsort` over the few hundred sets a batch touches. On
the real id stream its hit rate equals a true global LRU to eight digits at
0.5 GB and above (0.39394299 both) and loses 0.2 points at 0.25 GB, where the
cache is actually full. Ways 4/8/16/32 at 0.25 GB give 0.3847/0.3897/0.3920/0.3930.

Two smaller wins come free: batches are deduplicated before any read (1.39x
fewer rows per chunk), and misses are read as 100-byte `pread`s rather than
16 KB pages, leaving the OS page cache for the model's own weights.

## Traffic

`ngram.py` reproduces the model's hashing in numpy; `test_ngram.py` checks it
against the vendored mlx code lifted out of `language.py:1877-1915, 2603-2656`
by AST. Row ids are identical on five token patterns including eos boundaries,
and the derived padded vocabulary, 320,001,536, matches the manifest. Corpus is
5 Gutenberg books (1.72M tokens) plus 434 local Python files (1.68M tokens),
tokenized with the model's own `tokenizer.json`; measurements use a 300k-token
interleaved prose/code stream, 147 chunks, 4.8M lookups.

Each token does 16 lookups, 8 bigram heads over one hash and 8 trigram heads
over another. Over 300k tokens they touch 2,096,133 distinct rows, 210 MB of
payload; bigram rows are requested 3.31x each, trigram rows 1.75x.

## Exactness

`test_exact.py`: 6 batch shapes x 3 passes (duplicates, unsorted ids, a
16-lookup decode step, first and last rows of the file) against the uncached
reader on the real `layer1.rows.bin`, with a 0.5 MB cache so eviction runs
(2873 evictions). All three packed planes byte identical, the dequantized bf16
bit identical, and bit identical again after `* weight_scale`. Max abs error
0.0, 0 ULP.

## Speed

Medians. The OS page cache was never dropped, so the regimes are separated by
bypassing it with F_NOCACHE (block A) and by pass ordering on the deployed mmap
reader (block B).

| A: OS cache bypassed, paired, 36 chunks each | ms/chunk | rows from SSD per chunk |
|---|---|---|
| deployed algorithm, no cache, no dedup | 218.2 | 32768 |
| dedup only | 179.3 | 25380 |
| 2 GB cache, cold ids (13.1% hit) | 151.9 | 22053 |
| 2 GB cache, LRU warm | **4.17** | 0 |

| B: deployed mmap reader, 12 chunks | ms/chunk | ms/decode step |
|---|---|---|
| cold | 138.7 (203,095 pages, 3.33 GB read) | 0.139 |
| page-cache warm | **3.08** | **0.0068** |
| cached, LRU cold, page cache warm | 125.0 | |
| cached, LRU warm | 3.47 | 0.0125 |

| C: decode, 512 steps, after an 82k-token prefill | ms/step | hit rate |
|---|---|---|
| no cache, cold | 0.146 | |
| cached | 0.124 | 38.8% |
| cached, LRU warm | 0.0125 | 100% |

Hit rate after a full 300k-token prefill is 64.0% on continuation decode. The
rest of the lookup call (upload, `mx.dequantize`, the bf16 scale) costs 0.479 ms
per chunk and 0.224 ms per step, and the cache does not touch it.

Memory: `OMLX_PLE_LRU_GB=2` reserves 1.812 GB (16.78M entries, 1,048,576 sets
x 16 ways, 108 B each). Process RSS 1.92 GB with the default `pread` reader, and
8.77 GB of clean file pages with `OMLX_PLE_LRU_READER=mmap`, which is why pread
is the default.

## Honest read of the result

Audit item 6 projected +5-9% on decode and measurement does not support it. The
addressable work is 0.139 ms per step cold and 0.0068 ms page-cache warm against
a ~24 ms forward. The derived 2.3 ms is mostly the `mx.eval` host sync at
`LANG:2267` plus upload and dequantize, and no row cache removes any of it.

The cache earns its place when the table's pages are not in the OS page cache,
the production state with 78 GB of model plus a growing KV cache: repeated ids
then cost 4.17 ms per chunk instead of 218, and 0.0125 ms per step instead of
0.146. It also cuts bytes crossing the storage stack by 156x (21 MB of rows
against 3.33 GB of pages over 12 chunks), so the PLE reader stops evicting the
model's own mapped weights. When the page cache is healthy the OS does this
better than we do (3.08 against 3.47 ms), so treat the cache as insurance with a
bounded price rather than a speedup. Benchmarks ran with production idle
(`/health` reported `loaded_count: 0`, ~85 GB free), so the page-cache-warm
column is the most favourable case it will ever show.

## Cheaper alternative: a static hot set

Preloading the most frequent rows of a generic corpus, no eviction logic:

| rows preloaded | GB | hit rate on held-out traffic (same domain mix) |
|---|---|---|
| 100,000 | 0.01 | 32.2% |
| 1,000,000 | 0.10 | 47.0% |
| 1,543,770 (all seen in training) | 0.154 | 49.9% |
| adaptive 2 GB cache, same slice | 1.81 | 48.1% |

100 MB of static rows matches a 2 GB adaptive cache when the traffic matches the
corpus, and does not transfer at all when it does not: a prose-trained hot set
covers 4.1% of code lookups and a code-trained one 6.5% of prose, against
43.7-44.0% within domain. Worth it only for a known workload;
`CachedPLETable.preload(ids)` is there for that.

## How to enable, and how to run the tests

`install()` in `round2/ple-lru/patch.py`, gated by `OMLX_PLE_LRU=1`, on top of
`OMLX_PLE_PACKED=1` and `OMLX_PLE_PACKED_MODE=rows`. `OMLX_PLE_LRU_GB` sets the
size (default 2, clamped to 4) and `OMLX_PLE_LRU_READER` the miss reader
(`pread` default, `nocache`, `mmap`). Idempotent, returns False with the toggle
off or the pack missing, and safe either at import time or after model load: it
finds the base module by file path, so bootstrap's own `spec_from_file_location`
copy is the one that gets rebound.

```
cd ~/inference-server/kernels/round2/ple-lru
P=~/inference-server/kdev/bin/python
$P test_ngram.py      # hashing identical to the vendored mlx code
$P test_exact.py      # bit exactness against the uncached reader
$P test_install.py    # install contract, gating, clamp, idempotency
$P corpus.py          # fetch and tokenize (33 MB, cached under ./corpus)
$P analyze.py         # hit rates, policy comparison, static hot set
$P bench.py           # timings against the real 32 GB rows.bin (~30 s)
```

Limitations: the tests stub `mlx_vlm...language._PLE_IO_POOL` because kdev has
no mlx_vlm, and production imports the real pool. No end-to-end run was done,
since loading the model is forbidden here, so the end-to-end claim above is a
component measurement plus arithmetic.
