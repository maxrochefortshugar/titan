# Round-3 small items: fused top-K, reader workers, verify weighted sum, int8 down tables

`kernels/round3/small-items/`, 2026-09-12. No model loaded, no safetensors opened, no server
started, ports 8083 and 8084 untouched, peak GPU allocation under 150 MB. Every performance number
was taken after `staging/GPU_FREE` appeared at 13:53.

| item | env flag | install | timing | exactness | measured gain |
|---|---|---|---|---|---|
| 1 fused top-K, shortlist drafter | `OMLX_MTP_SHORTLIST_FASTTOPK=1` | `install_mtp_fasttopk()` | import time, after round2/mtp | exact top-K set | 0.04 ms per decode cycle, +0.1% |
| 2 n-gram reader workers | `OMLX_PLE_READ_WORKERS=<n>` | `install_ple_read_workers(model)` | after load | pure IO | none, 48 is already best |
| 3 MTP verify weighted sum | `OMLX_WSUM_TOPK10_VERIFY=1` | `install_wsum_verify()` | after load, after round2/wsum10 | bit-identical | 0.22 ms per decode cycle, +0.7% |
| 4 int8 down projections | `OMLX_MOE_INT8_DOWN=1` | `install_moe_int8_down(model)` | after load | accounting only | +2% prefill for 3.77 GB, declined |

## 1. Fused top-K for the shortlist drafter

Replaced: the selection line of `_refresh_shortlist`, `kernels/round2/mtp/patch.py:105`,
`ids = mx.argpartition(logits_1d, kth=v - k, axis=-1)[..., -k:]`. That file is not edited: the
function takes its `mx` module as its first argument, so the hook wraps it and passes a proxy whose
`argpartition` is the kernel. Install reads the function's source and returns False if that
expression ever changes.

`topk.py` is a three-launch radix select on the order-preserving uint32 key of each float: an
11-bit histogram, a pass that suffix-scans it and compacts the boundary bucket, then one threadgroup
resolving the candidates' remaining 21 bits. All 32 bits are resolved, so it is a true top-K.

**Exactness** (`test_topk.py`, 20 cases): the value multiset is bit-identical to `mx.topk`
everywhere, every index carries its returned value, all K distinct. The index set also matches
`mx.argpartition` whenever the K-th value is unique, which is every float32 case, and the live path
feeds float32 log-probs. In bf16 that value is shared by 34 to 115 tokens, so a few members differ
from argpartition's pick, all carrying the identical logit; the argmax never does.

**Speed**, V=248320, median of 41, CHAIN=10, quiet GPU, milliseconds:

| K | mx.topk | argpartition | fused f32 | speedup | fused bf16 | speedup |
|---|---|---|---|---|---|---|
| 512 | 0.104 | 0.104 | 0.064 | 1.62x | 0.075 | 1.38x |
| 1024 | 0.104 | 0.105 | 0.056 | 1.85x | 0.067 | 1.54x |
| 2048 | 0.104 | 0.104 | 0.074 | 1.40x | 0.076 | 1.36x |
| 4096 | 0.104 | 0.104 | 0.075 | 1.39x | 0.080 | 1.34x |

Per launch at K=2048: 0.025, 0.037, 0.040 ms, so it is dispatch bound.

**A correction to round 2.** On a quiet GPU `mx.topk` costs 0.104 ms here, not 0.267 to 0.383; that
figure was taken with the daemon holding the GPU. Under a synthetic load both paths queue to about
0.8 ms and the edge falls to 1.03 to 1.17x. The saving is 0.04 ms per decode cycle quiet, 0.07 to
0.13 contended, against a 29.8 ms cycle: **+0.1 to +0.4%**, not +0.8%. It does not earn a deployment
on its own.

## 2. Packed n-gram reader worker count

`_PLE_IO_POOL` is a hard-coded `ThreadPoolExecutor(max_workers=48)` at
`mlx_vlm/models/qwen4_exp/language.py:1921`, resolved by import on every call of
`PackedPLETable._prefetch_pages` (`kernels/ple-fix/patch.py:150`). `ple_workers.py` swaps that
global for a pool of `OMLX_PLE_READ_WORKERS` threads, default 48, so an unset flag changes nothing;
ple-fix is not edited.

`bench_ple_workers.py` replays the real chunk against the real 32 GB `layer1.rows.bin`: 32,768 row
ids, de-duplicated to about 32,690 pages and 535 MB, one 16 KB `pread` each, with fresh random rows
per repetition and `F_NOCACHE` on the descriptor, so the page cache is neither read nor filled.

| workers | 1 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| median ms/chunk | 2533 | 721 | 432 | 397 | 366 | 358 | 342 | 340 | 339 | 349 | 340 |
| GB/s | 0.21 | 0.74 | 1.24 | 1.35 | 1.46 | 1.50 | 1.57 | 1.58 | 1.58 | 1.54 | 1.57 |

Serial to 48 threads is 7.5x, the llama.cpp result reproduced. The curve is flat from 32 on and the
deployed 48 sits in the plateau; 64 is 0.3% better, inside the noise. **Recommendation: leave
`OMLX_PLE_READ_WORKERS` unset (48), never below 16**, and keep the knob for a future machine. Thread
count is not the ceiling: 16 KB reads at 10.4 us each is 1.58 GB/s against 13.6 sequential, and only
larger reads or a resident table take the rest.

## 3. MTP verify weighted sum

`round2/wsum10` refuses `target_verify=True`, so verify runs the stock three-pass tail at
`mlx_vlm/models/qwen3_5_moe/language.py:65` every decode cycle. That layout is the easy one:
`_target_verify_switch_glu` (`:15-32`) returns `[B, T, k, D]` in token order, so the kernel runs with
an identity permutation and no `inv_order` gather. `wsum_verify.py` wraps
`Qwen3_5MoeSparseMoeBlock.__call__` once more, claims only `target_verify=True` with T in 2..15, and
passes everything else down. round2/wsum10 is not edited: its `weighted_sum` is imported by path
under the module name its own patch uses, so either order works.

**Exactness** (`test_wsum_verify.py`): in the default `clone` mode, 0 differing elements out of
5,120 to 38,400 at every T from 2 to 15, plus B=2, k=8 and float16.

**Speed**, k=10, D=2560, bf16, median of 15, CHAIN=10:

| T | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 13 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| stock ms | 0.0307 | 0.0266 | 0.0262 | 0.0297 | 0.0292 | 0.0308 | 0.0350 | 0.0372 | 0.0340 |
| kernel ms | 0.0259 | 0.0224 | 0.0216 | 0.0219 | 0.0227 | 0.0214 | 0.0223 | 0.0225 | 0.0219 |
| speedup | 1.19x | 1.19x | 1.21x | 1.36x | 1.28x | 1.44x | 1.57x | 1.65x | 1.55x |

The kernel wins at every verify width. At depth 3 verify is T=4, so 4.6 us per layer over 48 layers
is **0.22 ms per decode cycle, +0.7%**, rising to 0.7 ms at T=13. One caveat: round 2 measured
wsum10 at decode widths as -7% in situ despite winning the same microbench, which is why
`OMLX_WSUM_TOPK10_MIN_TOKENS=64` is deployed. This gate is narrower, but it needs a paired workbench
A/B first.

## 4. int8 gather down projections

`memcalc.py` runs `moe-int8/kernel.py:prepare_weights` at a cut-down expert count and scales to the
real shapes. Each routed tensor gets three `[E, G, N]` tables: bf16 scales, bf16 folded bias and
uint16 nibble sums.

| tensor | N | K | G | per tensor | x48 layers |
|---|---|---|---|---|---|
| gate_up (fused, resident today) | 1280 | 2560 | 40 | 157.3 MB | 7.55 GB |
| down (skipped today) | 2560 | 640 | 10 | **78.6 MB** | **3.77 GB** (3.52 GiB) |

Budget against the 110 GB guard (soft 93.5, hard 104.5), from the 85.7 GB measured with the gate_up
tables resident and the 4 GB hot cache:

| line | GB | running |
|---|---|---|
| in use today | 85.70 | 85.70 |
| + down tables | 3.77 | 89.47 |
| + 65k of KV (2.5 GB per 100k) | 1.62 | 91.10 |
| soft limit | 93.50 | **+2.40 headroom** |
| hard limit | 104.50 | +13.40 headroom |

**It fits with 2.40 GB to spare, and I still recommend keeping `OMLX_MOE_INT8_SKIP_DOWN=1`.** A
second concurrent 65k stream leaves 0.78 GB and a fourth crosses the soft limit by 2.47 GB, which is
what serialised concurrent requests before the guard went to 110. The prize is small: down goes
2.485 to 1.916 ms per layer at T=2048, 27 ms per chunk, about +2% prefill.
`install_moe_int8_down(model)` refuses when `get_active_memory()` plus the tables would cross
`OMLX_MOE_INT8_DOWN_SOFT_GB` (default 93.5).

Defect found in passing: the skip test at `moe-int8/patch.py:191` computes
`weight.shape[-1] * 8 // bits`, which is K/4 for a uint32-packed tensor and so at most 1024 for
every Flash-Next tensor, meaning `warmup()` builds nothing when `SKIP_DOWN=1`. It should be
`* 32 // bits`. Production never calls it, so gate_up is built lazily instead.

## Commands and deployment

```
cd ~/inference-server/kernels/round3/small-items
~/inference-server/kdev/bin/python test_topk.py          # item 1 exactness
~/inference-server/kdev/bin/python test_wsum_verify.py   # item 3 exactness
~/inference-server/kdev/bin/python test_hooks.py         # all four installs
~/inference-server/kdev/bin/python memcalc.py            # item 4 accounting
sh run_benches.sh                                        # waits for GPU_FREE, runs all three benches
~/inference-server/kdev/bin/python bench_ple_workers.py --workers 8,16,32,48,64 --reps 7
```

Saved output: `test-20260912.txt`, `bench-20260912.txt`, `topk-*.txt`, `ple_workers*.json`. In
`prod/run-omlx.sh`:

```
export OMLX_MTP_SHORTLIST_FASTTOPK=1     # item 1, needs OMLX_MTP_SHORTLIST_DRAFT=1
export OMLX_WSUM_TOPK10_VERIFY=1         # item 3
OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft,$R3/small-items/patch.py:install_import_time"
OMLX_ROUND2_PATCHES="$R2/gdn-norm/patch.py,$R2/wsum10/patch.py,$R3/small-items/patch.py:install"
```

Item 1 must come after `round2/mtp/patch.py` in the import-time list, since it attaches to the
module object the bootstrap loaded. Item 3 goes after `round2/wsum10/patch.py`.

## Limitations

None of this has seen the real model. The verify gate was proved against the stock ops tail on
synthetic tensors, not a real `SwitchGLU`, since mlx_vlm is absent from the kdev venv. The contended
top-K figure used a synthetic GEMM loop, the reader sweep times the prefetch alone, and item 4 rests
on an 85.7 GB figure measured under the old 100 GB guard.
