# Weight-stationary bf16 expert gather (round 3, gather-ws)

2026-09-12. M5 Max 40-core, macOS 26.5, mlx 0.32.2, model Qwen3.8-Flash-Next-oQ4e-mtp.
No model loads, no safetensors reads, synthetic tensors under 1.2 GB, ports 8083/8084 untouched.

## 1. What is replaced

`mx.gather_qmm(..., sorted_indices=True)` on the routed-expert prefill GEMMs. On this GPU that
call lands in `affine_gather_qmm_rhs_nax` (dispatch read out at `libmlx.dylib` 0xe069c0, tile
choice at 0xe042b8), which stages dequantised bf16 weights into `threadgroup T Ws[]`
(`quantized_nax.h:638,646,673`) and feeds `matmul2d` bf16 x bf16 (`steel/gemm/nax.h:401-424`).
AUDIT section C names the defect: `rows_per_expert = 40 < 64` selects **BM=32**, so every expert
runs two row tiles and every expert weight is read and dequantised twice.

`kernel.py` builds a ragged tile table on device from the sorted indices (binary search for each
expert's first row, then one `(row_start, expert)` entry per 48-row block) and dispatches one
threadgroup per (tile, 256-column block). TM=48 covers a whole expert in one tile for 430 of 512
experts at T=2048, so those weights stream once. Each simdgroup dequantises only the 16 columns
it consumes, which makes the handoff a `simdgroup_barrier` rather than a threadgroup one, and the
group `g+1` weight loads are hoisted into registers above the group `g` matmul. The running sum
lives in the cooperative tensor (`mode::multiply_accumulate`, `relaxed_precision = false`), which
removed 24 fp32 adds per lane per group and was worth 5-8%. Skew is handled by the tile table
itself: an expert with 1120 rows gets 24 tiles, an expert with 6 gets one whose unused row slots
are computed and masked at the store. The last tile of the array slides its row window back so
the activation tensor never reads past row R, so no padded activation copy is needed.

Config `(TM, TN, SGN) = (48, 16, 16)`: 512 threads, 32 KB of staging, 24 fp32 accumulator
registers per lane. TN=32 spills (0.68x), TN=8 loses coalescing (0.58x), TM=64 wastes rows
(0.80x), a 128-column block doubles activation re-reads (0.96x). Double buffering the staging
slice was tried and lost (0.88-0.94x): it halves occupancy to one threadgroup per core.

## 2. Exactness

Bit-identical to stock, at every shape tested. `test_exact.py`, Zipf-like routing (popularity
`1/rank^0.7`, giving 6 to 1120 rows per expert at T=2048, 82 experts over 48 rows):

| shape | T | rows | identical bf16 words | max ULP | max abs diff | RMS vs fp32: ours / stock |
|---|---|---|---|---|---|---|
| gate_up [512,1280,2560] | 512 | 5120 | 100.0000% | 0 | 0 | 0.1662% / 0.1662% |
| gate_up | 2048 | 20480 | 100.0000% | 0 | 0 | 0.1662% / 0.1662% |
| down [512,2560,640] | 512 | 5120 | 100.0000% | 0 | 0 | 0.1662% / 0.1662% |
| down | 2048 | 20480 | 100.0000% | 0 | 0 | 0.1662% / 0.1662% |

The target was 1 ULP and the result is 0. That is not luck: both paths dequantise to bf16, feed
`matmul2d` with a fp32 destination, and sum the 40 affine groups in ascending order, so the
summation order is the same one stock uses. Tolerance achieved: exact equality, so nothing in
SwitchGLU, the router, or the deployed wsum10 weighted sum can see the difference.

## 3. Microbench

`bench.py`, 5 ops per `mx.eval`, 11 interleaved rounds, min over rounds, Zipf routing, flops
counted over the routed rows actually gathered. Ceiling 65.7 TFLOP/s bf16. The GPU is shared with
the live daemon; the T=2048 rows are from a quiet window (eight consecutive runs within 3%), the
T=512 and T=1024 rows are from a busier one, so read their **ratios**, not their absolute times.

| shape | T | rows | GFLOP | stock ms | TF/s | % peak | ws ms | TF/s | % peak | seg ms | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate_up | 2048 | 20480 | 134.2 | 5.96 | 22.5 | 34% | 5.18 | 25.9 | **39%** | 0.047 | **1.14x** |
| down | 2048 | 20480 | 67.1 | 2.68 | 25.1 | 38% | 2.34 | 28.7 | **44%** | 0.048 | **1.12x** |
| gate_up | 1024 | 10240 | 67.1 | 7.69 | 8.7 | 13% | 7.90 | 8.5 | 13% | 0.054 | 0.97x |
| down | 1024 | 10240 | 33.6 | 3.16 | 10.6 | 16% | 3.15 | 10.7 | 16% | 0.053 | 0.99x |
| gate_up | 512 | 5120 | 33.6 | 5.47 | 6.1 | 9% | 5.47 | 6.1 | 9% | 0.054 | 0.99x |
| down | 512 | 5120 | 16.8 | 2.33 | 7.2 | 11% | 2.82 | 5.9 | 9% | 0.048 | 0.87x |

Segmentation is 0.045-0.055 ms, measured as its own kernel and added to the totals above. It is
0.9% of the gate_up gather, and `kernel.py` memoises it on the indices array so a SwitchGLU pays
it once per layer instead of once per matmul, halving it to ~0.5%.

Against `../../moe-int8/` (its own report, same harness and machine): int8 x int4 reaches 1.48x on
gate_up and 1.30x on down at T=2048, taking a routed layer from 8.24 to 5.80 ms, for 11.3 GB of
transposed scale, folded-bias and nibble-sum tables. This kernel takes the same layer from 8.64 to
7.52 ms for 0 bytes, and is bit-exact where int8 costs 0.64% of output RMS.

Where the remaining time is: with the dequantise-and-store neutered, the same kernel runs
gate_up in 2.9-3.2 ms against 5.2 ms complete, so the unpack is roughly 60% of the kernel and the
matmul is already near the tensor ceiling. (The mirror-image ablation, matmul with staging
removed, did not isolate cleanly and its number is not quoted.) The lever is warp specialisation,
producer simdgroups unpacking into a ring while consumer simdgroups run `matmul2d`, so the ALU and
the tensor units overlap instead of alternating. That is the next thing to build here.

## 4. Deployment

`patch.py`, `install() -> bool`, idempotent, gated by **`OMLX_MOE_GATHER_WS=1`** (default off).
Install at **import time**: it monkeypatches `mx.gather_qmm` and `SwitchGLU.__call__`, not model
instances, so it does not need the model loaded, and there is nothing to warm up. Also
`OMLX_MOE_GATHER_WS_MIN_TOKENS` (default 2048, because 1024 and below is a wash or a loss) and
`OMLX_MOE_GATHER_WS_MIN_ROWS` (default 8192, used when the token count is unknown).

Routing conditions match the int8 patch: `sorted_indices=True`, `rhs_indices` present and
`lhs_indices` absent, `transpose=True`, affine 4-bit `group_size=64`, bf16 `[rows, 1, K]`
activations, `K % 64 == 0`, a column block dividing N, and the token floor. Everything else takes
the stock path, including every decode and MTP verify shape and the 5/6/8-bit tensors.

Composition. The `SwitchGLU` hook chains on top of whatever is installed, so oMLX's fused
`gate_up_proj` (`qwen35_moe_gate_up.py`) and the int8 patch's identical hook both keep working.
The `mx.gather_qmm` hook captures its predecessor as the fallback, so oMLX's M5 reroute survives
for everything not taken. **The int8 kernel wins** when both are enabled, at 1.42x against 1.14x
per layer: if int8 is installed below this patch, this patch calls the int8 module's own
`supported()` and declines anything it accepts; if int8 is installed above, it takes what it wants
and hands the rest down. Output layout and dtype are unchanged, so the deployed wsum10 fused
weighted sum sees exactly what it sees today. Neither of those two modules is touched.

## 5. Expected gain and test plan

Per 2048-token chunk: a routed layer goes 8.64 to 7.52 ms including segmentation, so 1.12 ms per
layer and **~54 ms across 48 layers**, about **5.4% of the ~1000 ms chunk body**. Prefill only;
decode is below every floor and unaffected. Nothing on top of a 70 GB model.

Workbench plan, in order:

1. `~/inference-server/kdev/bin/python test_exact.py` then `bench.py` then `test_patch.py`, in the
   workstream directory. Exactness must stay at 0 ULP; the patch test must show bit-identical
   output with the gate on and off, fall-through for 8-bit, and correct int8 precedence.
2. On the workbench box only, load the model with `OMLX_MOE_GATHER_WS=1` and confirm from
   `patch.stats()` that `routed` is 96 per 2048-token chunk (two gathers per layer) and 0 during
   decode and MTP verify.
3. Compare chunk-body timings over a 2048-token chunk from an empty cache, gate on against gate
   off, expecting ~54 ms. Because the kernel is bit-exact, logits must match the stock run word
   for word; anything else is a bug, not a tolerance.
4. Then a 65k prompt for the segmentation memo under real chunking, and a run with both this and
   `OMLX_MOE_INT8_PREFILL=1` to confirm `to_int8` accounts for every gather and `routed` is 0.

Commands: `cd ~/inference-server/kernels/round3/gather-ws && ~/inference-server/kdev/bin/python
test_exact.py && ~/inference-server/kdev/bin/python bench.py && ~/inference-server/kdev/bin/python
test_patch.py` (add `ablate.py` for the stage/matmul split).
