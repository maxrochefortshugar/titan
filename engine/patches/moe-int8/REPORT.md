# Workstream 4: routed-expert prefill on native int8 x int4 tensor operands

M5 Max 40-core, macOS 26.5, mlx 0.32.2, oMLX 0.7.0.dev2, model
Qwen3.8-Flash-Next-oQ4e-mtp (`qwen4_exp`, 48 MoE layers, 512 experts, top-10,
hidden 2560, expert intermediate 640).  Everything below was measured on this
machine. The GPU is shared with the live daemon on 8083, so every benchmark
alternates stock and kernel round-by-round and reports the **min** over rounds
as well as the median.

## 0. First: the baseline in the brief was wrong by 10x

The brief carried forward "MoE gather_qmm up-proj, T=2048 sorted, 0.655 ms,
~100 TFLOPS".  Both halves are wrong, and in opposite directions.

A 2048-token chunk at top-10 gathers **20,480 routed rows**, not 2,048.  The
0.655 ms figure is what you get timing `mx.gather_qmm` with `R = T` rows; the
"100 TFLOPS" then came from counting the flops of the *full* 20,480-row problem
against that time, which is why it exceeded the measured 65.7 TFLOP/s bf16 NAX
ceiling (workstream 3 already flagged this as impossible).

Measured here, with R = T * top_k and flops counted over the rows actually
gathered:

| shape | T | rows | stock sorted gather | TFLOP/s |
|---|---|---|---|---|
| gate_up [512,1280,2560] | 2048 | 20480 | **5.76 ms** | 23.3 |
| down [512,2560,640] | 2048 | 20480 | **2.48 ms** | 27.1 |

So a routed MoE layer costs ~8.2 ms of GEMM per 2048-token chunk, 48 layers is
**~396 ms**, and the sorted gather runs at ~25 TFLOP/s - **38% of the bf16 NAX
ceiling**, not 87% like the dense `quantized_matmul` path.  That is good news:
there is real headroom here, which is not true of the dense projections.

For calibration, a dense bf16 matmul of the same total shape
(`[20480,2560] x [2560,1280]`) runs at 55.9 TFLOP/s and dense 4-bit
`quantized_matmul` at 46.9.  The gather costs 2.2x a dense matmul of identical
flop count, because it streams all 839 MB of expert weights instead of reusing
a cached 1.6 MB tile.

## 1. Bit mix of the checkpoint: the experts are 100% 4-bit

`config.json`'s `quantization_config` carries a global default (4-bit, group 64,
affine) plus 631 per-tensor overrides.  **Not one override names a
`switch_mlp` tensor.**  All 441 routed-expert tensors - `gate_proj`, `up_proj`,
`down_proj` for the 48 language layers and the MTP layer - take the global
default: **4-bit, group_size 64, affine**.  Verified against the shard headers:
`layers.3.mlp.switch_mlp.gate_proj.weight` is `U32 [512, 640, 320]` (= 2560
nibbles per row) with `BF16 [512, 640, 40]` scales and biases.

The 5/6/8-bit tensors are all elsewhere: `linear_attn.*` (5/6-bit, some at
group 128), `shared_expert.*` and `shared_expert_gate` (8-bit), the hyper
connections (5/6/8-bit), `lm_head`/`embed_tokens` (8-bit), and the PLE n-gram
shards (4-bit group 32).  So the kernel targets exactly the bits that are
actually there, and the 4-bit gs64 case covers **100% of routed-expert
compute**.  Nothing else needs to be supported for this workstream.

## 2. Why the signed-nibble trick from prefill-int4 is not affordable here

Workstream 3's dense kernel re-centred the stored nibble with `q ^ 8` so the
packed weights could be fed verbatim to a signed `metal::int4b_format` operand
alongside int8 activations.  That produces a **second copy of the weight
tensor**.  For q_proj that is 7.9 MB.  For the routed experts it is 1.26 GB per
layer and 60 GB across the model - the experts *are* the model.  Dead on
arrival.

So this kernel feeds the stored **unsigned** nibbles to `metal::uint4b_format`
and quantizes activations to **uint8 with a fixed zero point of 128**
(`uint8 x uint4 -> int32` is in the supported operand table, measured at
113.5 TOP/s; mixed signedness is not supported, so int8 activations would force
the XOR).  With `x[m,k] = sc[m,g] * (xu[m,k] - 128)` and
`w[n,k] = q[n,k]*s[n,g] + b[n,g]`:

```
y[m,n] = sum_g  sc[m,g] * s[n,g] * ( D[m,n,g] - 128*qsum[n,g] - 8*(xusum[m,g] - 8192) )
              + rowsum[m,g] * (8*s[n,g] + b[n,g])
```

* `D` is the `uint8 x uint4 -> int32` tensor-unit dot over the 64-wide group.
* `qsum[n,g] = sum_k q[n,k]` in `[0, 960]`, stored as uint16, exact.
* `xusum[m,g]` is the integer sum of the activation codes, from the quantizer.
* `rowsum` is the exact fp32 group sum of the *unquantized* activations, so the
  affine bias term carries no activation-quantization error.

Two details that each cost a measurable amount:

* **The `-128*qsum` correction must stay exact.**  `D` is ~61,000 and
  `128*qsum` is ~61,400; their difference is the signal.  Folding
  `128*qsum*s` into a bf16 table (which would have removed a load and an op)
  loses 0.2% of the larger term to rounding, i.e. ~4% of the signal.  It is
  kept as an exact integer and combined in fp32, where every operand is below
  2^24 and the arithmetic is exact.
* **The `-8*(xusum - 8192)` term re-centres the nibble arithmetically.**
  Without it the quantized product carries the weight mean `q̄ ≈ 7.5` and the
  activation-quantization error is amplified: measured **1.60% output-RMS error
  without it, 0.64% with it** - exactly matching workstream 3's dense number.
  It costs one extra fp32 add per output element per group, about 5% of kernel
  time.  Worth it.

## 3. Ragged tiling: the 40-rows-per-expert problem

A 2048-token top-10 chunk is ~40 rows per expert (binomial, observed 20-65).
Stock tiles the row axis at BM=32, so it runs two 32-row tiles per expert and
**streams every expert weight twice**.

This kernel builds a **ragged tile table on the device** - no host sync, which
matters because the routing is data dependent and a sync per MoE layer per
chunk would cost more than the kernel saves:

* one 1024-thread kernel binary-searches the sorted `rhs_indices` for each
  expert's first row (`offsets[0..E]`), then thread 0 walks the experts and
  emits `(row_start, expert)` for every `TMT`-row block;
* the dispatch grid is a static bound `ceil(R/TMT) + E` tiles (939 for
  R=20480, TMT=48, against ~512 actual); surplus threadgroups read one integer
  and exit;
* activations are quantized into a buffer padded by `TMT` rows so a tail tile
  never reads out of bounds, and stores are masked by `offsets[e+1]`.

**TMT=48 with a single M-simdgroup** is the shipped configuration: 48 rows
covers a whole expert in one tile for ~99% of experts, so each expert weight is
streamed **once**, and the row waste is 48/40 = 1.2x instead of stock's 1.6x.

## 4. Tile sweep, and the two things that mattered

All at gate_up, T=2048, interleaved with stock in the same process
(`ablate.py`).  `none` = matmul only, no rescale (the ceiling for that tiling);
`fold` = the shipped rescale.

| config (TM, TN, SGM, SGN) | threadgroup tile | mode | min ms | TOP/s | vs stock |
|---|---|---|---|---|---|
| 32, 32, 1, 8 | 32 x 256 | none | 3.08 | 43.7 | 1.87x |
| 32, 32, 1, 8 | 32 x 256 | full, `[E,N,G]` scales | 5.59 | 24.0 | 1.04x |
| 32, 32, 1, 8 | 32 x 256 | full, `[E,G,N]` scales | 4.20 | 32.0 | 1.38x |
| 48, 32, 1, 8 | 48 x 256 | none | 2.81 | 47.7 | 2.04x |
| 48, 32, 1, 8 | 48 x 256 | fold | 4.94 | 27.2 | 1.16x |
| **48, 16, 1, 16** | **48 x 256** | **none** | **2.64** | **50.9** | **2.18x** |
| **48, 16, 1, 16** | **48 x 256** | **fold (shipped)** | **3.42** | **39.2** | **1.67x** |
| 64, 16, 1, 16 | 64 x 256 | fold | 3.84 | 35.0 | 1.49x |
| 32, 16, 1, 16 | 32 x 256 | fold | 3.87 | 34.7 | 1.48x |
| 80, 16, 1, 16 | 80 x 256 | fold | 5.73 | 23.4 | 1.00x |
| 96, 16, 1, 16 | 96 x 256 | fold | 9.54 | 14.1 | 0.60x |

(The shipped kernel adds the re-centring term on top of `fold`, ~5%, and the
bench table in section 6 includes the activation quantizer.)

**(a) Transposing the scale tables to `[E, G, N]` was worth 1.35x -> 1.67x.**
With mlx's `[E, N, G]` layout, the columns a simdgroup needs for one affine
group are `G` elements apart, so the simdgroup's scale/bias/qsum loads touch
one cache line *per column*.  Transposed, they are contiguous: one cache line
per array per group.  This was the single largest win in the whole workstream
and it is pure data layout, not arithmetic.

**(b) TN=16 instead of TN=32.**  The destination cooperative tensor for a
48x16 descriptor is 24 elements per lane over 6 rows and 4 columns; 48x32 is 48
elements over 6 rows and 8 columns and spills.  Narrowing N halves both the
accumulator register count and the per-group column loads, which is what lets
TM go to 48 at all.  The `TM=80/96` rows above are what spilling looks like.

### Things that were worse, recorded so nobody repeats them

* **Threadgroup staging of the weight tile.**  The hypothesis was that with
  SGM>1 each M-simdgroup re-reads the same B tile, so staging it once per
  threadgroup would cut DRAM traffic.  Measured: staged 64x256 `none` 5.41 ms
  against 3.28 ms unstaged; the full kernel went from 0.99x to 0.69x.  Same
  conclusion as workstream 3's dense kernel, for a different reason - the
  per-group barriers and the copy loop cost more than the redundant reads,
  which the cache largely absorbs anyway.  The winning answer was to remove the
  redundancy structurally (SGM=1) rather than to stage.
* **SGM > 1 at all.**  Every SGM=2/3/4 variant lost to the SGM=1 equivalent.
* **TM=64/80/96** for wider row coverage: register spill, catastrophic at 96
  (0.60x).
* **Moving the affine bias term to a rank-G bf16 matmul** (what the dense
  kernel does): worth ~7% (`fold` 3.42 vs `foldnb` 3.18 ms), but it needs the
  bias table padded to a multiple of 16 groups in the *un*transposed layout,
  i.e. a fourth per-tensor table.  Not worth another 63 MB per layer for 7%.

## 5. Accuracy (`test_exact.py`)

Not bit-exact and does not try to be.  References: `stock` =
`mx.gather_qmm(..., sorted_indices=True)` with bf16 activations; `fp32` = fp32
activations x fp32-dequantized weights accumulated per expert; `bf16 weights` =
the same matmul against the original unquantized weights, i.e. what the 4-bit
format itself costs.  RMS error is reported as a percentage of the reference
output's RMS.

Random weights, gate_up geometry `[512, 1280, 2560]`:

| case | stock vs fp32 | **ours vs fp32** | ours vs stock | q4 vs bf16 w | **ours vs bf16 w** |
|---|---|---|---|---|---|
| normal, T=512 | 0.166% | **0.642%** | 0.663% | **9.096%** | **9.116%** |
| normal, T=2048 | 0.166% | **0.641%** | 0.662% | - | - |
| outlier channels (1/128 at 10x) | 0.166% | **1.280%** | 1.290% | - | - |
| heavy-tailed activations | 0.166% | **0.638%** | 0.658% | - | - |
| heavy-tailed weights, T=2048 | 0.167% | **0.642%** | 0.662% | 9.069% | 9.091% |

**Real checkpoint tensors**, read straight out of the safetensors shard with
`mx.load` (layer 3 of Qwen3.8-Flash-Next-oQ4e):

| tensor | acts | T | stock vs fp32 | **ours vs fp32** | ours vs stock | max abs (ref max) |
|---|---|---|---|---|---|---|
| `gate_proj [512,640,2560]` | normal | 512 | 0.166% | **0.678%** | 0.698% | 0.030 (1.95) |
| `gate_proj` | normal | 2048 | 0.166% | **0.678%** | 0.698% | 0.033 (2.01) |
| `gate_proj` | outlier ch. | 512 | 0.166% | **1.372%** | 1.382% | 0.084 (4.81) |
| `down_proj [512,2560,640]` | normal | 512 | 0.166% | **0.673%** | 0.693% | 0.015 (1.27) |
| `down_proj` | normal | 2048 | 0.166% | **0.672%** | 0.692% | 0.020 (1.39) |
| `down_proj` | outlier ch. | 512 | 0.166% | **1.350%** | 1.360% | 0.062 (2.09) |

End to end through a real `SwitchGLU` (all three routed matmuls plus the
SwiGLU), `test_patch.py`: 1.41% of output RMS at T=512 and T=2048, and
**bit-identical** below the `min_tokens` floor.

**Is this acceptable for prefill?**  Yes, and the last two columns of the first
table are the argument.  Going 4-bit on the expert weights already moves the
layer output by 9.10% of its RMS.  Routing through this kernel moves it to
9.12% - the total error grows by **0.2% relative**.  The kernel is ~4x noisier
than the bf16-activation path in isolation (0.64% vs 0.17%) but both are far
inside the quantization the checkpoint was converted to tolerate, and the real
checkpoint tensors behave exactly like the random ones (0.67-0.70%), which is
the result that matters.  The outlier-channel case doubles the error to 1.35%;
that is the worst distribution tested and still 1/7th of the 4-bit weight
error.  I would route prefill here and would not route decode (where a single
token's error is not averaged over a 2048-token chunk and the kernel is slower
anyway).

## 6. Before / after (`bench.py`)

Stock `mx.gather_qmm(..., sorted_indices=True)` against this kernel.  5 ops
chained per `mx.eval`, 7 interleaved rounds, min and median both shown.  "int8"
is **end to end and includes** the fused activation quantizer, broken out
separately.  Flops counted over the routed rows actually gathered.

| shape | T | rows | GFLOP | stock min | med | TF/s | int8 min | med | TOP/s | of which quant | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate_up [512,1280,2560] | 512 | 5120 | 33.6 | 3.530 | 3.600 | 9.5 | 3.076 | 3.080 | 10.9 | 0.117 | **1.15x** |
| gate_up | 1024 | 10240 | 67.1 | 4.266 | 4.286 | 15.7 | 3.295 | 3.317 | 20.4 | 0.176 | **1.29x** |
| gate_up | 2048 | 20480 | 134.2 | 5.763 | 5.787 | 23.3 | 3.890 | 3.905 | 34.5 | 0.320 | **1.48x** |
| gate_up | 4096 | 40960 | 268.4 | 8.359 | 8.409 | 32.1 | 6.458 | 6.517 | 41.6 | 0.598 | **1.29x** |
| down [512,2560,640] | 512 | 5120 | 16.8 | 1.472 | 1.481 | 11.4 | 1.507 | 1.514 | 11.1 | 0.056 | 0.98x |
| down | 1024 | 10240 | 33.6 | 1.783 | 1.789 | 18.8 | 1.608 | 1.617 | 20.9 | 0.069 | **1.11x** |
| down | 2048 | 20480 | 67.1 | 2.479 | 2.485 | 27.1 | 1.910 | 1.916 | 35.1 | 0.111 | **1.30x** |
| down | 4096 | 40960 | 134.2 | 3.842 | 3.898 | 34.9 | 3.110 | 3.158 | 43.2 | 0.181 | **1.24x** |

**At the production chunk size (2048 tokens) a routed MoE layer goes
8.24 ms -> 5.80 ms, 1.42x.**  T=512 is a wash on `down` (0.98x) - the
`min_tokens` floor keeps it out.  T=4096 falls back to 1.29x/1.24x because
80 rows per expert needs two 48-row tiles again; TM=64 and TM=80 were tried
there and were worse (6.84 and 8.11 ms against 5.53 for TM=48), so one
configuration is shipped for all T.

## 7. Whole-model prefill projection

Given the brief's 79% routed-expert share of prefill compute, and 1.42x on that
share:

```
1 / (0.21 + 0.79/1.42) = 1.30x   ->  23% less prefill wall time
```

In absolute terms, 48 MoE layers x 8.24 ms = **396 ms of routed GEMM per
2048-token chunk today, 278 ms patched, saving ~118 ms per chunk**.  If prefill
today is ~500 ms per chunk, that is ~380 ms.

This is the number workstream 3 projected at "12-15% once the MoE gather is
converted"; it came out better than that because the sorted gather's real
starting point (38% of the bf16 ceiling) was much worse than the dense path's
(87%), which nobody knew until the baseline was re-derived in section 0.

Caveats on the 1.30x: it assumes the 79% is a share of *time*, not just of
flops, and it takes no credit for anything outside the routed GEMMs.  If only
`gate_up` is routed (halving the extra memory, see below), it is 1.20x.

## 8. Memory cost - the honest downside

Each routed weight tensor needs three `[E, G, N]` tables: transposed scales
(bf16), transposed folded bias `8s+b` (bf16), and the nibble sums (uint16).
That is 3x the size of the existing `scales` array, or **18.8% of the weight
tensor**:

| tensor | weights | scales+biases | new tables | prep time |
|---|---|---|---|---|
| gate_up (fused) | 839 MB | 105 MB | **157 MB** | 9.7 ms |
| down | 419 MB | 52 MB | **79 MB** | 4.3 ms |

**236 MB per MoE layer, ~11.3 GB across the 48 layers** on top of a ~70 GB
model.  On a 128 GB machine that is affordable but not free, and it is the main
reason to think before deploying this.  Options: route `gate_up` only (7.5 GB,
1.20x instead of 1.30x), or build the tables lazily per layer and evict
(rebuilding costs a full weight pass, ~1.5 ms per tensor, so per-chunk eviction
is not viable).  The transposed layout is what buys 1.35x -> 1.67x, so it is
not optional; only the `qsum` table (52/26 MB) is specific to the unsigned
scheme, and it exists only because duplicating the weights for the signed
scheme would cost 60 GB.

## 9. Where the remaining headroom is

For gate_up at T=2048 the kernel (excluding the quantizer) is 3.57 ms:

* **Weight bandwidth floor**: 839 MB at the 549 GB/s copy ceiling = 1.53 ms.
  We are at 43% of that roofline.
* **Compute**: 134 GFLOP useful, 1.2x row waste = 161 GOP at 45 TOP/s against
  the 113 TOP/s `uint8 x uint4` ceiling, so 40% of peak.
* **The rescale is the gap**: matmul-only is 2.64 ms, the full kernel 3.57 ms.
  0.93 ms, 26%, is the per-group extract-convert-scale, the same structural
  cost workstream 3 hit (104 -> 68 TOP/s on the dense kernel) and for the same
  reason: a 4-bit operand must come from memory, so the K loop is a chain of
  independent `matmul2d(K=64)` calls with a full accumulator extraction between
  each.  macOS 27's cooperative-tensors-direct-to-matmul would collapse that
  chain and apply the group scale to the operand instead of the result; on
  these measurements that is worth roughly another 1.35x, i.e. ~2.0x over
  stock and ~1.45x whole-model prefill.
* The other 30% between matmul-only and the bandwidth roofline is the 1.2x row
  waste plus imperfect activation-tile reuse (each threadgroup re-reads its
  48x2560 activation slice once per 256-column tile; a 512-column tile would
  halve that but needs 1024 threads and lost in the sweep).

## 10. How to enable it, and the end-to-end A/B

The patch is standalone; **nothing under /Applications was modified or is
modified at runtime**.

```python
import sys; sys.path.insert(0, "~/inference-server/kernels/moe-int8")
import patch
patch.install()                # wraps mx.gather_qmm + SwitchGLU.__call__
patch.warmup(model)            # optional: build all tables up front
```

`install()` captures whatever `mx.gather_qmm` currently is as the fallback, so
install it **after** `omlx.patches.m5_gather_qmm` and oMLX's MoE gate/up fusion
and both keep working for everything this kernel does not take.  The
`SwitchGLU.__call__` hook only records the token count of the call in progress
(so `min_tokens` can be expressed in tokens rather than routed rows) and
delegates straight through, so it composes with oMLX's fused `gate_up_proj`
rather than replacing it.

| env var | default | meaning |
|---|---|---|
| `OMLX_MOE_INT8_PREFILL` | `0` | must be `1`; opt-in |
| `OMLX_MOE_INT8_MIN_TOKENS` | `512` | chunk token floor |
| `OMLX_MOE_INT8_MIN_ROWS` | `4096` | routed-row floor when the token count is unknown |

Routes only: `sorted_indices=True`, `rhs_indices` present and `lhs_indices`
absent, `transpose=True`, affine 4-bit gs64, bf16 activations `[rows, 1, K]`,
`K % 64 == 0`, and `N` divisible by one of 256/128/64/32/16.  Everything else -
every decode shape, every MTP verify shape, every 5/6/8-bit tensor - falls
through bit-identically.  Any exception inside the kernel logs a warning and
falls back.  `patch.stats()` reports routed/fallback call counts;
`patch.uninstall()` restores everything.

### A/B procedure for the production server (quiet window)

1. **Second instance, not the live one.**  Start an isolated
   `omlx serve` on port 8084 with the same model, exactly as the workstream 1
   staging trial did.  Note that oMLX ships its own `sitecustomize.py` which
   will shadow a loader placed on `PYTHONPATH` - apply the patch from an
   in-process bootstrap module instead, and confirm no stale server holds the
   port.  Both of those cost the workstream 1 trial a false start.
2. **Confirm routing before trusting any number.**  Set
   `OMLX_MOE_INT8_PREFILL=1`, send one 2048-token prompt, and check
   `patch.stats()["routed"]` is `96` per chunk (48 layers x gate_up + down with
   oMLX's fusion on, or 144 without it).  A zero there means the shape gate
   rejected everything - that is the failure mode that made the workstream 1
   trial measure the unpatched server twice.
3. **Watch RSS.**  Expect +11.3 GB after `warmup()` (or after the first
   prefill chunk, spread over 96 tensors at 4-10 ms each).  If that is too
   much, gate to `gate_up` only.
4. **Measure prefill, not total.**  Time to first token on a long prompt with
   generation capped at 1 token, 3 rounds each, plain vs patched, prompts of
   2048 / 8192 / 32768 tokens (the last exercises the multi-chunk path).
   Expect ~1.30x on the prefill-dominated portion at 2048-token chunks.
5. **Check the output.**  Greedy decode of the same prompt plain vs patched
   will *not* be identical - this kernel is not bit-exact.  The check that
   matters is that the continuations stay semantically equivalent and that a
   short perplexity or multiple-choice eval does not move; a 0.2% relative
   increase on top of the 4-bit weight error should be invisible.  If outputs
   diverge badly, that is a bug, not tolerance - see section 11.
6. Decode throughput must be **unchanged**: the gate rejects every decode
   shape.  If single-stream tok/s moves at all, the gate is leaking.

## 11. Limitations and honest failures

* **11.3 GB of extra tables.**  Section 8.  This is the real cost and the main
  reason to deploy deliberately rather than by default.
* **Not bit-exact**, 0.64-1.37% of output RMS depending on activation
  distribution.  Acceptable for prefill (section 5), not proposed for decode.
* **1.48x, against a 2.18x matmul-only ceiling** for the same tiling.  The
  per-group rescale is 26% of kernel time and I did not find a way to remove it
  on macOS 26 - the same wall workstream 3 hit.
* **T=4096 regresses to 1.29x** because 80 rows per expert needs two tiles.  A
  second kernel binary with TM=80 would fix it if it did not spill; it does.
* Only 4-bit affine gs64 and only the `rhs_indices`-only sorted path.  No
  `lhs_indices`, no fp8/mxfp4, no unsorted path.
* Output is bf16; the fp32 accumulator is exact through the group loop and the
  store rounds, matching stock rather than improving on it.
* The element -> (row slot, col slot) map of the `matmul2d` destination tile is
  probed at build time and verified simd-uniform across all 32 lanes before
  being baked in as compile-time constants, rather than hard-coded as in
  workstream 3.  A future MPP release that changes it would be caught by the
  probe's `RuntimeError`, not by silent wrong answers.  The absolute rows and
  columns are still queried per lane at kernel start.
* **Not tried, and probably the next thing to try**: splitting the K loop so
  two affine groups are consumed per `matmul2d` call with a shared rescale
  would halve the rescale cost, but it needs the two groups' scales folded,
  which affine quantization does not allow.  A checkpoint quantized with
  group_size 128 would get it for free.

## 12. Files

* `kernel.py` - the kernel: uint8 activation quantizer, device-side ragged tile
  table, `uint8 x uint4 -> int32` matmul with the exact `qsum` correction,
  `prepare_weights`, `gather_qmm_int8`, drop-in `gather_qmm_sorted`.
* `patch.py` - the oMLX-style runtime patch (`install`, `uninstall`, `warmup`,
  `stats`).
* `test_exact.py` - section 5, including the real-checkpoint cases.
* `test_patch.py` - end-to-end through a real `SwitchGLU`, including the
  bit-identical below-floor case.
* `bench.py` - section 6.
* `ablate.py` - the section 4 sweep harness (rescale ablation, tile configs,
  scale layouts, threadgroup staging).
* `probe.py` - `matmul2d` destination-layout probe for a range of descriptors.
* `smoke.py` - 30-second sanity check.
