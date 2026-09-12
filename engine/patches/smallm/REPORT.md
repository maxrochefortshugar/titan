# Workstream 1: small-M affine quantized matmul

M5 Max 40-core, macOS 26.5, mlx 0.32.2, python `~/inference-server/kdev/bin/python`.
Files: `kernel.py` (kernel + patch), `test_exact.py`, `bench.py`.

## What I built

A `mx.fast.metal_kernel` for `x[M,K] @ dequant(Wq)[N,K]^T` that reads each
packed weight word once, unpacks it once, and multiplies it against all M
activation rows held in registers. fp32 accumulation throughout, bf16/fp16/fp32
activations, affine 4-bit and 8-bit, group_size 32/64/128.

One kernel, three tile parameters:

- **BN** output columns per simdgroup, full unroll over `BN*M` fp32
  accumulators that never leave registers.
- **K_PARTS** simdgroups splitting the K reduction, combined through
  threadgroup memory. This exists only to manufacture occupancy on small N.
- **NSG_N** column tiles per threadgroup.

Two details did most of the work and neither was obvious up front:

1. **Register pressure sets BN.** The accumulator array must never be indexed
   dynamically or the compiler spills it to thread-local memory and the inner
   loop collapses. Beyond that, `BN*M` accumulators plus the live activation
   values hit a cliff: BN=8 at M=8 measured 3.5 ms on lm_head against 1.15 ms
   for BN=4. So BN is chosen from M (8 at M<=2, 4 at M<=8, 2 above), and
   activations are loaded as `vec<T,4>` inside a half-step rather than
   `vec<T,8>` across the whole step.

2. **Unpacking is byte-parallel.** The obvious form costs 5 scalar ops per
   weight value (shift, mask, convert, mul, add) against only M useful FMAs.
   `p & 0x0F0F0F0F` reinterpreted as `uchar4` yields the even k values and
   `(p >> 4) & 0x0F0F0F0F` the odd ones, so 8 values cost ~3 integer ops plus
   two 4-wide converts and two 4-wide FMAs. For 8-bit the packed words already
   are `uchar4`.

`qmm_smallm(x, wq, scales, biases, group_size, bits)` falls back to
`mx.quantized_matmul` for M>16, K not a multiple of the group size, N%4, other
bit widths, non-affine modes, and batch>1 rank-3 inputs. Fallback output is
bit-identical to stock (asserted in the test).

## Correctness

`~/inference-server/kdev/bin/python test_exact.py [--bits 4|8]`

N,K in {(6144,2560),(2560,6144),(512,2560),(640,2560),(248320,2560)},
M in {1,2,3,4,6,8,16}, two weight distributions: N(0,1), and a heavy-tailed
lognormal scale mixture where a few percent of channels carry 20-50x outliers
and dominate their quantization group. Each M is also compared against an exact
fp32 dequantize-and-matmul, because the two kernels reduce K in different
orders and cannot agree bit for bit; the question worth asking is which one is
closer to the truth.

Relative error is measured against the largest magnitude in the same output
row. Element-wise relative error is not a usable metric here: a logit that
lands near zero by cancellation makes any two correct kernels look 1e6 apart.
Row scale is what decides softmax and argmax.

Inside the routed band (M=2..8), 4-bit:

| shape | dist | M | max abs | mean abs | ours vs fp32 | stock vs fp32 |
|---|---|---|---|---|---|---|
| q_proj | heavy-tail | 2 | 1.2e-4 | 9.9e-9 | 14.56 | 14.56 |
| q_proj | heavy-tail | 8 | 2.5e-1 | 1.5e-5 | 43.91 | 43.91 |
| o_proj | heavy-tail | 8 | 5.0e-1 | 5.2e-5 | 27.48 | 27.48 |
| kv_proj | heavy-tail | 2..8 | 0.0 | 0.0 | identical | identical |
| shared_mlp | heavy-tail | 2..8 | 0.0 | 0.0 | identical | identical |
| lm_head | normal | 8 | 5.0e-1 | 4.1e-6 | 0.738 | 0.738 |
| lm_head | heavy-tail | 8 | 8.0e0 | 4.2e-5 | 125.4 | 125.4 |

Worst row-relative error anywhere in the routed band is 4.5e-3, against a bf16
relative ULP of 2^-8 = 3.9e-3. So: at or just inside one output ULP. And the
`ours vs fp32` and `stock vs fp32` columns are identical to 4 significant
figures at every routed M, which is the stronger statement — the two kernels
are equally close to the truth, they just round differently.

**Greedy argmax on lm_head, 1000 rows:** 100% agreement with stock at M=2, 4,
8, 16, for both 4-bit and 8-bit.

Two rows in the full output look alarming and are not:

- **M=1** disagrees substantially (q_proj heavy-tail max abs 256, argmax 97.3%
  at 4-bit). Stock is the inaccurate one: `ours vs fp32` is 15.9 where `stock
  vs fp32` is 256. MLX's M=1 matvec accumulates in a lower-precision order.
  M=1 is not routed by the patch, so this never reaches production.
- **M=16** disagrees more than M=8 because *stock* switches to its GEMM path
  there. Mean abs error is 0.089 against outputs of magnitude ~50, i.e. below
  one bf16 ULP, and again ours is the closer of the two to fp32. Also not
  routed.

8-bit: worst row-relative error 1.0e-2 (that maximum is again at M=1/M=16),
argmax 100% at every M including M=1.

## Speed

`~/inference-server/kdev/bin/python bench.py [--bits 4|8]` — 10 ops chained per
`mx.eval`, 20 samples.

**A note on the numbers.** The GPU is shared with the live oMLX daemon on port
8083, and it was serving traffic throughout. Contention can only add time, so
`bench.py` reports the min alongside the median. The min reproduces and matches
the stock baselines in COMMON.md; the median on this box drifted 2-5x between
runs (I watched stock lm_head M=4 read 0.99 ms and 6.0 ms twenty minutes
apart). Everything below is min-of-20, and stock and ours were measured
back-to-back inside the same run so the comparison is fair even where the
absolute level is inflated.

### 4-bit, group 64

| shape | M | stock ms | ours ms | speedup | ours GB/s | ours vs own M=1 |
|---|---|---|---|---|---|---|
| q_proj 2560->6144 | 1 | 0.054 | 0.065 | 0.84x | 136 | 1.00x |
| | 2 | 0.072 | 0.079 | 0.91x | 112 | 1.22x |
| | 4 | 0.104 | 0.100 | 1.04x | 89 | 1.54x |
| | 8 | 0.086 | 0.080 | 1.07x | 112 | **1.24x** |
| | 16 | 0.104 | 0.077 | 1.35x | 119 | 1.18x |
| o_proj 6144->2560 | 1 | 0.024 | 0.029 | 0.85x | 308 | 1.00x |
| | 2 | 0.028 | 0.033 | 0.85x | 267 | 1.16x |
| | 8 | 0.052 | 0.052 | 1.00x | 173 | 1.81x |
| | 16 | 0.102 | 0.080 | 1.27x | 114 | 2.79x |
| kv_proj 2560->512 | 2 | 0.019 | 0.024 | 0.78x | 32 | 0.98x |
| | 8 | 0.019 | 0.026 | 0.74x | 30 | 1.07x |
| shared mlp 2560->640 | 2 | 0.019 | 0.024 | 0.78x | 38 | 1.13x |
| | 8 | 0.020 | 0.025 | 0.81x | 39 | 1.16x |
| lm_head 2560->248320 | 1 | 0.445 | 0.598 | 0.74x | 598 | 1.00x |
| | 2 | 0.602 | 0.645 | 0.93x | 556 | 1.08x |
| | 3 | 0.889 | 0.796 | 1.12x | 451 | 1.33x |
| | 4 | 1.472 | 1.100 | 1.34x | 327 | 1.84x |
| | 6 | 2.429 | 1.265 | 1.92x | 285 | 2.11x |
| | 8 | 3.161 | **1.307** | **2.42x** | 277 | 2.18x |
| | 16 | 1.389 | 6.474 | 0.21x | 56 | 10.8x |

### 8-bit, group 64

| shape | M | stock ms | ours ms | speedup | ours GB/s | ours vs own M=1 |
|---|---|---|---|---|---|---|
| q_proj 2560->6144 | 1 | 0.062 | 0.072 | 0.86x | 233 | 1.00x |
| | 8 | 0.064 | 0.052 | 1.24x | 326 | 0.72x |
| o_proj 6144->2560 | 8 | 0.056 | 0.052 | 1.08x | 326 | 1.40x |
| lm_head 2560->248320 | 1 | 0.814 | 1.108 | 0.74x | 610 | 1.00x |
| | 2 | 1.173 | 1.182 | 0.99x | 573 | 1.07x |
| | 4 | 1.477 | 1.155 | 1.28x | 587 | 1.04x |
| | 6 | 2.543 | 1.166 | 2.18x | 582 | 1.05x |
| | 8 | 3.936 | **1.245** | **3.16x** | 546 | **1.12x** |

## Did it hit the flatness goal?

Partly, and the shape of the miss is the interesting part.

**8-bit: yes, decisively.** lm_head costs 1.12x at M=8 what it costs at M=1,
against 4.83x for stock. It holds 546-610 GB/s across the whole range, which is
the copy ceiling. The design premise is fully realised: weight traffic is
independent of M.

**4-bit q_proj: yes on the stated metric, 1.24x** (0.080 ms at M=8 against
0.065 at M=1), inside the 1.3x target. Stock is 1.59x over the same range.

**4-bit lm_head: no, 2.18x.** This is the honest failure and it is not a bug in
the kernel. The weight traffic genuinely is flat. It is an arithmetic wall:

- lm_head at M=1 runs at 598 GB/s, i.e. at the copy ceiling. Memory bound.
- lm_head at M=8 runs at 277 GB/s but 7.8 TFLOPS of scalar fp32 FMA, which is
  50-60% of the ~14 TFLOPS scalar FMA peak of 40 cores. Compute bound.

The kernel does `BN*M` scalar fp32 FMAs per weight value. Once the weight bytes
are free, which is the whole point of the design, the remaining cost is that
FMA count, and it is linear in M by construction. The 8-bit case stays flat
precisely because doubling the weight bytes keeps it on the memory side of the
crossover. Same kernel, same tile, same M: 546 GB/s at 8-bit versus 277 GB/s at
4-bit is the cleanest evidence that what binds at 4-bit is arithmetic.

I confirmed this is arithmetic and not registers or unpacking by measuring
three separate fixes and getting nothing: chunking activation rows to cut live
registers (no change), moving the unpack inside the phase loop (no change at
M=8, though it did rescue BN=8 at M=4), and the byte-parallel unpack (no change
on q_proj). The only lever left is `simdgroup_matrix`, which is what MLX's own
large-M path uses to reach 55 TFLOPS at M=2048 — a different kernel, not a tile
tweak.

## How to enable in oMLX

```
OMLX_SMALLM_QMM=1
```

then `from kernel import apply; apply()`, which monkeypatches
`nn.QuantizedLinear.__call__`. Without the env var `apply()` returns False and
changes nothing. Any kernel fault inside the patched call falls back to stock
rather than failing the request. Optional `OMLX_SMALLM_MIN_N` (default 16384)
and `OMLX_SMALLM_TILE="bn,kp,ng"` to force a tile.

Verified end to end against a real `nn.QuantizedLinear` built with
`from_linear`: M=2 and M=8 route and produce identical argmax, M=1 and M=16 and
N=512 fall through bit-identically, and the `bias` path is applied after the
kernel.

## Routing band, and why it is narrower than what the kernel supports

The kernel is correct for 1 <= M <= 16 but is only *faster* in part of that
range, so `apply()` routes 2 <= M <= 8 with N >= 16384.

- **M=1 excluded.** Stock's matvec is already at the ceiling and this kernel
  carries more Python dispatch overhead. It also keeps single-stream decode
  bit-identical to the unpatched engine, which is worth something on its own.
- **M>8 excluded.** Stock's GEMM path takes over and wins: lm_head M=16 is
  1.39 ms stock against 6.47 ms here. The narrow BN=2 tile needed above M=8
  re-reads the activations once per column tile, and that amplification
  (4*M/BN = 32x at M=16, BN=2) swamps everything.
- **N < 16384 excluded.** kv_proj (N=512) and the shared MLP (N=640) run in
  18-25 us total, dominated by launch and by the Python-side dispatch of
  `mx.fast.metal_kernel`. Routed, they are 0.74-0.9x. q_proj and o_proj sit
  just below the floor too: they are 1.0-1.1x at M=4..8, a real but thin GPU
  win that the per-call Python overhead eats. Set `OMLX_SMALLM_MIN_N=4096` to
  route them and measure in situ — with ~330 projections per forward the CPU
  cost may or may not pay, which is the same trade the existing
  `qwen35_verify_qmm` patch resolved with its own N floor of 16384.

So in practice this patch accelerates **lm_head at batched decode and MTP
verify**: 2.4x at M=8 4-bit, 3.2x at M=8 8-bit, 1.9x at M=6, 1.3x at M=4. That
is the shape where stock was worst (7.1x from M=1 to M=8) and it is the single
largest dense matmul in the model.

## Limitations

- 4-bit above M=4 is scalar-ALU bound, as above.
- No `mx.gather_qmm` equivalent, so the MoE experts are untouched.
- Only affine mode. No MXFP4/NF4, no 2/3/5/6-bit.
- Requires K % group_size == 0 and N % 4 == 0. All Qwen3.8-Flash-Next shapes
  qualify; a tail-N guard exists in the kernel but non-multiple-of-4 N falls
  back.
- Rank-3 input must be batch-1. Prefill (rank 3, T=2048) is out of band anyway.
- The tile heuristic is fitted to K in {2560, 6144} on this machine. It is a
  heuristic over a fairly flat response surface (most tiles within 10% in the
  sweep), not a tuned autotuner.

## Next steps, in the order I would do them

1. **`simdgroup_matrix` for 4-bit M>=4.** Dequantize a `BN x 8` weight tile
   into a `simdgroup_matrix<float,8,8>` and accumulate against an `M x 8`
   activation fragment. This is the only thing that moves the 4-bit M=8 number,
   and the ceiling it is reaching for is ~4x away.
2. **Fold scale and bias out of the inner loop.** `sum_k x*(q*s+b)` becomes
   `s*sum_k(x*q) + b*sum_k(x)` with the row sums precomputed per group. Saves 2
   of the ~5 ops per weight value. Needs group-aligned lane assignment, which
   is awkward at K/GS = 40 groups against 32 lanes; a quad-per-group mapping
   works. Worth maybe 10-15%, so do it after (1) or not at all.
3. **Drop the Python dispatch cost** so the N floor can come down and q_proj /
   o_proj / kv can be routed. Cache the tile decision per (weight id, M) and
   skip the `mx.contiguous` when the input already is contiguous.
4. **Re-benchmark on a quiet GPU.** Every number here was taken against a live
   daemon and is a min-of-20 for that reason.
5. Extend to `gather_qmm` for the 512-expert MoE, where T=2..8 has the same
   per-row re-read problem (T=1 0.035 ms, T=8 0.128 ms per COMMON.md).
