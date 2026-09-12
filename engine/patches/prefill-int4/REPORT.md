# Workstream 3: prefill matmul on native low-precision tensor operands

M5 Max 40-core, macOS 26.5.2 (build 25F84), mlx 0.32.2, MSL 4.0 (`__METAL_VERSION__ 400`).
All numbers measured on this machine, median of 10 chained iterations per `mx.eval`.

## 1. Feasibility

A JIT `mx.fast.metal_kernel` compiles and runs `#include <metal_tensor>` plus
`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` with no MLX rebuild
and no Xcode installed (the Metal runtime compiler carries the headers; there is no
`xcrun metal` on this box). Feature macros read back from a probe kernel:

| macro | value |
|---|---|
| `__METAL_VERSION__` | 400 |
| `__HAVE_TENSOR__` | 1 |
| `__HAVE_BFLOAT__` | 1 |
| `__HAVE_INT4B_FORMAT_TYPE__` | **1** |

`mpp::tensor_ops::matmul2d` works both ways it is exposed:

* **register / cooperative-tensor operands** (what mlx's `steel/gemm/nax.h` uses):
  `get_left_input_cooperative_tensor` / `get_right_input_cooperative_tensor`, filled
  element-by-element. The destination index order is `ix[0] = n`, `ix[1] = m`
  (x is the inner/N dimension). Same for the A and B operands: `ix[0]` is the K index
  for A and the N index for B.
* **memory operands**: `metal::tensor<device T, dextents<int32_t,2>, tensor_inline>(ptr, extents, strides)`
  built inline from the raw device pointers MLX hands the kernel. Extents are
  `(inner, outer)`; for `transpose_right = true` the B tensor is `(K, N)` over a row-major
  `[N][K]` buffer. This turned out to be the faster path and is what the final kernel uses.

Operand-type matrix actually verified against a numpy reference (`feasibility.py`,
16x32x64 tile, NN and NT):

| left x right -> dest | compiles | correct |
|---|---|---|
| bfloat x bfloat -> float | yes | yes |
| int8 x int8 -> int32 | yes | yes |
| int8 x `metal::int4b_format` -> int32 | yes | yes |
| uint8 x `metal::uint4b_format` -> int32 | yes | yes |
| bfloat x `uint4b_format` -> float | yes | yes |
| bfloat x `int4b_format` -> float | yes | yes |
| int8 x `uint4b_format` -> int32 | **no** | mixed signedness is not in the supported table |

Two constraints that cost time to find, both `static_assert`s in
`__impl/MPPTensorOpsMatMul2dImpl.h`:

* `descriptor.k % 32 == 0` whenever either operand is sub-byte.
* `descriptor.k % 16 == 0` in general (this is why the rank-G bias matmul below pads
  G=40 to 48).

`int4b_format` / `uint4b_format` are empty tag structs in `<metal_packed_numeric>`, not
storage types. A cooperative tensor of that element type hands back a `uint4b_format&`
you cannot assign an integer to, so **4-bit operands can only be supplied from a
`metal::tensor` in device or threadgroup memory**, never built in registers. The tensor's
`data_handle_type` is `device uchar*` and extents/strides are counted in 4-bit elements.
This restriction is what macOS 27's cooperative-tensor-direct-matmul would lift.

## 2. Raw NAX peak per precision (`peak_table.py`)

Each simdgroup issues back-to-back `matmul2d` calls on threadgroup-resident tiles into
independent register accumulators; the accumulate dependency stops the compiler hoisting
the loop. 8 simdgroups/threadgroup, 640 threadgroups. Best tile per precision:

| precision | TFLOP/s (2*MACs) |
|---|---|
| float32 x float32 -> float32 | 15.7 |
| bfloat16 x bfloat16 -> float32 | **65.7** |
| float16 x float16 -> float32 | 65.7 |
| bfloat16 x int4/uint4 -> float32 | 65.5 |
| **int8 x int8 -> int32** | **129.8** |
| uint8 x uint8 -> int32 | 129.9 |
| int8 x int4 -> int32 | 120.2 |
| uint8 x uint4 -> int32 | 113.5 |

So the M5 Max NAX ceiling is **~65 TFLOP/s bf16 and ~130 TOP/s int8**, a clean 2x, with
fp32 at 1/4 bf16. Calibration against stock MLX on the same machine: a 4096^3 bf16 dense
matmul runs at 60.4 TFLOP/s and dense q_proj at M=2048 at 60.7 TFLOP/s, i.e. 92% of the
measured ceiling, so 65 is the real number rather than an artifact of the probe.

Two findings worth keeping:

* **A 4-bit right operand buys no tensor-core time.** `bfloat x int4` runs at exactly the
  bf16 rate (65.5 vs 65.7). The rate is set by the wider operand. 4-bit only saves
  bandwidth, registers and threadgroup traffic.
* The brief's "sorted MoE gather at 2048 tokens ~100 TFLOPS" figure cannot be a bf16
  tensor-path number: 100 > 65. Either that measurement counts padded/duplicated rows in
  its flop total, or it is counting the full `[512, 640, 2560]` expert tensor rather than
  the rows actually gathered. Worth rechecking before anyone plans against it.

## 3. The kernel (`kernel.py`)

The important consequence of section 2: **stock `mx.quantized_matmul` at M=2048 already
runs at 57.0 TFLOP/s, which is 87% of the 65 TFLOP/s bf16 ceiling.** The dequantize-to-bf16
threadgroup round trip in mlx's `quantized_nax.h` is therefore *not* costing meaningful
tensor-core time, and plan (b) from the brief (unpack 4-bit straight into the fragment
type, keeping bf16 operands) has at most 15% in it before it starts. Native `bfloat x int4`
would land in the same place. Only int8 activations reach a higher ceiling, so the kernel
takes plan (a).

### Math

With affine 4-bit gs64, `w[n,k] = q[n,k]*s[n,g] + b[n,g]`, `g = k/64`. Re-centre the
nibble to signed int4: `q ^ 8` is exactly `q - 8` in 4-bit two's complement, so

```
y[m,n] = sum_g  s[n,g] * xs[m,g] * (xi[m,:] . qs[n,:])_g          <- int8 x int4 -> int32
              + (8*s[n,g] + b[n,g]) * rowsum[m,g]                 <- rank-G bf16 matmul
```

* The weight repack is a single `Wq ^ 0x88888888` on the packed `uint32` tensor, done once
  at load and cached. mlx's nibble order (low nibble = even k, little-endian) is already
  exactly what a `uchar`-backed `int4b_format` tensor wants, so **the packed weights go to
  the tensor unit verbatim: no unpack, no dequant, no threadgroup staging, 0.5 bytes per
  weight in the load path instead of 2.**
* `xi`/`xs` are a per-row-group (64) symmetric int8 quantization of the activations,
  produced by a fused Metal kernel (one simdgroup per (row, group), two bf16 values per
  lane, `simd_max` / `simd_sum`).
* `rowsum` is the **exact** fp32 group sum of the *unquantized* activations, so the affine
  bias term carries no activation-quantization error at all. That term is itself a rank-G
  matmul and is folded into the same fp32 accumulator as one extra bf16 `matmul2d` before
  the main loop (G=40 padded to 48 for the `k % 16` rule) - about 2% overhead instead of
  the 33% it costs as scalar FMAs.

### Shape and the one real cost

32x32 tile per simdgroup, 8 simdgroups per threadgroup as 2(M) x 4(N), threadgroup tile
64 x 128. A and B tensors point straight at device memory; K is walked one 64-wide group
at a time so the group scale can be applied to the int32 accumulator.

The per-group rescale is the whole story of what is left on the table. Measured with the
identical tiling and memory pattern but with the rescale removed (a plain int8 GEMM,
`multiply_accumulate` straight through K):

| variant, q_proj M=2048 | ms | TOP/s-equiv |
|---|---|---|
| int8 x int4, no rescale (upper bound for this tiling) | 0.619 | 104.0 |
| int8 x int8, no rescale | 0.617 | 104.4 |
| bf16 x bf16, same tiling | 1.245 | 51.7 |
| **int8 x int4 with per-group rescale (shipped)** | **0.945** | **68.2** |

Getting from a naive rescale to the shipped one was most of the work. Each lane owns 32
destination elements that span only **4 rows and 8 columns** (probed from
`get_multidimensional_index`), so the group loop loads 12 scale values into registers, not
64 from threadgroup memory:

```
sv[j] = scales[(n0 + ncol[j]) * G + g];   // 8
xv[j] = xscale[(m0 + mrow[j]) * G + g];   // 4
acc[i] = fma(float(ct[i]) * xv[((i>>2)&1) + 2*((i>>4)&1)],
             sv[(i&3) + 4*((i>>3)&1)], acc[i]);
```

That took the kernel from 55 to 68 TOP/s-equiv. `mrow`/`ncol` are derived at kernel start
from `get_multidimensional_index` rather than hard-coded, so only the bit decomposition of
`i` is an assumption, and a wrong one fails the correctness test loudly.

### Things I tried that were worse, recorded so nobody repeats them

* **Staging A and B tiles in threadgroup memory** (what mlx's nax kernels do): 38 TOP/s vs
  55 for direct device tensors. `matmul2d` reading device memory directly is faster than
  anything I could hand-stage; the barriers and the copy cost more than the reuse saves.
* **Bigger threadgroup tiles** (128x128, 256x128, 128x256) to cut DRAM traffic: all slower.
  Register pressure from multiple 32-element fp32 accumulators dominates. 64x128 wins.
* **Unrolling the group loop 2x/4x/5x** to overlap dependent `matmul2d` calls: monotonically
  worse (49.8 -> 39.8 -> 27.5 TOP/s at unroll 1/2/4). Spills.
* **Per-row instead of per-group activation scales**, which would hoist `xv` out of the loop
  entirely: only 5% faster (74.0 vs 70.0 TOP/s in an early variant) and materially worse
  numerically. Not worth it.

## 4. Accuracy (`test_exact.py`)

**This kernel is not bit-exact and does not try to be.** The question is whether the int8
activation quantization matters next to the 4-bit weight quantization the model already
carries. References: `stock` = `mx.quantized_matmul`; `fp32` = fp32 activations x fp32
dequantized weights accumulated in float64; `unquantized` = the same matmul against the
original bf16 weights, i.e. what the 4-bit format itself costs.

q_proj, M=2048, N=6144, K=2560. RMS error as a fraction of the reference RMS:

| activation distribution | stock vs fp32 | ours vs fp32 | ours vs stock | **q4 vs unquantized** | **ours vs unquantized** |
|---|---|---|---|---|---|
| normal | 0.166% | 0.641% | 0.662% | **9.10%** | **9.12%** |
| outlier channels (1 in 128 at 10x) | 0.166% | 1.28% | 1.29% | **9.11%** | **9.20%** |
| heavy-tailed | 0.166% | 1.22% | 1.23% | **9.13%** | **9.21%** |

o_proj M=512 normal: ours vs stock 0.662% RMS, q4 vs unquantized 9.09%, ours vs
unquantized 9.11%. Max absolute error ours vs stock: 0.047 on q_proj (reference max 7.19),
0.063-0.073 on o_proj (reference max 10.9).

**Read the last two columns.** Going 4-bit on the weights already perturbs the layer output
by 9.1% of its RMS. Adding int8 activations moves that to 9.12% (normal), 9.20% (outliers),
9.21% (heavy-tailed) - the total error grows by **0.2% to 1.2% relative**. In absolute
terms the kernel is ~4x noisier than the stock bf16-activation path, but both are a rounding
error next to the quantization the model was trained/converted to tolerate. For prefill
feeding the KV cache, that is acceptable, and these are the numbers to argue with.

Greedy agreement, lm_head geometry (1024 rows, vocab 248320, K=2560, random weights):

```
ours  vs stock : 98.05%  (1004/1024)
stock vs fp32  : 96.68%  (990/1024)
ours  vs fp32  : 96.09%  (984/1024)
when ours differs from fp32, median fp32 logit gap to the true top-1: 0.0045 (logit std 0.506)
```

The 98% headline looks alarming until you notice **stock only agrees with fp32 96.7% on the
same sample**, and that the disagreements sit on logit gaps ~1% of a standard deviation -
random weights make near-degenerate logits, so this measures tie-breaking, not accuracy.
Our added disagreement over stock is 0.6 percentage points. That said, lm_head is a decode
op at M=1..8 and is excluded by the M >= 512 gate anyway; I would not route it here.

## 5. Before / after (`bench.py`)

Stock `mx.quantized_matmul` vs this kernel, 4-bit affine gs64, bf16 activations,
Qwen3.8-Flash-Next attention shapes. "int4 ms" is end to end and **includes** the fused
activation quantizer, whose cost is broken out separately.

| shape | M | N | K | stock ms | TF/s | int4 ms | TF/s | of which quant | speedup |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | 512 | 6144 | 2560 | 0.335 | 48.0 | 0.260 | 61.9 | 0.030 | **1.29x** |
| q_proj | 2048 | 6144 | 2560 | 1.131 | 57.0 | 0.945 | 68.2 | 0.050 | **1.20x** |
| o_proj | 512 | 2560 | 6144 | 0.305 | 52.8 | 0.266 | 60.5 | 0.038 | **1.15x** |
| o_proj | 2048 | 2560 | 6144 | 1.132 | 56.9 | 1.014 | 63.6 | 0.079 | **1.12x** |

The first version of the activation quantizer was written in MLX ops and cost 0.55-1.28 ms,
which turned the whole thing into a 0.46x regression. Fusing it into one Metal kernel took
it to 0.03-0.08 ms.

## 6. Enabling it in oMLX

```python
from omlx.patches import prefill_int4
prefill_int4.install()          # monkeypatches mx.quantized_matmul
```

`OMLX_PREFILL_INT4=0` disables routing at call time without uninstalling.
`kernel.uninstall()` restores the original. The patch routes only when all of:
`transpose=True`, `group_size == 64`, `bits == 4`, 2-D x and w_q, bf16 activations,
`M >= 512`, `M % 64 == 0`, `N % 128 == 0`, `K % 64 == 0`. Everything else falls through to
stock, including every decode and MTP-verify shape. Repacked weights are cached by
`id(w_q)`; the cache holds a second copy of the packed tensor (same size as the original -
7.9 MB for q_proj, 318 MB if it were ever applied to lm_head) plus the folded `8s+b` table.
Nothing in the running daemon was touched.

## 7. Limitations and honest failures

* **The MoE gather shape is not implemented.** This is the one that would actually matter,
  and it does not fit the current tiling: 2048 tokens top-10 over 512 experts is 40 rows per
  expert, and a 32-row M tile wastes 24 of every 64 rows (60% efficiency loss on the second
  tile of each expert). It needs either a 16-row M tile (3 tiles, 8 wasted rows) or a ragged
  row-offset scheme like mlx's `*_gather_qmm_rhs_nax`. I ran out of time-box before starting
  it. Everything else here - the signed-nibble repack, the rank-G bias matmul, the 12-load
  rescale - carries over unchanged.
* Only 4-bit gs64 affine. 5/6/8-bit and other group sizes fall back. 8-bit gs64 would
  actually be easier (int8 x int8 at 130 TOP/s, no nibble games) and is the obvious next
  target.
* 2-D only, no batching, no bias/output-transform fusion.
* Output is bf16. The fp32 accumulator is exact through the group loop but the store
  rounds; matching stock's behaviour, not an improvement on it.
* The `i -> (row, col)` bit decomposition inside the rescale is an assumption about the
  `matmul2d` destination layout for a 32x32x64 descriptor. It is validated numerically by
  `test_exact.py` on every shape, and `mrow`/`ncol` themselves are queried rather than
  hard-coded, but a future MPP release could reshuffle it. If the accuracy test ever shows
  errors at the 100% level rather than the 1% level, that is what broke.
* I did not chase the last 35%. The no-rescale ceiling for this tiling is 104 TOP/s and the
  kernel delivers 68. The gap is entirely the per-group int32 extract-convert-scale, and I
  do not see a way to close it on macOS 26 (see below).

## 8. What macOS 27 would add, given what was measured

**Cooperative tensors direct to matmul.** This is the one that matters. Today a 4-bit
operand *must* come from a `metal::tensor` in memory, because `int4b_format` is a tag with
no register representation. That forces the K loop to be a sequence of independent
`matmul2d(K=64)` calls with a full accumulator extraction between each, which is exactly the
104 -> 68 TOP/s loss. If a cooperative tensor can be produced and fed directly, the group
scale can be applied to the operand instead of to the result, and the K loop collapses back
into a single accumulating chain. That is worth roughly **1.5x on top of what is here**, i.e.
~100 TOP/s and ~1.8x over stock, and it is the single highest-value item in the macOS 27
list for this workload.

**E8M0 scale plane.** This is microscaling (MX) support: a power-of-two scale plane applied
by the tensor unit itself. It does not map onto mlx's affine gs64 (which has an fp scale
*and* a bias, and a bias term cannot be expressed as a scale plane), so it would not help
the oQ4e weights as they stand. For an MXFP4 or MXINT8 checkpoint it would remove the
per-group rescale entirely and land in the same place as the previous paragraph, without
needing the `8s+b` rank-G matmul at all. If there is any freedom in how future models are
quantized, MX-format weights are worth more on M5 than affine ones purely because of this.

**fp4 / int2 operands.** Measured here, a narrower *right* operand does not buy tensor-core
time: `int8 x int4` is 120 TOP/s against `int8 x int8` at 130, and `bf16 x int4` is
identical to `bf16 x bf16`. So fp4/int2 would be a bandwidth and register story, not a
throughput one. At M=2048 prefill is firmly compute-bound (q_proj is 64 GFLOP against 7.9 MB
of weights - 1.1 ms of compute against 0.014 ms of DRAM), so the gain there is close to
zero. Where they would pay is decode, which is memory-bound and belongs to a different
workstream.

## 9. End-to-end prefill estimate, 125B-A6B MoE at 2048-token chunks

6B active parameters is ~12 GFLOP per token of linear-layer work, so ~24.6 TFLOP per
2048-token chunk. At the stock 57 TFLOP/s that is ~0.43 s of GEMM; the linears are typically
75-85% of prefill wall time at this chunk size, with attention, norms, routing and KV writes
making up the rest.

**As shipped, the honest number is 2-3%.** Only the dense attention projections route
through `mx.quantized_matmul`; for a 125B-A6B the attention projections are maybe 10-15% of
the active-parameter flops, and 1.2x on 15% of 80% of the time is
`1/(0.88 + 0.12/1.2) = 1.02`. That is not worth a deployment on its own.

**With the MoE gather path converted, 12-15%.** The expert GEMMs are ~85% of the linear
flops. If they hit the same 1.2x (the ragged 40-rows/expert tiling will cost something, so
call it 1.15-1.2x), the linear layers as a whole go 1.18x and prefill goes
`1/(0.2 + 0.8/1.18) = 1.14`. This is the version worth building, and it is the obvious next
step.

**With macOS 27 cooperative tensors, 25-30%.** If the per-group rescale collapses and the
kernel reaches ~100 TOP/s (1.75x over stock) across both dense and MoE linears, prefill goes
`1/(0.2 + 0.8/1.75) = 1.52` on the GEMM-bound portion, which after Amdahl against the
non-linear 20% lands around 1.3x end to end. That assumes the rescale is genuinely the only
thing between 68 and 104 TOP/s, which the no-rescale measurement supports but does not prove.

For a 2048-token chunk at ~0.55 s of prefill today, those are roughly 0.54 s, 0.48 s and
0.42 s respectively.

## 10. Files

* `kernel.py` - patch module: fused activation quantizer, int8 x int4 matmul, weight
  repack, `install()` / `uninstall()`, fallback gate.
* `test_exact.py` - error tables vs stock / fp32 / unquantized, three activation
  distributions, lm_head argmax agreement.
* `bench.py` - the before/after table in section 5.
* `peak_table.py` + `peak_lib.py` - the per-precision NAX ceiling in section 2.
* `feasibility.py` - the operand-type matrix in section 1.
