# Workstream 2: single-token MoE decode fusion (Qwen3.8-Flash-Next, M5 Max)

Shapes throughout: hidden 2560, 512 routed experts, top-10, fused gate_up
`[512, 1280, 2560]`, down `[512, 2560, 640]`, affine 4-bit group 64. mlx 0.32.2,
mlx-lm 0.31.3, macOS 26.5, 40-core M5 Max. Copy ceiling measured at 549-561 GB/s
during these runs.

## What I built

Three JIT Metal kernels (`fused_kernels.py`, via `mx.fast.metal_kernel`, no MLX
rebuild) and a patch module (`kernel.py`).

| kernel | replaces | math |
|---|---|---|
| `omlx_moe_gate_up_silu` | `gather_qmm(gate_up)` + `split` + `SiLU*mul` | `h[t,j,:] = silu(Wg[e]x) * (Wu[e]x)` |
| `omlx_moe_down_wsum` | `gather_qmm(down)` + `mul` + `sum` | `y[t,:] = sum_j s_j (Wd[e_j] h[t,j,:])` |
| `omlx_moe_router_logits` + `omlx_moe_router_topk` (opt-in) | router `qmm` + `softmax` + `argpartition` + `take_along_axis` + `sum` + `divide` | fp32 logits, top-k by logit, scores renormalised over the top-k |

All three read the MLX affine layout directly (uint32 words of 8 nibbles,
`w = scale*q + bias`), dequantize on the fly, and accumulate in fp32. One SIMD
group produces one output element; the K reduction is a `simd_sum`. The gate and
up rows of a given output channel are reduced in the same loop so the activation
loads and the per-word `sum(x)` term (needed for the affine bias) are shared.
`down_wsum` folds the router score into the lane-local partial, so the ten
experts cost one `simd_sum`, not ten, and it computes `ROWS=2` output channels
per SIMD group so the `h` slice each lane holds is loaded once and reused.

Deliberately *not* done: folding RMSNorm into the gate/up kernel. In mlx-lm the
norm lives in the decoder layer, not the MoE block, and its output feeds the
router as well as the experts, so folding it into the expert kernel would mean
recomputing it in all 800 threadgroups while the router still needs its own copy.
There is nothing to save. The launches that were actually worth removing were the
router's six, which is what `router_logits`/`router_topk` do.

## Correctness

`test_exact.py`. Reference is an fp32 golden computed in numpy from the same
quantized weights (exact affine dequantisation, fp32 matmul), so the stock path
and the fused path are scored on the same ruler. One bf16 relative ULP is
2^-8 = 3.906e-3.

| case | stock abs | stock rel | fused abs | fused rel |
|---|---|---|---|---|
| E=64 T=1 normal | 1.20e-03 | 9.91e-03 | 2.44e-04 | 2.02e-03 |
| E=64 T=1 heavy-tailed | 3.10e+00 | 1.10e-02 | 9.03e-01 | 3.19e-03 |
| E=64 T=2 normal | 1.56e-03 | 1.08e-02 | 4.75e-04 | 3.29e-03 |
| E=64 T=2 heavy-tailed | 5.78e+00 | 4.61e-03 | 3.66e+00 | 2.92e-03 |
| E=64 T=8 normal | 1.78e-03 | 1.24e-02 | 4.55e-04 | 3.17e-03 |
| E=64 T=8 heavy-tailed | 1.24e+01 | 7.20e-03 | 3.94e+00 | 2.28e-03 |
| **E=512 T=1 normal** | 1.18e-03 | 1.05e-02 | 2.41e-04 | 2.14e-03 |
| **E=512 T=2 normal** | 1.44e-03 | 1.01e-02 | 3.40e-04 | 2.39e-03 |

The fused path is inside one bf16 ULP of the fp32 golden in every case, and is
2 to 5x closer to it than the stock path, which rounds the gate/up output, the
SiLU product, the down output and the weighted sum all to bf16. Comparing the
patched block against the stock block directly gives rel 9.1e-3 (T=1) and
1.0e-2 (T=2); that gap is stock's own bf16 intermediate error, not the kernels'.
At T=8 the patched block falls back and matches bit for bit (diff exactly 0).

"heavy-tailed" activations are `normal * exp(2*normal)`, which puts several
orders of magnitude between channels and is the case where fp32 accumulation
matters most. That is where the stock absolute error reaches 12.4 and the fused
error 3.9.

### The router fusion is opt-in, and here is why

`router_logits` reproduces the stock router matvec to within half a bf16 ULP
(max |logit diff| 7.8e-3 against a bf16 ULP of 1.95e-2 at |logit| ~ 5), and the
scores of commonly-selected experts agree to 2.4e-3. But the *set* of selected
experts disagrees on roughly 1 in 40 synthetic tokens. The cause is not an error
in either path: stock rounds its logits to bf16, which makes the 10th and 11th
experts exactly equal often enough to matter, and `mx.argpartition` then breaks
the tie arbitrarily while the kernel takes the lowest index. A swap there moves
the output by score_10 (5-10%), far outside any ULP bar, so the router fusion
sits behind its own flag, default off. My synthetic router (`normal * 0.02`)
produces an unusually flat logit distribution; a trained router is more decisive
and the disagreement rate should be lower, but I have not measured that on real
weights and will not claim it.

## Speed

`bench.py`, min-of-15 with stock and fused interleaved round-robin (the GPU is
shared with the live daemon on 8083; the median is contaminated by contention,
the minimum is the clean kernel time). The bench waits for the copy ceiling to
come back above 530 GB/s before measuring and prints it.

Full routed MLP block (router + experts + weighted sum), E=512:

| T | stock | fused (2 kernels) | fused + fused router | stock launches | fused launches |
|---|---|---|---|---|---|
| 1 | 88.1 us | 80.1 us (1.10x) | **71.3 us (1.23x)** | 14 | 10 / **4** |
| 2 | 130.9 us | 116.5 us (1.12x) | **106.4 us (1.23x)** | 14 | 10 / **4** |
| 8 | 447.8 us | 447.8 us (1.00x, falls back) | 1.00x | 23 | 23 |

Launch counts are MLX graph primitives excluding metadata-only nodes
(`Reshape`, `Broadcast`, `Squeeze`, ...), from `mx.export_to_dot`. Stock T=1:
`QuantizedMatmul, Softmax, ArgPartition, Slice, GatherAxis, Sum, Divide,
GatherQMM x2, Split, Compiled(silu*mul), Sum, Multiply, Arange x2` = 14. With
everything fused: `router_logits, router_topk, gate_up_silu, down_wsum` = 4.

Sub-step timings at T=1 (each measured alone, so they do not sum to the block
total, which pipelines):

| step | stock | fused |
|---|---|---|
| router | 28.3 us (6 kernels) | 23.3 us (2 kernels) |
| gate+up+act | part of 67.2 us | 36.4 us |
| down + weighted sum | part of 67.2 + 21.9 us | 30.1 us |

`gate_up_silu` moves 18.4 MB in 36.4 us = **505 GB/s, 92% of the measured copy
ceiling**. `down_wsum` moves 9.2 MB in 30.1 us = 306 GB/s; at T=8 the same kernel
reaches 586 GB/s, so its T=1 shortfall is dispatch latency and occupancy, not the
inner loop. The whole fused path moves 28.3 MB per layer, which is 51.5 us at
549 GB/s, so 71.3 us is 72% of the achievable floor. There is maybe 15 us per
layer still on the table and it is all in `down_wsum` and the router.

Expert kernels only (`switch_mlp` + weighted sum), to show where the fused
approach stops paying:

| T | stock | fused | ratio |
|---|---|---|---|
| 1 | 70.5 | 56.2 | 1.25x |
| 2 | 112.1 | 86.1 | 1.30x |
| 4 | 226.8 | 169.1 | 1.34x |
| 8 | 450.2 | 337.4 | 1.33x |
| 16 | 833.6 | 671.5 | 1.24x |
| 32 | 1587.6 | 1432.0 | 1.11x |
| 64 | 3506.4 | 3312.3 | 1.06x |

The kernels have no weight reuse across tokens: they re-read an expert's rows for
every routed (token, expert) pair. Below ~64 routed pairs per expert that costs
nothing because there is no reuse to be had, which is why they stay ahead out to
T=64. At prefill widths (2048 tokens, 40 rows per expert) the sorted `gather_qmm`
would win by a mile, which is what the threshold is for.

## Projected whole-model decode gain

48 layers (36 GDN + 12 attention) and the routed MLP runs in every one.

| regime | per layer | per token, 48 layers | delta |
|---|---|---|---|
| T=1 stock | 88.1 us | 4.23 ms | |
| T=1 fused experts | 80.1 us | 3.84 ms | **-0.38 ms** |
| T=1 fused experts + router | 71.3 us | 3.42 ms | **-0.81 ms** |
| T=2 (MTP verify) stock | 130.9 us | 6.28 ms | |
| T=2 fused + router | 106.4 us | 5.11 ms | **-1.17 ms** |

Against the observed oQ4e decode of 35-68 tok/s (14.7-28.6 ms per token):

- experts only (default config): **+1.3% to +2.6%** end-to-end tok/s
- experts + router (opt-in): **+2.9% to +5.8%** end-to-end tok/s
- MTP verify steps get 1.23x on their MoE portion, so a verify-heavy schedule
  sees somewhat more than the single-token number

That is the honest number and it is smaller than the launch-count reduction
suggests. The reason is the premise: 14 launches per layer does not cost
14 x 20 us, because MLX submits a whole layer's graph in one command buffer and
the small kernels pipeline. A dispatch measured in isolation costs 14-18 us
(I measured `tiny add` on 16 floats at 14.4 us, `rms_norm` on 2560 at 14.3 us, a
no-op JIT kernel at 17.8 us), but inside a chain the marginal cost of a small
kernel is closer to 4-5 us. So collapsing 14 launches to 4 buys about 17 us per
layer, not 200. What is left after that is real weight traffic: 28.3 MB per layer
per token, 1.36 GB per token across 48 layers, which is 2.5 ms at the copy
ceiling no matter how the kernels are written.

If the threshold is raised to 8 (`OMLX_MOE_DECODE_FUSED_MAX_TOKENS=8`, measured
safe above, 1.33x on the expert kernels), an 8-way batched decode step goes from
21.5 ms to about 17.5 ms of MoE time per step.

## How to enable in oMLX

```
export OMLX_MOE_DECODE_FUSED=1            # the two expert kernels, T<=2
export OMLX_MOE_DECODE_FUSED_MAX_TOKENS=8 # optional, measured safe to 16
export OMLX_MOE_DECODE_FUSED_ROUTER=1     # optional, see the caveat above
```

then call `kernel.apply()` once after load. It patches `__call__` on
`Qwen3MoeSparseMoeBlock` (mlx-lm), `SparseMoeBlock` (mlx-lm qwen3_5),
`Qwen3NextSparseMoeBlock`, and `Qwen3_5MoeSparseMoeBlock` (mlx-vlm), following
the structure of `qwen35_moe_weighted_sum.py`. The patched call does one shape
test before touching env or module state, so the cost on non-eligible calls is a
single comparison (issue #2132). Any exception inside the fast path logs a
warning and returns the stock result.

**Prerequisite**: the oMLX gate+up fusion (`qwen35_moe_gate_up.py`,
`OMLX_QWEN35_MOE_GATE_UP=1`, the default) must have run, so `switch_mlp` exposes
a single `gate_up_proj` of shape `[E, 2I, K]`. Blocks that still have separate
`gate_proj`/`up_proj` fall back, because the kernel takes one weight buffer.

## Limitations and honest failures

- **4-bit affine only.** Group size 64 or 128, no bias, `mode="affine"`. 8-bit
  gs64 (the second target in COMMON.md) is not implemented; it needs a different
  unpack (4 values per word instead of 8) and a separate template instantiation.
  Everything else falls back.
- **Not bit-identical to stock, by construction.** fp32 accumulation through the
  activation and the weighted sum means the output differs from stock at the
  ~2.5 bf16 ULP level, in the direction of the fp32 golden. If a caller needs
  bit-reproducibility against stock, this patch is not it.
- **The router fusion can pick a different expert.** See above. Default off.
- **No sharded-expert support.** `sharding_group is not None` falls back.
- **Target-verify / extra call arguments fall back.** The patched `__call__`
  declines as soon as it sees `*args`/`**kwargs`, so MTP verify keeps the stock
  path even at T=2. Wiring the verify signature in is straightforward but I did
  not want to guess at the mlx-vlm contract without a checkpoint to test on.
- **Never run on a real checkpoint.** Everything here is on synthetically
  quantized weights at the exact production shapes. The whole-model projection is
  arithmetic on per-layer measurements, not an end-to-end tok/s measurement.
- **Benchmarking on a shared GPU is the biggest source of error in this report.**
  Individual runs of the same measurement varied by 40% and one early run showed
  a 20x outlier. The paired min-of-N harness plus the wait-for-quiet gate
  handles it, but treat single digits of percent in the tables as noise.
- **A bug worth recording**: my first baseline measured the stock block at
  1071 us per layer, 12x too slow. `nn.RMSNorm` creates an fp32 weight, so the
  normed activation was fp32, and `gather_qmm` with fp32 activations against
  bf16 scales falls off its fast path (1647 us for a call that takes 46 us in
  bf16). That is a real trap for anyone assembling a synthetic baseline, and
  possibly worth checking in real model-loading code paths too.

## Next steps, in the order I would do them

1. Close the `down_wsum` gap at T=1 (306 GB/s vs 505 for the gate/up kernel).
   `ROWS` and `SPLIT` sweeps did not move it, so the next thing to try is
   restructuring the 80-word row so the last lane-strided iteration is not a
   half-warp (80 = 32+32+16), for instance by having a SIMD group cover two
   experts at once.
2. 8-bit affine gs64 instantiation. Same kernels, different unpack.
3. Wire up the target-verify signature so MTP verify at T=2 gets the 1.23x
   instead of falling back.
4. Measure the router top-k disagreement rate on real oQ4e router weights. If it
   is near zero, the router fusion can become default-on and the gain roughly
   doubles.
5. Raise the default threshold to 8 once there is an end-to-end tok/s
   measurement to back it.

## Files

- `fused_kernels.py` - the three JIT Metal kernels and their Python wrappers
- `kernel.py` - patch module, `apply()`, eligibility checks, fallbacks
- `moe_ref.py` - faithful stock block (mlx-lm `SwitchGLU` + `Qwen3MoeSparseMoeBlock`
  with the oMLX gate+up fusion applied), built at the production shapes without
  materialising the 6.7 GB fp weight tensor
- `test_exact.py` - fp32-golden correctness, end-to-end patched-vs-stock,
  router agreement
- `bench.py` - before/after timing, launch counts, crossover sweep
- `baseline.py`, `graphcount.py` - timing harness and primitive counter
