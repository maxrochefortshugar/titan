# Fused prefill hyper-connection block (`OMLX_QWEN4_HC_FUSE2=1`)

Workstream `kernels/round3/hc-fuse/`, 2026-09-12. No model loaded, synthetic tensors only, under
600 MB, ports 8083 and 8084 untouched.

## What was replaced

`Qwen4ExpGatedResidual.__call__` sends calls above 16 rows to `hc_fused.prefill_forward`
(`LANG:1669-1671`, `hc_fused.py:415-451`), which production rebinds through
`kernels/ple-fix/norm_patch.py`. That deployed path is eleven Metal launches:

| # | launch | traffic at T=2048 |
|---|---|---|
| 1 | `hc_fused._kernel_norm` | 42 MB read, 42 MB write |
| 2 | `input_mix_weight_down` qmm (`hc_fused.py:428`) | 42 MB read |
| 3-5 | `nn.silu(mix / hc)`: divide, sigmoid, multiply | 1.3 MB each |
| 6 | `block_inject_weight` qmm | 42 MB read |
| 7-9 | `2 * sigmoid(inj / hc)`: divide, sigmoid, multiply | 16 KB each |
| 10 | `input_mix_weight_up` qmm | 42 MB write |
| 11 | the compiled tail (`hc_fused.py:390-402`) | 84 MB read, 10.5 MB write |

`kernel.py` keeps launches 2, 6 and 10 on MLX and fuses the rest into three: the bf16 grouped norm,
one epilogue that splits both projection results and applies both activations, and one tail that
gates, multiplies by the normed stream and averages the four streams. Six launches instead of
eleven, fp32 accumulation in the norm reduction and every dot product, bf16 storage throughout.

**Why the projections stay on MLX, and why the stream is still read six times.** The three matmuls
measure 0.70 ms of the deployed call's 1.17 at T=2048; a hand written replacement runs on the fp32
ALU near 15 TF/s and would cost ~1.8 ms alone, which is why oMLX caps its own hand written
`fused_forward` at `MAX_ROWS = 16`. Each of the six stream passes is then an operand of, or a result
from, one of those matmuls, and MLX can neither consume a stream that was never written nor emit the
10240-wide up-projection anywhere but memory.

The patch **subsumes** `norm_patch.py`: the fused block calls the same `hc_fused._kernel_norm` and carries
the `_omlx_bf16_norm` marker `apply_bf16_norm_patch` tests, so whichever hook runs last leaves the
fused block in place (`test_patch.py` checks both orders). `OMLX_QWEN4_BF16_NORM=1` still governs
the three PLE norms under `_ALL`, untouched here.

## Exactness

Reference is the deployed path; canonical fp32-norm `_forward` (`LANG:1684-1757`) is alongside so
the bf16 norm's own cost stays visible. The measure is bf16 bit-pattern distance, the only one
meaningful where the stream terms cancel. Random [1, T, 10240] bf16, unit scale unless noted.

| case | tensor | max abs | max ULP vs deployed | mean ULP | max ULP vs canonical |
|---|---|---|---|---|---|
| attn 5-bit T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 29952 / 0 |
| mlp 5-bit T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 15104 / 0 |
| attn 4-bit T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 14848 / 2 |
| mlp 6-bit T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 29696 / 1 |
| attn 8-bit T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 128 / 0 |
| mixed 4/5-bit banks T=2048 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 29632 / 2 |
| mixer, no injection T=2048 | mixed | 0 | **0** | 0 | 14976 |
| 5-bit T=2048, stream x8 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 192 / 0 |
| 5-bit T=2048, stream /8 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 30080 / 0 |
| 5-bit T=512 | mixed / inject | 0 / 0 | **0 / 0** | 0 / 0 | 96 / 0 |
| **4 layers chained, T=2048** | stream | **0** | **0** | 0 | 30592 (mean 1.25) |

Bit identical to the deployed path everywhere, chain included, so no drift accumulates. That took
reproducing MLX's rounding rather than working in fp32. `Sigmoid` (`unary_ops.h:308-314`) is the
stable `y = 1/(1+exp(|x|)); x<0 ? y : 1-y` form whose every bf16 step rounds (`bf16_math.h:77`), and
`mx.mean` over a bf16 axis carries a **bf16** accumulator (`probe_mean.py`). fp32 accumulation of
the mean is more accurate but moves results up to 11 ULP where the four terms cancel, so it sits
behind `OMLX_QWEN4_HC_FUSE2_FP32_MEAN=1`, off. The last column is the deployed norm's own inherited
deviation: mean 0.006 to 0.018 ULP, big maxima only where the mean cancels, 1.25 mean ULP after
four layers.

### The merged input projection, rejected

Merging `input_mix_weight_down` and `block_inject_weight` into one [324, 10240] bank removes a 42 MB
read and a launch, and 91 of 96 blocks have matching bit widths for it. The 320 low-rank rows come
out **bit identical**, and so does `mixed`. The four injection rows do not: standalone an N=4 matmul,
in the bank four rows of a wide one, a different reduction order over K=10240, measured at 2 to 4
ULP (max abs 1.6e-2 on values in [0, 2]) and 10.7 mean ULP after four layers (`probe_concat.py`).
It stays behind `OMLX_QWEN4_HC_FUSE2_CONCAT=1`, off, at 0.22 GB when on. The same argument blocks
folding the injection into the norm kernel, otherwise nearly free.

## Microbenchmark

Median of 15, warm, `CHAIN=10` per eval, `mx.synchronize` around each timing, as in
`~/inference-server/kbench.py`, run at 13:53 while `GPU_FREE` existed. 5-bit banks, hidden 2560,
hc 4, lowrank 320. GB/s counts six stream passes, the output and ~3.4 MB of weights.

| path per call | T=512 | GB/s | T=2048 | GB/s | x96 at T=2048 | vs deployed |
|---|---|---|---|---|---|---|
| canonical `_forward` | 0.805 ms | 86 | 2.365 ms | 113 | 227.1 ms | 0.50x |
| shipped `prefill_forward` | 0.674 ms | 103 | 2.034 ms | 132 | 195.3 ms | 0.58x |
| **deployed** (bf16 norm) | 0.505 ms | 138 | 1.171 ms | 229 | 112.4 ms | 1.00x |
| **hc-fuse2** | **0.475 ms** | 146 | **1.011 ms** | 265 | **97.0 ms** | **1.16x** |

A second 21-iteration run gave 0.482 and 0.996 ms, so T=2048 is **1.16 to 1.19x, +15 to +18 ms per
chunk**, about 1.5% of a ~1000 ms body; deployed at 112 ms per chunk also puts a measurement under
the audit's 133 ms for this block. Stages at T=2048 show where it comes from:

| stage | stock | hc-fuse2 |
|---|---|---|
| grouped norm | 1.125 ms fp32, 0.158 ms bf16 kernel | 0.158 ms, same kernel |
| down qmm | 0.284 ms | unchanged |
| inject qmm | 0.135 ms | unchanged |
| silu chain + injection chain | 0.028 + 0.024 ms | 0.028 ms, one epilogue |
| up qmm | 0.280 ms | unchanged |
| tail | 0.349 ms compiled | **0.159 ms** |

The tail is most of it: `mx.compile` over four strided slice products costs 0.349 ms where one
custom kernel over the same bytes costs 0.159, worth 18 ms per chunk alone. At 265 GB/s the block
sits at 37% of the read ceiling, bound by kernel count and by two matmuls at small N and K rather
than by bandwidth. Decode is unchanged. The concat variant measured 0.934, 0.980 and 1.779 ms across
three runs, too unstable to quote.

## How to enable

```
OMLX_QWEN4_HC_FUSE2=1
OMLX_ROUND2_PATCHES="$HOME/inference-server/kernels/round3/hc-fuse/patch.py"
```

`install(model=None) -> bool` is idempotent and returns False, stock path intact, when the gate is
off or `hc_fused` is unavailable. It rebinds a module function, not instances, so import time and
post-load both work; any failure in a call returns None, which sends `__call__` to `_forward`.

```
cd ~/inference-server/kernels/round3/hc-fuse
~/inference-server/kdev/bin/python test_exact.py --rows 2048 --chain-rows 2048
~/inference-server/kdev/bin/python test_exact.py --rows 2048 --chain-rows 512 --concat
~/inference-server/kdev/bin/python test_patch.py
~/inference-server/kdev/bin/python bench.py --iters 15 --concat --stages --json bench.json
```

## Limitations

- +1.5% of prefill, against the audit's +6% for item 10: the stream still moves six times.
- The headroom left is fusing the up projection with the tail, worth 84 MB per call, which needs a
  quantized bf16 simdgroup-matrix GEMM in a custom kernel to hold the tensor units.
- Relaxing the bar to a tolerance on the block output rather than on the reduction order buys two
  more 42 MB reads and two launches, via the merged bank and a norm-fused injection.
- Decode (T=1) is untouched; batched decode and verify above 16 rows do take this path.
