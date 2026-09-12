# GDN NAX kernel: exact, but then no longer fast

2026-09-12. No model loads, no server, under 60 MB of GPU memory, ports 8083/8084 untouched.
Microbenchmarks ran only while `~/inference-server/staging/GPU_FREE` existed. Real shapes:
B=1, Hk=16, Dk=128, Hv=48, Dv=128, 36 GDN layers.

The 5.4e-4 state error is fixable, and the fix costs the speed it was buying. Corrected, the NAX
C=16 kernel reaches state rrmse 1.0e-6 worst case with output matching the stock bf16 path, and
runs at 3.06 ms per layer against C=8's 1.66 ms. Ship gdn-scan's C=8 kernel; leave
`OMLX_QWEN4_GDN_SCAN_NAX` off.

## 1. Diagnosis

Not the fragment copies, not C=16, not the input dtype: `matmul2d_descriptor`'s
`relaxed_precision`, and it is an input quantization rather than a loose accumulation.

Two probes on one isolated 16x16x16 product through the PR's own `mma` helper settle it.
`probe_format.py` multiplies by the identity, so the result is `round_F(A)` and reads the format
out directly: 11 significand bits, full fp32 exponent range, max relative error 9.3e-4, checked
from 1e-30 to 1e30. `probe_accum.py` shows the accumulator is honest, 2.3e-8 for operands already
exact in that format against 7.2e-4 for fp32 ones. Every matmul rounds both operands to 11 bits
and the fp32 state passes through several per chunk, which is exactly 5e-4. C=16 is innocent:
corrected, the same algebra matches C=8.

The `relaxed_precision=false` garbage is the indexing mismatch the brief suspected, now decoded.
`cooperative_tensor::get_multidimensional_index(i)` reports each thread's coordinates;
`probe_precise_layout.py` and `probe_precise_formula.py` dump both modes and verify a closed form
across all 32 lanes (section 6 states it). mlx's steel header hardcodes `true` in both overloads,
so there was no precise-mode reference to copy.

## 2. Fix

`_nax2_macros.h` splits each operand as `x = hi + lo`, `hi` masked to 11 bits, both exactly
representable, and runs `hi*hi + hi*lo + lo*hi` into one fp32 accumulator: three `run` calls, one
copy of C in and out, about 22 bits recovered. Each of the nine matmul sites carries a two-bit
mode (off, three-pass, correct the left operand, correct the right) as a template argument. Mode
0 reproduces the PR bit for bit, which is how the port was verified. `KK^T` and `QK^T` need
nothing, since q and k come from bf16 memory. The other seven matter. bf16 cooperative-tensor
operands, tried to shrink the operand registers, are slower (4.37 ms) and looser (8.1e-6).

## 3. Exactness

Reference is the exact per-token recurrence in fp32; inputs built as in gdn-scan. Worst case over
T = 512, 1024, 2048, 2049 and four seeds (`test_exact.py`, `sweep_seeds.py`).

| path | worst y rrmse | worst state rrmse |
|---|---|---|
| stock seq kernel, bf16 | 1.662e-03 | 0 (reference dtype) |
| oMLX blocked_seq | 1.662e-03 | 2.6e-08 |
| PR C=8 | 1.662e-03 | 6.14e-07 |
| PR NAX C=16, as ported | 2.004e-03 | 7.91e-04 |
| **NAX three-pass (default)** | **1.662e-03** | **9.97e-07** |
| NAX, W S^T corrected one side | 1.662e-03 | 2.89e-05 |
| NAX, four load-bearing sites only | 1.828e-03 | 2.43e-05 |

The default clears the 1e-5 bar by 10x, sits 1.6x off C=8, and its y error equals the stock
kernel's to four digits at every length and seed. `test_carry.py`, four 512-token chunks feeding
state forward against one fp32 pass over 2048, gives per-chunk y rrmse 1.662, 1.655, 1.652,
1.659e-03, flat and identical to stock, and a final state rrmse of 6.39e-07, the same as a single
2048 pass. No drift.

## 4. Microbench

Per layer, warm, median of 15, 4 ops per `mx.eval`, `mx.synchronize` around each sample.

| T | path | ms/layer | vs stock | ms/chunk x36 |
|---|---|---|---|---|
| 512 | stock seq kernel | 0.792 | 1.00x | 28.5 |
| 512 | oMLX blocked_seq (deployed) | 0.480 | 1.65x | 17.3 |
| 512 | PR C=8 | 0.451 | 1.76x | 16.2 |
| 512 | NAX as ported (6.8e-4) | 0.278 | 2.84x | 10.0 |
| 512 | NAX one-sided W S^T (2.9e-5) | 0.459 | 1.73x | 16.5 |
| 512 | **NAX three-pass (1.0e-6)** | **0.822** | **0.96x** | 29.6 |
| 2048 | stock seq kernel | 3.063 | 1.00x | 110.3 |
| 2048 | oMLX blocked_seq (deployed) | 1.825 | 1.68x | 65.7 |
| 2048 | PR C=8 | 1.660 | 1.85x | 59.8 |
| 2048 | NAX as ported (5.4e-4) | 0.980 | 3.12x | 35.3 |
| 2048 | NAX one-sided W S^T (2.9e-5) | 1.781 | 1.72x | 64.1 |
| 2048 | **NAX three-pass (1.0e-6)** | **3.060** | **1.00x** | 110.2 |

The cost is not spread evenly. `bench_bits.py` prices each site at T=2048 against the 0.98 ms
unsplit kernel: WY inverse +0.21, Q S^T +0.17, W panel +0.15, state update +0.09, U +0.07, out
+0.03. Then `W S^T` alone costs +1.99 ms, for 8 of the kernel's 66 mma units. That is a register
spill, not arithmetic: the one site where both operands are multi-fragment tiles kept live across
three `run` calls, on top of an 8-fragment state tile, in a kernel already near the register
limit. Correcting one operand there halves the penalty and gives up 30x of accuracy, landing at
2.9e-5. The accuracy is reachable, the combination is not, short of restructuring the inner loop
to shrink the live set around the state tile.

## 5. Deliverables

`kernel_nax2.py` (`mask()`, `DEFAULT`, `FAST`), `_nax2_macros.h`, `_nax2_body.metal` (the PR body
with a mode argument per call, otherwise verbatim), `naxhdr.py` (flattens mlx's shipped steel NAX
header for the JIT), five probes, `patch.py`.

`patch.py` exposes `install() -> bool`, idempotent, off unless `OMLX_QWEN4_GDN_SCAN_NAX=1`, same
hook point and fallthrough rules as gdn-scan's, which is untouched. Precedence: it clears
`OMLX_QWEN4_GDN_SCAN` once it binds, so a later gdn-scan `install()` returns False; an earlier one
is wrapped. Module functions only, no instances, so import time or post-load both work.
`OMLX_QWEN4_GDN_NAX_SPLIT` overrides the site mask. Leave the flag off; the kernel is here in case
the register pressure is ever fixed.

```
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/test_exact.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/test_carry.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/sweep_seeds.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/test_patch.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/probe_format.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/bench.py      # GPU_FREE only
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-nax/bench_bits.py # GPU_FREE only
```

## 6. Upstream text for ml-explore/mlx pull/4020

(Verbatim issue text, not counted in the report body.)

> `gated_delta_fused_nax` returns a final state about 5e-4 relative off the sequential recurrence,
> against 6e-7 for `gated_delta_fused_chunk`, at Dk=Dv=128, Hk=16, Hv=48, bf16 inputs, fp32 state,
> T from 64 to 2048. M5 Max, mlx 0.32.2 plus this PR.
>
> The cause is `relaxed_precision=true` in the `matmul2d_descriptor`s behind the kernel's `mma`
> helpers, which `steel/gemm/nax.h` also hardcodes. On this hardware that mode rounds both
> operands to an 11-bit significand and accumulates in fp32: multiplying by the identity returns
> the operand with its low 13 mantissa bits cleared, and operands already exact in that format
> come back at rrmse 2e-8 against fp64. Harmless for a GEMM over bf16 activations, but this kernel
> pushes the fp32 recurrent state through `W S^T`, `Q S^T` and the state update, so the state
> inherits that rounding once per chunk.
>
> `relaxed_precision=false` produces garbage because the register layout changes and the helpers
> assume the relaxed one. The precise layout is readable from
> `cooperative_tensor::get_multidimensional_index(i)`: rows map as in `BaseNAXFrag`, but a thread's
> four columns become `((qid & 2) | (lane & 1)) * 2 + {0,1,8,9}` rather than
> `((qid & 2) | (lane & 1)) * 4 + {0,1,2,3}`, and a 16x32 operand's second N-fragment sits at
> element indices 4 to 7, not 8 to 15.
>
> A workaround inside the relaxed path is to split each operand into an 11-bit `hi` and its
> residual and run `hi*hi + hi*lo + lo*hi` into one accumulator, which brings the state to 1e-6.
> At the seven sites that need it that costs 3.1 ms per layer against 0.98 ms as it stands, and
> most of the increase comes from `W S^T` alone, where both operands are multi-fragment tiles held
> live across the three passes. Shrinking the live set around the state tile, or a correct
> precise-mode path, is worth more than the split.
