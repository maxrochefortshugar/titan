# GDN prefill scan: mlx PR #4020 ported to a JIT kernel

2026-09-12. No model loads, no server, under 60 MB of GPU memory, ports 8083/8084 untouched.
All numbers at the real head shape B=1, Hk=16, Dk=128, Hv=48, Dv=128, 36 GDN layers.

## 1. The PR

pull/4020 at `c7e1a2a` (`mlx-src/`, branch `pr4020`) adds
`mx.fast.gated_delta_update(q, k, v, gamma, beta, initial_state, mask)`: a C++ sequential fallback
plus two WY-form chunked delta-rule Metal kernels, `gated_delta_fused_chunk` (C=8,
`simdgroup_float8x8`) and `gated_delta_fused_nax` (C=16, MPP cooperative tensors).
`gated_delta_update.cpp:76-84` takes C=16 when `is_nax_available()` and T >= 16, and `use_fallback` demands Dk = Dv = 128, one of six head pairs
including `(16, 48)`, and no mask. Layouts: q/k `[B,T,Hk,Dk]`, v/y `[B,T,Hv,Dv]`, g/beta
`[B,T,Hv]` with g in linear space and beta sigmoided, state `[B,Hv,Dv,Dk]` fp32 both ways.

## 2. What runs today

`Qwen4ExpGatedDeltaNet` (`LANG:1091`) inherits `Qwen3_5GatedDeltaNet.__call__`, which calls
`gated_delta_update` at site-packages `mlx_vlm/models/qwen3_5/language.py:1681` (reached because
the compat shim appends the vendor tree to `mlx_vlm.models.__path__` rather than shadowing it),
then `qwen3_5/gated_delta.py:126`, then `mlx_lm.models.gated_delta.gated_delta_kernel` (`:167`):
a per-token sequential loop, already a `mx.fast.metal_kernel`, byte-identical to the PR's own
`gated_delta_seq`. State is fp32 in `cache[1]` (`language.py:1714`), carried across prefill chunks
and into decode. oMLX rebinds the symbol at `patches/qwen35_gdn_chunked.py:99-100` to
`gated_delta_blocked_seq` (`qwen35_prefill/gdn.py:569`) for T >= 64 unmasked scalar gating, so the
deployed baseline is oMLX's kernel, not the stock one.

## 3. Strategy: (a), JIT metal_kernel

Option (b) is unnecessary. The stock path is already a JIT metal_kernel with the exact signature
the PR's kernel wants, so this is a source move, not a build, and the pinned mlx_vlm/mlx_lm APIs
are untouched. `kernel.py` inlines the C=8 template instantiation (InT, Dk, Dv, Hk, Hv, C become
mlx template args), puts the macros in `header=` and the body in `source=`, and lets metal_kernel
supply the `[[kernel]]` signature. NAX ports too, which I did not expect: mlx's JIT has no include
search path, but it ships the steel NAX header, which `kernel_nax.py` flattens.

## 4. Exactness

Reference is the exact per-token recurrence in fp32: the stock sequential kernel on fp32 inputs
and state. Inputs follow the real construction (q, k L2-normalised, q scaled by `Dk**-0.5`,
A ~ U(0,16), dt_bias 1 so g sits just under 1, beta = sigmoid(b-2) ~ 0.12, nonzero random initial
state). rrmse is `||x-ref||/||ref||`. `test_exact.py`.

| T | path | y absmax | y rrmse | state absmax | state rrmse |
|---|---|---|---|---|---|
| 64 | stock seq kernel, bf16 | 1.14e-04 | 1.666e-03 | 0 | 0 |
| 64 | oMLX blocked_seq, bf16 | 1.14e-04 | 1.666e-03 | 7.45e-09 | 6.03e-09 |
| 64 | **PR C=8, bf16** | 1.14e-04 | **1.666e-03** | 2.42e-07 | **4.76e-07** |
| 64 | PR NAX C=16, bf16 | 1.33e-04 | 1.914e-03 | 2.34e-04 | 6.69e-04 |
| 512 | stock seq kernel, bf16 | 1.17e-04 | 1.657e-03 | 0 | 0 |
| 512 | oMLX blocked_seq, bf16 | 1.17e-04 | 1.657e-03 | 1.49e-08 | 7.15e-09 |
| 512 | **PR C=8, bf16** | 1.17e-04 | **1.657e-03** | 1.62e-07 | **3.99e-07** |
| 512 | PR NAX C=16, bf16 | 1.65e-04 | 1.917e-03 | 4.22e-04 | 6.83e-04 |
| 2048 | stock seq kernel, bf16 | 1.22e-04 | 1.657e-03 | 0 | 0 |
| 2048 | oMLX blocked_seq, bf16 | 1.22e-04 | 1.657e-03 | 1.49e-08 | 4.95e-09 |
| 2048 | **PR C=8, bf16** | 1.22e-04 | **1.657e-03** | 4.53e-07 | **6.14e-07** |
| 2048 | PR NAX C=16, bf16 | 1.53e-04 | 1.915e-03 | 2.84e-04 | 5.37e-04 |

Fed fp32 inputs, C=8 drops to y rrmse 5.6e-07, so the bf16 row's 1.657e-03 is the rounding of y.

Tolerance. Judge on the state, since that is what survives the chunk. oMLX's deployed kernel
departs from the sequential scan by 5e-09 relative, C=8 by 6e-07: two orders worse, far under a
bf16 ULP here, and the output still matches stock bf16 to three decimals. `test_carry.py` runs
four 512-token chunks feeding state forward against one fp32 pass over 2048: per-chunk y
rrmse 1.662, 1.655, 1.652, 1.659e-03, flat, and the final state error is the same 6.1e-07 as a
single 2048 pass. The chunked error does not compound over a long prefill, the failure mode that
would bite at 65k. Bar: state rrmse under 1e-05, cleared 16x. Caveat: the WY inverse is
conditioned only because k is L2-normalised, bounding `KK^T` by 1; Qwen4-Exp normalises
(`LANG:1100-1107`).

NAX C=16 misses that bar at 5.4e-04, and its fp32 run is no better, so this is the algorithm, not
the dtype. The cause is `matmul2d_descriptor`'s sixth argument, `relaxed_precision`, set `true` by
the PR and by mlx's own steel NAX GEMM; `false` yields garbage (rrmse 1.1), so the fragment copies
are tied to the relaxed layout. A 5e-04 state error through 32 chunks is not something to ship on
synthetic evidence.

## 5. Microbench

Per layer, warm, median of 15, 4 ops chained per `mx.eval`, `mx.synchronize` around each sample;
chain 1 reproduces every figure within 2%. `bench.py`.

| T | path | ms/layer | vs stock | ms/chunk x36 |
|---|---|---|---|---|
| 512 | stock seq kernel | 0.799 | 1.00x | 28.8 |
| 512 | oMLX blocked_seq (deployed) | 0.480 | 1.66x | 17.3 |
| 512 | **PR C=8** | 0.457 | **1.75x** | 16.5 |
| 512 | PR NAX C=16 | 0.277 | 2.88x | 10.0 |
| 2048 | stock seq kernel | 3.097 | 1.00x | 111.5 |
| 2048 | oMLX blocked_seq (deployed) | 1.873 | 1.65x | 67.4 |
| 2048 | **PR C=8** | 1.683 | **1.84x** | 60.6 |
| 2048 | PR NAX C=16 | 0.979 | 3.16x | 35.2 |

1.84x at T=2048 lands on the PR's published 1.86x, so the port carries the number across intact.
Saved per 2048-token chunk over 36 layers: **6.8 ms** for C=8 against the deployed kernel, 50.9 ms
against stock, 32.2 ms for NAX. On the audit's ~1000 ms body, 0.7% and 3.2%.

One correction. The audit calls oMLX's kernel a wash, but that pair was unmatched: measured side
by side, `gated_delta_blocked_seq` is a real 1.65x and it is bound in production. The deployed
scan costs **67 ms per chunk, not 123**, so item D9's pool is 67 ms, not 123.

## 6. Deliverables

`kernel.py` is the C=8 kernel plus a `supported()` shape gate, `kernel_nax.py` with
`_nax_macros.h` and `_nax_body.metal` the C=16 variant, `mlx-src/` the PR checkout.

`patch.py` exposes `install() -> bool`, idempotent, off unless `OMLX_QWEN4_GDN_SCAN=1`. It rebinds
`gated_delta_update` in `mlx_vlm.models.qwen3_5.{gated_delta,language}` for T >=
`OMLX_QWEN4_GDN_SCAN_MIN_T` (default 64), unmasked, scalar gating, supported shape and dtype;
everything else falls through to whatever was bound first, so **install after oMLX's own
GDN patch**. It monkeypatches module functions and touches no instances, so import time or
post-load both work and only that ordering matters. `install()` JIT-compiles once and returns
False on failure. `OMLX_QWEN4_GDN_SCAN_IMPL=nax` selects C=16.

```
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-scan/test_exact.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-scan/test_carry.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round3/gdn-scan/bench.py
```

An end-to-end check against stub `mlx_vlm` modules (kdev has none) confirms install, idempotence,
prefill routed at y rrmse 1.660e-03 and state rrmse 4.93e-07, and T=1 and masked calls falling
through.

## 7. Limits and next steps

C=8's 6.8 ms per chunk is free and low risk but small. The interesting number is NAX's 32 ms,
blocked on whether its 5.4e-04 state error is inherent to relaxed-precision cooperative tensors or
a bug in the PR's fragment handling. The PR is open and was active two days ago, so post the
measurement there rather than debug it here. Not tested: the real model (banned), B > 1 and the
left-padded batch path, the contended 512-token chunk.
