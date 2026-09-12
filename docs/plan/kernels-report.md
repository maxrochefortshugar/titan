# Kernel workstreams: combined report (2026-09-11, M5 Max 40c, 128 GB, macOS 26.5, mlx 0.32.2)

Three Opus agents each delivered a JIT Metal kernel in oMLX patch style (mx.fast.metal_kernel, env-gated, stock fallback), an exactness test, a benchmark and a REPORT.md in their directory. All three exactness tests were re-run by the orchestrator and pass. Nothing was installed into the running daemon.

## Scoreboard

| Workstream | Kernel-level result | Whole-model effect (est.) | Correctness | Status |
|---|---|---|---|---|
| 1. Small-M matmul (smallm/) | lm_head at M=8: 4-bit 3.16 -> 1.31 ms (2.4x), 8-bit 3.94 -> 1.25 ms (3.2x); q_proj M=8/M=1 = 1.24x (stock ~2x) | Batched decode and MTP verify: removes the 7x lm_head penalty at 2..8 rows | argmax 100% vs stock at M=2..16; routed-band error at bf16 ULP; fallback bit-identical | Ready for in-situ trial |
| 2. MoE decode fusion (decode-fusion/) | Routed-MLP layer T=1: 88 -> 71 us (1.23x), launches 14 -> 4; T=2 1.23x; beats stock out to T=64 | Only +3% to +6% single-stream decode | within 1 bf16 ULP of fp32 golden, closer than stock; T>=8 falls back bit-exact; router fusion opt-in (1/40 top-k tie flips) | Correct, low payoff |
| 3. Native int4 prefill (prefill-int4/) | q_proj/o_proj at M=2048: 1.19x / 1.12x via int8 activations x int4 weights on the tensor units; measured NAX peaks | 2-3% now (dense projections only); 12-15% once the MoE gather is converted; ~30% with macOS 27 cooperative tensors | adds 0.2-1.2% relative error on top of the 9.1% the 4-bit weights already carry; argmax 98% vs stock (stock itself 96.7% vs fp32) | Prototype; MoE gather not done |

## What we learned (this changes the earlier estimates)
1. M5 Max NAX ceilings, measured for the first time: bf16 65.7 TFLOPS, int8 x int8 129.8, int8 x int4 120, bf16 x int4 65.5, fp32 15.7. A 4-bit weight operand buys no tensor-core time; the wider operand sets the rate. Stock quantized_matmul at M=2048 already runs at 87% of the bf16 ceiling, so "skip the bf16 expansion" was worth at most 15%, not 2x. Real prefill upside needs int8 activations (done, 1.1-1.2x) and macOS 27 cooperative tensors (est. +1.5x on the K loop).
2. The "20-35 us per small kernel" premise was an artefact of isolated timing. Inside a layer's command buffer a small kernel costs ~4-5 us, so launch fusion buys ~17 us per layer, not ~200. Decode is genuinely bandwidth-bound: 1.36 GB of routed-expert weights per token is 2.5 ms at the 549 GB/s ceiling, and the fused kernels already hit 92% of it.
3. The earlier "~100 TFLOPS sorted gather" figure over-counted FLOPs (it exceeds the measured 65.7 bf16 ceiling); the sorted path is still ~45x faster than the unsorted fallback, which is why the 0.7.0.dev2 upgrade (mlx 0.32.2) matters.
4. Small-M 4-bit above M~4 is scalar-ALU bound (BN x M fp32 FMAs per weight), not bandwidth bound; 8-bit stays on the memory side and goes flat in M. The remaining lever is simdgroup_matrix.
5. JIT kernels on macOS 26.5 CAN use Metal tensor ops with int8 and int4 operands (`__HAVE_INT4B_FORMAT_TYPE__` = 1), with no Xcode and no MLX rebuild. Constraints: 4-bit operands only from memory tensors, K % 32 == 0, no mixed signedness.

## Revised whole-model estimate for Flash-Next on this machine
- Single-stream decode: +5% from fusion; the rest of the gap to the 133 tok/s roofline is weight bandwidth, not kernels. Further gains need fewer bytes per token (lower-bit experts, MTP acceptance) rather than faster kernels.
- 2-8 concurrent sessions / MTP verify: 1.5-2.4x on the vocabulary head, modest elsewhere; whole-model gain depends on the share of lm_head time (largest at short contexts).
- Prefill: 1.1-1.2x now on dense projections; 1.15x whole-model with the MoE gather converted; ~1.3x with macOS 27.

## Recommended next steps (in order)
1. In-situ trial of workstream 1 on a staging server: second `omlx serve` on port 8084 with a sitecustomize that imports smallm/kernel.py and calls apply() (OMLX_SMALLM_QMM=1), run bench.py against it with 4 concurrent sessions and MTP on; adopt if tok/s improves with identical outputs.
2. Convert the sorted MoE gather to the int8 x int4 tensor path (workstream 3's missing piece; needs a 40-rows/expert tiling). This is the largest remaining prefill win on macOS 26.
3. Enable decode fusion only if the staging trial shows the 3-6%; it is correct but small.
4. On macOS 27 (after MLX/oMLX declare support): cooperative tensors direct to matmul for the prefill kernel; re-measure.
5. Upstream candidates to ml-explore/mlx: the small-M kernel (issue #4265 territory) and the int8-activation prefill path.

Per-workstream details, tables and failure notes: smallm/REPORT.md, decode-fusion/REPORT.md, prefill-int4/REPORT.md. Shared conventions: COMMON.md. Baseline microbench: ../kbench.py. Analysis notes: ../kernel-notes.md.
