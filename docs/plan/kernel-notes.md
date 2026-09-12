# MLX kernel headroom on M5 Max for Qwen3.8-Flash-Next (notes, 2026-09-11)

Anchors: ~500 GB/s achievable (82% of 614), NAX fp16 peak ~61 TFLOPS (4x the 15.4 measured on base M5), best kernel in tree hits 48 TF.
Model: 512 experts, top-10, expert mats 2560x640, 12 full-attn + 36 GDN layers, head_dim 256, MTP width 2.
Weight traffic per token at 4.6 bpw ~3.45 GB (+0.3 GB GDN state and KV) -> decode roofline ~133 tok/s; observed 35-68.

## Where MLX is today (mlx @ dfe17ba)
- NAX path dequantises 4-bit weights to bf16 in threadgroup memory (quantized_nax.h:1230, :638) before matmul2d. No MTLTensorDataType anywhere. macOS 26.2 int4/int8 tensor operands are unused.
- Decode never reaches NAX: qmv_* below get_qmv_batch_limit (13-33), qmm_splitk in between; NAX only when n_tiles*m_tiles >= 512 (~M>=97 at N=4096).
- MoE decode: GatherQMM NAX gate is M==1 && B>=16 && B/E>=4. With E=512, top_k=10, B=10 (decode) or 20 (MTP verify) -> always gather_qmv legacy SIMD.
- Prefill gather NAX: BM=32 when rows/expert<64. 2048-token chunk -> 40 rows/expert -> ~42-53% of ceiling (#3925, #4023). 512-token chunk -> ~15%.
- volatile compiler_barrier in every NAX inner loop (quantized_nax.h:1028/1045, fp_quantized_nax.h x4); STEEL_PRAGMA_NO_UNROLL on kk1; gather NAX only at BK=64; no fp_qmm_n_nax.
- Attention D256 fused NAX exists (#3842, #4416); decode vector path (#4477 open, +11-18%).
- GDN kernel: packed layout 1.9-2.0x at T=2048-8192 (#4020 review). fp8 KV: +8-12% decode (#3789).
- MSL 4.1 on macOS 27 only via JIT mx.fast.metal_kernel (#4052 merged); shipped metallib is Metal 4.0.

## Estimate (low / likely / high)
| Regime | Today util | 26.5 kernels only | + macOS 27 features |
|---|---|---|---|
| Decode M=1 | 25-30% of BW roofline | 1.4 / 1.9 / 2.6x (occupancy, gather_qmv for top_k<<E, GDN packed, fp8 KV, D256 decode attn) | +1.0 / 1.05 / 1.15x |
| MTP verify M=2 | ~55% of M=1 cost/row | 1.15 / 1.35 / 1.6x (simdgroup MMA small-M) | +1.1 / 1.25 / 1.4x (coop tensors, no threadgroup staging) |
| Batched decode M=4-8 | 16-30% | 1.3 / 1.55 / 1.8x | +1.2 / 1.4 / 1.7x |
| Prefill 2048 chunks | ~17-21% of NAX band | 1.6 / 2.3 / 3.2x (use 26.2 int4/int8 operands, BM tuning, drop barrier, threadgroup scope, bigger chunks) | 1.5 / 2.2 / 3.0x (native fp4 + E8M0 scale plane, coop tensors) partly overlapping |
| Attention prefill D256 | good | ~1.1x | 1.05 / 1.15 / 1.3x |
Key: the biggest prefill win (native int4/int8 tensor operands) shipped in macOS 26.2 and is unused by MLX. macOS 27 mainly removes threadgroup staging and adds fp4/int2/fp8 + scale planes. Whether M5 NAX has a real 2x fp4 datapath is unverified (int8 is ~1.8x fp16 on A19; fp8 on M4 was emulated at 0.94x).
Sources: mlx issues/PRs #4265 #3925 #4023 #4352 #4171 #3842 #4416 #4476 #4477 #4020 #3789 #4052 #2693; WWDC26 session 330; machinelearning.apple.com/research/exploring-llms-mlx-m5; tzakharko A19 NAX bench; michaelstinkerings M5 roofline.
