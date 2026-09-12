# Apple Silicon inference research, 2026-09-12

Scope: what changed in the ecosystem since roughly 2026-08-01, what hardware on this box is idle, and
what to try next. Baseline is the deployed configuration in `kernels/REPORT.md`: M5 Max 40-core GPU,
128 GB, macOS 26.5, oMLX 0.7.0.dev2 / mlx 0.32.2 / mlx-vlm 0.6.3, cold 65k prefill 1544 tok/s, decode
62 tok/s at MTP depth 3 (41 without). Vendor-published means the number comes from the shipping
party's own release notes and nobody has reproduced it.

## 1. Six weeks of releases

| Thing | Version / date | Status | Published number | URL |
|---|---|---|---|---|
| mlx | 0.32.2, 2026-08-25 | newest release, no 0.33.x | - | https://github.com/ml-explore/mlx/releases.atom |
| mlx `[Metal] global scale for qmm` | main, 2026-09-08 | merged, unreleased | - | https://github.com/ml-explore/mlx/commits/main |
| mlx PR #4481 gather_qqmm matrix kernels | opened 2026-09-08 | **open**, design contested | M5 Max prefill 1257 -> 3105 tok/s, decode flat | https://github.com/ml-explore/mlx/pull/4481 |
| mlx PR #4483 global scales in qmm_t | open | open | +8% alone, 4256 tok/s combined with #4481 | https://github.com/ml-explore/mlx/pull/4483 |
| mlx PR #4020 GDN Metal kernels | 2026-08-05, active 2026-09-10 | **open** | M5 Max 1.86-2.16x over sequential at T<=2048, covers (16,48) heads | https://github.com/ml-explore/mlx/pull/4020 |
| mlx PR #4487 SDPA D512 memory | open | open | 65k prompt 13.80 -> 9.58 GB | https://github.com/ml-explore/mlx/pull/4487 |
| mlx-lm | 0.31.3, 2026-04-22 | no release in window | - | https://github.com/ml-explore/mlx-lm/releases.atom |
| mlx-vlm | 0.7.0, 2026-09-07 | shipped, we are on 0.6.3 | Qwen4 n-gram shard gathers, APC redesign for hybrid caches, external PLE storage (PR #2045, merged 2026-08-28) | https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.7.0 |
| oMLX | 0.7.0.dev2, 2026-09-11 | current | INT8-activation prefill on M5, +34.4% at 32K (vendor) | https://github.com/jundot/omlx/releases.atom |
| llama.cpp | v0.4.0, 2026-09-04 | shipped | Metal sparse flash attention, initial qwen4exp | https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0 |
| llama.cpp PR #28136 direct PLE reads | 2026-09-01 | open | 2.0-3.1x cold prefill on non-Apple hardware, 4-6% warm regression, **no Apple measurements** | https://github.com/ggml-org/llama.cpp/pull/28136 |
| MTPLX | 2.11.2, 2026-09-06 | shipped | Flash-Next 68.4 tok/s at 16k on M5 Max 128 GB (vendor) | https://mtplx.com/releases/ |
| mlx-serve | v26.9.2, 2026-09-09 | shipped | M4 Max Flash-Next 83 -> 93 tok/s (vendor) | https://github.com/ddalcu/mlx-serve/releases.atom |
| vllm-metal | v0.29.0, 2026-09-11 | shipped, pins mlx 0.32.1 | paged attention now unconditional | https://github.com/vllm-project/vllm-metal/releases.atom |
| macOS 27 "Golden Gate" | GA 2026-09-14 | two days away | Metal tensors gain fp4/fp8/int2 and E8M0 block scales in 27; int4/int8 already in 26 | https://developer.apple.com/videos/play/wwdc2026/330/ |

Apple published nothing on inference performance in the window. The canonical M5 post is 2025-11-19
and still says prefill 3.5-4.1x over M4 while generation gains only 1.19-1.27x
(https://machinelearning.apple.com/research/exploring-llms-mlx-m5). No change found to
`iogpu.wired_limit_mb` behaviour, GPU scheduling, or third-party Neural Engine access in 26.6, 26.6.2
or the 27 announcements. Nothing in any runtime consumes the macOS 27 fp4/fp8 formats yet, so
upgrading on day one buys nothing and risks the oMLX patch stack.

BaseRT is real (arXiv 2607.00501, July 2026) but had no activity in the window.

## 2. Idle hardware on this box

**Neural Engine.** oMLX's `qwen35_ane_prefill.py` drives a private ANE runtime, not CoreML: it splits
dense Qwen MLP gate/up and the GDN `z` projection across ANE, CPU (Accelerate fp16) and GPU, with
per-output-channel INT8 requantisation on the ANE slice. It deliberately keeps QKV on the GPU because
the requantisation error would accumulate through the recurrent state
(`_recurrent_safe_gdn_ane_outputs`). It does not apply here. `omlx/model_settings.py:55`
`ane_prefill_backend()` returns a backend only for model types starting `qwen3_5`, `qwen3_6`,
`qwen3_8` or `k2_horizon`; our config reports `model_type: qwen4_exp`, so the path is unreachable, and
even if the gate were widened the eligible units are dense MLPs, which Flash-Next does not have
(`moe_intermediate_size` and `shared_expert_intermediate_size` are both 640). External evidence agrees
this is not the lever: mlx-serve shipped ANE prefill offload in v26.8.10 (2026-08-26) at +19-35% on 16k
prompts and **disabled it on M5-class Macs** because the GPU already wins
(https://github.com/ddalcu/mlx-serve/releases/tag/v26.8.10). ANE and GPU share the same unified
bandwidth, so there is no separate pipe to exploit. Verdict: skip.

**CPU with SME.** SMEPilot (arXiv:2606.16332, 2026-06-15) measures 4.10-4.44 TFLOP/s sustained on an
M4 Pro CPU for prefill FFN GEMM and 0.72-0.96x of that machine's GPU. Scaled against this box's
measured 65.7 bf16 TFLOPS, the CPU is worth under 10% of the GPU, and it is already busy with the
n-gram lookup. The one credible use is a concurrent lane rather than a share of the same matmul: a
small dense drafter or n-gram/suffix-tree search on CPU while the GPU runs verify. Nobody has published
that on Apple Silicon.

**Media engine.** No evidence of any general-compute use for the ProRes/HEVC blocks. Dead end.

**SSD.** The Verge measured 13.6 GB/s sequential read on a 4 TB M5 Max
(https://macdailynews.com/2026/03/10/apples-new-m5-max-macbook-pro-delivers-a-huge-ssd-read-write-speed-boost/,
2026-03-10), roughly double the 7 GB/s assumed in our notes. KVSwap (arXiv:2511.11907) finds 84% of
SSD KV read cost is kernel filesystem overhead, not device bandwidth, which matches the packed-table
result exactly. oMLX already ships `moe_expert_offload.py` for streaming non-resident experts, but it
refuses to combine with MTP (`model_settings.py`), so it is for models that do not fit, not for us.

**The 78 GB resident / 128 GB gap and concurrency.** This is the largest genuinely unexploited
resource. Published concurrency scaling on MoE models: oMLX 8x concurrency reaches 237.7 tok/s
aggregate against 65-80 single-stream on Qwen3.6 35B-A3B, measured on oMLX 0.3.8
(https://jacar.es/en/how-to-install-and-tune-omlx-on-m5-max-128-gb/), and mlx-serve reports 2.8x
aggregate at 4 streams (v26.8.10). An independent five-backend comparison on a 9B GDN model measured
oMLX at only 1.29-1.40x from concurrency 1 to 4 (https://jaesolshin.com/posts/apple-silicon-llm-backends/,
2026-05-20). All three are lower than the audit's decode arithmetic implies is available: the expert
gather runs at 300 GB/s at one row and 549 at eight, so rows are the currency.

## 3. Speculation

Our 1.51x is at the low end of what is published for this model class.

| Source | Hardware | Result | Type |
|---|---|---|---|
| MTPLX Flash-Next card | M5 Max | 43.8 -> 73.5 tok/s, 1.7x, adaptive depth ceiling 3 | vendor, https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed |
| MTPLX 2.11 | M5 Max 128 GB | 68.4 tok/s at 16k, 44.2 at 206k | vendor |
| MTPLX quality pack | M5 Max, Qwen3.8-27B | **3.0x with 8-bit MTP heads vs 2.3x with 4-bit** | vendor |
| oMLX 0.6.3 | M3 Ultra | 2.33-2.62x, 96.8-97.9% acceptance | vendor |
| llama.cpp #27836 community | - | 44-68% acceptance, ~78% with draft-n-max 2 | community |
| vLLM recipe | 4x H100 | MTP made throughput worse, ~36% acceptance | third party |
| ddtree-mlx | M3 Ultra, Qwen3.5-27B | chain 1.38x -> tree 1.52x, so trees add 10-15% | independent |
| mlx-lm EAGLE-3 prototype | M3 Ultra, Llama-3.1-8B | 1.05x, 34% acceptance | prototype, https://github.com/ml-explore/mlx-lm/discussions/890 |

Nothing exists for HASS, Medusa, SpecTr or Falcon on MLX, and there is no EAGLE-3 checkpoint for any
Qwen MoE on MLX. Draft trees are architecturally blocked in mlx-lm core because `KVCache` carries one
offset integer (issues #846, #250, open since February) and are worth only 10-15% over a chain where
they do work, so they are the wrong target.

Two levers stand out. First, the draft head precision: I checked the checkpoint, and
`mtp.fc_embedding` and `mtp.fc_hidden` are 4-bit group-64 (weight `U32[2560,320]` against
`scales[2560,40]`), while the shared expert inside the MTP block is 8-bit group-128. MTPLX's own A/B on
the same target family is 2.3x at 4-bit against 3.0x at 8-bit. The whole `mtp.*` block is 1.48 GB, so
requantising its 4-bit parts costs about 1 GB. Second, depth policy: llama.cpp's confidence-gated
`--spec-draft-p-min` (discussion #25198, July 2026) reports +15-20% global throughput at a high depth
ceiling with a probability floor, which is a policy change rather than a kernel. MTPLX 2.11.2
separately found its depth policy was assuming eager short drafts beat compiled depth 3 and gained
+28-37% on M5 Max by measuring instead of assuming. Our round-2 depth-4 sweep regressed, which is
consistent with a fixed-depth policy rather than evidence that depth is maxed out.

Third, the n-gram lane. MTPLX's context-copy proposes up to 24 tokens as one block when the stream tail
matches a prompt n-gram, verified with the same probability ratio so it is distribution-preserving:
73.8 -> 87.6 tok/s on a two-turn rewrite, and community reports of +46-54% on temperature-0 edits. Two
of their bug fixes transfer directly to any implementation here: copy blocks accepted past a stop token
poisoned the GDN recurrent session cache, and copy must be prompt-sliced rather than reading generated
output. SuffixDecoding (arXiv:2411.04975) is the research version and reports up to 5.3x on SWE-Bench,
with no MLX port.

## 4. Prefill

Our 1544 tok/s at 65k remains ahead of every published Flash-Next figure. Three things could beat the
current chunked-2048 setup.

The first is a warning rather than an idea. mlx-lm issue #980 (filed 2026-03-11, still open) reports
that prefix cache reuse is broken for every hybrid model, with Qwen3.5 attention+Mamba getting
*slower* across three sequential requests (5.02 -> 7.76 -> 8.00 s) where a pure-attention MoE gets
10.5x faster. Our own agentic suite measured a 49k cached follow-up at 2.2 s, so oMLX's paged SSD cache
is working here, but the failure mode is silent and worth a periodic assertion in the bench.

The second is CacheBlend (ACM TOCS 2026, arXiv:2405.16444): reuse KV for non-prefix chunks by
selectively recomputing only cross-attention, 2.2-3.3x lower TTFT, one reported two-turn eval at 98%
hit rate against 48% for plain prefix caching. That is the exact shape of agentic coding, where a tool
result lands mid-context and invalidates everything after it. No MLX or Metal implementation exists.

The third is KV quantisation. arXiv:2605.05699 reports a fused Metal int4 KV kernel that is *faster*
than fp16 across 256-4096 token prefixes on Apple Silicon because it moves a third of the bytes. oMLX
ships `turboquant_attention.py` with a decode kernel and a fused 2-pass MTP-verify kernel for
1 < L <= 15, which is exactly our verify width, but it patches `scaled_dot_product_attention` and our
12 QSA layers route through `qsa_fast.py`, so whether it engages is unknown and cheap to check.

On long-context attention, our measured 20% cost from 16k to 130k is already good. MTPLX's block-sparse
Metal prefill reports 98k prompt 175.7 -> 114.5 s on M5 Max 128 GB with peak memory 91.4 -> 83.0 GB
(vendor). SpecPrefill exists locally (`omlx/patches/specprefill.py`, arXiv:2502.02789) and drops
tokens, which is wrong for coding.

## 5. Ranked next experiments

Measured by someone on M5-class hardware:

| # | Experiment | Expected | Effort | Evidence |
|---|---|---|---|---|
| 1 | Requantise `mtp.fc_embedding` / `mtp.fc_hidden` from 4-bit gs64 to 8-bit, A/B decode | +10-25% decode | low, ~1 GB | MTPLX 2.3x vs 3.0x A/B, vendor; local checkpoint confirms 4-bit |
| 2 | Confidence-gated adaptive depth (p_min floor, ceiling 4-5) instead of fixed 3 | +10-20% decode | low, policy only | llama.cpp #25198 +15-20%; MTPLX 2.11.2 +28-37% on M5 Max |
| 3 | Enable oMLX INT8-activation prefill, verify it engages on 512-expert top-10 shapes | +10-34% prefill | low, config flag | oMLX 0.7.0.dev2 2026-09-11, vendor, M5 Max |
| 4 | Prompt-sliced n-gram copy lane beside MTP, stop-token safe, GDN-cache safe | +15-50% decode on edit-heavy turns | medium | MTPLX 73.8 -> 87.6 tok/s, vendor |
| 5 | Build mlx main with PR #4020 GDN kernels, measure the 123 ms scan | +5-10% prefill | medium, unmerged PR | 1.86-2.16x on M5 Max at our (16,48) head shape |
| 6 | Confirm the MTP verify pass uses the sparse QSA arm, not dense SDPA | up to +20% decode at long context | low to measure | MTPLX paired 16k test, 36.0 vs 64.5 tok/s |
| 7 | Concurrency sweep at 2/4/8 streams, aggregate and per-stream | 1.3-3x aggregate | low to measure | oMLX 237.7 tok/s at 8x (0.3.8); contradicted by 1.29-1.40x at 4x elsewhere |
| 8 | mlx-vlm 0.6.3 -> 0.7.0, diffing the n-gram shard gather against our packed patch | single digits | medium, pin conflict | shipped 2026-09-07 |

Theory, or measured only off Apple Silicon:

| # | Experiment | Expected | Effort | Evidence |
|---|---|---|---|---|
| 9 | CacheBlend-style selective recompute for mid-context tool output | 2-3x TTFT on invalidated turns | high, no Metal implementation | arXiv:2405.16444, GPU only |
| 10 | Route TurboQuant KV through the QSA path, or port the int4 KV Metal kernel | verify bandwidth, long-context reach | medium | arXiv:2605.05699 on Gemma-3 1B, not MoE |
| 11 | CPU/SME lane for suffix-tree drafting concurrent with GPU verify | unknown | high | SMEPilot 4.1-4.4 TFLOP/s on M4 Pro; nobody has tried the split |
| 12 | Re-measure real SSD read bandwidth and re-tune the packed reader worker count | +2-5% cold | low | 13.6 GB/s measured on M5 Max 4 TB; llama.cpp #28136 found 32 workers 6.6x over serialised faults |
| 13 | mlx PR #4481/#4483 gather_qqmm | up to 2.5x prefill | wait | open, contested, author-benchmarked only |
| 14 | macOS 27 fp4/fp8/MXFP4 tensor formats | unknown | blocked | GA 2026-09-14, no runtime consumes them |

Do not pursue: ANE offload (unreachable for `qwen4_exp`, and mlx-serve disables it on M5), draft trees
(10-15% for unmerged core changes), EAGLE-3 or Medusa on MLX (no ports, the one prototype gets 1.05x),
the media engine, and SpecPrefill (lossy).
