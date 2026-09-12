# Optimisation sources: what the ecosystem knows that Titan does not

2026-09-12. A survey of vLLM, SGLang, llama.cpp, mlx and mlx-lm, TensorRT-LLM,
MLC-LLM, the Apple-Silicon runtimes (BaseRT, MTPLX, mlx-serve) and the 2025 to
2026 speculative-decoding literature, read against Titan's own measured budget.
Every URL cited was fetched on the date above and returned 200. Where a claim
came from a search summary rather than the page itself, it says so.

Numbers to beat, from `docs/plan/IMPROVEMENTS.md` sections 11 and 12:

| measure | production today |
|---|---|
| short-context decode, thinking off | 84 to 91 tok/s |
| decode after a 64k prefill | 71 tok/s |
| aggregate at 1 / 2 / 4 / 8 streams | 60 / 73 / 97 / 130 tok/s |
| cold 65k prefill | 1600 tok/s |

Budget per decode cycle, from `engine/patches/round4/decode-profile/REPORT.md`
and IMPROVEMENTS section 12: cycle 26.9 ms, verify forward 75% (20.2 ms) at 45%
of the 718 GB/s measured read ceiling with 48 `mx.async_eval` calls, acceptance
host sync 13% (3.5 ms), draft chain 11% (3.0 ms). Three accepted tokens per
cycle is 111 tok/s. At 64k the plain step is 17.7 ms, the speculative cycle is
34.1 ms, the marginal verify row costs 9.3 ms, and the median accepted count
falls to 1.

---

## 1. Four findings that change the shape of the problem

Read these before the candidate list. Each of them moves obvious ideas down the
ranking and unobvious ones up.

### 1.1 Forty-eight `async_eval` calls per forward is not normal, and the 45% figure is its signature

`language.py:2774` calls `mx.async_eval(hidden_states)` after every decoder
layer whenever `_EAGER_DISPATCH` is on and the row count is at most 64, which is
every decode and every verify cycle. mlx-lm's own reference loop does one
`async_eval` per token, not one per layer, and it keeps the sampler in graph and
enqueues step n+1 before reading step n
(`https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/generate.py`).
MLX's lazy-evaluation guidance puts the good regime at tens to thousands of ops
per eval (https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
and `mx.async_eval` is still documented as experimental with no written ordering
semantics
(https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html).

Two independent measurements say what the consequence is.

MLX discussion #3939 (https://github.com/ml-explore/mlx/discussions/3939, opened
2026-07-28) profiles a 2.78T MoE across four M3 Ultras and states it plainly:
"Decode at batch 1 is dispatch-bound, not bandwidth-bound." They measure roughly
500 ms per token against a 27.4 ms bandwidth roofline. Their own caveat is the
one that matters here: "those per-module numbers were taken with `mx.eval`
around each module, a sync the real model does not pay per-module, so ~170 ms is
an upper bound." Titan's profiler wrapped `mx.eval`, `mx.async_eval`,
`mx.synchronize`, `tolist` and `item`, and its `REPORT.md` section 4 already
warns that "wall mode measures dispatch" and that the per-layer `async_eval`
means "a layer bucket absorbs whatever queue backpressure it happens to meet".

llama.cpp issue #23752 (https://github.com/ggml-org/llama.cpp/issues/23752,
opened 2026-05-27, open) isolates the same overhead on Metal: on an M1 Max with
Qwen3.5-9B-MTP at temperature 0, speculative decoding at n_max=0 loses 11% of
throughput **despite 100% acceptance**. That is pure per-round dispatch cost
with the acceptance variable held at its maximum.

So the 45%-of-ceiling number and the 9.3 ms marginal verify row are both
consistent with dispatch cost rather than bandwidth cost, and the profile that
produced them was taken with the syncs in place. Nothing in the ranking below
should be trusted until the dispatch experiments in section 5 have run.

There is a free knob here. MLX PR #1864
(https://github.com/ml-explore/mlx/pull/1864, merged 2025-02-14) added
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` with per-architecture
defaults, and the Max and Ultra default is 50 ops and 50 MB per command buffer.
Measured: M2 Ultra Qwen 0.5B 292.6 to 368.2 tok/s (+26%), Mistral-7B 124.3 to
131.2 (+5.6%), M4 Max Qwen 0.5B 479.0 to 532.7 (+11%). A 48-layer forward that
breaks the buffer at every layer never reaches the 50-op budget it is allowed.

One counter-argument to hold onto. mlx-lm issue #1332
(https://github.com/ml-explore/mlx-lm/issues/1332, closed) records DeepSeek-V4
crashing after about 11,300 tokens with
`[metal::malloc] Resource limit (499000) exceeded`, because the limit counts
buffers rather than bytes and detached per-step graphs retain every
intermediate. Explicitly evaluating cache state each step cut growth from
205 KB per step to 7 KB. Titan's 36 recurrent states are exactly that shape, and
oMLX's scheduler carries the same mitigation (`_eval_generation_batch_cache`
every 256 decode tokens, "Metal's 499,000-buffer limit", per
`OMLX-COMPONENT-MAP.md` section 4). So the per-layer evals may be load bearing
for a reason nobody wrote down. Removing them requires watching buffer count
over 10k tokens, not just watching tok/s over 600.

### 1.2 A verify row on this model does not amortise weights

The profiler's own byte model reads 2.34 GB of dense weights, plus
`512 * (1 - (1 - 10/512)^M) * 48 * 2.765 MB` of expert weights at M rows, plus
0.63 GB per `lm_head` pass. Expected distinct experts per layer is 10.0 at M=1,
19.8 at M=2, 29.4 at M=3, 38.8 at M=4 and 76 at M=8. That is close to linear all
the way to 8, because 10 draws from 512 almost never collide.

So the second verify row costs roughly what the first did. This is the opposite
of the dense-model intuition that speculation is nearly free because the weights
are already moving. Checking the model against the measurement: at M=4 it
predicts 2.34 + 5.15 + 2.52 = 10.0 GB, which is 13.9 ms at the roofline and
31 ms at the measured 45% efficiency, against a measured 34.1 ms cycle at 64k.

Three consequences.

Trees are the wrong target here. vLLM closed its tree-drafting request as not
planned (https://github.com/vllm-project/vllm/issues/18327). SGLang measures
accepted length going from 3.413 with a chain to 4.231 with topk=4 on
Qwen3-Next-80B at batch 1
(https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/,
2025-12-03), which is 24% more acceptance for roughly 2.7x the rows. On this
expert-gather curve that loses.

The lever is accepted tokens per row, not rows per cycle.

And the depth policy has to price the marginal row. Expected tokens per
millisecond at depth d is `(1 + E[accepted_d]) / (17.7 + 9.3 * d)` at 64k. With
the measured median accepted of 1, depth 1 beats depth 2, and Titan's
`DepthController` rule `round(mean_accepted) + 1` picks 2.

### 1.3 Our long-context acceptance is far below what this model is documented to give

Qwen's technical report (https://arxiv.org/abs/2608.30320, submitted
2026-08-31) section 2.1.2, Table 4: "Mean MTP accepted length with full
attention and QSA under four-step speculative decoding", averaging **4.06 with
full attention and 4.07 with QSA**, range 3.47 to 4.30. SGLang's day-0 post
measures 540 tok/s at batch 1 with MTP on NVFP4 TP4 B200 at an **accept length
of 3.3 including the bonus token**
(https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/, 2026-08-26).
SemiAnalysis lists 3.24 at MTP=3 with thinking off
(https://inferencex.semianalysis.com/model/qwen-3-8-flash-next). vLLM's own
recipe is the outlier the other way, reporting roughly 36% acceptance and MTP
making throughput worse at every concurrency on 4x H100
(https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).

Titan sits at 80% acceptance short-context and a median of 1 accepted at 64k.
Nobody publishes an acceptance-versus-context curve for this model, so 1 at 64k
is not directly contradicted, but the gap against 4.07 at four steps is large
enough to be a defect rather than a property.

The literature says what the defect usually is. Test-Time Speculation
(https://arxiv.org/abs/2605.09329, 2026-05-10) measures DFlash falling from 3.7
accepted over the first 10k tokens to 1.5 over the last 10k and EAGLE-3
collapsing to about 1.1 past 20k, and attributes it to speculators trained on
short sequences being run far outside that distribution.

There is also a cheap alternative explanation worth eliminating first.
llama.cpp issue #23658 (https://github.com/ggml-org/llama.cpp/issues/23658,
open, 2026-05) records MTP acceptance dropping to 16% at context 12,032 and
jumping to 71% at 12,288, a repeating pattern on roughly 2048-token boundaries,
attributed to how the draft context shares KV slots with the target. That is
CUDA data, but Titan's block grid is 512 and its snapshot grid is 2048, and the
64k measurement was taken at one context length. Sweep before concluding.

### 1.4 The MTP head is a QSA layer, and the verify block takes the prompt-shaped attention path

`Qwen4ExpMTPModule` (`language.py:2800`) builds its single layer with
`layer_types=["qwen_sparse_attention"]`. So each draft step at 64k pools index
keys, scores every completed micro-block and takes a top-512 over its own KV.

Worse, `qsa_fast.py` has two paths. The decode path takes one row and one shared
block bank. The `contiguous_causal` gathered path raises unless
`queries.shape[2] > 1`, then loops over query chunks computing block scores and
a fresh `argpartition` top-512 **per query row per layer**. A verify block of
width k+1 is not a prompt, but it takes the prompt path, so every one of the k+1
rows pays a full indexer pass in all 12 QSA layers.

This is exactly the shape of a bug mlx-lm just fixed elsewhere. PR #1817
(https://github.com/ml-explore/mlx-lm/pull/1817, open, 2026-09-02) found the
absorbed MLA attention path gated to L=1, so a multi-token verify fell back to
materialising full K and V across the whole cache. Removing the gate took the
verify step on GLM-4.7-Flash with an 8K cache from **363.7 ms to 3.3 ms**, and
speculative decoding from 2.8 to 14.7 tok/s, a 5.2x.

Both Qwen and Zhipu have already published the fix for the QSA version. The
Qwen report says they "follow GLM and reuse the top-k indices across speculative
decoding steps to further improve draft-model efficiency", with "no significant
change in the mean accepted length after QSA reuse". SGLang's implementation is
blunter: "The draft decode steps stop running the indexer altogether ... each
request's last accepted row is captured there and reused by the whole draft
loop", and separately they cut "QSA's index-cache overhead by 80%" by not
retaining raw index keys for the full context. Titan retains them
(`Qwen4ExpQSAKVCache`, `language.py:452`, documented as "KV cache with the raw
indexer keys") and recomputes selection per row.

For scale, llama.cpp merged a sparse flash-attention path on Metal on 2026-09-03
explicitly targeting Qwen3.8-Flash-Next query-sparse attention
(https://github.com/ggml-org/llama.cpp/pull/28098) and measured a DeepSeek-V4
context of 65,536 going from 107.08 to 323.54 t/s, roughly 3x, with no change at
2048. That is the size of the prize sitting in the long-context attention path.

---

## 2. How the gains below are estimated

Holding accepted tokens fixed, saving `s` ms of a 26.9 ms cycle multiplies
short-context decode by `26.9 / (26.9 - s)`. Removing the whole host sync is
1.15x. Taking 20% off the verify forward is 1.18x. Halving the draft chain is
1.06x.

At 64k, tokens per cycle is `1 + accepted` and the cycle is `17.7 + 9.3 * depth`
ms. Today that is roughly 2.4 tokens per 34.1 ms, which is the measured 71
tok/s. Raising accepted from 1 to 3 at the same width gives 4 / 34.1 = 117
tok/s. Cutting the marginal row from 9.3 ms to 4 ms at depth 3 gives 2 / 29.7 =
67 tok/s at accepted 1, and 4 / 29.7 = 135 at accepted 3.

Every estimate below is a range and every one of them competes with the thermal
drift IMPROVEMENTS section 10 measured at 10 to 16%. Nothing here should be
believed without a paired A/B with a cooldown.

---

## 3. Candidates

Each entry: what it is, primary source with date and status, mechanism against a
named stage, estimated gain with the reasoning, the Titan seam, risks, and
whether it collides with something already rejected.

### D. Dispatch and the verify forward (75% of the cycle)

#### D1. Stop breaking the command buffer at every layer

Turn off `_EAGER_DISPATCH` and let one `async_eval` cover the whole forward, as
mlx-lm's reference loop does.

- Sources: mlx-lm `generate.py` on main (one `async_eval` per token, sampler in
  graph, next step enqueued before the current one is read); MLX discussion
  #3939 (https://github.com/ml-explore/mlx/discussions/3939, 2026-07-28),
  "Decode at batch 1 is dispatch-bound, not bandwidth-bound", ~500 ms per token
  against a 27.4 ms roofline, and the explicit caveat that per-module `mx.eval`
  inflates attribution; llama.cpp issue #23752
  (https://github.com/ggml-org/llama.cpp/issues/23752, open, 2026-05-27), -11%
  from speculative decoding on Metal at 100% acceptance, which is pure dispatch.
- Stage: verify forward and draft chain.
- Estimate: the gap between 45% and 60% of the read ceiling is 1.33x on the
  verify stage, which at 75% of the cycle is 1.25x overall. Reaching Hazy
  Research's 78% at batch 1
  (https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) would be 1.7x
  on the stage. Estimate -5% to +20%, wide because the flag exists for a reason.
- Titan seam: `TITAN_QWEN4_EAGER_DISPATCH=0`, one environment variable that the
  vendored model already reads at `language.py:998`, and which the comment at
  `:997` says is "Scheduling only: outputs are bit-identical".
- Cost: half a day to measure.
- Risks: mlx-lm issue #1332
  (https://github.com/ml-explore/mlx-lm/issues/1332, closed) is the reason to be
  careful. Metal's resource limit counts buffers, not bytes, and detached
  per-step graphs over 36 recurrent states can reach 499,000 buffers. oMLX
  mitigates with `_eval_generation_batch_cache` every 256 decode tokens. Any run
  of this experiment has to go past 10k generated tokens, not 600.
- Collisions: none.

#### D2. Raise the command-buffer op and byte budgets

- Source: https://github.com/ml-explore/mlx/pull/1864, merged 2025-02-14.
  Per-architecture defaults, Max and Ultra at 50 ops and 50 MB. Measured M2
  Ultra Qwen 0.5B 292.6 to 368.2 tok/s (+26%), Mistral-7B +5.6%, M4 Max Qwen
  0.5B +11%, M2 Air +31%.
- Stage: verify forward and draft chain.
- Estimate: the measured range on small dense models is +5 to +31%, and it
  shrinks with model size (Mistral-7B got 5.6%). On a 48-layer forward whose
  buffers are already being broken by D1's `async_eval`, this only pays after
  D1. Estimate +2 to 8% together with D1.
- Titan seam: environment variables set by `prod/`, and a note in the resolved
  config so `/metrics` records them. Titan's no-environment-variable rule
  (ARCHITECTURE section 8) applies to Titan's own configuration; these belong to
  MLX and should be set in the launcher and echoed.
- Cost: half a day, run together with D1.
- Risks: larger command buffers mean coarser cancellation granularity.
- Collisions: none.

#### D3. Route the verify block through the decode-shaped QSA path

Give the k+1 verify rows one shared block selection computed at the pending
token, and the direct-index attention kernel, instead of the prompt-shaped
gathered path that recomputes a top-512 per row per layer.

- Sources: Qwen technical report https://arxiv.org/abs/2608.30320 section 2.1.2,
  "reuse the top-k indices across speculative decoding steps", with "no
  significant change in the mean accepted length after QSA reuse"; SGLang's
  implementation note https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/,
  "The draft decode steps stop running the indexer altogether"; the closest
  measured analogue is mlx-lm PR #1817
  (https://github.com/ml-explore/mlx-lm/pull/1817, open, 2026-09-02), where
  removing an L=1 gate on the MLA absorbed path took a verify step from 363.7 ms
  to 3.3 ms and speculative decoding from 2.8 to 14.7 tok/s.
- Stage: verify forward at long context, and the marginal row directly.
- Estimate: at short context the indexer is nearly a no-op, because with 512
  blocks and compression ratio 4 the selection only binds past 2048 tokens. At
  64k there are 16,384 complete micro-blocks and the top-512 runs per row per
  layer. If this takes the marginal row from 9.3 ms to 5 ms, a depth-2 cycle
  goes from 34.1 to 27.7 ms and 64k decode goes 71 to 87. Estimate +10 to 25% at
  64k, near zero at short context. This is the largest well-sourced
  long-context number in the document.
- Titan seam:
  `titan/adapters/mlx/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py`, the
  `selected_block_rows` construction in the gathered path, plus a slot on
  `Qwen4ExpQSAKVCache` for the last accepted selection, plus the `PHASE_VERIFY`
  branch in `titan/adapters/mlx/state.py`. Register it as a kernel op so
  `kernels.disabled` can bisect it.
- Cost: 3 to 4 days, most of it in the exactness work.
- Risks: this is not bit-identical. The model definition computes a selection
  per query, and reusing it is an approximation. The only evidence it is safe is
  Qwen's own sentence about accepted length, and accepted length is not output
  quality. It needs the coding probes and its own tolerance class.
- Collisions: none. IMPROVEMENTS records "Confirmed already right (no change
  needed): MTP verify uses the sparse attention arm", which is true and
  different: the arm is right, the selection inside it is recomputed per row.

#### D4. Land the fused `verify_accept` metal kernel

`titan/kernels/verify_accept.py` exists with a reference implementation and no
fast path, and the registry entry has never been made (`ENGINE.md` section 9).
Fusing argmax, compare, cumprod, sum and gather removes four or five small
dispatches immediately before the sync.

- Source: the closest published analogue is vLLM's Triton rejection sampler
  (https://github.com/vllm-project/vllm/pull/14930, merged 2025-03-18), which
  flattened the logits to `[num_tokens, vocab]`, removed concat, gather and all
  CPU-GPU syncs, and measured Llama 3.1 8B on one H100 going from 51.49 to 64.16
  req/s, +18% against main with speculative decoding on.
- Stage: acceptance host sync, the part that is not the wait.
- Estimate: +2 to 5%.
- Titan seam: the fast path in `titan/kernels/verify_accept.py` plus two lines in
  `titan/kernels/registry.py:_op_modules`. The exactness test already covers
  k=1 to 8, rejection at every position, row independence and the stochastic
  path against numpy.
- Cost: 2 days. The cheapest item here.
- Collisions: none.

#### D5. Compiled verify lane with bucketed row widths

`mx.compile` recompiles when input shapes change, and the docs say so
(https://ml-explore.github.io/mlx/build/html/usage/compile.html). Titan's
`DepthController` hands out a different depth most cycles, so the verify block
width changes most cycles, so a compiled verify would miss its cache almost
every time. Quantise the width to a small set and compile per bucket.

- Sources: MTPLX 2.7.0 (2026-08-15) "Compiled verify to 32k ... +6.9% at 20k";
  2.11 (2026-09-04) ships "a compiled verify lane" alongside M5 Max
  Flash-Next decode going 53.2 to 68.4 tok/s at 16k, 47.5 to 60.9 at 100k and
  32.2 to 44.2 at 206k (https://mtplx.com/releases/,
  https://mtplx.com/benchmarks/). 2.11.2 (2026-09-06) connects the two
  explicitly: "The expected-value depth policy measures draft and verify cost
  per depth on the running machine, so Flash-Next's compiled verify route
  carries 90 to 96 percent of cycles in agent turns instead of about 11"
  (https://releasebot.io/updates/mtplx). Kernel detail at
  https://mtplx.com/how-it-works/: verify shapes are "GraphBank compiled" with
  "unroll_count(4) for verify shapes M=3..6". The source is Apache-2.0 at
  https://github.com/youssofal/MTPLX.
- Stage: verify forward.
- Estimate: MTPLX's paired 2.10.2-to-2.11 numbers are +28 to +37% on the same
  model on the same machine class, with the compiled verify lane named as the
  cause. Not all of that is compilation and their base was lower than ours.
  Estimate +8 to 20%.
- Titan seam: `titan/adapters/mlx/model.py:verify` under `mx.compile`, width
  bucketing in `DepthController.plan_depth`, and the fixed-shape commit from H3.
  `ModelState.batch` already pads to a common width, so bucketing is a clamp on
  what the backend is asked for.
- Cost: 4 to 5 days. The bucketing is a day; the rest is finding what in the
  forward is not compilable. MLX's documented breakers are evaluating or
  printing arrays inside the function, side effects, and control flow branching
  on array values. A 512-expert router that branches on selection is the
  textbook case of the last one, and the n-gram gather and the QSA top-k are
  both suspect.
- Risks: SGLang documents piecewise CUDA graph as incompatible with speculative
  decoding for the same reason
  (https://lmsysorg.mintlify.app/docs/advanced_features/piecewise_cuda_graph).
  Their escape hatch is Breakable CUDA Graph, compiling the segments around the
  data-dependent parts rather than the whole cycle
  (https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/, 2026-08-17,
  measured on gpt-oss-120b at TP4: full capture 1.93x, breakable 1.70x,
  piecewise 1.45x against eager, with builds 3.8 to 5.2x faster).
- Collisions: none, and it pairs with A2. MTPLX's note says the depth policy
  exists to keep cycles on the compiled route, so the two should ship together.

#### D6. Fuse the hyper-connection Mix and Combine at the verify layout

- Source: https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/, 2026-08-26.
  "On NVIDIA B300 at M = 4, the fused path reduces Mix latency from 12.36 to
  6.03 us, a 2.05x kernel-level speedup"; "At M = 4, the split path reduces
  Combine latency from 4.17 to 2.13 us, a 1.96x kernel-level speedup"; "In an
  end-to-end speculative-decode benchmark against the previous Triton path,
  throughput improves by 7.6%".
- Stage: verify forward.
- Estimate: M=4 is Titan's verify width at depth 3 and the hyper connection is a
  4-wide residual stream, so the shape matches exactly. The 7.6% is end-to-end
  against a Triton baseline on a different backend. Estimate +3 to 8%.
- Titan seam: a new op beside `titan/kernels/hc_prefill.py`, registered in
  `registry.py`, called from `Qwen4ExpGatedResidual` (`language.py:1625`).
- Cost: 3 to 4 days, and bit-identical is achievable because it is a fusion of
  exact arithmetic.
- Collisions: partial and worth checking first. IMPROVEMENTS records "fused
  prefill hyper-connection block (bit-identical, +2%)" as deployed and "fused
  weighted sum on the MTP verify layout (bit-identical, neutral)". Those are two
  different ops and neither is Mix and Combine at decode layout. Read
  `docs/kernels/REPORT.md` before building; if the neutral result was this
  kernel at this shape, drop it.

#### D7. Small-M flash-decoding verify attention

A verify block is 2 to 8 query rows against a 64k cache, which is neither the
one-row decode case nor the whole-prompt case that generic SDPA is tuned for.

- Sources: MTPLX 2.11: "TensorOps flash-decoding kernel cuts verify attention
  per layer by 35%" on the 27B (https://mtplx.com/releases/); their kernel list
  names `sdpa_nax_flash` and a `verify_qmv` small-M qmv with "4-simdgroup" and
  "unroll_count(4) for verify shapes M=3..6" (https://mtplx.com/how-it-works/).
  MLX 0.32.2 already shipped "Add a fused full-attention path for head_dim 256
  on NAX devices" (https://github.com/ml-explore/mlx/releases/tag/v0.32.2,
  2026-08-25), and QSA's main attention head dim is 256, so part of this may be
  available already. Open follow-ups: mlx #4476 "Use NAX attention for short
  causal D256 prefill" and #4477 "Add D256 to GQA-8 two-pass vector attention".
  llama.cpp's merged Metal sparse flash attention
  (https://github.com/ggml-org/llama.cpp/pull/28098, merged 2026-09-03) measured
  a 65,536-token DeepSeek-V4 context going 107.08 to 323.54 t/s.
- Stage: verify forward at long context.
- Estimate: attention is 12 of 48 layers and the audit puts the 16k-to-130k
  attention cost at about 20%. Estimate +3 to 8% at 64k after D3, less before it,
  because D3 removes the selection cost that dominates today.
- Titan seam: `titan/kernels/qsa_gathered_attention.py`, which exists.
- Cost: 5 to 6 days for a kernel at the house exactness standard, or 1 day to
  check whether the 0.32.2 NAX D256 path is already engaging.
- Collisions: none. Round 4 already banked a batched sparse attention arm at 1
  ULP worth 20 tok/s per stream at two 68k streams against 12 dense.

#### D8. Re-measure the weight-stationary expert gather at verify width

Section 1.2 says the marginal verify row is dominated by expert bytes, which
makes this the only candidate attacking the dominant term directly.
IMPROVEMENTS records "the gather kernel recovers about half of int8's gain at
zero memory", measured at decode width 1.

- Source: `titan/kernels/moe_gather_ws.py` and `docs/kernels/REPORT.md`.
- Stage: verify forward.
- Estimate: +2 to 5%, more at M=4 than at M=1 because the whole point is
  avoiding a double weight stream, and at M=4 there are 39 experts per layer to
  stream rather than 10.
- Cost: 1 day to re-measure at the verify layout.
- Collisions: it was shelved, not rejected. Measuring it at M=4 is a different
  experiment from measuring it at M=1.

#### D9. mlx gather_qqmm matrix kernels (PRs #4481, #4483)

- Sources: https://github.com/ml-explore/mlx/pull/4481 (open, 2026-09-08,
  design contested), M5 Max prefill 1257.0 to 3105.3 tok/s;
  https://github.com/ml-explore/mlx/pull/4483 (open, approved, 2026-09-09),
  1267.5 to 1365.9 alone and 4256.2 combined.
- Stage: prefill only. Both PRs state generation throughput is effectively
  unchanged, around 68 to 82 tok/s. This is the pattern across every recent
  quantised-MoE kernel PR: prefill moves, decode does not. mlx #2078
  (`gather_qmm`, merged 2025-04-17) is the same story, with Mixtral 8x7B going
  171 to 590 tps of prompt processing and the PR explicitly calling out smaller
  gains with 256 experts and short prompts.
- Estimate: prefill +30 to 100% if it applies to our quant layout, decode 0.
- Cost: 2 days to build and measure, plus the risk of an unreleased mlx.
- Collisions: none.

### A. Acceptance at long context

#### A1. Sweep the context length for a KV-boundary artefact

Before any theory, check whether the 64k acceptance figure is a boundary effect.

- Source: https://github.com/ggml-org/llama.cpp/issues/23658, open, 2026-05.
  MTP acceptance 16% at context 12,032 and 71% at 12,288, repeating on roughly
  2048-token boundaries, on both 35B and 9B, hypothesised as a slot-indexing bug
  in how the draft context shares KV with the target. CUDA data.
- Stage: acceptance at long context.
- Estimate: if it reproduces, the fix could be worth the whole gap. If it does
  not, the experiment costs a day and eliminates the cheapest explanation.
- Titan seam: none for the measurement. Titan's block grid is 512 and its
  snapshot grid is 2048, and the MTP head keeps its own QSA cache with its own
  `index_offset` that has to stay aligned with the target's
  (`language.py:719` raises on misalignment, which is a good sign).
- Cost: 1 day.
- Collisions: none.

#### A2. Expected-value depth policy in place of `round(mean) + 1`

Maximise expected tokens per unit of cycle time using a measured cost table
indexed by row width and a per-depth acceptance estimate. Titan already records
the cost table: `CycleProfile` carries rows and stage times.

- Sources: MTPLX 2.11.2 (https://releasebot.io/updates/mtplx, 2026-09-06),
  "measures draft and verify cost per depth on the running machine". SGLang's
  adaptive speculative decoding issue
  (https://github.com/sgl-project/sglang/issues/23705, opened 2026-04-25, open)
  builds the same thing with an EMA controller over pre-built runtime tiers,
  with step=0 merged so speculation can be turned off at runtime. vLLM's DSpark
  chooses its budget "by maximizing expected tokens per unit step time from a
  profiled cost table"
  (https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification, 2026-08-14),
  and reports the key datum that "the 7th drafted token survives under 10% of
  the time vs over 70% for the first".
- Stage: acceptance at long context, and the draft and verify stages indirectly.
- Estimate: at 64k with accepted 1 the EV rule picks depth 1 for 2 / 27.0 = 74
  tok/s against the current depth-2 choice at 2 / 34.1 = 59. Measured production
  is 71, so realistically +5 to 15% at long context and near zero at short
  context where the current rule already lands on 3. The second-order effect is
  larger: a stable depth is the precondition for D5.
- Titan seam: `titan/engine/decode_cycle.py`, `DepthController.plan_depth` and
  `AcceptanceEstimator`. Pure Python, no device work, and the depth-control
  cases in `tests/engine/test_mtp_parity.py` already cover it.
- Cost: 1 to 2 days.
- Risks: a policy that reads its own cost table can oscillate. TapOut's answer
  is a bandit rather than a threshold.
- Collisions: this is the closest thing here to the rejected confidence-gated
  depth (-6%), and the difference is the signal. The rejected gate drafted
  *deeper* on the draft head's own confidence. TapOut
  (https://arxiv.org/abs/2511.02017, 2025-11) measures that whole family
  failing: AdaEDL at 0.93x against a static depth, SpecDec++ at 0.99x despite
  raising acceptance from 0.55 to 0.65, with the diagnosis that "entropy decays
  with generation length ... a static threshold across all prompts and positions
  is suboptimal". The EV policy uses observed acceptance history and measured
  cost, never the drafter's confidence, and its usual effect is to draft
  shallower. Opposite intervention.

#### A3. Gate the draft on a probability floor and raise the ceiling

Rather than tuning depth, keep a high maximum and terminate the chain when the
draft's top-token probability falls below a floor.

- Source: https://github.com/ggml-org/llama.cpp/discussions/25198, open,
  2026-07. Across 184 real coding sessions, `--spec-draft-n-max 16
  --spec-draft-p-min 0.8` beat the legacy `(4, 0.0)` at **119.53 against 99.29
  tok/s, +20.4%**, while acceptance *fell* from 67.8% to 55.7% and mean accepted
  run length rose from 3.71 to 5.20. User-run, not maintainer CI.
- Stage: acceptance at long context, and the draft chain.
- Estimate: this is the strongest published counter to optimising acceptance
  rate, and the metric it optimises (accepted run length) is the one Titan's
  cycle actually spends rows on. Estimate +5 to 20%, overlapping with A2.
- Titan seam: `MTPDecodeCycle._propose` terminating the chain early, which
  requires a device-side comparison the drafter can do without a sync, or a
  one-cycle-late signal.
- Cost: 2 days on top of A2.
- Risks: reading the draft probability needs either a sync or a device-side
  early exit. A device-side early exit means a variable-width row block, which
  fights D5.
- Collisions: this is a confidence gate, and IMPROVEMENTS rejected one at -6%.
  The difference is direction and pairing: the rejected gate raised depth on high
  confidence, whereas p_min *truncates* on low confidence while the ceiling goes
  up. TapOut's measurement is that gates of the first kind lose; llama.cpp's
  measurement is that a gate of the second kind wins by 20%. If the Titan
  version ends up raising depth on confidence, drop it.

#### A4. Suffix-tree drafting as a second lane

Build a suffix tree over the prompt plus a bounded cache of recent generations,
match the output suffix against it, and propose a variable-length continuation
when the match is long enough. Model-free, host-side, and it strengthens as the
prompt grows.

- Source: https://arxiv.org/abs/2411.04975 (SuffixDecoding, NeurIPS 2025
  Spotlight), production write-up
  https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
  (2025-12-02), merged into vLLM as
  https://github.com/vllm-project/vllm/pull/25784.
- Measured: up to 5.3x on agentic workloads, 1.4 to 3.9x faster than vLLM's
  n-gram on SWE-Bench, 1.96 to 3.12x on BlazeEdit, 1.0 to 1.28x on SpecBench,
  roughly 10% CPU overhead at concurrency 64. Knobs:
  `suffix_decoding_max_tree_depth` 24, `max_cached_requests` 10000,
  `max_spec_factor` 1.0, `min_token_prob` 0.1.
- Stage: acceptance, on edit-heavy and agentic turns.
- Estimate: +20 to 50% on the `edit` and `code` prompts in
  `bench/decode_bench.py`, near zero on `prose` and `json`. Call it +10 to 20%
  on the median.
- Titan seam: a `Drafter` composed with the MTP drafter, chosen per cycle in
  `MTPDecodeCycle._propose`. The tree lives on the scheduler thread and costs no
  GPU, and the port already allows an empty tuple per sequence so falling back to
  MTP is legal today.
- Cost: 3 to 4 days, including the two bug classes MTPLX documented publicly
  (copy blocks accepted past a stop token poison the recurrent state, and the
  copy must be prompt-sliced rather than reading generated output).
- Risks: CPU time on the scheduler thread, which also owns templating,
  detokenisation, acceptance bookkeeping and profiling. Variable proposal length
  fights the fixed-width verify block, so it wants D5's bucketing.
- Collisions: IMPROVEMENTS rejected an n-gram copy lane as neutral. The
  difference is real: the rejected lane copied fixed-length blocks from the
  prompt with no frequency statistics, whereas SuffixDecoding adapts proposal
  length per step from the empirical continuation frequency in the tree, and its
  own paper measures it beating tuned n-gram by 1.4 to 3.9x with one fixed
  configuration. If the Titan version is a fixed-length copy again, drop it.

#### A5. Check what the MTP chain is fed, and at what precision

Two separate questions, both cheap, both with published claims behind them.

First, normalisation. EAGLE 3.1 attributes long-context acceptance collapse to
attention drift and fixes it by normalising each target hidden state before the
FC layer and feeding post-norm hidden states into the next drafting step, so the
drafter behaves like a recursive invocation of the target
(https://vllm.ai/blog/2026-05-26-eagle-3-1, 2026-05-26, merged, ships in
v0.22.0, claiming up to 2x longer acceptance specifically on long context and
2.03x at concurrency 1 on Kimi K2.6). `Qwen4ExpMTPModule.fuse_inputs` already
applies `pre_fc_norm_embedding` and `pre_fc_norm_hidden`, so step 1 looks right.
Step 2 is the question: oMLX's chain re-enters the head on its own output hidden
(`_chain_next_drafts:2422`, `h = head_hidden[:, -1:]`) rather than the
backbone's, and whether the trunk RMSNorm is applied on that path is not
recorded anywhere.

Second, precision. mlx-lm PR #990
(https://github.com/ml-explore/mlx-lm/pull/990, open, opened 2026-03-13)
implements MTP draft and verify with SSM state rollback for hybrids and reports
Qwen3.5-27B dense 4-bit on M4 Pro going 15.7 to 24.6 tok/s at 88.3% acceptance,
but MoE results of 1.11x at about 11% acceptance on 35B-A3B and 1.09x at about
9% on 122B-A10B, with the stated finding that MTP weights must stay in BF16
because quantising the head alongside the backbone degrades MoE acceptance to
near zero, and the attribution that a single MTP layer struggles to predict
expert routing.

- Stage: acceptance at long context.
- Estimate: the normalisation check is a half-day for a possible 30%. The
  precision claim is harder to credit here, because IMPROVEMENTS already
  measured 4-bit against 8-bit at 80.7% and 80.9% acceptance, and 80% is nothing
  like the 9 to 11% the PR reports. That says Titan's head is not in the broken
  regime. Full BF16 was never tested, and the whole `mtp.*` block is 1.48 GB, so
  a BF16 sidecar is roughly 3 GB of the 50 GB of headroom. Estimate 0 to 10% for
  BF16, and only worth doing if the 8-bit A/B is repeated at 64k rather than at
  short context, which it was not.
- Titan seam: `titan/adapters/mlx/model.py:mtp_draft`, the vendored
  `Qwen4ExpMTPModule`, and `titan/adapters/mlx/checkpoint.py` for the sidecar.
- Cost: half a day to check normalisation, 2 days for a BF16 sidecar A/B.
- Collisions: the BF16 half collides directly with the rejected 8-bit MTP draft
  block. The only thing that makes it a different experiment is running it at
  64k, where the acceptance problem actually is, and going to BF16 rather than
  8-bit. If a 64k A/B of the existing 8-bit sidecar shows nothing, drop it.

#### A6. Skip the MTP head's own indexer

The narrow half of D3, restricted to the draft head. Cheaper and safer, because
the head is one layer and its selection is never used by the target.

- Source: as D3, https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/.
- Estimate: +3 to 6% at 64k.
- Titan seam: `Qwen4ExpMTPModule.__call__` and the head's QSA cache.
- Cost: 1 to 2 days. Do this before D3 as the cheap half.
- Collisions: none.

#### A7. Online distillation of the MTP head (Test-Time Speculation)

The verify forward already computes the target's logits at every drafted
position, which is exactly the supervision a draft head needs. TTS treats the
draft as a student and the target as a teacher and applies a small online update
every few speculation rounds, so the head tracks the target as the session moves
out of its training distribution.

- Source: https://arxiv.org/abs/2605.09329, submitted 2026-05-10, revised
  2026-05-19, arXiv preprint. Abstract verbatim on the acceptance curve: "the
  acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3
  and PARD degrade with generation length, reaching values close to 1 (i.e. no
  speedup) within just a few thousand output tokens". Measured: acceptance
  length up to +72% and +41% on average across Qwen-3, Qwen-3.5 and Llama-3.1,
  "with the benefits scaling with increased generation lengths".
- Stage: acceptance at long context, and nothing else.
- Estimate: the only candidate here with a plausible path to doubling 64k
  decode. If accepted goes 1 to 2.5 at the current width, 3.5 / 34.1 = 103
  tok/s, so +45%. Short context gains little, because acceptance is already 80%.
- Titan seam: a `Drafter` implementation owning a small adapter over
  `mtp.fc_embedding` and `mtp.fc_hidden`, plus a richer payload than
  `VerifyOutcome` through `Drafter.observe`.
- Cost: 8 to 12 days. Gradients inside a serving loop, a bounded update budget so
  the GPU does not lose the cycle it saved, and a rule for the adapter across
  requests and across prefix-cache restores.
- Risks: the output stays exact, because a changed drafter does not change what
  the target verifies, but the adapter is per-session state the prefix-cache
  signature does not cover, and the update competes with decode for the GPU.
- Collisions: none, and it explains the rejected 8-bit MTP result rather than
  contradicting it. Precision was never the limiter; distribution shift is.

#### A8. Turn speculation off when it stops paying

- Sources: mlx-serve v26.9.2, 2026-09-09: "Past some conversation length a
  speculative step costs more than it saves; Flash Next now measures both and
  switches speculation off there and back on when it pays again"
  (https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.2), with the M4 Max
  headline going 83 to 93 tok/s and "+30% at 64k and 128k". SGLang's merged
  step=0 support under https://github.com/sgl-project/sglang/issues/23705.
- Estimate: this is A2 with `min_depth` allowed to reach 0, which Titan already
  supports. The plain step at 64k is 17.7 ms for one token, 56 tok/s, against a
  measured 71, so on today's numbers speculation still pays at 64k and the switch
  would rarely fire. It becomes valuable past 128k. Estimate 0% at 64k, +10 to
  30% past 128k.
- Cost: half a day on top of A2.
- Collisions: the rejected "MTP park policy" is this idea with a heuristic
  trigger and an exponential 128-to-4096-token cooldown. A8 triggers on a
  measured cost comparison and is a depth clamp, not a separate parked state
  machine with its own reconciliation path. If the Titan version reintroduces a
  parked state, drop it.

### H. Host sync (13%)

#### H1. Never read the accepted count

Keep the accepted count and the bonus token as device arrays, index the
embedding table with them, and build the next cycle's inputs on device. The host
learns what was committed one cycle late.

- Sources: SGLang's zero-overhead batch scheduler with placeholder "future
  tokens" (https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/, 2024-12-04,
  shipped in v0.4, 1.1x over v0.3, 1.3x over baselines, Nsight showing zero GPU
  idle across five consecutive decode batches). vLLM async scheduling
  (https://github.com/vllm-project/vllm/pull/19970, merged 2025-07-15, 3 to
  15%) and its speculative variant caching draft tokens in the GPU model runner
  (https://github.com/vllm-project/vllm/pull/24799, merged 2025-11-17, +1.8% at
  24 prompts rising to +7.1% at 96, bitwise identical). vLLM Model Runner V2
  builds `input_ids`, positions and block tables on device and states that
  "GPU-resident preparation kernels can directly consume speculative decoding
  results without CPU intervention" (https://vllm.ai/blog/2026-03-24-mrv2,
  2026-03-24, experimental, -6.3% latency for spec decoding on GLM-4.7-FP8, and
  explicitly not yet supporting linear attention). TensorRT-LLM's overlap
  scheduler documents the trade as "costs one extra decoding step of latency"
  (https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html).
- Stage: acceptance host sync, all 13%.
- Estimate: the ceiling is 1.15x and nobody reaches it. vLLM's measured numbers
  trend down toward batch 1, at 1.8 to 3%. Call it +4 to 10%.
- Titan seam: this is the hardest architectural item here, because Titan is
  built on the single-host-sync invariant rather than a zero-host-sync one.
  `MLXModelBackend.verify` at `titan/adapters/mlx/backend.py:210` does
  `mx.eval(accepted_counts, bonus)` then two `.tolist()` calls, and everything
  downstream (`commit`, `TextEmitter`, stop conditions, `truncate_state`) reads
  Python ints. Note that mlx-lm's own `speculative_generate_step` has exactly the
  same structure, so this is a limitation of the reference design rather than a
  Titan mistake.
- Cost: 10 to 15 days, invalidating a large part of `tests/engine/`. This is an
  M5 project, not an experiment.
- Risks: the invariant that `host_syncs != 1` is a bug becomes meaningless, and
  `TextEmitter`'s stop-string machinery depends on knowing exactly which tokens
  survived.
- Collisions: none, but ARCHITECTURE section 10 lists "in-graph acceptance with
  one host sync" as one of the five things Titan was built to reach. This
  candidate says the target should have been zero.

#### H2. Fixed-shape masked commit

Size every per-cycle buffer to the maximum row width and mask, rather than
slicing at Python level on the accepted count. The prerequisite for both H1 and
D5.

- Source: https://github.com/sgl-project/sglang/pull/26128, merged 2026-05-29.
  A backend capability flag so that when every backend in a spec-v2 forward opts
  out, the scheduler skips the device-to-host transfer of `seq_lens` and the
  `int(sum())`, and runs a fully device-resident metadata path.
- Estimate: nothing alone. A 2-day enabler for two items worth 10 to 25%.
- Titan seam: `MLXModelBackend.verify` and `ModelState.truncate`.
- Risks: masking rather than slicing means reading padding. At 45% of the ceiling
  there is headroom, but measure it.
- Collisions: none.

#### H3. Measure and then slim the per-verify snapshot

`TitanQwenFlashNext.verify` calls `state.stage_snapshot()` before every forward,
and `stage_snapshot` does `mx.array(value)` over every slot of every
`ArraysCache`, which is 36 recurrent states plus conv windows, about 110 MiB.
That is a device copy per cycle whether or not the draft is rejected, plus 36
or more dispatches that break fusion.

- Source: llama.cpp PR #28123
  (https://github.com/ggml-org/llama.cpp/pull/28123, "qwen4exp: support
  recurrent state rollback", merged 2026-09-01) is the same problem on the same
  model. Before the fix "the server serialized the entire recurrent state to
  host memory on every speculative round, negating drafting benefits"; the fix
  writes one snapshot per rollback slot for the delta-net QKV convolution and
  the PLE convolution. Measured on Qwen3.8-Flash-Next UD-Q4_K_XL at n-max 3, one
  slot, on an RTX PRO 6000: no draft 108 tok/s; before the fix 123 code and 83
  prose; after the fix **183 code and 144 prose**, so +49% and +73%. Note that
  prose before the fix (83) was *slower than not drafting at all* (108).
  Also relevant: llama.cpp PR #19493
  (https://github.com/ggml-org/llama.cpp/pull/19493, merged 2026-04-19)
  introduced speculative checkpointing because recurrent modules cannot do a
  partial sequence removal, and its author notes checkpoints are slower than a
  trim. SpecLA's alternative is factor buffering rather than snapshots
  (https://arxiv.org/abs/2607.16673, 2026-07-18, GDN-1.3B on H100, up to 1.70x
  end to end).
- Stage: the commit phase, which the profiler folds into `accept`.
- Estimate: 110 MiB at the 549 GB/s copy ceiling is 0.2 ms, under 1% of the
  cycle, so if the copy stays on device this is small. If it does not, or if the
  36 extra dispatches matter at the rate section 1.1 suggests they might, it is
  larger. The llama.cpp numbers are on a discrete GPU where a host round trip is
  catastrophic and unified memory changes that, so do not expect +49% here.
  Estimate 0 to 8%, and measure before building. This is the measurement step 4
  of the workbench plan in `REPORT.md` section 3 was written for and which has
  never been run.
- Titan seam: `titan/adapters/mlx/state.py:stage_snapshot` and
  `titan/adapters/mlx/model.py:verify`.
- Cost: 1 day to measure, 4 to 8 days to replace copy-per-cycle with
  replay-on-rejection or factor buffering.
- Collisions: none. oMLX already does replay rather than copy
  (`OMLX-COMPONENT-MAP.md` section 2, `_chain_rollback:3205`, replaying the kept
  prefix through `_process_chunk` per linear layer, paid only on rejection).
  Titan's replay-free rollback was a deliberate design choice; this candidate
  asks whether it was the right one at 80% acceptance.

### C. Cache and prefill

#### C1. Packed GDN Metal kernel from mlx-lm

- Source: https://github.com/ml-explore/mlx-lm/pull/1559, "Add packed gated
  delta kernel, bitwise-pinned by an explicit-tree comparator", merged
  2026-08-27. Eight value rows per 32-lane SIMD group, four lanes per row, 32
  contiguous state elements per lane in registers, an explicit butterfly tree
  replacing `simd_sum`. Guards: `Dk == 128`, `Dv % 8 == 0`, scalar gate, no
  padding mask, `g.dtype == float32` and `state.dtype == float32`. Kill switch
  `MLX_GDN_PACKED=0`. Measured on M5 Max with Qwen3.6-35B-A3B: per-layer 2.54 to
  2.56 ms down to 1.34 to 1.38 ms, roughly 1.8 to 2.0x, with independent
  validation at 1.93 to 2.04x and token-identical greedy output.
- Stage: prefill only. The PR states plainly that it targets the prefill path
  and that decode at T=1 uses the existing kernels.
- Estimate: end-to-end prefill on their model was **1.064x, +6.4%**, despite the
  2x per-layer number, which is the honest figure. Titan has 36 GDN layers of 48
  against their 30 of 40, so a similar ratio. Estimate +4 to 8% prefill, 0
  decode. Titan's config has `linear_key_head_dim` 128 and
  `linear_value_head_dim` 128, so both shape guards pass; the fp32 state guard
  needs checking.
- Titan seam: `titan/kernels/gdn_chunk_scan.py`, which already carries the
  chunked scan from mlx PR #4020 at +0.7%.
- Cost: 2 days to port and check the guards.
- Collisions: none. Also note mlx PR #4020
  (https://github.com/ml-explore/mlx/pull/4020, still open, active to
  2026-09-10) is measured at 1.6 to 2.2x over sequential only for batched
  prefill at B>=4 and T<=2048, and its successor #4409 targeting the sequential
  path reports its *smallest* gain, 1.25x, at exactly B=1, which is decode. The
  whole GDN kernel effort upstream moves prefill.

#### C2. GDN snapshots at the fine grid

IMPROVEMENTS lists this as open: 58% of warm-turn recompute is text already
processed. Round 6 shipped `OMLX_CACHE_FINE_TAIL=512` on the overlay and Titan's
CACHE.md already decouples the two grids and snapshots at every prompt end.

- Source: internal, `docs/architecture/CACHE.md` sections 1 and 2. External
  corroboration that this is the binding problem for hybrids: mlx-lm issue #980
  (https://github.com/ml-explore/mlx-lm/issues/980, closed 2026-03-11 with no
  merged fix) records sliding-window, Mamba and mixed-attention models "silently
  falling back to full prompt recomputation on every request", with a
  pure-attention model getting 10.5x on warm requests and Qwen 3.5 getting zero
  or worse. vLLM's equivalent fix measured Qwen3.5-35B-A3B going from 37 s to
  2.6 s on a repeat run, 14x
  (https://github.com/vllm-project/vllm/pull/36649, closed 2026-05-05 in favour
  of #26807), with the hard-won details being chunk-size-64 block alignment and
  fp32 SSM state dtype.
- Stage: prefill on warm turns. This moves latency, not tok/s.
- Estimate: further 20 to 40% on the multi-turn median beyond round 6's 2.59 s
  to 2.10 s.
- Cost: 2 days.
- Collisions: "fine cache boundary (broke the store path)" is on the rejected
  list, and CACHE.md section 2 explains exactly why the overlay's version broke
  (a chunk from 26112 to 27648 stepped over the grid multiple at 26624 without
  landing on it) and that
  `test_every_coarse_multiple_inside_a_suffix_still_ends_a_chunk` now pins it.
  The Titan version differs because `plan_chunks` guarantees every snapshot
  point ends a chunk.

#### C3. Two-LRU recurrent state eviction

- Source: https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/,
  2025-12-03, shipped in SGLang v0.5.5. MambaRadixCache: match returns the
  deepest node with a non-None recurrent state and copies it out, insert forks a
  checkpoint after chunked prefill, and eviction keeps two LRU lists so KV
  evicts leaf-to-root while recurrent states evict from any node.
- Estimate: Titan already has the two-grid design, chain block hashing and
  snapshot-bounded restore. The two-list eviction rule is what is missing, and
  it is worth a few points of hit rate.
- Titan seam: `titan/adapters/cache/store.py`, the hot tier's ordered dict.
- Cost: 3 days.
- Collisions: none.

#### C4. Stop retaining raw QSA indexer keys for the whole context

- Source: https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/, "reduces QSA's
  index-cache overhead by 80%".
- Stage: memory, therefore admission and concurrency.
- Estimate: no tok/s at one stream. At 8 streams it is headroom against the
  110 GB guard, which is what caps aggregate throughput.
- Cost: 4 days, and it interacts with D3, which wants the pooled selection
  cached instead of the raw keys. Do D3 first; it may make this free.
- Collisions: none.

#### C5. INT8-activation prefill

oMLX 0.7.0.dev2 reports "32K prefill 615.2 tok/s (INT8 OFF) vs 826.7 tok/s
(+34.4%)" on M5 Max. Method public: W8A8, per-token dynamic symmetric
activations, per-channel weights derived from the 4-bit weights, eligibility
`N%128==0, K%32==0, min dim >= 1024, N <= 32768`.

- Source: https://github.com/jundot/omlx/releases, 0.7.0.dev2, 2026-09-11.
- Estimate: +10 to 34% prefill if the 512-expert top-10 shapes satisfy
  `N%128==0`. Zero decode.
- Cost: 5 days to port, because Titan is not oMLX and this has to become a
  kernel in `titan/kernels/`.
- Collisions: IMPROVEMENTS round 1 records oMLX's INT8 prefill flag as
  "excludes this model type". Check eligibility before building.

#### C6. CacheBlend-style selective recompute

- Source: https://arxiv.org/abs/2405.16444, ACM TOCS 2026, 2.2 to 3.3x lower
  TTFT and 2.8 to 5x higher throughput against full recompute.
- Estimate: it would apply to the 12 QSA layers only. The 36 GDN layers are a
  fold with no inverse, so they still need a full replay from the last snapshot,
  and that replay is the whole cost. Estimate near zero as a result.
- Cost: 15+ days.
- Collisions: none. Ranked at the bottom, and the GDN argument probably kills it
  outright.

### M. Multi-stream

#### M1. Take the ragged-width verify seam (M4b)

`ENGINE.md` section 8 describes it: `plan_depth` may return different depths per
sequence, the backend already pads, and nothing in `commit` knows the width.

- Sources: internal for the seam. External evidence that per-sequence depth
  matters: SGLang's load-aware policy keyed on batch size
  (https://github.com/sgl-project/sglang/issues/23705) and DSpark's global top-B
  selection across requests, where "position 5 of a confident request can
  outrank position 1 of a low-confidence one"
  (https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification). The
  bandwidth argument is Titan's own: the expert gather runs at 300 GB/s at one
  row and 549 at eight, so rows are the currency.
- Estimate: +5 to 15% aggregate at 4 to 8 streams, zero at 1.
- Titan seam: exactly the two places ENGINE.md names.
- Cost: 3 days.
- Risks: it makes D5's bucketing harder, because the block width becomes the max
  over the batch. Bucket on the max.
- Collisions: "batched MTP verify (lost to plain batching)" is on the rejected
  list. The overlay's version lost because it split rows:
  `OMLX-COMPONENT-MAP.md` section 2 records `_mtp_batch_next:2590` extracting a
  single-row cache per row, running a singleton cycle and merging back, with the
  source's own comment at `:640` reading "standard batched decode is faster at
  batch >= 2". That is decomposition, not batching. Titan's `MTPDecodeCycle`
  makes one `verify` call for the whole batch and never splits. The difference is
  structural, not a tuning change.

#### M2. Batching as the largest available lever, if the workload allows it

MLX discussion #3939 recovers a dispatch-bound model from 1.32 tok/s at batch 1
to 21.2 tok/s aggregate at 32 concurrent streams. Titan gets 60 to 130 across 1
to 8. The published Apple-Silicon concurrency scaling in
`docs/research/APPLE-SILICON-2026-09-12.md` section 2 ranges from 1.29x at 4
streams to 2.9x at 8, so Titan's 2.2x at 8 is mid-range.

- Estimate: not a candidate so much as a framing. If the agentic workload can be
  made concurrent, that is a larger win than anything else here. If it is a
  single interactive stream, it is worth nothing.
- Collisions: the memory guard already serialised concurrency once and was
  fixed. `AdmissionConfig` and the 110 GB guard are the levers.

#### M3. Two-batch overlap

- Source: https://www.lmsys.org/blog/2025-05-05-large-scale-ep/, 2025-05-05,
  merged, prefill +27 to 35%, decode +25.5% at 256 tokens per device and +35%
  under simulated MTP. The transferable rule is to submit GPU compute before any
  host-blocking work.
- Estimate: nothing at batch 1, +5 to 10% aggregate at 8.
- Titan seam: `titan/engine/scheduler.py`, and it breaks the lockstep batching
  rule ARCHITECTURE section 5 states as law.
- Cost: 6 days.
- Collisions: lockstep is a stated invariant.

#### M4. Mixed prefill and decode batches (D14)

- Estimate: ARCHITECTURE section 3 already gives the counter-argument with
  numbers. Batched prefill loses the gathered sparse-attention arm and
  materialises a 134 MB mask per QSA layer at 65k. Listed for completeness. Do
  not build.

### R. Things measured and ruled out, restated so they are not retried

- **Draft trees.** Section 1.2's expert arithmetic, vLLM closing
  https://github.com/vllm-project/vllm/issues/18327 as not planned, ddtree-mlx
  getting only 1.38x to 1.52x on M3 Ultra, and mlx-lm core being unable to
  express them (issues #846, #250). TreeWY (https://arxiv.org/abs/2608.20961,
  2026-08-21) is the best paper on our exact 3:1 GDN-to-attention ratio and it
  makes wide trees affordable in state memory, but its own abstract says "a
  wider, higher-acceptance draft becomes possible, though not yet a throughput
  win". Revisit if the M5 Ultra changes the arithmetic.
- **Layer-skip self-drafting.** https://arxiv.org/abs/2605.01106, 2026-05-01,
  measures acceptance at k=2 of 0.68 on Falcon-H1-0.5B, a parallel hybrid, and
  **0.038 on Qwen3.5-0.8B, a sequential hybrid**, an 18x gap, with wall-clock
  speedup below 1.0x. Titan's model is a sequential hybrid. Do not build.
- **An external draft model.** mlx-lm issue #1132
  (https://github.com/ml-explore/mlx-lm/issues/1132, open, 2026-04-08) measures
  Qwen3.5-397B-A17B with a 9B draft at -35% average throughput, worst -45% on
  Python codegen. With 6B active parameters an external drafter cannot pay for
  itself. MTP is the only viable speculation route on this model.
- **Confidence-gated depth in the direction of drafting deeper.** Rejected
  locally at -6%, and TapOut measures the family at 0.93x to 0.99x against a
  static depth. A2 and A3 are different signals in the other direction.
- **8-bit MTP draft block.** Rejected locally at 80.7% against 80.9%, and TTS
  explains why precision was never the limiter. A5 revisits only the untested
  BF16 case, only at 64k, and only if the normalisation check comes back clean.
- **Graph capture on Metal.** It does not exist in MLX. MLX issue #2358
  (closed, low priority) is CUDA only. There is no
  `MTLIndirectCommandBuffer` PR or issue in MLX or llama.cpp. Both projects
  answer dispatch overhead with graph-level op fusion instead: llama.cpp PR
  #28164 (https://github.com/ggml-org/llama.cpp/pull/28164, merged 2026-09-11)
  declares fusable Metal patterns in one table consumed by both the optimiser
  and the encoders, and measures about +5% token generation. A true persistent
  megakernel is also blocked, because Metal has no device-side kernel launch, no
  forward-progress guarantee across threadgroups, and 32 KB of threadgroup
  memory against the 213 KB the Hazy Research design assumes.
- **Neural Engine, media engine, native int4 tensor math.** Ruled out with
  numbers in `docs/research/APPLE-SILICON-2026-09-12.md` section 2 and
  `docs/kernels/RESEARCH-2026-09-12.md` section 7. NAX itself is real and shipped
  in MLX v0.30.0 with quantised paths (https://github.com/ml-explore/mlx/pull/2772,
  merged 2025-11-19, requires macOS 26.2+), but Apple's own figures put the
  benefit at 3.33 to 4.06x on time-to-first-token and only 1.19 to 1.27x on
  generation, tracking the 28% bandwidth increase rather than the compute one
  (https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- **KV quantisation as a decode lever.** vLLM's FP8 KV post
  (https://vllm.ai/blog/2026-04-22-fp8-kvcache, 2026-04-22) measures the
  inter-token-latency slope at 54% of BF16 on a full-attention model but 96% on
  an unskipped sliding-window model, and QSA reads at most 2051 positions
  regardless of context, so the bytes it moves per row do not grow with context.
  MTPLX measures "KV q8 costs about 4 percent for double the context headroom".
  MLX community data puts quantised KV at +1.1% on a small dense model
  (https://github.com/ml-explore/mlx/discussions/3134). The gain here is memory
  headroom for concurrency, not tok/s. One warning if it is done anyway: MLX
  issue #3480 (closed via PR #3497) records `quantized_matmul` producing errors
  up to 12.46 in magnitude with GQA stride-0 batch dims at M>=2, manifesting only
  in speculative-decoding two-token verification passes.
- **Cascade attention, batch-size-keyed speculation length.** Pure batching wins
  needing concurrency in the hundreds.
- **SpecPrefill and token-dropping prefill.** Lossy, wrong for coding.

---

## 4. Ranking by expected tok/s gain per day of work

Gain is the midpoint of the range as a percentage of the relevant production
number, divided by the midpoint of the day estimate. Long and short context are
scored separately where they differ, and the rank is set by whichever is larger,
because 64k is where the agentic workload lives.

| # | candidate | stage | est. gain | days | gain/day | confidence |
|---|---|---|---|---|---|---|
| 1 | D1 `TITAN_QWEN4_EAGER_DISPATCH=0` A/B | verify | -5 to +20% | 0.5 | 15%/d | low, free |
| 2 | D2 raise `MLX_MAX_OPS_PER_BUFFER` | verify | +2 to 8% | 0.5 | 10%/d | medium, free |
| 3 | A5a check MTP chain normalisation | acceptance 64k | 0 or +30% | 0.5 | 30%/d | low, cheap |
| 4 | A1 context-length sweep for KV boundary | acceptance 64k | 0 or large | 1 | n/a | low, eliminates a cause |
| 5 | A2 EV depth policy | acceptance 64k | +5 to 15% | 1.5 | 7%/d | high |
| 6 | A6 skip the MTP head's indexer | draft, 64k | +3 to 6% | 1.5 | 3%/d | high |
| 7 | D8 re-measure weight-stationary gather at M=4 | verify | +2 to 5% | 1 | 3.5%/d | medium |
| 8 | D4 fused `verify_accept` kernel | host sync | +2 to 5% | 2 | 1.8%/d | high |
| 9 | H3 measure the per-verify snapshot cost | commit | 0 to 8% | 1 measure | 4%/d | medium |
| 10 | D3 verify block through the decode QSA path | verify, 64k | +10 to 25% | 3.5 | 5%/d | high |
| 11 | A3 p_min draft gate with a raised ceiling | acceptance | +5 to 20% | 2 | 6%/d | medium-high |
| 12 | D5 compiled verify lane, bucketed widths | verify | +8 to 20% | 4.5 | 3.1%/d | medium-high |
| 13 | A4 suffix-tree drafting lane | acceptance | +10 to 20% median | 3.5 | 4.3%/d | medium |
| 14 | C1 packed GDN kernel (mlx-lm #1559) | prefill | +4 to 8% prefill | 2 | 3%/d prefill | high |
| 15 | M1 ragged-width verify (M4b seam) | multi-stream | +5 to 15% at 4-8 | 3 | 3.3%/d | medium |
| 16 | D6 fused hyper-connection Mix/Combine | verify | +3 to 8% | 3.5 | 1.6%/d | medium |
| 17 | H2 fixed-shape masked commit | enabler | 0 alone | 2 | 0 | high |
| 18 | A8 speculation off switch | acceptance >128k | 0 at 64k | 0.5 | n/a | high |
| 19 | C2 GDN snapshots at the fine grid | warm latency | 20 to 40% latency | 2 | n/a | high |
| 20 | C5 INT8-activation prefill port | prefill | +10 to 34% prefill | 5 | 4.4%/d prefill | medium |
| 21 | D7 small-M flash-decoding verify attention | verify 64k | +3 to 8% | 5.5 | 1%/d | medium |
| 22 | D9 mlx PRs #4481 / #4483 | prefill | +30 to 100% prefill | 2 | high, prefill | low, PRs open |
| 23 | C3 two-LRU recurrent state eviction | warm hit rate | hit rate | 3 | n/a | medium |
| 24 | A5b BF16 MTP sidecar, A/B at 64k | acceptance 64k | 0 to 10% | 2 | 2.5%/d | low |
| 25 | C4 stop retaining raw indexer keys | memory | concurrency headroom | 4 | n/a | medium |
| 26 | A7 online distillation of the MTP head | acceptance 64k | +30 to 60% | 10 | 4.5%/d | medium |
| 27 | M3 two-batch overlap | multi-stream | +5 to 10% at 8 | 6 | 1.2%/d | low |
| 28 | H3b replay or factor buffering for rollback | commit | 0 to 8% | 6 | 0.7%/d | low |
| 29 | H1 zero host sync, device-resident commit | host sync | +4 to 10% | 12 | 0.6%/d | high on mechanism |
| 30 | C6 CacheBlend selective recompute | prefill | near zero on hybrids | 15 | ~0 | low |
| 31 | TreeWY, layer-skip drafting, external drafter, mixed batch | - | negative or zero | - | - | do not build |

Two entries deserve a note against their rank.

A7 lands mid-table on gain per day, but it is the only candidate with a
plausible path to doubling 64k decode and the only one attacking the cause the
literature identifies rather than a symptom. If experiments 1 to 5 confirm that
acceptance rather than cost is the binding constraint at 64k, promote it to the
top of the next round regardless of its cost.

H1 lands near the bottom on gain per day and is still probably right. Every
major engine has done it, the mechanism is not in dispute, and it is the only
item here that makes Titan's decode loop structurally the same shape as vLLM V1
and SGLang v0.4. Schedule it as an M5 milestone, not as an experiment.

---

## 5. The first five experiments

Run every one on the workbench, paired against a baseline inside the same round,
45 s cooldown, quiet GPU, fan floor, two rounds, per the protocol in
IMPROVEMENTS section 4. A result inside noise is a negative result and gets
recorded as one.

### Experiment 1: is the verify forward dispatch-bound?

Two free variables in one sweep: `TITAN_QWEN4_EAGER_DISPATCH` in {1, 0} and
`MLX_MAX_OPS_PER_BUFFER` in {50, 200, 500} with `MLX_MAX_MB_PER_BUFFER` scaled
alongside. Four to six paired rows.

The flag's own comment says the change is bit-identical, so this tests one
assumption cleanly: that 48 graph breaks per forward are worth their dispatch
cost at M between 1 and 8. Every other item in the ranking is calibrated against
the 45%-of-ceiling figure, and that figure was measured with the breaks in
place.

- Scripts: `bench/decode_bench.py --tag dispatch-<cfg>` for short context,
  `bench/e2e_cold.py --tag dispatch-<cfg>` for the 64k tail and cold prefill.
- Acceptance metric: decode median moves by more than 3% on a paired two-round
  run at any configuration. A move down is as useful as a move up, because it
  pins the flag with a Titan measurement instead of an inherited comment. If
  anything moves, re-run the decode profiler and re-derive the stage split
  before anything else in this document is acted on.
- Second, mandatory check: run one generation past 12,000 tokens with the winning
  configuration and watch the Metal buffer count and `mx.get_active_memory`, per
  mlx-lm issue #1332. A configuration that is faster for 600 tokens and dies at
  11,300 is not a win.
- Half a day.

### Experiment 2: what does the marginal verify row actually cost, and why?

Run step 4 of the workbench plan in `REPORT.md` section 3, which has never been
run: the depth-0 plain-step budget at 64k against the depth-3 budget, in sync
mode, with the sub-stage breakdown. Then answer three questions the ranking
depends on and nobody has measured.

1. How much of the 9.3 ms marginal row is QSA selection? The gathered path in
   `qsa_fast.py` runs a top-512 over 16,384 blocks per query row per layer, and
   section 1.4 says the verify block should not be on that path at all.
2. How much of the `accept` phase is the per-verify `stage_snapshot`, which
   copies 36 recurrent states every cycle whether or not the draft is rejected?
3. How much is the expert gather, which section 1.2's byte model says should be
   most of it?

- Scripts: `analyze.py <dir> --ctx-min 60000 --k 0` against `--k 3`, and
  `analyze.py --compare`.
- Acceptance metric: a stage table that attributes at least 85% of the marginal
  row to named sub-stages. That number decides whether experiment 5 is worth
  three days or thirty minutes.
- 1 day.

### Experiment 3: EV depth policy and a probability floor

Replace `DepthController.plan_depth` with an expected-value rule over a cost
table built from `CycleProfile`, keyed on row width, plus a per-depth acceptance
window. Keep `round(mean) + 1` behind a config flag so the A/B is one setting.
Then add the p_min truncation from llama.cpp discussion #25198 with the ceiling
raised, as a second row in the same sweep.

- Scripts: `bench/decode_bench.py` for short context, the 64k tail from
  `bench/e2e_cold.py`, and `bench/mtp_stats.py` for tokens per cycle and the
  per-depth accept counts.
- Acceptance metric: 64k decode above 78 tok/s (from 71) with short-context
  decode not below 84. Secondary and more informative: mean accepted run length
  rises even if the acceptance percentage falls, which is the effect llama.cpp
  measured at +20.4%. Third: the depth histogram at 64k concentrates rather than
  sitting at 2 by construction.
- 2 to 3 days.

### Experiment 4: is the 64k acceptance figure a boundary artefact?

Sweep decode acceptance at contexts of 8k, 16k, 32k, 48k, 60k, 62k, 64k, 66k,
96k and 128k, recording per-depth acceptance at each. This is step 5 of the
workbench plan plus the boundaries llama.cpp issue #23658 identified.

This is cheap and it is the only experiment that can invalidate the whole
acceptance story. If acceptance is 3 at 62k and 1 at 64k, the problem is an
alignment bug between the MTP head's QSA `index_offset` and the target's, not
distribution shift, and A7 is ten days of work aimed at the wrong thing.

- Scripts: `bench/e2e_cold.py --words N` across the sweep, `bench/mtp_stats.py`
  on each, plus the decode profiler at each context.
- Acceptance metric: a monotone acceptance-versus-context curve, or a
  discontinuity. Either answer is worth having; there is currently no recorded
  per-depth acceptance at long context anywhere in the docs, which is itself the
  gap this closes.
- 1 day.

### Experiment 5: put the verify block on the decode-shaped QSA path

Cache the block selection computed for the last accepted row on the QSA cache
and reuse it for the draft chain (A6) and for rows 1..k of the verify block
(D3). Do A6 first as a separate row in the sweep, because it is the cheap half
and it is safe: the head's selection is never used by the target. Gate both
behind kernel config names so `kernels.disabled` can bisect them.

- Scripts: `bench/e2e_cold.py --tag qsareuse` for the 64k tail,
  `bench/decode_bench.py` for short context, `bench/mtp_stats.py` for
  acceptance, and the coding probes from `bench/bench.py` because this one is not
  exact.
- Acceptance metric: 64k decode up by at least 8%, short-context decode unchanged
  within noise, coding probes still 5/5, and per-depth acceptance unchanged
  within 2 points. That last one is the check Qwen's report predicts will pass
  ("no significant change in the mean accepted length after QSA reuse"). If it
  fails here, the reuse is wrong for this implementation and it comes out.
- 3 to 4 days, or half a day if experiment 2 shows QSA selection is not where
  the marginal row goes.

Together these cost about eight days, three of them are pure measurement, and
they answer the two questions the whole ranking rests on: whether 45% of the
bandwidth ceiling is a dispatch artefact, and whether the 64k acceptance
collapse is a cost problem, an alignment bug, or a drafter problem. Everything
in D5, A4 and A7 should wait on those answers.

---

## 6. Source index

All fetched on 2026-09-12, all returning 200.

**Qwen3.8-Flash-Next primary**
- https://arxiv.org/abs/2608.30320 technical report, 2026-08-31 (MTP Table 4, QSA index reuse)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next card and config.json
- https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/ SGLang day 0, 2026-08-26
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next vLLM recipe
- https://inferencex.semianalysis.com/model/qwen-3-8-flash-next
- https://qwen.ai/blog?id=qwen3.8-flash-next (JS shell, no extractable content)

**MLX and mlx-lm**
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html
- https://github.com/ml-explore/mlx/pull/1864 command-buffer budgets, merged 2025-02-14
- https://github.com/ml-explore/mlx/pull/2772 NAX support, merged 2025-11-19
- https://github.com/ml-explore/mlx/pull/2078 gather_qmm, merged 2025-04-17
- https://github.com/ml-explore/mlx/pull/4020 GDN Metal kernels, open
- https://github.com/ml-explore/mlx/pull/4409 packed gated_delta_seq, open
- https://github.com/ml-explore/mlx/pull/4481 gather_qqmm matrix kernels, open 2026-09-08
- https://github.com/ml-explore/mlx/pull/4483 global scales in qmm_t, open
- https://github.com/ml-explore/mlx/pull/4487 SDPA D512 memory, open
- https://github.com/ml-explore/mlx/releases/tag/v0.32.2 2026-08-25
- https://github.com/ml-explore/mlx/discussions/3939 dispatch-bound decode, 2026-07-28
- https://github.com/ml-explore/mlx/discussions/3134 quantised KV community data
- https://github.com/ml-explore/mlx-lm/pull/1559 packed GDN kernel, merged 2026-08-27
- https://github.com/ml-explore/mlx-lm/pull/990 MTP with SSM rollback, open
- https://github.com/ml-explore/mlx-lm/pull/1817 MLA L=1 gate removal, open 2026-09-02
- https://github.com/ml-explore/mlx-lm/pull/1788 qwen4_exp support, open
- https://github.com/ml-explore/mlx-lm/pull/1870 chunkwise gated delta rule, open
- https://github.com/ml-explore/mlx-lm/issues/1132 external drafter -35%, open
- https://github.com/ml-explore/mlx-lm/issues/980 hybrid prefix caching broken, closed
- https://github.com/ml-explore/mlx-lm/issues/1332 Metal buffer limit, closed
- https://github.com/ml-explore/mlx-lm/discussions/890 EAGLE-3 prototype, 1.05x

**llama.cpp**
- https://github.com/ggml-org/llama.cpp/pull/28123 qwen4exp recurrent state rollback, merged 2026-09-01
- https://github.com/ggml-org/llama.cpp/pull/28098 Metal sparse flash attention, merged 2026-09-03
- https://github.com/ggml-org/llama.cpp/pull/28164 Metal fusion rework, merged 2026-09-11
- https://github.com/ggml-org/llama.cpp/pull/27461 Metal 4 tensor path fix, merged 2026-09-01
- https://github.com/ggml-org/llama.cpp/pull/19493 speculative checkpointing, merged 2026-04-19
- https://github.com/ggml-org/llama.cpp/pull/22673 MTP head support, merged 2026-05-16
- https://github.com/ggml-org/llama.cpp/discussions/25198 p_min gating, +20.4%, open
- https://github.com/ggml-org/llama.cpp/issues/23752 spec decode a loss on Metal, open
- https://github.com/ggml-org/llama.cpp/issues/23658 acceptance at KV boundaries, open
- https://github.com/ggml-org/llama.cpp/pull/28136 direct PLE reads, open

**vLLM**
- https://vllm.ai/blog/2026-03-24-mrv2 Model Runner V2, 2026-03-24
- https://vllm.ai/blog/2026-05-26-eagle-3-1 EAGLE 3.1, 2026-05-26
- https://vllm.ai/blog/2026-04-22-fp8-kvcache 2026-04-22
- https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification 2026-08-14
- https://github.com/vllm-project/vllm/pull/19970 async scheduling, merged 2025-07-15
- https://github.com/vllm-project/vllm/pull/24799 async plus spec, merged 2025-11-17
- https://github.com/vllm-project/vllm/pull/14930 Triton rejection sampler, merged 2025-03-18
- https://github.com/vllm-project/vllm/pull/25784 suffix decoding, merged
- https://github.com/vllm-project/vllm/pull/36649 hybrid prefix caching, closed 2026-05-05
- https://github.com/vllm-project/vllm/issues/18327 tree drafting, closed not planned
- https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/ 2025-11-05

**SGLang**
- https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ zero-overhead scheduler
- https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/ breakable graphs, 2026-08-17
- https://www.lmsys.org/blog/2026-07-06-dspark-sglang/ 2026-07-06
- https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ two-batch overlap
- https://github.com/sgl-project/sglang/issues/23705 adaptive spec decoding, open
- https://github.com/sgl-project/sglang/pull/26128 device-resident metadata, merged 2026-05-29
- https://github.com/sgl-project/sglang/discussions/36891 RecoverSSM, 2026-08-28
- https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ 2025-12-03
- https://arxiv.org/abs/2312.07104 RadixAttention
- https://lmsysorg.mintlify.app/docs/advanced_features/piecewise_cuda_graph

**TensorRT-LLM, MLC, Apple Silicon runtimes**
- https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html
- https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm 2024-12-18
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5 2025-11-19
- https://arxiv.org/abs/2607.00501 BaseRT, 2026-07-01 (1.35x decode over MLX, M3/M4 Pro)
- https://arxiv.org/abs/2607.19438 BaseRT M5 neural accelerators, 2026-07-21
- https://github.com/basecompute/baseRT prebuilt engine, Apache-2.0 CLI only
- https://mtplx.com/releases/ , https://mtplx.com/benchmarks/ , https://mtplx.com/how-it-works/
- https://releasebot.io/updates/mtplx
- https://github.com/youssofal/MTPLX Apache-2.0
- https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.2 2026-09-09
- https://github.com/mlc-ai/mlc-llm (no Metal graph-capture or spec-decode design doc found)
- https://github.com/jundot/omlx/releases 0.7.0.dev2, 2026-09-11

**Papers**
- https://arxiv.org/abs/2605.09329 Test-Time Speculation, 2026-05-10
- https://arxiv.org/abs/2411.04975 SuffixDecoding, NeurIPS 2025 Spotlight
- https://arxiv.org/abs/2608.20961 TreeWY for GDN hybrids, 2026-08-21
- https://arxiv.org/abs/2607.16673 SpecLA, 2026-07-18
- https://arxiv.org/abs/2605.01106 component-aware self-speculation on hybrids, 2026-05-01
- https://arxiv.org/abs/2511.02017 TapOut, 2025-11
- https://arxiv.org/abs/2502.17421 LongSpec, ACL 2025
- https://arxiv.org/abs/2512.02337 SpecPV, 2025-12
- https://arxiv.org/abs/2606.00144 BudgetDraft, 2026-06
- https://arxiv.org/abs/2503.01840 EAGLE-3
- https://arxiv.org/abs/2402.12374 Sequoia
- https://arxiv.org/abs/2603.12201 IndexCache cross-layer index reuse, 2026-03-12
- https://github.com/THUDM/IndexCache
- https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html
- https://arxiv.org/abs/2405.16444 CacheBlend
- https://arxiv.org/abs/2605.05699 int4 KV cache on Apple Silicon, 2026-05
- https://arxiv.org/abs/2512.22219 Mirage Persistent Kernel
- https://arxiv.org/abs/2604.07609 Blink, 2026-04
- https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles megakernel, 78% of bandwidth
- https://arxiv.org/abs/2506.20675 utility-driven speculative decoding for MoE
