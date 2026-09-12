# MTP decode: draft shortlisting, verify fusion, depth policy

Workstream `kernels/round2/mtp/`, 2026-09-12. No model loaded, no safetensors opened, peak 1.4 GB
GPU. The live daemon held the GPU throughout, so absolute times run 1.3 to 1.7x the audit's quiet
numbers; ratios within a run are reliable and every projection uses the audit's quiet figures.

## 1. The qwen4_exp MTP path

`LANG` = `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`,
`BG` = `omlx/patches/mlx_lm_mtp/batch_generator.py`.

| stage | file:line | what runs |
|---|---|---|
| depth | `omlx/utils/model_loading.py:721` | `set_mtp_depth(3)`, adaptive 0..3 after |
| chain markers | `LANG:3068-3075` | `_omlx_mtp_chain`, `_omlx_mtp_head_prenorm` |
| draft chain | `BG:2303-2434` `_chain_next_drafts` | 1 fold `mtp_forward(logits_keep=1)` + `depth-1` chained calls |
| draft head | `LANG:3101-3126` | 1 decoder layer (QSA + 512-expert MoE), then `self.lm_head(...)`, all 248320 rows |
| draft sampling | `BG:2385-2412`, `:2183-2187` | greedy target gives greedy drafts, else temp 0.6 / top_p 0.95 / top_k 20 |
| verify | `BG:2908-3010` | one backbone forward on `[next_main, d1..dk]`, **M = k+1 = 4** |
| verify MoE | `LANG:2841` -> `qwen3_5_moe/language.py:15-32` -> `qwen35_moe_gate_up.py:146-177` | rebound to the fused `gate_up_proj` |
| verify lm_head | `qwen35_verify_qmm.py:418-431` | oMLX `vk_qmm` msg kernel, M=3..6 and N>=16384, so head only |
| acceptance | `BG:2978-2990` greedy, `:2992-3020` stochastic | in-graph cumprod, one host sync per cycle |
| rollback | `LANG:3202+`, `BG:3221-3260` | GDN state replay plus PLE snapshot |
| depth policy | `BG:1823-2182` `_DepthController` | already adaptive: EMA acceptance, measured per-depth cost, 3% hysteresis |

**Four vocabulary passes per cycle: confirmed and measured.** `test_shortlist.py` check [0] counts
3 full-vocabulary `lm_head` calls in the real `_chain_next_drafts` at depth 3 (fold plus 2 chain
steps; the last samples and breaks at `:2417`), plus 1 in verify.

**gate_up leak: refuted.** `engine/vlm.py:1978-1991` runs the fusion post-load, which calls
`_ensure_vlm_verify_patch` and rebinds the module global that
`Qwen3_5MoeSparseMoeBlock.__call__` resolves at call time. qwen4_exp imports the class
(`LANG:32`), so the rebind reaches it. `test_verify_gate_up.py`: fused output bit-identical at M=4
(max abs 0.000e+00), worth **1.03x** on the verify MoE block. At M=4 with top-10 at most 40 experts
are touched either way, so the fusion saves one launch of three per layer and zero bytes. The
audit's "+2-4% decode" for this item is roughly 10x too high.

## 2. mlx-vlm 0.7.0 vs bundled 0.6.3

0.6.3 ships no `qwen4_exp`, so the comparison is against oMLX's vendored copy. Wheel at
`wheels/mlx_vlm-0.7.0-py3-none-any.whl`, unpacked under `wheels/unpacked/`.

| change | 0.7.0 location | judgement for speed |
|---|---|---|
| `_target_verify_switch_glu` deleted, MoE verify calls `switch_mlp(x, inds)` | `qwen3_5_moe/language.py` | irrelevant: upstream closed the same leak, oMLX already fused |
| SwitchGLU moved into `mlx_vlm.models.switch_layers` | `qwen3_5_moe/language.py:11` | blocker for a version bump (breaks `qwen35_moe_gate_up.py`), no speed content |
| quantized verify QMV, `RESULTS_PER_SIMDGROUP=4`, token-tiled and streamed | `qwen3_5/speculative_verifier.py:135-1000` | portable as a monkeypatch, loses to `vk_qmm` (section 4) |
| fused quantized matvec + argmax | `qwen3_5/speculative_verifier.py:1003-1078` | portable, loses at M<=4 (section 4) |
| dense bf16 verify GEMV | `models/exact_speculative_verify.py` | irrelevant, unquantized weights only |
| standalone `Qwen4ExpMTPDraftModel` | `speculative/drafters/qwen4_exp_mtp/` | whole module, and worse: rollback replays the accepted prefix one token at a time (`qwen4_exp/language.py:2077-2081`) |
| adaptive draft depth, `_effective_mtp_block_size` | `speculative/mtp.py:449-495` | irrelevant, a 32-cycle hit-rate ceiling against oMLX's richer controller |
| uniform batch acceptance, batched parity | `speculative/common.py`, `mtp.py:377-390` | irrelevant single-stream |
| external PLE storage, n-gram shard gathers | `qwen4_exp/ple_storage.py` | whole module, collides with the deployed packed-rows patch, do not merge |
| QSA cache reuse | `qwen4_exp/qsa_kernel.py` | whole module, not evaluated |

Nothing in 0.7.0 is worth porting for decode speed here, and the SwitchGLU move plus the PLE
rewrite make a straight bump expensive. The research doc's "decode +10 to 25%" has no support in
the code.

## 3. What was built

`patch.py`: three env-gated installs, idempotent, each returns False and leaves the stock path when
preconditions fail. Load by path with `importlib.util.spec_from_file_location`.

| install | env var | timing | replaces |
|---|---|---|---|
| `install_shortlist_draft()` | `OMLX_MTP_SHORTLIST_DRAFT=1` | **import time**, module function swap | `BG._chain_next_drafts` (`:2303-2434`) |
| `install_verify_gate_up(model)` | `OMLX_MTP_VERIFY_GATE_UP=1` | **after load**, needs the model | `qwen3_5_moe.language._target_verify_switch_glu`, only if oMLX did not |
| `install_depth_trace()` | `OMLX_MTP_DEPTH_TRACE=1` | import time | read-only wrapper on `_DepthController.observe` |

Knobs: `OMLX_MTP_SHORTLIST_K` (2048), `OMLX_MTP_SHORTLIST_FROM_STEP` (1, keeping the first draft on
the full vocabulary), `OMLX_MTP_SHORTLIST_REFRESH` (1 cycle), `OMLX_MTP_DEPTH_TRACE_EVERY` (64).
Rebuilding the shortlist needs a full step-0 pass, so `FROM_STEP=0` only differs from `1` once
`REFRESH >= 2`.

**(a) Shortlist drafter.** Step 1 keeps its full pass and its top-K becomes the shortlist. Steps
2..depth run the MTP head without its vocabulary projection, then `quantized_matmul` against the K
gathered head rows, and scatter the log-probs into a full-width row that is `-inf` elsewhere. The
scatter keeps the patch to one function: that row is the true proposal density of a drafter which
can only emit shortlist tokens, so greedy comparison, the Leviathan/Chen ratio and the residual
`max(p-q,0)` stay correct with `_run_verify_cycle_chain` untouched.

| `test_shortlist.py` check | result |
|---|---|
| [0] full-vocabulary passes per cycle | 3+1 -> **1+1** |
| [1] argmax reproduced whenever it is in the shortlist | 200/200 |
| [2] first 400 greedy tokens identical to stock, K=32/64/128 | **identical 3/3** |
| [2b] identical at step correlation 0.0 / 0.5 / 0.8 / 0.95 | identical 4/4 |
| [2b] acceptance 66.7 -> 34.0 / 66.2 -> 59.1 / 2.1 -> 2.0 / 41.6 -> 41.6 % | correlation-dependent |
| [3] off-list max -3.0e38, on-list logsumexp -2.4e-7, mass 1.000000 | pass |

The toy's head is close to random, so its zero-correlation column is a worst case with no bearing
on a real LM.

**(b) gate_up.** Not leaking, so the install returns False on a stock server and logs the
fused/unfused counts. It stays as a guard: with `moe_gate_up_fusion_enabled` off, or MoE expert
offload active (`engine/vlm.py:1971-1977` skips fusion then), verify silently returns to 3 gathers.

**(c) Adaptive depth.** Already shipped. `_DepthController` scores
`(1 + sum_j prod_{i<=j} p_i) / t[d]` over measured per-depth cycle times, with EMA acceptance, 3%
hysteresis, staleness probes and a hand-off gate. That score is the break-even, so this workstream
adds instrumentation only. With `p_k` the probability that the first k drafts all pass,
`S_k = sum_{j<=k} p_j`, `R(k) = (1+S_k)/C(k)`, `C(k) = C(k-1) + D_k`:

    R(k) > R(k-1)  <=>  p_k * C(k-1) > D_k * (1 + S_{k-1})  <=>  p_k > D_k * R(k-1)

Depth k pays exactly when cumulative acceptance through k beats k's marginal cost times the current
tokens per millisecond. Calibration from the audit:

| quantity | value | source |
|---|---|---|
| one-row forward | 24.4 ms | 41 tok/s with MTP off |
| C(3) | 31.06 ms | 61.5 tok/s at ~1.91 tokens/cycle |
| draft chain, depth 3 | ~4.0 ms | 3 x (0.93 lm_head + ~0.4 head layer) |
| one extra verify row | ~1.6 ms | gather 0.151 vs 0.061 ms/layer x 48 over 3 rows; lm_head 1.51 vs 0.93 |
| `D_1` | 1.6 ms | verify row only (depth 1 reuses the fold's logits) |
| `D_2 = D_3` | 2.93 ms | verify row + draft head forward + lm_head |

At `a1 = 0.90, a2 = 0.80`: `C(2) = 28.13` ms, `R(2) = 2.62/28.13 = 0.0931` tok/ms, so depth 3 pays
while `p_3 > 2.93 x 0.0931 = 0.273`, i.e. **conditional acceptance at depth 3 above 38%**. MTPLX
reports 88% at D3 on a smaller Qwen and vLLM ~36% here, so depth 3 sits on the line. The shortlist
cuts `D_2 = D_3` to ~1.92 ms, dropping the threshold to 24% and making depth 4 worth one sweep.

Not patched: `_marginal_est` (`BG:2058-2071`) fits one slope across measured depths, but `D_1` is
1.6 against 2.93 ms and `t[0]` carries no verify row. At `max_depth=3` warmup measures every depth
so `_t_est` never uses the slope; it bites at depth 4.

**(d) Small-M verify kernel: nothing shipped.** `bench.py`, 15 iterations, CHAIN=10,
`mx.synchronize` per sample. "stock" is `mx.quantized_matmul`, "vk" oMLX's `vk_qmm` (live for the
head), "v070" `_target_verify_quantized_linear` lifted from the 0.7.0 wheel.

lm_head 2560 -> 248320 at M=4, milliseconds:

| bits | stock | vk (production) | v070 | v070 argmax vs stock+argmax |
|---|---|---|---|---|
| 4 | 1.064 | **0.639** | 1.005 | 0.83x |
| 8 (3 repeats) | 1.848 / 2.656 / 2.696 | **1.385 / 1.771 / 2.042** | 1.933 / 2.640 / 2.901 | 0.78 to 1.05x |

Dense projections at M=4, 4-bit, forced past the N>=16384 gate (stock / vk / v070): q_proj 0.099 /
0.113 / 0.097; o_proj 0.103 / 0.108 / 0.086; shared 640 0.050 / 0.043 / 0.051. All launch-bound at
19 to 104 GB/s, nothing wins beyond noise.

At M=1 the 4-bit head runs at 809 GB/s, essentially the read ceiling, so no kernel helps a pass that
is pure weight streaming. At M=4 it falls to 338 GB/s because cost becomes M fp32 FMAs per weight
element (`kernels/REPORT.md` finding 4), and `vk_qmm` recovers most of that: 0.639 against a 0.443
ms floor, leaving 0.196 ms, 0.6% of a cycle. `kernels/smallm/` lost in situ because its edge was at
M=8, which oMLX never presents; mine would lose because at M=4 there is no edge to start from.
Better target: `mx.topk` over 248320 costs 0.267 to 0.383 ms against `mx.argmax`'s 0.027 ms on the
same array, so a fused top-K saves ~0.25 ms per shortlist cycle, ~0.8% of decode.

## 4. Expected gain per part

C(3) = 31.06 ms, 1.91 tokens/cycle, 8-bit head at 0.930 ms per M=1 pass. Shortlist overhead
measured at the real head shape: top-K 0.267 + row gather 0.081 + 2 x (small qmm 0.039 + scatter
0.091) = 0.608 ms per cycle (`bench_shortlist.py`).

| part | arithmetic | per cycle | C(3) | tok/s | delta |
|---|---|---|---|---|---|
| baseline | | | 31.06 | 61.5 | |
| (a) shortlist FROM_STEP=1 | 2 x 0.930 - 0.608 | **-1.25 ms** | 29.81 | **64.1** | **+4.2%** |
| (a) shortlist FROM_STEP=0 | 3 x 0.930 - 0.738 | -2.05 ms | 29.01 | 65.8 | +7.0% if tokens/cycle falls under 2.7% |
| (b) gate_up | already fused | 0 | 31.06 | 61.5 | 0% |
| (c) adaptive depth | already shipped | 0 | 31.06 | 61.5 | 0% |
| (d) verify kernel | `vk_qmm` unbeaten at M=4 | 0 | 31.06 | 61.5 | 0% |
| future fused top-K | -0.25 ms | -0.25 ms | 29.56 | 64.6 | +0.8% on top of (a) |

**The case rests on the head being 8-bit.** The audit measured 0.930 ms at M=1 and says the
checkpoint ships 8-bit; `COMMON.md` still records 4-bit at 0.443 ms, where the shortlist saves
2 x 0.443 - 0.608 = 0.28 ms, so +0.9% rather than +4.2%. Settle that first. Per 2048-token prefill
chunk: no change, nothing here runs in prefill.

## 5. Workbench test plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, two rounds each, patched and
unpatched paired within a round.

| step | change | prompts | measure | expect |
|---|---|---|---|---|
| 0 | none | n/a | log `lm_head.bits`, time 200 calls at `[1,1,2560]` | 8 bits and ~0.93 ms: continue. 4 bits and ~0.45 ms: stop, the patch is worth under 1% |
| 1 | `OMLX_MTP_DEPTH_TRACE=1`, `..._EVERY=64` | agentic suite | the `MTP depth:` lines: `p`, `t`, scores | `p` near [0.9, 0.75, 0.6], `t[3]-t[2]` near 2.9 ms. If depth 3 is not the argmax score, the default is already wrong |
| 2 | `OMLX_MTP_SHORTLIST_DRAFT=0` vs `1`, K in {512, 2048, 4096} | 200-token code completion; 2k prose continuation; 25k tool-calling turn; 50k uncached prefill then 300 decoded | decode tok/s, plus `accept=A/D`, `depth[d1,d2,d3]` and `tok/cycle` from the `MTP[...]` line | 61.5 -> 63.5 to 64.5 tok/s. d2/d3 acceptance down at most 2 to 3 points (75/60% to 73/57%), code least, prose most. `tok/cycle` within 2%; below -3% raise K. Greedy output **byte-identical** across all prompts and all K, which is the acceptance criterion |
| 3 | `FROM_STEP=0`, then `REFRESH=2`, then `set_mtp_depth(4)` | as step 2 | same | each trades acceptance for cycle time; depth 4's threshold is ~20% conditional acceptance once the shortlist is on |
| 4 | temperature 0.7, fixed seed | as step 2 | accepted-length histogram, per-token logprob distribution | not token-identical, but both distributions must match baseline; a systematic shift means the scattered row is not reaching `q` |

## 6. Commands

    cd ~/inference-server/kernels/round2/mtp
    ~/inference-server/kdev/bin/python test_shortlist.py
    ~/inference-server/kdev/bin/python test_verify_gate_up.py
    ~/inference-server/kdev/bin/python bench.py --bits 8 --ms 1,2,4,6,8 --only lm_head
    ~/inference-server/kdev/bin/python bench.py --bits 4 --ms 4 --force-vk
    ~/inference-server/kdev/bin/python bench_shortlist.py --bits 8

To enable, import `patch.py` from `prod/bootstrap.py` by path: `install_shortlist_draft()` and
`install_depth_trace()` at import time, `install_verify_gate_up(model)` once
`VLMBatchedEngine.start` has the model, and `OMLX_MTP_SHORTLIST_DRAFT=1` in `prod/run-omlx.sh`.

## 7. Limitations

The shortlist has never seen the real model: output identity is proved, acceptance is a projection
from a near-random toy head whose bracket (-33 to 0 points) is too wide to act on. The `-inf`
scatter costs 0.091 ms and two launches per draft step, 15% of the saving. Benches ran on a
contended GPU. The 0.7.0 comparison is a code diff; the wheel was never installed.
