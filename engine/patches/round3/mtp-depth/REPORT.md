# Confidence-gated MTP draft depth

Workstream `kernels/round3/mtp-depth/`, 2026-09-12. No model loaded, no safetensors opened, peak
70 MB GPU. Ports 8083/8084 untouched.

## 1. oMLX's existing depth controller

`BG` = `omlx/patches/mlx_lm_mtp/batch_generator.py`.

| question | answer | evidence |
|---|---|---|
| is it disabled? | no, it runs on every chain sequence with depth > 1 | `BG:2504-2517` constructs `_DepthController(depth, ...)` |
| what signal? | conditional acceptance per position as a token-domain EMA (`ALPHA=0.08`) plus per-depth whole-cycle wall time as a wall-clock EMA (`TAU_MS=400`, spike-damped) | `BG:1959-1972`, `BG:2027-2058` |
| decision rule | `score(d) = (1 + p1 + p1p2 + ...) / t_est(d)`, argmax with 3% hysteresis, ascending scan so ties go shallow | `BG:2093-2100`, `BG:2169-2178` |
| how often? | every cycle after a `max_depth + 3` warmup sweep; probe bursts of 4 at a duty-bounded ~1 s cadence re-measure the best rival or the stalest depth | `BG:1985-2025` |
| bounds | 0 to `max_depth`; depth 0 is a plain step and a sustained park hands the sequence to the standard decoder | `BG:2110-2129`, `EXIT_MARGIN`/`EXIT_STREAK` at `BG:1897-1911` |
| where does 3 come from? | `max_depth` is `mtp_num_draft_tokens` or the literal default 3 | `omlx/utils/model_loading.py:713-721`, stamped at `.../qwen4_exp/language.py:3060`, read at `BG:1609-1628` |

So 3 is a ceiling, not a fixed choice. From the audit's 1.91 tokens per cycle and
`C(3) = 31.06` ms, depth 2 costs `28.13` ms for `1.87` tokens: **66.5 tok/s against 61.5**. The
controller's own score prefers depth 2 by 8% and is not taking it, so either the baseline is a
mixture rather than a steady depth 3, or `t[2]` is stale-high in the way the class docstring warns
about at `BG:1846-1857`. `OMLX_MTP_DEPTH_TRACE=1` settles that in one run. Depth 4's regression is
consistent: a geometric fit to 1.91 tokens per cycle, `a_j = 0.610 * 0.705^(j-1)`, predicts 61.3
at depth 3 and **56.6 at depth 4** against the measured 56.

## 2. What was built

`patch.py`, two idempotent env-gated installs, loaded by path.

| install | env | timing | replaces |
|---|---|---|---|
| `install_conf_depth()` | `OMLX_MTP_CONF_DEPTH=1` | import time | `BG._chain_next_drafts` (`BG:2303-2434`) and wraps `_DepthController._best` (`BG:2169-2178`) |
| `install_conf_depth_ceiling(model)` | same | after load | raises `_omlx_mtp_depth` (`language.py:3060`) to `OMLX_MTP_CONF_MAX_DEPTH` |

The chain drafts to the ceiling and stops at the first step where the running product of the
drafter's proposal probabilities falls under a floor; `k = state.drafts.shape[0]` then sizes the
verify window, so M varies per cycle.

**Product, not per-step.** Step `j+1` pays when `P_j * p[j] / D_{j+1} > E_j / C(j)`, with `P_j`
the probability the prefix survives. A constant per-step floor `f` implies a product floor `f^j`
that loosens with depth while the true requirement rises (section 3); `breakeven.py` section 6 has
the per-step form losing in five of nine cells.

**Which probability.** The gate reads `lp_2d`, the row the sampler drew from on that step, post
processors and post shortlist renormalisation. Not `_accept_lp_for(...)`: under a stochastic
target that is the temp-0.6/top-20 sharpened acceptance density (`BG:2183-2187`), near-certain for
tokens the head is guessing at. Greedy makes the two identical (`BG:1412-1415`).

**Composition with the shortlist.** Both patches own `_chain_next_drafts`, so this one subsumes
round-2's: under `OMLX_MTP_SHORTLIST_DRAFT=1` it imports round-2's helpers by path and runs steps
`>= FROM_STEP` on the shortlist head, so the gated probability is that head's own softmax.

**Not fighting the controller.** `_best` is the only place a post-warmup depth is chosen. The
wrapper collapses it to 0, leaving the depth-0 escape hatch and `_park_mtp_to_standard` untouched,
or to `max_depth`, meaning the gate picks. Warmup and probe cycles stay fixed-depth
(`ctl._warmup`, `ctl.probe_left`), so `t[]` keeps honest samples and the probes double as a live
A/B. `observe` gets the real `k`.

**Variable M audit.** Nothing assumes a constant window.

| site | file:line | verdict |
|---|---|---|
| verify window | `BG:2934` `k = int(state.drafts.shape[0])` | already per-cycle |
| GDN replay | `mlx_vlm/models/qwen3_5/language.py:1963-2110` | `block_size` is a parameter; `intermediate_states` is captured at the live window |
| PLE snapshot | `.../qwen4_exp/language.py:2739-2772`, restore at `:3171-3200` | window read from `snapshot.input_ids.shape[1]`; capture is skipped when `input_ids.shape[1] == 1`, which is the k=0 park |
| rollback call | `BG:3205-3248` | passes `num_drafts + 1` |
| per-depth stats | `BG:3088-3096` | sized from `state.depth`, which is why the ceiling install must raise the marker before the first cycle |

The one real risk is Metal shape specialisation: M takes six values instead of one, so each new M
pays a first-run warmup.

## 3. Break-even

`C(0)=24.4`, `D_1=1.6`, `D_k=2.93` stock and `1.92` with the shortlist, which the tables use.
Fixed depth, tokens per cycle and tok/s:

| profile | d=1 | d=2 | d=3 | d=4 | d=5 |
|---|---|---|---|---|---|
| code, repetitive `0.78*0.90^j` | 1.78/68.5 | 2.33/83.4 | 2.67/89.6 | 2.87/90.4 | 2.97/88.2 |
| measured, fitted `0.61*0.705^j` | 1.61/61.9 | 1.87/67.1 | 1.95/65.4 | 1.97/62.0 | 1.97/58.5 |
| free prose `0.42*0.78^j` | 1.42/54.6 | 1.56/55.8 | 1.59/53.4 | 1.60/50.4 | 1.60/47.5 |

Gate at ceiling 5, Monte Carlo over Beta confidences (concentration 6) with the profile as mean
and realised acceptance equal to the drawn confidence, so it wins only if confidence predicts
acceptance. Sync bubble 0:

| profile | fix3 | fix4 | p_min .15 | .25 | .35 | adaptive floor | mean k |
|---|---|---|---|---|---|---|---|
| code | 89.6 | 90.4 | 90.0 | 91.1 | 91.1 | 91.2 | 3.63 |
| measured | 65.4 | 62.0 | 65.9 | 66.9 | 67.2 | 67.3 | 2.04 |
| prose | 53.4 | 50.4 | 55.3 | 56.0 | 56.2 | 56.2 | 1.73 |

Against the best fixed depth per profile the gate is worth 1 to 2%. **It earns its keep on mixed
content**, where one depth serves both regimes and the EMA sees only the blend (sync 0.20 ms in):

| stream | fix1 | fix2 | fix3 | fix4 | fix5 | gate p.25 | gate adaptive | mean k | vs best fixed |
|---|---|---|---|---|---|---|---|---|---|
| 25% code | 58.1 | 62.7 | 62.4 | 60.4 | 57.7 | 65.0 | 64.4 | 2.10 | +2.7% |
| 50% code | 61.5 | 69.6 | 71.5 | 70.4 | 67.9 | 73.8 | 73.1 | 2.54 | +2.3% |
| 75% code | 65.0 | 76.5 | 80.5 | 80.4 | 78.0 | 81.8 | 81.7 | 3.07 | +1.5% |

Against the deployed fixed 3 those rows are +4.2%, +3.2% and +1.6%. **The gate wins when the
acceptance profile varies within a stream faster than the 400 ms cost EMA and the 0.08 acceptance
EMA track it, and the drafter's confidence carries that variation.** It does not win on a
homogeneous stream, where a fresh `t[]` lets the controller find the right fixed depth.

`OMLX_MTP_CONF_PMIN` default **0.25**. Break-even floors from the fitted profile are 0.107, 0.277,
0.425, 0.587, 0.789 for steps 1 to 5, so no constant is right everywhere; 0.25 sits between the
step-1 and step-2 floors, where almost every stop happens, and clears round-2's 0.18 threshold. With
`OMLX_MTP_CONF_ADAPT=1` (default) the floor is instead `D_{j+1} * E_j / (p[j] * C(j))` from the
controller's live `t[]` and `p[]`, clamped to `[0.02, 0.90]`; static 0.25 tracks that within 1%
above, so `ADAPT` is insurance against a different cost regime, not a win today.

Sync cost, `bench_sync.py`: a chain of dependent 2.7 ms steps costs **0.22 to 0.28 ms per
intermediate host sync** (bare `mx.eval` plus `.item()` is 0.042 ms) on a contended GPU. The gate
keeps +2.3% on the 50% code stream at 0.25 ms per step and still beats fixed 3 at 0.50 ms.

## 4. Synthetic test

`test_conf_depth.py` registers a fake `batch_generator` under the real module name, so the patch
monkeypatches the same symbol it will in production, and execs the **real** `_DepthController`
class body out of the shipped file. The fake 512-token model has a per-cycle logit gain making
alternating four-cycle blocks sharp (3.0) or flat (0.25). The loop mirrors
`_run_verify_cycle_chain` at ceiling 5, including the one-pass state capture and the rollback to
`accepted + 1`.

| check | result |
|---|---|
| [0] gate installs; wrapped `_best` chooses only {0, 5} over 400 post-warmup cycles; ceiling 3 to 5 | pass |
| [1] greedy stream over 600 tokens identical to fixed depth 3, at p_min 0.15/0.25/0.40, shortlist off and on | **identical 6/6** |
| [2] mean depth 4.02 on sharp cycles against 1.00 on flat, histogram k=0..5 `[0,116,16,21,16,61]` | pass |
| [3] recurrent state at every shared emitted-token count, against fixed depth 3, with M varying | **0 mismatches over 128-137 checkpoints, 4 configs** |
| [3] head cache committed prefix equals the emitted stream every cycle | pass |
| [1] cycles at p_min 0.15/0.25/0.40, shortlist off | 243 / 268 / 300 for the same 600 tokens |
| [4] adaptive floor equals the closed form at five depths | exact |

Without the gain schedule the histograms shift monotonically with p_min: `[0,72,147,23,1,0]`,
`[0,155,106,7,0,0]`, `[0,255,45,0,0,0]` at 0.15/0.25/0.40.

## 5. Workbench plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, two rounds, paired within a
round. Every row carries `OMLX_MTP_SHORTLIST_DRAFT=1 OMLX_MTP_DEPTH_TRACE=1`.

| step | env added | prompts | record | expect |
|---|---|---|---|---|
| 0 | none | agentic suite | the `MTP depth:` lines: `cur`, `p`, `t`, `scores` | settles the section-1 question. If `cur` is 2 and not 3, the 62 tok/s baseline is already a mixture and every delta below shrinks |
| 1 | `OMLX_MTP_CONF_DEPTH=1 OMLX_MTP_CONF_MAX_DEPTH=5 OMLX_MTP_CONF_ADAPT=0`, p_min in {0.15, 0.25, 0.35, 0.50} | A: 200-token completion inside a 400-line file, repetitive edit. B: 2k free prose continuation. C: 25k tool-calling turn. D: alternating A and B in one session, 5 turns | decode tok/s, `accept=A/D`, `tok/cycle`, `depth[...]` from the `MTP[...]` line, plus the `MTP conf-depth:` histogram (`OMLX_MTP_CONF_TRACE_EVERY=64`) | A rises with p_min low (mean k 3.5+), B falls (mean k under 2). D is the row that decides the patch: +2 to 4% over the best single fixed depth, and the histogram must be bimodal across turns, not a single spike |
| 2 | `OMLX_MTP_CONF_ADAPT=1` at p_min 0.25 | same | same, plus `t` from the depth trace | within 1% of the best static p_min from step 1. A large gap means `t[]` is stale, which is a controller problem, not a gate problem |
| 3 | `OMLX_MTP_CONF_PSTEP` in {0.3, 0.5, 0.7}, `OMLX_MTP_CONF_PMIN=0` | same | same | the llama.cpp form. Expect it to lose to the product floor; if it wins, the head's confidence is miscalibrated across depth and the floor schedule needs refitting |
| 4 | `OMLX_MTP_CONF_MAX_DEPTH` in {4, 5, 8} | A and D | mean k, tail of the histogram | mean k should stop growing before the ceiling. If it pins at the ceiling on A the floor is too low for the real head |
| 5 | greedy, all of the above | A to D | full output bytes | **byte-identical to the unpatched run on every prompt and every setting**. This is the acceptance criterion |
| 6 | temperature 0.7, fixed seed | A and B | accepted-length histogram, per-token logprob distribution | not token-identical, but both distributions must match the baseline |
| 7 | `OMLX_MTP_CONF_DEPTH=1`, 50k uncached prefill then 300 decoded | | first-cycle latency, whether depth-4/5 cycles appear in the first 50 cycles | flags Metal shape-warmup cost for the new M values |

## 6. Commands

    cd ~/inference-server/kernels/round3/mtp-depth
    ~/inference-server/kdev/bin/python test_conf_depth.py
    ~/inference-server/kdev/bin/python breakeven.py
    ~/inference-server/kdev/bin/python bench_sync.py --ms 1.9 --reps 15

To enable, import `patch.py` from `prod/bootstrap.py` by path after round-2's:
`install_conf_depth()` at import time, `install_conf_depth_ceiling(model)` once
`VLMBatchedEngine.start` has the model, `OMLX_MTP_CONF_DEPTH=1` in `prod/run-omlx.sh`.

## 7. Limitations

Section 3's rates are a model fitted to two field numbers. Confidence is assumed to track
acceptance: `over=1.25` costs about 1.5 points on code and still beats fixed 3, but a head whose
confidence carries none makes the gate a 0.25 ms per step tax. That bubble came from synthetic
matmuls on a contended GPU, and nothing here has seen the real model. The gate also puts a host
sync inside a previously pipelined dispatch, which a synthetic chain cannot exercise against
`mx.async_eval` scheduling.
