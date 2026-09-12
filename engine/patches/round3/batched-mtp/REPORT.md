# Fused batched MTP verify: one forward for B concurrent requests

Workstream `kernels/round3/batched-mtp/`, 2026-09-12. No model loaded, no safetensors opened, peak
0.2 GB GPU, ports 8083/8084 untouched. Shipped option **(a)**, batched verify.

## 1. How a concurrent request loses MTP today

Flash-Next never takes the external-drafter `vlm_mtp` path: `_vlm_mtp_drafter` is None here, and
"scheduler contention" and "drafter is busy" appear zero times in the production log. The live path
is oMLX's own MTP head chain in `omlx/patches/mlx_lm_mtp/batch_generator.py`, the one that logs
`MTP[n]`; its cycle is mapped in `round2/mtp/REPORT.md` section 1 and unchanged here. Every refusal,
and what kind it is:

| refusal | file:line | correctness or heuristic |
|---|---|---|
| singleton only: `len(uids) != 1` | `BG:556-578` `_is_mtp_eligible` | heuristic; the row-wise path below exists for the rest |
| row-wise batch MTP is opt-in | `BG:432-451`, `:600-604` | **heuristic**, and its own docstring gives the measurement: 53.3 / 52.5 tok/s aggregate at batch 2 / 4 against 65.2 / 86.5 for plain batched decode |
| late arrival pinned out | `BG:271-296` `patched_bg_next` forces `completion_batch_size = 0` while `_generation_batch_has_active_mtp` | heuristic; `patched_extend` (`BG:219-247`) already reconciles MTP state to standard before any merge |
| multirow marker locks a row out after a shared step | `BG:452-466`, `:468-485` | correctness for *singleton* re-activation; the row-wise path is explicitly exempt |
| grammar processors | `BG:398-431` | correctness |
| `vlm_mtp` "drafter is busy" | `scheduler.py:9096-9107` | correctness: the drafter keeps per-request state on the module instance |
| `vlm_mtp` "scheduler contention" | `scheduler.py:9113-9130` | heuristic, and the comment says so ("Prefer ordinary batching") |

Per-sequence state is already clean. `_MtpState` (`BG:692-766`) holds the emit queue, the MTP head
cache, `next_main`, drafts, `hist_offset`, the sampler and a `_DepthController`, one per uid inside
`_MtpBatchState`; only the backbone cache, the model and the module functions are shared. Rollback
is batch-aware end to end: `qwen4_exp:3202` normalises a per-row accepted list, `_restore_ple_state`
(`:3171-3200`) gathers per-row retained positions, and `qwen3_5:1963-2110` right-trims rows by
different amounts through `prepare(right_padding=...)` / `finalize()`.

The stock row-wise path is slow for a different reason: `_mtp_batch_next` (`BG:2590-2629`) runs
**one backbone forward per row per cycle** and calls `gen_batch.extract_cache(idx)` plus
`_merge_row_caches` every cycle, copying each row's whole KV cache out and back
(`QSAKVCache.extract` at `LANG:527` does `mx.contiguous` over `[..., :offset, :]`).

## 2. What was built

`patch.py`, two idempotent env-gated installs, both **import time**, loaded by path.

| install | env | swaps |
|---|---|---|
| `install_batched_mtp()` | `OMLX_MTP_BATCHED=1` | `_mtp_batch_next` -> fused verify; `_rowwise_batch_mtp_enabled` -> True; `_generation_batch_has_active_mtp` -> False (item 3) |
| `install_batched_mtp_scheduler()` | same | wraps `Scheduler._route_to_vlm_mtp`, hiding `waiting`/`running`/`prefilling` for the call so the contention preference cannot fire. `OMLX_MTP_BATCHED_MULTI_DRAFTER=1` also hides `_vlm_mtp_active`, which is unsafe on a shared drafter and stays off |

The fused cycle: `(B, k+1)` inputs from every row's `next_main` and drafts at `k = min_b k_b`; one
`_call_backbone` against `gen_batch.prompt_cache`, no extract and no merge; greedy acceptance for
all rows in graph (`argmax`, `cumprod`, one `tolist`); one `rollback_speculative_cache` with the
per-row accepted list; then per row the stock `_chain_next_drafts` on a `_make_row_batch` view, so
`copy-lane` and `mtp-depth` still compose. Knobs: `MAX_B` 4, `MAX_ROWS` 16, `MAX_CTX` 16384,
`QUEUE_CAP` 8, `TRACE_EVERY`. Anything outside the envelope, plus stochastic sampling, logits
processors, commit alignment and `mtp_clamp_accept`, falls back to the stock loop.

**The constraint that shapes the design.** The batch cache has one row per sequence, so a forward
covers all B rows or none: a subset cannot be advanced. Rows drain their emit queue at one token per
`next()` call but refill `m+1` at a time, so the first cycle where acceptance differs desynchronises
them permanently. The fused path therefore steps every row on every cycle, a row that accepts more
than its peers banks an emit queue, and `QUEUE_CAP` bounds that queue by clamping its accepted count
(always legal, the same lever the stock cycle uses for boundary alignment). A cycle runs only when
some row is dry; otherwise the call just emits.

## 3. Synthetic test

`test_batched.py` registers a miniature `batch_generator` under the real module name and execs
eleven blocks verbatim out of the shipped file (`_MtpState`, `_DepthController`,
`_chain_next_drafts`, `_run_verify_cycle_chain`, `_emit_batch_responses` and the rest), so the
reference stream comes from oMLX's real singleton cycle. The 256-token toy carries a recurrent
accumulator (GDN) and a rolling history window (PLE) in its cache, restored per row by a
batch-aware `rollback_speculative_cache`; each row has its own drafter skill, so acceptance differs
within a cycle.

| check | result |
|---|---|
| [0] three module functions swapped, late-arrival pin lifted, idempotent | pass |
| [1] B=2 token streams identical to the alone-run, skills 95 / 60 % | **identical, 40 tokens each** |
| [1] B=4 identical, skills 95 / 60 / 30 / 80 % | **identical, 40 tokens each** |
| [1] every fused forward carries all B rows at M = k+1 = 4 | pass |
| [2] per-row cache offsets diverge every cycle (ragged accepts) | 100 % of cycles |
| [2] GDN accumulator and PLE window exact at every shared commit point | 6 to 28 points per row, 0 mismatches |
| [3] MTP head cache exact at every shared `hist_offset` | 0 mismatches |
| [3] head caches distinct, depth histograms monotone, stats self-consistent | pass |
| [4] non-greedy, B over `MAX_B`, and the context guard all fall back | pass |
| [5] ragged per-row depth collapses to k=min and still matches | pass |
| forwards for the same token count, B=2 | 26 row-wise -> **16 fused, 1.62x fewer** |
| forwards for the same token count, B=4 | 68 row-wise -> **28 fused, 2.43x fewer** |

## 4. Expected gain

`simulate.py` prices a forward from measured components only: shared weights solved so
T(1 row) = 24.4 ms (the MTP-off step), the expert gather at 0.061 ms per layer per row scaled by the
measured 300 / 490 / 549 GB/s at 1 / 4 / 8 rows, QSA per sequence from `qsa-verify` section 2, the
2.3 ms host n-gram lookup, and the 8-bit head at 0.93 / 1.51 / 4.23 / 1.70 ms at M = 1 / 4 / 8 / 16.
Tokens per cycle come from a Monte Carlo of the lockstep queue rule over round-2's fitted acceptance
profile. Anchors: plain batched 74.5 (measured 82) at B=2, 130.2 (measured 116) at B=4,
single-stream MTP 60.2 (measured 62 to 79).

Short context, measured acceptance profile:

| B | k | tok/cycle | cycle ms | fused aggregate | per stream | plain batched | today |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 3.19 | 31.9 | **100.0** | 50.0 | 74.5 | 82 |
| 2 | 3 | 3.78 | 44.4 | 85.2 | 42.6 | 74.5 | 82 |
| 4 | 1 | 6.31 | 43.2 | **146.1** | 36.5 | 130.2 | 116 |
| 4 | 3 | 7.33 | 61.7 | 118.8 | 29.7 | 130.2 | 116 |
| 8 | 1 | 12.50 | 59.4 | 210.5 | 26.3 | 195.7 | |
| 8 | 3 | 14.27 | 103.2 | 138.3 | 17.3 | 195.7 | |

Against the plain-batched column that is **1.34x at B=2, 1.12x at B=4, 1.08x at B=8**, all at draft
depth 1, not 3. Depth 3 loses because 8 rows land in `affine_qmv_wide` (4.23 ms against 1.51 at
4 rows) and 16 rows triple the gather. On copy-lane's repetitive profile the same table gives
1.63x / 1.30x / 1.20x with depth 2 winning at B=2 and B=4. On free prose it is 1.18x at B=2 and a
wash at B=4.

Two results decide deployment. Heterogeneous rows are where lockstep hurts: one code row plus one
prose row gives 3.19 tokens per cycle where the rows alone would give 4.27, and the fused run loses
to plain batching (71.9 against 74.5). And long context is fatal unless the sparse arm survives.
`BatchQSAKVCache` fails the strict `type(c) is QSAKVCache` test at `LANG:1292, 1327, 1376`, and a
fused batch is also on rank-three mRoPE ids because `set_step_rope_deltas` (`models/vlm.py:303`)
keeps Qwen4's rank-two positions only at `len(uids) == 1`. At 65k the dense arm gives 61.5 at B=4
k=3 against 98 for plain batched; the sparse arm preserved gives 128.2 at B=4 k=1 against 113.6.
Hence `MAX_CTX` 16384. The per-sequence sparse loop inside the attention layer is designed but not
shipped: per-row slicing of a left-padded batched cache needs its own exactness proof.

## 5. Workbench plan

`~/inference-server/staging/concurrency_sweep.py`, port 8084 only (it refuses 8083), streams
1 2 4 8, 300 tokens, two rounds, 45 s cooldown, paired inside a round. It scrapes the per-request
`MTP[n]` line for `tok/cycle`, `accept=A/D` and `depth[...]` per stream, plus the new
`MTP batched:` line under `OMLX_MTP_BATCHED_TRACE_EVERY=64`.

| step | env | expect / stop |
|---|---|---|
| 0 | none | reproduce 79 / 82 / 116 at 1 / 2 / 4. If B=2 is not flat, the premise moved |
| 1 | `OMLX_MTP_BATCHED=1`, depth 3 | the `MTP batched:` line appears with `mean_B` at the stream count. Greedy output **byte-identical** to step 0; a diff is a bug, stop |
| 2 | `OMLX_MTP_CONF_MAX_DEPTH` 1 / 2 / 3 | depth 1 should win at B=4 and depth 2 at B=2. If depth 3 wins, the head cliff at 8 rows is not real in situ |
| 3 | `OMLX_MTP_BATCHED_QUEUE_CAP` 2 / 8 / 32 | cap 2 approximates uniform acceptance, cap 32 lets rows drift. Watch per-stream fairness, not just aggregate |
| 4 | 65k prompts, `MAX_CTX` 16384 then 0 | the dense-arm cliff. Expect a large regression with the guard off, which sizes the per-sequence sparse loop |
| 5 | mixed traffic: one code stream, one prose stream | the heterogeneous case above. If it regresses, gate on acceptance spread |
| 6 | late arrival: start one stream, add a second after 100 tokens | the second must get MTP, not `BatchGenerator`. Check the log for a second `MTP[n]` line |

## 6. Commands

    export OMLX_MTP_BATCHED=1
    OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft,\
    $R3/batched-mtp/patch.py:install"

    cd ~/inference-server/kernels/round3/batched-mtp
    ~/inference-server/kdev/bin/python test_batched.py
    ~/inference-server/kdev/bin/python simulate.py
    python3 ~/inference-server/staging/concurrency_sweep.py --label batched \
        --streams 1 2 4 8 --out sweep-batched.json

## 7. Limitations

Nothing here has seen the model. Section 4 is a cost model calibrated on five measured numbers, and
it brackets plain batched decode at 74.5 against a measured 82 at B=2 and 130 against 116 at B=4, so
trust the ratios rather than the absolutes. Acceptance is drawn i.i.d. per row per cycle; real rows
are autocorrelated, which makes the lockstep collapse worse than the table says. The fused path is
greedy only, so temperature traffic keeps the stock loop. The queue cap lets a row over-run a stop
token by up to `cap - 1` tokens beyond MTP's existing depth over-run, poisoning GDN state for a
follow-up turn the way copy-lane described. Option (b) was not shipped separately: oMLX already has
round-robin row-wise MTP behind `OMLX_MTP_ROWWISE_BATCH=1` with a measurement saying it loses, and
the fused verify is that same path with the per-row forwards and the per-cycle cache copies
removed.
