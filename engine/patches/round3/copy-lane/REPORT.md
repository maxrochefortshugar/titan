# Prompt-lookup copy lane beside MTP (qwen4_exp)

`kernels/round3/copy-lane/`, 2026-09-12. No model loaded, peak 0.4 GB GPU, ports 8083/8084
untouched. Source reading plus synthetic tests only.

## 1. Hook

`install_copy_lane()` under `OMLX_MTP_COPY_LANE=1`, at import time, wraps the module function
`_chain_next_drafts` (`batch_generator.py:2303-2434`) and delegates to whatever was bound there
before, composing with the round-2 shortlist drafter and with `mtp-head8` / `mtp-depth`. Install
**last**, outermost, so a miss falls back to the MTP chain (test [10]).

The verify pass needed no patch: it reads `k = int(state.drafts.shape[0])` per cycle (`BG:2930`), so
a copy block is verified, clamped, rolled back and rewound by stock code, and the lane swaps only
`state.drafts` / `draft_lps` / `draft_accept_lps`. It **replaces** the MTP chain, because
`inputs = concat([next_main, drafts])` (`BG:2933`) is one linear chain: one hypothesis per cycle.

`gen_batch.tokens[idx]` "always represents the tokens contained in the KV cache"
(`mlx_lm/generate.py:1372`) and `_num_tokens[idx]` counts generated tokens only, so
`prompt_len = len(tokens) - _num_tokens` is the exact prompt slice, taken once per request.
`SOURCE=context` is refused. Under sampling q is a real one-hot row, the standard Leviathan/Chen
case (test [7]).

## 2. The M cap: 14, not 24

| component | envelope | file:line | above it |
|---|---|---|---|
| `vk_qmm` lm_head | `3 <= M <= 6` | `qwen35_verify_qmm.py:421-430` | stock qmm, correct, slower |
| verify SDPA split | `q_len*gqa <= 32`, rows <= 5 or 4 | `qwen35_verify_sdpa_split.py:10,38` | composed unfused path |
| TurboQuant multi-row | `L <= 15` | `turboquant_attention.py:30` | **prefill fallbacks that re-dequantize the whole KV cache with per-chunk syncs** |
| TurboQuant fold | `n_repeats*L <= 24` | `turboquant_attention.py:34,362` | quantized multi-row decode |
| qwen4_exp gathered QSA | needs `>= 16` | `qwen4_exp/language.py:95,1287` | excluded on `target_verify` |

`L <= 15` is the cliff, so `M = block + 1 <= 15` and `MAX` defaults to 14, clamped by `M_LIMIT`.
None is a correctness limit, which is why the cap is a knob. Rollback itself is window-generic
(`qwen4_exp:2767,3159`, `qwen3_5/language.py:2114`).

## 3. Hazards, each tested

**(a) Stop inside a block.** The block is cut so its last token is the first stop, found by
dry-running the pure `state_machine.match` (multi-token stops) and checking EOS ids. Emission halts
there, but the backbone consumed the whole window and a finished request hands its cache out for
prefix reuse (`BG:2668`); GDN state is recurrent and untrimmable, so the next turn inherits the
over-run: at most 3 for stock MTP, 14 for an untruncated block. Test [4]: over-run 0, 5 dropped.

**(b) Partial accept.** No new code; `_chain_rollback` (`BG:3221`) runs unchanged. The fake cache
carries a recurrent accumulator and a PLE-style window snapshot, checked every cycle against the
emitted prefix. Test [5]: 101 partial accepts, all exact.

**(c) MTP head input.** oMLX caches no MTP hidden across cycles: verify passes `hidden[:, :m+1]` and
`committed` from its own forward (`BG:3125`), so the head always folds the target's state at the new
position. The lane still runs the fold, leaving head history byte-for-byte stock, and skips only the
chain steps. MLX is lazy and nothing consumes the fold's logits, so its `lm_head` never evaluates: a
copy cycle pays one head layer, ~0.4 ms. Test [6].

**(d) Merges.** Keys are id tuples, blocks are slices of the prompt id list. Test [8]: a re-split
tail does not match a merged id.

**(e) Found by the tests.** The verify histogram loop breaks at the first rejection (`BG:3086`),
touching `depth_drafted[0..m]` not `[0..k-1]`, so the lane's undo must mirror it or counts go
negative. And `observe` clamps `used` to `max_depth` (`BG:1955`), so a 14-row block books as a
full-accept depth-3 cycle at double the wall time, corrupting `p`, `t` and `mtp-depth`'s policy. So
the lane skips `observe` on copy cycles, and never speculates during warmup. Test [9].

## 4. Index

One C-level pass per order over zipped shifted slices into a dict from id tuple to following
position, last occurrence winning; lookup is one dict get per order, descending `NGRAM` to
`NGRAM_MIN`. `MAX_PROMPT` (65536) bounds the indexed window.

| prompt tokens | build ms | lookup us | table MB |
|---|---|---|---|
| 4 096 | 0.93 | 0.152 | 0.73 |
| 16 384 | 2.58 | 0.152 | 2.9 |
| 65 536 | 12.3 | 0.129 | 12.1 |

## 5. Synthetic test

`test_copy.py` registers a miniature `batch_generator` under the real module name, so `patch.py`
patches and drives the same entry point it patches in production; its cycle loop transcribes
`_run_verify_cycle_chain`'s accept, clamp, emit, rollback and queue-drain order. Fake model: greedy
argmax at t is `intended[t]`, MTP head hits at 0.72. Prompt: a 600-token code file. Edit: that file
with docstrings inserted. Prose: unrelated ids.

| regime | N | adapt | cycles | hit % | copy cycles | mean block | mean accept | tok/cycle | ms/tok | tok/s | vs stock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edit | - | - | 152 | 0.0 | 0 | 0.00 | 0.00 | 2.63 | 12.18 | 82.1 | 1.00x |
| edit | 4 | on | 139 | 45.0 | 63 | 5.17 | 2.33 | 2.88 | 11.26 | 88.8 | 1.08x |
| edit | 4 | off | 135 | 43.4 | 59 | 13.76 | 2.56 | 2.96 | 12.96 | 77.2 | **0.94x** |
| edit | 6 | on | 148 | 34.9 | 52 | 4.67 | 2.12 | 2.70 | 11.86 | 84.3 | 1.03x |
| edit | 6 | off | 126 | 23.6 | 30 | 14.00 | 4.40 | 3.17 | 11.20 | 89.3 | 1.09x |
| edit | 8 | on | 125 | 15.1 | 19 | 9.58 | 5.79 | 3.20 | 10.38 | 96.4 | **1.17x** |
| edit | 8 | off | 125 | 15.1 | 19 | 14.00 | 5.79 | 3.20 | 10.71 | 93.3 | 1.14x |
| prose | 4/6/8 | on/off | 152 | 0.0 | 0 | 0.00 | 0.00 | 2.63 | 12.18 | 82.1 | 1.00x |

`ms/tok` prices each cycle with round-2's costs. Greedy output was identical to stock in every row;
prose never fires, and a miss costs one dict lookup. The N=4 row is why the default is 8: short
matches raise the hit rate and cut the accepted length, which at 1.6 ms per row loses money.
Adaptive sizing (`k = clamp(1.5*EMA + 2)`) rescues N=4 and costs N=8 three points, so it stays on.
These ratios are my corpus, not the model.

## 6. Expected gain

Round-2: one-row forward 24.4 ms, `C(3) = 31.06` ms at 1.91 tokens/cycle (61.5 tok/s, 16.3
ms/token), one extra verify row 1.6 ms. `bench_rows.py` brackets that row cost at the real shapes:
the 4-bit MoE up-proj gather (512 experts, top-10) runs 0.1715 to 0.4106 ms from M=1 to M=15, which
is 0.82 ms per row over 48 layers for one projection, so 1 to 2.5 ms with gate_up and down; the
lm_head adds 0.028. A copy cycle costs `24.4 + 1.6k + 0.4` ms and yields `1 + m` tokens.

| block k | m | cycle ms | tokens | ms/token | vs 16.3 |
|---|---|---|---|---|---|
| 8 | 8 | 37.6 | 9 | 4.18 | 3.9x |
| 8 | 4 | 37.6 | 5 | 7.52 | 2.2x |
| 8 | 2 | 37.6 | 3 | 12.5 | 1.30x |
| 14 | 14 | 47.2 | 15 | 3.15 | 5.2x |
| 14 | 4 | 47.2 | 5 | 9.44 | 1.73x |
| 14 | 1 | 47.2 | 2 | 23.6 | **0.69x** |

Break-even is `1 + m > (24.8 + 1.6k)/16.3`: two accepted at k=14, one at k=8. A request where a
third of cycles are copies with mean accepted 6 gives `(2/3)(16.3) + (1/3)(34.0/7) = 12.5` ms/token,
**+30%** there and 0% on prose, json and code, the shape of MTPLX's 73.8 to 87.6.

## 7. Workbench plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, two rounds, paired. Record
decode tok/s, the `MTP[...]` line and the lane's new `COPY[...]` line.

| step | change | prompts | expect / stop |
|---|---|---|---|
| 0 | lane on, `MAX=1` | decode_bench, all four | no behaviour change, `COPY[...]` appears, output byte-identical. A diff is a bug: stop |
| 1 | `NGRAM` 6/8/10, `MAX=14` | decode_bench `edit` (the intended case) plus a real two-turn rewrite | hit 20-50%, mean accept above 4 at N=8, tok/s +15 to 35% |
| 2 | same | `prose`, `json`, `code` | hit under 2%, tok/s within noise. A regression above 2% means the miss path is not free |
| 3 | `MAX` 6/10/14/20, `M_LIMIT=32` | `edit` at 2k and 50k | the 50k run decides: a collapse between 14 and 20 confirms the `L <= 15` cliff. Also time verify at M=7 vs M=16 |
| 4 | `ADAPTIVE` 0/1, `MIN_ACCEPT` 0/0.75/1.5 | `edit` plus a mixed agentic suite | adaptive wins on mixed traffic, loses slightly on pure copy |
| 5 | temp 0.7, fixed seed, `MAX_SAMPLED` 4/8 | `edit` | not token-identical; accepted-length histogram and logprob distribution must match baseline. Watch the 24 MB of one-hot q traffic at k=14 |
| 6 | lane + `mtp-head8` + `mtp-depth`, lane last | full suite | all installs True, both log lines present, depth histogram non-negative, gains roughly additive |
| 7 | stop token inside a copyable prompt run, then a second turn on the same session | hand-built | `stop_cut` non-zero and the follow-up turn sane. This is the GDN poisoning MTPLX hit |

## 8. Commands

    export OMLX_MTP_COPY_LANE=1
    OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft,\
    $R3/copy-lane/patch.py:install_copy_lane"     # copy lane LAST

    cd ~/inference-server/kernels/round3/copy-lane
    ~/inference-server/kdev/bin/python test_copy.py
    ~/inference-server/kdev/bin/python bench_rows.py --bits 4

## 9. Limitations

Never run against the real model, so hit rate, accepted length and the gain are projections from a
corpus I wrote. My lm_head bench is unusable above M=6 (M=14 timed faster than M=12) with the daemon
on the GPU. The `L <= 15` cap assumes TurboQuant KV is live; step 3 settles that. Logprobs for
copy-accepted tokens are the proposal's degenerate one-hot row, so deployments serving logprobs
should leave the lane off. Nothing here helps prefill or a no-overlap request.
