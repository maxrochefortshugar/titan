# MTP park-and-probe policy

Workstream `kernels/round3/mtp-park/`, 2026-09-12. No model loaded, no safetensors opened, no GPU
used: source reading, log arithmetic and a host-side simulation. Ports 8083/8084 untouched.

## 1. The premise does not survive the logs

The brief says prose decodes at 33-35 tok/s in every round-3 configuration and that the park policy
is why. Neither half holds. `decode_bench.py`'s `prose` prompt (`MTP[3]`, 600 tokens) **never
parks** in any of the six, and its rate tracks per-cycle backbone time and nothing else.

| config | prose tok/s | cycles | backbone+head per cycle | implied ms/token |
|---|---|---|---|---|
| head8 | 35.4 | 277 | 59.6 ms | 27.5 |
| conf-depth | 46.1 | 239 | 51.4 ms | 20.5 |
| head8+conf-depth | 64.8 | 243 | 36.2 ms | 14.7 |
| gdn-scan | 73.2 | 276 | 28.1 ms | 12.9 |
| copy-lane | 74.4 | 268 | 28.4 ms | 12.7 |
| decode-all | 71.7 | 242 | 32.8 ms | 13.3 |

35.4 is the `head8` run alone, and that whole run was slow: its cold prefill measured 1056 tok/s
against 1424 for `gdn-scan` fifteen minutes later. Contention, not policy.

The only request that parks is `e2e_cold.py`'s tail, "Explain the CAP theorem in detail with
examples.", 300 greedy tokens issued 15 s after a 64k cold prefill: the `MTP[7]` line and the
`decode X tok/s` number. `analyze_logs.py` splits it into segments by the park and probe
timestamps.

| config | MTP tok | MTP tok/s | std tok | std tok/s | probe tok | probe tok/s | request tok/s |
|---|---|---|---|---|---|---|---|
| head8 | 121 | 41.2 | 128 | 49.8 | 47 | **17.9** | 36.9 |
| conf-depth | 140 | 46.2 | 128 | 52.5 | 32 | 40.4 | 47.9 |
| head8+conf-depth | 163 | 51.0 | 128 | 56.4 | 9 | 55.2 | 53.3 |
| gdn-scan | 201 | 55.7 | 99 | 56.4* | - | - | 55.9 |
| copy-lane | 203 | 54.7 | 97 | 56.4* | - | - | 55.2 |
| decode-all | 164 | 51.2 | 128 | 57.0 | 8 | 51.0 | 53.5 |

(*) no probe fired, so the standard rate is the median of the four measured ones.

The standard decoder beats MTP here in every configuration, so **parking is the right call**. The
loss is the re-entry probe, worst in `head8`: 47 tokens in 2.62 s, 17.9 tok/s against the 49.8 of
the mode it left.

## 2. Why the probes fail

| stage | file:line | cost |
|---|---|---|
| eligibility unblocks after 128 standard tokens | `BG:398-412`, `BG:777` | - |
| fresh `_MtpState` and a **fresh** `_DepthController` | `BG:1199-1224` | `p` resets to `[0.6]*d`, `t` empties |
| `_post_init_mtp`: one extra 1-token backbone forward, new head cache, 2 `init` emits | `BG:2443-2530` | ~25 ms plus Metal shape warmup for M=2..d+1 |
| warmup sweep max..1 then three depth-0 cycles | `BG:1949-1953` | `max_depth+3` cycles before any score is used |
| probe may succeed on the **first** non-losing post-warmup cycle | `BG:832-854` | deletes the park state |
| exit needs 16 consecutive losing decisions | `BG:1911` | a losing probe therefore runs at least 22 cycles |

Two of these interact badly. `BG:849` deletes the park state at the first non-losing cycle, so a
probe four cycles old counts as a success, the next park builds a fresh `_MtpParkState` at 128
tokens instead of doubling, and the backoff at `BG:800-806` never engages. `head8` shows it live:
park at 13:00:57.180, "probe succeeded" at 13:00:59.980, park again 2.4 s later, again at 128.

## 3. The break-even, stated

One depth-1 cycle costs `t[1]` and yields `1 + p1` tokens. The standard decoder emits one token per
`t[0]/tax`, where `t[0]` is measured inside the MTP loop and `tax` is the loop's synchronous
round-trip (`EXIT_MARGIN`, 1.15 by default, measured per machine at `BG:1728`). So

    p1 > tax * t[1] / t[0] - 1.

From the `copy-lane` segment: standard step 17.7 ms, so `t[0] = 20.4` ms; the MTP cycle is 34.1 ms
at mean drafted `k = 1.48`, giving a marginal verify row `D = 9.3` ms and `t[1] = 29.7` ms. **The
floor is 67.5%** against a measured 60.2%. At `decode_bench`'s short context `t[0] = 25.6` and
`D = 2.5` give **15%** against a measured 80%. The verify row is what moves: 2.5 ms at 40 tokens of
context, 9.3 ms at 64k, while the plain step barely changes.

## 4. Alternative policies, priced on the measured segments

| config | stock | (a) never park | (b) break-even park, 512-token cooldown, 32-cycle probe | (c) MTP off after 2 failed probes |
|---|---|---|---|---|
| head8 | 36.9 | 41.2 | **45.9** | 36.5 |
| conf-depth | 47.9 | 46.2 | **49.3** | 47.9 |
| head8+conf-depth | 53.3 | 51.0 | 53.3 | 53.3 |
| gdn-scan | 55.9 | 55.7 | 55.9 | 55.9 |
| copy-lane | 55.2 | 54.7 | 55.2 | 55.2 |
| decode-all | 53.5 | 51.2 | 53.7 | 53.5 |

(a) loses everywhere: a depth floor alone holds a sequence in a mode that is genuinely slower.
(b) never loses and gains 24% on the worst row, purely by making the cooldown outlast the request.
(c) is neutral: no configuration reached two failed probes.

## 5. What was built

`patch.py`, one idempotent env-gated install, **import time**, loaded by path.

| install | env | wraps |
|---|---|---|
| `install_park_policy()` | `OMLX_MTP_PARK_POLICY=1` | `_DepthController._best` / `.should_exit` / `.observe`, plus `_park_mtp_to_standard`, `_maybe_finish_mtp_reentry_probe`, `_prepare_mtp_state_for_next` and `_MtpParkState.__init__`'s cooldown default |

| knob | default | effect |
|---|---|---|
| `OMLX_MTP_PARK_MIN_DEPTH` | 1 | `_best` may not return 0 unless `p[0]` is below the floor; 0 restores stock |
| `OMLX_MTP_PARK_ACCEPT_FLOOR` | -1 | static `p1` floor; negative means the live break-even of section 3 |
| `OMLX_MTP_PARK_PROBE_CYCLES` | 32 | post-warmup cycles a probe must run before it may succeed or park again, with a hard abort below half the floor after 8 |
| `OMLX_MTP_PARK_TOKENS` | 512 | initial cooldown |
| `OMLX_MTP_PARK_MAX_TOKENS` | 8192 | cooldown ceiling |
| `OMLX_MTP_PARK_MAX_PROBES` | 2 | failed probes before MTP is off for the rest of the request |
| `OMLX_MTP_PARK_STICKY` | 1 | carry the backoff across a probe that declared success |
| `OMLX_MTP_PARK_TRACE` | 0 | log every decision |

Nothing here touches `_chain_next_drafts`, so round-2's shortlist, `mtp-depth` and `copy-lane` keep
it to themselves. Install **after** `install_conf_depth()` so this `_best` wrapper sits outside
`mtp-depth`'s: when that one collapses to 0 and the floor refuses, the answer becomes the best
speculative depth. The `observe` wrapper reads `copy-lane`'s `_omlx_copy_active` marker without
consuming it, so a copy cycle never counts toward the probe window in either install order.

## 6. Synthetic test

`test_park_policy.py` registers a miniature `batch_generator` under the real module name and execs
the **real** `_DepthController`, `_MtpParkState`, `_mtp_park_state_for_batch` and
`_maybe_finish_mtp_reentry_probe` out of the shipped file; the driver mirrors `patched_next` and
`_mtp_next`. Costs are section 3's; the fake model's acceptance switches between prose-like
(d1 0.60) and code-like (d1 0.90) at tokens 400 and 800.

| check | result |
|---|---|
| [0] installs, wraps all six symbols, idempotent, composes on top of a `mtp-depth`-style `_best` | pass |
| [1] greedy stream over 1200 tokens identical to stock, to never-park and to MTP-off | **identical 3/3** |
| [3] stock cooldowns `[128,128,128,128,512,512,128,...]`, non-monotone; patched `[512,512]` | pass |
| [4] `breakeven_p1` equals `tax*t1/t0-1` exactly | pass |

Modelled tok/s over 1200 tokens, with parks and probes in brackets:

| stream | stock | park policy | never park | MTP off |
|---|---|---|---|---|
| prose | 52.5 (4, 3) | **54.0** (2, 1) | 43.1 | 56.5 |
| code | 65.0 (0, 0) | 65.0 (0, 0) | 65.0 | 56.5 |
| mixed | 53.1 (4, 4) | **54.2** (2, 1) | 48.7 | 56.5 |

Depth per 100 tokens on the mixed stream: stock oscillates, spending 31 to 97% of most buckets in
standard mode, while the policy parks at token 200, returns at 600 inside the code regime at mean
depth 2.75 to 2.90, and parks again at 900. The knob sweep ([5]) says cooldown length is what
matters: 512 beats 128 by 1.6 to 3.6%, 1024 adds nothing, `PROBE_CYCLES` 8 against 32 is worth
0.1 to 0.4 tok/s. 32 is the default anyway, since stationary synthetic acceptance cannot reward a
better estimate.

## 7. Workbench plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, two rounds, paired. Every row
carries `OMLX_MTP_SHORTLIST_DRAFT=1`.

| step | env added | prompts | expect / stop |
|---|---|---|---|
| 0 | none | `decode_bench.py` all four, then `e2e_cold.py` | reproduce section 1: prose does not park, only the post-64k request does. If prose parks, the short-context floor is wrong and everything below changes |
| 1 | `OMLX_MTP_PARK_POLICY=1 OMLX_MTP_PARK_TRACE=1` at defaults | same | the `parked for` line must read 512, and at most one probe per request |
| 2 | `OMLX_MTP_PARK_TOKENS` in {128, 256, 512, 1024} | `e2e_cold.py` | +2 to 6% at 512 over 128 on the CAP-theorem tail, flat by 1024 |
| 3 | `MIN_DEPTH=0` (stock park) against `MIN_DEPTH=1 ACCEPT_FLOOR=0` (never park) | `e2e_cold.py` plus a 32k prose continuation of 1000 tokens | never-park should **lose** 1 to 4%, confirming section 4. If it wins, the loop tax exceeds `EXIT_MARGIN` and the floor is too low |
| 4 | defaults | new mixed prompt: 300 tokens of prose then "now write the Python implementation", 900 tokens, at 40 tokens and at 32k of context | the policy should re-enter for the code half. This is the row the patch is for |
| 5 | `OMLX_MTP_PARK_MAX_PROBES` in {1, 2, 0} | `e2e_cold.py` at `max_tokens` 1200 | 2 should beat 0 on a long prose request and lose on the step-4 mixed one |
| 6 | greedy, all of the above, plus `mtp-depth` and `copy-lane` | full suite | **byte-identical output to the unpatched run everywhere**. This is the acceptance criterion; the patch changes only when the loop is entered |
| 7 | `OMLX_MTP_PARK_TRACE=1` | 50k uncached prefill then 600 decoded | prices the re-prime plus Metal shape warmup that section 2 estimates at 25 ms |

## 8. Commands

    cd ~/inference-server/kernels/round3/mtp-park
    ~/inference-server/kdev/bin/python analyze_logs.py
    ~/inference-server/kdev/bin/python test_park_policy.py

To enable, import `patch.py` from `prod/bootstrap.py` by path at import time, after round-2's
shortlist and after `mtp-depth`'s `install_conf_depth`:

    OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft,\
    $R3/mtp-depth/patch.py:install_conf_depth,\
    $R3/mtp-park/patch.py:install_park_policy,\
    $R3/copy-lane/patch.py:install_copy_lane"        # copy lane still LAST
    export OMLX_MTP_PARK_POLICY=1

## 9. Limitations

Nothing here has seen the real model. Section 4 reuses each configuration's own measured rates, so
it holds only if those rates hold for the tokens the policy moves between modes, and the `head8`
row comes from a contended run. The 25 ms re-prime and 40 ms shape warmup are estimates, and the
simulator cannot exercise `mx.async_eval` scheduling. `EXIT_MARGIN` stays at its 1.15 fallback
until a hand-off measures the real loop tax, so where the true tax is 1.3 the floor is 15 points
too low. Blocking `should_exit` for 32 cycles costs real time on a doomed probe; the half-floor
abort bounds that at eight, but only well below the line. The sticky backoff lives on the
`GenerationBatch` keyed by uid, so a request that migrates between batch objects restarts at 512.
