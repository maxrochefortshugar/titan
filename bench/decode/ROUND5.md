# ROUND5: the depth policy's resting points, where a prefill kernel's decode cost lives, and the prefill round

M5 Max, 128 GB, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`, greedy, one model
instance at a time on 127.0.0.1:8085. Written as the work was done, which this
round means: the diagnosis, the code and the fake-driven tests are here and
green, and every real-model number is missing for the reason in section 1d.
Read the section headed "not measured" before quoting anything.

ROUND4 closed with three ranked levers and this round works them in order. The
first is the largest thing in the workstream and it is not a kernel: the
adaptive depth policy measured the identical reference configuration at 93.5
and 79.0 tok/s an hour apart, an 18% spread on the control arm, larger than
every kernel effect ROUND4 went on to attribute. The second is the question
ROUND4 raised and could not answer, which is why a kernel that only fires on
prefill-shaped calls costs 13.2% of a 64k decode. The third is the prefill
round proper, against oMLX's 1600 tok/s cold on 65k.

## How to reproduce

    # step 1, the depth policy: four repeats of one config, plus pinned depth 3
    bash bench/decode/round5/step1_depth.sh
    python bench/decode/round5/summarise.py step1

    # step 2, the residency arms, with the memory line either side of prefill
    bash bench/decode/round5/step2_residency.sh
    python bench/decode/round5/summarise.py step2

    # step 3a, per-chunk prefill rate against chunk size
    bash bench/decode/round5/step3_prefill.sh
    python bench/decode/round5/summarise.py step3

    # the unit half, which does run
    pytest tests/engine/test_depth_policy.py tests/model/test_qsa_arms.py

Every log and every result is under `bench/decode/round5/`.

---

## Step 1. The depth policy's resting points

### 1a. What the loop actually was

ROUND4 section 1a named the shape of the problem -- "a loop whose input is its
own output has more than one resting point" -- and left the mechanism to this
round. There are three, and they compound.

**The cost model learned from the cycles the policy shortened.**
`CycleCostModel` pooled every observation into one weighted least-squares fit
over `(width, wall_ms)`. A policy that settles on depth 2 only ever feeds it
width 3. Two things follow. The mass at a single `x` drives the regression's
determinant toward zero, so `fit` flips between a line through one point and
the seed fallback, and which of the two a process is in depends on how many
cycles it has run rather than on the machine. And the seed points -- the only
other widths in the sum -- were decayed by every observation, so after 512
cycles half the seed was gone and after an hour the policy priced width 5, the
number that decides whether depth 4 is worth trying, off nothing at all.

**The acceptance estimate was conditioned on the depth the policy chose.**
`PositionAcceptance` can only learn about position *i* from a chain that
drafted *i* tokens. At depth 2, positions 3 and above hold the prior of 0.7 for
the life of the process, whatever the text is doing. Holding still is the one
move that cannot resolve this, and the policy held still.

**Nothing damped the switch.** The choice was a plain argmax over values that
are routinely a per-cent apart, so it flipped on noise, and each flip changed
which width the cost model was fed, which changed the argmax.

Depth 0 deserves its own line, because it is an absorbing state. A cycle that
drafts nothing teaches the acceptance curve nothing at any position and the
cost model only width 1, so a policy that reaches depth 0 for any reason --
including a bad reason, such as a stale price at width 2 left behind by a
64k request -- has no route back out.

### 1b. The policy this round ships

Four changes, in `titan/engine/decode_cycle.py`.

**One decayed mean per width, and the line is fitted over the means.** A width
contributes one point to the regression whatever its sample count, so the
incumbent depth can no longer bend the price of the alternatives toward itself.
A width with at least `min_samples` observations is quoted from its own mean
and never from the line; the line prices only the widths the policy has not
run, which is the question a depth policy has to ask every cycle.

**Two clocks, because a price and a recency are different questions.** A
width's price is an average over that width's *own* last `own_half_life`
cycles. Under a single clock shared with every width, a stale price at a rarely
run width takes thousands of cycles to wash out, and while it is washing out it
is what pins the policy to the depth that produced it. A separate weight,
decayed on every observation at every width, decides only whether a width has
been seen recently enough to quote at all.

**Hysteresis and dwell.** A candidate has to beat the incumbent by 6% in
expected committed tokens per millisecond, and the incumbent has to have held
for 8 decisions before it can be displaced. The first decision on a track is
still the plain argmax: there is nothing to be hysteretic about yet.

**Probe runs, with a staleness discount.** Every 48 decisions the policy spends
8 consecutive cycles at the depth one above the incumbent, then next time one
below, and it discounts what it already believes about that depth by 70% before
it starts. Both halves are load-bearing. A single-cycle probe moves either
estimator by about one part in ninety, so a policy in a wrong resting point
needs roughly 140 probe cycles to climb out; eight in a row after a discount is
a measurement it can act on within a couple of hundred cycles, which is inside
one short request. The discount is not a thumb on the scale: whatever that
depth's price was, it was measured under a regime the policy has not observed
since, and averaging eight fresh cycles against four hundred stale ones is how
a wrong price survives the measurement sent to correct it. When one neighbour
is out of range the probe takes the other, which is what gets a policy out of
depth 0.

The probe duty cycle is 8 in 48, so about 17% of decisions run at a
neighbouring depth. That is a real cost and it is inside every number below.

### 1c. Convergence, on fakes

`tests/engine/test_depth_policy.py` grew a `FakeWorkload`: a deterministic
machine (cycle ms linear in width) and a deterministic text (a fixed
low-discrepancy sweep, so position *i* is accepted at its true rate without a
random seed anyone has to trust). Three controllers are steered to three
different starting points -- a run of total rejections, a run of fully accepted
depth-6 chains, and nothing at all -- and then run on the same workload.

| start | first 200 cycles | after 400 | tail mode | mean depth over the tail |
|---|---:|---:|---:|---:|
| low (120 cycles of total rejection) | 0 | 2 | 2 | 2.00 |
| high (120 cycles of 6-of-6 accepted) | 6 | 2 | 2 | 2.00 |
| fresh | 2 | 2 | 2 | 2.00 |

All three are at the same depth inside the first 400 cycles, which is inside
one short request. `test_the_two_starting_points_disagreed_before_the_workload_ran`
is the check that the steer is a real steer, so that the agreement afterwards
is convergence rather than a test that could not fail. The 16% of the tail that
is not the mode is the probe duty and nothing else.

### 1d. The real-model spread, measured

Twelve server lifetimes, three arms by four repeats, short and 64k in every
one, rotated so no arm sits at the same position in the pass twice. The bar
this round set itself: a spread under the machine's own drift, about 5%, and a
median not below pinned depth 3.

`adaptive` is the converging policy this round ships. `fixed3` is
`adaptive_depth = false` at depth 3. `legacy` is ROUND4's
`round(mean_accepted) + 1`, reachable again through
`speculation.depth_policy = "mean_accepted"`, because "the new policy is
steadier" is a claim about two policies and measuring one of them from a
previous round's notes measures the machine's drift as well.

**short**

| arm | n | tok/s each | median | spread | mean rows |
|---|---:|---|---:|---:|---|
| `adaptive` | 4 | 90.5, 86.8, 88.6, 86.1 | 87.7 | 5.0% | 2.92, 2.70, 3.16, 2.84 |
| `fixed3` | 4 | 90.1, 91.5, 91.2, 88.9 | 90.7 | 2.9% | 3.97 throughout |
| `legacy` | 4 | 93.1, 91.8, 89.5, 89.0 | 90.7 | 4.5% | 3.95 throughout |

**64k**

| arm | n | tok/s each | median | spread | mean rows |
|---|---:|---|---:|---:|---|
| `adaptive` | 4 | 60.8, 66.6, 68.2, 66.1 | 66.3 | 11.2% | 2.87, 2.95, 3.77, 2.86 |
| `fixed3` | 4 | 65.9, 66.3, 67.6, 66.2 | 66.2 | 2.6% | 3.97 throughout |
| `legacy` | 4 | 68.3, 67.3, 65.7, 66.7 | 67.0 | 3.9% | 3.97 throughout |

The policy misses both halves of the bar. Its 64k spread is 11.2% against the
two pinned arms' 2.6 and 3.9, and its short median is 3.3% under pinned depth
3. What it did fix is real and is visible in the same table: ROUND4's 18% on
the control arm is gone, and the two pinned arms now repeat themselves to
within the drift, which is the evidence that the 18% was the loop and not the
machine. The policy stopped oscillating. It settled in the wrong place.

Where it settles is the mean-rows column. Both pinned arms sit at 3.97 rows,
which is depth 3 every cycle. The adaptive arm sits at 2.86 to 2.95 in three
repeats and at 3.77 in one, and the one that reached depth 3 is also the
fastest 64k number in the whole table at 68.2. The arm is not noisy; it is
bimodal, and the two modes are depth 2 and depth 3.

### 1e. Why depth 2, and the one thing worth changing

The arithmetic is in the arms themselves. At 64k the pinned arm runs a
41.74 ms cycle at width 4 and commits 2.752 tokens; the adaptive arm runs a
37.10 ms cycle at width 3 and commits 2.256. That is 0.0659 committed tokens
per millisecond at depth 3 against 0.0628 at depth 2, so depth 3 is worth
about 5% more. The policy's hysteresis is 6%. A candidate that is 5% better
can never clear a 6% bar, so whichever depth the policy is in first is the
depth it stays in, and the seed cost table decides that: under the seed and
the 0.7 prior, depth 2 prices at 0.0928 against depth 3's 0.0918, so the very
first decision on a fresh process is depth 2 and nothing afterwards can undo
it.

The same reading explains the short arm. There the depths are within a per
cent of each other -- 0.08998 at depth 2 against 0.08985 at depth 3 -- so a 6%
bar freezes the policy wherever it lands, and on the short workload that is
sometimes depth 1, which costs 5%.

`bench/decode/round5/depth_tune.py` prices the settings against this, on a
deterministic fake whose acceptance curve and cost line are taken from the
pinned arms above, driven through the real controller. Two things about how it
is read. It is priced at the horizons the arms actually have, 912 cycles short
and 133 at 64k, because a policy's asymptote is not what a request sees: the
64k arm is 300 tokens and the convergence test in 1c runs to 400 cycles. And
the column that matters for a default is the fresh-start one, because every
arm here is a fresh process; the worst-of-three column is what the policy is
worth if it ever arrives at a resting point it did not choose, which is the
ROUND4 failure and the reason the probe stays.

| setting | duty | fresh, short | fresh, 64k | worst of three, short | worst of three, 64k |
|---|---:|---:|---:|---:|---:|
| shipped, h=0.06, probe 48/8 | 17% | 94.4% | 90.2% | 94.4% | 78.0% |
| h=0.03, probe 48/8 | 17% | 99.0% | 90.2% | 97.6% | 78.0% |
| h=0.02, probe 48/8 | 17% | 98.8% | 89.2% | 97.7% | 78.0% |
| h=0.02, probe 32/6 | 19% | 98.9% | 88.2% | 98.1% | 82.8% |
| h=0.02, probe 96/8 | 8% | 99.1% | 88.7% | 96.6% | 64.8% |
| h=0.02, no probe | 0% | 99.3% | 89.9% | 76.4% | 52.2% |
| h=0.06, no probe | 0% | 95.0% | 89.9% | 76.4% | 52.2% |

Percentages are of what a pinned depth is worth on that workload. Two things
fall out and only one of them is the round's question. The probe duty is not
what costs the fresh-start numbers: at 64k, no probe at all realises 89.9%
from fresh and the shipped 17% duty realises 90.2%, so the duty pays for
itself even before the robustness column, where dropping it is the difference
between 78% and 52%. The probe stays. What costs the fresh-start numbers is
the hysteresis, and 3% is the whole change: inside both workloads' true
margins, still outside the per-cent noise the bar exists to damp.

---

## Step 2. Where a prefill kernel's decode cost lives

### 2a. The first arm is degenerate, and that is the finding

The round asked for an arm with "the kernel on for prefill only with its decode
arm disabled". Neither kernel has a decode arm to disable, and establishing
that is worth more than the arm would have been.

`gdn_chunk_scan` is offered from `gated_delta_update`, and `gated_delta_update`
is the `gdn_sink is None` branch of `Qwen4ExpAttention`. Every drafted decode
takes the other branch: a verify block needs the per-row intermediate states
for replay-free rollback, so it goes through
`_gated_delta_update_verify_decode` and `gated_delta_update_with_states`, which
never reaches the chunk scan. A plain width-1 decode does reach
`gated_delta_update`, and is refused there by the kernel's own `q.shape[1] > 1`
gate. `grouped_rmsnorm_bf16` is offered from `hc_fused.prefill_forward` and
from nowhere else.

So both kernels are prefill-only in fact and not merely in intent, and ROUND4's
premise holds exactly. Whatever they cost a 64k decode, they cost it without
running: it is state left behind, and the second arm is the whole experiment.

### 2b. The instrument

`MLXModelBackend` now samples `mx.get_active_memory()`,
`mx.get_cache_memory()` and `mx.get_peak_memory()` when the last prefill chunk
lands, and the scheduler emits them as a `prefill_done` event. Three host-side
counters and no device query, so this is inside the one-sync-per-step rule the
2026-09-12 panic produced and nowhere near the per-scope evaluation that caused
it. This is the sample ROUND4 section 5 called cheap and not done.

`scheduler.release_after_prefill` is the second arm: `mx.clear_cache()` once
the prompt is in. The buffer cache is a free list rather than live data, so
dropping it cannot lose anything the decode needs. What it costs is the
reallocation of whatever the decode would have reused; what it buys, if the
residency hypothesis is right, is a decode that is not allocating underneath a
64k prefill's leftovers.

### 2c. Not measured

Blocked on 1d. The three arms are the control, the two prefill kernels added,
and the two prefill kernels plus `release_after_prefill=true`, each short and
64k, with the `prefill_done` memory line read off every one.

---

## Step 3. The prefill round

### 3a. The cold plan is already flat 2048, which moves the target

Before measuring per-chunk rate it is worth reading what the planner emits,
because the round's proposed change turns out to be close to a no-op on the
path it was aimed at. `plan_chunks(0, 64779, chunk=2048, block=512, grid=2048)`:

| suffix | chunks | sizes |
|---|---:|---|
| cold 64,779 | 33 | 31 x 2048, 1 x 1024, 1 x 267 |
| cold 12,000 | 7 | 5 x 2048, 1 x 1536, 1 x 224 |
| 3,000 matched, 64,779 total | 32 | 1 x 1096, 29 x 2048, 1 x 1024, 1 x 267 |

The cold 65k prefill is already 31 full 2048-token chunks and two tail chunks:
the fine cut at 64,512 (the prompt end rounded down to the block grid) and the
267 tokens past it. Snapshot boundaries and the block grid coincide at 2048, so
nothing in the plan is being shortened by them.

The 400-token chunks the integration report priced at 695 tok/s are the
*contended* path, `_contended_chunk = 512`, and `PortAdmitter` passes
`contended=False` unconditionally, so a served prefill never takes it. The
planner is therefore not the prefill lever it was expected to be, and the
remaining 3(b), 3(c) and 3(d) are where the 830-to-1600 gap has to come from.
Two chunks out of thirty-three being short is worth at most a couple of
per cent.

What is left to measure, and is not measured: the per-chunk rate against chunk
size on the real model (256 needs `cache.block_tokens=256` as well, since the
planner refuses a chunk that is not a multiple of the block), the per-chunk
fixed cost, and the two prefill kernels with step 2's answer in hand.

### 3b. The instrument

`bench/decode/round5/probe.py` gained a `prefill` mode that reads the
scheduler's own `prefill_chunk` events back out of `/metrics` and groups them
by chunk size, so the rate table needs no new instrumentation on the model.
`/metrics` gained an `events` parameter and the profiler's event ring went from
256 to 4096, because a 65k prefill at 256-token chunks emits 254 chunk events
and the old default of sixty is a rate table with the first three quarters of
the prompt missing.

### 3c. Not measured

Blocked on 1d.

---

## Step 4. The QSA indexer seams

### 4a. The two Metal kernels are not written, on purpose

`qsa.indexer_scores` and `qsa.topk_indices` are seams that already exist in
`qsa_fast.py`: `_native_indexer_scores` and `_native_topk_indices` check their
shapes, ask the adapter for an op, and fail closed to MLX when the map returns
`None`, which it does for both. Filling them means two new Metal kernels.

`docs/ops/INCIDENTS.md` rule 2 says the first real-model run of any new kernel
path happens with `kernels.reference_only` as the control, then with only that
kernel enabled, short before 64k; rule 3 says every custom Metal kernel has a
bounds test at the largest real shape before it is enabled by default. Neither
can be done this session, and shipping two unvalidated Metal kernels into a
tree whose kernel defaults are measured per op would be the opposite of what
ROUND4 established. They are left unwritten and this is the reason.

### 4b. The half that needs no kernel, which is written

ROUND4 named it: "the cheap half of it -- a persistent fp32 bank on the cache,
no Metal kernel -- removes the cast without a new kernel".

`_portable_indexer_scores` scores in float32 because which blocks win is a
discrete choice and rounding the products flips the ones near the cut-off. It
got there by casting the entire pooled bank on every decode step. At 64k that
is 16,000 slots of 128 bf16 values read and 8 MB written, per layer, per token,
over a bank that has not changed since the prompt went in.

`Qwen4ExpQSAKVCache.pooled_indexer_keys_f32` holds the widened copy and extends
it by the blocks that are new, mirroring the offset bookkeeping the bf16 bank
already does. The decode path passes it down when the `qsa_pooled_bank_f32`
forward path is on. The cast moves from once a step to once a block, and during
a 64k decode no blocks are added at all, so it moves to never.

This is bit-exact rather than close: widening bf16 to float32 is lossless, so a
step reads exactly the numbers it read before. The tests say so in those terms
--- `np.array_equal` on the scores, not a ULP bound --- across both indexer
thresholds and just past each of them, because a tolerance would hide the only
bug this path can have, which is the bank and its offset drifting apart.

The cost is the second bank: 8 MB a layer at 64k against the first bank's 4 MB,
so 96 MB over the twelve QSA layers. That is why it is a forward path defaulted
off rather than a default. The A/B that would turn it on is blocked on 1d.

---

## Step 5. Where this leaves things

Green: the full suite at 1,234 passed, 40 skipped, 13 deselected, under
`-m "not slow" -p timeout --timeout=60`. No oMLX imports in `titan/`. The only
environment read in `titan/` is the config file path, which predates this
round. No tests were deleted; the depth policy file gained eleven and the QSA
file gained twelve.

Not green: every real-model number. `~/.config/titan/titan.toml` is unchanged,
because nothing this round measured a win on the machine, and a default set
from a unit test is not a measured default.

### What is queued behind the 1.2 GB

1. Step 1's four repeats plus pinned depth 3, `step1_depth.sh`. The spread and
   the median against fixed depth 3 are the round's first deliverable and the
   only one whose fix is already written and tested.
2. Step 2's three arms with the `prefill_done` memory line.
3. Step 3's chunk-size sweep and the per-chunk rate table.
4. Step 4's A/B on `qsa_pooled_bank_f32` at 64k, plain and drafted, with the
   lossless digest.
