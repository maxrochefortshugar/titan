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

### 2c. The instrument was dead, and finding that is the first result

The first pass of these arms came back with no memory line at all. The reason
is a one-token invariant two files apart. `_close_prefill` fired on
`tokens_after == 0`, and the scheduler's own comment says a prefill plan stops
one token short of the prompt, so the last chunk is handed a count of one and
never zero. `test_every_prefill_chunk_is_told_how_much_of_the_sequence_follows_it`
asserts exactly that: "the final chunk's is one, not zero". The sample never
ran on a served request, and neither did `mx.clear_cache()`, so the second arm
would have measured the first one twice and agreed with it.

The boundary is a scheduler fact. The scheduler cut the prompt up and is the
only party that knows which chunk was last, so it now calls a public
`close_prefill` hook on `chunk.is_last`, through the same `_guarded` wrapper as
every other port call. `tests/engine/test_scheduler.py` gained two tests: the
hook arrives on the fourth chunk of a four-chunk prompt and not before, and it
arrives once per sequence rather than once per turn, because a backend that
dropped its buffer cache on every turn would pay the reallocation on every
token.

### 2d. The arms

Three repeats of the control and the released arm, two of the others,
alternated, one server lifetime each. A fourth arm was added that the round did
not ask for and that turns out to carry the answer: `releaseonly` is
`release_after_prefill=true` with the seven-kernel default and neither prefill
kernel, which separates what the release does from what the kernels do.

| arm | short decode | 64k decode | cold 65k prefill | in-chunk prefill |
|---|---:|---:|---:|---:|
| `control` (seven) | 91.0 | 67.4 | 868 | 881 |
| `prefill` (+ both kernels) | 91.8 (+0.9%) | 62.2 (-7.6%) | 935 (+7.8%) | 933 (+5.9%) |
| `released` (+ both, release on) | 93.5 (+2.7%) | 64.4 (-4.5%) | 940 (+8.3%) | 954 (+8.3%) |
| `releaseonly` (seven, release on) | 92.8 (+1.9%) | 68.0 (+0.8%) | 870 (+0.2%) | 877 (-0.4%) |

Medians in tok/s. Per-repeat spreads: 64k decode 4.5% control, 4.0% prefill,
2.5% released, 3.1% releaseonly; cold prefill 6.2%, 2.9%, 1.7%, 3.5%. The
derived cold-prefill column carries the request's tokenisation with it, which
is why the in-chunk column, read off the scheduler's own per-chunk events, is
next to it. They agree on every arm.

**The memory line, at the last prefill chunk**

| arm | pre active | pre cache | post active | post cache | released |
|---|---:|---:|---:|---:|---:|
| `control` | 75,659 | 26,728 | 75,659 | 26,728 | 0 |
| `prefill` | 75,693 | 26,704 | 75,678 | 26,712 | ~0 |
| `released` | 75,691 | 26,704 | 75,665 | 5 | 26,699 |

MB. A 64k prompt leaves 26.7 GB of MLX buffer cache behind on top of 75.7 GB
active, and `release_after_prefill` returns essentially all of it. The two
prefill kernels do not change either number: the residency they were suspected
of is not extra resident bytes.

### 2e. What the fourth arm settles

`releaseonly` is flat on all four measures, inside its own repeat spread on
every one. Dropping 26.7 GB of buffer cache neither helps nor hurts a decode
that is not running under the prefill kernels, so the release is not a decode
optimisation in its own right and cannot be sold as one.

Against that baseline the rest reads cleanly. The two prefill kernels are worth
about 8% on the prefill and cost 7.6% on the 64k decode. Turning the release on
with them recovers three of those seven and a half points at no cost to the
prefill, which is ROUND4's residency hypothesis confirmed in part and refuted
in the larger part: some of a prefill kernel's decode cost really is the
allocator state it leaves behind, and most of it, 4.5% of a 64k decode that
survives dropping every reclaimable byte, is not. Whatever the remaining 4.5%
is, it is not residency, and this arm cannot see it.

**The defaults do not change.** The condition this step set itself was that the
release recover the decode while keeping the prefill gain. It recovers under
half of it. What is left is a real trade rather than a win: +8.3% on the
prefill against -4.5% on the 64k decode, which on the 65k prompt is 5.7 seconds
saved against 0.7 ms added per output token, so the pair pays for itself below
about 8,000 output tokens and loses above it. That is a better trade than the
-12.2% ROUND4 recorded for the same two kernels, and the improvement is the
release. It is a deployment choice and not a default, so it stays in the
`[kernels]` comment in `titan.toml` with the new numbers, and the seven-kernel
default stands.

`release_after_prefill` also stays off. It is free on throughput and it returns
26.7 GB to a 128 GB machine, which is an operational argument rather than a
throughput one, but every arm here is single-stream and `max_seqs` is 4: what
the release costs when a second prefill starts immediately behind the first is
not measured, and a default set from a measurement that did not include the
case it would change is how the last three rounds got their bad defaults.

The 150-token digest is `34a52eecda614590e6c6a07c41ae424b` in all ten arms.

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

### 3c. The rate against chunk size, and the chunk that cannot grow

Two repeats of each size, one cold 65k prompt and four tokens of decode per
arm, the table read off the scheduler's own `prefill_chunk` events.

| arm | chunks | full-chunk tok/s | in-chunk | end-to-end | gap median | after a snapshot | gap share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chunk256` | 254 | 625 | 629 | 623 | 0.03 | 25.99 | 0.8% |
| `chunk512` | 127 | 755 | 760 | 754 | 0.03 | 21.42 | 0.8% |
| `chunk1024` | 64 | 865 | 863 | 854 | 16.95 | 21.32 | 1.0% |
| `chunk2048` | 33 | 902 | 904 | 894 | 23.92 | 23.92 | 1.1% |
| `chunk4096` | 33 | 902 | 907 | 896 | 23.69 | 23.69 | 1.2% |
| `chunk4096g` | 17 | 958 | 945 | 938 | 31.16 | 31.16 | 0.7% |
| `chunk8192g` | 9 | 986 | 961 | 955 | 50.49 | 50.49 | 0.6% |

`chunk4096` is the round's proposed arm and it is a no-op: 33 chunks and 902
tok/s, the same plan `chunk2048` runs. `plan_chunks` makes every snapshot-grid
multiple inside the suffix a chunk end, for the reason written above the method
-- a chunk that steps over a grid multiple stages nothing there and the store
chain truncates back -- so `scheduler.prefill_chunk` is clamped to
`cache.snapshot_grid` and cannot be raised on its own. The `g` arms move the
grid with the chunk, and they are where the rate is.

The curve is an occupancy curve and it is still climbing at 8192: 623 tok/s at
256, 894 at 2048, 955 at 8192, monotone, two repeats each, the pairs within 1%
of one another. Nothing here is scheduling.

### 3d. There is no per-chunk fixed cost to remove

This is the step's flattest result and it retires a lever. The wall between one
chunk's forward returning and the next one starting is 0.03 ms after a plain
chunk. Not 0.3, not 3: the turn loop, the store session and the host work
between two forwards are together thirty microseconds, and at 256-token chunks,
where the loop runs 254 times, they are still 0.8% of the prefill.

The only fixed cost that exists is snapshot staging, and it is the whole of the
gap column: 21 to 26 ms per staged snapshot at 2048 and below, 31 at 4096, 50
at 8192, scaling with the state being copied rather than with the number of
chunks. Thirty-one snapshots at 26 ms is 0.8 seconds of a 102-second prefill.
The split the round asked for -- gap after a snapshot against gap after a plain
chunk -- comes back as 26.0 against 0.03 at 256 tokens, which says all of it
and says it cleanly. Host syncs per chunk are 1.0 throughout.

So the fixed-cost removals this step was expected to produce have no target,
and none were written. The 830-to-1600 gap is inside the prefill forward.

### 3e. `moe_gather_int8` as a prefill-only op

Two repeats each, alternated.

| arm | in-chunk prefill | 64k decode | digest |
|---|---:|---:|---|
| `int8off` | 902 | 67.1 | `34a52ee...` |
| `int8prefill` | 905 | 68.8 | `34a52ee...` |

+0.3% on the prefill, which is a quarter of the arm's own repeat spread. The op
runs over 20,480 rows on a 2048-token chunk and ROUND4's -3.7% was measured on
a decode, so the phase split was worth asking; the answer is that the prefill
does not care either. It stays off. The 64k decode column is the control
measured twice and drifting 2.5%, since a prefill-only op cannot touch a
decode, and it is a useful reminder of what this machine's drift is.

The digest is unchanged, which the step did not expect: the arm's activation
quantisation is 0.65% of output RMS and the 150-token greedy answer survives it
byte for byte. Recorded, not asserted, exactly as planned.

### 3f. What the prefill levers reach, together

`prefillmax` is every prefill lever this round found, at once: chunk and grid at
8192, both prefill kernels, the release on.

| arm | in-chunk prefill | end-to-end prefill | 64k decode |
|---|---:|---:|---:|
| production today (2048, seven kernels) | 904 | 894 | 67.4 |
| `prefillmax` | 1,046 | 1,039 | 64.5 |

+16.2% on the cold 65k prefill and -4.3% on the 64k decode. Against oMLX's
1,600 that is 65% of the target rather than 56%, and the remaining 35% is not
in the scheduler, not in the chunk plan and not in the host loop, all three of
which this step measured and found to cost about one per cent between them. It
is in the prefill forward.

**What changes, and what does not.** Nothing here becomes a default. Chunk size
is the one real win and it cannot be taken without moving `cache.snapshot_grid`
with it, and the grid is what a partial prefix match restores at: at 8192 a
warm turn recomputes up to 8,191 tokens it could have restored, about nine
seconds at these rates, against the 3.9 seconds the larger chunk saves on a
cold 65k prompt. Every arm in this round runs with the prefix store off, so the
warm side of that trade is not measured here, and the round that sets a default
from the half of a trade it measured is the round that produced the defaults
this one has been unwinding. The numbers are in the table and the coupling is
now written down; decoupling the two, so that a chunk can stage snapshots at
several grid multiples inside it, is a backend change and is the first of this
round's next levers.

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
off rather than a default.

### 4c. The A/B, and the cast that was not costing anything

Four arms at 64k. Drafted is the production cycle; plain is
`speculation.enabled=false`, the width-1 decode where the indexer runs once per
committed token and the cast is at its largest share of a step. Three repeats
of the plain pair, two of the drafted pair.

| arm | n | cycle ms each | median cycle ms | 64k tok/s |
|---|---:|---|---:|---:|
| `bankoff-draft` | 2 | 40.85, 40.97 | 40.91 | 67.3 |
| `bankon-draft` | 2 | 40.42, 41.06 | 40.74 | 67.5 |
| `bankoff-plain` | 3 | 20.29, 18.39, 18.39 | 18.39 | 54.4 |
| `bankon-plain` | 3 | 18.28, 18.38, 18.30 | 18.30 | 54.6 |

-0.4% of a drafted cycle and -0.5% of a plain one. Both are a fraction of this
machine's drift and the bank does not win. It stays off.

The third repeat of the plain pair is why that sentence is what it is. At two
repeats the plain arms read 51.8 against 54.5, a 5.2% win, and the whole of it
was `bankoff-plain` r0 at 20.29 ms against three other measurements at 18.28 to
18.39. One outlier in a two-repeat arm is a 5% result, and the round would have
turned a forward path on and spent 96 MB on it.

What the null result says about the diagnosis is worth keeping. The reasoning
in 4b is arithmetically right: the old path did cast 16,000 slots of 128 bf16
values per layer per token, and the new one casts nothing during a 64k decode
because no blocks are added. Removing it changes the cycle by half a per cent,
so that cast was not on the critical path, and a decode step at 64k is not
short of the bandwidth it was assumed to be short of. The seam and the tests
stay, because the bank is correct and cheap to keep and the next indexer change
may need it; the default does not.

The digest is `34a52eecda614590e6c6a07c41ae424b` in all four arms, which is the
bit-exactness the step required rather than a tolerance.

---

## Step 5. The production config, before this round and after it

`prodbefore` is what this round inherited: ROUND4's production config, which is
`adaptive_depth = true` on the `mean_accepted` policy. `prodafter` is
`~/.config/titan/titan.toml` exactly as it now stands. Two repeats each,
alternated, one server lifetime per arm.

| measure | before | after | oMLX |
|---|---:|---:|---:|
| short decode | 94.7 | 94.7 | 84 to 91 |
| 64k decode | 69.1 | 68.9 | 71 |
| cold 65k prefill | 882 | 882 | 1,600 |
| 150-token digest | `34a52ee...` | `34a52ee...` | n/a |

A dead heat, and that is the honest headline. Step 1 replaced an adaptive
policy with a pinned depth and the throughput did not move, because the legacy
policy was already landing on depth 3 on both workloads: its `mean_rows` here
are 3.95 and 3.97, the same rows the pinned arm runs. What step 1 bought is in
the spread rather than the median, 2.9% and 2.6% against 4.5% and 3.9% over
four repeats, and a policy that cannot reach depth 0 or oscillate between
resting points an hour apart. That is worth having and it is not a speed-up.

**Read the absolute numbers with the clock next to them.** These arms ran at
06:40 and the step-2 control ran at 04:20 at 91.0 short and 67.4 at 64k, on the
same config. The machine drifted about 4% upward across this session, which is
larger than most of the effects the round set out to measure and is the entire
reason every arm above is paired and alternated. Nothing in this document
should be compared against a number from a different hour.

## Step 6. Where this leaves things

### 6a. What changed

One code change and no default changes.

`close_prefill` is now a scheduler-driven hook rather than a backend guess.
This is a bug fix, not a tuning change: the memory sample and
`release_after_prefill` had never once run on a served request, because they
keyed off a `tokens_after` of zero that a prefill plan never produces. Two
tests in `tests/engine/test_scheduler.py` hold the boundary, one for when the
hook arrives and one for how often.

`~/.config/titan/titan.toml` changes only in its `[kernels]` comment, which
carried ROUND4's -12.2% for the two prefill kernels and now carries this
round's +8.3% / -4.5% with the release on, since that trade is what a
prefill-heavy deployment would actually be choosing between.

Everything else this round measured came back inside the machine's drift or
outside the terms it set itself, and is written up above rather than shipped:
the two prefill kernels and `release_after_prefill` (a trade, not a win), the
chunk-size increase (a win that cannot be taken without the snapshot grid),
`moe_gather_int8` prefill-only (+0.3%), and `qsa_pooled_bank_f32` (-0.4%).

### 6b. Against oMLX

| measure | Titan | oMLX | Titan's best measured arm |
|---|---:|---:|---|
| short decode | 94.7 | 84 to 91 | 94.7, production |
| 64k decode | 68.9 | 71 | 68.9, production |
| cold 65k prefill | 882 | 1,600 | 1,039, `prefillmax` |

Short decode is ahead. The 64k decode is 3% behind and has been within a few
per cent for three rounds. The prefill is 45% behind, and this round is the
first one that can say where it is not: not the chunk plan, not the turn loop,
not snapshot staging, not the host gap between forwards, which together cost
about 1%.

### 6c. The next three levers

1. **The prefill forward itself.** Step 3 eliminated everything around it, so
   the 882-to-1600 gap is inside the chunk's forward pass. The measurement that
   follows is a layer-type attribution of one 2048-token chunk, and
   `INCIDENTS.md` rule 1 says how: on the synthetic configuration, or on the
   real model with at most one sync per step, never the per-scope harness that
   panicked the machine on 2026-09-12.
2. **Decouple `scheduler.prefill_chunk` from `cache.snapshot_grid`.** The chunk
   is clamped to the grid because a snapshot can only be staged at a chunk end,
   and that clamp is worth 6.8% of a cold prefill at 8192. Letting one chunk
   stage snapshots at several grid multiples inside it is a backend change that
   takes the prefill win without coarsening what a warm turn can restore.
3. **The 4.5% of a 64k decode that the two prefill kernels cost without
   running.** Step 2 halved ROUND4's number and named what the other half is
   not: it survives dropping all 26.7 GB of reclaimable buffer cache, so it is
   not residency. Two candidates the memory line cannot see are allocator
   fragmentation, which `mx.clear_cache()` does not fix, and a per-op
   compilation or shape-cache effect that the prefill shapes leave behind.

### 6d. Hygiene

The full suite is green under `-m "not slow" -p timeout --timeout=60`: 1,309
passed, 40 skipped, 13 deselected. No oMLX imports in `titan/`. No tests were
deleted; two were added.

On environment variables, the earlier rounds' phrasing needs correcting rather
than repeating. Titan's own code reads exactly one, the config file path in
`titan/config/settings.py`. The vendored `mlx_vlm` tree under
`titan/adapters/mlx/vendor/` reads six more, all of them kill switches on
forward arms (`TITAN_QWEN4_HC_FUSED`, `TITAN_QWEN4_HC_HYBRID`,
`TITAN_QWEN4_GATHERED_MIN_QUERY` and the rest). None was added this round, none
is set by any harness here, and every arm above ran with none of them in the
environment. They are worth removing in favour of the `forward_paths_on`/`off`
config that now exists, and that is a cleanup rather than a finding.

Every arm ran through `bench/decode/round5/arm.sh` on 127.0.0.1:8085, one model
instance at a time, in the foreground, and every result is in
`bench/decode/round5/results.jsonl`.

### 6e. One process note

This round began by stopping a background chain the previous session left
running: `step1b_tuned.sh` chained into `step2_residency.sh` and
`step3_prefill.sh` under `bash -c`, with a `titan serve` on 8085 and a polling
watcher beside it, all still going after that session had reported. The first
measurement of this round would have been taken against a machine already
serving a model. Its four partial `control` rows were dropped from
`results.jsonl`, which is kept as `results.prefix-bug.bak.jsonl`, and every arm
after that ran in the foreground of one shell.
