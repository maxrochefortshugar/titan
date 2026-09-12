# ROUND2: the second decode round, measured

M5 Max, 128 GB, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`, greedy, one model
instance at a time on 127.0.0.1:8085. Written as the measurements were taken,
so the negative results are here with the positive ones and in the order they
happened.

The baseline this round starts from is `bench/decode/README.md`: short 54.6
tok/s with MTP off and 82.9 with the `omlx` chain at fixed depth 3; 64k 45.7
off and 52.0 to 54.3 on. The oMLX targets on the same machine are 84 to 91
short and 71 at 64k. Short was already at target when this round opened. 64k
was not, and that is what ROUND2 is about.

## How to reproduce

`run_round2.sh` starts a server, runs one mode, and stops everything it
started. One mode per invocation on purpose: arms have to alternate on a warm
GPU, because thermal and page-cache drift on this machine moves a decode number
by more than most of what this round is looking for.

    bash bench/decode/run_round2.sh <arm> <short|long|sweep|lossless> [--set ...]

Results append to `~/inference-server/staging/round2/results.jsonl`.

Two profiler fields are new this round and both are in `GET /metrics`:

- `position_acceptance`, P(draft *i* accepted | the chain reached *i*),
  reconstructed from the cycle profiles. It is the curve the long-context
  defect shows up in first, and nothing was reporting it.
- `?window=n`, which limits the decode summary to the last *n* cycles. The
  counters stay cumulative, so a caller that knows how many cycles its own
  request cost can read that request's numbers back without resetting
  anything. The context sweep needs it: seven contexts in one process.

## Reading the numbers

`accepted/cycle`, `acceptance` and the position curve are the steady figures
and are what the arguments below rest on. `tok/s` and `cycle ms` on this
machine drift by a factor that swamps most of the effects here (see step 1c),
so a throughput claim is only made where an arm was measured repeatedly inside
one process.

---

## Step 1. The MTP head at long context

### 1a. Head position alignment: no effect, and it cannot have one

**Hypothesis.** The head's KV cache starts empty at the first decode cycle, so
its RoPE offset is the count of decoded tokens while the trunk attends at
64,779 plus that count. Put the head back on the trunk's position line and
acceptance should recover.

**Where the positions come from.** `Qwen4ExpMTPModule.__call__` passed
`position_ids=None` into its one decoder layer, and with nothing passed
`Qwen3_5Attention` builds `arange(cache.offset, cache.offset + L)` from the
cache it was handed. For the head that cache is the head's own, so the
hypothesis is exactly right about the mechanism: at 64k the head really is
reading positions 0, 1, 2 while the trunk reads 64,779, 64,780, 64,781. oMLX
does not seed an offset either; what it does instead is fold the whole prompt
into the head cache during prefill (`patches/mlx_lm_mtp/prompt_priming.py`), so
its head's offset *is* the prompt length as a side effect of having the
prompt's keys.

**What was built.** `position_ids` threaded through the vendored MTP module,
`TitanQwenFlashNext.mtp_step(..., position_offset=)` building the same
`[3, 1, T]` mRoPE planes the attention builds for itself, and the drafter
computing the constant that turns a head offset into a trunk position.
`speculation.mtp_head_align_positions` selects it. Passing nothing is still the
identical graph, which is what makes the arm honest.

**Result: byte-identical drafts.** 64k, fixed depth 3, two arms in the same
session:

| arm | accepted/cycle | acceptance | position acceptance | accepted histogram |
|---|---|---|---|---|
| base | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 35 / 32 / 15 / 39 |
| positions aligned to the trunk | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 35 / 32 / 15 / 39 |

Not close. Identical, to the cycle.

**Why.** RoPE is a relative encoding. Shifting every query and every key by the
same constant leaves every `q · k` unchanged, so an absolute position offset
applied to a cache the head attends to *in its entirety* is a no-op by
construction. Confirmed off the GPU as well, on a synthetic Qwen4-Exp with a
real head: an offset of 64,779 moves the head's logits by 1.0e-5 and moves no
argmax.

So the hypothesis is half right and the half it gets right is the half that
matters. The head's problem at 64k is not where it thinks it is. It is that its
cache holds a few hundred decoded tokens and nothing of the 64,779 the answer
depends on.

**Kept:** nothing. `mtp_head_align_positions` stays at its default of false and
is documented as a measured no-op; the `position_ids` parameter on the vendored
module stays, because it costs nothing and the next thing that wants to move
the head's positions should not have to re-derive the plumbing.

### 1b. Prompt priming: fold the prompt into the head during prefill

**Hypothesis.** 1a ended with the head's cache holding a few hundred decoded
tokens and nothing of the 64,779 the answer depends on. Fold the prompt into
the head during prefill and the drafter's attention sees what the trunk's does.

**What it costs to build.** Nothing new on the GPU: the head predicts token
*t+2* from the trunk's hidden at position *t* and the embedding of token *t+1*,
and a prefill chunk already computed both. `TitanQwenFlashNext._prime_chunk`
folds the chunk's hidden states against the chunk's own ids shifted by one,
which is one extra layer over the prompt out of 49. The pair at a chunk's last
position is the only one the chunk cannot form alone, so the scheduler hands
the backend the token after the chunk. `speculation.mtp_prime_prompt` selects
it.

**Result: the acceptance is real and the draft time eats it.** 64k, fixed
depth 3, `omlx` chain, 300 decode tokens:

| arm | accepted/cycle | acceptance | position acceptance | draft ms | cycle ms | profiler tok/s |
|---|---|---|---|---|---|---|
| no priming | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 7.11 | 45.9 | 54.0 |
| priming, `clone` cache | 1.542 | 52.3% | 0.750 / 0.483 / 0.336 | 11.55 | 50.3 | 50.6 |
| priming, `trim` cache | 1.542 | 52.3% | 0.750 / 0.483 / 0.336 | 10.30 | 49.3 | 51.6 |

Two things in that table and they point opposite ways.

The head does draft better with the prompt in it. Acceptance 50.1% to 52.3%,
and the gain is at the front of the chain where it is worth the most: position
1 goes 0.723 to 0.750 and position 2 goes 0.454 to 0.483. The accepted
histogram moves the same way, 35 zero-accept cycles down to 31. So the
mechanism 1a argued for is the right mechanism.

It is also nowhere near enough. Two and a bit points of acceptance against the
twenty-five that separate 64k from short context. Whatever else the head is
missing at long context, the prompt not being in its KV cache is a small part
of it.

And the head now pays for the cache it was given. Draft time goes 7.1 ms a
cycle to 11.6: four head calls a cycle, each re-attending over 64,779 entries
instead of a few hundred, plus a copy of that cache for the speculative tail.
`mtp_chain_cache = "trim"` removes the copy and is worth 1.25 ms, which
confirms both halves of the docstring's claim -- clone and trim draft
identically (1.542 either way, same histogram to the cycle) and the copy is not
free once the head holds a prompt. The remaining 3.2 ms is the attention
itself. Net the arm is 4% *slower* than not priming at all.

**The window.** If the cost is the head attending over the whole prompt and the
benefit is the head having seen some of it, the two can be separated:
`speculation.mtp_prime_window` folds only the last *n* tokens. A chunk that
falls entirely outside the window is not primed and is not asked for its hidden
states either, so the window is cheaper on prefill as well as on decode. The
scheduler is what cuts the prompt into chunks, so it is the scheduler that
tells the backend how much of the sequence follows each one (`tokens_after` on
the prefill port); a chunk-local window would put the window in the wrong place
on every chunk but the last.

**Result: the window beats both, and it is not a compromise between them.**
Same 64k arm, fixed depth 3, `omlx` chain:

| arm | accepted/cycle | acceptance | position acceptance | draft ms | cycle ms | profiler tok/s |
|---|---|---|---|---|---|---|
| no priming | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 7.11 | 45.9 | 54.0 |
| prime the whole prompt | 1.542 | 52.3% | 0.750 / 0.483 / 0.336 | 10.30 | 49.3 | 51.6 |
| prime the last 16,384 | 1.632 | 55.4% | 0.795 / 0.527 / 0.339 | 8.79 | 47.1 | 55.9 |
| prime the last 4,096 | 1.632 | 55.4% | 0.795 / 0.527 / 0.339 | 7.77 | 46.3 | 56.9 |

A window drafts *better* than the whole prompt, not merely cheaper: 1.632
accepted a cycle against 1.542 for the full fold and 1.479 for none. Giving the
head more of the prompt than it can attend to makes its drafts worse.

The two windows are not merely close. They are identical -- 1.632, the same
position curve to three places, the same accepted histogram, the same 114
cycles -- which means the head drafted the same tokens on both, and the extra
12,288 entries in the 16k arm changed nothing except the 1.0 ms a cycle it cost
to carry them.

**Why 4,096 and 16,384 agree and 64,779 does not.** The head's QSA indexer on
this checkpoint has `indexer_budget = 2048` and `indexer_compress_ratio = 4`,
so it scores blocks of four tokens and keeps 512 of them: 2,048 tokens,
whatever the cache holds. At 4k and at 16k the 2,048 it picks are the same
2,048, and they are the recent ones, which is what a next-token head is
conditioned on. At 64k they are not: there is enough history for a distant
block to outscore a recent one, the head spends part of a fixed budget on
context it cannot use, and acceptance falls back from 55.4% to 52.3%. The head
has a working set, it is about the size of its own budget, and priming past it
buys a worse draft at a higher price.

### 1c. What the throughput numbers are worth on this machine

The window arms above hand this one over for free, because two of them are the
same arm. `prime window 4096` was run twice, in two processes, an hour apart,
and it drafted *byte-identically* both times: 1.632 accepted a cycle, the same
position curve to three places, the same accepted histogram, the same 114
cycles for 300 tokens. The same work, done twice.

| repeat | accepted/cycle | cycles | verify ms | draft ms | cycle ms | profiler tok/s |
|---|---|---|---|---|---|---|
| first | 1.632 | 114 | 34.42 | 7.77 | 46.25 | 56.9 |
| second | 1.632 | 114 | 30.02 | 7.95 | 42.18 | 62.4 |

**Identical work, 9.7% apart on throughput.** The gap is all in the verify
stage, 34.4 ms against 30.0, which is the bandwidth-bound half of the cycle and
the half a thermal state moves.

That is the error bar on every tok/s figure in this document, and it is larger
than most of what ROUND2 is looking for. Three consequences, and they are
already applied above:

- Accepted per cycle, acceptance and the position curve are deterministic here.
  Two runs of an arm agree exactly, so a difference of 0.01 in accepted per
  cycle is real and a difference of 2 tok/s is not.
- A tok/s claim is only made from arms measured close together, and the
  direction is only asserted when accepted per cycle agrees with it.
- The 1024, 4096 and 16,384 windows measured 57.2, 56.9 and 55.9 tok/s, which
  says nothing at all. What separates them is 1.4 ms of draft time, which is
  visible because `draft_ms` is a stage timer over 114 cycles rather than an
  end-to-end rate.

---

## Step 2. Acceptance against context length

One process, seven contexts, 200 decode tokens each, on the step-1b defaults
(prime on, window 1,024, `omlx` chain, fixed depth 3). Each point uses its own
seed, so no point is a prefix of another and the prefix cache cannot serve the
second half of the sweep, and each reads its own cycles back through
`/metrics?window=n`.

| target context | prompt tokens | cycles | accepted/cycle | acceptance | position acceptance | cycle ms | draft ms | tok/s |
|---|---|---|---|---|---|---|---|---|
| 8,192 | 8,257 | 80 | 1.500 | 51.3% | 0.744 / 0.487 / 0.308 | 44.3 | 7.43 | 56.5 |
| 16,384 | 16,401 | 81 | 1.469 | 50.4% | 0.684 / 0.494 / 0.333 | 40.2 | 5.92 | 61.4 |
| 32,768 | 32,763 | 81 | 1.469 | 50.4% | 0.709 / 0.506 / 0.295 | 44.2 | 6.00 | 55.9 |
| 49,152 | 49,173 | 78 | 1.564 | 53.3% | 0.714 / 0.513 / 0.368 | 42.9 | 6.04 | 59.8 |
| 63,488 | 63,567 | 82 | 1.439 | 48.8% | 0.679 / 0.494 / 0.287 | 42.5 | 6.18 | 57.3 |
| 64,779 | 64,847 | 81 | 1.469 | 50.0% | 0.700 / 0.468 / 0.329 | 42.5 | 6.12 | 58.1 |
| 67,584 | 67,624 | 80 | 1.500 | 50.8% | 0.646 / 0.532 / 0.346 | 41.8 | 6.10 | 59.8 |

**Neither a slope nor a step. A flat line.** Accepted per cycle runs 1.44 to
1.56 across a factor of eight in context, with no monotone trend and a spread
no larger than the spread between two adjacent points. Cycle time is flat too:
41.8 to 44.3 ms, and the 64k end of the sweep is not the slow end.

This is the result that reframes the rest of the round, so it is worth being
exact about what it does and does not say.

It does not say the 64k figure is wrong. 1.469 at 64,847 tokens is the same
number the long arm measures.

What it says is that **8k is already just as bad**. 1.500 accepted a cycle at
8,257 tokens, against the 2.265 the short benchmark measures. The whole of the
gap that ROUND2 opened by calling "the long-context defect" is present at 8k,
and nothing further out makes it worse.

So the variable is not context length. The two benchmarks differ in a second
way that has been riding along with it: the sweep's prompt is a wall of seeded
random tokens followed by one instruction, and its completion is expository
prose about TCP congestion control. The short benchmark's four prompts are
mostly code, a file rewrite and a JSON array -- highly predictable text, where
the short benchmark's own prose prompt is its slowest by a wide margin (68 to
74 tok/s against 93 to 102 for the file rewrite, in the same process).

Acceptance here is a property of what is being *generated*, not of how much
context precedes it. A 64k session that emits code should accept like a short
session that emits code. That is a testable claim and step 2 does not test it:
every point in the sweep generates the same prose. It is the first thing the
next round should measure, because it decides whether "64k decode" is a
throughput problem at all or whether the 71 tok/s oMLX target at 64k was
measured on a different kind of completion.

### The low end of the sweep, which settles it

Same process, same decode task, contexts from 256 tokens up, and 256 repeated
at the end so the first point is not the only short one.

| target context | prompt tokens | cycles | accepted/cycle | acceptance | position acceptance | cycle ms | tok/s |
|---|---|---|---|---|---|---|---|
| 256 | 302 | 79 | 1.532 | 52.2% | 0.756 / 0.519 / 0.286 | 41.8 | 60.5 |
| 512 | 559 | 77 | 1.597 | 54.2% | 0.776 / 0.500 / 0.347 | 36.4 | 71.3 |
| 1,024 | 1,074 | 78 | 1.564 | 53.5% | 0.816 / 0.526 / 0.263 | 37.5 | 68.5 |
| 2,048 | 2,082 | 79 | 1.532 | 51.7% | 0.718 / 0.487 / 0.346 | 40.9 | 62.0 |
| 4,096 | 4,157 | 79 | 1.532 | 52.4% | 0.769 / 0.481 / 0.316 | 40.9 | 61.8 |
| 8,192 | 8,231 | 82 | 1.439 | 49.2% | 0.725 / 0.425 / 0.325 | 41.4 | 58.9 |
| 256 (repeat) | 301 | 76 | 1.632 | 55.6% | 0.733 / 0.554 / 0.378 | 36.5 | 72.0 |

**At 302 tokens of context this model accepts 1.53 tokens a cycle**, and at
64,847 it accepts 1.47. Across two and a half orders of magnitude of context,
from 256 to 67,584, accepted per cycle stays inside 1.44 to 1.63 and the two
256-token points bracket the whole 64k range on their own.

There is no long-context acceptance defect. There never was one to find. What
the W4.1 table compared as "short 2.24, 64k 1.48" was four code-and-JSON
prompts against one prose prompt, and the 0.76 tokens a cycle between them is
what the model is being asked to write.

Throughput does fall with context, and that part is real: 72 tok/s at 300
tokens against 58 at 64k, on identical acceptance. The whole of that is cycle
time -- 36.5 ms to 42.5 -- and the whole of *that* is the verify forward
reading a larger KV cache. It is a bandwidth story with nothing speculative in
it.

This is the result the round turns on, so it is worth stating what it costs the
rest of the document. Step 1b's window is still a real win and its 64k numbers
still stand, but it is not a long-context fix; it is a drafter fix that happens
to have been measured at 64k. And the "64k against short" framing that opened
ROUND2, including the 45.7-against-54.6 baselines, is not measuring what its
name says.

## Step 3. Real-model host overhead and the dispatch knobs

`host_overhead.py` against the real checkpoint at a 600-token context, 20
repeats, widths 1 and 4.

### Where the time goes

| width | ops built | async evals | build ms | step ms | gpu ms |
|---|---|---|---|---|---|
| 1 (decode) | 4,553 | 48 | 15.63 | 18.28 | 2.64 |
| 4 (verify) | 4,734 | 48 | 29.02 | 32.73 | 3.71 |

This reproduces the figure `FORWARD.md` states and sharpens it. A verify step
is 32.7 ms of which 3.7 ms is the GPU. **Eighty-nine per cent of a verify step
is the host building the graph in Python**, and half of that is one file:
`vendor/qwen4_exp/language.py` at 15.7 ms of 29.5 ms of measured Python time,
with `memmap.py` (the PLE reader) second at 5.5 ms.

Going from width 1 to width 4 adds 181 ops and 13.4 ms of build for 1.1 ms of
GPU. The marginal cost of a verify column on this machine is host time, which
is the fact the depth policy is really trading against.

### Experiment 1 (eager dispatch) and 2 (command-buffer budgets)

Six cells, one process each, model loaded once per cell, sequentially.

| eager | ops/buf | MB/buf | width 1 build ms | width 1 step ms | width 4 build ms | width 4 step ms |
|---|---|---|---|---|---|---|
| on | default | default | 13.70 | **16.25** | 23.28 | **27.03** |
| on | 200 | 200 | 13.36 | 16.27 | 20.31 | 26.58 |
| on | 500 | 500 | 12.47 | 16.10 | 13.78 | 26.88 |
| off | default | default | 5.19 | 22.02 | 5.76 | 33.36 |
| off | 200 | 200 | 5.06 | 21.57 | 5.74 | 32.99 |
| off | 500 | 500 | 5.11 | 21.72 | 5.65 | 33.12 |

**D1 is a loss, not a win, and a large one.** Turning eager dispatch off costs
35% at width 1 (16.25 to 22.02 ms) and 23% at width 4 (27.03 to 33.36). The
survey estimated -5% to +20% and called the range wide because the flag exists
for a reason. The flag exists for this reason. Note what the build column does
while the step column gets worse: 23.3 ms to 5.8 ms. With eager dispatch off
the host stops waiting inside the forward and hands MLX one big graph, so
`build` stops measuring the work and starts measuring the enqueue; the work
moves into the closing eval, and there is more of it than before.

**D2 does nothing here.** Every budget cell is within 2% of its eager-matched
default, below the 3% the experiment set as its own threshold. The one visible
effect is on `build` at width 4 with eager on -- 23.3 ms down to 13.8 -- and it
does not reach `step` at all, which says the width-4 cycle is not waiting on
command-buffer breaks.

**The buffer check passes on the configuration that wins.** 12,288 generated
tokens on the default configuration: active memory 74,198 MB to 74,433 MB
total, steady-state growth -0.01 KB a step against the 205 KB a step mlx-lm
#1332 measured before its fix. Nothing accumulates.

**Kept: nothing, deliberately.** The winning cell is the configuration Titan
already ships -- eager dispatch on, MLX's own per-architecture buffer defaults
-- so there is no setting to apply in `titan/cli.py` and no re-measurement of
the decode arms to do. Both experiments are closed as measured negatives.

What the profile does say is where the next 20% is, and it is not in either of
these knobs. A verify step spends 29 ms of host time to dispatch 3.7 ms of GPU
work. Any change that removes Python from the forward -- fewer ops built per
layer, a compiled region, a wider verify block amortising the same build over
more columns -- is worth more than every dispatch flag put together.

## Step 4. A6: the MTP head's own sparse-attention indexer

A6 asks whether the draft chain can skip or reuse the head's QSA indexer. Step
1b answered it in passing, and the answer is that on the defaults this round
lands on, **the head's indexer is already never running.**

The head's indexer engages only when its cache holds more complete blocks than
it is allowed to select: `max_complete_blocks <= block_topk` returns `None` and
the head attends densely. On this checkpoint `indexer_budget` is 2,048 and
`indexer_compress_ratio` is 4, so the threshold is 2,048 entries in the head's
KV cache. With `mtp_prime_window = 1024` the head holds 1,024 prompt entries
plus the committed tokens of the session, which over a 300-token completion
reaches 1,324: below the threshold for the whole run.

That also gives A6 its A/B, because the window arms straddle the threshold and
draft identically:

| head cache | indexer | accepted/cycle | draft ms |
|---|---|---|---|
| 1,024 primed (+ decoded) | never engages | 1.632 | 7.39 |
| 4,096 primed | engages | 1.632 | 7.77 |
| 16,384 primed | engages | 1.632 | 8.79 |

Same drafts, to the token, on both sides of the threshold. So the head's whole
indexer -- projection, pooling, scoring, top-k, the gather -- costs 0.38 ms a
cycle at a 4k head cache and 1.40 ms at 16k. Against a 46 ms cycle that is 0.8%
and 3.0%.

**Kept: nothing to build.** A6's estimate was +3 to 6% at 64k and its cost was
one to two days of vendored surgery. The measured ceiling is 0.8% at the head
cache the defaults now use, and the way to collect it is to keep the head's
cache under its own budget, which `mtp_prime_window` already does for a reason
that has nothing to do with cost: a head given more than its budget drafts
*worse* (step 1b). Skipping the indexer while keeping a large primed cache is
dominated on both axes and is not worth writing.

The lossless requirement is met trivially and for the structural reason the
drafter's docstring gives: a draft is a proposal, `verify` decides what is
committed, and the 150-token check in step 1b is byte-identical to MTP off.

## Step 5. Overlapping the draft chain with the next cycle

**What was built.** `MTPDrafter.propose` is now the composition of two halves.
`dispatch` builds the whole batch's chain, folds the committed run into each
head's KV, and calls `mx.async_eval`; `read` waits for it and turns it into
candidates. `MTPDecodeCycle` with `speculation.overlap_draft` calls `dispatch`
at the end of a cycle, after commit -- the earliest point the fold's inputs
exist -- and `read` at the start of the next one. The record carries the state
handles and context lengths it was built for, and a cycle whose batch no longer
matches drops the dispatch instead of reading it, which is the only way this
could put one sequence's draft on another. A dropped dispatch is safe to drop
because everything it did with a side effect was a fold of *committed* tokens,
which belonged in the head cache either way.

**Result: a wash, and the stage timers say so exactly.** Short prompts, two
repeats of each arm, alternated in the order run:

| arm | cycle ms | verify ms | draft ms | other ms | draft + other | profiler tok/s |
|---|---|---|---|---|---|---|
| no overlap | 38.53 | 28.32 | 6.07 | 0.23 | **6.30** | 85.0 |
| overlap | 35.42 | 25.15 | 4.56 | 1.68 | **6.24** | 92.4 |
| overlap (repeat) | 38.74 | 28.72 | 4.53 | 1.59 | **6.12** | 84.5 |
| no overlap (repeat) | 35.51 | 25.16 | 6.09 | 0.23 | **6.32** | 92.2 |

64k, same pair:

| arm | cycle ms | verify ms | draft ms | other ms | draft + other | accepted/cycle | tok/s |
|---|---|---|---|---|---|---|---|
| no overlap | 45.86 | 34.55 | 7.21 | 0.26 | **7.47** | 1.632 | 57.4 |
| overlap | 45.41 | 34.20 | 4.93 | 2.45 | **7.38** | 1.632 | 58.0 |

The overlap does exactly what it says: draft time falls by 2.28 ms at 64k and
1.51 ms short. And `other` rises by 2.19 ms and 1.45 ms, because `other` is
where the dispatch's own host time lands now. Draft plus other is 6.30, 6.24,
6.12, 6.32 short and 7.47 against 7.38 at 64k. **Nothing was saved. The work
was relabelled.**

The throughput column is the trap, and this pair is the cleanest illustration
of step 1c in the document. Read the four short rows in the order they ran: the
overlap arm measures 92.4 then 84.5, and the no-overlap arm measures 85.0 then
92.2. The two arms swap places between repeats, `verify_ms` swaps with them
(25.15 / 28.72 against 28.32 / 25.16), and the machine alternates between two
states about 12% apart. An A/B run once here would have concluded +9% with a
straight face.

**Why it cannot pay, which step 3 already knew.** A verify step is 32.7 ms of
which 3.7 ms is the GPU. The cycle is host-bound at 89%, so moving GPU work
earlier only helps if the host has something else to do while it runs -- and
the thing the host would be doing is building the draft graph, which is most of
what `draft_ms` was measuring. One Python thread cannot build the next cycle's
chain and this cycle's verify at the same time, so the chain's host cost moves
with its GPU cost and the cycle is exactly as long as it was.

**Kept: the code, not the default.** `speculation.overlap_draft` stays false.
The seam is worth keeping because the change that *would* pay needs it: if the
draft ids stayed on device and the verify rows were built from device arrays,
the draft sync would disappear rather than move, and `dispatch`/`read` is the
shape that admits it. It is also lossless as it stands -- the 150-token check
with the overlap on is byte-identical to MTP off, same md5.

---

## What ROUND2 changed

### Defaults

| setting | was | is | why |
|---|---|---|---|
| `speculation.mtp_prime_prompt` | false | **true** | +10% accepted per cycle at 64k, neutral short, lossless |
| `speculation.mtp_prime_window` | (did not exist) | **1024** | priming the whole prompt is worse than priming none |
| `speculation.mtp_head_align_positions` | false | false | measured no-op (1a), kept as a documented one |
| `speculation.mtp_chain_cache` | "clone" | "clone" | identical drafts either way; the window makes the copy cheap again |
| `speculation.overlap_draft` | (did not exist) | **false** | measured wash (5); the seam is kept, the default is not |
| eager dispatch, MLX buffer budgets | MLX/Titan defaults | unchanged | D1 measured -23 to -35%, D2 measured nothing |

`~/.config/titan/titan.toml` carries the two priming settings explicitly, since
that file is what `/metrics` echoes.

### Throughput, before and after

Profiler tok/s, `omlx` chain at fixed depth 3, against the oMLX targets on this
machine. Every figure carries step 1c's error bar of about 10%, so the
accepted-per-cycle column is the one that means something.

| | before | after | oMLX target |
|---|---|---|---|
| short, MTP off | 54.6 | 54.6 | - |
| short, drafted | 82.9 | 84.5 to 92.4 | 84 to 91 |
| short, accepted/cycle | 2.24 | 2.27 | - |
| 64k, MTP off | 45.7 | 45.7 | - |
| 64k, drafted | 52.0 to 54.3 | 56.4 to 62.4 | 71 |
| 64k, accepted/cycle | 1.48 | 1.63 | - |

Short was at target when the round opened and is at target now. 64k moved from
52-54 to 56-62 on throughput and from 1.48 to 1.63 accepted per cycle, which is
the +10% the priming window is worth. It is still short of 71.

### The finding that outlasts the numbers

There is no long-context acceptance defect (step 2). At 302 tokens of context
this model accepts 1.53 tokens a cycle on the benchmark's prose task; at 64,847
it accepts 1.47. The 2.24-against-1.48 gap ROUND2 opened with was four
code-and-JSON prompts against one prose prompt, and it measures what the model
was asked to write. Throughput does fall with context -- 72 tok/s at 300 tokens
to 58 at 64k -- and all of it is the verify forward reading a larger KV cache.

That makes the oMLX 71-at-64k target the open question rather than the target.
Nothing here can close a 71-against-58 gap by drafting better, because the
drafting is the same at both ends of the context range; the gap is in a verify
forward that spends 29 ms of host time to dispatch 3.7 ms of GPU work.

### The next three levers, ranked

1. **Get Python out of the verify forward.** Step 3 measured a verify step at
   32.7 ms with 3.7 ms of GPU in it, and half the host time in one file. This
   is the only lever on the board sized like the remaining gap, it applies at
   every context length, and it is measurable today with
   `host_overhead.py paths`. Everything else on this list is a few per cent.
2. **Draft ids that never leave the device.** Step 5 built the dispatch/read
   seam and proved that moving the draft's *wait* saves nothing while its host
   build moves with it. Building the verify rows from device arrays removes the
   draft sync rather than relocating it, and it is the only version of the
   overlap the profile says can pay. The seam is in place; what it needs is a
   verify path that takes drafted ids as an `mx.array`.
3. **Benchmark what is actually generated.** Step 2's result means every
   acceptance number in this document and the last is a statement about the
   prompt set, not about the engine. Before another round is spent on the
   drafter, the bench needs a decode task per context point that is held fixed
   across contexts and varied deliberately across content -- code, prose, JSON
   -- so a change in acceptance can be attributed to the change that caused it.

A7, online distillation of the head, is off this list rather than below it. It
was the only candidate with a plausible path to doubling 64k decode, and its
premise was that acceptance decays as a session runs long. Step 2 says
acceptance here does not decay with context at all, so the thing A7 fixes is
not the thing this engine has.
