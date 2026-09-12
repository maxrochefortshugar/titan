# ROUND3: the per-token cost at long context, by layer type

M5 Max, 128 GB, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`, greedy, one model
instance at a time on 127.0.0.1:8085. Written as the measurements were taken.

ROUND2 closed on a number it could not explain. There is no long-context
acceptance defect: the drafter accepts 1.53 tokens a cycle at 300 tokens of
context and 1.47 at 64,847, so the whole of the throughput fall with context is
cycle time. Plain decode is 18.3 ms a cycle at 600 tokens and 21.9 at 64k;
oMLX's plain cycle at 64k is 14 ms. Titan pays about 8 ms a token more at 64k
and 2 ms more at short, and ROUND2's last line was that the gap lives in a
verify forward spending 29 ms of host time to dispatch 3.7 ms of GPU work.

This round asks which part of the forward that is, and fixes what the answer
exposes.

## Before anything else: the harness that panicked the machine

The previous agent's `layer_breakdown.py` ran on the real checkpoint at 64k
with an evaluation barrier and an `mx.synchronize()` inside every scope, in a
loop that rewound the cache after each step. The machine took a watchdog panic
about ninety seconds later, the second GPU stall that day
(`docs/ops/INCIDENTS.md`).

The script survived and the rule that followed did not, so the rule is now in
the script rather than beside it:

- `--sync-level top` and `--sync-level all` are **refused** with `--model`, by
  name, citing the incident. There is no way to ask this file for per-scope
  barriers on the checkpoint.
- The real-checkpoint mode times the Python call only and synchronises exactly
  once per step, at the end. Nothing inside a step is evaluated out of order,
  so the graph the GPU sees is the graph an ordinary step builds.
- Per-scope attribution moved to a synthetic configuration that is worth
  measuring: `--config real-shapes` carries the checkpoint's per-layer shapes
  (hidden 2560, 24 query heads over 2 K/V heads at head dimension 256, the
  4-head indexer at dimension 128 with budget 2048 and ratio 4, the 48/16-head
  Gated DeltaNet, the 4-wide hyper-connection at low rank 320) with the layer
  count cut to 8 and the expert count to 16. A QSA layer there costs what a QSA
  layer on the checkpoint costs, which is the only thing this round needed.

Everything below was produced by that file or by the server, and every log is
under `bench/decode/round3/`.

## How to reproduce

    # attribution, synthetic, per-scope barriers allowed
    python bench/decode/layer_breakdown.py --config real-shapes \
        --contexts 600,64000 --widths 1,4 --repeats 10

    # elimination: the same steps with one arm turned off, paired A B B A
    python bench/decode/layer_breakdown.py --config real-shapes \
        --contexts 2052,4096,6144,8192,16384,64000 --widths 1,4 \
        --repeats 20 --eliminate --sync-level none

    # the real model, whole-step host time only, one sync per step
    python bench/decode/layer_breakdown.py --model <checkpoint> \
        --context 600 --widths 1,4

    # the real model, end to end, one arm per invocation
    bash bench/decode/round3/ab.sh

---

## Step 1. Which arm does each forward actually take

The engine runs three different one-model forwards and, before this round,
they took three different attention paths. That turned out to be most of the
story, so it comes first.

| forward | what runs it | `target_verify` | arm it took |
|---|---|---|---|
| decode, width 1, no hidden state | `speculation.enabled = false` | no | gathered sparse |
| decode, width 1, hidden state | the depth-0 speculative cycle, and any `decode(want_hidden=True)` | **yes** | **dense masked** |
| verify, width > 1 | every speculative cycle | yes | gathered sparse |

The middle row is a routing hole rather than a design. `return_hidden` sets
`capture_layer_ids`, which sets `target_verify` on every layer, which is how
the Gated DeltaNet knows to capture the intermediates a rollback replays from.
Attention has no intermediates to capture, so the flag says nothing about it --
but `_gathered_text_decode_eligible` tested `not target_verify`, so a one-row
forward that asked for a hidden state read the whole cache on all twelve QSA
layers.

`--arm decode-hidden` in `layer_breakdown.py` is that forward, and it is why
the arm list exists.

## Step 2. Where the extra time is, by elimination

Per-scope barriers cut the graph at every scope boundary, which blocks the
fusion and the host/GPU overlap a real step gets. That makes the per-scope GPU
column an upper bound rather than a marginal cost, and the floor is not small:
at a 600-token context the `update_indexer` scope reports 0.39 ms of GPU for
two calls that write 128 numbers, so a barrier costs about 0.19 ms of its own.
Read the barriered table for *which arm ran*, and read the elimination table
for *what it cost*.

### 2a. The barriered table says which arm, and roughly where

Real-shapes synthetic, 8 layers of which 2 are QSA, context 600 against 64,000,
GPU column from the barriered pass. Only the QSA rows are reproduced; every
other group is flat in context to within the noise, which is itself the result.

| group | 600, decode | 64k, decode | 64k, decode-hidden | 64k, verify w4 |
|---|---|---|---|---|
| `arm.dense_sdpa` | 0.61 | -- | 0.89 | -- |
| `arm.dense_indexer` | 0.04 | -- | 0.66 | -- |
| `arm.gathered_decode` | -- | 0.10 | -- | -- |
| `arm.gathered_block` | -- | -- | -- | 0.81 |
| `gather` | -- | 1.01 | -- | 1.37 |
| `scores` | -- | 0.44 | -- | 0.50 |
| `pool` | -- | 0.08 | 0.08 | 0.43 |
| `update_indexer` | 0.39 | 0.39 | 0.39 | 0.40 |
| whole step, unbarriered | **3.73** | **4.37** | **4.44** | **5.87** |

Two things to take from it. The QSA layers are the only ones that change with
context, and at 64k the `decode-hidden` arm is on `dense_sdpa` while
`decode` is on `gathered_decode` -- the routing hole, visible.

The whole-step numbers scale honestly to the checkpoint. Two QSA layers here
against twelve there: 4.37 - 3.73 = 0.64 ms for two layers, so 3.8 ms for
twelve, against the 3.6 ms the real plain cycle actually gains between 600 and
64k. The synthetic is measuring the right thing.

### 2b. The elimination table says what each arm is worth

Whole-step milliseconds, real-shapes synthetic, twenty repeats, arms alternated
A B B A inside one process so thermal drift cannot favour one. `default` is
this round's configuration; each other row turns one thing off and re-measures
the same step. The number after each cell is the difference from `default`.

Multiply by six for the checkpoint's twelve QSA layers.

| context | arm | w1 plain | w1 hidden | w4 verify |
|---:|---|---|---|---|
| 2052 | default | 3.89 | 3.92 | 5.45 |
| | dense-qsa | +0.01 | +0.02 | -0.00 |
| | gathered-at-budget | **+0.08** | **+0.12** | **+0.20** |
| 4096 | default | 3.91 | 3.98 | 5.69 |
| | dense-qsa | +0.00 | +0.00 | +0.00 |
| | gathered-at-budget | **+0.08** | +0.04 | -0.02 |
| 6144 | default | 3.92 | 3.98 | 5.69 |
| | dense-qsa | +0.02 | -0.00 | **+0.21** |
| | gathered-at-budget | +0.04 | +0.04 | -0.00 |
| 8192 | default | 3.98 | 4.02 | 5.70 |
| | dense-qsa | -0.05 | -0.01 | **+0.58** |
| | gathered-at-budget | +0.01 | +0.02 | +0.02 |
| 16384 | default | 4.03 | 4.03 | 5.74 |
| | dense-qsa | -0.02 | -0.00 | **+1.63** |
| 64000 | default | 4.15 | 4.15 | 5.83 |
| | dense-qsa | **+0.21** | **+0.24** | **+8.02** |
| | no-singleton-verify | -0.03 | **+0.23** | -0.01 |

Four results, and they are the whole of step 2.

**The gathered verify arm is the largest single thing in the forward at 64k,
and it was already on.** Turning it off costs 8.02 ms for two QSA layers, so
about 48 ms over twelve. Titan's 64k verify at width 4 measures 30 to 34 ms;
without this arm it would be near 80. Nothing in this round improves it and
nothing should touch it without measuring this cell first.

**The one-row target-verify hole is worth 1.4 ms a forward on the checkpoint.**
`no-singleton-verify` is +0.23 for two layers at 64k on exactly the arm it
affects, and zero on the other two, which is what a correct routing change
looks like: it moves one arm and no others.

**The vendored crossover fires thousands of tokens too early.** The gathered
arm becomes *legal* past the QSA budget, 2048, because that is where there are
more complete blocks than the indexer may select. It does not become *cheaper*
there: it reads a flat 2,051 rows whatever the context, so at 2048 it is
reading more than the dense arm and paying for selection on top.
`gathered-at-budget` is the old gate, and it costs 0.08 to 0.12 at width 1 and
0.20 at width 4 at 2052.

**The two widths cross over in different places.** At width 1 the gathered arm
is still behind at 6144 (+0.04) and level at 8192. At width 4 it is level at
4096 and 0.21 ahead by 6144. The dense arm's cost rises with the block width
and the gathered arm's barely does, so a four-row block is worth gathering
thousands of tokens before a one-row decode is. That is two numbers, not one.

**What is left after all of it.** `default` at 64k against `default` at 2052 is
4.15 against 3.89 for a plain decode: 0.26 ms for two QSA layers, 1.6 ms over
twelve. That is the sparse arm's own growth with context -- the pooled bank the
indexer scores against is 16,000 slots at 64k and 512 at 2052, and the fp32
cast of that bank plus the `argpartition` over it are the parts that scale.
It is the residue this round does not remove, and section 5 says what would.

---

## Step 3. What was wired

### 3a. `kernels.reference_only` now reaches the forward

This is a prerequisite rather than an optimisation, and it was quietly broken.
`titan.config.wiring` builds a registry from the `kernels` section and hands it
to the engine; `build_backend` ignores it, and
`titan/adapters/mlx/kernels.py` built its own with default settings. So
`kernels.reference_only = true` disabled nothing inside the forward. The
control arm of every kernel A/B was not a control, and INCIDENTS rule 2 could
not be followed as written.

`build_registry` and `reference_only` now publish the registry they build
(`titan.kernels.registry.set_current`), and the adapter prefers the published
one. A configuration that turns an op off makes the adapter's `get` return
`None`, which is its documented contract for "no kernel at this site, keep the
stock path" -- not the op's own reference, which for the gathered-QSA op is
still the sparse algorithm.

The measurement that proves it took: the same short benchmark, same tree, same
session, `kernels.reference_only = true` measures 73.1 median wall tok/s
against 95.4 with only `qsa_gathered_attention` enabled. Before this change
both cells would have measured the same thing.

### 3b. The one-row target-verify forward takes the sparse arm

`_gathered_text_decode_eligible` no longer refuses `target_verify`. At one row
the projections are the same call either way -- `_target_verify_linear` short
circuits a quantised projection at batch one and every attention projection on
this checkpoint is quantised -- so the only difference is which attention arm
runs, and the two agree bit-for-bit on these shapes at width 1
(`tests/model/test_qsa_arms.py::test_the_shipped_arms_are_the_same_distance_apart`).

Route: `qsa_sparse_singleton_verify`, default on.

### 3c. The crossover is a route, and it has two values

`_gather_min_context(token_budget, width)` returns the context from which a
gathered arm is preferred, clamped up to the budget because below the budget
the arm has nothing to select. Defaults from the table in 2b: 8192 at width 1,
4096 above it.

Routes: `qsa_gather_min_context`, `qsa_gather_min_context_verify`.

### 3d. The batched sparse arm, for `BatchQSAKVCache`

This is the gap `titan/adapters/mlx/VENDORED.md` recorded: the kernel existed
(`titan/kernels/qsa_gathered_attention.py`) but it wants the pooled index bank
in phased slot space, and the vendored attention did not build one, so
`qsa.gathered_batched` stayed unwired and every batched row read the whole
cache.

What was needed was the bank. Rows are left padded and right aligned, so row
`b` with `pads[b]` dead columns starts its own four-token block grid at column
`pads[b]`, and two rows whose padding differs by something that is not a
multiple of the ratio have block grids in different *phases*. There is no
single pooled key that serves both. The bank is therefore indexed in phased
slot space: slot `j` of row `b` pools physical columns `ratio*j + phase_b`
through `+ ratio - 1`, and row `b`'s own logical block `k` is slot
`k + pads[b] // ratio`. Selection runs in slot space and only the final gather
converts to physical columns, which is what makes the batched output equal the
per-row output.

`BatchQSAKVCache.pooled_index_slots` builds it incrementally. A slot is
*settled* once every row's columns for it have been written, which is
`(width - max_phase) // ratio` slots; settled slots are written once and never
recomputed, and the one or two past that are recomputed every step in place,
because a row with a small phase completes a slot a step before a row with a
large one. Recomputing a slot early is harmless rather than wrong: a slot that
is not yet complete for a row is masked out for that row by the op.

Route: `qsa_batched_sparse`, default on. Op: `qsa.gathered_batched`, which
`kernels.disabled` and `kernels.reference_only` both switch off, sending
batched rows back to the dense path.

### 3e. Two things the batched path was doing that it should not

**An `mx.concatenate` of the whole indexer history, per layer, per step.**
`BatchQSAKVCache.update_indexer` appended by concatenating, which copies
16.6 MB a layer at 64k -- 200 MB a step over twelve layers -- to add 128
numbers. It is a capacity buffer now, as the singleton cache has been since
`_QSAIndexerCache` landed. The buffer is private and `index_keys` stays the
logical view, so `extend`, `filter`, `trim` and the state setter can keep
replacing the arrays outright; the update notices and starts a fresh buffer
rather than writing into one that no longer backs them.

**A `TypeError` on every batched decode past a QSA layer.**
`Qwen4ExpQSAIndexer.from_projected` built default positions with
`mx.arange(past_len, past_len + seq_len)`, and a batched cache's `offset` is one
logical length per row, an `mx.array`, which `mx.arange` does not take. So two
or more sequences decoding concurrently did not fall back to a slow dense path,
they raised. The positions are now built row-wise, exactly as
`Qwen3_5Attention` already builds them for the same case. This is a crash, not
a cost, and it was found by writing the "arm switched off" test.

---

## Step 4. The real model, before and after

Two source trees, both built from HEAD with `git archive` so the concurrent
engine, scheduler and config work in the repo cannot move these numbers; the
"after" tree is that plus the four files this round changed
(`titan/kernels/registry.py`, `titan/kernels/__init__.py`,
`titan/adapters/mlx/kernels.py`, and the vendored `qwen4_exp/language.py`).
One arm per server, one server at a time, order A B B A within each mode.

INCIDENTS rule 2 was followed before either arm ran: `kernels.reference_only =
true` at short context as the control, then `kernels.enabled =
["qsa_gathered_attention"]` at short context, then 64k.

### The arms

Profiler tok/s and cycle milliseconds, from `GET /metrics`. The wall rate is in
`~/inference-server/staging/round3/results.jsonl` and says the same thing more
noisily. Short is four prompts at 600 tokens; 64k is one seeded cold prompt of
64,779 tokens followed by 300 tokens of decode. Every cell is the median of the
runs listed after it.

| arm | context | before | after | oMLX on this machine |
|---|---|---|---|---|
| plain decode, cycle ms | 600 | 17.38 (n=2) | 17.37 (n=2) | -- |
| plain decode, tok/s | 600 | 57.6 | 57.6 | -- |
| plain decode, cycle ms | 64k | 21.47 (n=4) | **20.13** (n=4) | 14 |
| plain decode, tok/s | 64k | 46.6 | **49.6** | 71 |
| drafted, cycle ms | 600 | 35.71 (n=2) | 33.31 (n=2) | -- |
| drafted, tok/s | 600 | 86.2 | 88.5 | 84 to 91 |
| drafted, cycle ms | 64k | 40.27 (n=2) | 40.28 (n=2) | -- |
| drafted, tok/s | 64k | 59.5 | 58.9 | 71 |

**Three of the four cells are unchanged, and that is what the routing
predicts.** None of what this round wired is reachable from a single stream at
600 tokens or from a drafted cycle at 64k:

- at 600 tokens the context never passes the QSA budget, so no arm changes;
- a drafted cycle at 64k is a verify block two to four rows wide, which was
  already on the gathered arm, plus a draft chain whose head cache is windowed
  to 1,024 tokens and never reaches the budget at all;
- the one-row target-verify fix needs a one-row forward that asks for a hidden
  state, which the depth policy produces only on a depth-0 cycle, and
  `mean_rows` never fell below 3.13 in these runs;
- the batched arm needs two sequences decoding in lockstep.

The short-context pair is the cleanest illustration of ROUND2 step 1c in this
document. Four plain-decode runs in order before, after, after, before measured
55.1, 58.9, 55.1, 58.9 tok/s: the machine alternated by run index, not by arm,
and the two arms have identical medians.

**The 64k plain-decode cell did move, and I cannot say why.** Four runs an arm,
order balanced across two scripts: before 21.38, 21.62, 21.56, 20.21 ms; after
20.39, 19.11, 20.24, 20.02. Seven of the eight pairwise comparisons favour the
after tree and the medians are 1.34 ms apart, 6.2%. But with
`speculation.enabled = false` there is no MTP module, so `draft_depth_max` is
zero, so `decode` is called with `want_hidden=False`, so `target_verify` is
false and **both trees take the same gathered decode arm**. The crossover does
not fire at 64k and the batched arm needs two streams. There is no mechanism in
the diff that reaches this arm, so this is recorded as an unexplained
difference and not claimed as a win. If it is real, the next round should find
it with `--eliminate` on the real model rather than by rerunning this.

### Lossless

Byte-identical at 150 tokens, and the md5 is the same one the four W4.1 arms
and every ROUND2 arm produced:

    md5 lossless-r3_before.txt lossless-r3_after.txt
    -> e54dbfb37cac5d18d6c942f8e407cf4b, twice

### The control arm, which is now a control

The 2x2 that proves section 3a took, all short context, all in one session:

| tree | `kernels.reference_only` | wall tok/s | cycle ms |
|---|---|---|---|
| before | false | 88.6, 87.4 | 36.06, 35.36 |
| before | **true** | 87.3 | 36.57 |
| after | false | 88.9, 86.1 | 32.23, 34.38 |
| after | **true** | 93.5 | 34.13 |
| after | only `qsa_gathered_attention` enabled | 95.4 | 33.56 |

On the before tree, turning every kernel off changes nothing, which is the
defect. On the after tree it changes the throughput, which is the fix.

It also says something nobody was looking for. On the after tree, **turning the
registry's fast kernels off makes the model faster**: 93.5 and 95.4 against
87.5 with them on, two measurements of each, an hour and a half apart. That is
about 7%, which is inside ROUND2's error bar for a single pair and outside it
for four. The first suspect is `moe.gather_qmm_int8`, which
`titan/adapters/mlx/VENDORED.md` records as approximate and as costing 11.3 GB
of extra residency on a machine with 128 GB. This is now a one-line bisect --
`--set kernels.disabled=["moe_gather_int8"]` -- which it was not before this
round, and it should be the next thing anyone measures.


---

## Step 5. What remains

**The 64k residue is a kernel problem now, not a routing one.** With every arm
routed correctly, a plain decode at 64k still costs 1.6 ms more over twelve QSA
layers than the same decode at 2052, and the two parts that scale are the fp32
cast of the pooled block bank in `_portable_indexer_scores` and the
`argpartition` over the scores. At 64k that bank is 16,000 slots of 128 bf16
values a layer, cast to fp32 every step and read twice. oMLX does not pay it:
`qsa.indexer_scores` and `qsa.topk_indices` consume the bf16 bank directly.
Those are two of the four narrow seams `VENDORED.md` lists as unwired and they
are the next lever, in that order. The cheap half of it is a persistent fp32
bank on the cache, which removes the cast without a Metal kernel; it was not
built here because the elimination says the whole prize is under 2 ms.

**Bisect the registry.** Section 4's control says the fast kernels cost about
7% at short context. That is larger than everything this round wired, it now
has a one-line switch per op, and nobody has looked.

**The batched arm has no real-model measurement.** It is proved correct on
synthetic tensors at the checkpoint's shapes, bit-for-bit against the op's
per-row reference through the whole attention module at widths 1 to 6 and
across the 2048 and 8192 crossings (`tests/model/test_qsa_arms.py`), and its
bounds are tested at 64k by width 8 (`tests/kernels/`). What it does not have
is a throughput number on the checkpoint, because reaching it needs two
sequences decoding in lockstep and this round could not establish that the
current scheduler produces that: two concurrent 12k-token streams completed on
both trees, with and without speculation, and the drafted cycle never batches
at all -- `MLXModelBackend.verify` dispatches one forward per sequence, which
its own comment records as W4.2. Until W4.2 lands, the batched arm is
insurance rather than throughput. `bench/decode/round3/concurrent_check.py` is
the harness to point at it when it does.

**One crash, found by writing the switched-off test.**
`Qwen4ExpQSAIndexer.from_projected` raised a `TypeError` on any batched cache,
because `mx.arange` will not take the per-row offset array a `BatchQSAKVCache`
carries. It is reachable through the production `to_batch`/`extend` API and the
test exercises it; whether today's scheduler reaches it was not established.
Fixed either way.

**Two files this round could not edit and should be corrected.**
`titan/adapters/mlx/VENDORED.md` still says `qsa.gathered_batched` is unwired,
which is now wrong: it is wired, at `Qwen4ExpAttention._batched_sparse_qsa`,
for batched decode and batched verify. And the three new routes belong in
`models/forward_paths.py` with the rest of the arm switches rather than in
`titan/adapters/mlx/kernels.py`, which is where they are.

