# ROUND4: which fast kernels actually pay, and what the compiled blocks cost on the checkpoint

M5 Max, 128 GB, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`, greedy, one model
instance at a time on 127.0.0.1:8085. Written as the measurements were taken.

ROUND3 ended with a control arm that finally worked and an unexplained result
from it: with every fast kernel off, the model measured 93.5 wall tok/s at
short context against 87.5 with them all on, and with only
`qsa_gathered_attention` on it measured 95.4. Turning the kernel library off
made the model faster by about 7%, nobody had looked at why, and ROUND3's own
closing line was that this should be the next thing anyone measures.

This round measures it, one op at a time.

## How to reproduce

    # the bisect: the control, each fast op alone, the production default,
    # at short context and at 64k, twice each, alternated
    bash bench/decode/round4/bisect.sh
    python bench/decode/round4/summarise.py

    # one arm on its own
    bash bench/decode/round4/arm.sh refonly short,long,lossless 0 \
        --set 'kernels.reference_only=true'

    # the compiled path, synthetic parity and the buffer check
    python bench/decode/host_overhead.py buffers --tokens 12288 --compiled
    python bench/decode/compiled_path.py real \
        --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
        --contexts 600,64000 --widths 1,4 --repeats 20

Every log and every result is under `bench/decode/round4/`.

## The harness, and why it is not ROUND3's

`bench/decode/round4/arm.sh` is `run_round2.sh`'s stop-and-wait logic
unchanged, for its reasons: closing the port is not the process exiting and the
process exiting is not the 73 GB coming back, and starting the next arm before
it does overlaps two model images and swaps. What is different is that a kernel
configuration is fixed at startup, so a bisect of eleven ops at two contexts
with two repeats is fifty-two model loads if each mode gets its own server.
This runs every mode inside one server lifetime and costs thirteen.

`probe.py` imports `mtp_ab.py`'s client, prompts and metrics view rather than
copying them, so a ROUND4 number and a ROUND3 number are the same measurement.
Short is the same four 600-token prompts; 64k is the same seeded cold prompt of
64,779 tokens followed by 300 tokens of decode.

Ordering is paired alternation, and it is the ordering rather than a flag: the
arm list runs forwards for repeat 0 and backwards for repeat 1, so an arm at
position *i* in the first pass sits at position *2N-i* in the second and drift
that is monotone in time cancels in the pair. ROUND2 step 1c measured that
drift at 10 to 16% on this machine, which is larger than most of what this is
looking for.

Cold prefill is derived from the 64k run rather than measured separately. The
prompt is cold and the prefix store is off, so the request's wall time is
prefill plus decode, and decode is the profiler's own cycle count times its own
cycle time. Subtracting leaves the prefill seconds for the same 64,779 tokens
every arm sees. It carries the request's tokenisation with it, so it is a
number to compare across arms rather than the kernel's own prefill rate. It is
free this way, which is why every arm has one rather than only the four
prefill-oriented ops.

---

## Step 1. The bisect, and the thing that had to be removed first

### 1a. The production configuration cannot attribute anything

The first pass ran the production configuration, which is what the round asked
for: drafter on, `speculation.adaptive_depth = true`. Thirteen arms, two
contexts, two repeats each, alternated. Here are the two repeats of four of
those arms at short context, and they are the reason this section exists.

| arm | repeat 0 | repeat 1 |
|---|---|---|
| `refonly` | 93.5 tok/s, 3.60 rows, 2.133 accepted | 79.0 tok/s, 2.97 rows, 1.688 accepted |
| `all_on` | 85.9, 3.38, 1.956 | 81.4, 2.72, 1.487 |
| `only_moe_gather_int8` | 87.8, 3.71, 2.187 | 81.2, 2.50, 1.321 |
| `only_moe_gather_ws` | 88.1, 2.97, 1.715 | 88.3, 2.97, 1.712 |

Same tree, same configuration, same prompts, fourteen tok/s apart on the
control arm. The column that moves with the throughput is `mean_rows`: the
depth the drafter chose. `AdaptiveDepth` picks it by argmax over
`E[committed | k] / cycle_ms(k + 1)`, and `cycle_ms` there is a *measured*
model fed by `observe(width, wall_ms)`. So the depth an arm settles on is
downstream of how fast that arm happened to run, accepted-per-cycle is
downstream of the depth, and throughput is downstream of both. The loop has
more than one resting point and a run picks one early and stays in it: the last
row above is an arm that landed on the same point twice and reported the same
number twice, to two decimal places.

This is what ROUND3 section 4 was looking at when it recorded that turning the
fast kernels off made the model 7% faster and called it the next thing to
measure. Its 93.5 against 87.5 is the first row of this table against the
second, and the pair 93.5 / 79.0 is the same arm against itself. There is no
7% kernel effect in it to find.

So the production table is reported below because it is what an instance
delivers, and nothing is attributed from it.

### 1b. The bisect, with the depth pinned

`bench/decode/round4/fixed_depth.sh` is the same thirteen arms with
`speculation.adaptive_depth = false` and `mtp_depth_min = mtp_depth_max = 3`,
which is W4.1's `b_chain_d3`. The feedback loop is gone and every arm reports
`accepted_per_cycle` of exactly 2.310 and `mean_rows` of exactly 3.97, at short
context, on every arm and every repeat. That equality is the check that the
arms are now comparable: the drafter is doing identical work in all of them, so
what is left in the cycle time is the kernel.

Read this table for what an op costs. Read 1c for what an instance delivers.

Profiler tok/s, median of two repeats, `fd_refonly` as the control. Short is
four 600-token prompts; 64k is the seeded 64,779-token cold prompt then 300
tokens of decode; prefill is the same prompt's cold prefill rate.

| arm | short tok/s | vs control | 64k tok/s | vs control | cold prefill tok/s |
|---|---:|---:|---:|---:|---:|
| `reference_only` (control) | 87.2 | -- | 65.0 | -- | 830 |
| `gdn_norm_gate` | 89.1 | **+2.1%** | 66.6 | **+2.5%** | 839 |
| `moe_gather_ws` | 89.0 | **+2.1%** | 66.1 | **+1.8%** | 830 |
| `moe_weighted_sum` | 89.0 | **+2.0%** | 66.1 | **+1.7%** | 833 |
| `verify_accept` | 88.5 | **+1.5%** | 66.2 | **+1.9%** | 825 |
| `topk_radix` | 88.5 | **+1.5%** | 65.8 | **+1.2%** | 825 |
| `hc_prefill` | 88.4 | **+1.4%** | 66.3 | **+2.1%** | 827 |
| `qsa_gathered_attention` | 86.7 | -0.6% | 65.9 | +1.5% | 826 |
| `grouped_rmsnorm_bf16` | 88.3 | +1.3% | 59.8 | **-7.9%** | 853 |
| `gdn_chunk_scan` | 88.2 | +1.2% | 56.3 | **-13.2%** | 850 |
| `moe_gather_int8` | 88.2 | +1.1% | 62.5 | **-3.7%** | 826 |
| `ple_packed_lookup` | 83.7 | **-4.0%** | 62.7 | **-3.5%** | 798 |
| every op on | 84.1 | **-3.6%** | 58.6 | **-9.7%** | 872 |

Six ops pay for themselves at both contexts, by one to two and a half per cent
each. One is neutral. Four cost throughput, and three of those cost it only at
64k:

**`gdn_chunk_scan` is the largest single item, at -13.2% at 64k.** It is a
prefill kernel -- the C = 8 chunked delta-rule scan -- and
`gated_delta_update` offers it every prefill-shaped call with scalar gating and
no mask. A 64k decode is not prefill, but a 64k *request* is, and the arm that
is slower here is the one whose prefill ran through the chunked scan. Its own
cold-prefill column is 850 against the control's 830, so it buys 2.4% of
prefill and gives back 13.2% of decode.

**`grouped_rmsnorm_bf16` at -7.9% and `moe_gather_int8` at -3.7%, both only at
64k.** `moe_gather_int8` is the op ROUND3 named as the first suspect, and it is
a real cost, though a third of what ROUND3's uncontrolled pair suggested. It is
also the op `VENDORED.md` records as costing 11.3 GB of extra residency for
approximate arithmetic, so the memory argument and the throughput argument now
point the same way.

**`ple_packed_lookup` is the only op that loses at both contexts**, -4.0% and
-3.5%, and it has the worst cold-prefill column of the thirteen arms at 798
against 830. The packed reader is bit-exact and the loss is not numerics; the
n-gram table is streamed from SSD by rows either way, and on this machine the
kernel's device-side dequantise is not buying back what its host-side page
reads cost.

**Every op on is worse than every op off, at both contexts.** That is ROUND3's
observation, reproduced under a controlled drafter, and it is now attributable:
-3.6% short and -9.7% at 64k is roughly the four losers, and the six winners do
not cancel them.

### 1c. The subset, and the prefill trade it exposes

Two subsets were measured against the same control, at pinned depth. `decode`
is the six winners plus the neutral `qsa_gathered_attention`. `prefill` is that
plus `gdn_chunk_scan` and `grouped_rmsnorm_bf16`, the two losers that were
built for prefill and whose own cold-prefill columns are the best of the
single-op arms. `moe_gather_int8` is in neither: it is a 64k loser *and* has a
below-control prefill column, so nothing is being traded for it.

| arm | short tok/s | 64k decode tok/s | cold 65k prefill tok/s |
|---|---:|---:|---:|
| `reference_only` | 87.2 | 65.0 | 830 |
| every op on | 84.1 | 58.6 | 872 |
| **subset, decode** | **88.8** | **65.8** | 840 |
| subset, prefill | 86.2 | 57.8 | **892** |

The decode subset is the best decode arm measured at both contexts and it beats
the shipped default by 5.6% short and 12.3% at 64k. It is also the arm whose
150-token completion is byte-identical to every other arm's.

The prefill subset is the honest complication, and it is the reason the round
asked for a prefill column at all. Putting the two prefill ops back buys 6.2%
of cold prefill and gives back 12.2% of 64k decode. On the 64,779-token prompt
those are 72.6 seconds of prefill against 77.1, and 17.30 ms a token against
15.20. They cross at about **2,000 output tokens**: below that the prefill
subset finishes the request sooner, above it the decode subset does.

The default set here is the decode subset, for two reasons and against the
third. The metric this workstream is measured on is decode tok/s against
oMLX's 71, and the decode subset is the only arm that moves toward it. And a
64k agentic turn on this node is usually well past 2,000 output tokens. Against
that: a prefill-dominated workload -- long context, short answers, retrieval --
should set `kernels.enabled` to the prefill list, and the table above is what
it is buying. That is one line in `titan.toml` and it is documented next to the
defaults.

**What is not established is why a prefill kernel costs a decode.**
`gdn_chunk_scan` fires on prefill-shaped calls with scalar gating and no mask,
which a decode step is not, and it costs nothing at short context, where the
600-token prefill never reaches the shape it fires on. Its whole -13.2% appears
only in the arm whose prefill ran through it. The obvious suspect is
residency -- `moe_gather_int8` is already known to hold 11.3 GB of qsum tables,
and a scan that allocates chunk workspace during a 64k prefill and never frees
it would slow every decode step after it without ever running again. That is a
mechanism, not a measurement: nothing here measured resident bytes per arm. It
is the second lever in section 5 and it is worth more than either subset,
because if it is right then freeing the tables after prefill gives both
columns at once rather than making anyone choose.

### 1d. The production configuration, for the record

Adaptive depth back on, three repeats each, alternated. These are delivered
numbers rather than attribution, for the reason in 1a.

| arm | short tok/s | 64k decode tok/s | cold prefill tok/s | 150-token digest |
|---|---:|---:|---:|---|
| every op on (the shipped default) | 83.0 | 61.2 | 872 | `34a52eec` |
| the subset, by `--set` | 84.6 | 64.9 | 826 | `34a52eec` |
| the subset, from `titan.toml` | 84.1 | 65.5 | 831 | `34a52eec` |

The third row is an accident that turned into a check. `titan.toml` gained the
winning subset while the last arm of the pass was in flight, so an arm named
`prod_all_on` ran the subset instead. Every row in `results.jsonl` records the
kernel configuration the server resolved, which is what made that visible, and
the row is renamed to what it measured. Two independent routes to the same
configuration, one through `--set` and one through the file, land within 0.6%
of each other at both contexts, which is what says the file reproduces the arm.

### 1e. Lossless

Every arm in this round produced the same 150 tokens, byte for byte:

    34a52eecda614590e6c6a07c41ae424b

Sixteen arms, thirteen ops, both subsets, the control, both depth policies.
That is the answer to the only question the numerics had to answer: nothing
selected here changes what the model says. It is a different digest from
ROUND3's `e54dbfb3` because that one was 400 tokens and this one is 150, which
is the length `bench/decode/README.md` records as the one every arm agrees at.

---

## Step 2. The compiled decode blocks on the checkpoint

COMPILED.md section 6 carried a command and the line "nothing in this
workstream has run it". It has now been run, and it did not run on the first
try for two reasons that are worth more than the timing it eventually produced.

### 2a. Section 5's diff, applied

**5.1, the gates on the compiled gated residual.** Three came off: the MTP
gate, the width gate and the dtype gate. The MTP gate was guarding the wrong
thing -- it exists for the *hybrid projection* arm inside `_forward`, which
reads the same stamp and still refuses -- and the other two existed because the
closure was traced once. The closure now takes `target_verify` as a keyword, so
MLX keys a trace per value of it as well as per shape.

Applying it exposed that the parity test for this path had never tested
anything. `bench/decode/host_overhead.py`'s `build_synthetic` calls
`prepare_rmsnorm_scales` but not `compile_hyper_connections`, which the real
loader calls in `Model.load_weights`. So on the synthetic model
`_compiled_forward` was absent, the compiled arm was refused whatever the
forward path said, and
`test_compiled_gated_residual_holds_its_ulp_bound` ran `_forward` on both sides
of its pair. There is now a `compiled_hc` fixture that installs the closures
and removes them afterwards, the bound test uses it, and a new test counts the
calls so that a refused arm fails instead of passing quietly. The bound holds
at widths 1, 4 and 6 with the MTP stamp on.

**5.2, the compiled decode layer.** `Qwen4ExpDecoderLayer.compile_step` builds
the layer's step and holds it; `__call__` takes it when
`compiled_decode_layer` is on, the caller holds a `LayerState`, and there is no
`gdn_sink`. The path is in `forward_paths.DEFAULTS` at `False` and it is still
`False` at the end of this round, because the measurement below says it should
be.

### 2b. Two things that had to be fixed before the command could run at all

**The checkpoint has a PLE layer and `build_model` refused the whole model.**
`build_layer` raises for a layer whose n-gram embedding reads rows out of a
32 GB packed table through an mmap mid-forward, which is right for a caller
asking for that layer and wrong for a caller walking all 48. Layer 1 of the
checkpoint has one. `build_model` now builds a *plan* when it meets one: traced
runs of layers with the untraceable ones eager between them, one extra dispatch
per such layer rather than the whole saving. The state contract grew a third
kind, `EAGER`, a placeholder with no arrays that `grow_state`,
`truncate_state` and `recurrent_arrays` skip and whose layer keeps its own
vendored cache. A model with no such layer -- every synthetic model in the
tests -- takes exactly the single-graph path it always did.

**A single graph past the indexer budget is a wrong answer, and nothing
refused it.** COMPILED.md section 7 calls this the worst failure mode in the
document and says the builder should refuse rather than trust the caller.
`CompiledModel.__call__` now refuses, at call time because the budget is a
property of the length:

    ValueError: context 64001 is past the QSA indexer budget 2048, where the
    eager attention selects key blocks and a single traced layer does not.

So the 64k row of the table below is not a slow number, it is an absent one.
Getting it needs `build_layer_split` wired into `build_model`, which is a
separate change and, given what the 600-token row says, not an urgent one.

### 2c. The buffer check, which the compiled path passes

`host_overhead.py buffers --tokens 12288 --compiled`, synthetic, which FORWARD.md
section 5 requires of any dispatch change and COMPILED.md section 7 asks for by
name because `grow_state` briefly holds the old buffer and the new one at every
growth point.

| arm | steady-state growth | active MB at 12,288 | peak MB | wall for 12,288 steps |
|---|---:|---:|---:|---:|
| eager | 0.00 KB/step | 13.1 | 19.4 | 16.3 s |
| compiled | 0.00 KB/step | 6.8 | 18.2 | 12.6 s |

Both flat, against mlx-lm #1332's 205 KB/step before its fix. Active memory
steps only at the capacity growth points and is flat between them, and the
compiled path holds about half the live memory the eager one does, which is the
fixed-capacity KV doing what section 4 says it does. Nothing here is a reason
not to enable the compiled path.

### 2d. The measurement, and it loses

`compiled_path.py real --contexts 600 --widths 1,4 --repeats 20`, the
checkpoint through `titan.adapters.mlx.loader.load_model`, one `mx.synchronize`
on each side of each timed step, paired A B B A.

| | width 1 eager | width 1 compiled | width 4 eager | width 4 compiled |
|---|---:|---:|---:|---:|
| build ms | 14.03 | **2.64 (-81%)** | 23.05 | **2.88 (-87%)** |
| step ms | **16.29** | 27.91 (+71%) | **26.74** | 33.28 (+24%) |
| gpu ms | 2.26 | 25.27 | 3.69 | 30.40 |

**The host saving is real and it is the size COMPILED.md predicted.** 14.03 ms
of Python becomes 2.64, a factor of five, and it is flat in the width: 23.05
becomes 2.88, a factor of eight. FORWARD.md's 15.8 ms of host build per token
is the thing this workstream set out to remove and it is removed.

**The step is slower anyway, at both widths.** That is the whole result. Step
time is the honest end-to-end number and the compiled arm is 71% worse at width
1 and 24% worse at width 4.

Two reasons, and the second is the larger.

*The eager arm overlaps and the compiled arm does not.* Titan's forward
`async_eval`s the residual stream after every decoder layer, so the GPU is
running layer *i* while the host builds layer *i+1* and most of the GPU time
has already happened by the time the host finishes. The compiled arm submits
nothing until the caller asks. COMPILED.md says this about its own `gpu ms`
column and treats it as a reporting artefact; on the checkpoint it is not an
artefact, it is the mechanism. The host time the compiled path removes was not
on the critical path to begin with, because it was already hidden behind the
GPU.

*The compiled layer gives up `hc_fused`.* COMPILED.md's own qualification in
section 1c says it: as a standalone replacement the compiled gated residual is
not the fastest thing available, because `hc_fused` takes rows one to sixteen
on checkpoint-shaped inputs with three fused Metal kernels before `__call__`
ever looks at the compiled arm. `build_layer` composes `build_gated_residual`,
so a compiled layer runs the slower arm twice, and a step runs 96 of them. The
document expected the Python between blocks to be worth more than the kernels
inside them. On the synthetic model, whose GPU work is trivial, it was. On the
checkpoint it is not. This is the leading explanation for a 2.26 ms to 25.27 ms
move that capacity waste cannot account for -- `capacity_for(600)` is 768, so
attending over capacity is 1.28x on twelve layers and nowhere near the rest of
it -- but it is an explanation and not a measurement, and the experiment that
would settle it is a `build_layer` variant that calls `hc_fused` where the
shapes allow.

**So the compiled blocks are not wired into the served decode and
`compiled_decode_layer` stays off.** The round's condition for wiring them was
that they win, and they lose by 71% at the width that matters most. The server
A/B was not run, because there is nothing to A/B.

---

## Step 3. Housekeeping

**`VENDORED.md` said `qsa.gathered_batched` was unwired.** It has been wired
since ROUND3, at `Qwen4ExpAttention._batched_sparse_qsa`, for batched decode
and batched verify, with `BatchQSAKVCache.pooled_index_slots` building the
pooled bank in phased slot space. The table row and the paragraph under it now
say so, and say what the row still lacks, which is a throughput number on the
checkpoint: reaching it needs two sequences decoding in lockstep and
`MLXModelBackend.verify` dispatches one forward per sequence. The other unwired
rows gained the reason each one is unwired rather than being left as a single
sentence covering four different situations.

**The four QSA routes moved to `models/forward_paths.py`.**
`qsa_sparse_singleton_verify`, `qsa_batched_sparse`, `qsa_gather_min_context`
and `qsa_gather_min_context_verify` were in `titan/adapters/mlx/kernels.py`
because ROUND3 did not own the vendored file, with a comment saying they
belonged in `forward_paths`. They are there now, so the attention switchboard
is one file. `forward_paths` holds two kinds of switch: a *path* is a boolean
read with `enabled`, a *route* carries a value read with `route`, and
`set_paths` and `overridden` take either, so a bench pairing arms does not have
to know which kind each one is. `FORWARD.md`'s section on where the routes live
was rewritten to match.

**The adapter stopped reading the environment.**
`titan/adapters/mlx/kernels.py` read `TITAN_DISABLE_OPS` and
`TITAN_REFERENCE_ONLY` and applied them on top of the registry's policy, which
`titan/kernels/README.md` already claimed was gone. A bisect with two policies
cannot say which one produced a number, and this round is a bisect. The one
caller that set the variable, `bench/parity/greedy_parity.py --kernels
reference`, now publishes a reference-only registry instead, which is the same
mechanism `kernels.reference_only` uses, so the arm that bench calls the
control is the arm the server would run.

---

## Step 4. The defaults this round sets

Per-op, in code, next to the op: `KernelOp.default_off` with a
`default_off_reason` carrying the measurement. An op that is `default_off`
takes its fast path only when a configuration names it in `kernels.enabled`;
`kernels.disabled` still forces one off. So an empty `kernels.enabled` now
means "every op's default" rather than "every op's fast path", which is what
the production configuration needs, since it names nothing.

| op | default | why |
|---|---|---|
| `gdn_norm_gate` | **on** | +2.1% short, +2.5% at 64k |
| `moe_gather_ws` | **on** | +2.1% short, +1.8% at 64k |
| `moe_weighted_sum` | **on** | +2.0% short, +1.7% at 64k |
| `verify_accept` | **on** | +1.5% short, +1.9% at 64k |
| `topk_radix` | **on** | +1.5% short, +1.2% at 64k |
| `hc_prefill` | **on** | +1.4% short, +2.1% at 64k |
| `qsa_gathered_attention` | **on** | -0.6% short, +1.5% at 64k; neutral now, and the batched arm behind it is the one that pays when W4.2 lands |
| `moe_gather_int8` | **off** | -3.7% at 64k, below-control prefill, and 11.3 GB of qsum tables |
| `gdn_chunk_scan` | **off** | -13.2% at 64k for +2.4% prefill |
| `grouped_rmsnorm_bf16` | **off** | -7.9% at 64k for +2.7% prefill |
| `ple_packed_lookup` | **off** | -4.0% short and -3.5% at 64k, worst prefill of the thirteen arms |

`~/.config/titan/titan.toml` names the seven positively in `kernels.enabled`
rather than leaving the list empty. The two spellings select the same set; the
list is written out because a measured configuration should be readable in the
file that produced it, and because `/metrics` echoes this file. The comment
above it carries the numbers and the one-line change for a prefill-dominated
workload.

Nothing about the forward paths changed except the addition of
`compiled_decode_layer`, which is off.

---

## Step 5. Where this leaves the numbers, and the next three levers

Profiler tok/s, drafted, production configuration.

| | before this round | after | oMLX on this machine |
|---|---:|---:|---:|
| short context | 83.0 | **84.6** | 84 to 91 |
| 64k | 61.2 | **64.9** | 71 |
| cold 65k prefill | 872 | 826 | -- |

At pinned drafter depth, where the comparison is the kernels rather than the
control loop, the same change is 84.1 to 88.8 short and 58.6 to 65.8 at 64k.
Short context is now inside oMLX's band. 64k is 8.6% behind it, down from 14%.

The prefill column went the other way, by 5.3%, and section 1c is the trade
that bought the decode numbers. It is reversible in one line.

### The next three levers, ranked

**1. The adaptive depth policy, which is worth more than every kernel in this
round put together.** Section 1a measured the same configuration at 93.5 and
79.0 tok/s in two runs an hour apart, entirely because the policy settled on
`mean_rows` 3.60 once and 2.97 the other time. That is an 18% spread on the
control arm. Pinning the depth at 3 costs nothing and removes it: `fd_refonly`
and `fd_all_on` reported `accepted_per_cycle` of 2.310 to three decimals on
every arm and every repeat. The policy optimises
`E[committed | k] / cycle_ms(k + 1)` against a cycle-time model it feeds
itself, and a loop whose input is its own output has more than one resting
point. The cheap half is a floor under the depth; the real fix is to stop the
estimator learning from cycles the policy itself shortened. Nothing else here
is an 18% lever.

**2. Find out why a prefill kernel costs a decode, because it is worth both
columns.** `gdn_chunk_scan` fires only on prefill-shaped calls and costs 13.2%
of 64k decode; `grouped_rmsnorm_bf16` costs 7.9% the same way; neither costs
anything at short context, where the prefill never reaches the shape they fire
on. If the mechanism is residency left behind by the prefill -- and
`moe_gather_int8` is already known to hold 11.3 GB of tables -- then freeing it
after prefill gives the 6.2% of prefill *and* the 12.2% of decode, instead of
making anyone choose between them. The experiment is one arm with
`mx.get_active_memory()` sampled either side of the prefill, per op. It is
cheap and it has not been done.

**3. The two QSA indexer seams, which is ROUND3's lever and is still open.** At
64k the pooled block bank is 16,000 slots of 128 bf16 values a layer, and
`_portable_indexer_scores` casts it to fp32 every step and reads it twice, then
`argpartition` reads the scores. oMLX's `qsa.indexer_scores` and
`qsa.topk_indices` consume the bf16 bank directly and Titan has no equivalent;
they are two of the four unwired rows in `VENDORED.md`. ROUND3 priced the whole
residue at 1.6 ms over twelve layers, so this is smaller than the two above it,
but it is the one that is squarely on the 64k gap to oMLX and the cheap half of
it -- a persistent fp32 bank on the cache, no Metal kernel -- removes the cast
without a new kernel.

Ranked below all three, and recorded so it is not rediscovered: the compiled
decode blocks. They do what they claimed to the host, five to eightfold, and
they lose anyway because the host time was already hidden behind the GPU and
because a compiled layer gives up `hc_fused`. A `build_layer` that keeps
`hc_fused` where the shapes allow is the experiment that would reopen it.
