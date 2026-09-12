# The forward pass, and what it costs the host

One decode cycle on the real checkpoint at a 600-token context measures 18.1 ms
with the device synchronised. Host graph build in Python is 15.8 ms of that and
GPU execution is 2.0 ms, so the GPU is idle 87% of the time. Under target
verification the host cost rises with the block width: 14.3, 16.8, 24.0 and
32.0 ms at widths 1, 2, 4 and 6.

This document says where that host time goes, which parts of the forward were
built once per token and are now built once per block, what was compiled and
what could not be, and how to reproduce every number.

Everything here is measured on the synthetic model in
`bench/decode/host_overhead.py`, not on the checkpoint. The last section says
what that costs in fidelity and gives the single command for the real
measurement.

---

## 1. How the numbers are taken

MLX is lazy. "How many ops did this step build" is a question about the graph,
not the clock, so `Trace` in `bench/decode/host_overhead.py` wraps `mx.eval` and
`mx.async_eval`, walks the graph reachable from each call's arguments with
`mx.export_to_dot` before letting the call through, and counts the primitive
nodes. A node an earlier call already evaluated is a leaf by the time a later
call sees it, so summing over a step's eval boundaries counts each primitive
once. The closing `finish` sweeps up whatever the step left unevaluated.

The count is primitives the host built, which is what dispatch cost tracks. It
is not the number of Metal dispatches: MLX fuses some primitives, and a
`mx.fast.metal_kernel` is one node whatever it does inside.

Three columns come out of a step:

- `build ms`, the Python time to construct the graph with the step's evaluation
  pushed to the end,
- `step ms`, the same plus the wait for the GPU,
- `gpu ms`, the difference, which is a lower bound on GPU time and not a
  measurement of it. Work the GPU finished while the host was still building
  does not appear.

Timings on this machine are noisy because the workbench shares the GPU. Op
counts are exact and reproducible; step times are only trustworthy when the two
settings are paired inside one process, which is what the `paths` subcommand
does, in A B B A order so a machine warming up over the run cannot favour
either side.

### The synthetic model

`SyntheticSpec` builds a Qwen4-Exp with four decoder layers, hidden 128, eight
experts top-2, the 4-wide hyper-connection residual, the QSA indexer at the
checkpoint's 2048-token budget, and 4-bit affine quantisation applied by the
same rule the checkpoint follows (a projection is quantised when its input is a
whole number of 64-element groups). It weighs a few megabytes. `--layers 24`
makes per-layer effects visible above the timing noise without making the model
large.

What it does not cover:

- **PLE.** The n-gram embedding layers read a 32 GB packed table from the
  checkpoint on SSD (`Qwen4ExpNGramEmbedding.__init__`,
  `qwen4_exp/language.py:2501`, which raises without a model path). The
  synthetic config sets `ple_layer_ids=[]`. On the checkpoint one layer of 48
  has PLE, and its two projections are bf16, so it is one of the places the
  batched linear arm matters.
- **The Lightning MTP head.** Built only when the checkpoint carries MTP
  weights. `mtp_draft` and the MTP path are another workstream's.
- **The 512-expert router width and the real hidden sizes.** Which arm each
  projection takes depends on its shape, and section 3 works the real shapes
  out by hand where the synthetic model cannot reach them.

---

## 2. Where a step's host time goes

Synthetic model, 24 layers, context 600, batch one, quantised. Decode is width
1 and verify is width 4.

| width | ops | eval | async_eval | build ms | step ms | gpu ms |
|---|---|---|---|---|---|---|
| 1 | 3042 | 0 | 24 | 6.6 | 7.0 | 0.4 |
| 4 | 3211 | 0 | 24 | 7.1 | 7.5 | 0.4 |

Two things to read off this.

**There are no `mx.eval` calls in a step and one `async_eval` per layer.** The
per-layer `async_eval` is `qwen4_exp/language.py:2841`, fired whenever the row
count is at most 64, which is every decode and every verify. Forty-eight of
them per forward on the checkpoint.

**The graph is about 127 primitives per layer at width 1.** Scaled to 48 layers
that is roughly 6100 primitives for a decode step, and 15.8 ms of host build
over 6100 primitives is 2.6 microseconds each. That is the right order for
Python-level MLX op construction, so the reported host cost is explained by op
count rather than by anything unusual.

Python time by module, cProfile `tottime`, decode width 1 at 24 layers. Read
the shares, not the milliseconds: the profiler inflates every call.

| share | module |
|---|---|
| 69% | `vendor/mlx_vlm/models/qwen4_exp/language.py` |
| 11% | `vendor/mlx_vlm/models/qwen3_5/language.py` |
| 3.2% | builtins and the MLX C extension |
| 2.7% | `mlx/nn/layers/quantized.py` |
| 2.6% | `mlx/nn/layers/base.py` |
| 2.3% | `vendor/mlx_lm/models/switch_layers.py` |
| 1.9% | `vendor/mlx_vlm/models/qwen3_5_moe/language.py` |
| 1.6% | `titan/kernels/moe_weighted_sum.py` |
| 1.5% | `vendor/mlx_vlm/models/qwen3_5/gated_delta.py` |
| 1.2% | `titan/kernels/registry.py` |

The 69% in `qwen4_exp/language.py` is not one hot function. It is the decoder
layer body, the hyper-connection block, the QSA indexer, and the per-layer
`async_eval`, whose submission cost cProfile bills to its caller. Section 5
separates that last part out, because it is the largest single item.

The primitive histogram says the same thing from the other side. Decode width
1, 24 layers, top ten:

| count | primitive |
|---|---|
| 445 | Reshape |
| 395 | Broadcast |
| 333 | QuantizedMatmul |
| 285 | Multiply |
| 163 | Sigmoid |
| 132 | ExpandDims |
| 121 | Divide |
| 115 | RMSNorm |
| 87 | Add |
| 84 | Slice |

Reshape, Broadcast, ExpandDims and Slice together are 1056 of 3042, so a third
of the graph is shape metadata rather than arithmetic. Most of those are free on
the GPU and none of them are free on the host.

---

## 3. The per-token work under `target_verify`

`target_verify` is the vendored model's name for a family of arms that keep a
verify block's rows independent of each other: each output row reads the whole
weight matrix, as it would if the row were decoded alone. That is the right
trade for a block a few rows wide. Written as a Python loop it is also one
kernel launch per row per token.

`target_verify` is not passed in. It is derived, in three places, from whether
the caller asked for a hidden state:

- `qwen3_5/language.py:2634`, `gdn_sink` and `target_verify` from
  `capture_layer_ids`,
- `qwen4_exp/language.py:2834`, `target_verify=gdn_sink is not None` per layer,
- `qwen3_5/language.py:1672`, `target_verify = target_verify or gdn_sink is not
  None` inside the Gated DeltaNet.

`LanguageModel.__call__` (`qwen4_exp/language.py:3003`) sets
`capture_layer_ids=[]` whenever `return_hidden` is true, so asking for a hidden
state is what turns all of this on. That is why a prefill chunk that wanted the
hidden state took the verify arms, and why `model.py`'s `prefill` docstring
recorded a 150-token chunk at 200 seconds against 0.1 without.

### The loops, as found

| site | shape it loops over | fires when |
|---|---|---|
| `qwen3_5/language.py:629` `_target_verify_timewise` | one call per token | quantised projection, batch above one, input not a whole number of 512-element groups |
| `qwen3_5/language.py:634` `_target_verify_singletons` | one call per row per token | bf16 projection the dense GEMV declines |
| `qwen3_5/language.py:1575` per-row SDPA in `Qwen3_5Attention` | one attention call per query row per sparse layer | every verify block wider than one |
| `qwen3_5/language.py:1383` per-row SDPA in `_target_verify_left_padded_attention` | one call per query row per padding group | batched rows with ragged left padding |

The dense GEMV at `qwen3_5/language.py:187` is a single launch over all rows, so
a projection it accepts costs one dispatch whatever the width. It declines three
shapes: an output narrower than four columns, an output that is not a multiple
of four, and an input more than sixteen times the output. Before this change,
declining meant `_target_verify_singletons`.

Two things that look like per-token loops and are not. The Gated DeltaNet's
`for t in range(T)` at `qwen3_5/gated_delta.py:284` and `:515` is the reference
implementation, reached only without Metal; on the GPU the recurrent step and
its intermediate capture are one fused `mx.fast.metal_kernel`
(`gated_delta_update_with_states`, `gated_delta.py:304`). And the query-chunk
loop at `qwen4_exp/qsa_fast.py:598` has a minimum chunk of 32
(`contiguous_causal_query_chunk`, `:75`), so every verify width the engine uses
is a single chunk.

### Which loops the checkpoint actually reaches

`_target_verify_linear` short-circuits a quantised projection at batch one
(`qwen3_5/language.py:652`), and on the checkpoint almost every projection is
quantised. Walking the quantisation config against the module list:

- **Quantised, so the short circuit takes them at batch one.** The attention
  Q/K/V/O, the QSA indexer projection, all four Gated DeltaNet input
  projections and its output projection, all three hyper-connection
  projections, the shared expert and its gate, the embedding and the head.
- **bf16, so the dense GEMV decides.** The MoE router at 2560 into 512 (input
  is five times the output, accepted, one launch), and the PLE key and value
  projections at 2560 into 10240 and 2560 (accepted).

So at batch one on the checkpoint, no linear was looping per token. The
width-linear host cost at batch one was the per-row attention loop: twelve
sparse layers times the block width, so 48 attention calls at width 4 where 12
would do. At batch two and above the picture changes, because the custom verify
QMV needs the input to be a whole number of 512-element groups and two shapes
per layer are not: the hyper-connection up projection at 320 and the shared
expert down projection at 640. Those were one call per token per row.

On the synthetic model the same walk is visible directly. With bf16 weights, 22
of 89 projections per forward fell to `_target_verify_singletons`; with the
checkpoint's quantisation at batch one, none did.

### What changed

`_narrow_verify_block` (`qwen3_5/language.py:169`) with
`_TARGET_VERIFY_MAX_ROWS = 64` now gates every `target_verify` arm on the
block's width, and every per-row fallback has a block-wide arm in front of it
behind a named switch:

- `batched_verify_linear`: a projection the dense GEMV declines takes the
  ordinary batched call instead of the per-row loop
  (`qwen3_5/language.py:657`, `:674`, `:703`).
- `batched_verify_attention`: one masked attention call over the block instead
  of one per query row (`qwen3_5/language.py:1530`).
- The recurrent capture is created only for a narrow block
  (`qwen3_5/language.py:2634`), so a prefill chunk that asks for its hidden
  state gets chunk-shaped work and does not allocate one intermediate state per
  recurrent layer per row.

The last one closes the trap `model.py` documented. A 150-token chunk with
`return_hidden=True` now makes one attention call per sparse layer, no per-row
linear calls at all, and returns its hidden state.
`tests/model/test_forward_overhead.py` asserts each of those.

### What the batched attention costs

The per-row loop computes row i against the first `prefix_len + i + 1` keys,
which is causality expressed by slicing. One call with the block's own mask is
the same reduction over the same keys, because a masked-off column contributes
an exact zero. It is not free in the last bits. Measured against a float32
reference at the checkpoint's head shapes (24 query heads, 2 KV heads, head
dimension 256, 604 keys, width 4), the block-wide call sits at 4.3e-3 relative
RMS error and the per-row loop at 1.7e-3, against a bf16 epsilon of 3.9e-3. The
two arms differ from each other by 4.6e-3 relative RMS, so about one bf16 ULP,
which is the tolerance class Titan already accepts for the batched QSA arm.

There is a subtlety worth stating rather than fixing. The per-row loop drops a
mask of rank below four and relies on slicing for causality. The block-wide arm
reproduces that exactly, passing the string causal mask in the same case.
Changing it would be a numerics change and does not belong in a dispatch fix.

---

## 4. What was compiled, and what could not be

`mx.compile` is worth reaching for where a block is elementwise, shape-stable
and free of side effects. Three parts of the forward qualify.

**The hyper-connection injection** (`qwen4_exp/language.py:2686`,
`_hyper_inject`). Writing a branch's output back into the 4-wide residual stream
was five nodes inline, run twice per layer, so 480 nodes of a 48-layer decode
step. Compiled with `shapeless=True`, one traced graph serves decode width 1 and
every verify width. Measured saving: 4 nodes per layer.

**The gated residual** (`Qwen4ExpGatedResidual._forward`). A compiled decode
path already existed (`compile_hyper_connections`, `qwen4_exp/language.py:1883`)
and is now under the `compiled_gated_residual` switch. Its existing gate keeps
it off while Lightning MTP is enabled, which is production, so this one does not
yet pay on the checkpoint. Lifting that gate is a small change and an untested
one; the switch is in place for whoever measures it.

**The RMSNorm scale** (`Qwen4ExpRMSNorm.prepare_scale`,
`qwen4_exp/language.py:1023`). `1 + weight` was a cast, an add and a grouped
reshape over a constant, on every norm on every step. It is now folded once by
`prepare_rmsnorm_scales` (`:1873`) from `Model.load_weights`. Bit-identical, the
same fp32 value computed once, and 7 nodes per layer.

What could not be compiled, and why:

- **The whole single-token forward.** MLX's documented breakers are evaluating
  arrays inside the function, side effects, and control flow that branches on
  array values. The forward has all three. `_causal_conv1d_decode`
  (`qwen3_5/language.py:1651`) calls `mx.eval` on a cached weight.
  `KVCache.update_and_fetch` reallocates its buffer and mutates the cache
  object. The QSA path reads `int(cache.offset)` and calls
  `kv_seq_len.max().item()` (`qwen3_5/language.py:1507`). The PLE layer reads
  rows out of an mmap through numpy. None of these is a small fix.
- **The Gated DeltaNet step.** There is nothing to compile. On Metal the step
  and its intermediate capture are already one fused kernel; the Python loop is
  the CPU fallback.
- **The MoE routing.** `mx.argpartition` followed by gathers is compilable in
  principle, but the routed expert call goes through `SwitchGLU`'s gather
  matmuls and the registry dispatch in `titan/kernels`, which resolve per call
  in Python. Compiling around that seam is a larger change than this one.
- **A compiled verify lane.** The survey's D5 wants the whole verify under
  `mx.compile` with the width bucketed. That needs the cache passed as compiled
  inputs and outputs and a fixed-shape commit, and it is blocked by the same
  three breakers as the full forward.

---

## 5. The per-layer `async_eval`, which is the largest single item

Turning off `eager_dispatch` on the synthetic model at 24 layers takes build
time from 6.6 ms to 1.9 ms. Twenty-four `async_eval` calls therefore cost 4.7 ms
of host time, about 0.20 ms each. Scaled to 48 layers that is 9.4 ms, which
would be most of the checkpoint's 15.8 ms host build.

That is not the same as saying 9.4 ms is recoverable. On the same run, step time
went the other way: 7.0 ms with eager dispatch on and 8.1 ms with it off, so
removing the submissions cost 14% more end to end. Evaluating one 3000-node
graph is slower here than evaluating 24 pieces of it as they are built. Eager
dispatch does not create host work so much as relocate it, and on a model whose
GPU work is trivial, relocating it loses.

The checkpoint's arithmetic is different: 2.0 ms of GPU work against 15.8 ms of
host build, so there is far less to overlap and far more submission cost to
save. That is exactly why experiment 1 has to run there, and why the survey's
`-5% to +20%` range is honest. The `dispatch` subcommand runs it.

Two notes for whoever does. The flag was `TITAN_QWEN4_EAGER_DISPATCH`, read once
at import; it is now the `eager_dispatch` forward path, so both settings can be
paired inside one process and the resolved configuration can record which one
ran. And the check in experiment 1 is mandatory rather than advisory: mlx-lm
issue #1332 records a model dying at 11,300 tokens on a limit that counts
buffers rather than bytes, and Titan holds 36 recurrent states. `host_overhead.py
buffers --tokens 12288` reports active, peak and cache memory and the per-step
growth. MLX 0.32.2 does not expose the buffer count itself, so growth in active
memory is the proxy, and a flat line after the cache settles is the pass.

---

## 6. Before and after

Synthetic model, context 600, batch one, quantised, every added path off against
on, paired A B B A inside one process.

At 24 layers:

| | width 1 before | width 1 after | width 4 before | width 4 after |
|---|---|---|---|---|
| ops | 3309 | 3042 (-8.1%) | 3700 | 3211 (-13.2%) |
| build ms | 8.03 | 7.55 (-6.0%) | 8.67 | 7.97 (-8.0%) |
| step ms | 8.49 | 8.02 (-5.5%) | 9.11 | 8.44 (-7.3%) |

At the default four layers, where the timings are inside the noise of a shared
GPU and only the op counts mean anything:

| | width 1 before | width 1 after | width 4 before | width 4 after |
|---|---|---|---|---|
| ops | 584 | 537 (-8.0%) | 650 | 566 (-12.9%) |
| build ms | 1.21 | 1.18 | 1.35 | 1.24 |

Attribution, in nodes per step at 24 layers:

| path | width 1 | width 4 |
|---|---|---|
| `cached_norm_scale` | -171 | -171 |
| `compiled_gated_residual` | -96 | -96 |
| `batched_verify_attention` | 0 | -222 |
| `batched_verify_linear` | 0 | 0 at batch one on a quantised model |

Per layer that is 11.1 nodes at width 1 and 20.4 at width 4, so on 48 layers
roughly 530 and 980 nodes. At the 2.6 microseconds per node the checkpoint's
own numbers imply, 1.4 ms and 2.5 ms of host build.

`batched_verify_linear` shows zero here because the synthetic model at batch one
is exactly the case where the quantised short circuit already applied. Its value
is at batch two and above, and on the checkpoint's bf16 projections, which
section 3 works out.

---

## 7. The forward paths

`titan/adapters/mlx/vendor/mlx_vlm/models/forward_paths.py` is the switchboard.
No environment variables: a path is a name with a default and a docstring line,
flipping one is a function call, and `overridden` scopes a change to a block.

| name | default | what it does |
|---|---|---|
| `eager_dispatch` | on | `async_eval` the residual stream after every layer at small row counts |
| `batched_verify_linear` | on | one launch over the verify width instead of one per token |
| `batched_verify_attention` | on | one attention call over the verify width instead of one per row |
| `compiled_gated_residual` | on | the compiled gated residual and the compiled injection |
| `cached_norm_scale` | on | the RMSNorm scale folded at load time |

Every path off is a supported configuration, and off is always the slower arm
rather than the wrong one. If a greedy-parity probe regresses,
`batched_verify_attention` is the first switch to flip: it is the only added
path that moves the last bits on the checkpoint's shapes.

---

## 8. Reproducing this

Synthetic, which is what every number above came from:

```
python bench/decode/host_overhead.py profile --context 600 --widths 1,4 --layers 24
python bench/decode/host_overhead.py paths   --context 600 --widths 1,4 --layers 24
python bench/decode/host_overhead.py dispatch --context 600 --widths 1,4 --layers 12
python bench/decode/host_overhead.py buffers --tokens 12288
```

The real model, one command, which nothing in this workstream has run:

```
python bench/decode/host_overhead.py profile \
    --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
    --context 600 --widths 1,4 --repeats 20
```

It loads the checkpoint through `titan.adapters.mlx.loader.load_model`, prefills
to the requested context, and prints the table in section 2. The other three
subcommands take the same `--model`, one at a time, and `dispatch` is
experiments 1 and 2 from `docs/research/OPTIMISATION-SOURCES.md` section 5:
eager dispatch in {on, off} crossed with `MLX_MAX_OPS_PER_BUFFER` in {default,
200, 500}, each cell in its own process because MLX reads its command-buffer
budget once at start.

Run the sweeps on a quiet GPU, paired, per the protocol in IMPROVEMENTS section
4, and run `buffers --tokens 12288` on the winning configuration before acting
on any of it.

---

## 9. The attention arms, and which forward takes which

Appended by ROUND3. Sections 1 to 8 are about the host cost of building the
graph, which is context-independent. This section is about the one part of the
forward whose cost is not: the twelve full-attention layers, which read a cache
that grows.

### The three forwards are not the same forward

The engine runs three one-model forwards and they take three different
attention paths. Which one a call gets is decided by two flags it does not set
on purpose.

| forward | rows | `return_hidden` | `target_verify` | arm |
|---|---|---|---|---|
| plain decode | 1 | no | no | gathered sparse |
| decode with a hidden state | 1 | yes | yes | gathered sparse *(was dense)* |
| verify | 2 to 8 | yes | yes | gathered sparse |

`target_verify` is derived, not passed: `LanguageModel.__call__` sets
`capture_layer_ids=[]` whenever `return_hidden` is true, and section 3 lists the
three places that turns into `target_verify` on a layer. It exists so the Gated
DeltaNet captures the intermediates a rollback replays from. Attention has no
intermediates to capture, so the flag should not reach it -- but
`_gathered_text_decode_eligible` tested `not target_verify`, so the middle row
read the whole cache on all twelve QSA layers. That is 1.4 ms a forward at 64k
on the checkpoint's geometry, measured by elimination in `bench/decode/
ROUND3.md` step 2b. It is now routed to the same arm as the other two, behind
the `qsa_sparse_singleton_verify` route.

### The gathered arm is legal before it is cheaper

The vendored gate is the QSA token budget, 2048: past that there are more
complete blocks than the indexer may select, so selection has something to
remove. Cheaper is a different question. The gathered arm reads a flat
`token_budget + compress_ratio - 1` rows -- 2,051 -- whatever the context, so at
2048 it is reading more than the dense arm and paying for selection on top.

Measured whole-step on the real-shapes synthetic, paired, in milliseconds for
two QSA layers (multiply by six for the checkpoint):

| context | width 1 | width 4 |
|---:|---|---|
| 2052 | dense ahead by 0.08 | dense ahead by 0.20 |
| 4096 | dense ahead by 0.08 | level |
| 6144 | dense ahead by 0.04 | gathered ahead by 0.21 |
| 8192 | level | gathered ahead by 0.58 |
| 16384 | level | gathered ahead by 1.63 |
| 64000 | gathered ahead by 0.21 | gathered ahead by 8.02 |

Two crossovers, because the dense arm's cost rises with the block width and the
gathered arm's barely does. `_gather_min_context(token_budget, width)` carries
both, from the `qsa_gather_min_context` and `qsa_gather_min_context_verify`
routes, and clamps up to the budget so a route cannot make the arm legal
earlier than the algorithm allows.

The bottom-right cell is the one to remember before touching any of this: the
gathered verify arm is worth about 48 ms a forward on the checkpoint at 64k. It
is the largest single item in the long-context forward and it was already on.

### Where the routes live, and why not in `forward_paths`

`titan/adapters/mlx/kernels.py` carries them, next to the registry lookups, on
the same contract as this document's section 7: a name, a default, a docstring
line, no environment variables, and `overridden` to scope a change to a block.
They belong in `models/forward_paths.py` with the rest and should be moved
there; ROUND3 did not own that file.

### What is still context-dependent, after all of it

A plain decode at 64k costs 1.6 ms more over twelve QSA layers than the same
decode at 2052, with every arm chosen correctly. That is the sparse arm's own
growth: the pooled block bank the indexer scores against holds 16,000 slots at
64k against 512 at 2052, and the two parts that read all of it are the fp32
cast in `_portable_indexer_scores` and the `mx.argpartition` over the scores.
Neither is removable by routing. oMLX does not pay them because its
`qsa.indexer_scores` and `qsa.topk_indices` kernels consume the bf16 bank
directly; those are two of the four narrow seams
`titan/adapters/mlx/VENDORED.md` lists as unwired, and they are the next lever.
