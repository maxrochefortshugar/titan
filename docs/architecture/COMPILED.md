# The compiled decode path

`docs/architecture/FORWARD.md` measured a decode token on the checkpoint at
18.1 ms, of which 15.8 ms is Python building a graph the GPU runs in 2.0 ms.
The graph is about 127 primitives per layer, 6100 for a 48-layer step, at 2.6
microseconds each. That arithmetic does not improve by making primitives
faster. The only lever is building fewer of them from Python, and `mx.compile`
is that lever: a traced function replays its graph from the C++ side, so the
second call and every later one costs a dispatch of a cached graph instead of a
few hundred Python-level op constructions.

FORWARD.md section 4 listed four reasons the forward could not be compiled.
Every one of them is a property of how the forward is *written*, not of what it
computes, and `titan/adapters/mlx/compiled.py` is the rewrite.

| FORWARD.md said | what it was | what it is here |
|---|---|---|
| `mx.eval` inside the function | `_causal_conv1d_decode` evaluated a transposed conv weight on first use | `conv_decode_weight` hoists the transpose to build time; the step evaluates nothing |
| state mutated in place | `KVCache.update_and_fetch` reallocates and assigns into the cache object | a fixed-capacity buffer in, `mx.slice_update` at an array offset, a buffer out; growth outside the traced region |
| control flow on array values | the QSA path read `int(cache.offset)` and `kv_seq_len.max().item()` | the offset stays an `int32` array and the causal mask is derived from it in the graph |
| registry dispatch per call | `titan.kernels` resolves per site per step | every op is resolved once at build time and baked into the trace |

---

## 1. What compiled

Every block is a factory. It takes the module, reads out everything that is not
an array -- dimensions, epsilons, quantisation bits, whether a projection is
fused -- and returns a compiled function plus a pytree of arrays. The function
is pure, so the caller owns the state, which is what makes section 4 work at
all.

### (a) The Gated DeltaNet decode step

`build_gdn_step(module) -> (x, conv_state, ssm_state, weights) -> (y,
conv_state', ssm_state')`.

One graph for the whole linear-attention branch: the four input projections,
the causal conv window and its next state, the gated delta recurrence, the
gated output norm and the output projection. No `mx.eval`, no in-place write,
no `.item()`.

`causal_conv1d_decode` is the conv written functionally: the window is a
concatenate of the state and the new rows, the taps are a Python loop over a
compile-time constant, and the next state is a slice of the window. It
accumulates in fp32, which is what the vendored *decode* arm does. The vendored
*verify* arm calls `nn.Conv1d` and accumulates in bf16, so at width above one
the two arms differ; at kernel size four the difference is well under a bf16
ULP and the fp32 one is the more accurate of the pair.

Measured against the eager module at widths 1, 2, 4 and 6: output within one
bf16 ULP, the conv state bit-identical, the recurrent state within one ULP.

### (b) The MoE block

`build_moe_step(module) -> (x, weights) -> y`, widths 1 to 8.

Routing is in the graph: the softmax, `mx.argpartition` for the top-k, the
renormalised scores, the gathered expert matmuls on the fused `gate_up` and on
`down`, the weighted sum, and the shared expert with its own sigmoid gate. The
`moe.weighted_sum` kernel is resolved once at build time rather than looked up
per call.

There is no sort in it, and that is not a simplification. `SwitchGLU`'s sort
fires at `indices.size >= 64`; a decode is `top_k` indices and a width-8 verify
is `8 * top_k`, so at every width this block is built for, the stock path is
the unsorted one and the two Titan gather ops -- which require
`sorted_indices` -- decline. There never was a registry call to remove at these
widths; what was removed is the Python around `SwitchGLU`.

Bit-identical to the eager block at every width 1 to 8.

### (c) The hyper-connection blocks

`build_gated_residual(module)` for both per-layer residuals and for the final
`hyper_connection_mixer`, and `hyper_inject` for the write back into the
four-wide stream. Both are `shapeless=True`: one trace serves every width.

Bit-identical to `Qwen4ExpGatedResidual._forward` at widths 1, 4 and 8, and to
`_hyper_inject_ops`.

This is the block FORWARD.md said "does not yet pay on the checkpoint", because
the vendored compiled arm is gated off whenever Lightning MTP is enabled, which
is production. Here it carries no MTP gate, no width gate and no dtype gate, so
it is the compiled default including when MTP is on. Section 5 is the diff that
makes that true of the vendored module too.

One honest qualification. As a *standalone* replacement in the eager path the
compiled residual is not the fastest thing available: `hc_fused` takes rows one
to sixteen on checkpoint-shaped inputs with three fused Metal kernels, before
`__call__` ever looks at the compiled arm. Where the compiled residual earns
its place is inside `build_layer`, where what it removes is not the ops but the
Python between them.

Writing a shapeless block has one rule that is easy to get wrong and was got
wrong first time here: a shapeless trace still hands the function the *first*
call's concrete shapes, so a reshape spelled with `x.shape` bakes that width
into the graph. The second width then raises `ValueError: [reshape] Cannot
reshape array of size 2048 into shape (1,1,512)`. `mx.unflatten` and
`mx.flatten` say the same thing relative to the trailing axis and survive the
width change. The spelling with a `-1` in it is worse than the one that raises:
it reshapes to a valid but wrong shape and says nothing.
`tests/model/test_compiled_path.py` pins all three behaviours.

### (d) The dense attention step

`build_attention_step(module) -> (x, k_buf, v_buf, offset, weights,
sparse_mask) -> (y, k_buf', v_buf', offset', index_keys)`.

The KV cache is two fixed-capacity arrays and a rank-zero `int32` offset array.
The step writes the new rows with `mx.slice_update` at that offset -- so nothing
calls `int()` on it and nothing reallocates -- and attends over the *whole*
capacity with a causal mask derived from the offset inside the graph. Columns
past the write are masked, and a masked column contributes an exact zero to the
softmax, so this is the same reduction the sliced eager arm performs.

Attending over capacity rather than over the live length is what buys the
stable shape: one trace covers every length that fits, and only a capacity
change retraces. The cost is arithmetic on masked columns, bounded under 2x by
`capacity_for`, against a GPU that is idle 87% of the time.

`capacity_for` steps by 256 up to 2048 and doubles after, so the number of
traces is logarithmic in the context and the waste is under 2x. `grow_kv` does
the reallocation, outside any traced region, because a shape change is a
retrace by definition.

The step also returns the QSA indexer's raw key row. The selection is not in
this graph (see below), but the raw keys are one projection and a reshape, and
leaving them out would let the indexer's cache drift out of alignment with the
KV it indexes, which section 4 forbids.

Measured against the eager attention at a 600-token context, widths 1 and 4:
output within one bf16 ULP, KV buffer bit-identical, index keys bit-identical.

### Whole layers and the whole model

`build_layer(layer)` composes the four blocks *uncompiled* and traces the
composition, so a layer is one graph rather than four graphs with Python
between them. That distinction is the whole point: the saving from compiling a
block is the Python it replaces, and the Python between blocks is not replaced
by compiling the blocks.

`build_model(model)` chains every layer, the final mixer and the head into one
traced step. The embedding is deliberately outside it: on the checkpoint it is
a `DiskBackedShardedEmbedding` reading rows out of an mmap through numpy, which
is a side effect no trace can hold. It is one gather per step.

---

## 2. What did not compile, and the exact reason

**The QSA selection above the indexer budget.** Below the budget the eager
attention is dense and the compiled step reproduces it exactly. Above it the
eager path selects key blocks, and the selection needs *this layer's* mixed
hyper-connection output -- which is computed inside the single layer graph. So
a sparse layer cannot be one graph, and `build_layer_split` is the cut:
`mix_step` gives the mixed input, the caller computes the selection on the
eager path, and `rest_step` takes it padded to capacity with `pad_sparse_mask`.
Measured at a 2400-token context on the synthetic model, the split with the
selection passed in lands within one bf16 ULP of eager while the dense single
graph is two orders of magnitude further away -- that gap is not rounding, it
is a different set of keys.

The selection itself is host work by construction rather than by formulation:
`Qwen4ExpQSAIndexer.from_projected` reads `int(cache.offset)` and pools the
indexer keys through `pooled_indexer_keys`, a Python-level cache with its own
invalidation. Compiling it needs the indexer cache reformulated the way the KV
cache is here -- a capacity-shaped key buffer and an array offset -- and that is
a separate change.

**PLE layers.** `build_layer` raises rather than silently dropping one.
`Qwen4ExpNGramEmbedding` reads rows from a 32 GB packed table on SSD through an
mmap and numpy, mid-forward. One layer of 48 on the checkpoint has one; that
layer stays eager.

**`shapeless=True` for anything but the hyper-connection blocks.** Three
primitives these blocks cannot avoid refuse to infer output shapes without
concrete inputs. All three raise `ValueError` from `[Primitive::output_shapes]`
on MLX 0.32.2, and `tests/model/test_compiled_path.py` asserts each message,
because a claim about an error string is a claim about a version:

| error | what raises it here |
|---|---|
| `CustomKernel cannot infer output shapes` | any `mx.fast.metal_kernel`: `gated_delta_kernel`, and `_mrope_apply_kernel` in the attention step |
| `Slice cannot infer output shapes` | `mx.argpartition` followed by a negative-index slice -- the MoE top-k -- and the conv window slice |
| `Split cannot infer output shapes` | `mx.split` at explicit indices -- the GDN q/k/v split, the attention query/gate split |

Two things that do infer shapes and are therefore *not* what forces the
specialisation, which is worth stating because both were suspected:
`mx.slice_update` at an array offset, and `mx.gather_qmm`.

So the shape-specialised trace is the default, MLX keys its compile cache on
input shapes, and widths 1 to 8 cost eight traces. Section 3 prices them.

**The whole eager `LanguageModel.__call__`.** Not attempted. It carries the
mRoPE delta bookkeeping, the mask construction, the batched left-padding paths
and the MTP capture, all of which branch on Python state. The compiled model
step replaces the decode-shaped subset of it.

---

## 3. Before and after

Synthetic Qwen4-Exp, context 600, batch one, 4-bit quantised, eager against
compiled, paired A B B A inside one process.
`python bench/decode/compiled_path.py synthetic --layers 24 --context 600
--widths 1,4`.

At 24 layers:

| | width 1 eager | width 1 compiled | width 4 eager | width 4 compiled |
|---|---|---|---|---|
| ops | 3042 | 2815 (-7.5%) | 3211 | 2815 (-12.3%) |
| build ms | 6.28 | 0.52 (-91.8%) | 6.77 | 0.52 (-92.4%) |
| step ms | 6.70 | 4.96 (-26.0%) | 7.19 | 5.05 (-29.8%) |
| gpu ms | 0.42 | 4.44 | 0.42 | 4.53 |

A second run of the same command an hour later, on a busier machine, gave 6.39
against 0.54 and 6.70 against 0.60 for build, and -26.9% and -27.5% for step.
The build-time factor is the stable number here; step time carries the GPU's
mood.

At 4 layers:

| | width 1 eager | width 1 compiled | width 4 eager | width 4 compiled |
|---|---|---|---|---|
| ops | 537 | 490 (-8.8%) | 566 | 490 (-13.4%) |
| build ms | 1.09 | 0.09 (-91.3%) | 1.16 | 0.09 (-92.1%) |
| step ms | 1.35 | 1.05 (-22.2%) | 1.42 | 1.07 (-25.1%) |

Read the build column, not the op column. The graph is nearly the same graph --
that is the point, it computes the same thing -- and the op count only drops
because the compiled attention step has no per-row verify arm and the layer has
no per-layer `async_eval`. What collapses is the Python: 6.28 ms to 0.52 ms at
24 layers, a factor of twelve, and it is flat in the width because a traced
graph costs the same to dispatch whatever it contains.

The `gpu ms` column moves the other way and the reason is not a regression. In
the eager arm the per-layer `async_eval` submits work while the host is still
building, so most of the GPU time has already happened by the time the host
finishes and does not appear in the difference. The compiled arm submits
nothing until the caller asks, so all of its GPU time lands in that column.
Step time, which is the honest end-to-end number, is 26% to 30% better.

That 26% is the *floor*, not the expectation. The synthetic model's GPU work is
trivial, so removing host time mostly exposes GPU time. The checkpoint's
arithmetic is the opposite -- 15.8 ms of host against 2.0 ms of GPU -- and
there a twelvefold cut in host build is most of the step. Section 6 is the
command that measures it; nothing in this workstream has run it.

### First-call compile cost and the cache

`python bench/decode/compiled_path.py cache --layers 24 --context 600`. Build
ms is host time to return from the step; step ms adds the wait for the device.

| width | 1st build | 2nd build | 1st step | 2nd step |
|---|---|---|---|---|
| 1 | 8.29 | 0.64 | 21.35 | 6.19 |
| 2 | 8.57 | 0.59 | 15.68 | 5.96 |
| 3 | 8.71 | 0.52 | 14.29 | 5.75 |
| 4 | 12.42 | 0.53 | 17.66 | 5.17 |
| 5 | 8.24 | 0.51 | 13.21 | 4.99 |
| 6 | 8.55 | 0.57 | 13.16 | 4.95 |
| 7 | 8.33 | 0.54 | 12.98 | 4.84 |
| 8 | 8.39 | 0.55 | 13.35 | 4.89 |

And across the buffer growth points, at width 1:

| tokens | capacity | 1st build | 2nd build |
|---|---|---|---|
| 200 | 256 | 8.20 | 0.52 |
| 500 | 512 | 8.51 | 0.50 |
| 1000 | 1024 | 8.31 | 0.53 |
| 2000 | 2048 | 8.18 | 0.58 |
| 3000 | 4096 | 8.11 | 0.59 |
| 5000 | 8192 | 8.32 | 0.61 |

Three things to read off this.

A new trace costs about 8.3 ms of host time at 24 layers, which is roughly the
eager build time for one step -- the trace is the same work done once. Widths 1
to 8 is eight traces, about 66 ms, paid once per process. Scaled to the
checkpoint's 48 layers that is roughly 130 ms of one-off cost, against 15.8 ms
per token saved: it pays for itself inside ten tokens.

Capacity is part of the trace key, not just width. Every growth point is
another 8.3 ms. `capacity_for` doubling above 2048 is what keeps that
logarithmic: a 64k context crosses nine growth points, not 250.

The two 12.4 ms and 21.4 ms outliers are a shared GPU, not a width effect. The
machine this ran on has a workbench on it; the 24-layer numbers in the table
above were taken paired for exactly that reason.

---

## 4. The state contract

The compiled step returns its state instead of mutating one, which is what
makes rollback a reassignment rather than a replay.

`DecodeState` holds one `LayerState` per layer. A linear layer carries
`(conv_state, ssm_state)`. An attention layer carries `(k_buf, v_buf,
index_buf, offset)`, the three buffers fixed-capacity and the offset a
rank-zero `int32` array, never a Python integer.

**Truncation.** `truncate_state(state, length, snapshot)`. The attention half is
free: moving the offset back is an assignment, and the rows past it are masked
out of the next step's softmax by the same comparison that made them causal.
Nothing is zeroed and nothing is copied. The recurrent half cannot be rewound,
exactly as `ModelState.truncate` already says, so it takes a snapshot -- the
dict `DecodeState.recurrent_arrays()` produced at that length, keyed
`layer{i}.slot{j}`, which is what `titan.adapters.mlx.state.Snapshot` already
stores. Without one it raises rather than continuing from a recurrent state a
few tokens ahead of the KV cache it is paired with.

**Speculative rollback.** `rollback_speculative_state(state, accepted,
block_size, snapshot)` is the compiled equivalent of
`LanguageModel.rollback_speculative_cache`: come back to the length before the
block plus `accepted + 1`. The vendored version trims each cache in place and
replays the recurrent state out of the captured intermediates; here the trim is
an offset assignment and the recurrent state is the snapshot staged before the
block. A test runs a width-1 step, runs a width-4 block, rolls back, runs the
width-1 step again and asserts the result is bit-identical to the first.

**Growth.** `grow_state(state, tokens)` reallocates every attention buffer to
`capacity_for(tokens)`, outside the traced region. `CompiledModel.__call__`
calls it before every step, so growth is automatic and retracing happens only
at the growth points.

**Bridging.** `read_layer_state(caches)` builds a `DecodeState` out of the
vendored caches; `write_layer_state(state, caches)` puts it back. This is the
reason the compiled path is rollback-compatible rather than a parallel
universe: everything downstream of the model -- `ModelState.truncate`,
`stage_snapshot`, the prefix-cache codec, `rollback_speculative_cache` -- keeps
working on the caches it already knows, because after the write they hold
exactly what the compiled step computed. A test takes a state out, runs a
compiled step, writes it back, and then runs an *eager* forward that carries on
from it; another test hands the result to `ModelState` and truncates.

---

## 5. The change in `qwen4_exp`, as a diff

**Applied in ROUND4.** Both hunks are in the tree; what follows is kept as the
statement of what changed and why, because the reasoning is the part that rots
if it is only in a commit message. Two differences from the diff as written:
the compiled-residual branch carries the reasoning as a comment at the call
site, and `compiled_decode_layer` reads its switch from `forward_paths`, which
now also holds the four QSA routes that ROUND3 left in
`titan/adapters/mlx/kernels.py`.

### 5.1 Make the compiled gated residual the default, MTP or not

The compiled arm in `Qwen4ExpGatedResidual.__call__` is refused whenever
Lightning MTP is on, at any width above one, and for anything but bfloat16.
None of those three gates is needed. The MTP gate guarded the *hybrid
projection* arm, which the compiled forward does not take. The width and dtype
gates existed because the compiled closure was traced once and a second shape
would have retraced -- which is fine, and is exactly what the compile cache is
for.

In `titan/adapters/mlx/vendor/mlx_vlm/models/qwen4_exp/language.py`, in
`Qwen4ExpGatedResidual.__call__`:

```diff
         compiled_forward = getattr(self, "_compiled_forward", None)
         if (
             compiled_forward is not None
             and forward_paths.enabled("compiled_gated_residual")
-            and not target_verify
             and hyper_input.ndim == 3
-            and hyper_input.shape[:2] == (1, 1)
-            and hyper_input.dtype == mx.bfloat16
-            and not getattr(self, "_titan_mtp_enabled", False)
+            and hyper_input.shape[0] == 1
         ):
-            return compiled_forward(hyper_input)
+            return compiled_forward(hyper_input, target_verify=target_verify)
         return self._forward(hyper_input, target_verify=target_verify)
```

`_forward` already takes `target_verify`, so the compiled closure has to be
built to pass it. In `compile_hyper_connections`:

```diff
     compiled = 0
     for module in _unique_hyper_connections(model):
         module._titan_mtp_enabled = bool(mtp_enabled)
         if hasattr(module, "_compiled_forward"):
             continue
-        module._compiled_forward = mx.compile(module._forward)
+        # target_verify is a Python bool, so MLX keys a trace per value of it
+        # as well as per shape. Two traces, both wanted.
+        module._compiled_forward = mx.compile(module._forward)
         compiled += 1
     return compiled
```

That second hunk is a comment only; `mx.compile` already handles a Python
keyword by tracing per value. The stamp assignment stays because
`_titan_exact_hybrid_projection` in `_forward` still reads it.

Verified before proposing, on the synthetic model. `mx.compile(module._forward)`
called with a `target_verify` keyword is bit-identical to the uncompiled
`_forward` at both values of it, at widths 1 and 4, and in float32 as well as
bfloat16 -- so all three gates come off without a numerics change. With the
stamp cleared at a 300-token context, the whole eager forward is bit-identical
at width 1 and width 4.
`titan.adapters.mlx.compiled.install_compiled_hyper_connections` does the stamp
half of this at runtime, without touching the file, for anyone who wants to
measure before merging.

Note that on checkpoint-shaped inputs `hc_fused` takes rows one to sixteen
before this branch is reached, so the diff pays on the shapes `hc_fused`
declines: anything with an `input_inject_weight`, a non-affine or unsupported
bit width, or more than sixteen rows.

### 5.2 Call the compiled blocks from the decoder layer

The compiled blocks are plain functions, so the layer does not need to import
anything structural -- it needs a place to hold the built step and its weights.
The minimal form, in `Qwen4ExpDecoderLayer`:

```diff
+    def compile_step(self, *, use_kernel: bool = True):
+        """Build this layer's compiled step once, after the weights are loaded.
+
+        Titan modification: see docs/architecture/COMPILED.md. The step is a
+        pure function of (hidden, state, weights); the caller owns the state.
+        """
+        from .....compiled import build_layer
+
+        if "ple" in self:
+            self._titan_compiled = None
+            return None
+        self._titan_compiled = build_layer(self, use_kernel=use_kernel)
+        return self._titan_compiled
+
     def __call__(
         self,
         hidden_states: mx.array,
         input_ids: mx.array,
         mask: Optional[mx.array],
         cache: Optional[Any],
         position_ids: Optional[mx.array],
         gdn_sink=None,
         target_verify: bool = False,
     ):
+        compiled = getattr(self, "_titan_compiled", None)
+        if (
+            compiled is not None
+            and forward_paths.enabled("compiled_decode_layer")
+            and gdn_sink is None
+            and "ple" not in self
+            and isinstance(cache, _TitanCompiledState)
+        ):
+            return compiled.step(hidden_states, *cache.arrays, compiled.weights)
+
         if "ple" in self:
```

with a new forward path in `forward_paths.py`, which is a name in `DEFAULTS`
and a line in `DESCRIPTIONS`:

```diff
 DEFAULTS: dict[str, bool] = {
     "eager_dispatch": True,
     "batched_verify_linear": True,
     "batched_verify_attention": True,
     "compiled_gated_residual": True,
     "cached_norm_scale": True,
+    "compiled_decode_layer": False,
 }
```

```diff
     "cached_norm_scale": (
         "Use the RMSNorm scale folded at load time rather than rebuilding "
         "1 + weight every call."
     ),
+    "compiled_decode_layer": (
+        "Run a decoder layer as one traced graph, state in and state out. "
+        "Needs the caller to hold a compiled DecodeState; see COMPILED.md."
+    ),
 }
```

It is deliberately not in `ADDED`: `ADDED` is what the before/after table in
FORWARD.md turns off to get the old forward, and this path is off by default
already.

`_TitanCompiledState` is `titan.adapters.mlx.compiled.LayerState`; the layer
only needs `isinstance` and `.arrays`. The default is off because turning it on
means the engine holds a `DecodeState` rather than a cache list, which is
`backend.py`'s decision to make, and `backend.py` is not this workstream's
either. Section 4 says what that state has to support and
`read_layer_state`/`write_layer_state` are the two functions that bridge it.

The `gdn_sink is None` condition is load-bearing: the recurrent-intermediate
capture that a replay-free rollback uses is not in the compiled graph, because
the compiled path does not need it -- it rolls back from a snapshot instead.

---

## 6. The real-model command, which ROUND4 ran

```
python bench/decode/compiled_path.py real \
    --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
    --contexts 600,64000 --widths 1,4 --repeats 20
```

It loads the checkpoint through `titan.adapters.mlx.loader.load_model`, prefills
to each context through the ordinary eager path, snapshots the recurrent state,
runs eager and compiled alternately at each width and writes
`bench/decode/results/compiled_real_<context>.json`.

It obeys the two rules in `docs/ops/INCIDENTS.md` that the 2026-09-12 panic
produced. There is exactly one `mx.synchronize()` on each side of a timed step
and no per-scope evaluation anywhere: the whole step is built, evaluated once at
the end, and synchronised once. Op counting is off by default on this
subcommand (`--ops` turns it on) and even then it only walks the graph with
`mx.export_to_dot`, which reads it without evaluating it. Results are written
under the repo, not /tmp.

### What it measured

`bench/decode/ROUND4.md` step 2 has the run and the reasoning. The short
version, at a 600-token context on the checkpoint, twenty repeats, paired:

| | width 1 eager | width 1 compiled | width 4 eager | width 4 compiled |
|---|---:|---:|---:|---:|
| build ms | 14.03 | 2.64 (-81%) | 23.05 | 2.88 (-87%) |
| step ms | 16.29 | 27.91 (+71%) | 26.74 | 33.28 (+24%) |
| gpu ms | 2.26 | 25.27 | 3.69 | 30.40 |

**The host prediction held and the step prediction did not.** Host build falls
from 14.03 ms to 2.64, five to eightfold, flat in the width, which is what this
document said it would do. Step time gets worse at both widths anyway.

Two reasons. The eager forward `async_eval`s after every decoder layer, so the
GPU is running layer *i* while the host builds layer *i+1*; the host time this
path removes was already hidden behind the GPU rather than sitting on the
critical path. And a compiled layer gives up `hc_fused` -- section 1c's own
qualification, which said the compiled residual is not the fastest thing
available on checkpoint-shaped inputs, turns out to be the larger of the two
once a step runs 96 of them. The synthetic model's 26% win in section 3 came
from a model whose GPU work is trivial; the checkpoint's is not.

So `compiled_decode_layer` is off by default and is not wired into the served
decode. The experiment that would reopen this is a `build_layer` that calls
`hc_fused` where the shapes allow it rather than always composing
`build_gated_residual`.

### Two things ROUND4 had to fix before the command ran at all

**The checkpoint has a PLE layer, and `build_model` refused the whole model.**
`build_layer` still raises for one, which is right for a caller that asked for
that layer. `build_model` now builds a *plan* instead: traced runs of layers
with the untraceable ones eager between them, one extra dispatch per such
layer. `untraceable_layers(model)` names them. The state contract grew a third
kind, `EAGER`, a placeholder carrying no arrays; `grow_state`,
`truncate_state` and `recurrent_arrays` skip it and the layer keeps its own
vendored cache, because its state is state a trace cannot hold. A model with no
such layer takes exactly the single-graph path it always did.

**`CompiledModel.__call__` now refuses a length past the indexer budget**,
which is what section 7 said the builder should do and did not. The refusal is
at call time because the budget is a property of the length. That is why the
64k half of this command has no numbers: getting them needs `build_layer_split`
wired into `build_model`, and given the 600-token result that is not urgent.

---

## 7. Risks

**Capacity waste at 64k.** The compiled attention step attends over the whole
buffer. `capacity_for(64000)` is 65536, so the waste is 2.4%, but a context
just above a doubling point wastes close to 2x of the *arithmetic*, on masked
columns. At 64k with 36 layers that is real GPU time, against a host saving
that is real too. The synthetic model cannot answer which wins; section 6 can.

**Trace count against process lifetime.** Widths 1 to 8 crossed with nine
capacity growth points is up to 72 traces, about 600 ms of one-off host cost at
48 layers, plus whatever MLX's compile cache holds in memory. A server that
serves one long sequence pays it once; a server churning short sequences at
many widths pays it once too, since the cache is keyed on shapes and not on
sequences. What has not been measured is the memory the cache holds.

**Buffer count, not bytes.** mlx-lm issue #1332 records a model dying at 11,300
tokens on a limit that counts buffers rather than bytes. The compiled path holds
*fewer* live buffers than the eager one at steady state -- fixed-capacity KV
instead of a reallocating one -- but `grow_state` briefly holds both the old and
the new buffer at every growth point, and at 64k both are large. **Run and
clear in ROUND4**: `host_overhead.py buffers --tokens 12288 --compiled` reports
0.00 KB/step of steady-state growth against the eager arm's 0.00, with active
memory stepping only at the capacity growth points and flat between them, and
the compiled arm holding 6.8 MB live at 12,288 tokens against the eager arm's
13.1. The `--compiled` flag is new; it drives the compiled step instead of the
eager forward.

**Numerics at the sparse seam.** Below the indexer budget the compiled path is
within one bf16 ULP of eager everywhere it was measured, and agrees on the
argmax at widths 1 to 6. Above the budget, the dense single graph is *not* the
same computation -- it attends to keys the eager path drops -- and
`build_layer_split` is not optional there. A caller that forgets it gets a
plausible answer that is wrong, which is the worst failure mode in this
document. **Closed in ROUND4**: `CompiledModel.__call__` refuses a length past
the budget by name, at call time, and
`tests/model/test_compiled_path.py::test_a_single_graph_is_refused_past_the_indexer_budget`
pins the refusal.

**The snapshot is the rollback.** The compiled path drops the recurrent
intermediate capture and rolls back from `ModelState`'s snapshot instead. That
is simpler and it is a real behaviour change: a rollback to a length with no
staged snapshot is an error rather than a slow path. `ModelState.truncate`
already behaves that way, so the engine's contract does not change, but the
verify cycle has to keep staging before every block.
