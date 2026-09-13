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

---

## Generation 2: the same kernels, inside the trace

Section 6 is the honest failure. Host build fell from 14.03 ms to 2.64 on the
checkpoint, five to eightfold, exactly as sections 1 to 3 said it would, and
the step got 71% worse at width 1 and 24% worse at width 4 anyway. Section 6
named two reasons and one of them is the larger: a compiled layer gave up
`hc_fused`. Section 1c had already written the qualification down -- "as a
standalone replacement in the eager path the compiled residual is not the
fastest thing available" -- and then `build_layer` composed it 96 times a step
regardless. The Gated DeltaNet's gated output norm went the same way: the eager
`Qwen4ExpRMSNormGated` takes `gdn_norm_gate`, one fused launch, and the
compiled block spelled it as an `rms_norm`, two casts, a sigmoid and a
multiply.

So generation 2 asks the narrower question. Not "can the decode forward be
traced" -- generation 1 answered that -- but "can a trace hold the kernels".

### 1. It can, at fixed shapes, and it cannot shapeless

`mx.compile` at fixed shapes accepts an `mx.fast.metal_kernel` as one opaque
node and replays it from the C++ side like any other primitive. Shapeless it
refuses, and the refusal is not a gap in MLX: a custom kernel's output shape is
something the caller *declares* in `output_shapes`, so there is nothing for a
shapeless trace to infer it from.

`python bench/decode/compiled_path.py kernels` generates this table on the
synthetic model rather than restating it from memory, because an error string
is a fact about a version. On MLX 0.32.2:

| kernel | traced at fixed shapes | shapeless |
|---|---|---|
| `hc_fused` (norm, down/inject, up) | yes | `ValueError: [Primitive::output_shapes] CustomKernel cannot infer output shapes.` |
| `gdn_norm_gate` | yes | same |
| `gated_delta_kernel` | yes | same |
| `_mrope_apply_kernel` | yes | same |
| `moe_weighted_sum` | yes | same |
| `topk_radix` | yes | same |
| `verify_accept` | yes | `ValueError: [Primitive::output_shapes] Slice cannot infer output shapes.` |
| `ple_packed_lookup` | **no** | not attempted |
| `qsa_gathered_attention` | **no** | not attempted |

The two that do not trace do not fail with an MLX error, which is the more
useful way to say it: they are not pure array functions. `ple_packed_lookup`
reads rows off SSD through an mmap and numpy in the middle of the forward.
`qsa_gathered_attention` builds its `QSAGeometry` from `int(cache.offset)` and
the pooled index bank's host-side lengths, offsets and phases, so the selection
is host work by construction rather than by formulation. Those two are the
islands in section 3 below, and no amount of retracing changes that.

`verify_accept` is worth a line of its own: it traces, and its shapeless error
is `Slice`, not `CustomKernel`, because today it is plain MLX ops. The fused
launch is a later change and the seam already exists, so when it lands its row
here will move to `CustomKernel` and nothing else about the verify step will.

### 2. What that buys, block by block

`titan/adapters/mlx/compiled_kernels.py` is the new file: each fused kernel the
eager decode path calls, written as a function of arrays alone. No module
reads, no `mx.eval`, no `try/except` around a lazy Metal compile, no
Python-level cache keyed on a shape. The Metal sources are not copied -- they
come from `hc_fused` and from `titan.kernels` -- so there is still one body per
kernel in the tree.

**The hyper-connection block.** `hc_fused_plan(module)` decides eligibility
once at build time, out of the same predicates `hc_fused.compatible` applies
per call: the layout, the quantisation, the 64-element alignment, a Metal
device. `hc_fused_body(hyper_input, arrays, plan)` is the three kernels. The
row bound is the one thing a build-time plan cannot decide, and it is decided
at *trace* time, where the shape is a Python tuple: at most sixteen rows takes
the kernels, more composes the ops form. So a compiled step never changes which
arithmetic it runs while it runs.

**The Gated DeltaNet gate.** `gdn_norm_gate_plan(module, width=...)` resolves
`gdn.norm_gate_fused` through the adapter's registry door, at build time,
exactly as the eager site resolves it -- which means `kernels.reference_only`
and `kernels.disabled` reach the compiled path, and a control arm measured here
is the control arm the server would run. The plan declines width one, because
the eager site declines width one (`x.shape[0] * x.shape[1] > 1`). The kernel
is bit-identical at width one too; declining it anyway is the whole discipline
of this generation. Same kernels means the same kernels in the same places.

Everything generation 1 already had inside the trace stays there: the gated
delta recurrence, the fused rotary application, `moe_weighted_sum`.

### 3. Islands, and how the checkpoint's 48 layers split

`island_layers(model, length=...)` names the layers that run eagerly between
the compiled segments, and `plan_layout(...)` computes the same thing from a
config's `layer_types`, `ple_layer_ids` and `indexer_budget` with nothing
loaded -- which matters, because the checkpoint is 70 GB and one process at a
time. `bench/decode/compiled_path.py layout --config .../config.json` is the
command.

For `Qwen3.8-Flash-Next-oQ4e-mtp`: 48 layers, `full_attention` at every fourth
index (3, 7, ... 47), PLE at index 2, indexer budget 2048.

| context | islands | traced segments | capacity in the trace key |
|---|---:|---:|---|
| at or under 2048 | 1 (the PLE layer) | 2, being layers 0-1 and 3-47 | yes |
| over 2048 | 13 (PLE plus the twelve sparse layers) | 12, being 0-1 and eleven runs of three | **no** |

The second row is the interesting one and the reason the trace count stops
growing exactly where the context starts getting long. Past the budget every
attention layer is an island, so no traced segment holds a KV buffer, so no
trace of theirs depends on capacity. Twelve segments across widths 1 to 8 is
**96 traces**, at 64k and at 262k alike. Under the budget it is two segments
across eight widths and, for a session that grew into that context, the nine
capacity buckets it crossed on the way: 144 traces worst case.

That replaces generation 1's estimate of "up to 72 across widths 1..8 and
capacity buckets" with a number, and it replaces generation 1's *refusal* past
the budget with a plan. `CompiledModel` still refuses a length past the budget
when the sparse layers are inside its traces, which is every generation 1
build; it stops refusing exactly when they are islands, and
`_sparse_layers_are_islands` is the condition, not a flag.

What an island costs is one extra dispatch and the Python of one eager layer.
On the synthetic model at a 3000-token context, where six of 24 layers are
islands, host build goes from 0.54 ms (all traced, under the budget) to 2.25 ms
against eager's 7.71 -- still a 3.4x cut, and a quarter of the layers now
paying full Python price.

### 4. Numerics: bit-identical, not within a ULP

Generation 1 held itself to one bf16 ULP because it computed the
hyper-connection block a different way. Generation 2 computes it the same way,
so anything short of equality is a bug rather than a tolerance, and
`tests/model/test_compiled_kernels.py` asserts equality.

Measured on the synthetic model **cast to bfloat16**, against the ordinary
eager forward with the kernels on:

* logits bit-identical at widths 1, 2, 4 and 6, at contexts 200, 300, 600 and
  2000 -- either side of the 256 and 512 growth points and up to the budget;
* logits bit-identical at 2100 and 3000, past the budget, where the sparse
  layer is an island;
* the Gated DeltaNet conv state and recurrent state compared directly, layer by
  layer, bit-identical at widths 1 and 4 at 600 and 2100;
* `hc_fused_body` bit-identical to `hc_fused.fused_forward` at widths 1, 2, 4,
  8 and 16, and `gdn_norm_gate_body` bit-identical to
  `Qwen4ExpRMSNormGated.__call__`.

The cast is load-bearing and it is the thing generation 1's synthetic numbers
were missing. `build_synthetic` leaves the model in float32, and in float32 not
one fused decode kernel is eligible: `hc_fused` declines a float32 norm weight
by layout and `gdn_norm_gate` declines a float32 input by dtype. So generation
1's whole synthetic table was measured against an eager arm with no kernels in
it, which is why it promised 26% and delivered +71%. `bf16_synthetic` in the
bench is the fix, and two tests assert both halves of it: the bfloat16 model is
one the kernels take, and the float32 model is not.

**The rollback contract is unchanged.** `truncate_state` and
`rollback_speculative_state` are exactly section 4's, including the requirement
that a rollback to a length with no staged snapshot is an error. A test runs a
width-1 step, runs a width-4 block, comes back to the length before the block
and asserts the next width-1 step is bit-identical to the first, and another
pins the `accepted + 1` arithmetic. What generation 2 changed is what a step
computes, not what the state promises.

### 5. The synthetic table

`python bench/decode/compiled_path.py generations --layers 24 --context 600
--widths 1,4 --repeats 20`. All three arms in one process on one prefill,
measured forwards and then backwards and averaged, on a bfloat16 model so the
eager arm has the fused kernels in it. Medians of four consecutive runs on a
quiet machine; a fifth run taken while the workbench beside it was busy moved
every absolute number by up to 3x and is not in here, which is why the arms are
measured together rather than in separate processes.

| | w1 eager | w1 gen 1 | w1 gen 2 | w4 eager | w4 gen 1 | w4 gen 2 |
|---|---:|---:|---:|---:|---:|---:|
| ops | 2243 | 2936 (+30.9%) | 2350 (+4.8%) | 2322 | 2936 (+26.4%) | 2332 (+0.4%) |
| build ms | 5.64 | 0.54 (-90.4%) | 0.55 (-90.3%) | 5.88 | 0.55 (-90.6%) | 0.54 (-90.9%) |
| step ms | 6.03 | 5.03 (-16.7%) | 4.25 (-29.6%) | 6.27 | 5.14 (-18.1%) | 4.39 (-30.0%) |
| gpu ms | 0.39 | 4.49 | 3.70 | 0.40 | 4.59 | 3.86 |

Read the op column first this time, not the build column. Generation 1 builds a
graph 31% larger than eager's at width 1 and 26% larger at width 4; generation
2 builds one 4.8% and 0.4% larger. That difference *is* the fused blocks: 48
layers times two hyper-connections is 96 sites where generation 1 expands three
Metal launches into a norm, two quantised matmuls, a silu, two sigmoids, a
reshape and a mean, and generation 2 does not. The GPU column says the same
thing from the other side: 3.70 against 4.49 ms at width 1, a 17.6% cut in GPU
work, from running the arithmetic the eager path already runs.

The step column is 30% better than eager and 13 points better than generation
1, and it is the *least* transferable number in the table. The synthetic
model's GPU work is 0.4 ms against 5.6 ms of host; the checkpoint's is 2.0
against 15.8 at width 1 and 3.7 against 23.1 at width 4. A model whose GPU is
idle 93% of the time cannot say what happens to a step whose regression came
from GPU work, which is exactly the mistake section 3 made and section 6 paid
for. What the synthetic model *can* say is that the mechanism section 6 blamed
is gone: the graph is the size of eager's graph again, the kernels are the
eager kernels, and the bits are the eager bits.

### 6. The trace cache

`python bench/decode/compiled_path.py warm --layers 24 --length 600
--capacities 256,512,1024,2048,4096`. Forty grid points, eight widths across
five capacities, in 1.04 seconds.

| capacity | median first call | total for eight widths | median cached call |
|---|---:|---:|---:|
| 256 | 70.7 ms | 440.7 ms | 5.50 ms |
| 512 | 11.7 ms | 93.9 ms | 4.45 ms |
| 1024 | 12.0 ms | 95.9 ms | 4.61 ms |
| 2048 | 12.5 ms | 99.9 ms | 4.85 ms |
| 4096 | 12.6 ms | 100.7 ms | 5.08 ms |

The first capacity costs six times what the later ones cost and the difference
is not the trace. It is Metal specialising every custom kernel for a row count
it has not seen: `output_shapes` changes with the width, so a new width is a
new kernel specialisation as well as a new trace. A new *capacity* at a width
already seen is the trace alone, 12 ms at 24 layers -- close to generation 1's
8.3 ms, a little more because a fused block is more nodes to construct than the
ops it replaces are cheap to construct.

So the warm cost has two terms, and only one of them scales with the grid: about
70 ms once per verify width, and about 12 ms per (width, capacity) pair after
that, at 24 layers. Scaled to 48 layers, warming widths 1 to 8 at one capacity
is roughly 1.1 seconds; the whole 96-trace past-the-budget grid is that same
1.1 seconds, because past the budget there is only one capacity.

`warm_traces` takes a time budget and a memory budget and stops on the first
breach rather than part way through a point, so a cut grid is reported as cut.
Warming runs on throwaway state -- a fresh `DecodeState` per capacity and a
fresh vendored cache per island, both dropped afterwards -- because an island's
cache is state the functional rollback cannot reach, and a warm-up that stepped
one would leave the model a token ahead of its own KV, silently, and only past
the budget. A test asserts the islands come back untouched.

**Nothing is evicted.** MLX's compile cache has no eviction and neither does
this. The measured cost of the forty-point grid is 8.1 MB of MLX active memory
and a 23.5 MB peak, on a 24-layer synthetic; the standing cost is the 8.1 MB.
That is the number generation 1's section 7 said had not been measured. At 48
layers and the checkpoint's shapes it will be larger, and the honest thing to
say is that it is measured on synthetic and predicted nowhere else.

### 7. The real-model command

Nothing in this section has run on the checkpoint. Another process owns the GPU
and the 70 GB of weights. This is the command for whoever does, and it obeys
both rules the 2026-09-12 panic produced: exactly one `mx.synchronize()` on
each side of a timed step, no per-scope evaluation anywhere, op counting off by
default and graph-only when on, results under the repo.

```
python bench/decode/compiled_path.py real --gen2 \
    --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
    --contexts 600,64000 --widths 1,4 --repeats 20
```

It writes `bench/decode/results/compiled_real_gen2_<context>.json`, alongside
generation 1's `compiled_real_<context>.json`, so the two are directly
comparable. The 64k half is now runnable, which it was not in ROUND4: past the
budget the sparse layers are islands rather than a refusal.

Run it in the order INCIDENTS rule 2 asks for. The control arm is
`kernels.reference_only`, which turns the fused paths off on *both* arms and
should leave them agreeing exactly, because the compiled step resolves its ops
through the same registry door the eager step does. Then the default
configuration, at 600 before 64k.

What to expect, stated in advance so it can be wrong. Host build should land
near generation 1's 2.64 and 2.88 ms at 600, because the tracing is the same
tracing; at 64k it should be higher, roughly four times, because a quarter of
the layers are islands paying full Python. GPU time should land near the eager
arm's 2.26 and 3.69 ms rather than generation 1's 25.27 and 30.40, because the
graph is now the eager graph. If it does, the step is GPU-bound at a few
milliseconds and this workstream is done. If host build lands where predicted
and GPU time does not, the remaining difference is something in the traced
graph that is not in the eager graph, and the op count from `--ops` is where to
look for it.

### 8. Risks

**The synthetic model cannot answer the question this section exists to
answer.** Its GPU work is 0.4 ms against 5.6 ms of host; the checkpoint's ratio
is the other way. Every step-time number in section 5 is a floor, and section
6 of this document is the record of what happens when a synthetic floor is
read as a prediction. The only thing that settles it is the command in section
7.

**Metal specialisation per verify width is a new first-call cost.** Generation
1 paid about 8.3 ms for a new trace at 24 layers. Generation 2 pays about 70 ms
the first time it sees a width, because every custom kernel in the graph gets a
new specialisation with it. A server that warms widths 1 to 8 at startup never
notices; one that meets a ninth width mid-flight pays it once, on a token.

**A build is tied to one side of the indexer budget.** `build_gen2(length=...)`
decides at build time whether the sparse layers are traced or islands, and
crossing 2048 mid-sequence needs a rebuild. That is a real thing the engine has
to own and it is not owned yet: `backend.py` is not this workstream's file. The
safe default for a server that serves long contexts is to build with
`length=None`, which islands every sparse layer at every length and gives up
the sub-budget single-graph win to avoid the rebuild.

**Capacity waste at 64k is unchanged and now matters less.** The compiled
attention step attends over the whole buffer, and a context just above a
doubling point wastes close to 2x of the arithmetic on masked columns. Past the
budget no attention layer is in a trace at all, so at 64k this is no longer a
compiled-path cost; below 2048 the absolute waste is small. The risk moved
rather than closing.

**Two paths through one builder.** `build_gated_residual` now composes either
the fused kernels or the ops form, chosen at trace time by the row count, and
the two agree bit for bit only where the kernels are eligible. A layout the
kernels decline -- a merged input projection, an unsupported bit width, a
non-bfloat16 model -- silently gets the ops form, which is correct and slower
and gives back exactly what generation 1 gave back. The failure mode is a
performance cliff with no error, and the mitigation is that
`CompiledModel.fused_kernels` records what was asked for while
`hc_fused_plan` returning `None` records what was possible. A caller that wants
to know it is on the fast path has to check the plan, not the flag.

**`verify_accept` and `topk_radix` are not in any trace yet.** They trace --
section 1's table says so -- but the verify block still calls them from Python
between compiled model steps. Folding them into the step is the obvious next
change and it is not this one; what this section establishes is that nothing
about them blocks it.
