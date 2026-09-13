"""A compiled decode path: one graph per layer type, state in and state out.

Why this module exists is in ``docs/architecture/FORWARD.md``. One decode token
on the checkpoint costs 15.8 ms of Python to build a graph the GPU runs in 2.0
ms, and the graph is about 127 primitives per layer at 2.6 microseconds each.
Nothing in that arithmetic is fixed by making the primitives faster. The only
lever is building fewer of them, and ``mx.compile`` is the lever: a traced
function replays its graph from the C++ side, so the second and every later
call costs one dispatch of a cached graph rather than a few hundred Python-level
op constructions.

FORWARD.md section 4 lists why the whole forward could not be compiled. Every
one of those reasons is a *formulation* problem rather than a property of the
arithmetic, and this module is the reformulation:

``mx.eval`` inside the function
    ``Qwen3_5GatedDeltaNet._causal_conv1d_decode`` evaluated a transposed copy
    of the conv weight on the first call so it would not rebuild the transpose
    every step. The transpose is a function of the weights alone, so it is
    hoisted to build time here (:func:`conv_decode_weight`) and the step
    function has no eval in it.

state mutated in place
    ``KVCache.update_and_fetch`` reallocates its buffer and assigns into the
    cache object. Here the KV buffer is a fixed-capacity array passed in and
    returned (:func:`build_attention_step`), written with ``mx.slice_update``
    at an offset that is itself an array, so the offset never becomes a Python
    integer and the traced shapes never change while capacity holds. Growth is
    outside the compiled region (:func:`grow_kv`).

control flow on array values
    The QSA path read ``int(cache.offset)`` and ``kv_seq_len.max().item()``.
    The compiled attention step takes the offset as an ``int32`` array and
    derives the causal mask from it in the graph.

registry dispatch per call
    ``titan.kernels`` memoises resolution, but it is still a dict lookup and a
    Python call per site per step. Every op this module uses is resolved once,
    at build time, and baked into the traced graph.

What a compiled block looks like
--------------------------------

Every block is a factory. The factory takes the module (or the config), reads
the static facts out of it -- dimensions, epsilons, quantisation bits, whether
a projection is fused -- and returns ``(step, weights)``: an ``mx.compile``'d
function and a pytree of arrays to hand it. The function is pure::

    y, conv_state, ssm_state = gdn_step(x, conv_state, ssm_state, weights)

so the caller owns the state, which is what makes rollback work at all (see
"the state contract" in ``docs/architecture/COMPILED.md``).

Two facts about ``mx.compile`` shape the design and are worth stating because
they were measured rather than assumed (``tests/model/test_compiled_path.py``
asserts both):

* ``shapeless=True`` is not available to most of this. Three primitives these
  blocks cannot avoid refuse to infer their output shapes without concrete
  inputs, each with its own error, all of them ``ValueError`` from
  ``[Primitive::output_shapes]``:

  - ``CustomKernel cannot infer output shapes`` -- any
    ``mx.fast.metal_kernel``, which is the Gated DeltaNet recurrence
    (``gated_delta_kernel``) and the fused rotary application
    (``_mrope_apply_kernel``).
  - ``Slice cannot infer output shapes`` -- ``mx.argpartition`` followed by a
    negative-index slice, which is the MoE router's top-k, and the conv window
    in :func:`causal_conv1d_decode`.
  - ``Split cannot infer output shapes`` -- ``mx.split`` at explicit indices,
    which is the Gated DeltaNet's q/k/v split and the attention block's
    query/gate split.

  So the default is a shape-specialised trace, and MLX's own compile cache
  keys on the input shapes: widths 1..8 cost eight traces, which is eight
  first-call compiles and then nothing.
* Where shapeless *does* work it is used, because it collapses those eight
  traces into one. That is the hyper-connection blocks, which are projections
  and elementwise arithmetic. ``mx.slice_update`` at an array offset and
  ``mx.gather_qmm`` both infer shapes fine; they are not what blocks the
  others.
* A shapeless trace still reports the *first* call's concrete shapes inside the
  function, so a block written for it must not read ``x.shape``.
  ``mx.unflatten`` and ``mx.flatten`` say what a reshape would say and survive
  a width change; a literal reshape bakes the first width into the graph and
  the second width raises ``[reshape] Cannot reshape array of size N``.

What is not here, and why, is in COMPILED.md section "what did not compile".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from .kernels import get as _titan_op
from .vendor.mlx_lm.models.gated_delta import (
    gated_delta_kernel,
    gated_delta_ops,
)

__all__ = [
    "ATTENTION",
    "LINEAR",
    "CompiledLayer",
    "CompiledModel",
    "DecodeState",
    "LayerState",
    "LayoutPlan",
    "LinearSpec",
    "WarmReport",
    "build_gen2",
    "capacity_buckets",
    "island_layers",
    "plan_layout",
    "warm_traces",
    "build_attention_step",
    "build_gdn_step",
    "build_gated_residual",
    "build_layer",
    "build_layer_split",
    "build_model",
    "build_moe_step",
    "capacity_for",
    "EAGER",
    "conv_decode_weight",
    "causal_conv1d_decode",
    "eval_weights",
    "grow_kv",
    "grow_state",
    "hyper_inject",
    "hyper_inject_body",
    "install_compiled_hyper_connections",
    "layer_weights",
    "new_state",
    "pad_sparse_mask",
    "read_layer_state",
    "rollback_speculative_state",
    "split_linear",
    "truncate_state",
    "write_layer_state",
]


# ---------------------------------------------------------------------------
# linears: the static half and the array half
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearSpec:
    """Everything about a projection that is not an array.

    A compiled function may not branch on an array, but it may branch on any of
    this: it is all known when the trace is taken. Splitting a projection into
    a spec and a tuple of arrays is what lets the arrays be function arguments
    (so the caller, not a closure, owns them) while the quantisation bits stay
    a Python constant folded into the graph.
    """

    quantized: bool
    bits: int = 4
    group_size: int = 64
    mode: str = "affine"
    biases: bool = True
    bias: bool = False


def split_linear(linear: nn.Module) -> tuple[LinearSpec, tuple[mx.array, ...]]:
    """Split ``nn.Linear`` or ``nn.QuantizedLinear`` into (spec, arrays)."""
    quantized = "scales" in linear
    if not quantized:
        arrays = [linear.weight]
        if "bias" in linear:
            arrays.append(linear.bias)
        return LinearSpec(quantized=False, bias="bias" in linear), tuple(arrays)

    biases = linear.get("biases", None) is not None
    arrays = [linear.weight, linear.scales]
    if biases:
        arrays.append(linear.biases)
    if "bias" in linear:
        arrays.append(linear.bias)
    spec = LinearSpec(
        quantized=True,
        bits=int(linear.bits),
        group_size=int(linear.group_size),
        mode=getattr(linear, "mode", "affine"),
        biases=biases,
        bias="bias" in linear,
    )
    return spec, tuple(arrays)


def call_linear(spec: LinearSpec, weights: Sequence[mx.array], x: mx.array):
    """Apply a projection split by :func:`split_linear`. Graph-only."""
    if not spec.quantized:
        y = x @ weights[0].T
        if spec.bias:
            y = y + weights[1]
        return y
    scales = weights[1]
    biases = weights[2] if spec.biases else None
    y = mx.quantized_matmul(
        x,
        weights[0],
        scales,
        biases,
        transpose=True,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    if spec.bias:
        y = y + weights[-1]
    return y


def _norm_scale(norm: nn.Module) -> mx.array:
    """The folded ``1 + weight`` an ``Qwen4ExpRMSNorm`` normalises with.

    The same value ``prepare_scale`` caches, computed here so a compiled block
    works on a module the loader never prepared (every synthetic model, and any
    module whose weights were replaced after load).
    """
    scale = 1.0 + norm.weight.astype(mx.float32)
    group_size = getattr(norm, "group_size", None)
    if group_size is not None:
        scale = scale.reshape(-1, group_size)
    return scale


def _rms_norm(x: mx.array, scale: mx.array, eps: float, group_size: Optional[int]):
    """``Qwen4ExpRMSNorm.__call__``, op for op, with the scale passed in.

    Written without reading ``x.shape``. A ``shapeless=True`` trace still hands
    the function the *first* call's concrete shapes, so a reshape spelled with
    them bakes that width into the graph and the second width raises
    ``ValueError: [reshape] Cannot reshape array of size N into shape (...)``.
    ``mx.unflatten`` and ``mx.flatten`` say the same thing relative to the
    trailing axis and survive a width change.
    """
    dtype = x.dtype
    if group_size is None:
        return mx.fast.rms_norm(x, scale, eps).astype(dtype)
    y = mx.unflatten(x.astype(mx.float32), -1, (-1, group_size))
    y = mx.fast.rms_norm(y, None, eps) * scale
    return mx.flatten(y, -2, -1).astype(dtype)


# ---------------------------------------------------------------------------
# (a) the causal depthwise conv, without the eval
# ---------------------------------------------------------------------------


def conv_decode_weight(conv1d: nn.Module) -> mx.array:
    """The transposed fp32 conv taps, ``[kernel, channels]``.

    ``Qwen3_5GatedDeltaNet._causal_conv1d_decode`` built this on the first
    decode and called ``mx.eval`` on it so the transpose would not reappear in
    every later step's graph. That eval is the first of FORWARD.md's three
    blockers. It is not needed: the value depends only on the weights, so it
    belongs at build time, and the caller evaluates it once with the rest of
    the extracted weights.
    """
    return conv1d.weight[:, :, 0].T.astype(mx.float32)


def causal_conv1d_decode(
    conv_state: mx.array,
    x: mx.array,
    weight: mx.array,
    kernel_size: int,
) -> tuple[mx.array, mx.array]:
    """One causal depthwise conv step over a width, functionally.

    ``conv_state`` is ``[B, kernel_size - 1, C]`` and ``x`` is ``[B, W, C]``.
    Returns the conv output over the W new positions and the next conv state.
    Nothing is mutated and nothing is evaluated: the window is a concatenate,
    the taps are a Python loop over a compile-time constant, and the new state
    is a slice of the window.

    The accumulation is fp32, which is what the vendored decode arm does
    (``_qwen3_5_decode_depthwise_conv``). The vendored *verify* arm calls
    ``nn.Conv1d`` and accumulates in bf16; at kernel size 4 the two differ by
    well under a bf16 ULP and this one is the more accurate of the pair.
    """
    window = mx.concatenate([conv_state, x], axis=1)
    width = x.shape[1]
    acc = None
    for tap in range(kernel_size):
        piece = window[:, tap : tap + width, :].astype(mx.float32)
        term = piece * weight[tap][None, None, :]
        acc = term if acc is None else acc + term
    keep = kernel_size - 1
    return acc.astype(x.dtype), mx.contiguous(window[:, -keep:, :])


# ---------------------------------------------------------------------------
# (a) the Gated DeltaNet decode step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GDNStatic:
    hidden_size: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_dim: int
    kernel_size: int
    eps: float
    gate_activation: str
    use_kernel: bool


def gdn_weights(module: nn.Module) -> dict[str, Any]:
    """Every array ``build_gdn_step`` needs, keyed by name."""
    specs = {}
    arrays: dict[str, Any] = {}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        spec, values = split_linear(getattr(module, name))
        specs[name] = spec
        arrays[name] = values
    arrays["conv"] = conv_decode_weight(module.conv1d)
    arrays["A_log"] = module.A_log
    arrays["dt_bias"] = module.dt_bias
    arrays["norm"] = module.norm.weight
    return specs, arrays


def build_gdn_step(
    module: nn.Module,
    *,
    use_kernel: bool = True,
    compiled: bool = True,
    fused_kernels: bool = False,
):
    """``(x, conv_state, ssm_state, weights) -> (y, conv_state, ssm_state)``.

    One compiled graph for the whole linear-attention branch: the four input
    projections, the causal conv window and its state, the gated delta
    recurrence, the gated output norm and the output projection.

    Not shapeless. ``gated_delta_kernel`` is a ``mx.fast.metal_kernel`` and a
    custom kernel cannot infer output shapes under a shapeless trace, so this
    specialises per width. Widths 1..8 are eight traces.

    ``fused_kernels`` is generation 2: the gated output norm goes through
    ``gdn_norm_gate``, the fused Metal launch the eager
    ``Qwen4ExpRMSNormGated`` takes, rather than the ``rms_norm`` plus cast plus
    sigmoid plus multiply the ops form spells. The eager site declines a single
    row, so this declines width one for the same reason: bit-identity with
    eager means taking the kernel exactly where eager takes it.
    """
    specs, _ = gdn_weights(module)
    static = GDNStatic(
        hidden_size=module.hidden_size,
        num_v_heads=module.num_v_heads,
        num_k_heads=module.num_k_heads,
        head_k_dim=module.head_k_dim,
        head_v_dim=module.head_v_dim,
        key_dim=module.key_dim,
        value_dim=module.value_dim,
        conv_dim=module.conv_dim,
        kernel_size=module.conv_kernel_size,
        eps=module.layer_norm_epsilon,
        gate_activation=getattr(module.norm, "activation", "sigmoid"),
        use_kernel=use_kernel and mx.metal.is_available(),
    )
    recurrence = gated_delta_kernel if static.use_kernel else gated_delta_ops
    # Resolved once, here, exactly as the eager site resolves it. ``None`` when
    # the registry is in ``reference_only`` or the op is disabled, which is
    # what makes the compiled path's control arm the server's control arm.
    norm_gate_op = (
        _titan_op("gdn.norm_gate_fused") if fused_kernels else None
    )

    def step(x, conv_state, ssm_state, weights):
        batch, width, _ = x.shape

        mixed_qkv = call_linear(specs["in_proj_qkv"], weights["in_proj_qkv"], x)
        z = call_linear(specs["in_proj_z"], weights["in_proj_z"], x)
        b = call_linear(specs["in_proj_b"], weights["in_proj_b"], x)
        a = call_linear(specs["in_proj_a"], weights["in_proj_a"], x)
        z = z.reshape(batch, width, -1, static.head_v_dim)

        conv_out, conv_state = causal_conv1d_decode(
            conv_state, mixed_qkv, weights["conv"], static.kernel_size
        )
        conv_out = nn.silu(conv_out)

        q, k, v = [
            t.reshape(batch, width, heads, dim)
            for t, heads, dim in zip(
                mx.split(conv_out, [static.key_dim, 2 * static.key_dim], -1),
                (static.num_k_heads, static.num_k_heads, static.num_v_heads),
                (static.head_k_dim, static.head_k_dim, static.head_v_dim),
            )
        ]
        inv_scale = static.head_k_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # ``_compute_g_beta`` is itself a compiled shapeless function in the
        # vendored code; inlined here so the whole block is one graph.
        g = mx.exp(
            -mx.exp(weights["A_log"].astype(mx.float32))
            * nn.softplus(a + weights["dt_bias"])
        )
        beta = mx.sigmoid(b)
        out, ssm_state = recurrence(q, k, v, g, beta, ssm_state, None)

        # The eager ``Qwen4ExpRMSNormGated`` takes the fused kernel at more
        # than one row and the ops below at one row; both are the same bits.
        if (
            norm_gate_op is not None
            and batch * width > 1
            and out.dtype in (mx.bfloat16, mx.float16)
            and z.shape == out.shape
            and z.dtype == out.dtype
        ):
            y = norm_gate_op(
                out,
                z,
                weights["norm"].astype(out.dtype),
                eps=static.eps,
                activation=0 if static.gate_activation == "sigmoid" else 1,
            )
        else:
            y = mx.fast.rms_norm(out, weights["norm"], static.eps).astype(mx.float32)
            gate = z.astype(mx.float32)
            gate = (
                mx.sigmoid(gate)
                if static.gate_activation == "sigmoid"
                else nn.silu(gate)
            )
            y = (y * gate).astype(out.dtype)

        y = call_linear(
            specs["out_proj"], weights["out_proj"], y.reshape(batch, width, -1)
        )
        return y, conv_state, ssm_state

    return mx.compile(step) if compiled else step


# ---------------------------------------------------------------------------
# (b) the MoE block
# ---------------------------------------------------------------------------


def moe_weights(module: nn.Module) -> tuple[dict, dict]:
    specs: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    for name in ("gate", "shared_expert_gate"):
        spec, values = split_linear(getattr(module, name))
        specs[name] = spec
        arrays[name] = values
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(module.shared_expert, name)
        spec, values = split_linear(proj)
        specs[f"shared_{name}"] = spec
        arrays[f"shared_{name}"] = values

    switch = module.switch_mlp
    fused = "gate_up_proj" in switch
    specs["fused_gate_up"] = fused
    names = ("gate_up_proj", "down_proj") if fused else (
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    for name in names:
        proj = switch[name]
        spec, values = split_linear(proj)
        specs[f"switch_{name}"] = spec
        arrays[f"switch_{name}"] = values
    return specs, arrays


def _switch_linear(spec: LinearSpec, weights, x, indices):
    """One routed expert gather, the ``SwitchLinear`` body without the sort.

    The sort in ``SwitchGLU.__call__`` fires at ``indices.size >= 64``. A decode
    is ``1 * 1 * top_k`` indices and a width-8 verify is ``8 * top_k``, so for
    every width this module is built for the stock path is the unsorted one and
    the two Titan gather ops (which require ``sorted_indices``) decline. That
    is why there is no registry call here: at these widths there never was one.
    """
    if spec.quantized:
        out = mx.gather_qmm(
            x,
            weights[0],
            weights[1],
            weights[2] if spec.biases else None,
            rhs_indices=indices,
            transpose=True,
            group_size=spec.group_size,
            bits=spec.bits,
            mode=spec.mode,
            sorted_indices=False,
        )
    else:
        out = mx.gather_mm(
            x, weights[0].swapaxes(-1, -2), rhs_indices=indices, sorted_indices=False
        )
    if spec.bias:
        out = out + mx.expand_dims(weights[-1][indices], -2)
    return out


def build_moe_step(module: nn.Module, *, compiled: bool = True):
    """``(x, weights) -> y`` for the sparse MoE block, widths 1..8.

    The routing is in the graph: softmax, ``argpartition`` for the top-k, the
    renormalised scores, the gathered expert matmuls on the fused gate/up and
    on down, the weighted sum, and the shared expert with its own gate. No
    ``.item()``, no Python loop over tokens, no per-call registry lookup -- the
    weighted-sum kernel is resolved once, here, and traced into the graph.

    Not shapeless: ``mx.argpartition`` feeding a negative-index slice raises
    ``Slice cannot infer output shapes`` under a shapeless trace.
    """
    specs, _ = moe_weights(module)
    top_k = int(module.top_k)
    fused = specs["fused_gate_up"]
    weighted_sum = _titan_op("moe.weighted_sum")

    def step(x, weights):
        gates = call_linear(specs["gate"], weights["gate"], x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        routed = mx.expand_dims(x, (-2, -3))
        if fused:
            gate_up = _switch_linear(
                specs["switch_gate_up_proj"],
                weights["switch_gate_up_proj"],
                routed,
                indices,
            )
            x_gate, x_up = mx.split(gate_up, 2, axis=-1)
        else:
            x_up = _switch_linear(
                specs["switch_up_proj"], weights["switch_up_proj"], routed, indices
            )
            x_gate = _switch_linear(
                specs["switch_gate_proj"], weights["switch_gate_proj"], routed, indices
            )
        hidden = nn.silu(x_gate) * x_up
        y = _switch_linear(
            specs["switch_down_proj"], weights["switch_down_proj"], hidden, indices
        ).squeeze(-2)

        summed = None
        if weighted_sum is not None:
            summed = weighted_sum(y, None, scores, (*y.shape[:-2], y.shape[-1]))
        if summed is None:
            summed = (y * scores[..., None]).sum(axis=-2)

        gate_shared = call_linear(
            specs["shared_gate_proj"], weights["shared_gate_proj"], x
        )
        up_shared = call_linear(specs["shared_up_proj"], weights["shared_up_proj"], x)
        shared = call_linear(
            specs["shared_down_proj"],
            weights["shared_down_proj"],
            nn.silu(gate_shared) * up_shared,
        )
        shared = (
            mx.sigmoid(
                call_linear(
                    specs["shared_expert_gate"], weights["shared_expert_gate"], x
                )
            )
            * shared
        )
        return summed + shared

    return mx.compile(step) if compiled else step


# ---------------------------------------------------------------------------
# (c) the hyper-connection gated residual, the injection and the final mixer
# ---------------------------------------------------------------------------


def _compiled_kernels():
    """``titan.adapters.mlx.compiled_kernels``, imported late.

    Late because that module imports the vendored ``hc_fused`` for its Metal
    sources, and this module is imported by the vendored decoder layer. A
    module-level import would close the circle.
    """
    from . import compiled_kernels

    return compiled_kernels


def _hc_fused_takes(hyper_input: mx.array) -> bool:
    """Whether the three ``hc_fused`` kernels take this concrete input.

    The row bound and the dtype, which is everything ``hc_fused.compatible``
    checks that a build-time plan cannot. Inside a per-shape trace the shape is
    a Python tuple, so this is a Python branch resolved when the trace is
    taken, not a branch on an array.
    """
    ck = _compiled_kernels()
    if hyper_input.ndim != 3 or hyper_input.dtype != mx.bfloat16:
        return False
    rows = hyper_input.shape[0] * hyper_input.shape[1]
    return 1 <= rows <= ck.HC_MAX_ROWS


def gated_residual_weights(
    module: nn.Module, *, fused_kernels: bool = False
) -> tuple[dict, dict]:
    """The static half and the array half of one ``Qwen4ExpGatedResidual``.

    ``fused_kernels`` adds the arrays the three ``hc_fused`` Metal kernels take
    (raw norm weight, the epsilon array, the packed quantised projections) on
    top of the ones the ops form takes. Both sets travel, because a layer built
    for the fused kernels still composes the ops form for any width the kernels
    decline; see the generation 2 section of COMPILED.md.
    """
    specs: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    fused = getattr(module, "input_inject_weight", None) is not None
    specs["fused_inject"] = fused
    specs["combine"] = "block_inject_weight" in module or fused
    names = ["input_mix_weight_up"]
    if fused:
        names.append("input_inject_weight")
    else:
        names.append("input_mix_weight_down")
        if "block_inject_weight" in module:
            names.append("block_inject_weight")
    for name in names:
        spec, values = split_linear(getattr(module, name))
        specs[name] = spec
        arrays[name] = values
    arrays["hc_norm"] = _norm_scale(module.hc_norm)
    if fused_kernels:
        plan = _compiled_kernels().hc_fused_plan(module)
        specs["hc_plan"] = plan
        if plan is not None:
            arrays["hc_fused"] = _compiled_kernels().hc_fused_arrays(module)
    return specs, arrays


def build_gated_residual(
    module: nn.Module, *, compiled: bool = True, fused_kernels: bool = False
):
    """``(hyper_input, weights) -> mixed`` or ``(mixed, hyper_input, gates)``.

    This is ``Qwen4ExpGatedResidual._forward`` written functionally. The
    vendored module already compiles ``_forward`` under the
    ``compiled_gated_residual`` path, but that arm is gated off whenever
    Lightning MTP is enabled (``_titan_mtp_enabled``), which is production, and
    it is gated to width 1 and bfloat16. Neither gate is needed. The MTP gate
    guarded the *hybrid projection* arm, which this function does not take; the
    width gate existed because the compiled closure was traced once and MLX
    would have retraced per width anyway, which is fine and is what happens
    here. :func:`install_compiled_hyper_connections` lifts both on a live model
    without touching the vendored file; COMPILED.md carries the diff that makes
    it the default.

    Shapeless: this block is projections, elementwise arithmetic, a mean and
    reshapes, all of which infer shapes. One trace serves every width.

    ``fused_kernels`` is generation 2. It calls the same three ``hc_fused``
    Metal kernels the eager path calls, as opaque nodes inside the trace, for
    every width those kernels take. That gives up shapelessness -- a custom
    kernel cannot infer output shapes -- which is the trade the generation 2
    section of COMPILED.md is about: one trace per width against ninety-six
    fused blocks a step that generation 1 was computing out of plain ops.
    Widths the kernels decline (above sixteen rows, or a layout with a merged
    input projection) fall back to the ops form at *trace* time, so a compiled
    step never silently changes which arithmetic it runs.
    """
    specs, _ = gated_residual_weights(module, fused_kernels=fused_kernels)
    hc_plan = specs.get("hc_plan")
    hc_count = int(module.hc_count)
    hidden_size = int(module.hidden_size)
    hc_lowrank = int(module.hc_lowrank)
    eps = float(module.hc_norm.eps)
    combine = specs["combine"]
    fused = specs["fused_inject"]

    def step(hyper_input, weights):
        if hc_plan is not None and _hc_fused_takes(hyper_input):
            return _compiled_kernels().hc_fused_body(
                hyper_input, weights["hc_fused"], hc_plan
            )
        normed = _rms_norm(hyper_input, weights["hc_norm"], eps, hidden_size)
        if fused:
            combined = call_linear(
                specs["input_inject_weight"], weights["input_inject_weight"], normed
            )
            mix = combined[..., :hc_lowrank]
            block_injection = combined[..., hc_lowrank : hc_lowrank + hc_count]
        else:
            mix = call_linear(
                specs["input_mix_weight_down"], weights["input_mix_weight_down"], normed
            )
            block_injection = (
                call_linear(
                    specs["block_inject_weight"],
                    weights["block_inject_weight"],
                    normed,
                )
                if combine
                else None
            )
        mix = nn.silu(mix / hc_count)
        mix = mx.sigmoid(
            call_linear(
                specs["input_mix_weight_up"], weights["input_mix_weight_up"], mix
            )
        )
        mix = mx.unflatten(mix, -1, (hc_count, hidden_size))
        streams = mx.unflatten(normed, -1, (hc_count, hidden_size))
        mixed_input = mx.mean(mix * streams, axis=-2)
        if block_injection is None:
            return mixed_input
        return mixed_input, hyper_input, 2 * mx.sigmoid(block_injection / hc_count)

    if not compiled:
        return step
    # A fused block holds custom kernels, which have no shapeless trace. That
    # is not a loss here: this function is composed *into* a layer trace, which
    # is per-shape anyway because the Gated DeltaNet recurrence and the rotary
    # application are custom kernels too.
    return mx.compile(step, shapeless=hc_plan is None)


def hyper_inject_body(
    hyper_input: mx.array, branch: mx.array, injection_weights: mx.array
) -> mx.array:
    """Write a branch's output back into the n-wide residual stream.

    Identical to the vendored ``_hyper_inject``; repeated here so the compiled
    layer does not import the layer module it is meant to replace. The body is
    separate from the compiled wrapper because the whole-layer step inlines it
    into its own trace, and a compiled function called from inside another
    trace is a seam MLX does not need to see.
    """
    injection = branch[..., None, :] * injection_weights[..., None]
    return hyper_input + mx.flatten(injection, -2, -1)


hyper_inject = mx.compile(hyper_inject_body, shapeless=True)


def install_compiled_hyper_connections(model: nn.Module) -> int:
    """Lift the MTP stamp so the vendored compiled ``_forward`` can run.

    ``compile_hyper_connections`` stamps ``_titan_mtp_enabled`` on each
    hyper-connection and ``Qwen4ExpGatedResidual.__call__`` then refuses the
    compiled arm, which in production means it never runs. This clears the
    stamp and makes sure a compiled forward exists. It is reversible and it
    touches no vendored file.

    What it does *not* do is lift the other three gates in that ``__call__``:
    width one, bfloat16, and not target-verify. Those are in a file this
    module does not own, so the change that makes the compiled residual the
    default at every width is the diff in COMPILED.md.

    It is also worth being plain about what this is worth on its own. Rows one
    to sixteen on a checkpoint-shaped hyper-connection are taken by
    ``hc_fused`` before ``__call__`` looks at the compiled arm at all, and
    ``hc_fused`` is three fused Metal kernels, which beats a compiled graph of
    the ops form. The compiled residual earns its place *inside*
    :func:`build_layer`, where what it removes is not the ops but the Python
    between them.
    """
    seen: set[int] = set()
    installed = 0
    for _, module in _walk_modules(model):
        if type(module).__name__ != "Qwen4ExpGatedResidual":
            continue
        if id(module) in seen:
            continue
        seen.add(id(module))
        module._titan_mtp_enabled = False
        if not hasattr(module, "_compiled_forward"):
            module._compiled_forward = mx.compile(module._forward)
        installed += 1
    return installed


def _walk_modules(root: nn.Module):
    stack = [("", root)]
    while stack:
        prefix, module = stack.pop()
        yield prefix, module
        children = module.children()
        for name, child in children.items():
            if isinstance(child, nn.Module):
                stack.append((f"{prefix}.{name}", child))
            elif isinstance(child, (list, tuple)):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        stack.append((f"{prefix}.{name}.{index}", item))


# ---------------------------------------------------------------------------
# (d) the dense attention step, on a preallocated KV buffer
# ---------------------------------------------------------------------------


def capacity_for(tokens: int, step: int = 256) -> int:
    """The buffer capacity that holds *tokens*, rounded to a growth point.

    Capacity is what the compiled attention step's traced shapes depend on, so
    the growth schedule is the trace schedule: every distinct capacity is one
    more compile. Stepping by 256 up to 2048 and doubling after keeps the
    number of traces logarithmic in the context and the waste under 2x.
    """
    if tokens <= 0:
        return step
    if tokens <= 2048:
        return ((tokens + step - 1) // step) * step
    capacity = 2048
    while capacity < tokens:
        capacity *= 2
    return capacity


def grow_kv(buffer: Optional[mx.array], capacity: int, sample: mx.array) -> mx.array:
    """Reallocate a KV buffer to *capacity*, outside any compiled region.

    Growth is the one thing the compiled step cannot do: it changes a shape,
    and a shape change is a retrace. Keeping it here means the traced graph
    sees a stable capacity and the host pays for growth only at the points
    :func:`capacity_for` chose.
    """
    batch, heads, _, dim = sample.shape
    grown = mx.zeros((batch, heads, capacity, dim), dtype=sample.dtype)
    if buffer is not None and buffer.shape[2]:
        held = min(buffer.shape[2], capacity)
        grown = mx.slice_update(
            grown, buffer[:, :, :held, :], mx.array([0], mx.int32), axes=(2,)
        )
    return grown


def attention_weights(module: nn.Module) -> tuple[dict, dict]:
    specs: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        spec, values = split_linear(getattr(module, name))
        specs[name] = spec
        arrays[name] = values
    arrays["q_norm"] = _norm_scale(module.q_norm)
    arrays["k_norm"] = _norm_scale(module.k_norm)
    arrays["inv_freq"] = module.rotary_emb.inv_freq
    selector = module.rotary_emb.position_selector
    arrays["position_selector"] = (
        selector if selector is not None else mx.zeros((1,), mx.int32)
    )
    indexer = getattr(module, "indexer", None)
    specs["has_indexer"] = indexer is not None
    if indexer is not None:
        spec, values = split_linear(indexer.index_qk_proj)
        specs["index_qk_proj"] = spec
        arrays["index_qk_proj"] = values
    return specs, arrays


def build_attention_step(module: nn.Module, *, compiled: bool = True):
    """``(x, k_buf, v_buf, offset, weights) -> (y, k_buf, v_buf, offset)``.

    The KV cache is two fixed-capacity arrays and an ``int32`` offset array.
    The step writes the new rows with ``mx.slice_update`` at the offset -- the
    offset stays an array, so nothing calls ``int()`` on it and nothing
    reallocates -- and attends over the *whole* capacity with a causal mask
    derived from the offset in the graph. Columns past the write are masked,
    and a masked column contributes an exact zero to the softmax, so the result
    is the same reduction the sliced eager arm performs.

    Attending over capacity rather than over the live length is what buys the
    stable shape: one trace covers every sequence length that fits, and only a
    capacity change retraces. The cost is arithmetic on masked columns, bounded
    at under 2x by :func:`capacity_for`, against a GPU that is idle 87% of the
    time.

    Not shapeless: the rotary application is a custom Metal kernel.

    ``sparse_mask`` is the seam for QSA. Below the indexer's budget the eager
    path is dense and this function reproduces it exactly. Above the budget the
    eager path selects blocks, and the selection is host work this function does
    not do; a caller that wants the sparse arm computes the boolean selection
    outside and passes it in padded to capacity. See COMPILED.md.
    """
    from .vendor.mlx_vlm.models.rope_utils import (
        _fast_mrope_apply,
        _mrope_apply_kernel,
    )

    specs, _ = attention_weights(module)
    heads = int(module.num_attention_heads)
    kv_heads = int(module.num_key_value_heads)
    head_dim = int(module.head_dim)
    scale = float(module.scale)
    eps_q = float(module.q_norm.eps)
    eps_k = float(module.k_norm.eps)
    group_q = getattr(module.q_norm, "group_size", None)
    group_k = getattr(module.k_norm, "group_size", None)
    rope = module.rotary_emb
    rope_kernel = _mrope_apply_kernel(rope.dim, 2, rope.pairing)
    has_indexer = specs["has_indexer"]
    indexer_heads = (
        module.indexer.n_heads + module.indexer.kv_heads if has_indexer else 0
    )
    indexer_dim = module.indexer.head_dim if has_indexer else 0
    indexer_qheads = module.indexer.n_heads if has_indexer else 0

    def step(x, k_buf, v_buf, offset, weights, sparse_mask=None):
        batch, width, _ = x.shape
        capacity = k_buf.shape[2]

        q_out = call_linear(specs["q_proj"], weights["q_proj"], x)
        keys = call_linear(specs["k_proj"], weights["k_proj"], x)
        values = call_linear(specs["v_proj"], weights["v_proj"], x)

        queries, gate = mx.split(q_out.reshape(batch, width, heads, -1), 2, axis=-1)
        gate = gate.reshape(batch, width, -1)
        queries = _rms_norm(queries, weights["q_norm"], eps_q, group_q).transpose(
            0, 2, 1, 3
        )
        keys = _rms_norm(
            keys.reshape(batch, width, kv_heads, -1), weights["k_norm"], eps_k, group_k
        ).transpose(0, 2, 1, 3)
        values = values.reshape(batch, width, kv_heads, -1).transpose(0, 2, 1, 3)

        positions = offset + mx.arange(width, dtype=offset.dtype)
        position_ids = mx.broadcast_to(positions[None, :], (batch, width))
        if rope_kernel is not None:
            queries, keys = _fast_mrope_apply(
                rope_kernel,
                queries,
                keys,
                position_ids,
                weights["inv_freq"],
                weights["position_selector"],
            )
        else:
            cos, sin = rope(keys, position_ids)
            from .vendor.mlx_vlm.models.rope_utils import (
                apply_multimodal_rotary_pos_emb,
            )

            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin, mrope_section=rope.mrope_section,
                unsqueeze_dim=1, style=rope.style,
            )

        start = mx.reshape(offset, (1,)).astype(mx.int32)
        k_buf = mx.slice_update(k_buf, keys, start, axes=(2,))
        v_buf = mx.slice_update(v_buf, values, start, axes=(2,))

        columns = mx.arange(capacity, dtype=mx.int32)
        allowed = columns[None, None, None, :] <= positions[None, None, :, None]
        if sparse_mask is not None:
            allowed = allowed & sparse_mask
        out = mx.fast.scaled_dot_product_attention(
            queries, k_buf, v_buf, scale=scale, mask=allowed
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, width, -1)
        y = call_linear(specs["o_proj"], weights["o_proj"], out * mx.sigmoid(gate))

        # The QSA indexer's raw key row for this step. The selection itself is
        # not in this graph (see the docstring), but the raw keys are cheap --
        # one projection and a reshape -- and leaving them out would let the
        # indexer's own cache drift out of alignment with the KV it indexes,
        # which the state contract in COMPILED.md forbids. The caller pushes
        # them into the cache with ``cache.update_indexer``.
        index_keys = None
        if has_indexer:
            projected = call_linear(
                specs["index_qk_proj"], weights["index_qk_proj"], x
            ).reshape(batch, width, indexer_heads, indexer_dim)
            index_keys = projected[:, :, indexer_qheads:].squeeze(2)
        return y, k_buf, v_buf, offset + width, index_keys

    return mx.compile(step) if compiled else step


# ---------------------------------------------------------------------------
# whole-layer steps
# ---------------------------------------------------------------------------

LINEAR = "linear"
ATTENTION = "attention"
#: A layer that cannot be traced and runs eagerly against a vendored cache,
#: with a placeholder in the :class:`DecodeState` so the indices still line up.
#: One layer of 48 on the checkpoint is one; see :func:`build_model`.
EAGER = "eager"


@dataclass
class CompiledLayer:
    """One decoder layer as a single traced graph, plus its arrays.

    ``kind`` decides the calling convention, which is the only thing about a
    layer the model step has to know:

    ``linear``
        ``(hidden, conv_state, ssm_state, weights) -> (hidden, conv_state,
        ssm_state)``
    ``attention``
        ``(hidden, k_buf, v_buf, index_buf, offset, weights, sparse_mask)
        -> (hidden, k_buf, v_buf, index_buf, offset)``

    Both take the whole 4-wide hyper-connection residual in and give it back:
    the two gated residuals, the branch, the two injections and the MoE block
    are one graph, so a layer costs one dispatch of a cached trace instead of
    the 127 Python-level op constructions FORWARD.md counted.
    """

    kind: str
    step: Callable[..., Any]
    weights: dict[str, Any]
    module: Any = None


def layer_weights(layer: nn.Module, *, fused_kernels: bool = False) -> dict[str, Any]:
    """Every array :func:`build_layer` needs, in one pytree."""
    weights: dict[str, Any] = {
        "attn_hc": gated_residual_weights(
            layer.attn_hyper_connection, fused_kernels=fused_kernels
        )[1],
        "mlp_hc": gated_residual_weights(
            layer.mlp_hyper_connection, fused_kernels=fused_kernels
        )[1],
        "moe": moe_weights(layer.mlp)[1],
    }
    if layer.is_linear:
        weights["branch"] = gdn_weights(layer.linear_attn)[1]
    else:
        weights["branch"] = attention_weights(layer.self_attn)[1]
    return weights


def build_layer(
    layer: nn.Module,
    *,
    compiled: bool = True,
    use_kernel: bool = True,
    fused_kernels: bool = False,
) -> CompiledLayer:
    """One ``Qwen4ExpDecoderLayer`` as one compiled step.

    The four blocks are built uncompiled and composed here, so the trace is the
    whole layer rather than four traces with Python between them. That matters
    more than it sounds: the saving from compiling a block is the Python it
    replaces, and the Python *between* blocks is not replaced by compiling the
    blocks.

    A layer carrying a PLE sub-layer is refused rather than silently dropped.
    The n-gram embedding reads rows out of an mmap through numpy, which is a
    side effect in the middle of the graph and cannot be traced; on the
    checkpoint one layer of 48 has one, and that layer stays on the eager path.
    """
    if "ple" in layer:
        raise ValueError(
            "a PLE layer cannot be compiled: Qwen4ExpNGramEmbedding reads rows "
            "from an mmap through numpy inside the forward"
        )

    attn_hc = build_gated_residual(
        layer.attn_hyper_connection, compiled=False, fused_kernels=fused_kernels
    )
    mlp_hc = build_gated_residual(
        layer.mlp_hyper_connection, compiled=False, fused_kernels=fused_kernels
    )
    moe = build_moe_step(layer.mlp, compiled=False)
    inject = hyper_inject_body
    weights = layer_weights(layer, fused_kernels=fused_kernels)

    if layer.is_linear:
        branch = build_gdn_step(
            layer.linear_attn,
            use_kernel=use_kernel,
            compiled=False,
            fused_kernels=fused_kernels,
        )

        def step(hidden, conv_state, ssm_state, weights):
            mixed, hyper_input, injection = attn_hc(hidden, weights["attn_hc"])
            out, conv_state, ssm_state = branch(
                mixed, conv_state, ssm_state, weights["branch"]
            )
            hidden = inject(hyper_input, out, injection)
            mixed, hyper_input, injection = mlp_hc(hidden, weights["mlp_hc"])
            hidden = inject(hyper_input, moe(mixed, weights["moe"]), injection)
            return hidden, conv_state, ssm_state

        kind = LINEAR
    else:
        branch = build_attention_step(layer.self_attn, compiled=False)
        has_indexer = attention_weights(layer.self_attn)[0]["has_indexer"]

        def step(hidden, k_buf, v_buf, index_buf, offset, weights, sparse_mask=None):
            mixed, hyper_input, injection = attn_hc(hidden, weights["attn_hc"])
            out, k_buf, v_buf, new_offset, index_keys = branch(
                mixed, k_buf, v_buf, offset, weights["branch"], sparse_mask
            )
            if has_indexer and index_keys is not None:
                index_buf = mx.slice_update(
                    index_buf,
                    index_keys.astype(index_buf.dtype),
                    mx.reshape(offset, (1,)).astype(mx.int32),
                    axes=(1,),
                )
            hidden = inject(hyper_input, out, injection)
            mixed, hyper_input, injection = mlp_hc(hidden, weights["mlp_hc"])
            hidden = inject(hyper_input, moe(mixed, weights["moe"]), injection)
            return hidden, k_buf, v_buf, index_buf, new_offset

        kind = ATTENTION

    return CompiledLayer(
        kind=kind,
        step=mx.compile(step) if compiled else step,
        weights=weights,
        module=layer,
    )


# ---------------------------------------------------------------------------
# the state contract
# ---------------------------------------------------------------------------


@dataclass
class LayerState:
    """One layer's decode state, as arrays the caller owns.

    Nothing here is mutated by a step. A step takes these arrays and returns
    new ones, which is the whole reason a rollback is a reassignment rather
    than a replay: holding a reference to a ``LayerState`` holds that state,
    whatever later steps do.

    ``linear`` carries ``(conv_state, ssm_state)``; ``attention`` carries
    ``(k_buf, v_buf, index_buf, offset)`` where the three buffers are
    fixed-capacity and ``offset`` is a rank-zero ``int32`` array, never a
    Python integer.
    """

    kind: str
    arrays: tuple[mx.array, ...]

    @property
    def offset(self) -> Optional[mx.array]:
        return self.arrays[3] if self.kind == ATTENTION else None

    @property
    def capacity(self) -> int:
        return self.arrays[0].shape[2] if self.kind == ATTENTION else 0


@dataclass
class DecodeState:
    """The whole model's decode state: one :class:`LayerState` per layer."""

    layers: list[LayerState]
    length: int = 0

    def arrays(self) -> list[tuple[mx.array, ...]]:
        """The pytree the compiled model step takes and returns."""
        return [state.arrays for state in self.layers]

    def replaced(self, arrays: Sequence[Sequence[mx.array]], length: int) -> "DecodeState":
        return DecodeState(
            layers=[
                LayerState(kind=state.kind, arrays=tuple(new))
                for state, new in zip(self.layers, arrays)
            ],
            length=length,
        )

    @property
    def capacity(self) -> int:
        for state in self.layers:
            if state.kind == ATTENTION:
                return state.capacity
        return 0

    def recurrent_arrays(self) -> dict[str, mx.array]:
        """The recurrent halves, keyed the way ``ModelState`` keys a snapshot.

        ``titan.adapters.mlx.state.Snapshot`` stores ``layer{i}.slot{j}``, and
        a compiled state has to round-trip through the same store, so the keys
        are the same and slot 0 and slot 1 mean what ``ArraysCache`` means by
        them: the conv window and the recurrent state.
        """
        out: dict[str, mx.array] = {}
        for index, state in enumerate(self.layers):
            if state.kind != LINEAR:
                continue
            for slot, value in enumerate(state.arrays):
                out[f"layer{index}.slot{slot}"] = value
        return out


def new_state(
    model: nn.Module,
    *,
    batch: int = 1,
    capacity: int = 256,
    dtype=mx.bfloat16,
    eager_indices: Sequence[int] = (),
) -> DecodeState:
    """An empty :class:`DecodeState` for *model* with the KV buffers allocated.

    ``eager_indices`` names layers that run as eager islands and therefore hold
    a vendored cache instead of arrays in this state. A PLE layer is always
    one; generation 2 adds the sparse-attention layers when the context is past
    the indexer budget. See :func:`island_layers`.
    """
    inner = _language_model(model).model
    skip = set(eager_indices)
    layers: list[LayerState] = []
    for index, layer in enumerate(inner.layers):
        if index in skip or "ple" in layer:
            layers.append(LayerState(kind=EAGER, arrays=()))
            continue
        if layer.is_linear:
            gdn = layer.linear_attn
            layers.append(
                LayerState(
                    kind=LINEAR,
                    arrays=(
                        mx.zeros(
                            (batch, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=dtype
                        ),
                        mx.zeros(
                            (batch, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
                            dtype=mx.float32,
                        ),
                    ),
                )
            )
        else:
            attn = layer.self_attn
            indexer = getattr(attn, "indexer", None)
            index_dim = indexer.head_dim if indexer is not None else 1
            layers.append(
                LayerState(
                    kind=ATTENTION,
                    arrays=(
                        mx.zeros(
                            (batch, attn.num_key_value_heads, capacity, attn.head_dim),
                            dtype=dtype,
                        ),
                        mx.zeros(
                            (batch, attn.num_key_value_heads, capacity, attn.head_dim),
                            dtype=dtype,
                        ),
                        mx.zeros((batch, capacity, index_dim), dtype=dtype),
                        mx.array(0, mx.int32),
                    ),
                )
            )
    return DecodeState(layers=layers, length=0)


def grow_state(state: DecodeState, tokens: int, *, step: int = 256) -> DecodeState:
    """Grow every attention buffer to hold *tokens*, outside the compiled region.

    Growth is a shape change and a shape change is a retrace, which is exactly
    why it is here and not in the step. :func:`capacity_for` picks the growth
    points, so the number of traces is logarithmic in the context rather than
    linear in it.
    """
    needed = capacity_for(tokens, step)
    if needed <= state.capacity:
        return state
    layers = []
    for entry in state.layers:
        if entry.kind != ATTENTION:
            layers.append(entry)
            continue
        k_buf, v_buf, index_buf, offset = entry.arrays
        layers.append(
            LayerState(
                kind=ATTENTION,
                arrays=(
                    grow_kv(k_buf, needed, k_buf),
                    grow_kv(v_buf, needed, v_buf),
                    _grow_index(index_buf, needed),
                    offset,
                ),
            )
        )
    return DecodeState(layers=layers, length=state.length)


def _grow_index(buffer: mx.array, capacity: int) -> mx.array:
    batch, held, dim = buffer.shape
    grown = mx.zeros((batch, capacity, dim), dtype=buffer.dtype)
    if held:
        keep = min(held, capacity)
        grown = mx.slice_update(
            grown, buffer[:, :keep, :], mx.array([0], mx.int32), axes=(1,)
        )
    return grown


def truncate_state(
    state: DecodeState,
    length: int,
    snapshot: Optional[dict[str, mx.array]] = None,
) -> DecodeState:
    """Roll back to exactly *length* tokens.

    The attention half is free: the offset is an array the caller owns, so
    moving it back is an assignment and the rows past it are masked out of the
    next step's softmax by the same comparison that made them causal. Nothing
    is zeroed and nothing is copied.

    The recurrent half cannot be rewound, exactly as
    ``ModelState.truncate`` says, so it takes a *snapshot* -- the dict
    :meth:`DecodeState.recurrent_arrays` produced at *length*. Without one this
    raises, rather than silently continuing from a state that is a few tokens
    ahead of the KV cache it is paired with.
    """
    if length > state.length:
        raise ValueError(f"cannot grow a state from {state.length} to {length}")
    if length == state.length:
        return state
    layers: list[LayerState] = []
    for index, entry in enumerate(state.layers):
        if entry.kind == EAGER:
            # The layer that stays eager holds its own vendored cache and is
            # truncated by the vendored path, the same way it always was. The
            # placeholder carries no arrays, so there is nothing to rewind
            # here and nothing to demand a snapshot for.
            layers.append(entry)
            continue
        if entry.kind == ATTENTION:
            k_buf, v_buf, index_buf, _offset = entry.arrays
            layers.append(
                LayerState(
                    kind=ATTENTION,
                    arrays=(k_buf, v_buf, index_buf, mx.array(length, mx.int32)),
                )
            )
            continue
        if snapshot is None:
            raise ValueError(
                f"no recurrent snapshot for truncation to {length}: the conv "
                "window and the recurrent state cannot be rewound"
            )
        restored = []
        for slot in range(len(entry.arrays)):
            key = f"layer{index}.slot{slot}"
            if key not in snapshot:
                raise ValueError(f"snapshot is missing {key}")
            restored.append(snapshot[key])
        layers.append(LayerState(kind=LINEAR, arrays=tuple(restored)))
    return DecodeState(layers=layers, length=length)


def rollback_speculative_state(
    state: DecodeState,
    accepted: int,
    block_size: int,
    snapshot: Optional[dict[str, mx.array]] = None,
) -> DecodeState:
    """The compiled equivalent of ``LanguageModel.rollback_speculative_cache``.

    A verify block of *block_size* rows was committed and *accepted* of the
    drafted tokens survived, so the state has to come back to the length it had
    before the block plus ``accepted + 1``. The vendored version trims each
    cache in place and replays the recurrent state out of the captured
    intermediates; here the trim is an offset assignment and the recurrent
    state is the snapshot staged before the block.
    """
    keep = int(accepted) + 1
    if keep > block_size:
        raise ValueError(f"accepted {accepted} is outside a block of {block_size}")
    return truncate_state(state, state.length - (block_size - keep), snapshot)


def read_layer_state(
    caches: Sequence[Any],
    *,
    capacity: Optional[int] = None,
    step: int = 256,
    eager_indices: Sequence[int] = (),
) -> DecodeState:
    """Build a :class:`DecodeState` from the vendored caches.

    The bridge in. ``ArraysCache`` slots become the recurrent pair as they
    stand; a ``KVCache``'s keys and values are copied into a fixed-capacity
    buffer and its integer offset becomes a rank-zero array. Called once when a
    sequence moves from the eager prefill onto the compiled decode.

    ``eager_indices`` names the layers that stay on the eager path, which is
    :func:`untraceable_layers` for a model with a PLE layer in it. Those get an
    :data:`EAGER` placeholder here and keep their vendored cache, because their
    state is state a trace cannot hold -- the PLE layer's short-conv window
    lives in slot 2 of the same cache, which this function does not carry.
    """
    skip = set(eager_indices)
    from .vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache

    length = 0
    for cache in caches:
        if not isinstance(cache, ArraysCache):
            length = max(length, int(cache.offset))
    target = capacity_for(length, step) if capacity is None else capacity

    layers: list[LayerState] = []
    for index, cache in enumerate(caches):
        if index in skip:
            layers.append(LayerState(kind=EAGER, arrays=()))
            continue
        if isinstance(cache, ArraysCache):
            slots = list(cache.state or ())
            layers.append(LayerState(kind=LINEAR, arrays=tuple(slots[:2])))
            continue
        offset = int(cache.offset)
        keys = cache.keys[:, :, :offset, :]
        values = cache.values[:, :, :offset, :]
        k_buf = mx.slice_update(
            grow_kv(None, target, keys), keys, mx.array([0], mx.int32), axes=(2,)
        )
        v_buf = mx.slice_update(
            grow_kv(None, target, values), values, mx.array([0], mx.int32), axes=(2,)
        )
        index_keys = getattr(cache, "index_keys", None)
        if index_keys is None:
            index_buf = mx.zeros((keys.shape[0], target, 1), dtype=keys.dtype)
        else:
            index_buf = _grow_index(index_keys[:, :offset, :], target)
        layers.append(
            LayerState(
                kind=ATTENTION,
                arrays=(k_buf, v_buf, index_buf, mx.array(offset, mx.int32)),
            )
        )
    return DecodeState(layers=layers, length=length)


def write_layer_state(state: DecodeState, caches: Sequence[Any]) -> None:
    """Push a :class:`DecodeState` back into the vendored caches.

    The bridge out, and the reason the compiled path is rollback-compatible
    rather than a parallel universe: everything downstream of the model --
    ``ModelState.truncate``, ``stage_snapshot``, the prefix-cache codec,
    ``rollback_speculative_cache`` -- keeps working on the caches it already
    knows, because after this call they hold exactly what the compiled step
    computed.
    """
    from .vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache

    for cache, entry in zip(caches, state.layers):
        if entry.kind == EAGER:
            continue
        if entry.kind == LINEAR:
            if not isinstance(cache, ArraysCache):
                raise ValueError("linear layer state written to a non-recurrent cache")
            slots = list(cache.state or ())
            for slot, value in enumerate(entry.arrays):
                if slot < len(slots):
                    slots[slot] = value
            cache.state = slots
            continue
        k_buf, v_buf, index_buf, offset = entry.arrays
        held = int(offset)
        cache.keys = k_buf
        cache.values = v_buf
        cache.offset = held
        if getattr(cache, "index_keys", None) is not None or hasattr(
            cache, "update_indexer"
        ):
            cache.index_keys = index_buf[:, :held, :]
            cache.index_position_ids = mx.broadcast_to(
                mx.arange(held, dtype=mx.int32)[None, :], (index_buf.shape[0], held)
            )


# ---------------------------------------------------------------------------
# the whole-model decode step
# ---------------------------------------------------------------------------


def _language_model(model: nn.Module):
    return getattr(model, "language_model", model)


@dataclass
class CompiledModel:
    """Every layer chained into one traced decode step.

    ``step(hidden, state, sparse_masks) -> (logits, hidden, state)``, with the
    embedding outside. The embedding is the one part of the forward that is not
    arithmetic: on the checkpoint it is a ``DiskBackedShardedEmbedding`` that
    reads rows out of an mmap through numpy, which is a side effect no trace
    can hold. It is one gather per step, so leaving it out costs one op and
    keeps the compiled region honest about what it can promise.
    """

    step: Callable[..., Any]
    weights: dict[str, Any]
    layers: list[CompiledLayer]
    hc_count: int
    model: Any = None
    #: The QSA indexer budget of the sparse layers, or 0 when the model has
    #: none. Past it the eager forward selects key blocks and a single graph
    #: does not: see :meth:`_refuse_above_the_budget`.
    sparse_budget: int = 0
    #: ``[(kind, payload)]`` in layer order when the model has a layer that
    #: cannot be traced; empty when the whole model is one graph. ``kind`` is
    #: ``"traced"`` with ``(start, stop, step)`` or ``"eager"`` with the layer
    #: index. See :func:`build_model`.
    plan: list[tuple[str, Any]] = field(default_factory=list)
    #: Vendored caches for the eager layers, by layer index. Those layers own
    #: their own state, because it is state a trace cannot hold.
    eager_caches: dict[int, Any] = field(default_factory=dict)
    tail: Optional[Callable[..., Any]] = None
    #: Layer indices that run as eager islands. Empty when the whole model is
    #: one graph. Generation 2 puts the sparse-attention layers in here as well
    #: as the PLE layers; see :func:`island_layers`.
    island_indices: tuple[int, ...] = ()
    #: True when the layer traces call the eager path's fused Metal kernels.
    fused_kernels: bool = False

    def embed(self, tokens: mx.array) -> mx.array:
        inner = _language_model(self.model).model
        return mx.tile(inner.embed_tokens(tokens), (1, 1, self.hc_count))

    def _refuse_above_the_budget(self, length: int) -> None:
        """Refuse a length where a single graph would be a different answer.

        COMPILED.md section 2: below the indexer budget the eager attention is
        dense and the compiled step reproduces it exactly. Above it the eager
        path *selects* key blocks, using this layer's mixed hyper-connection
        output, which is computed inside the layer graph -- so a sparse layer
        cannot be one graph, and :func:`build_layer_split` is the cut.

        A caller that forgets gets a plausible answer that is wrong, which
        section 7 calls the worst failure mode in that document and says the
        builder should refuse rather than trust the caller. This is that
        refusal. It is at call time rather than at build time because the
        budget is a property of the length, and the length is not known until
        a step runs.
        """
        if (
            self.sparse_budget
            and length > self.sparse_budget
            and not self._sparse_layers_are_islands()
        ):
            raise ValueError(
                f"context {length} is past the QSA indexer budget "
                f"{self.sparse_budget}, where the eager attention selects key "
                "blocks and a single traced layer does not. Use "
                "build_layer_split and pass the selection in; a single graph "
                "here attends to keys the eager path drops and returns a "
                "plausible wrong answer. See COMPILED.md sections 2 and 7."
            )

    def _sparse_layers_are_islands(self) -> bool:
        """Whether every layer with a QSA indexer runs eagerly.

        That is generation 2's answer to the refusal above: the layer that
        would need the selection is not in a trace at all, so a length past the
        budget is not a wrong answer, it is one more eager dispatch. The
        refusal stands for a model built without those islands, which is every
        generation 1 build.
        """
        inner = _language_model(self.model).model
        islands = set(self.island_indices)
        return all(
            index in islands
            for index, layer in enumerate(inner.layers)
            if getattr(getattr(layer, "self_attn", None), "indexer", None) is not None
        )

    def __call__(self, tokens: mx.array, state: DecodeState):
        """Embed, run the compiled step, and return ``(logits, hidden, state)``."""
        hidden = self.embed(tokens)
        width = tokens.shape[1]
        self._refuse_above_the_budget(state.length + width)
        state = grow_state(state, state.length + width)
        if not self.plan:
            logits, mixed, arrays = self.step(hidden, state.arrays(), self.weights)
            return logits, mixed, state.replaced(arrays, state.length + width)
        return self._segmented(tokens, hidden, state, width)

    def _segmented(self, tokens, hidden, state, width):
        """The mixed path: traced runs with an untraceable layer between them.

        The layer that cannot be traced runs eagerly, on its own vendored
        cache, exactly as it would in the ordinary forward. Everything either
        side of it is still one graph per run, so what the split costs is one
        extra dispatch per untraceable layer rather than the whole saving.
        """
        inner = _language_model(self.model).model
        arrays = state.arrays()
        out: list[tuple[mx.array, ...]] = [() for _ in state.layers]
        for kind, payload in self.plan:
            if kind == "eager":
                index = payload
                layer = inner.layers[index]
                cache = self.eager_caches.get(index)
                # The same mask ``Qwen4ExpModel.__call__`` builds for this
                # layer's type. At width 1 both helpers return ``None``; at a
                # verify width the block's own causality is in here, so
                # passing ``None`` would be quietly wrong rather than slow.
                from .vendor.mlx_vlm.models.qwen4_exp.language import (
                    _create_qwen3_5_attention_mask,
                    _create_qwen3_5_ssm_mask,
                )

                build_mask = (
                    _create_qwen3_5_ssm_mask
                    if layer.is_linear
                    else _create_qwen3_5_attention_mask
                )
                hidden = layer(hidden, tokens, build_mask(hidden, cache), cache, None)
                continue
            start, stop, step = payload
            hidden, produced = step(
                hidden,
                arrays[start:stop],
                self.weights["layers"][start:stop],
            )
            out[start:stop] = list(produced)
        logits, mixed = self.tail(hidden, self.weights)
        return logits, mixed, state.replaced(out, state.length + width)


def untraceable_layers(model: nn.Module) -> list[int]:
    """Indices of the layers :func:`build_layer` cannot take.

    Today that is exactly the PLE layers: ``Qwen4ExpNGramEmbedding`` reads rows
    out of a 32 GB packed table on SSD through an mmap and numpy, in the middle
    of the forward, and no trace can hold a side effect. One layer of 48 on the
    checkpoint has one; the synthetic model has none, which is why the whole
    compiled model was a single graph until ROUND4 pointed it at the
    checkpoint and found that section 6's command had never been runnable.
    """
    inner = _language_model(model).model
    return [index for index, layer in enumerate(inner.layers) if "ple" in layer]


def build_model(
    model: nn.Module,
    *,
    compiled: bool = True,
    use_kernel: bool = True,
    fused_kernels: bool = False,
    eager_indices: Optional[Sequence[int]] = None,
) -> CompiledModel:
    """Chain every layer, the final mixer and the head into one graph.

    A model with an untraceable layer in it cannot be one graph, so it becomes
    a *plan*: traced runs of layers with the untraceable ones eager between
    them. The checkpoint has one such layer, so it is two traced runs and one
    eager layer rather than one graph, and what the split costs is one extra
    dispatch rather than the saving. A model with none -- every synthetic model
    here -- takes exactly the single-graph path it always did.

    ``eager_indices`` overrides which layers become islands. The default is
    :func:`untraceable_layers`, which is the PLE layers; generation 2 passes
    :func:`island_layers` instead, which adds the sparse-attention layers when
    the context is past the indexer budget.
    """
    language = _language_model(model)
    inner = language.model
    eager_ids = (
        untraceable_layers(model) if eager_indices is None else sorted(eager_indices)
    )
    layers = [
        None
        if index in eager_ids
        else build_layer(
            layer, compiled=False, use_kernel=use_kernel, fused_kernels=fused_kernels
        )
        for index, layer in enumerate(inner.layers)
    ]
    mixer = build_gated_residual(
        inner.hyper_connection_mixer, compiled=False, fused_kernels=fused_kernels
    )
    head_spec, head_arrays = split_linear(language.lm_head)
    kinds = [None if layer is None else layer.kind for layer in layers]
    steps = [None if layer is None else layer.step for layer in layers]

    weights = {
        "layers": [{} if layer is None else layer.weights for layer in layers],
        "mixer": gated_residual_weights(
            inner.hyper_connection_mixer, fused_kernels=fused_kernels
        )[1],
        "head": head_arrays,
    }

    def run(index, hidden, state_tuple, layer_weights, sparse_mask=None):
        """One traced layer, dispatched on its kind. Shared by both paths."""
        if kinds[index] == LINEAR:
            conv_state, ssm_state = state_tuple
            hidden, conv_state, ssm_state = steps[index](
                hidden, conv_state, ssm_state, layer_weights
            )
            return hidden, (conv_state, ssm_state)
        k_buf, v_buf, index_buf, offset = state_tuple
        hidden, k_buf, v_buf, index_buf, offset = steps[index](
            hidden, k_buf, v_buf, index_buf, offset, layer_weights, sparse_mask
        )
        return hidden, (k_buf, v_buf, index_buf, offset)

    if eager_ids:
        return _planned_model(
            model,
            inner,
            layers,
            weights,
            mixer,
            head_spec,
            run,
            eager_ids,
            compiled,
            fused_kernels,
        )

    def step(hidden, states, weights, sparse_masks=None):
        out_states = []
        for index, (kind, layer_step) in enumerate(zip(kinds, steps)):
            layer_weights = weights["layers"][index]
            if kind == LINEAR:
                conv_state, ssm_state = states[index]
                hidden, conv_state, ssm_state = layer_step(
                    hidden, conv_state, ssm_state, layer_weights
                )
                out_states.append((conv_state, ssm_state))
            else:
                k_buf, v_buf, index_buf, offset = states[index]
                mask = None if sparse_masks is None else sparse_masks[index]
                hidden, k_buf, v_buf, index_buf, offset = layer_step(
                    hidden, k_buf, v_buf, index_buf, offset, layer_weights, mask
                )
                out_states.append((k_buf, v_buf, index_buf, offset))
        mixed = mixer(hidden, weights["mixer"])
        logits = call_linear(head_spec, weights["head"], mixed)
        return logits, mixed, out_states

    return CompiledModel(
        step=mx.compile(step) if compiled else step,
        weights=weights,
        layers=layers,
        hc_count=int(inner.args.hc_count),
        model=model,
        sparse_budget=_sparse_budget(inner),
        fused_kernels=fused_kernels,
    )


def _sparse_budget(inner) -> int:
    """The QSA indexer budget shared by this model's sparse layers, or 0."""
    for layer in inner.layers:
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if indexer is not None:
            return int(indexer.token_budget)
    return 0


def _planned_model(
    model,
    inner,
    layers,
    weights,
    mixer,
    head_spec,
    run,
    eager_ids,
    compiled,
    fused_kernels=False,
):
    """``build_model`` for a model with an untraceable layer in it."""
    count = len(inner.layers)
    plan: list[tuple[str, Any]] = []
    start = 0
    for index in [*eager_ids, count]:
        if index > start:
            plan.append(("traced", (start, index, None)))
        if index < count:
            plan.append(("eager", index))
        start = index + 1

    def make_segment(begin: int, end: int):
        def segment(hidden, states, seg_weights, sparse_masks=None):
            produced = []
            for offset in range(end - begin):
                mask = None if sparse_masks is None else sparse_masks[offset]
                hidden, out = run(
                    begin + offset, hidden, states[offset], seg_weights[offset], mask
                )
                produced.append(out)
            return hidden, produced

        return mx.compile(segment) if compiled else segment

    plan = [
        (kind, (payload[0], payload[1], make_segment(payload[0], payload[1])))
        if kind == "traced"
        else (kind, payload)
        for kind, payload in plan
    ]

    def tail(hidden, weights):
        mixed = mixer(hidden, weights["mixer"])
        return call_linear(head_spec, weights["head"], mixed), mixed

    return CompiledModel(
        step=None,
        weights=weights,
        layers=[layer for layer in layers if layer is not None],
        hc_count=int(inner.args.hc_count),
        model=model,
        plan=plan,
        eager_caches={},
        tail=mx.compile(tail) if compiled else tail,
        sparse_budget=_sparse_budget(inner),
        island_indices=tuple(eager_ids),
        fused_kernels=fused_kernels,
    )


def eval_weights(weights: Any) -> None:
    """Evaluate every array in a weight pytree. Call once, before the first step."""
    flat: list[mx.array] = []

    def walk(value: Any) -> None:
        if isinstance(value, mx.array):
            flat.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(weights)
    if flat:
        mx.eval(flat)


# ---------------------------------------------------------------------------
# the QSA seam: a sparse layer above the indexer budget
# ---------------------------------------------------------------------------


def pad_sparse_mask(mask: mx.array, capacity: int) -> mx.array:
    """Widen a ``[B, 1, W, key_len]`` selection out to the buffer capacity.

    The compiled attention step attends over the whole capacity and lets the
    causal comparison switch off the columns past the write, so a selection
    computed against the live length has to be padded with ``False`` rather
    than left short.
    """
    held = mask.shape[-1]
    if held == capacity:
        return mask
    if held > capacity:
        raise ValueError(f"selection covers {held} columns, capacity is {capacity}")
    pad = mx.zeros((*mask.shape[:-1], capacity - held), dtype=mx.bool_)
    return mx.concatenate([mask, pad], axis=-1)


def build_layer_split(
    layer: nn.Module, *, compiled: bool = True
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """An attention layer as two compiled steps with the QSA seam between them.

    Below the indexer's budget the eager attention is dense and
    :func:`build_layer` covers it in one graph. Above the budget the eager path
    selects key blocks, and the selection needs *this layer's* mixed
    hyper-connection output -- which is computed inside that single graph. So a
    caller that wants the sparse arm needs the graph cut where the selection
    happens, and this is that cut:

        mixed, hyper_input, injection = mix_step(hidden, weights)
        sparse = pad_sparse_mask(layer.self_attn.indexer(mixed, cache, None), capacity)
        hidden, k, v, index, offset = rest_step(
            mixed, hyper_input, injection, k, v, index, offset, weights, sparse
        )

    The selection itself stays on the eager path. It reads ``int(cache.offset)``
    and pools the indexer keys through a Python-level cache with its own
    invalidation, so it is the one part of the sparse forward that is host work
    by construction rather than by formulation. Compiling it needs the indexer
    cache reformulated the way the KV cache is here -- a capacity-shaped key
    buffer and an array offset -- which is a separate change.
    """
    if layer.is_linear:
        raise ValueError("build_layer_split is for attention layers")

    mix = build_gated_residual(layer.attn_hyper_connection, compiled=False)
    mlp_hc = build_gated_residual(layer.mlp_hyper_connection, compiled=False)
    moe = build_moe_step(layer.mlp, compiled=False)
    branch = build_attention_step(layer.self_attn, compiled=False)
    inject = hyper_inject_body
    has_indexer = attention_weights(layer.self_attn)[0]["has_indexer"]

    def mix_step(hidden, weights):
        return mix(hidden, weights["attn_hc"])

    def rest_step(
        mixed, hyper_input, injection, k_buf, v_buf, index_buf, offset, weights,
        sparse_mask=None,
    ):
        out, k_buf, v_buf, new_offset, index_keys = branch(
            mixed, k_buf, v_buf, offset, weights["branch"], sparse_mask
        )
        if has_indexer and index_keys is not None:
            index_buf = mx.slice_update(
                index_buf,
                index_keys.astype(index_buf.dtype),
                mx.reshape(offset, (1,)).astype(mx.int32),
                axes=(1,),
            )
        hidden = inject(hyper_input, out, injection)
        mixed, hyper_input, injection = mlp_hc(hidden, weights["mlp_hc"])
        hidden = inject(hyper_input, moe(mixed, weights["moe"]), injection)
        return hidden, k_buf, v_buf, index_buf, new_offset

    if not compiled:
        return mix_step, rest_step
    return mx.compile(mix_step, shapeless=True), mx.compile(rest_step)


# ---------------------------------------------------------------------------
# generation 2: the same kernels the eager path runs, inside the trace
# ---------------------------------------------------------------------------
#
# Generation 1 asked whether the decode forward could be traced. It could, and
# on the checkpoint that removed 81 to 87% of the host build and added 24 to
# 71% to the step, because the traced layer computed out of plain MLX ops what
# the eager layer next to it computed with fused Metal kernels: the
# hyper-connection block, ninety-six times a step, and the Gated DeltaNet's
# gated output norm.
#
# Generation 2 asks the narrower question that follows: can a trace hold the
# kernels? At fixed shapes it can. ``mx.compile`` treats an
# ``mx.fast.metal_kernel`` as one opaque node, replays it from the C++ side
# like any other primitive, and produces bit-identical results. Shapeless it
# cannot, and that is not a bug to route around -- the output shape of a custom
# kernel is a declaration, not an inference -- so per-shape tracing is the
# price of the kernels rather than a compromise. What that costs is one trace
# per (verify width, KV capacity) and :func:`warm_traces` pays it at startup.
#
# Three pieces here. :func:`island_layers` and :func:`plan_layout` say which
# layers cannot be in a trace and how the rest group into segments.
# :func:`build_gen2` is the builder. :func:`warm_traces` walks the grid inside
# a budget and reports what it cost.


def island_layers(model: nn.Module, *, length: Optional[int] = None) -> list[int]:
    """Layer indices that run eagerly between the compiled segments.

    Two kinds, and they are the only two:

    *PLE layers.* ``Qwen4ExpNGramEmbedding`` reads rows out of a 32 GB packed
    table on SSD, through an mmap and numpy, in the middle of the forward. A
    trace cannot hold a disk read. One layer of 48 on the checkpoint.

    *Sparse-attention layers past the indexer budget.* Above the budget the
    eager attention selects key blocks, and the selection needs this layer's
    own mixed hyper-connection output plus ``int(cache.offset)`` and a pooled
    key cache with Python-level invalidation. COMPILED.md section 2 has the
    argument; the short form is that a dense trace there attends to keys the
    eager path drops and returns a plausible wrong answer. Generation 1
    refused such a length outright. Generation 2 runs those layers eagerly,
    which is what makes 64k reachable at all.

    ``length`` is the context the plan is for. ``None`` means "any length",
    which puts every indexed attention layer on the eager side; a length at or
    under the budget leaves them in the traces, where they are dense and exact.
    """
    inner = _language_model(model).model
    budget = _sparse_budget(inner)
    islands = []
    for index, layer in enumerate(inner.layers):
        if "ple" in layer:
            islands.append(index)
            continue
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if indexer is None:
            continue
        if length is None or (budget and length > budget):
            islands.append(index)
    return islands


@dataclass(frozen=True)
class LayoutPlan:
    """How a layer stack splits into traced segments and eager islands.

    Computed from the config alone, so the checkpoint's numbers do not need
    the checkpoint loaded -- which matters, because loading it is 70 GB and one
    process at a time.
    """

    layers: int
    islands: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]
    #: Whether any traced segment contains an attention layer. When none does,
    #: KV capacity is not part of any trace key and the grid collapses to the
    #: widths alone, which is the case at every context past the budget.
    capacity_in_trace_key: bool

    @property
    def island_count(self) -> int:
        return len(self.islands)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def traces(self, widths: Sequence[int], capacities: Sequence[int]) -> int:
        """Distinct traces this layout needs over a (width, capacity) grid."""
        buckets = len(capacities) if self.capacity_in_trace_key else 1
        return self.segment_count * len(widths) * buckets


def plan_layout(
    *,
    layer_types: Sequence[str],
    ple_layer_ids: Sequence[int] = (),
    length: Optional[int] = None,
    indexer_budget: int = 0,
) -> LayoutPlan:
    """:func:`island_layers` and the segmentation, from a config dict's fields.

    ``layer_types`` is the checkpoint config's list of ``"linear_attention"``
    and ``"full_attention"``; ``ple_layer_ids`` is its PLE list. Nothing here
    touches a model, so ``bench/decode/compiled_path.py layout --config ...``
    can answer "how many traces and how many islands" for the checkpoint while
    another process owns the GPU.
    """
    count = len(layer_types)
    ple = set(int(i) for i in ple_layer_ids)
    sparse_islanded = length is None or (
        bool(indexer_budget) and length > indexer_budget
    )
    islands = sorted(
        index
        for index in range(count)
        if index in ple
        or (sparse_islanded and layer_types[index] == "full_attention")
    )
    segments: list[tuple[int, int]] = []
    start = 0
    for index in [*islands, count]:
        if index > start:
            segments.append((start, index))
        start = index + 1
    capacity_in_key = any(
        layer_types[i] == "full_attention"
        for begin, end in segments
        for i in range(begin, end)
    )
    return LayoutPlan(
        layers=count,
        islands=tuple(islands),
        segments=tuple(segments),
        capacity_in_trace_key=capacity_in_key,
    )


def build_gen2(
    model: nn.Module,
    *,
    length: Optional[int] = None,
    compiled: bool = True,
    use_kernel: bool = True,
    fused_kernels: bool = True,
) -> CompiledModel:
    """The generation 2 build: fused kernels in the traces, islands around them.

    ``length`` is the context this build is for, and it decides whether the
    sparse-attention layers are inside the traces (at or under the indexer
    budget, where they are dense) or islands beside them (past it). A build is
    valid for any length on its own side of the budget; crossing the budget
    needs a rebuild, which is one more thing the engine has to own and is
    stated plainly in COMPILED.md rather than hidden behind a rebuild-on-demand.
    """
    return build_model(
        model,
        compiled=compiled,
        use_kernel=use_kernel,
        fused_kernels=fused_kernels,
        eager_indices=island_layers(model, length=length),
    )


# ---------------------------------------------------------------------------
# the trace cache
# ---------------------------------------------------------------------------


def capacity_buckets(*, up_to: int, step: int = 256) -> list[int]:
    """Every distinct capacity :func:`capacity_for` produces up to *up_to*.

    The trace grid's other axis. The schedule steps by 256 to 2048 and doubles
    from 4096 up, so a 262k context is fifteen buckets rather than a thousand,
    and the waste stays under 2x of the arithmetic on masked columns.
    """
    buckets: list[int] = []
    tokens = 1
    while tokens <= up_to:
        value = capacity_for(tokens, step)
        if not buckets or value != buckets[-1]:
            buckets.append(value)
        if value >= up_to:
            break
        tokens = value + 1
    return buckets


def _capacity_in_trace_key(compiled_model: CompiledModel) -> bool:
    """Whether any traced segment of *compiled_model* holds an attention layer."""
    inner = _language_model(compiled_model.model).model
    islands = set(compiled_model.island_indices)
    return any(
        index not in islands and not layer.is_linear and "ple" not in layer
        for index, layer in enumerate(inner.layers)
    )


@dataclass
class WarmReport:
    """What warming the grid cost: one row per (width, capacity) point."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    #: Points the budget stopped before reaching.
    skipped: list[tuple[int, int]] = field(default_factory=list)
    seconds: float = 0.0
    #: MLX active memory before and after, in bytes. The compile cache never
    #: evicts, so the difference is the standing cost of the warm grid plus the
    #: throwaway state the warm-up allocated and dropped.
    active_before: int = 0
    active_after: int = 0
    peak: int = 0

    @property
    def warmed(self) -> int:
        return len(self.rows)

    def first_call_ms(self) -> float:
        return sum(row["first_ms"] for row in self.rows)

    def summary(self) -> dict[str, Any]:
        cached = [row["second_ms"] for row in self.rows]
        return {
            "warmed": self.warmed,
            "skipped": len(self.skipped),
            "seconds": self.seconds,
            "first_call_total_ms": self.first_call_ms(),
            "cached_call_median_ms": _median_of(cached),
            "active_growth_mb": (self.active_after - self.active_before) / 2**20,
            "peak_mb": self.peak / 2**20,
        }


def _median_of(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def warm_traces(
    compiled_model: CompiledModel,
    *,
    widths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    capacities: Sequence[int] = (4096,),
    budget_seconds: float = 30.0,
    budget_mb: float = 2048.0,
    vocab: Optional[int] = None,
) -> WarmReport:
    """Take every trace on the (width, capacity) grid, inside a budget.

    A first call at a new shape pays the trace; every later call at that shape
    pays a dispatch. Paying them all at startup is the difference between a
    server whose first token after a width change costs an extra 8 ms per
    segment and one whose steady state is the steady state from the first
    token.

    The grid is walked width-major inside each capacity, and the loop stops on
    the first budget breach rather than part way through a point, so a report
    with ``skipped`` in it is a report of a grid that was cut, not corrupted.
    Nothing is evicted -- MLX's compile cache has no eviction and neither does
    this -- so ``active_growth_mb`` is the standing cost and the caller decides
    whether the grid is worth it.

    Warming runs on throwaway state: a fresh :class:`DecodeState` per capacity
    and a fresh vendored cache for every island, both dropped afterwards. The
    caller's own state is not touched, and the islands' caches are put back.
    """
    import time

    model = compiled_model.model
    language = _language_model(model)
    if vocab is None:
        vocab = int(language.args.vocab_size)
    islands = list(compiled_model.island_indices)
    held_caches = dict(compiled_model.eager_caches)

    # When every attention layer is an island, no traced segment holds a KV
    # buffer, so capacity is not part of any trace key and the grid collapses
    # to the widths. That is the case at every context past the indexer budget,
    # which is where a long-context server lives, and it is the reason the
    # trace count does not grow with the context there.
    if not _capacity_in_trace_key(compiled_model):
        capacities = capacities[:1]

    report = WarmReport(active_before=int(mx.get_active_memory()))
    started = time.perf_counter()
    stop = False
    for capacity in capacities:
        if stop:
            report.skipped.extend((int(w), int(capacity)) for w in widths)
            continue
        for width in widths:
            elapsed = time.perf_counter() - started
            grown = (int(mx.get_active_memory()) - report.active_before) / 2**20
            if elapsed > budget_seconds or grown > budget_mb:
                stop = True
                report.skipped.append((int(width), int(capacity)))
                continue
            if islands:
                fresh = language.make_cache()
                compiled_model.eager_caches = {i: fresh[i] for i in islands}
            state = new_state(
                model, capacity=capacity, eager_indices=islands
            )
            state = DecodeState(layers=state.layers, length=0)
            mx.eval([a for entry in state.layers for a in entry.arrays])
            tokens = mx.array(
                [[(3 + i) % max(2, vocab - 1) + 1 for i in range(width)]], mx.int64
            )
            first = _one_warm_call(compiled_model, tokens, state)
            second = _one_warm_call(compiled_model, tokens, state)
            report.rows.append(
                {
                    "width": int(width),
                    "capacity": int(capacity),
                    "first_ms": first,
                    "second_ms": second,
                }
            )
    report.seconds = time.perf_counter() - started
    report.active_after = int(mx.get_active_memory())
    report.peak = int(mx.get_peak_memory())
    compiled_model.eager_caches = held_caches
    return report


def _one_warm_call(compiled_model: CompiledModel, tokens: mx.array, state: DecodeState):
    """One end-to-end call, timed, with exactly one synchronize on each side.

    ``docs/ops/INCIDENTS.md`` rule 1: no per-scope evaluation and no per-scope
    ``mx.synchronize()``. The whole step is built, evaluated once, synchronised
    once.
    """
    import time

    mx.synchronize()
    started = time.perf_counter()
    logits, hidden, _next = compiled_model(tokens, state)
    mx.eval(logits, hidden)
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0
