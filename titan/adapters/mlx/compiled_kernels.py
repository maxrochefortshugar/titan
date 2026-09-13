"""The eager path's fused Metal kernels, written as traceable pure functions.

Generation 1 of the compiled decode path (``compiled.py``, and section 3 of
``docs/architecture/COMPILED.md``) cut host build time by 81 to 87% on the
checkpoint and made the step *slower*: +71% at width 1, +24% at width 4. The
reason was not the tracing. It was that a compiled layer computed the
hyper-connection block out of plain MLX ops while the eager layer next to it
called ``hc_fused``, three fused Metal kernels, ninety-six times a step. The
compiled arm won the host and lost the GPU by more.

This module is the fix. Every fused kernel the eager decode path uses is
re-expressed here as a function of arrays alone -- no module reads, no
``mx.eval``, no ``try/except`` around a lazy compile, no Python-level cache
keyed on a shape -- so it can sit *inside* a traced step as an opaque node.
Two things make that possible and one thing makes it necessary:

* ``mx.compile`` at fixed shapes accepts ``mx.fast.metal_kernel``. The custom
  kernel becomes one node in the traced graph and MLX replays it from the C++
  side like any other primitive.
* ``mx.compile(..., shapeless=True)`` does not, and cannot: the shape of a
  custom kernel's output is whatever the caller declared in ``output_shapes``,
  and MLX has no way to infer it from shapeless inputs. It raises
  ``ValueError: [Primitive::output_shapes] CustomKernel cannot infer output
  shapes``. ``tests/model/test_compiled_kernels.py`` pins that message.
* So per-shape tracing is not a compromise forced on generation 2, it is the
  thing that buys the kernels back. One trace per (verify width, KV capacity)
  is the price and section "generation 2" of COMPILED.md prices it.

What is here
------------

``hyper_connection``
    The three kernels of
    ``vendor/mlx_vlm/models/qwen4_exp/hc_fused.py``: the per-stream RMS norm,
    the fused down/inject projection with its activation, and the up
    projection with the stream mix. Sources and kernel handles come from that
    module rather than being copied, so there is one Metal body per kernel in
    the tree and a change there is a change here. What this module supplies is
    the *calling convention*: arrays in, arrays out, eligibility decided once
    at build time instead of per call.

``gdn_norm_gate``
    ``titan.kernels.gdn_norm_gate``, the fused grouped RMS norm and output
    gate on the Gated DeltaNet value heads. Resolved through the adapter's
    registry door at build time, exactly as the eager site resolves it, so
    ``kernels.reference_only`` and ``kernels.disabled`` reach the compiled path
    too.

Both carry the eager path's own gate as a build-time predicate rather than a
runtime one, because bit-identity with eager is the whole claim: the compiled
step has to take a fused kernel exactly where the eager step takes it and the
ops path exactly where the eager step takes that. ``gdn_norm_gate``'s eager
site declines a single row (``x.shape[0] * x.shape[1] > 1``), so
:func:`gdn_norm_gate_plan` declines width one, even though the kernel is
bit-identical there too.

Nothing here monkeypatches anything and nothing here reads a module at call
time. A plan is a frozen dataclass of Python scalars and kernel handles; the
arrays travel in the weight pytree the caller owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .kernels import get as _titan_op
from .vendor.mlx_vlm.models.qwen4_exp import hc_fused as _hc

__all__ = [
    "GDNNormGatePlan",
    "HCFusedPlan",
    "OPAQUE_SHAPELESS_ERROR",
    "gdn_norm_gate_body",
    "gdn_norm_gate_plan",
    "hc_fused_arrays",
    "hc_fused_body",
    "hc_fused_plan",
    "opaque_under_shapeless",
]

#: The error every ``mx.fast.metal_kernel`` raises under a shapeless trace on
#: MLX 0.32.2. A claim about an error string is a claim about a version, so a
#: test asserts it rather than this docstring being the only record.
OPAQUE_SHAPELESS_ERROR = "CustomKernel cannot infer output shapes"

#: ``hc_fused`` takes at most this many rows; above it the eager path goes to
#: the compiled prefill mean, which is not a decode shape. Mirrored from the
#: vendored module rather than restated, so the two cannot drift.
HC_MAX_ROWS = _hc.MAX_ROWS


# ---------------------------------------------------------------------------
# the fused hyper-connection block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HCFusedPlan:
    """Everything about one ``Qwen4ExpGatedResidual`` that is not an array.

    Built once, at build time, from the same predicates
    ``hc_fused.compatible`` applies per call. A plan existing at all is the
    statement that this module's layout is one the three kernels support; the
    only thing left to decide at trace time is the row count, which is the
    verify width and is a Python integer inside a per-shape trace.
    """

    hc_count: int
    hidden_size: int
    hc_lowrank: int
    width: int
    bits_down: int
    bits_inject: int
    bits_up: int
    has_inject: bool
    dtype: Any
    eps: float


def hc_fused_plan(module: nn.Module) -> Optional[HCFusedPlan]:
    """A plan for *module*, or ``None`` when the eager path would decline too.

    The eligibility is ``hc_fused._layout_compatible`` minus the parts that
    need an input array: the quantisation of the two (or three) projections,
    the 64-element alignment the up grid and the quantisation groups require,
    ``hc_count == 4``, the norm's layout, and a Metal device. The row bound and
    the input dtype are the two runtime halves and they are checked in
    :func:`hc_fused_body`, where the shapes are concrete.

    Returning ``None`` is not a failure. It is the same answer the eager path
    gives for a layout the kernels do not cover, and the caller composes the
    plain-MLX block instead -- which is what generation 1 did everywhere.
    """
    if not _hc.enabled():
        return None
    if getattr(module, "input_inject_weight", None) is not None:
        # The merged down|inject bank. hc_fused declines this layout by name
        # ("combined input projection layout") and so does this.
        return None
    hc_count = getattr(module, "hc_count", None)
    hidden = getattr(module, "hidden_size", None)
    lowrank = getattr(module, "hc_lowrank", None)
    if not (
        isinstance(hc_count, int)
        and isinstance(hidden, int)
        and isinstance(lowrank, int)
        and hc_count == 4
        and hidden % 64 == 0
        and lowrank % 64 == 0
    ):
        return None
    norm = getattr(module, "hc_norm", None)
    if not (
        norm is not None
        and getattr(norm, "group_size", None) == hidden
        and isinstance(getattr(norm, "weight", None), mx.array)
        and norm.weight.dtype == mx.bfloat16
        and norm.weight.shape == (hc_count * hidden,)
    ):
        return None
    down = getattr(module, "input_mix_weight_down", None)
    up = getattr(module, "input_mix_weight_up", None)
    if not (_hc._quantized_ok(down) and _hc._quantized_ok(up)):
        return None
    inject = module.block_inject_weight if "block_inject_weight" in module else None
    if inject is not None and not _hc._quantized_ok(inject):
        return None
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return None
    return HCFusedPlan(
        hc_count=hc_count,
        hidden_size=hidden,
        hc_lowrank=lowrank,
        width=hc_count * hidden,
        bits_down=int(down.bits),
        bits_inject=int(inject.bits) if inject is not None else int(down.bits),
        bits_up=int(up.bits),
        has_inject=inject is not None,
        dtype=mx.bfloat16,
        eps=float(norm.eps),
    )


def hc_fused_arrays(module: nn.Module) -> dict[str, Any]:
    """The arrays :func:`hc_fused_body` needs, keyed by name.

    ``eps`` is an array rather than a float because the norm kernel takes it as
    an input buffer. The vendored module builds it lazily and calls ``mx.eval``
    on it, which is a side effect in the middle of a forward; here it is built
    once with the rest of the weights and evaluated with them.
    """
    down = module.input_mix_weight_down
    up = module.input_mix_weight_up
    inject = module.block_inject_weight if "block_inject_weight" in module else down
    return {
        "hc_norm_raw": module.hc_norm.weight,
        "eps": mx.array([float(module.hc_norm.eps)], dtype=mx.float32),
        "down": (down.weight, down.scales, down.biases),
        "up": (up.weight, up.scales, up.biases),
        "inject": (inject.weight, inject.scales, inject.biases),
    }


def hc_fused_body(hyper_input: mx.array, arrays: dict, plan: HCFusedPlan):
    """``hc_fused.fused_forward`` as a pure function of arrays. Traceable.

    Returns ``mixed`` when the module has no ``block_inject_weight`` and
    ``(mixed, hyper_input, injection)`` when it has one, which is the contract
    ``Qwen4ExpGatedResidual.__call__`` returns and the contract
    ``build_gated_residual`` already returns.

    Three differences from the vendored function, all of them the difference
    between a call site and a graph node:

    * no ``try/except``. The vendored version catches a lazy Metal compile
      failure and falls back, which needs an ``mx.eval`` to find out. Inside a
      trace there is nothing to fall back *to* -- the graph is already built --
      so the validation happens once, outside, when the caller warms the trace.
    * no ``mx.eval``. The one-off specialisation check moved to the warm-up.
    * no module reads. Everything is an argument.

    Raises ``ValueError`` for a row count the kernels do not take, rather than
    silently producing the ops path: the caller decided to use this and a quiet
    substitution would make the compiled step's numerics depend on a width.
    """
    batch, seq, _ = hyper_input.shape
    rows = batch * seq
    if not 1 <= rows <= HC_MAX_ROWS:
        raise ValueError(
            f"hc_fused takes 1..{HC_MAX_ROWS} rows, got {rows}; above that the "
            "eager path takes the compiled prefill mean, which is not a decode "
            "shape. Compose the plain-MLX block instead."
        )
    if hyper_input.dtype != mx.bfloat16:
        raise ValueError(
            f"hc_fused is a bfloat16 kernel, got {hyper_input.dtype}"
        )
    hc, hidden, lowrank = plan.hc_count, plan.hidden_size, plan.hc_lowrank
    width = plan.width
    dtype = hyper_input.dtype
    flat = hyper_input.reshape(rows, width)

    normed = _hc._kernel(
        "titan_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], _hc._N_SOURCE
    )(
        inputs=[flat, arrays["hc_norm_raw"], arrays["eps"]],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[0]

    down_w, down_s, down_b = arrays["down"]
    inject_tensors = arrays["inject"] if plan.has_inject else arrays["down"]
    act, injection = _hc._kernel(
        "titan_qwen4_hc_fused_down",
        ["xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
        ["act", "inj"],
        _hc._D_SOURCE,
        header=_hc._HEADER,
    )(
        inputs=[normed, down_w, down_s, down_b, *inject_tensors],
        template=[
            ("T", dtype),
            ("BITS_D", plan.bits_down),
            ("BITS_I", plan.bits_inject),
            ("K", width),
            ("R", lowrank),
            ("HC", hc),
            ("INJ", 1 if plan.has_inject else 0),
        ],
        grid=(32, 8 * (lowrank // 8 + 1), rows),
        threadgroup=(32, 8, 1),
        output_shapes=[(rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )

    up_w, up_s, up_b = arrays["up"]
    mixed = _hc._kernel(
        "titan_qwen4_hc_fused_up",
        ["xn", "act", "up_w", "up_s", "up_b"],
        ["mixed"],
        _hc._U_SOURCE,
        header=_hc._HEADER,
    )(
        inputs=[normed, act, up_w, up_s, up_b],
        template=[
            ("T", dtype),
            ("BITS_U", plan.bits_up),
            ("K", width),
            ("R", lowrank),
            ("HC", hc),
            ("H", hidden),
            ("S", rows),
        ],
        grid=(256, hidden // 64, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[dtype],
    )[0]

    mixed = mixed.reshape(batch, seq, hidden)
    if not plan.has_inject:
        return mixed
    return mixed, hyper_input, injection.reshape(batch, seq, hc)


# ---------------------------------------------------------------------------
# the fused Gated DeltaNet norm and output gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GDNNormGatePlan:
    """The resolved ``gdn.norm_gate_fused`` op plus its non-array arguments."""

    op: Callable[..., Any]
    eps: float
    activation: int


def gdn_norm_gate_plan(
    module: nn.Module, *, width: int, dtype=mx.bfloat16
) -> Optional[GDNNormGatePlan]:
    """A plan for ``Qwen4ExpRMSNormGated``, or ``None`` where eager declines.

    The gate is the eager site's gate, condition for condition
    (``language.py``, ``Qwen4ExpRMSNormGated.__call__``): four dimensions, more
    than one row, a half-precision dtype, and a registry that has the op. Width
    one is declined here because the eager site declines it, and matching eager
    exactly is the point -- not because the kernel would be wrong there. The
    kernel is bit-identical at every width its ``supports`` accepts.

    A registry with ``kernels.reference_only`` set returns ``None`` from
    ``_titan_op`` and so does this, which is what makes the compiled path's
    control arm the same control arm the server would run.
    """
    if width <= 1:
        return None
    if dtype not in (mx.bfloat16, mx.float16):
        return None
    op = _titan_op("gdn.norm_gate_fused")
    if op is None:
        return None
    activation = getattr(module, "activation", "sigmoid")
    return GDNNormGatePlan(
        op=op,
        eps=float(module.eps),
        activation=0 if activation == "sigmoid" else 1,
    )


def gdn_norm_gate_body(
    x: mx.array, gate: mx.array, norm_weight: mx.array, plan: GDNNormGatePlan
) -> mx.array:
    """One fused launch for ``rms_norm(x, w) * activation(gate)``. Traceable.

    ``x`` and ``gate`` are ``[B, W, heads, head_dim]``. The kernel is
    bit-identical to the ops form it replaces; see ``titan/kernels/README.md``.
    """
    return plan.op(
        x,
        gate,
        norm_weight.astype(x.dtype),
        eps=plan.eps,
        activation=plan.activation,
    )


# ---------------------------------------------------------------------------
# the probe the tests and the bench share
# ---------------------------------------------------------------------------


def opaque_under_shapeless(fn: Callable[..., Any], *args) -> Optional[str]:
    """Trace *fn* shapeless and hand back the error, or ``None`` if it traced.

    The report generation 2 owes is "which kernels compiled as opaque nodes and
    which did not, with the errors", and an error message is a fact about an
    MLX version rather than a fact about a kernel. This is how both the test
    suite and ``bench/decode/compiled_path.py kernels`` get one, so the table
    in COMPILED.md is generated from the machine rather than remembered.
    """
    try:
        mx.eval(mx.compile(fn, shapeless=True)(*args))
    except Exception as exc:  # noqa: BLE001 - the message is the result
        return f"{type(exc).__name__}: {exc}"
    return None
