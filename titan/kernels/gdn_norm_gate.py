# SPDX-License-Identifier: MIT
"""Fused grouped RMSNorm + output gate for the Gated DeltaNet value heads.

Ported from ``engine/patches/round2/gdn-norm/kernel.py``.

The reference is the arithmetic the stock model runs::

    y = rms_norm(x, w, eps).astype(float32)
    g = sigmoid(gate.astype(float32))        # or silu
    out = (y * g).astype(x.dtype)

At [B, S, 48, 128] bf16 that is a bf16 norm result, two fp32 casts, an fp32
product and a bf16 store: about 250 MB of traffic for 75 MB of payload. The
Metal implementation does it in one dispatch, reading bf16 and accumulating the
sum of squares in fp32, and never materialises an fp32 tensor.

Exactness: bit-identical. The reduction copies mlx's own ``rms_single_row``
(one 32-lane simdgroup per row, four contiguous elements per lane, ``simd_sum``
of the squares, ``metal::precise::rsqrt(acc / axis + eps)``, the same
``w * static_cast<T>(x * normalizer)`` rounding), and the gate product then goes
through fp32 exactly as the reference does. At DV = 128 the reduction order
matches mlx bit for bit.

The op takes T >= 1: unlike the overlay, which deliberately declined single-row
input so oMLX's own decode kernel kept it, Titan owns both paths.
"""

from __future__ import annotations

import mlx.core as mx

from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["GATE_SIGMOID", "GATE_SILU", "OP", "gate_code", "key", "metal",
           "reference", "supports"]

GATE_SIGMOID = 0
GATE_SILU = 1

_ACTIVATIONS = {"sigmoid": GATE_SIGMOID, "silu": GATE_SILU, "swish": GATE_SILU}
_DTYPES = (mx.bfloat16, mx.float16)


def gate_code(activation) -> int | None:
    """Map ``"sigmoid"`` / ``"silu"`` / an int code onto the kernel's code."""
    if isinstance(activation, int):
        return activation if activation in (GATE_SIGMOID, GATE_SILU) else None
    return _ACTIVATIONS.get(activation)


_SOURCE = """
    constexpr int NPT = DV / 32;                 // elements per lane
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.y;   // flattened B*S token index
    const uint head = thread_position_in_grid.z;  // value head
    const uint base = (row * uint(HV) + head) * uint(DV) + lane * NPT;
    const uint wbase = lane * NPT;

    float xs[NPT];
    float sumsq = 0.0f;
    for (int i = 0; i < NPT; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    const float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));

    for (int i = 0; i < NPT; ++i) {
        // rms_norm materialises the norm in T before the gate product casts
        // back to fp32, so the intermediate round trip is kept.
        const T normed = norm_w[wbase + i] * T(xs[i] * inv);
        const float zv = float(z[base + i]);
        float gate;
        if (GATE == 0) {
            // sigmoid, computed on the stable side to avoid exp overflow
            const float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
            gate = zv < 0.0f ? sy : 1.0f - sy;
        } else {
            const float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
            gate = zv * (zv < 0.0f ? sy : 1.0f - sy);
        }
        out[base + i] = T(float(normed) * gate);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="titan_gdn_norm_gate",
            input_names=["y", "z", "norm_w", "eps"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def _heads_per_group(hv: int) -> int:
    """Pack several heads into one threadgroup; each head is its own simdgroup."""
    for hpg in (8, 6, 4, 3, 2):
        if hv % hpg == 0:
            return hpg
    return 1


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def reference(x, gate, norm_w, *, eps, activation=GATE_SIGMOID):
    """Plain-MLX grouped RMSNorm + gate. ``x``, ``gate``: [B, S, HV, DV]."""
    dtype = x.dtype
    code = gate_code(activation)
    y = mx.fast.rms_norm(x, norm_w, eps).astype(mx.float32)
    g = gate.astype(mx.float32)
    g = mx.sigmoid(g) if code == GATE_SIGMOID else g * mx.sigmoid(g)
    return (y * g).astype(dtype)


def metal(x, gate, norm_w, *, eps, activation=GATE_SIGMOID):
    """One fused dispatch. Identical signature to :func:`reference`."""
    b, s, hv, dv = x.shape
    return _kernel()(
        inputs=[x, gate, norm_w, mx.array(eps, dtype=mx.float32)],
        template=[("T", x.dtype), ("HV", hv), ("DV", dv),
                  ("GATE", int(gate_code(activation)))],
        grid=(32, b * s, hv),
        threadgroup=(32, 1, _heads_per_group(hv)),
        output_shapes=[(b, s, hv, dv)],
        output_dtypes=[x.dtype],
    )[0]


def key(x, gate, norm_w, *, eps, activation=GATE_SIGMOID) -> ShapeClass:
    return shape_class(x, gate, norm_w, extra=(gate_code(activation),))


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) != 3:
        return False
    xs, gs, ws = k.shapes
    xd, gd, wd = k.dtypes
    if len(xs) != 4 or gs != xs or len(ws) != 1:
        return False
    if xd not in _DTYPES or gd != xd or wd != xd:
        return False
    dv = xs[3]
    if ws[0] != dv or dv % 32 or dv // 32 > 8 or xs[2] < 1:
        return False
    return bool(k.extra) and k.extra[0] in (GATE_SIGMOID, GATE_SILU)


OP = KernelOp(
    name="gdn_norm_gate",
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=0.0,
    shapes=(
        {"B": 1, "S": 8, "HV": 48, "DV": 128},
        {"B": 1, "S": 64, "HV": 48, "DV": 128},
        {"B": 1, "S": 1, "HV": 48, "DV": 128},
        {"B": 1, "S": 2048, "HV": 48, "DV": 128},
    ),
    exactness="bit-identical",
    source="engine/patches/round2/gdn-norm/REPORT.md",
)
