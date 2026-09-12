#!/usr/bin/env python3
"""Fused grouped RMSNorm + output gate for the Qwen4-Exp Gated DeltaNet, T >= 1.

Stock code being replaced (``Qwen4ExpRMSNormGated.__call__``,
``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:1080-1088``)::

    y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
    gate = gate.astype(mx.float32)
    gate = mx.sigmoid(gate)            # output_gate_type == "sigmoid"
    return (y * gate).astype(dtype)

x and gate are ``[B, S, 48, 128]`` bf16 (the GDN value heads), so at a
2048-token chunk that is 25 MB in each, a 50 MB bf16 norm result, two 50 MB
fp32 casts, a 50 MB fp32 product and a 25 MB bf16 store: about 250 MB of
traffic for 75 MB of payload, 36 times per chunk.

This kernel does the whole thing in one dispatch: read bf16, accumulate the
sum of squares in fp32, and write bf16.  No fp32 tensor is ever materialised.

Exactness.  The arithmetic is copied from mlx's own ``rms_single_row`` kernel
(one 32-lane simdgroup per row, four contiguous elements per lane,
``simd_sum`` of the squares, ``metal::precise::rsqrt(acc / axis + eps)``, and
the same ``w * static_cast<T>(x * normalizer)`` rounding), and the gate
product then goes through fp32 exactly as the stock Python does.  At DV=128
the reduction order matches mlx's bit for bit, so the result is identical, not
merely close.  ``test_exact.py`` reports the measured distribution.

oMLX already ships this fusion for one-row decode
(``omlx/patches/qwen35_gdn_prework.py:383-411`` ``qwen4_decode_norm_gate_fused``),
reached from the fused decode arm at ``qwen35_gdn_prework.py:669``, which
bypasses ``self.norm`` entirely.  This module deliberately engages only for
4-D input with ``B * S > 1``, so decode keeps oMLX's kernel untouched.
"""
from __future__ import annotations

import mlx.core as mx

# gate activation codes
GATE_SIGMOID = 0
GATE_SILU = 1

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
        // mx.fast.rms_norm materialises bf16 before Qwen4 casts back to fp32
        // for the gate product, so the intermediate round trip is kept.
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
            name="gdn_norm_gate_fused",
            input_names=["y", "z", "norm_w", "eps"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def _heads_per_group(hv: int, dv: int) -> int:
    """Pack several heads into one threadgroup; each head is its own simdgroup."""
    max_heads = max(1, 1024 // 32)
    for hpg in (8, 6, 4, 3, 2):
        if hv % hpg == 0 and hpg <= max_heads:
            return hpg
    return 1


def norm_gate_fused(x, gate, norm_w, *, eps, activation=GATE_SIGMOID):
    """Fused grouped RMSNorm + gate.

    x, gate: ``[B, S, HV, DV]`` of the same floating dtype.
    norm_w:  ``[DV]``, same dtype.
    Returns an array of x's shape and dtype.
    """
    b, s, hv, dv = x.shape
    rows = b * s
    hpg = _heads_per_group(hv, dv)
    return _kernel()(
        inputs=[x, gate, norm_w, mx.array(eps, dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("HV", hv),
            ("DV", dv),
            ("GATE", int(activation)),
        ],
        grid=(32, rows, hv),
        threadgroup=(32, 1, hpg),
        output_shapes=[(b, s, hv, dv)],
        output_dtypes=[x.dtype],
    )[0]


# ---------------------------------------------------------------------------
# eligibility


_ACTIVATIONS = {"sigmoid": GATE_SIGMOID, "silu": GATE_SILU, "swish": GATE_SILU}
_DTYPES = (mx.bfloat16, mx.float16)


def gate_code(activation):
    return _ACTIVATIONS.get(activation)


def eligible(x, gate, norm_w, activation) -> bool:
    """Prefill-only gate.  One-row input is left to oMLX's decode kernel."""
    if gate_code(activation) is None:
        return False
    if not isinstance(x, mx.array) or not isinstance(gate, mx.array):
        return False
    if x.ndim != 4 or gate.shape != x.shape:
        return False
    b, s, hv, dv = x.shape
    if b * s <= 1:                      # decode / single-row verify: stock path
        return False
    if x.dtype not in _DTYPES or gate.dtype != x.dtype or norm_w.dtype != x.dtype:
        return False
    if norm_w.ndim != 1 or norm_w.shape[0] != dv:
        return False
    if dv % 32 or dv // 32 > 8 or hv < 1:
        return False
    return True


# ---------------------------------------------------------------------------
# reference (the stock arithmetic, for tests)


def norm_gate_stock(x, gate, norm_w, *, eps, activation=GATE_SIGMOID):
    dtype = x.dtype
    y = mx.fast.rms_norm(x, norm_w, eps).astype(mx.float32)
    g = gate.astype(mx.float32)
    if activation == GATE_SIGMOID:
        g = mx.sigmoid(g)
    else:
        g = g * mx.sigmoid(g)
    return (y * g).astype(dtype)
