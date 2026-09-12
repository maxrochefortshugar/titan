#!/usr/bin/env python3
"""Fused Qwen4 hyper-connection block for the prefill path (T > 16 rows).

The stock prefill path (``hc_fused.prefill_forward``, optionally with the
deployed bf16 norm patch) runs the block as roughly eleven Metal launches over
a [T, 10240] bf16 residual stream:

    normed = hc_norm(x)                     read  42 MB, write 42 MB
    mix    = silu(down(normed) / hc)        read  42 MB   (+3 elementwise)
    inj    = 2 * sigmoid(injw(normed) / hc) read  42 MB   (+3 elementwise)
    up_out = up(mix)                        write 42 MB
    mixed  = mean_g sigmoid(up_out)_g * normed_g   read 84 MB, write 10.5 MB

This module keeps the two quantized matmuls on MLX (they reach the NAX tensor
units at M = 2048 and a hand written replacement would fall back to the fp32
ALU rate) and fuses everything around them:

  1. the grouped RMS norm runs on ``hc_fused._kernel_norm``, bf16 in and bf16
     out, no fp32 tensor (this subsumes ``kernels/ple-fix/norm_patch.py``),
  2. the down and block-inject projections become one ``quantized_matmul``
     over a concatenated [hc_lowrank + hc_count, width] weight bank, so the
     stream is read once instead of twice,
  3. one epilogue kernel splits that result and applies silu and the injection
     sigmoid, replacing six elementwise launches over a small tensor,
  4. one tail kernel applies the gate sigmoid, multiplies by the normed stream
     and averages the hc streams, so the [T, 10240] up-projection result is
     consumed where it is produced.

Five launches instead of eleven, and the stream is touched five times instead
of eight. fp32 accumulation throughout, bf16 storage everywhere.

Gate: OMLX_QWEN4_HC_FUSE2=1.  Two sub-flags, both off by default because both
move results by more than 1 bf16 ULP: OMLX_QWEN4_HC_FUSE2_CONCAT=1 merges the
down and injection banks into one matmul, OMLX_QWEN4_HC_FUSE2_FP32_MEAN=1
accumulates the stream mean in fp32 instead of matching MLX's bf16 reduce.
"""
from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

STATS = {"fused": 0, "concat": 0, "split": 0, "fallback": 0}

_KERNELS: dict[str, object] = {}
_VALIDATED: set = set()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Metal sources
# --------------------------------------------------------------------------

# MLX's Sigmoid (mlx/backend/metal/kernels/unary_ops.h:308-314) is the stable
# form, and with T = bfloat16_t every intermediate goes through bf16_math.h,
# which rounds to bf16 after each step.  Reproducing that rounding exactly is
# what keeps the fused block inside 1 ULP of the stock elementwise chain.
_HEADER = r"""
using namespace metal;

template <typename T>
inline float hc2_sigmoid(float x) {
    const float e = float(T(metal::exp(metal::abs(x))));
    const float d = float(T(1.0f + e));
    const float y = float(T(1.0f / d));
    return (x < 0.0f) ? y : float(T(1.0f - y));
}
"""

# Both projection results, both activations, one launch.  d is [rows, R] from
# the low-rank bank and j is [rows, HC] from the injection bank (or a dummy
# when INJ == 0); act is silu(v / HC) and inj is 2 * sigmoid(v / HC).
_EPI_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    const int width = R + (INJ ? HC : 0);
    const int c = int(idx) % width;
    const int row = int(idx) / width;
    const float raw = (c < R) ? float(d[(size_t)row * R + c])
                              : float(j[(size_t)row * HC + (c - R)]);
    const float v = float(T(raw / float(HC)));
    const float s = hc2_sigmoid<T>(v);
    if (c < R) {
        act[(size_t)row * R + c] = T(v * s);
    } else {
        inj[(size_t)row * HC + (c - R)] = T(2.0f * s);
    }
"""

# Gate, stream product and the mean over the hc streams in one pass:
#   mixed[r, h] = mean_g sigmoid(up[r, g*H + h]) * normed[r, g*H + h].
# ACC32 = 0 reproduces MLX's reduction, which carries a bf16 accumulator for a
# bf16 input, and is therefore bit identical to the stock tail.  ACC32 = 1 is
# the strictly more accurate fp32 accumulation; it drifts from stock by more
# than 1 ULP wherever the four stream terms cancel, so it is not the default.
_TAIL_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    // VEC consecutive hidden lanes per thread: 2 * VEC bytes per load keeps the
    // simdgroup's requests wide over the two 42 MB reads.
    const int h = int(idx % (uint)(H / VEC)) * VEC;
    const int row = int(idx / (uint)(H / VEC));
    const size_t base = (size_t)row * (size_t)(H * HC) + (size_t)h;
    float acc[VEC];
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;
    for (int g = 0; g < HC; ++g) {
        const size_t o = base + (size_t)(g * H);
        for (int v = 0; v < VEC; ++v) {
            const float gate = hc2_sigmoid<T>(float(up[o + v]));
            const float prod = float(T(gate * float(xn[o + v])));
            acc[v] = ACC32 ? (acc[v] + prod) : float(T(acc[v] + prod));
        }
    }
    const size_t ob = (size_t)row * (size_t)H + (size_t)h;
    for (int v = 0; v < VEC; ++v) mixed[ob + v] = T(acc[v] / float(HC));
"""


# Concat variant: one [rows, R + HC] result to split.
_EPI_COMB_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    const int c = int(idx) % (R + HC);
    const int row = int(idx) / (R + HC);
    const float v = float(T(float(comb[idx]) / float(HC)));
    const float s = hc2_sigmoid<T>(v);
    if (c < R) {
        act[(size_t)row * R + c] = T(v * s);
    } else {
        inj[(size_t)row * HC + (c - R)] = T(2.0f * s);
    }
"""


def _kernel(name, input_names, output_names, source):
    k = _KERNELS.get(name)
    if k is None:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            header=_HEADER,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = k
    return k


def _epilogue(d, j, rows, lowrank, hc, dtype):
    """(act, inj) from the two projection results; j may be None."""
    inj_flag = 0 if j is None else 1
    total = rows * (lowrank + (hc if inj_flag else 0))
    tg = 256
    dummy = d if j is None else j
    out = _kernel(
        "omlx_hc2_epilogue", ["d", "j"], ["act", "inj"], _EPI_SOURCE
    )(
        inputs=[d, dummy],
        template=[("T", dtype), ("R", lowrank), ("HC", hc), ("INJ", inj_flag),
                  ("TOTAL", total)],
        grid=((total + tg - 1) // tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )
    return (out[0], None) if j is None else (out[0], out[1])


def _tail(up, normed, rows, hc, hidden, dtype, acc32=False, vec=4):
    vec = vec if hidden % vec == 0 else 1
    total = rows * (hidden // vec)
    tg = 256
    return _kernel(
        "omlx_hc2_tail", ["up", "xn"], ["mixed"], _TAIL_SOURCE
    )(
        inputs=[up, normed],
        template=[("T", dtype), ("H", hidden), ("HC", hc), ("VEC", vec),
                  ("ACC32", 1 if acc32 else 0), ("TOTAL", total)],
        grid=((total + tg - 1) // tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[dtype],
    )[0]


# --------------------------------------------------------------------------
# Concatenated down|inject bank
# --------------------------------------------------------------------------

def _concat_bank(module):
    """[R + HC, width] quantized bank, or None when the two banks disagree."""
    bank = getattr(module, "_omlx_hc2_bank", None)
    if bank is not None:
        return bank if bank != () else None
    down = module.input_mix_weight_down
    inject = module.block_inject_weight if "block_inject_weight" in module else None
    ok = (
        inject is not None
        and getattr(down, "bits", None) == getattr(inject, "bits", None)
        and getattr(down, "group_size", None) == getattr(inject, "group_size", None)
        and getattr(down, "mode", "affine") == getattr(inject, "mode", "affine")
        and down.weight.dtype == inject.weight.dtype
        and down.weight.shape[1] == inject.weight.shape[1]
        and down.scales.shape[1] == inject.scales.shape[1]
        and down.scales.dtype == inject.scales.dtype
        and down.biases.dtype == inject.biases.dtype
    )
    if not ok:
        module._omlx_hc2_bank = ()
        return None
    bank = (
        mx.concatenate([down.weight, inject.weight], axis=0),
        mx.concatenate([down.scales, inject.scales], axis=0),
        mx.concatenate([down.biases, inject.biases], axis=0),
        int(down.bits),
        int(down.group_size),
    )
    mx.eval(bank[0], bank[1], bank[2])
    module._omlx_hc2_bank = bank
    return bank


def bank_bytes(module) -> int:
    bank = getattr(module, "_omlx_hc2_bank", None)
    if not bank:
        return 0
    return bank[0].nbytes + bank[1].nbytes + bank[2].nbytes


# --------------------------------------------------------------------------
# The fused prefill forward
# --------------------------------------------------------------------------

def fused_prefill_forward(module, hyper_input):
    """Drop-in for ``hc_fused.prefill_forward``; returns None to fail closed."""
    from mlx_vlm.models.qwen4_exp import hc_fused  # noqa: PLC0415

    return _fused_prefill_forward(hc_fused, module, hyper_input)


def _fused_prefill_forward(hc_fused, module, hyper_input, use_concat=None):
    try:
        hc = module.hc_count
        hidden = module.hidden_size
        lowrank = module.hc_lowrank
        width = hc * hidden
        dtype = hyper_input.dtype
        batch, seq = hyper_input.shape[0], hyper_input.shape[1]
        rows = batch * seq
        flat = hyper_input.reshape(rows, width)

        normed = hc_fused._kernel_norm(module, flat, rows, hc, hidden, dtype)

        down = module.input_mix_weight_down
        inject = module.block_inject_weight if "block_inject_weight" in module else None
        if use_concat is None:
            use_concat = _env_flag("OMLX_QWEN4_HC_FUSE2_CONCAT")
        bank = _concat_bank(module) if (use_concat and inject is not None) else None

        if bank is not None:
            # One projection instead of two: the low-rank rows come out bit
            # identical, the four injection rows do not (see REPORT).
            w, s_, b_, bits, gs = bank
            comb = mx.quantized_matmul(
                normed, w, s_, b_, transpose=True, group_size=gs, bits=bits
            )
            total = rows * (lowrank + hc)
            tg = 256
            act, inj = _kernel(
                "omlx_hc2_epilogue_comb", ["comb"], ["act", "inj"],
                _EPI_COMB_SOURCE,
            )(
                inputs=[comb],
                template=[("T", dtype), ("R", lowrank), ("HC", hc),
                          ("TOTAL", total)],
                grid=((total + tg - 1) // tg * tg, 1, 1),
                threadgroup=(tg, 1, 1),
                output_shapes=[(rows, lowrank), (rows, hc)],
                output_dtypes=[dtype, dtype],
            )
            STATS["concat"] += 1
        else:
            act, inj = _epilogue(
                down(normed),
                None if inject is None else inject(normed),
                rows, lowrank, hc, dtype,
            )
            STATS["split"] += 1

        up_out = module.input_mix_weight_up(act)
        mixed = _tail(up_out, normed, rows, hc, hidden, dtype,
                      acc32=_env_flag("OMLX_QWEN4_HC_FUSE2_FP32_MEAN"))

        signature = (
            "hc2", dtype, hc, hidden, lowrank, module.input_mix_weight_up.bits,
            bank is not None, inject is not None,
        )
        if signature not in _VALIDATED:
            if inj is None:
                mx.eval(mixed)
            else:
                mx.eval(mixed, inj)
            _VALIDATED.add(signature)

        STATS["fused"] += 1
        mixed = mixed.reshape(batch, seq, hidden)
        if inj is None:
            return mixed
        return mixed, hyper_input, inj.reshape(batch, seq, hc)
    except Exception as exc:  # noqa: BLE001 - optional native path
        STATS["fallback"] += 1
        logger.warning("hc-fuse2 failed closed, canonical path in effect: %s", exc)
        return None


__all__ = [
    "STATS",
    "bank_bytes",
    "fused_prefill_forward",
    "_fused_prefill_forward",
]
