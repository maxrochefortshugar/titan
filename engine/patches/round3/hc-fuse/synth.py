#!/usr/bin/env python3
"""Synthetic Qwen4ExpGatedResidual for exactness and bench work.

Nothing here loads the model.  The two vendored oMLX modules that the fused
path builds on (``hc_fused``, ``hc_projection``) are imported from the app
bundle read only, through a throwaway package alias so their relative import
resolves.  ``Qwen4ExpRMSNorm`` and the canonical ``_forward`` are transcribed
verbatim from
``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py``
(lines 1049-1070 and 1684-1757) so the reference path is the stock arithmetic.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

VENDOR = Path(
    "/Applications/oMLX.app/Contents/Resources/omlx/patches/"
    "mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp"
)


def load_hc_fused():
    """Import the app's hc_fused / hc_projection without importing mlx_vlm."""
    if "hcvendor" not in sys.modules:
        pkg = types.ModuleType("hcvendor")
        pkg.__path__ = [str(VENDOR)]
        sys.modules["hcvendor"] = pkg
        for name in ("hc_projection", "hc_fused"):
            spec = importlib.util.spec_from_file_location(
                f"hcvendor.{name}", VENDOR / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"hcvendor.{name}"] = mod
            spec.loader.exec_module(mod)
    return sys.modules["hcvendor.hc_fused"]


class Qwen4ExpRMSNorm(nn.Module):
    """Verbatim from language.py:1049-1070."""

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"{dim=} must be divisible by {group_size=}")
        self.weight = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        scale = 1.0 + self.weight.astype(mx.float32)
        if self.group_size is None:
            return mx.fast.rms_norm(x, scale, self.eps).astype(dtype)
        y = x.astype(mx.float32).reshape(*x.shape[:-1], -1, self.group_size)
        y = mx.fast.rms_norm(y, None, self.eps) * scale.reshape(-1, self.group_size)
        return y.reshape(x.shape).astype(dtype)


class GatedResidual(nn.Module):
    """Synthetic Qwen4ExpGatedResidual with the canonical _forward."""

    def __init__(self, hidden=2560, hc_count=4, lowrank=320, bits=5,
                 inject_bits=None, eps=1e-6, use_combine=True, seed=0):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden
        self.hc_lowrank = lowrank
        width = hc_count * hidden
        mx.random.seed(seed)
        self.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=eps)
        # checkpoint norm weights are centred at zero and stored bf16
        self.hc_norm.weight = (
            mx.random.normal((width,)) * 0.05
        ).astype(mx.bfloat16)

        def qlin(nout, nin, b):
            lin = nn.Linear(nin, nout, bias=False)
            lin.weight = (mx.random.normal((nout, nin)) * 0.02).astype(mx.bfloat16)
            return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=b)

        self.input_mix_weight_down = qlin(lowrank, width, bits)
        self.input_mix_weight_up = qlin(width, lowrank, bits)
        if use_combine:
            self.block_inject_weight = qlin(
                hc_count, width, bits if inject_bits is None else inject_bits
            )

    # language.py:1684-1757, target_verify=False, no hybrid, no
    # input_inject_weight (neither exists on this checkpoint's prefill path)
    def _forward(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        mix = self.input_mix_weight_down(normed)
        block_injection = (
            self.block_inject_weight(normed)
            if "block_inject_weight" in self
            else None
        )
        mix = nn.silu(mix / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * streams, axis=-2)
        if block_injection is None:
            return mixed_input
        injection_weights = 2 * mx.sigmoid(block_injection / self.hc_count)
        return mixed_input, hyper_input, injection_weights


def make_input(rows, width=10240, scale=1.0, seed=1):
    """A residual stream at a realistic magnitude: unit-ish, bf16, [1, T, W]."""
    mx.random.seed(seed)
    return (mx.random.normal((1, rows, width)) * scale).astype(mx.bfloat16)


def _ordered_bits(a: mx.array) -> mx.array:
    """bf16 bit pattern as a monotone signed integer, so |x - y| counts ULPs."""
    u = a.astype(mx.bfloat16).view(mx.uint16).astype(mx.int32)
    sign = u >> 15
    mag = u & 0x7FFF
    return mx.where(sign == 1, -mag, mag)


def ulp_stats(a: mx.array, b: mx.array):
    """max abs error, max relative error, max and mean error in bf16 ULPs.

    The ULP count is the distance between bf16 bit patterns, which is the only
    measure that stays meaningful across exponents and near zero.
    """
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    diff = mx.abs(af - bf)
    rel = diff / mx.maximum(mx.abs(bf), 1e-6)
    d = mx.abs(_ordered_bits(a) - _ordered_bits(b)).astype(mx.float32)
    return (
        float(mx.max(diff).item()),
        float(mx.max(rel).item()),
        float(mx.max(d).item()),
        float(mx.mean(d).item()),
    )
