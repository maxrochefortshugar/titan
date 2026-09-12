#!/usr/bin/env python3
"""oMLX patch: run the prefill hyper-connection norm on the bf16 Metal kernel.

``Qwen4ExpGatedResidual`` runs 96 times per chunk (two per layer plus the final
mixer) over a [1, T, 10240] residual stream.  Above 16 rows the module takes
``hc_fused.prefill_forward``, which is "fused" only in its compiled mean: it
still calls ``module.hc_norm``, and ``Qwen4ExpRMSNorm`` with a ``group_size``
cannot hand ``mx.fast.rms_norm`` a per-group weight, so it does

    x.astype(float32) -> rms_norm -> * scale_fp32 -> astype(bf16)

which at T=2048 is 42 MB in, an 84 MB fp32 intermediate, and ~420 MB of traffic
per call.  The profile measured the round trip at **85 ms per 2048-token chunk,
7.8% of the body**.

``hc_fused._kernel_norm`` is the same grouped RMS norm as a Metal kernel that
reads bf16, accumulates the sum of squares in fp32 per stream, and writes bf16 --
no fp32 tensor ever exists.  It already ships, is already validated, and is
already used by ``fused_forward`` on the decode path; ``prefill_forward`` simply
never calls it.  This patch is that one-line change, applied at runtime.

Numerics: identical arithmetic, different fp32 rounding order (``metal::rsqrt``
and a simd-tree reduction against MLX's ``rms_norm``), so results agree to
bf16 ULP.  ``test_norm_exact.py`` reports the actual distribution.

Enable with ``OMLX_QWEN4_BF16_NORM=1`` and call :func:`apply_bf16_norm_patch`
(no model argument needed -- it patches the module function).  Setting
``OMLX_QWEN4_BF16_NORM_ALL=1`` additionally routes every grouped
``Qwen4ExpRMSNorm`` (the three PLE norms) through the same bf16 arithmetic
without the fp32 round trip; that is a strictly larger change and is off by
default.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_APPLIED = False
_APPLIED_ALL = False
_ORIGINAL_PREFILL = None
_ORIGINAL_NORM_CALL = None

# instrumentation
STATS = {"kernel_calls": 0, "fallback_calls": 0}


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _make_prefill_forward(hc_fused):
    import mlx.core as mx
    import mlx.nn as nn

    def prefill_forward(module, hyper_input):
        """prefill_forward with the bf16 grouped-norm kernel instead of hc_norm."""
        try:
            hc, hidden = module.hc_count, module.hidden_size
            dtype = hyper_input.dtype
            rows = hyper_input.shape[0] * hyper_input.shape[1]
            width = hc * hidden
            try:
                normed = hc_fused._kernel_norm(
                    module, hyper_input.reshape(rows, width), rows, hc, hidden, dtype
                ).reshape(hyper_input.shape)
                STATS["kernel_calls"] += 1
            except Exception as exc:  # noqa: BLE001
                # a shape the kernel will not take: keep the canonical norm
                logger.debug("bf16 grouped norm kernel declined: %s", exc)
                normed = module.hc_norm(hyper_input)
                STATS["fallback_calls"] += 1
            mix = nn.silu(module.input_mix_weight_down(normed) / hc)
            mixed = hc_fused._tail(hc, hidden)(module.input_mix_weight_up(mix), normed)
            inject = (module.block_inject_weight
                      if "block_inject_weight" in module else None)
            injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
            signature = ("prefill_bf16norm", dtype, hc, hidden, module.hc_lowrank,
                         module.input_mix_weight_down.bits)
            if signature not in hc_fused._VALIDATED:
                mx.eval(mixed) if injection is None else mx.eval(mixed, injection)
                hc_fused._VALIDATED.add(signature)
            if injection is None:
                return mixed
            return mixed, hyper_input, injection
        except Exception as exc:  # noqa: BLE001
            # Same contract as upstream: returning None sends the caller to the
            # canonical _forward, so a failure here can never change results.
            logger.warning("bf16 prefill hyper-connection failed closed: %s", exc)
            return None

    prefill_forward._omlx_bf16_norm = True
    return prefill_forward


def _make_norm_call():
    import mlx.core as mx

    def __call__(self, x):
        """Grouped RMSNorm without the fp32 round trip on the activations."""
        dtype = x.dtype
        if self.group_size is None or dtype not in (mx.bfloat16, mx.float16):
            return _ORIGINAL_NORM_CALL(self, x)
        scale = (1.0 + self.weight).astype(dtype)
        y = x.reshape(*x.shape[:-1], -1, self.group_size)
        y = mx.fast.rms_norm(y, None, self.eps)
        y = y * scale.reshape(-1, self.group_size)
        return y.reshape(x.shape).astype(dtype)

    __call__._omlx_bf16_norm = True
    return __call__


def apply_bf16_norm_patch(model=None, *, force=False, also_all_norms=None) -> bool:
    """Route hc_fused.prefill_forward through the bf16 grouped-norm kernel."""
    global _APPLIED, _APPLIED_ALL, _ORIGINAL_PREFILL, _ORIGINAL_NORM_CALL
    if not force and not _env_flag("OMLX_QWEN4_BF16_NORM"):
        return False
    from mlx_vlm.models.qwen4_exp import hc_fused

    if not _APPLIED:
        if getattr(hc_fused.prefill_forward, "_omlx_bf16_norm", False):
            _APPLIED = True
        else:
            _ORIGINAL_PREFILL = hc_fused.prefill_forward
            hc_fused.prefill_forward = _make_prefill_forward(hc_fused)
            _APPLIED = True
            logger.info("Qwen4 prefill hyper-connection norm: bf16 Metal kernel")

    if also_all_norms is None:
        also_all_norms = _env_flag("OMLX_QWEN4_BF16_NORM_ALL")
    if also_all_norms and not _APPLIED_ALL:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

        _ORIGINAL_NORM_CALL = Qwen4ExpRMSNorm.__call__
        Qwen4ExpRMSNorm.__call__ = _make_norm_call()
        _APPLIED_ALL = True
        logger.info("Qwen4ExpRMSNorm: grouped path kept in bf16")
    return True


def remove_bf16_norm_patch() -> bool:
    global _APPLIED, _APPLIED_ALL
    removed = False
    from mlx_vlm.models.qwen4_exp import hc_fused

    if _APPLIED and _ORIGINAL_PREFILL is not None:
        hc_fused.prefill_forward = _ORIGINAL_PREFILL
        _APPLIED = False
        removed = True
    if _APPLIED_ALL and _ORIGINAL_NORM_CALL is not None:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

        Qwen4ExpRMSNorm.__call__ = _ORIGINAL_NORM_CALL
        _APPLIED_ALL = False
        removed = True
    return removed


def is_applied():
    return _APPLIED, _APPLIED_ALL
