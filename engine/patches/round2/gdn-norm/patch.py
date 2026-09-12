#!/usr/bin/env python3
"""oMLX patch: fused grouped norm + output gate for Qwen4-Exp GDN prefill.

Replaces the body of ``Qwen4ExpRMSNormGated.__call__``
(``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:1080-1088``)
with one Metal dispatch whenever the input is 4-D ``[B, S, HV, DV]`` with
``B * S > 1``.  Everything else, including every one-row call, falls through to
the original method, so oMLX's own ``qwen4_decode_norm_gate_fused``
(``omlx/patches/qwen35_gdn_prework.py:395``) stays the decode path.

Composition with ~/inference-server/kernels/ple-fix/norm_patch.py: no overlap.
That patch rebinds ``hc_fused.prefill_forward`` and, under
``OMLX_QWEN4_BF16_NORM_ALL=1``, ``Qwen4ExpRMSNorm.__call__`` (the *ungated*
grouped norm on the hyper-connection and PLE streams).  This patch rebinds
``Qwen4ExpRMSNormGated.__call__``, a different class on a different tensor.
Both can be installed in either order.

Enable with ``OMLX_QWEN4_GDN_NORM_GATE=1``.  Install AFTER the model is loaded:
the target class lives in the vendored ``mlx_vlm`` tree that oMLX's compat
patch puts on the import path, so the import is only reliable post-load.  The
patch itself touches the class, not instances, so it applies to every layer at
once and is idempotent.
"""
from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_APPLIED = False
_ORIGINAL_CALL = None

STATS = {"fused_calls": 0, "fallback_calls": 0, "errors": 0}

_ENV = "OMLX_QWEN4_GDN_NORM_GATE"


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _load_kernel_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
    spec = importlib.util.spec_from_file_location("gdn_norm_gate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_call(k):
    def __call__(self, x, gate):
        if not k.eligible(x, gate, self.weight, self.activation):
            STATS["fallback_calls"] += 1
            return _ORIGINAL_CALL(self, x, gate)
        try:
            out = k.norm_gate_fused(
                x,
                gate,
                self.weight,
                eps=self.eps,
                activation=k.gate_code(self.activation),
            )
            STATS["fused_calls"] += 1
            return out
        except Exception as exc:  # noqa: BLE001
            # Never change results because of a kernel-build failure.
            STATS["errors"] += 1
            logger.warning("fused GDN norm+gate declined, stock path: %s", exc)
            return _ORIGINAL_CALL(self, x, gate)

    __call__._omlx_gdn_norm_gate = True
    return __call__


def install(*, force: bool = False) -> bool:
    """Idempotent.  Returns False and leaves the stock path when unavailable."""
    global _APPLIED, _ORIGINAL_CALL
    if _APPLIED:
        return True
    if not force and not _env_flag(_ENV):
        return False
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return False
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated
    except Exception as exc:  # noqa: BLE001
        logger.warning("fused GDN norm+gate not installed: %s", exc)
        return False

    if getattr(Qwen4ExpRMSNormGated.__call__, "_omlx_gdn_norm_gate", False):
        _APPLIED = True
        return True

    try:
        k = _load_kernel_module()
        # build and validate the kernel once, off the hot path, on a tiny
        # synthetic shape; a failure here keeps the stock path.
        x = mx.random.normal((1, 4, 48, 128)).astype(mx.bfloat16)
        w = mx.random.normal((128,)).astype(mx.bfloat16)
        ref = k.norm_gate_stock(x, x, w, eps=1e-6, activation=k.GATE_SIGMOID)
        got = k.norm_gate_fused(x, x, w, eps=1e-6, activation=k.GATE_SIGMOID)
        if not bool(mx.array_equal(ref, got).item()):
            logger.warning("fused GDN norm+gate self-check mismatch, stock path kept")
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("fused GDN norm+gate self-check failed: %s", exc)
        return False

    _ORIGINAL_CALL = Qwen4ExpRMSNormGated.__call__
    Qwen4ExpRMSNormGated.__call__ = _make_call(k)
    _APPLIED = True
    logger.info("Qwen4 GDN gated norm: fused bf16 Metal kernel for T>1")
    return True


def remove() -> bool:
    global _APPLIED
    if not _APPLIED or _ORIGINAL_CALL is None:
        return False
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated

    Qwen4ExpRMSNormGated.__call__ = _ORIGINAL_CALL
    _APPLIED = False
    return True


def is_applied() -> bool:
    return _APPLIED
