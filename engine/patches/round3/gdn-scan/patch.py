# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Route Qwen4-Exp GDN prefill to PR #4020's chunked delta-rule Metal kernel.

Rebinds ``gated_delta_update`` in ``mlx_vlm.models.qwen3_5.{gated_delta,language}``
(qwen4_exp's ``Qwen4ExpGatedDeltaNet`` inherits ``Qwen3_5GatedDeltaNet.__call__``,
which resolves the symbol from that module's globals). Only the T > 1 prefill
scan with scalar per-head gating and no mask is replaced. Decode (T == 1),
verify, masked and vectorized-gating calls fall through to whatever was bound
before, which on oMLX is ``qwen35_gdn_chunked``'s ``gated_delta_blocked_seq``
wrapper, and under that the stock mlx_lm sequential kernel.

Import-time monkeypatch of module functions: no model instances are touched, so
``install()`` may run before or after the model is loaded.

Env:
  OMLX_QWEN4_GDN_SCAN=1       enable (default off)
  OMLX_QWEN4_GDN_SCAN_IMPL    simd (default, C=8) | nax (C=16, faster, looser)
  OMLX_QWEN4_GDN_SCAN_MIN_T   minimum T to engage (default 64)
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> bool:
    """Idempotent. Returns False and leaves the existing path on any precondition failure."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("OMLX_QWEN4_GDN_SCAN", "0") != "1":
        return False
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
        from mlx_vlm.models.qwen3_5 import language as lang
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlx_vlm qwen3_5 not importable; GDN scan patch skipped: %s", exc)
        return False

    impl = os.environ.get("OMLX_QWEN4_GDN_SCAN_IMPL", "simd")
    min_t = int(os.environ.get("OMLX_QWEN4_GDN_SCAN_MIN_T", "64"))

    try:
        if impl == "nax":
            from .kernel_nax import gated_delta_fused_nax as scan  # type: ignore
            from .kernel_nax import supported
        else:
            from .kernel import gated_delta_fused_chunk as scan  # type: ignore
            from .kernel import supported
    except ImportError:
        # loaded by path (importlib.spec_from_file_location), not as a package
        import importlib.util
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        mod_name = "kernel_nax" if impl == "nax" else "kernel"
        spec = importlib.util.spec_from_file_location(
            f"_omlx_gdn_{mod_name}", os.path.join(here, f"{mod_name}.py")
        )
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        scan = getattr(mod, "gated_delta_fused_nax" if impl == "nax" else "gated_delta_fused_chunk")
        supported = mod.supported

    # Smoke the JIT compile once at install time so a compile failure downgrades
    # to the existing path instead of exploding mid-prefill.
    try:
        B, T, Hk, Dk, Hv, Dv = 1, 64, 16, 128, 48, 128
        z = mx.zeros
        y, s = scan(
            z((B, T, Hk, Dk), mx.bfloat16),
            z((B, T, Hk, Dk), mx.bfloat16),
            z((B, T, Hv, Dv), mx.bfloat16),
            mx.ones((B, T, Hv), mx.float32),
            z((B, T, Hv), mx.float32),
            z((B, Hv, Dv, Dk), mx.float32),
        )
        mx.eval(y, s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PR #4020 GDN scan kernel failed to compile; not installed: %s", exc)
        return False

    original = gd.gated_delta_update

    def gated_delta_update_pr4020(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    ):
        if (
            use_kernel
            and mask is None
            and a.ndim == 3  # scalar per-head gating
            and q.shape[1] >= min_t
        ):
            g, beta = gd._compute_g_beta(A_log, a, b, dt_bias)
            if supported(q, k, v, g, beta, state):
                return scan(q, k, v, g, beta, state)
        return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)

    lang.gated_delta_update = gated_delta_update_pr4020
    gd.gated_delta_update = gated_delta_update_pr4020
    _INSTALLED = True
    logger.info("Qwen4-Exp GDN prefill scan patched to mlx PR #4020 (impl=%s, min_t=%d)", impl, min_t)
    return True
