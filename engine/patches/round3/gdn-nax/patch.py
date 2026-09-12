# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Route Qwen4-Exp GDN prefill to the split-precision NAX (C=16) kernel.

Same hook point as round3/gdn-scan/patch.py: rebinds ``gated_delta_update`` in
``mlx_vlm.models.qwen3_5.{gated_delta,language}``, replacing only the unmasked
T > 1 prefill scan with scalar per-head gating. Decode, verify, masked and
vectorized-gating calls fall through to whatever was bound before.

Module-level monkeypatch, no instances touched, so install() may run at import
time or post-load. Bootstrap it post-load at the same point as gdn-scan.

Precedence: when both OMLX_QWEN4_GDN_SCAN_NAX and OMLX_QWEN4_GDN_SCAN are set,
NAX wins. This install() clears OMLX_QWEN4_GDN_SCAN in the process environment
after it binds, so a gdn-scan install() running later returns False and leaves
this kernel in place; a gdn-scan install() that already ran is simply wrapped.

Env:
  OMLX_QWEN4_GDN_SCAN_NAX=1        enable (default off)
  OMLX_QWEN4_GDN_SCAN_NAX_MIN_T    minimum T to engage (default 64)
  OMLX_QWEN4_GDN_NAX_SPLIT         split-precision site mask (default 0x1DE)
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_INSTALLED = False


def _load_kernel_module():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        from . import kernel_nax2 as mod  # type: ignore

        return mod
    except ImportError:
        pass
    import importlib.util
    import sys

    # kernel_nax2 imports naxhdr as a top-level module
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(
        "_omlx_gdn_kernel_nax2", os.path.join(here, "kernel_nax2.py")
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def install() -> bool:
    """Idempotent. Returns False and leaves the existing path on any precondition failure."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("OMLX_QWEN4_GDN_SCAN_NAX", "0") != "1":
        return False
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
        from mlx_vlm.models.qwen3_5 import language as lang
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlx_vlm qwen3_5 not importable; GDN NAX patch skipped: %s", exc)
        return False

    mod = _load_kernel_module()
    if mod is None:
        return False
    scan = mod.gated_delta_fused_nax2
    supported = mod.supported
    split = mod.DEFAULT_SPLIT
    min_t = int(os.environ.get("OMLX_QWEN4_GDN_SCAN_NAX_MIN_T", "64"))

    # Smoke the JIT compile once so a compile or NAX-availability failure
    # downgrades to the existing path instead of exploding mid-prefill.
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
        logger.warning("GDN NAX kernel failed to compile; not installed: %s", exc)
        return False

    original = gd.gated_delta_update

    def gated_delta_update_nax(
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

    lang.gated_delta_update = gated_delta_update_nax
    gd.gated_delta_update = gated_delta_update_nax
    if os.environ.get("OMLX_QWEN4_GDN_SCAN") == "1":
        os.environ["OMLX_QWEN4_GDN_SCAN"] = "0"
        logger.info("OMLX_QWEN4_GDN_SCAN disabled: the NAX kernel takes precedence")
    _INSTALLED = True
    logger.info(
        "Qwen4-Exp GDN prefill scan patched to split-precision NAX C=16 "
        "(split=%#05x, min_t=%d)",
        split,
        min_t,
    )
    return True
