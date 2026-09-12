#!/usr/bin/env python3
"""oMLX integration for the fused prefill hyper-connection block.

``install()`` rebinds ``mlx_vlm.models.qwen4_exp.hc_fused.prefill_forward``,
the function ``Qwen4ExpGatedResidual.__call__`` takes for every call above 16
rows (language.py:1669-1671).  It touches a module function, not instances, so
it can run at import time or post-load; the bootstrap's post-load hook is the
simpler place because mlx_vlm is certainly imported by then.

Composition with kernels/ple-fix/norm_patch.py: this patch subsumes it.  The
fused block calls the same ``hc_fused._kernel_norm`` bf16 grouped norm, so
``OMLX_QWEN4_BF16_NORM=1`` becomes a no-op for the hyper-connections (it still
governs ``OMLX_QWEN4_BF16_NORM_ALL``, the three PLE grouped norms, which this
patch does not touch).  The installed function carries the
``_omlx_bf16_norm`` marker that norm_patch checks, so norm_patch treats it as
already applied and will not overwrite it whichever order the two hooks run
in.

    OMLX_QWEN4_HC_FUSE2=1
    OMLX_ROUND2_PATCHES="~/inference-server/kernels/round3/hc-fuse/patch.py"
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_APPLIED = False
_ORIGINAL = None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _load_kernel():
    if "hc_fuse2_kernel" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "hc_fuse2_kernel", _HERE / "kernel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hc_fuse2_kernel"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["hc_fuse2_kernel"]


def install(model=None, *, force: bool = False) -> bool:
    """Idempotent. False (and the stock path) when the gate is off or unusable."""
    global _APPLIED, _ORIGINAL
    if not force and not _env_flag("OMLX_QWEN4_HC_FUSE2"):
        return False
    if _APPLIED:
        return True
    try:
        from mlx_vlm.models.qwen4_exp import hc_fused
    except Exception as exc:  # noqa: BLE001
        logger.warning("hc-fuse2: qwen4_exp not present (%s), stock path kept", exc)
        return False
    if not hc_fused.enabled():
        logger.info("hc-fuse2: hc_fused disabled, stock path kept")
        return False
    if getattr(hc_fused.prefill_forward, "_omlx_hc_fuse2", False):
        _APPLIED = True
        return True

    kernel = _load_kernel()

    def prefill_forward(module, hyper_input):
        return kernel.fused_prefill_forward(module, hyper_input)

    prefill_forward._omlx_hc_fuse2 = True
    # norm_patch.py checks this marker and will not replace us.
    prefill_forward._omlx_bf16_norm = True
    _ORIGINAL = hc_fused.prefill_forward
    hc_fused.prefill_forward = prefill_forward
    _APPLIED = True
    logger.info("hc-fuse2: fused prefill hyper-connection installed")
    return True


def remove() -> bool:
    global _APPLIED
    if not _APPLIED or _ORIGINAL is None:
        return False
    from mlx_vlm.models.qwen4_exp import hc_fused

    hc_fused.prefill_forward = _ORIGINAL
    _APPLIED = False
    return True


def stats():
    return dict(_load_kernel().STATS)


def is_applied() -> bool:
    return _APPLIED
