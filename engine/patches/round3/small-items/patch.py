# SPDX-License-Identifier: Apache-2.0
"""Round-3 small items: four independent installs, one env flag each.

``prod/bootstrap.py`` loads this file by path (importlib.spec_from_file_location),
so nothing here may depend on the module being importable as ``patch``.

| install                       | env flag                        | when |
|-------------------------------|---------------------------------|------|
| ``install_mtp_fasttopk()``    | ``OMLX_MTP_SHORTLIST_FASTTOPK`` | import time, AFTER round2/mtp |
| ``install_ple_read_workers()``| ``OMLX_PLE_READ_WORKERS``       | after load |
| ``install_wsum_verify()``     | ``OMLX_WSUM_TOPK10_VERIFY``     | after load, after round2/wsum10 |
| ``install_moe_int8_down()``   | ``OMLX_MOE_INT8_DOWN``          | after load (needs the model) |

Each one is idempotent and returns False, leaving the stock path in place,
when its flag is off or its preconditions fail.  They share no state.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MOE_INT8_DIR = os.path.expanduser("~/inference-server/kernels/moe-int8")

ENV_DOWN = "OMLX_MOE_INT8_DOWN"
ENV_DOWN_SOFT_GB = "OMLX_MOE_INT8_DOWN_SOFT_GB"     # refuse above this (default 93.5)


# ------------------------------------------------------------------ item 1
def install_mtp_fasttopk() -> bool:
    """Fused Metal top-K for the MTP shortlist drafter (import time)."""
    try:
        import mtp_fasttopk
        return bool(mtp_fasttopk.install())
    except Exception:                                        # noqa: BLE001
        logger.warning("fused top-K install failed", exc_info=True)
        return False


# ------------------------------------------------------------------ item 2
def install_ple_read_workers(model=None) -> bool:
    """Configurable thread count for the packed n-gram reader (after load)."""
    try:
        import ple_workers
        return bool(ple_workers.install(model))
    except Exception:                                        # noqa: BLE001
        logger.warning("PLE read workers install failed", exc_info=True)
        return False


# ------------------------------------------------------------------ item 3
def install_wsum_verify() -> bool:
    """top_k=10 weighted-sum kernel for the MTP verify layout."""
    try:
        import wsum_verify
        return bool(wsum_verify.install())
    except Exception:                                        # noqa: BLE001
        logger.warning("MTP verify weighted-sum install failed", exc_info=True)
        return False


# ------------------------------------------------------------------ item 4
def _moe_int8():
    for name, rel in (("moe_int8_kernel", "kernel.py"), ("moe_int8_patch", "patch.py")):
        if name in sys.modules:
            continue
        if MOE_INT8_DIR not in sys.path:
            sys.path.insert(0, MOE_INT8_DIR)
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(MOE_INT8_DIR, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["moe_int8_kernel"], sys.modules["moe_int8_patch"]


def down_table_bytes(model) -> int:
    """Bytes the down-projection tables will add, from the kernel's own sizes."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    k, _ = _moe_int8()
    total = 0
    for _, m in model.named_modules():
        if not isinstance(m, QuantizedSwitchLinear):
            continue
        if (m.bits, m.group_size, getattr(m, "mode", "affine")) != (4, 64, "affine"):
            continue
        if k.pick_cfg(m["scales"].shape[1]) is None:
            continue
        w = m["weight"]
        kdim = w.shape[-1] * 32 // m.bits          # uint32 words -> input width
        if kdim > 1024:                            # gate_up: already resident
            continue
        e, n, _ = w.shape
        g = kdim // m.group_size
        total += 3 * e * g * n * 2                 # scales, folded bias, qsum
    return total


def install_moe_int8_down(model=None) -> bool:
    """Let the int8 gather kernel take the down projections too (after load).

    Refuses, and leaves ``OMLX_MOE_INT8_SKIP_DOWN`` alone, when the projected
    resident total would cross the scheduler's soft memory limit: crossing it
    is what serialised concurrent requests before the guard was raised
    (kernels/REPORT.md, 2026-09-12 12:40).
    """
    if os.environ.get(ENV_DOWN, "0") != "1":
        return False
    if os.environ.get("OMLX_MOE_INT8_PREFILL", "0") != "1":
        logger.warning("int8 down projections: OMLX_MOE_INT8_PREFILL is off")
        return False
    if model is None:
        logger.warning("int8 down projections: needs the loaded model")
        return False
    try:
        import mlx.core as mx

        need = down_table_bytes(model)
        soft = float(os.environ.get(ENV_DOWN_SOFT_GB, "93.5")) * 1e9
        active = float(mx.get_active_memory())
        if active + need > soft:
            logger.warning("int8 down projections skipped: %.1f GB active + "
                           "%.2f GB of tables crosses the %.1f GB soft limit",
                           active / 1e9, need / 1e9, soft / 1e9)
            return False
        os.environ["OMLX_MOE_INT8_SKIP_DOWN"] = "0"
        k, _ = _moe_int8()
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear

        built = 0
        for _, m in model.named_modules():
            if not isinstance(m, QuantizedSwitchLinear):
                continue
            if (m.bits, m.group_size, getattr(m, "mode", "affine")) != (4, 64, "affine"):
                continue
            if k.pick_cfg(m["scales"].shape[1]) is None:
                continue
            if m["weight"].shape[-1] * 32 // m.bits > 1024:
                continue
            k._prepared(m["weight"], m["scales"], m["biases"])
            built += 1
        logger.info("int8 down projections enabled: %d tensors, %.2f GB of "
                    "tables, %.1f GB active after", built, need / 1e9,
                    float(mx.get_active_memory()) / 1e9)
        return built > 0
    except Exception:                                        # noqa: BLE001
        logger.warning("int8 down projections install failed", exc_info=True)
        return False


# ------------------------------------------------------------------ combined
def install_import_time() -> bool:
    """The one import-time item."""
    return install_mtp_fasttopk()


def install(model=None) -> bool:
    """All three post-load items; True when at least one installed."""
    done = install_wsum_verify()
    done |= install_ple_read_workers(model)
    done |= install_moe_int8_down(model)
    return done


__all__ = [
    "install",
    "install_import_time",
    "install_mtp_fasttopk",
    "install_ple_read_workers",
    "install_wsum_verify",
    "install_moe_int8_down",
    "down_table_bytes",
]
