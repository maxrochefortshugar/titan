# SPDX-License-Identifier: Apache-2.0
"""Make the packed n-gram reader's parallelism configurable.

``kernels/ple-fix/patch.py:150`` (``PackedPLETable._prefetch_pages``) touches
one 16 KB page per n-gram row through ``_PLE_IO_POOL``, the thread pool the
vendored model file creates at
``mlx_vlm/models/qwen4_exp/language.py:1921``::

    _PLE_IO_POOL = ThreadPoolExecutor(max_workers=48, thread_name_prefix="ple-io")

That 48 is a hard-coded constant, and it is the only knob over the read path:
a 2048-token chunk issues ~31,855 scattered single-page ``pread`` calls for
522 MB, which the ple-fix report measured at 456 ms cold, or 1.14 GB/s against
13.6 GB/s sequential on this SSD.  llama.cpp PR #28136 found 32 parallel
readers worth 6.6x over serialised page faults, so the count is worth a sweep.

This module swaps the module-level pool for one with ``OMLX_PLE_READ_WORKERS``
threads.  ``_prefetch_pages`` resolves ``_PLE_IO_POOL`` by import on every
call, and oMLX's own stock reader (``language.py:2031``) reads the same global,
so the swap reaches both paths without editing either file.

Install AFTER the model is loaded: the vendored module has to be imported
before its global can be replaced, and the pool is idle until the first
prefill.  Default is 48, the value in the code today, so installing without
setting the env var changes nothing.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

ENV_WORKERS = "OMLX_PLE_READ_WORKERS"
DEFAULT_WORKERS = 48

_STATE: dict = {}


def _targets():
    """Every loaded module that owns a ``_PLE_IO_POOL`` global.

    oMLX serves qwen4_exp from a vendored copy under
    ``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor``, which may be aliased into
    ``sys.modules`` under more than one name, so match on the attribute.
    """
    out = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("qwen4_exp.language"):
            continue
        if hasattr(mod, "_PLE_IO_POOL"):
            out.append((name, mod))
    return out


def workers() -> int:
    try:
        n = int(os.environ.get(ENV_WORKERS, str(DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS
    return max(1, min(n, 512))


def install(model=None) -> bool:
    """Idempotent; returns False and leaves the stock pool on any failure."""
    if _STATE.get("installed"):
        return True
    n = workers()
    targets = _targets()
    if not targets:
        logger.warning("PLE read workers: no qwen4_exp.language module loaded "
                       "(install after the model, not at import time)")
        return False
    if n == DEFAULT_WORKERS and os.environ.get(ENV_WORKERS) is None:
        logger.info("PLE read workers: left at the stock %d", DEFAULT_WORKERS)
        _STATE["installed"] = True
        _STATE["workers"] = DEFAULT_WORKERS
        return True
    pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ple-io")
    old = []
    for name, mod in targets:
        old.append((mod, mod._PLE_IO_POOL))
        mod._PLE_IO_POOL = pool
    _STATE.update(installed=True, workers=n, pool=pool, old=old)
    logger.info("PLE read workers: %d (was %d) across %d module(s)",
                n, DEFAULT_WORKERS, len(targets))
    return True


def uninstall() -> bool:
    old = _STATE.pop("old", None)
    if not old:
        return False
    for mod, pool in old:
        mod._PLE_IO_POOL = pool
    p = _STATE.pop("pool", None)
    if p is not None:
        p.shutdown(wait=False)
    _STATE.pop("installed", None)
    return True


__all__ = ["install", "uninstall", "workers", "DEFAULT_WORKERS", "ENV_WORKERS"]
