# SPDX-License-Identifier: Apache-2.0
"""Integration entry point: ``install() -> bool``.

``prod/bootstrap.py`` loads this file by path (importlib.spec_from_file_location),
so nothing here may depend on the module being importable as ``patch``.
The kernel itself lives next to this file in ``kernel.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_NAME = "omlx_wsum_topk10_kernel"


def _load():
    mod = sys.modules.get(_MODULE_NAME)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, os.path.join(_HERE, "kernel.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def install() -> bool:
    """Idempotent. False (and the stock path untouched) when preconditions fail."""
    try:
        return bool(_load().install())
    except Exception:
        return False


__all__ = ["install"]
