#!/usr/bin/env python3
"""Integration contract: install() is idempotent, gated, and composes with the
deployed packed patch without editing it.

There is no model here (loading it is forbidden), so this exercises the two
things install() actually does: rebind the class the deployed patch
instantiates, and promote tables that already exist.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stub_io_pool import install_stub  # noqa: E402

install_stub()

MODEL = os.environ.get(
    "PROFILE_MODEL",
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")


def main():
    results = {}

    os.environ.pop("OMLX_PLE_LRU", None)
    import patch as lru
    importlib.reload(lru)
    results["off_returns_false"] = lru.install() is False

    os.environ["OMLX_PLE_LRU"] = "1"
    importlib.reload(lru)
    results["on_returns_true"] = lru.install() is True
    base = lru.load_base_module()
    # install() rebinds the class on its own cached copy of the base module
    module = lru._STATE["base"]
    results["class_rebound"] = getattr(module.PackedPLETable, "_omlx_ple_lru", False) is True
    results["idempotent"] = lru.install() is True
    results["still_one_class"] = module.PackedPLETable is lru._STATE["cls"]

    # a table built the way the deployed patch builds it now comes out cached
    directory, manifest = module.load_manifest(MODEL)
    entry = manifest["layers"]["1"]
    table = module.PackedPLETable(directory, entry, "rows")
    results["new_table_is_cached_class"] = isinstance(table, lru.CachedPLETable)
    module._APPLIED[1] = table
    lru.install()
    results["existing_table_promoted"] = getattr(table, "_lru", None) is not None
    stats = table.cache_stats()
    results["budget_gb"] = stats["gb_reserved"]
    results["ways"] = stats["ways"]
    results["reader_default"] = stats["reader"]

    # capacity clamp
    os.environ["OMLX_PLE_LRU_GB"] = "64"
    results["clamped_to_4gb"] = lru.budget_bytes() == 4 * (1 << 30)
    os.environ["OMLX_PLE_LRU_GB"] = "0.5"
    results["respects_half_gb"] = lru.budget_bytes() == (1 << 29)
    os.environ["OMLX_PLE_LRU_GB"] = "2"

    # uninstall puts the stock class back
    lru.uninstall()
    results["uninstall_restores_class"] = module.PackedPLETable is lru._STATE["original_cls"]
    table.close()

    results["all_ok"] = all(v is True for k, v in results.items()
                            if isinstance(v, bool))
    print(json.dumps(results, indent=1))
    (HERE / "test_install.json").write_text(json.dumps(results, indent=1))
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
