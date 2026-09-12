"""The adapter's one door to ``titan.kernels``.

The vendored model code never imports a kernel. It asks here, by name, and uses
what comes back only if something comes back:

    op = get("gdn.norm_gate_fused")
    if op is not None:
        out = op(x, gate, w, eps=eps, activation=0)

Two properties follow, and both are deliberate. There is no monkeypatching: a
call site that wants a kernel says so in its own body, in the open. And the
adapter is correct without the registry -- every site keeps the stock MLX path,
so this module returning ``None`` for everything is a supported configuration,
which is what makes bisecting a numeric change a matter of one env var.

``ADAPTER_OPS`` maps the names the call sites use to the names
``titan.kernels`` registers. The adapter's names are dotted and describe the
site; the registry's are flat and describe the kernel. Keeping the two apart
means a kernel can be renamed, aliased or split without touching vendored code.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: adapter call-site name -> ``titan.kernels`` op name, or ``None`` when the
#: registry has nothing for that site yet and the stock MLX path stands.
ADAPTER_OPS: dict[str, str | None] = {
    "norm.grouped_rms_bf16": "grouped_rmsnorm_bf16",
    "gdn.norm_gate_fused": "gdn_norm_gate",
    "gdn.chunk_scan": "gdn_chunk_scan",
    "moe.weighted_sum": "moe_weighted_sum",
    "moe.gather_qmm_ws": "moe_gather_ws",
    "moe.gather_qmm_int8": "moe_gather_int8",
    "hc.prefill_block_fused": "hc_prefill",
    "ple.packed_rows_lookup": "ple_packed_lookup",
    "sample.fast_topk": "topk_radix",
    # The batched gathered-QSA arm. The kernel exists
    # (``qsa_gathered_attention``) but it replaces the whole attention block
    # and needs the pooled index bank in phased slot space, which the vendored
    # attention does not build. Wiring it is the one remaining hook; until
    # then batched decode takes the dense path and is merely slow, not wrong.
    "qsa.gathered_batched": None,
    # The narrow per-kernel seams inside qsa_fast.py. oMLX filled these from
    # its own extension; Titan has no equivalent yet, so they stay on MLX.
    "qsa.indexer_scores": None,
    "qsa.topk_indices": None,
    "qsa.sparse_gqa": None,
    "qsa.decode_sdpa": None,
    "ple.packed_rows_prefetch": None,
}

_DISABLED = {
    name.strip()
    for name in os.environ.get("TITAN_DISABLE_OPS", "").split(",")
    if name.strip()
}
_ALL_DISABLED = os.environ.get("TITAN_REFERENCE_ONLY", "") == "1"

_lock = threading.Lock()
_registry: Any = None
_resolved: dict[str, Optional[Callable[..., Any]]] = {}
_build_failed = False


def _get_registry() -> Any:
    """Build the registry once. A failure is remembered, not retried per call."""
    global _registry, _build_failed
    if _registry is not None or _build_failed:
        return _registry
    with _lock:
        if _registry is not None or _build_failed:
            return _registry
        try:
            from titan.kernels import registry as registry_module

            _registry = registry_module.build_registry()
        except Exception as exc:  # noqa: BLE001 - the adapter runs without it
            _build_failed = True
            logger.info(
                "titan.kernels is unavailable (%s); the model runs on the "
                "stock MLX paths",
                exc,
            )
        return _registry


def registry_available() -> bool:
    return _get_registry() is not None


def reset() -> None:
    """Forget the registry and every resolution. Tests and bisects use this."""
    global _registry, _build_failed
    with _lock:
        _registry = None
        _build_failed = False
        _resolved.clear()


def get(name: str) -> Optional[Callable[..., Any]]:
    """The callable for an adapter op name, or ``None``.

    ``None`` is never an error. It means "no kernel for this site", and the
    site keeps its stock path.
    """
    if _ALL_DISABLED or name in _DISABLED:
        return None
    if name in _resolved:
        return _resolved[name]
    registry = _get_registry()
    op: Optional[Callable[..., Any]] = None
    if registry is not None:
        if name not in ADAPTER_OPS:
            logger.warning("unknown adapter op %s", name)
            registry_name = None
        else:
            registry_name = ADAPTER_OPS[name]
        if registry_name is None:
            op = None
        elif registry_name in _DISABLED:
            op = None
        else:
            try:
                op = registry.resolve(registry_name)
            except Exception as exc:  # noqa: BLE001 - a missing op is normal
                logger.debug("op %s unavailable: %s", registry_name, exc)
                op = None
    _resolved[name] = op
    return op


def has(name: str) -> bool:
    return get(name) is not None


def wired() -> dict[str, bool]:
    """Which call sites currently have a kernel behind them."""
    return {name: has(name) for name in ADAPTER_OPS}


# ---------------------------------------------------------------------------
# Glue the vendored modules cannot carry themselves
# ---------------------------------------------------------------------------


def hc_weights(module):
    """Build the ``HCWeights`` the fused hyper-connection block wants.

    The kernel takes a flat description of the block's three projections rather
    than the module, so this translates once per module and caches the result on
    it. Returns ``None`` when the block is not the affine-quantised shape the
    kernel supports, which sends the call site back to the MLX path.
    """
    cached = getattr(module, "_titan_hc_weights", None)
    if cached is not None:
        return cached or None
    try:
        from titan.kernels.hc_prefill import HCWeights, QuantLinear
    except Exception:
        module._titan_hc_weights = False
        return None

    def quant(projection):
        if projection is None:
            return None
        weight = projection.get("weight") if hasattr(projection, "get") else None
        if weight is None or weight.dtype.__str__() != "uint32":
            raise TypeError("not an affine-quantized projection")
        return QuantLinear(
            weight=projection["weight"],
            scales=projection["scales"],
            biases=projection["biases"],
            bits=int(projection.bits),
            group_size=int(projection.group_size),
        )

    try:
        weights = HCWeights(
            norm_weight=module.hc_norm.weight,
            down=quant(module.input_mix_weight_down),
            up=quant(module.input_mix_weight_up),
            hc_count=int(module.hc_count),
            hidden_size=int(module.hidden_size),
            eps=float(module.hc_norm.eps),
            inject=quant(
                module.block_inject_weight
                if "block_inject_weight" in module
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - shape mismatch is a fallback
        logger.debug("hyper-connection block not eligible for the kernel: %s", exc)
        module._titan_hc_weights = False
        return None
    module._titan_hc_weights = weights
    return weights


_ple_tables: dict[str, Any] = {}


def ple_table(prefix: str, model_path):
    """The packed n-gram table for a layer, or ``None`` if none is built.

    The kernel takes an open table object rather than a path, and the reader is
    ``titan/adapters/mlx/ngram.py``'s job once that lands; until it does, this
    opens the pack directly from the checkpoint's ``ple-packed/manifest.json``.
    Tables are cached per prefix: each one memory-maps a 32 GB file.
    """
    if prefix in _ple_tables:
        return _ple_tables[prefix] or None
    table = None
    try:
        from titan.adapters.mlx.ngram import packed_table  # type: ignore

        table = packed_table(prefix, model_path)
    except Exception:
        table = None
    if table is None:
        table = _open_pack(prefix, model_path)
    _ple_tables[prefix] = table or False
    return table


def _open_pack(prefix: str, model_path):
    from pathlib import Path

    from .checkpoint import ple_manifest

    try:
        from titan.kernels.ple_packed_lookup import LayerEntry, PackedRowTable
    except Exception:
        return None
    manifest = ple_manifest(Path(model_path))
    if not manifest:
        return None
    for layer_id, entry in (manifest.get("layers") or {}).items():
        if entry.get("prefix") and prefix.endswith(entry["prefix"].split(".", 2)[-1]):
            directory = Path(model_path) / "ple-packed"
            try:
                return PackedRowTable(directory, LayerEntry.from_manifest(entry))
            except Exception as exc:  # noqa: BLE001
                logger.warning("packed n-gram table for layer %s unusable: %s",
                               layer_id, exc)
                return None
    return None
