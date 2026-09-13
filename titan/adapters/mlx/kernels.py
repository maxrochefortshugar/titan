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
which is what makes bisecting a numeric change a matter of one line in the
``kernels`` section.

``ADAPTER_OPS`` maps the names the call sites use to the names
``titan.kernels`` registers. The adapter's names are dotted and describe the
site; the registry's are flat and describe the kernel. Keeping the two apart
means a kernel can be renamed, aliased or split without touching vendored code.
"""

from __future__ import annotations

import logging
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
    # The batched gathered-QSA arm. Wired: ``Qwen4ExpAttention`` builds the
    # pooled index bank in phased slot space on the ``BatchQSAKVCache`` and
    # calls this for batched decode and batched verify. ``None`` here (which
    # ``kernels.reference_only`` and ``kernels.disabled`` both produce) sends
    # batched rows back to the dense masked path.
    "qsa.gathered_batched": "qsa_gathered_attention",
    # The narrow per-kernel seams inside qsa_fast.py. oMLX filled these from
    # its own extension; Titan has no equivalent yet, so they stay on MLX.
    "qsa.indexer_scores": None,
    "qsa.topk_indices": None,
    "qsa.sparse_gqa": None,
    "qsa.decode_sdpa": None,
    "ple.packed_rows_prefetch": None,
}

# There is no environment switch here. Turning an op off is
# ``kernels.disabled`` in the config file and turning them all off is
# ``kernels.reference_only``; both reach this module through the registry the
# wiring publishes, which is the same path the server takes, so a control arm
# measured here is the control arm the server would run. The two ``TITAN_*``
# variables this module used to read were a second, quieter policy on top of
# that one, and a bisect with two policies is a bisect that cannot say which
# one produced a number.

_lock = threading.Lock()
_registry: Any = None
_resolved: dict[str, Optional[Callable[..., Any]]] = {}
_build_failed = False


def _get_registry() -> Any:
    """The process's configured registry, or a default one, built once.

    ``titan.config.wiring`` builds a registry from the ``kernels`` section at
    startup and publishes it (``titan.kernels.registry.set_current``). It never
    reached here before, so ``kernels.reference_only`` and ``kernels.disabled``
    changed nothing inside the forward: the adapter built its own registry with
    default settings and used every fast path regardless. That made the control
    arm of a kernel A/B not a control, which is the arm INCIDENTS rule 2 asks
    for by name. Preferring the published one fixes it without the adapter
    having to be handed anything.

    A failure is remembered, not retried per call.
    """
    global _registry, _build_failed
    if _registry is not None or _build_failed:
        return _registry
    with _lock:
        if _registry is not None or _build_failed:
            return _registry
        try:
            from titan.kernels import registry as registry_module

            _registry = registry_module.current() or registry_module.build_registry()
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
        else:
            try:
                # A configuration that turned this op off means "this site has
                # no kernel", not "this site runs the op's reference". The two
                # differ: the reference is the registry's own MLX call
                # sequence, which for the gathered-QSA op is still the sparse
                # algorithm, while the stock path is what the vendored code
                # does without any kernel at all. ``kernels.reference_only``
                # asks for the second, so that is what ``None`` here gives.
                if registry.fast_disabled(registry_name):
                    op = None
                else:
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


# ---------------------------------------------------------------------------
# Arm routing lives in ``models/forward_paths.py``
# ---------------------------------------------------------------------------
#
# ROUND3 added four routes here -- ``qsa_sparse_singleton_verify``,
# ``qsa_batched_sparse`` and the two gather crossovers -- because
# ``forward_paths`` belonged to another workstream's files at the time, and
# recorded that they should be folded in by whoever owned it next. ROUND4 did
# that. They are arm switches for the vendored attention, the same kind of
# thing as ``batched_verify_attention``, and there is no reason for the
# attention switchboard to be in two files. Read them with
# ``forward_paths.route(name)`` and pair them with
# ``forward_paths.overridden(...)``, which now takes routes as well as paths.

# ---------------------------------------------------------------------------
# The gathered-QSA value objects
# ---------------------------------------------------------------------------
#
# ``qsa_gathered_attention`` takes the head geometry and the batch's padding as
# value objects rather than reading them off a cache, which is what keeps the
# op free of the model. The vendored attention builds them here so it does not
# import ``titan.kernels`` itself; the door stays one module wide.


def qsa_config(**fields: Any):
    """A :class:`titan.kernels.qsa_gathered_attention.QSAConfig`."""
    from titan.kernels.qsa_gathered_attention import QSAConfig

    return QSAConfig(**fields)


def qsa_geometry(width: int, pads, compress_ratio: int):
    """A :class:`titan.kernels.qsa_gathered_attention.QSAGeometry`."""
    from titan.kernels.qsa_gathered_attention import QSAGeometry

    return QSAGeometry(width, pads, compress_ratio)


def qsa_pool_slots(*args: Any, **kwargs: Any):
    """``qsa_gathered_attention.pool_slots``: fresh slots of the phased bank."""
    from titan.kernels.qsa_gathered_attention import pool_slots

    return pool_slots(*args, **kwargs)
