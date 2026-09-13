"""Weight loading for Qwen3.8-Flash-Next: plan first, then a lazy mmap load.

The loader is split in two on purpose.

``build_plan`` reads nothing but the safetensors headers and ``config.json``.
It decides, for every tensor in the checkpoint, exactly one of: load it at a
module parameter path, fuse it with a sibling, or drop it -- and it says why.
That makes the mapping testable without a machine that can hold the model:
``tests/model/test_loader_plan.py`` asserts the plan covers all 3748 tensors
exactly once.

``load_weights`` executes a plan.  It maps each shard with ``mx.load`` (lazy,
memory mapped), applies the plan's transforms, and evaluates in small batches so
peak memory stays near the model size rather than twice it.

Two transforms are worth naming, because they are where a loader usually goes
wrong on this checkpoint:

*Fused gate_up.*  oMLX loads ``gate_proj`` and ``up_proj`` separately and fuses
them afterwards (``omlx/patches/qwen35_moe_gate_up.py``), which needs both
copies resident at once.  Titan produces the fused layout directly:
``concatenate([gate, up], axis=1)`` over ``weight``, ``scales`` and ``biases``
alike, giving ``[512, 1280, ...]`` per MoE layer with gate rows first.  The
concatenation is exact because affine quantisation groups run along the input
axis, so every output row carries its own scale and bias.

*Names.*  The checkpoint is already in runtime spelling
(``language_model.model.layers.N...``), so ``sanitize_key`` is close to a no-op
here, but the plan applies the full chain anyway -- key rewrite, ``shard_N`` to
``shards.N``, the ``conv1d`` transpose guard -- so a differently exported
checkpoint of the same model still lands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import TensorMeta, checkpoint_tensors, load_config
from .vendor.mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key
from .vendor.mlx_vlm.models.qwen4_exp.config import ModelConfig
from .vendor.mlx_vlm.models.qwen4_exp.qwen4_exp import Model

logger = logging.getLogger(__name__)

NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")
NGRAM_STORAGE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\."
    r"(?:shard_\d+|shards\.\d+)\.(?:weight|scales|biases)$"
)
MTP_PREFIXES = (
    "model.language_model.mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "mtp.",
)
QUANT_SUFFIXES = ("weight", "scales", "biases")

LOAD = "load"
FUSE = "fuse"
DROP = "drop"


@dataclass(frozen=True)
class PlanEntry:
    """One decision about one or two checkpoint tensors."""

    action: str
    sources: tuple[str, ...]
    target: Optional[str]
    shape: Optional[tuple[int, ...]]
    dtype: Optional[str]
    reason: str = ""

    @property
    def suffix(self) -> Optional[str]:
        if self.target is None:
            return None
        return self.target.rsplit(".", 1)[-1]


@dataclass
class LoaderPlan:
    """Every checkpoint tensor, assigned exactly once."""

    model_dir: Path
    entries: list[PlanEntry] = field(default_factory=list)
    quant: dict[str, dict] = field(default_factory=dict)
    mtp_enabled: bool = False
    fuse_gate_up: bool = True

    # -- coverage ----------------------------------------------------------
    def sources(self) -> list[str]:
        out: list[str] = []
        for entry in self.entries:
            out.extend(entry.sources)
        return out

    def targets(self) -> dict[str, PlanEntry]:
        return {e.target: e for e in self.entries if e.target is not None}

    def by_action(self, action: str) -> list[PlanEntry]:
        return [e for e in self.entries if e.action == action]

    # -- quantisation ------------------------------------------------------
    def quant_spec(self, module_path: str) -> Optional[dict]:
        """``{"group_size":..,"bits":..}`` if this module is quantised.

        Driven by the checkpoint, not by the config: a module is quantised iff
        the plan produces a ``.scales`` tensor for it.  The bit width comes from
        the per-tensor override in ``config.json`` when there is one, and from
        the global quantisation block otherwise.
        """
        return self.quant.get(module_path)


def _module_path(target: str) -> str:
    return target.rsplit(".", 1)[0]


def _quant_override(config: dict, module_path: str) -> Optional[dict]:
    """Per-tensor quantisation override, tried under every spelling.

    ``config.json`` writes these keys with the ``language_model.`` root; oMLX
    expands them at load time (``expand_per_layer_quant_keys``).  Titan looks
    the variants up instead of rewriting the config.
    """
    block = config.get("quantization") or {}
    candidates = [module_path]
    if module_path.startswith("language_model."):
        candidates.append(module_path[len("language_model.") :])
    else:
        candidates.append("language_model." + module_path)
    if module_path.startswith("model.language_model."):
        candidates.append("language_model.model." + module_path[len("model.language_model.") :])
    for candidate in candidates:
        value = block.get(candidate)
        if isinstance(value, dict):
            return {"group_size": value["group_size"], "bits": value["bits"]}
    return None


def _runtime_name(name: str) -> str:
    """The module parameter path a checkpoint tensor binds to."""
    name = sanitize_key(name)
    return NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", name)


def _mtp_normalised(name: str) -> Optional[str]:
    for prefix in MTP_PREFIXES:
        if name.startswith(prefix):
            return "mtp." + name[len(prefix) :]
    return None


def build_plan(
    model_dir,
    *,
    mtp_enabled: bool = True,
    fuse_gate_up: bool = True,
    tensors: Optional[dict[str, TensorMeta]] = None,
    config: Optional[dict] = None,
) -> LoaderPlan:
    """Decide where every tensor in the checkpoint goes. Reads headers only."""
    model_dir = Path(model_dir)
    config = load_config(model_dir) if config is None else config
    tensors = checkpoint_tensors(model_dir) if tensors is None else tensors
    text_config = config.get("text_config") or {}
    global_quant = config.get("quantization") or {}
    default_quant = {
        "group_size": global_quant.get("group_size", 64),
        "bits": global_quant.get("bits", 4),
    }

    plan = LoaderPlan(
        model_dir=model_dir,
        mtp_enabled=mtp_enabled,
        fuse_gate_up=fuse_gate_up,
    )
    tie = bool(text_config.get("tie_word_embeddings", False))

    # Pass 1: classify, and collect the MoE pairs that will be fused.
    gate_up: dict[tuple[str, str], dict[str, TensorMeta]] = {}
    straight: list[TensorMeta] = []

    for name in sorted(tensors):
        meta = tensors[name]
        if name.startswith("vision_tower.") or ".visual." in name:
            plan.entries.append(
                PlanEntry(DROP, (name,), None, meta.shape, meta.dtype,
                          "vision tower: Titan runs the language path only")
            )
            continue
        if NGRAM_STORAGE_RE.search(name):
            plan.entries.append(
                PlanEntry(DROP, (name,), None, meta.shape, meta.dtype,
                          "PLE n-gram shard: read from SSD by the packed rows "
                          "reader, never resident")
            )
            continue
        mtp_name = _mtp_normalised(name)
        if mtp_name is not None:
            if not mtp_enabled:
                plan.entries.append(
                    PlanEntry(DROP, (name,), None, meta.shape, meta.dtype,
                              "MTP head disabled for this load")
                )
                continue
            target_source = mtp_name
        else:
            target_source = name
        if tie and target_source in ("lm_head.weight", "language_model.lm_head.weight"):
            plan.entries.append(
                PlanEntry(DROP, (name,), None, meta.shape, meta.dtype,
                          "tied word embeddings: lm_head comes from embed_tokens")
            )
            continue

        target = _runtime_name(target_source)
        fuse_match = re.match(r"^(.*\.switch_mlp)\.(gate_proj|up_proj)\.(\w+)$", target)
        if fuse_gate_up and fuse_match:
            prefix, which, suffix = fuse_match.groups()
            gate_up.setdefault((prefix, suffix), {})[which] = meta
            continue
        straight.append(meta)

    # Pass 2: straight loads.
    for meta in straight:
        mtp_name = _mtp_normalised(meta.name)
        target = _runtime_name(mtp_name if mtp_name is not None else meta.name)
        shape = meta.shape
        reason = ""
        if "conv1d.weight" in target and shape[-1] != 1:
            shape = (shape[0], shape[2], shape[1])
            reason = "conv1d transposed from HF (C,1,K) to MLX (C,K,1)"
        plan.entries.append(
            PlanEntry(LOAD, (meta.name,), target, shape, meta.dtype, reason)
        )

    # Pass 3: fused gate_up.
    for (prefix, suffix), parts in sorted(gate_up.items()):
        missing = {"gate_proj", "up_proj"} - set(parts)
        if missing:
            raise ValueError(f"{prefix}: cannot fuse, missing {sorted(missing)}")
        gate, up = parts["gate_proj"], parts["up_proj"]
        if gate.dtype != up.dtype or gate.shape[0] != up.shape[0]:
            raise ValueError(f"{prefix}: gate/up mismatch {gate.shape} {up.shape}")
        shape = (gate.shape[0], gate.shape[1] + up.shape[1], *gate.shape[2:])
        plan.entries.append(
            PlanEntry(
                FUSE,
                (gate.name, up.name),
                f"{prefix}.gate_up_proj.{suffix}",
                shape,
                gate.dtype,
                "routed experts fused [gate | up] on the output axis",
            )
        )

    # Pass 4: quantisation, from the tensors the plan actually produces.
    produced = plan.targets()
    for target in produced:
        if not target.endswith(".scales"):
            continue
        module_path = _module_path(target)
        spec = _quant_override(config, module_path)
        if spec is None and module_path.endswith(".gate_up_proj"):
            # The fused module has no override of its own; both halves must
            # agree, and for this checkpoint both take the global 4b/gs64.
            spec = _quant_override(config, module_path[: -len("gate_up_proj")] + "gate_proj")
        plan.quant[module_path] = spec or dict(default_quant)

    plan.entries.sort(key=lambda e: (e.action, e.target or "", e.sources[0]))
    return plan


def plan_summary(plan: LoaderPlan) -> str:
    counts: dict[str, int] = {}
    for entry in plan.entries:
        counts[entry.action] = counts.get(entry.action, 0) + 1
    return ", ".join(f"{action}={n}" for action, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def build_config(
    model_dir,
    *,
    mtp_enabled: bool = True,
    mtp_depth: int = 3,
    mtp_checkpoint_prefix: Optional[str] = None,
) -> ModelConfig:
    """The model config, with Titan's runtime fields filled in."""
    model_dir = Path(model_dir)
    raw = dict(load_config(model_dir))
    text = dict(raw.get("text_config") or {})
    text["titan_model_path"] = str(model_dir)
    text["titan_mtp_enabled"] = bool(mtp_enabled)
    text["titan_mtp_checkpoint_prefix"] = mtp_checkpoint_prefix
    text["titan_mtp_depth"] = int(mtp_depth)
    raw["text_config"] = text
    return ModelConfig.from_dict(raw)


def _shard_groups(plan: LoaderPlan) -> dict[Path, list[PlanEntry]]:
    tensors = checkpoint_tensors(plan.model_dir)
    groups: dict[Path, list[PlanEntry]] = {}
    for entry in plan.entries:
        if entry.action == DROP:
            continue
        shard = tensors[entry.sources[0]].shard
        groups.setdefault(shard, []).append(entry)
    return groups


def materialise(plan: LoaderPlan, *, eval_every: int = 8) -> dict[str, mx.array]:
    """Execute *plan*, returning the weight dict the model binds.

    Shards are mapped one at a time with ``mx.load`` (memory mapped, lazy) and
    the result is evaluated every ``eval_every`` arrays, which keeps the Metal
    command buffer short and stops the transient from stacking up.
    """
    tensors = checkpoint_tensors(plan.model_dir)
    by_shard: dict[Path, list[PlanEntry]] = {}
    for entry in plan.entries:
        if entry.action == DROP:
            continue
        for source in entry.sources:
            by_shard.setdefault(tensors[source].shard, []).append(entry)

    weights: dict[str, mx.array] = {}
    pending: list[mx.array] = []
    loaded_shards: dict[Path, dict[str, mx.array]] = {}

    def shard_for(name: str) -> dict[str, mx.array]:
        shard = tensors[name].shard
        data = loaded_shards.get(shard)
        if data is None:
            data = mx.load(str(shard))
            loaded_shards[shard] = data
        return data

    done: set[str] = set()
    for entry in plan.entries:
        if entry.action == DROP or entry.target in done:
            continue
        if entry.action == LOAD:
            value = shard_for(entry.sources[0])[entry.sources[0]]
            if "conv1d.weight" in entry.target and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
        else:  # FUSE
            gate = shard_for(entry.sources[0])[entry.sources[0]]
            up = shard_for(entry.sources[1])[entry.sources[1]]
            value = mx.concatenate([gate, up], axis=1)
        weights[entry.target] = value
        done.add(entry.target)
        pending.append(value)
        if len(pending) >= eval_every:
            mx.eval(*pending)
            pending.clear()
            loaded_shards.clear()
    if pending:
        mx.eval(*pending)
    return weights


def load_model(
    model_dir,
    *,
    mtp_enabled: bool = True,
    mtp_depth: int = 3,
    fuse_gate_up: bool = True,
    strict: bool = True,
    wait_for_memory: bool = True,
) -> tuple[Model, LoaderPlan]:
    """Build the model and bind the checkpoint. Loads the whole thing."""
    model_dir = Path(model_dir)
    from .checkpoint import checkpoint_mtp_weight_prefix

    if wait_for_memory:
        # Every real load goes through here, server or bench harness. macOS
        # releases a dead process's image lazily, so a load started seconds
        # after another model process exited can overlap two 73 GB images and
        # swap the machine into a freeze (docs/ops/INCIDENTS.md). Wait for the
        # headroom first; refuse after the timeout rather than start anyway.
        from titan.observability.memory_guard import wait_for_headroom

        need_gb = _checkpoint_size_gb(model_dir) + 8.0
        wait_for_headroom(need_gb, timeout_s=180.0, log=logger.warning)

    prefix = checkpoint_mtp_weight_prefix(model_dir) if mtp_enabled else None
    config = build_config(
        model_dir,
        mtp_enabled=bool(mtp_enabled and prefix),
        mtp_depth=mtp_depth,
        mtp_checkpoint_prefix=prefix,
    )
    plan = build_plan(
        model_dir,
        mtp_enabled=bool(mtp_enabled and prefix),
        fuse_gate_up=fuse_gate_up,
    )
    model = Model(config)
    if fuse_gate_up:
        # Before quantisation: the plan's quant specs are keyed by the fused
        # path, so the fused module must exist when the predicate runs.
        fused_modules = fuse_switch_glu_modules(model)
        logger.info("fused gate/up on %d SwitchGLU modules", fused_modules)

    def class_predicate(path, module):
        spec = plan.quant_spec(path)
        if spec is None or not hasattr(module, "to_quantized"):
            return False
        return spec

    global_quant = (load_config(model_dir).get("quantization") or {})
    nn.quantize(
        model,
        group_size=global_quant.get("group_size", 64),
        bits=global_quant.get("bits", 4),
        class_predicate=class_predicate,
    )
    weights = materialise(plan)
    model.load_weights(list(weights.items()), strict=strict)
    model.eval()
    logger.info("loaded %s (%s)", model_dir.name, plan_summary(plan))
    return model, plan


def fuse_switch_glu_modules(model) -> int:
    """Restructure every SwitchGLU so it owns one ``gate_up_proj`` module.

    The plan writes the routed experts in the fused [gate | up] layout, so the
    module tree has to carry a single projection of doubled output width before
    ``load_weights`` binds the arrays (strict loading checks names and shapes).
    The still-lazy initial parameters are concatenated on the output axis and
    the gate module is reused as the container, so quantisation attributes
    (bits, group size, mode) carry over. Nothing is evaluated here. Returns the
    number of modules fused.
    """
    from .vendor.mlx_lm.models.switch_layers import SwitchGLU

    count = 0
    for _, module in model.named_modules():
        if not isinstance(module, SwitchGLU):
            continue
        if "gate_up_proj" in module or "gate_proj" not in module or "up_proj" not in module:
            continue
        gate, up = module.gate_proj, module.up_proj
        for name in ("weight", "scales", "biases", "bias"):
            g = gate.get(name) if hasattr(gate, "get") else None
            u = up.get(name) if hasattr(up, "get") else None
            if g is None or u is None:
                continue
            axis = -1 if name == "bias" else 1
            setattr(gate, name, mx.concatenate([g, u], axis=axis))
        module.gate_up_proj = gate
        del module.gate_proj
        del module.up_proj
        count += 1
    return count


def _checkpoint_size_gb(model_dir: Path) -> float:
    """Sum of the safetensors shards, the memory a load will need."""
    total = 0
    for f in Path(model_dir).glob("*.safetensors"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total / 1024 ** 3 if total else 75.0
