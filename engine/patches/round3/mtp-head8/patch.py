# SPDX-License-Identifier: Apache-2.0
"""Serve the Qwen4-Exp MTP draft block at 8 bits.

The deployed checkpoint quantises seven ``mtp.*`` projections at 4-bit group 64.
This patch swaps them for 8-bit group 64 affine weights read from the sidecar
``mtp-8bit.safetensors``, which was produced from the bf16 originals in
``Qwen/Qwen3.8-Flash-Next`` (a 4-bit tensor cannot be requantised upward).

Timing: **after model load**, it mutates live module instances.
Env gate: ``OMLX_MTP_HEAD8=1``. Sidecar path override: ``OMLX_MTP_HEAD8_SIDECAR``.

The swap is done in place on the existing module objects rather than by
constructing replacements, so every reference oMLX has already cached stays
valid: ``qwen35_moe_gate_up._fuse_one`` reuses the ``gate_proj`` object as the
fused ``gate_up_proj`` container, ``SwitchGLU.__call__`` and the vlm verify
patch both resolve ``gate_up_proj`` off ``self`` at call time, and
``hc_fused`` reads ``input_mix_weight_down`` / ``_up`` plus their ``.bits`` on
every call. Both quantized classes read ``bits`` / ``group_size`` / ``mode``
and the arrays at call time, so mutating them is sufficient and total.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

ENV = "OMLX_MTP_HEAD8"
SIDECAR_ENV = "OMLX_MTP_HEAD8_SIDECAR"
DEFAULT_SIDECAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "mtp-8bit.safetensors")
BITS, GROUP_SIZE, MODE = 8, 64, "affine"
_FLAG = "_omlx_mtp_head8_installed"

# module path inside the mtp block -> sidecar key prefix
TARGETS = (
    "fc_embedding",
    "fc_hidden",
    "hyper_connection_mixer.input_mix_weight_down",
    "hyper_connection_mixer.input_mix_weight_up",
    "layers.0.mlp.switch_mlp.gate_proj",
    "layers.0.mlp.switch_mlp.up_proj",
    "layers.0.mlp.switch_mlp.down_proj",
)
# When oMLX has already fused gate+up, gate_proj/up_proj are gone and the fused
# container carries [gate; up] concatenated on the output axis.
FUSED_PATH = "layers.0.mlp.switch_mlp.gate_up_proj"
FUSED_PARTS = ("layers.0.mlp.switch_mlp.gate_proj",
               "layers.0.mlp.switch_mlp.up_proj")


def _resolve(root: Any, path: str):
    """Walk a dotted path, treating integer components as list indices."""
    node = root
    for part in path.split("."):
        if part.isdigit() and isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def find_mtp_module(model: Any):
    """Return the live Qwen4ExpMTPModule, or None."""
    if model is None:
        return None
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        return mtp
    lm = getattr(model, "language_model", None)
    if lm is not None:
        getter = getattr(lm, "get_mtp_module", None)
        if getter is not None:
            return getter()
        return getattr(lm, "mtp", None)
    getter = getattr(model, "get_mtp_module", None)
    return getter() if getter is not None else None


def _out_in(module) -> tuple[int, int]:
    """(output_dims, input_dims) of a quantized linear or switch linear."""
    w = module["weight"]
    return int(w.shape[-2]), int(w.shape[-1]) * 32 // int(module.bits)


def _swap(module, w, s, b) -> None:
    """Mutate a quantized module to 8-bit group 64 in place."""
    module.weight = w
    module.scales = s
    if b is None:
        module.biases = None
    else:
        module.biases = b
    module.bits = BITS
    module.group_size = GROUP_SIZE
    module.mode = MODE


def _invalidate_caches(mtp) -> None:
    """Drop anything that could still hold pre-swap arrays.

    ``Qwen4ExpGatedResidual`` keeps ``_compiled_forward`` (an ``mx.compile`` of
    its own forward, which captures the projection arrays as graph constants)
    and ``_omlx_exact_hybrid_projection``. Both are skipped while the MTP
    runtime is enabled, so this is belt and braces, but a stale compiled graph
    would silently serve the old 4-bit weights.
    """
    dropped = 0
    seen = set()
    nodes = [mtp]
    try:
        nodes += [m for _, m in mtp.named_modules()]
    except Exception:  # noqa: BLE001
        pass
    for node in nodes:
        if id(node) in seen:
            continue
        seen.add(id(node))
        for attr in ("_compiled_forward", "_omlx_exact_hybrid_projection",
                     "_omlx_hc_fused_signature", "_omlx_hc_fused_eps"):
            if hasattr(node, attr):
                try:
                    delattr(node, attr)
                    dropped += 1
                except Exception:  # noqa: BLE001
                    pass
    return dropped


def _check_forward(module, out_dims: int, switch: bool) -> None:
    """Confirm the swapped module still produces the pre-swap output shape."""
    in_dims = _out_in(module)[1]
    if switch:
        x = mx.zeros((1, 1, 1, in_dims), dtype=mx.bfloat16)
        idx = mx.zeros((1, 1), dtype=mx.uint32)
        y = module(x, idx)
    else:
        x = mx.zeros((1, 1, in_dims), dtype=mx.bfloat16)
        y = module(x)
    mx.eval(y)
    if int(y.shape[-1]) != out_dims:
        raise RuntimeError(
            f"post-swap output width {y.shape[-1]} != {out_dims}"
        )


def install(model: Any = None, sidecar: str | None = None) -> bool:
    """Swap the 4-bit mtp.* projections to 8-bit. Run after the model loads."""
    if os.environ.get(ENV, "0") != "1":
        return False

    mtp = find_mtp_module(model)
    if mtp is None:
        logger.info("MTP head8: no mtp block on the model, leaving 4-bit")
        return False
    if getattr(mtp, _FLAG, False):
        return True

    path = sidecar or os.environ.get(SIDECAR_ENV) or DEFAULT_SIDECAR
    if not os.path.exists(path):
        logger.warning("MTP head8: sidecar %s missing, leaving 4-bit", path)
        return False

    # Optional subset: OMLX_MTP_HEAD8_TARGETS="fc_embedding,fc_hidden,..." (names relative to mtp.)
    _sel = [x.strip() for x in os.environ.get("OMLX_MTP_HEAD8_TARGETS", "").split(",") if x.strip()]
    targets = [t for t in TARGETS if not _sel or t in _sel]
    fused = _resolve(mtp, FUSED_PATH) is not None and any(t in FUSED_PARTS for t in targets)
    wanted = [t for t in targets if t not in FUSED_PARTS or not fused]

    # Every module must exist and be quantized before anything is touched.
    modules = {}
    for t in wanted:
        m = _resolve(mtp, t)
        if m is None or not hasattr(m, "bits") or "weight" not in m:
            logger.warning("MTP head8: %s not a quantized module, aborting", t)
            return False
        if int(m.bits) == BITS and int(m.group_size) == GROUP_SIZE:
            logger.info("MTP head8: %s already 8-bit", t)
        modules[t] = m
    if fused:
        m = _resolve(mtp, FUSED_PATH)
        if m is None or "weight" not in m:
            logger.warning("MTP head8: fused gate_up_proj unusable, aborting")
            return False
        modules[FUSED_PATH] = m

    try:
        weights = mx.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP head8: cannot read %s: %s", path, exc)
        return False

    need = list(wanted) + list(FUSED_PARTS if fused else ())
    logger.info("MTP head8: swapping %s (fused=%s)", need, fused)
    for t in need:
        for suffix in (".weight", ".scales", ".biases"):
            if "mtp." + t + suffix not in weights:
                logger.warning("MTP head8: sidecar lacks mtp.%s%s", t, suffix)
                return False

    # Shape agreement against the live 4-bit modules, checked before mutating.
    plan = {}
    for t in wanted:
        m = modules[t]
        out_dims, in_dims = _out_in(m)
        w = weights["mtp." + t + ".weight"]
        s = weights["mtp." + t + ".scales"]
        b = weights["mtp." + t + ".biases"]
        if int(w.shape[-2]) != out_dims or int(w.shape[-1]) * 32 // BITS != in_dims:
            logger.warning(
                "MTP head8: %s shape mismatch, live (%d,%d) sidecar %s",
                t, out_dims, in_dims, tuple(w.shape),
            )
            return False
        if int(s.shape[-1]) != in_dims // GROUP_SIZE:
            logger.warning("MTP head8: %s scales width wrong", t)
            return False
        plan[t] = (m, w, s, b, out_dims)

    if fused:
        m = modules[FUSED_PATH]
        out_dims, in_dims = _out_in(m)
        gp, up = FUSED_PARTS
        w = mx.concatenate(
            [weights["mtp." + gp + ".weight"], weights["mtp." + up + ".weight"]],
            axis=1,
        )
        s = mx.concatenate(
            [weights["mtp." + gp + ".scales"], weights["mtp." + up + ".scales"]],
            axis=1,
        )
        b = mx.concatenate(
            [weights["mtp." + gp + ".biases"], weights["mtp." + up + ".biases"]],
            axis=1,
        )
        if int(w.shape[-2]) != out_dims or int(w.shape[-1]) * 32 // BITS != in_dims:
            logger.warning("MTP head8: fused gate_up shape mismatch")
            return False
        plan[FUSED_PATH] = (m, w, s, b, out_dims)

    for t, (m, w, s, b, out_dims) in plan.items():
        switch = w.ndim == 3
        _swap(m, w, s, b)
        mx.eval(m["weight"], m["scales"], m["biases"])
        _check_forward(m, out_dims, switch)

    dropped = _invalidate_caches(mtp)
    setattr(mtp, _FLAG, True)
    mx.clear_cache()
    logger.info(
        "MTP head8: %d mtp projections now 8-bit group 64 (fused=%s, %d caches dropped)",
        len(plan), fused, dropped,
    )
    return True


__all__ = ["install", "find_mtp_module", "ENV"]
