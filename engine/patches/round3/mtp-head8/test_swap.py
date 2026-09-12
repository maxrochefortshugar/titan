"""Exactness test for the 8-bit MTP head swap. No real model is loaded.

A synthetic Qwen4ExpMTPModule is built from the real oMLX class with random
weights at a shrunken geometry, quantised to 4-bit group 64 the way the
deployed checkpoint is, then swapped by patch.install(). Every swapped module
is compared against a directly constructed 8-bit module carrying the same
sidecar arrays; the two must be bit-identical.

    ~/inference-server/kdev/bin/python test_swap.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace

RES = "/Applications/oMLX.app/Contents/Resources"
sys.path.append(RES)
sys.path.append(os.path.join(RES, "Python/framework-mlx-base/lib/python3.11/site-packages"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpMTPModule,
    TextConfig,
)
from omlx.patches.qwen35_moe_gate_up import (  # noqa: E402
    _can_fuse,
    _ensure_call_patch,
    _fuse_one,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch as head8  # noqa: E402
import stio  # noqa: E402

HIDDEN, EXPERTS, INTER, LOWRANK = 256, 8, 128, 64
GS = 64
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def build_config():
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=INTER,
        moe_intermediate_size=INTER,
        rms_norm_eps=1e-6,
        vocab_size=1024,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=LOWRANK,
        head_dim=64,
        layer_types=["qwen_sparse_attention"],
        full_attention_interval=1,
        ple_layer_ids=[],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=256,
        indexer_compress_ratio=4,
    )


def quantized_predicate(path, module):
    return isinstance(module, (nn.Linear,)) or type(module).__name__ in (
        "SwitchLinear",
    )


def make_sidecar(bf16, path):
    """Quantise the captured bf16 weights to 8-bit group 64 into a sidecar."""
    spec, arrays = [], {}
    for name, w in bf16.items():
        qw, qs, qb = mx.quantize(w, group_size=GS, bits=8, mode="affine")
        mx.eval(qw, qs, qb)
        arrays[name] = (qw, qs, qb)
        spec += [
            (name + ".weight", list(qw.shape), mx.uint32),
            (name + ".scales", list(qs.shape), mx.bfloat16),
            (name + ".biases", list(qb.shape), mx.bfloat16),
        ]
    w = stio.StreamWriter(path, spec)
    for name in bf16:
        for i, suffix in enumerate((".weight", ".scales", ".biases")):
            w.write(name + suffix, arrays[name][i])
    w.close()
    return arrays


def direct_quantized_linear(qw, qs, qb):
    m = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(m)
    m.group_size, m.bits, m.mode = GS, 8, "affine"
    m.weight, m.scales, m.biases = qw, qs, qb
    m.freeze()
    return m


def direct_switch_linear(qw, qs, qb):
    m = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(m)
    m.group_size, m.bits, m.mode = GS, 8, "affine"
    m.weight, m.scales, m.biases = qw, qs, qb
    m.freeze()
    return m


def main():
    mx.random.seed(0)
    cfg = build_config()
    mtp = Qwen4ExpMTPModule(cfg)

    # Capture the bf16 weights of the seven targets, then quantise the module
    # to 4-bit group 64 the way the deployed checkpoint is quantised.
    paths = [
        "fc_embedding",
        "fc_hidden",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
        "layers.0.mlp.switch_mlp.gate_proj",
        "layers.0.mlp.switch_mlp.up_proj",
        "layers.0.mlp.switch_mlp.down_proj",
    ]
    bf16 = {}
    for p in paths:
        m = head8._resolve(mtp, p)
        assert m is not None, p
        bf16["mtp." + p] = m["weight"].astype(mx.bfloat16)
    mx.eval(list(bf16.values()))

    nn.quantize(mtp, group_size=GS, bits=4, mode="affine")
    live = {p: head8._resolve(mtp, p) for p in paths}
    check(
        "[0] synthetic head quantised to 4-bit group 64",
        all(m is not None and m.bits == 4 and m.group_size == GS
            for m in live.values()),
        ", ".join(f"{p.split('.')[-1]}={live[p].bits}b" for p in paths),
    )

    tmp = tempfile.mkdtemp(prefix="mtp8-")
    side = os.path.join(tmp, "mtp-8bit.safetensors")
    arrays = make_sidecar(bf16, side)

    # Pre-swap output shapes, to prove the forward geometry is unchanged.
    x2 = mx.random.normal((1, 3, HIDDEN)).astype(mx.bfloat16)
    x3 = mx.expand_dims(x2, (-2, -3))  # SwitchGLU's own expert layout
    idx = mx.array([[0, 1], [2, 3], [4, 5]], dtype=mx.uint32)[None]
    before = {
        "fc_embedding": live["fc_embedding"](x2).shape,
        "gate_proj": live["layers.0.mlp.switch_mlp.gate_proj"](x3, idx).shape,
    }

    os.environ[head8.ENV] = "1"
    ok = head8.install(_Holder(mtp), sidecar=side)
    check("[1] install() returned True", ok is True)

    swapped = {p: head8._resolve(mtp, p) for p in paths}
    check(
        "[2] every target now 8-bit group 64",
        all(m.bits == 8 and m.group_size == GS and m.mode == "affine"
            for m in swapped.values()),
    )
    check(
        "[3] module identity preserved (in-place swap)",
        all(swapped[p] is live[p] for p in paths),
    )

    # Bit-identity against directly constructed 8-bit modules.
    worst = 0.0
    for p in paths:
        qw, qs, qb = arrays["mtp." + p]
        m = swapped[p]
        in_dims = int(qw.shape[-1]) * 32 // 8
        xi = mx.random.normal((1, 3, in_dims)).astype(mx.bfloat16)
        if qw.ndim == 3:
            xe = mx.expand_dims(xi, (-2, -3))
            ref = direct_switch_linear(qw, qs, qb)
            a, b = m(xe, idx), ref(xe, idx)
        else:
            ref = direct_quantized_linear(qw, qs, qb)
            a, b = m(xi), ref(xi)
        mx.eval(a, b)
        d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        worst = max(worst, d)
        if d != 0.0:
            check(f"[4] {p} bit-identical", False, f"max abs {d:.3e}")
    check("[4] all seven bit-identical to a direct 8-bit module",
          worst == 0.0, f"max abs {worst:.3e}")

    after = {
        "fc_embedding": swapped["fc_embedding"](x2).shape,
        "gate_proj": swapped["layers.0.mlp.switch_mlp.gate_proj"](x3, idx).shape,
    }
    check("[5] forward output shapes unchanged", before == after,
          f"{before} -> {after}")

    # A full fuse_inputs pass still runs and keeps its shape.
    tok = mx.random.normal((1, 3, HIDDEN)).astype(mx.bfloat16)
    hid = mx.random.normal((1, 3, 4 * HIDDEN)).astype(mx.bfloat16)
    fused_out = mtp.fuse_inputs(tok, hid)
    mx.eval(fused_out)
    check("[6] mtp.fuse_inputs shape unchanged",
          tuple(fused_out.shape) == (1, 3, 4 * HIDDEN), str(fused_out.shape))

    check("[7] install() is idempotent",
          head8.install(_Holder(mtp), sidecar=side) is True)

    # ---- the fused gate_up path -------------------------------------------
    mx.random.seed(0)
    mtp2 = Qwen4ExpMTPModule(build_config())
    bf2 = {"mtp." + p: head8._resolve(mtp2, p)["weight"].astype(mx.bfloat16)
           for p in paths}
    mx.eval(list(bf2.values()))
    nn.quantize(mtp2, group_size=GS, bits=4, mode="affine")
    smlp = head8._resolve(mtp2, "layers.0.mlp.switch_mlp")
    fusable = _can_fuse(smlp)
    if fusable:
        _ensure_call_patch()  # the same SwitchGLU.__call__ swap oMLX installs
        _fuse_one(smlp)
    check("[8] oMLX gate+up fusion applied to the synthetic head",
          fusable and hasattr(smlp, "gate_up_proj")
          and not hasattr(smlp, "gate_proj"))

    side2 = os.path.join(tmp, "mtp-8bit-2.safetensors")
    arrays2 = make_sidecar(bf2, side2)
    ok2 = head8.install(_Holder(mtp2), sidecar=side2)
    check("[9] install() handles the fused container", ok2 is True)
    gu = smlp.gate_up_proj
    check("[10] fused gate_up now 8-bit and still one object",
          gu.bits == 8 and gu.group_size == GS
          and gu.weight.shape[1] == 2 * INTER,
          f"shape {tuple(gu.weight.shape)}")

    gq = direct_switch_linear(*arrays2["mtp.layers.0.mlp.switch_mlp.gate_proj"])
    uq = direct_switch_linear(*arrays2["mtp.layers.0.mlp.switch_mlp.up_proj"])
    got = gu(x3, idx)
    want = mx.concatenate([gq(x3, idx), uq(x3, idx)], axis=-1)
    mx.eval(got, want)
    d = float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))))
    check("[11] fused 8-bit output == concat of separate 8-bit gate and up",
          d == 0.0, f"max abs {d:.3e}")

    smlp_out = smlp(mx.random.normal((1, 3, HIDDEN)).astype(mx.bfloat16),
                    mx.array([[0, 1], [2, 3], [4, 5]], dtype=mx.uint32))
    mx.eval(smlp_out)
    check("[12] SwitchGLU forward still runs post-swap",
          tuple(smlp_out.shape) == (1, 3, 2, HIDDEN), str(smlp_out.shape))

    # ---- fail-closed behaviour --------------------------------------------
    mx.random.seed(0)
    mtp3 = Qwen4ExpMTPModule(build_config())
    nn.quantize(mtp3, group_size=GS, bits=4, mode="affine")
    os.environ[head8.ENV] = "0"
    r = head8.install(_Holder(mtp3), sidecar=side)
    check("[13] returns False with the env var off", r is False
          and head8._resolve(mtp3, "fc_embedding").bits == 4)
    os.environ[head8.ENV] = "1"
    r = head8.install(_Holder(mtp3), sidecar=os.path.join(tmp, "missing.st"))
    check("[14] returns False on a missing sidecar, weights untouched",
          r is False and head8._resolve(mtp3, "fc_embedding").bits == 4)
    r = head8.install(None, sidecar=side)
    check("[15] returns False when the model has no mtp block", r is False)

    print()
    print("FAILED:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


class _Holder:
    """Stands in for the mlx_vlm Model, which holds the head as ``.mtp``."""

    def __init__(self, mtp):
        self.mtp = mtp


if __name__ == "__main__":
    sys.exit(main())
