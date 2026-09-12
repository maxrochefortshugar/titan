"""The loader plan covers the checkpoint exactly once.

Header-only: this test reads ``config.json`` and the safetensors headers of the
21 shards.  It never maps a tensor's data, never touches the GPU, and is safe to
run while the workbench is holding the model.

Skipped when the checkpoint is not on this machine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from titan.adapters.mlx import checkpoint, loader

MODEL_DIR = Path(
    os.environ.get(
        "TITAN_MODEL_DIR",
        Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
    )
).expanduser()

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "config.json").exists(),
    reason=f"checkpoint not present at {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def tensors():
    return checkpoint.checkpoint_tensors(MODEL_DIR)


@pytest.fixture(scope="module")
def config():
    return checkpoint.load_config(MODEL_DIR)


@pytest.fixture(scope="module")
def plan(tensors, config):
    return loader.build_plan(MODEL_DIR, tensors=tensors, config=config)


def test_every_tensor_is_claimed_exactly_once(plan, tensors):
    sources = plan.sources()
    assert len(sources) == len(set(sources)), "a tensor is claimed twice"
    assert set(sources) == set(tensors), "plan and checkpoint disagree"


def test_no_target_is_written_twice(plan):
    targets = [e.target for e in plan.entries if e.target is not None]
    assert len(targets) == len(set(targets))


def test_dropped_tensors_are_only_the_three_expected_families(plan):
    for entry in plan.by_action(loader.DROP):
        name = entry.sources[0]
        assert (
            name.startswith("vision_tower.")
            or loader.NGRAM_STORAGE_RE.search(name)
            or name.startswith("mtp.")
        ), f"unexpected drop: {name}"
        assert entry.reason, "every drop states a reason"


def test_vision_tower_is_dropped_whole(plan, tensors):
    vision = {n for n in tensors if n.startswith("vision_tower.")}
    dropped = {
        e.sources[0]
        for e in plan.by_action(loader.DROP)
        if e.sources[0].startswith("vision_tower.")
    }
    assert vision and dropped == vision


def test_ngram_shards_are_never_resident(plan, tensors):
    shard_tensors = {n for n in tensors if loader.NGRAM_STORAGE_RE.search(n)}
    assert len(shard_tensors) == 384, "128 shards x weight/scales/biases"
    loaded = {
        s
        for e in plan.entries
        if e.action != loader.DROP
        for s in e.sources
    }
    assert not (shard_tensors & loaded)


def test_mtp_is_bound_when_enabled_and_dropped_when_not(tensors, config):
    mtp = {n for n in tensors if n.startswith("mtp.")}
    assert mtp, "this checkpoint ships an MTP head"

    on = loader.build_plan(MODEL_DIR, tensors=tensors, config=config, mtp_enabled=True)
    off = loader.build_plan(MODEL_DIR, tensors=tensors, config=config, mtp_enabled=False)

    on_targets = on.targets()
    assert any(t.startswith("mtp.") for t in on_targets)
    assert not any(
        (e.target or "").startswith("mtp.") for e in off.entries
    )
    assert {e.sources[0] for e in off.by_action(loader.DROP)} >= mtp
    # Coverage still holds with the head off.
    assert set(off.sources()) == set(tensors)


def test_gate_up_is_fused_for_every_moe_layer(plan, config):
    fused = plan.by_action(loader.FUSE)
    layers = config["text_config"]["num_hidden_layers"]
    # every decoder layer plus the single MTP layer, three tensors each
    assert len(fused) == (layers + 1) * 3

    for entry in fused:
        assert entry.target.endswith(
            (".gate_up_proj.weight", ".gate_up_proj.scales", ".gate_up_proj.biases")
        )
        gate, up = entry.sources
        assert gate.endswith(f".gate_proj.{entry.suffix}")
        assert up.endswith(f".up_proj.{entry.suffix}")
        # gate rows first, then up rows, on the output axis
        assert re.match(r".*\.switch_mlp\.gate_up_proj\.\w+$", entry.target)


def test_fused_shapes_double_the_output_axis(plan, tensors):
    for entry in plan.by_action(loader.FUSE):
        gate = tensors[entry.sources[0]]
        up = tensors[entry.sources[1]]
        assert gate.shape == up.shape, "this checkpoint's halves are symmetric"
        assert entry.shape == (
            gate.shape[0],
            gate.shape[1] * 2,
            *gate.shape[2:],
        )
        assert entry.dtype == gate.dtype


def test_expected_fused_expert_shapes(plan):
    weight = plan.targets()[
        "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight"
    ]
    scales = plan.targets()[
        "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.scales"
    ]
    assert weight.shape == (512, 1280, 320) and weight.dtype == "U32"
    assert scales.shape == (512, 1280, 40) and scales.dtype == "BF16"


def test_quantisation_bits_follow_the_per_tensor_overrides(plan):
    quant = plan.quant
    # global 4-bit / group 64 for the routed experts
    assert quant["language_model.model.layers.0.mlp.switch_mlp.gate_up_proj"] == {
        "group_size": 64,
        "bits": 4,
    }
    assert quant["language_model.model.layers.0.mlp.switch_mlp.down_proj"] == {
        "group_size": 64,
        "bits": 4,
    }
    # 8-bit embeddings and head
    assert quant["language_model.model.embed_tokens"]["bits"] == 8
    assert quant["language_model.lm_head"]["bits"] == 8
    # 8-bit / group 128 shared experts
    assert quant["language_model.model.layers.0.mlp.shared_expert.gate_proj"] == {
        "group_size": 128,
        "bits": 8,
    }
    # 5-bit / group 128 GDN output projections
    assert quant["language_model.model.layers.0.linear_attn.out_proj"] == {
        "group_size": 128,
        "bits": 5,
    }


def test_every_quantised_module_has_all_three_tensors(plan):
    targets = plan.targets()
    for module_path in plan.quant:
        for suffix in ("weight", "scales", "biases"):
            assert f"{module_path}.{suffix}" in targets, module_path


def test_router_and_norms_stay_unquantised(plan):
    assert "language_model.model.layers.0.mlp.gate" not in plan.quant
    entry = plan.targets()["language_model.model.layers.0.mlp.gate.weight"]
    assert entry.dtype == "BF16"


def test_conv1d_layout_is_left_alone_when_already_mlx(plan):
    entry = plan.targets()[
        "language_model.model.layers.0.linear_attn.conv1d.weight"
    ]
    assert entry.shape[-1] == 1
    assert entry.reason == ""


def test_plan_reads_no_tensor_data(plan, tensors):
    """The plan's shapes come from the headers, so they must match them."""
    for entry in plan.by_action(loader.LOAD):
        meta = tensors[entry.sources[0]]
        expected = meta.shape
        if "conv1d.weight" in entry.target and expected[-1] != 1:
            expected = (expected[0], expected[2], expected[1])
        assert entry.shape == expected
        assert entry.dtype == meta.dtype


def test_ple_manifest_matches_the_checkpoint_table(plan):
    manifest = checkpoint.ple_manifest(MODEL_DIR)
    if manifest is None:
        pytest.skip("no packed n-gram table built yet")
    layer = manifest["layers"]["1"]
    assert layer["rows"] == 320_001_536
    assert layer["row_stride"] == layer["weight_bytes"] + 2 * layer["scale_bytes"]
    assert layer["bits"] == 4 and layer["group_size"] == 32
    assert layer["layouts"]["rows"]["order"] == ["weight", "scales", "biases"]
    assert "resident" not in layer["layouts"], "rows mode only"
