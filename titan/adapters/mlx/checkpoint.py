"""Checkpoint inspection: safetensors headers, no tensor data.

Everything here reads JSON only -- the safetensors header of each shard and the
sidecar manifests.  Nothing in this module maps or materialises a tensor, so it
is safe to run on a machine whose GPU is busy.  The loader (``loader.py``) uses
it to build its plan; ``tests/model/test_loader_plan.py`` uses it to check that
plan covers the checkpoint exactly once.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

MTP_WEIGHT_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)

# safetensors dtype string -> (bytes per element, mlx dtype name)
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorMeta:
    """One tensor as the checkpoint declares it."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: Path
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def read_safetensors_header(path: Path) -> dict:
    """Return the JSON header of a safetensors file. Reads the header only."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len <= 0 or header_len > (256 << 20):
            raise ValueError(f"implausible safetensors header length in {path}")
        return json.loads(f.read(header_len))


def shard_paths(model_dir: Path) -> list[Path]:
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
        names = sorted(set(weight_map.values()))
        return [model_dir / n for n in names]
    return sorted(model_dir.glob("*.safetensors"))


def checkpoint_tensors(model_dir: Path) -> dict[str, TensorMeta]:
    """Every tensor in the checkpoint, from the shard headers."""
    out: dict[str, TensorMeta] = {}
    for shard in shard_paths(Path(model_dir)):
        header = read_safetensors_header(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            if name in out:
                raise ValueError(f"tensor {name} appears in two shards")
            out[name] = TensorMeta(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(entry["shape"]),
                shard=shard,
                begin=begin,
                end=end,
            )
    return out


def iter_tensor_names(model_dir: Path) -> Iterator[str]:
    for shard in shard_paths(Path(model_dir)):
        for name in read_safetensors_header(shard):
            if name != "__metadata__":
                yield name


def checkpoint_mtp_weight_prefix(model_path) -> Optional[str]:
    """The MTP weight prefix this checkpoint uses, or ``None``.

    Titan replacement for oMLX's ``_checkpoint_qwen4_mtp_weight_prefix``: reads
    the shard index when present, otherwise the shard headers.
    """
    p = Path(model_path)
    if not p.is_dir():
        return None
    index = p / "model.safetensors.index.json"
    keys: list[str]
    try:
        if index.exists():
            keys = list((json.loads(index.read_text()).get("weight_map") or {}).keys())
        else:
            keys = list(iter_tensor_names(p))
    except Exception:
        return None
    for prefix in MTP_WEIGHT_PREFIXES:
        if any(k.startswith(prefix) for k in keys):
            return prefix
    return None


def load_config(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())


def ple_manifest(model_dir: Path) -> Optional[dict]:
    """The packed n-gram table manifest, when one has been built."""
    manifest = Path(model_dir) / "ple-packed" / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text())
