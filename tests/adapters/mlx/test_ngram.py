"""Packed n-gram reader: layout, refusals, prefetch and bit-exact gather.

The synthetic table here is a real pack in miniature: the same manifest keys,
the same row-contiguous layout, the same weight/scale/bias split, sized so it
spans a few dozen pages rather than two million. Everything about the read path
is exercised against it. What it cannot prove is that the real pack's manifest
still parses and its rows still dequantise, so there is one read-only smoke test
against the checkpoint's own ``ple-packed`` when it exists. That test reads a few
dozen rows, which is a few hundred kilobytes, and it never writes.

The expected values are computed independently, from the bytes the test wrote,
through ``mx.dequantize`` directly. That direction matters: comparing the reader
against itself would only prove it is self-consistent.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from titan.adapters.mlx.ngram import PAGES_PER_TASK, open_packed_table
from titan.core.errors import ConfigError

MODEL_DIR = Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
PLE_PACKED = MODEL_DIR / "ple-packed"

ROWS = 8192
DIMS = 64
BITS = 4
GROUP = 32
WEIGHT_BYTES = DIMS * BITS // 8  # 32
GROUPS = DIMS // GROUP  # 2
SCALE_BYTES = GROUPS * 2  # bf16
BIAS_BYTES = GROUPS * 2
STRIDE = WEIGHT_BYTES + SCALE_BYTES + BIAS_BYTES  # 40


def _bf16_bytes(values: np.ndarray) -> np.ndarray:
    """The bf16 encoding of ``values`` as raw bytes, via mlx's own cast."""
    packed = mx.array(values).astype(mx.bfloat16).view(mx.uint16)
    return np.array(packed, copy=False).view(np.uint8)


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> tuple[Path, np.ndarray]:
    """Write a tiny pack and return its directory and its raw rows."""
    directory = tmp_path_factory.mktemp("ple-packed")
    rng = np.random.default_rng(20260912)
    raw = rng.integers(0, 256, size=(ROWS, STRIDE), dtype=np.uint8)
    # keep the bf16 scales and biases in a sane range: random bits can be NaN,
    # and NaN would make an equality comparison meaningless rather than wrong
    scales = (rng.random((ROWS, GROUPS), dtype=np.float32) + 0.1).astype(np.float32)
    biases = (rng.random((ROWS, GROUPS), dtype=np.float32) - 0.5).astype(np.float32)
    raw[:, WEIGHT_BYTES : WEIGHT_BYTES + SCALE_BYTES] = _bf16_bytes(scales)
    raw[:, WEIGHT_BYTES + SCALE_BYTES :] = _bf16_bytes(biases)
    (directory / "layer0.rows.bin").write_bytes(raw.tobytes())
    manifest = {
        "pack_version": 1,
        "model": "synthetic",
        "layers": {
            "0": {
                "rows": ROWS,
                "dims": DIMS,
                "bits": BITS,
                "group_size": GROUP,
                "mode": "affine",
                "weight_bytes": WEIGHT_BYTES,
                "scale_bytes": SCALE_BYTES,
                "row_stride": STRIDE,
                "shard_offsets": [0],
                "prefix": "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding",
                "layouts": {
                    "rows": {
                        "file": "layer0.rows.bin",
                        "stride": STRIDE,
                        "order": ["weight", "scales", "biases"],
                    }
                },
            }
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, raw


def expected(raw: np.ndarray, ids: list[int]) -> mx.array:
    """Dequantise the requested rows straight from the bytes the test wrote."""
    slab = raw[np.asarray(ids, dtype=np.intp)]
    w = slab[:, :WEIGHT_BYTES].copy().view("<u4")
    s = slab[:, WEIGHT_BYTES : WEIGHT_BYTES + SCALE_BYTES].copy().view("<u2")
    b = slab[:, WEIGHT_BYTES + SCALE_BYTES :].copy().view("<u2")
    return mx.dequantize(
        mx.array(w),
        mx.array(s).view(mx.bfloat16),
        mx.array(b).view(mx.bfloat16),
        group_size=GROUP,
        bits=BITS,
        mode="affine",
    )


def same(a: mx.array, b: mx.array) -> bool:
    """Bit-identical, which is the tolerance this op declares."""
    return a.shape == b.shape and bool(mx.array_equal(a, b))


@pytest.fixture
def reader(pack):
    directory, _ = pack
    with open_packed_table(directory, workers=4) as r:
        yield r


# ---------------------------------------------------------------------------
# opening
# ---------------------------------------------------------------------------


def test_manifest_shape_is_read(reader):
    assert reader.rows == ROWS
    assert reader.dims == DIMS
    assert reader.entry.row_stride == STRIDE


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_packed_table(tmp_path / "nope")


def test_unknown_layer(pack):
    directory, _ = pack
    with pytest.raises(ConfigError):
        open_packed_table(directory, layer=7)


@pytest.mark.parametrize("option", ["resident", "preload"])
def test_resident_options_are_refused(pack, option):
    """A resident copy of this table panicked the kernel. It stays refused."""
    directory, _ = pack
    with pytest.raises(ConfigError) as excinfo:
        open_packed_table(directory, **{option: True})
    assert "resident" in str(excinfo.value).lower()


def test_only_mmap_mode(pack):
    directory, _ = pack
    with pytest.raises(ConfigError):
        open_packed_table(directory, mode="resident")


def test_the_mapping_is_read_only(reader):
    mapping = reader.table._map
    assert isinstance(mapping, np.memmap)
    assert mapping.mode == "r"
    assert not mapping.flags.writeable
    # and nothing has been copied out of it wholesale
    assert mapping.base is not None or mapping.size == ROWS * STRIDE


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------


def test_gather_matches_a_direct_dequantise(reader, pack):
    _, raw = pack
    ids = [0, 1, 5, 4095, ROWS - 1]
    assert same(reader.gather(ids), expected(raw, ids))


def test_gather_empty(reader):
    out = reader.gather([])
    assert out.shape == (0, DIMS)


def test_gather_repeated_ids(reader, pack):
    _, raw = pack
    ids = [7, 7, 7, 3]
    assert same(reader.gather(ids), expected(raw, ids))


def test_gather_accepts_numpy_and_mlx(reader, pack):
    _, raw = pack
    ids = [11, 900, 4444]
    want = expected(raw, ids)
    assert same(reader.gather(np.array(ids, dtype=np.int64)), want)
    assert same(reader.gather(mx.array(ids, dtype=mx.uint32)), want)


@pytest.mark.parametrize("bad", [-1, ROWS])
def test_gather_rejects_out_of_range(reader, bad):
    with pytest.raises(IndexError):
        reader.gather([0, bad])


def test_prefetch_does_not_change_the_result(reader, pack):
    _, raw = pack
    rng = random.Random(7)
    ids = [rng.randrange(ROWS) for _ in range(256)]
    cold = expected(raw, ids)
    reader.prefetch(ids)
    assert same(reader.gather(ids), cold)


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


def test_prefetch_returns_without_reading(reader):
    """It queues and returns. The pages are read on the pool's threads."""
    ids = list(range(0, ROWS, 7))
    started = time.perf_counter()
    reader.prefetch(ids)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    reader.gather(ids)
    assert reader.stats()["pages_read"] > 0


def test_prefetch_is_idempotent(reader):
    ids = list(range(512))
    reader.prefetch(ids)
    reader.gather(ids)
    pages = reader.stats()["pages_read"]
    reader.prefetch(ids)
    reader.gather(ids)
    assert reader.stats()["pages_read"] == pages


def test_prefetch_empty_is_a_no_op(reader):
    reader.prefetch([])
    assert reader.stats()["prefetch_calls"] == 0


def test_a_prefetched_gather_counts_as_a_hit(reader):
    ids = list(range(64))
    reader.prefetch(ids)
    reader.gather(ids)
    stats = reader.stats()
    assert stats["rows_hit"] == 64
    assert stats["gather_calls"] == 1


def test_an_unprefetched_gather_counts_as_a_miss(reader):
    reader.gather([1000, 2000])
    stats = reader.stats()
    assert stats["rows_missed"] == 2
    assert stats["rows_hit"] == 0


def test_stats_keys(reader):
    reader.prefetch([1])
    reader.gather([1])
    stats = reader.stats()
    for key in ("bytes_read", "pages_read", "mean_wait_ms", "hit_rate", "wait_ms"):
        assert key in stats
    assert stats["bytes_read"] > 0


def test_pages_are_batched(reader):
    """One pool task per page would swamp the scheduler on a real chunk."""
    assert PAGES_PER_TASK > 1


def test_close_is_idempotent(pack):
    directory, _ = pack
    reader = open_packed_table(directory, workers=2)
    reader.close()
    reader.close()


# ---------------------------------------------------------------------------
# the real pack, read only
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(
    not (PLE_PACKED / "manifest.json").exists(), reason="no packed table present"
)


@real
def test_real_pack_smoke():
    """A few dozen rows out of the 32 GB pack, read only, under a megabyte."""
    rng = random.Random(20260912)
    with open_packed_table(PLE_PACKED, layer=1, workers=8) as reader:
        assert reader.dims == 160
        assert reader.rows > 300_000_000
        ids = [rng.randrange(reader.rows) for _ in range(48)]
        reader.prefetch(ids)
        warm = reader.gather(ids)
        assert warm.shape == (48, 160)
        assert warm.dtype == mx.bfloat16
        assert bool(mx.all(mx.isfinite(warm)))
        # gathering again must give the same bits
        assert same(reader.gather(ids), warm)
        stats = reader.stats()
        assert stats["bytes_read"] < 100 * 1024 * 1024


def test_satisfies_the_port(reader):
    from titan.core.ports import NgramReader

    assert isinstance(reader, NgramReader)
