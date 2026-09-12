# SPDX-License-Identifier: MIT
"""ple_packed_lookup: bit-identical, and the pooled reader changes only timing.

Bar from engine/patches/ple-fix/REPORT.md section 4.1: the runtime lookup is
bitwise identical to the unpacked path in rows mode. Here the two
implementations differ only in whether the pages are touched through the
caller's worker pool first, so the output must be identical word for word, and
the page accounting must show the prefetch actually did the reads.

The pack is built in a tmp dir from random rows, in the same layout the repack
tool writes: 160 dims, 4-bit affine, group size 32, so one row is 80 + 10 + 10
= 100 contiguous bytes, which is the 100 B the report quotes.

There is no resident mode to test. It is not ported: on a 128 GB machine the
32 GB of resident planes panicked the kernel.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from titan.kernels import ple_packed_lookup as op

ROWS, DIMS, BITS, GROUP = 4096, 160, 4, 32


@pytest.fixture()
def pack(tmp_path):
    """A synthetic pack: manifest.json plus a row-contiguous rows.bin."""
    mx.random.seed(0)
    dense = mx.random.normal((ROWS, DIMS)).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=GROUP, bits=BITS)
    mx.eval(w, s, b)
    wb = np.array(w, copy=True).view(np.uint8).reshape(ROWS, -1)
    sb = np.array(s.view(mx.uint16), copy=True).view(np.uint8).reshape(ROWS, -1)
    bb = np.array(b.view(mx.uint16), copy=True).view(np.uint8).reshape(ROWS, -1)
    raw = np.concatenate([wb, sb, bb], axis=1)
    (tmp_path / "layer0.rows.bin").write_bytes(raw.tobytes())
    manifest = {"layers": {"0": {
        "rows": ROWS, "dims": DIMS, "bits": BITS, "group_size": GROUP,
        "weight_bytes": int(wb.shape[1]), "scale_bytes": int(sb.shape[1]),
        "row_stride": int(raw.shape[1]), "shard_offsets": [0, ROWS],
        "layouts": {"rows": {"file": "layer0.rows.bin"}},
    }}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, dense


def _table(pack):
    directory, layers = op.load_manifest(pack[0])
    return op.PackedRowTable(directory, layers[0])


@pytest.mark.exactness
@pytest.mark.parametrize("n", [1, 9, 512])
def test_pooled_read_is_bit_identical(pack, n):
    rng = np.random.default_rng(0)
    rows = rng.integers(0, ROWS, size=n)
    with _table(pack) as plain, _table(pack) as pooled, op.ReaderPool(8) as io:
        a = op.reference(plain, rows)
        b = op.metal(pooled, rows, io)
        mx.eval(a, b)
        assert bool(mx.all(a == b).item())


@pytest.mark.exactness
def test_dequantises_to_the_original_rows(pack):
    """The pack is the same nibbles the quantiser produced, so the lookup must
    reproduce mx.dequantize of those rows exactly."""
    _dir, dense = pack
    rows = np.arange(64)
    with _table(pack) as table:
        got = op.reference(table, rows)
        w, s, b = mx.quantize(dense[:64], group_size=GROUP, bits=BITS)
        want = mx.dequantize(w, s, b, group_size=GROUP, bits=BITS, mode="affine")
        mx.eval(got, want)
        assert bool(mx.all(got == want).item())


def test_prefetch_reads_each_page_once(pack):
    """Second call over the same rows must read nothing: the seen-page bitmap
    is what keeps a warm chunk off the SSD."""
    rows = np.arange(0, 2048)
    with _table(pack) as table, op.ReaderPool(8) as io:
        first = table.prefetch(rows, io)
        second = table.prefetch(rows, io)
    assert first > 0
    assert second == 0


def test_worker_count_is_the_callers(pack):
    """The count is a constructor argument, not a module constant: it is the
    only knob over the read path and it is worth several times serialised
    faults."""
    for n in (1, 4, 32):
        with op.ReaderPool(n) as io:
            assert io.workers == n
    with op.ReaderPool(100000) as io:
        assert io.workers == 512


def test_accepts_device_indices(pack):
    rows = mx.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=mx.uint32)
    with _table(pack) as table:
        assert op.reference(table, rows).shape == (10, DIMS)


def test_empty_lookup(pack):
    with _table(pack) as table:
        assert op.reference(table, np.array([], dtype=np.int64)).shape == (0, DIMS)


def test_manifest_rejects_a_missing_rows_file(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": {"0": {
        "rows": 4, "dims": DIMS, "bits": BITS, "group_size": GROUP,
        "weight_bytes": 80, "scale_bytes": 10, "row_stride": 100,
        "shard_offsets": [0, 4], "layouts": {"rows": {"file": "gone.bin"}},
    }}}))
    with pytest.raises(FileNotFoundError):
        op.load_manifest(tmp_path)


def test_no_resident_mode_exists():
    """Asserted, not just documented: a resident copy of this table panicked
    the kernel on a 128 GB machine, and it must not come back by accident."""
    assert not hasattr(op, "resident")
    assert not any("resident" in name.lower() for name in dir(op))


def test_supports_needs_a_pool_and_enough_rows(pack):
    with _table(pack) as table, op.ReaderPool(4) as io:
        rows = np.arange(64)
        assert op.supports(op.key(table, rows, io))
        assert not op.supports(op.key(table, rows))
        assert not op.supports(op.key(table, np.arange(4), io))


@pytest.mark.slow
@pytest.mark.exactness
def test_real_row_count(pack):
    """The row count a 2048-token chunk touches: about 31,855 scattered rows."""
    rng = np.random.default_rng(1)
    rows = rng.integers(0, ROWS, size=31855)
    with _table(pack) as plain, _table(pack) as pooled, op.ReaderPool(16) as io:
        a = op.reference(plain, rows)
        b = op.metal(pooled, rows, io)
        mx.eval(a, b)
        assert bool(mx.all(a == b).item())
