"""The two tiers: budgets, the writer thread, and every way a read can fail."""

from __future__ import annotations

import os

import pytest

from titan.core.types import BlockHash
from titan.adapters.cache.format import encode_record
from titan.adapters.cache.store import TwoTierStateStore

from tests.cache.fakes import signature


def key(name: str) -> BlockHash:
    return BlockHash(name.encode().ljust(32, b"\0"))


@pytest.fixture
def ram_store():
    store = TwoTierStateStore(signature(), hot_budget_bytes=4096)
    yield store
    store.close()


def make_disk_store(tmp_path, **kwargs):
    defaults = dict(
        ssd_dir=str(tmp_path),
        hot_budget_bytes=1024,
        ssd_capacity_bytes=1024 * 1024,
        pending_budget_bytes=1024 * 1024,
        max_stall_s=0.05,
    )
    defaults.update(kwargs)
    return TwoTierStateStore(signature(), **defaults)


def test_ram_only_round_trip(ram_store):
    ram_store.put_block(key("a"), b"hello")
    assert ram_store.get_block(key("a")) == b"hello"
    assert ram_store.get_block(key("missing")) is None
    assert ram_store.stats.hot_hits == 1
    assert ram_store.stats.misses == 1


def test_hot_tier_evicts_least_recently_used_under_a_byte_budget():
    store = TwoTierStateStore(signature(), hot_budget_bytes=300)
    try:
        for name in ("a", "b", "c"):
            store.put_block(key(name), b"x" * 100)
        store.get_block(key("a"))  # a is now the most recent
        store.put_block(key("d"), b"x" * 100)
        assert store.get_block(key("b")) is None
        assert store.get_block(key("a")) == b"x" * 100
        assert store.hot_bytes() <= 300
        assert store.stats.hot_evictions == 1
    finally:
        store.close()


def test_pinned_entries_survive_eviction():
    store = TwoTierStateStore(signature(), hot_budget_bytes=200)
    try:
        store.put_block(key("pinned"), b"x" * 100)
        store.pin_block(key("pinned"))
        for name in ("a", "b", "c"):
            store.put_block(key(name), b"x" * 100)
        assert store.get_block(key("pinned")) == b"x" * 100
        store.unpin_block(key("pinned"))
        for name in ("d", "e", "f"):
            store.put_block(key(name), b"x" * 100)
        assert store.get_block(key("pinned")) is None
    finally:
        store.close()


def test_duplicate_put_does_not_store_twice(ram_store):
    ram_store.put_block(key("a"), b"payload")
    ram_store.put_block(key("a"), b"payload")
    assert ram_store.stats.puts == 1
    assert ram_store.stats.duplicate_puts == 1


def test_write_behind_reaches_disk_and_reads_back(tmp_path):
    store = make_disk_store(tmp_path)
    try:
        store.put_block(key("a"), b"durable")
        assert store.flush(2.0)
        assert store.pending_bytes() == 0
        assert store.stats.bytes_written > 0
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(tmp_path)
            for name in names
        ]
        assert len(files) == 1
        assert not os.path.basename(files[0]).startswith(".writing-")
        assert signature().slug() in files[0]
    finally:
        store.close()

    reopened = make_disk_store(tmp_path)
    try:
        assert reopened.get_block(key("a")) == b"durable"
        assert reopened.stats.ssd_hits == 1
    finally:
        reopened.close()


def test_backpressure_drops_the_write_and_keeps_the_bytes(tmp_path):
    store = make_disk_store(tmp_path, pending_budget_bytes=600, max_stall_s=0.02)
    try:
        store.set_writer_paused(True)
        store.put_block(key("a"), b"x" * 400)
        store.put_block(key("b"), b"x" * 400)
        assert store.stats.writes_dropped == 1
        assert store.stats.max_stall_s < 0.5
        assert store.get_block(key("b")) == b"x" * 400
        store.set_writer_paused(False)
        assert store.flush(2.0)
    finally:
        store.close()


def test_signature_mismatch_on_load_is_a_miss_not_a_crash(tmp_path):
    store = make_disk_store(tmp_path)
    try:
        store.put_block(key("a"), b"payload")
        assert store.flush(2.0)
        path = next(
            os.path.join(root, name)
            for root, _, names in os.walk(tmp_path)
            for name in names
        )
    finally:
        store.close()

    other = TwoTierStateStore(
        signature(snapshot_dtype="int8"),
        ssd_dir=str(tmp_path),
        hot_budget_bytes=1024,
    )
    try:
        # The other model's directory is a different slug, so point at the file
        # directly by writing it under the new slug with the old signature.
        target = path.replace(signature().slug(), signature(snapshot_dtype="int8").slug())
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(path, "rb") as handle:
            payload = handle.read()
        with open(target, "wb") as handle:
            handle.write(payload)
        other._scan_existing()  # the index is rebuilt at start in production
        assert other.get_block(key("a")) is None
        assert other.stats.signature_rejected == 1
    finally:
        other.close()


def test_corrupt_file_is_skipped_and_deleted(tmp_path):
    store = make_disk_store(tmp_path)
    try:
        store.put_block(key("a"), b"payload")
        assert store.flush(2.0)
        path = next(
            os.path.join(root, name)
            for root, _, names in os.walk(tmp_path)
            for name in names
        )
        with open(path, "r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            handle.write(b"\x00")
    finally:
        store.close()

    reopened = make_disk_store(tmp_path)
    try:
        assert reopened.get_block(key("a")) is None
        assert reopened.stats.corrupt_skipped == 1
        assert not os.path.exists(path)
    finally:
        reopened.close()


def test_half_written_temporary_files_are_cleaned_at_start(tmp_path):
    store = make_disk_store(tmp_path)
    try:
        store.put_block(key("a"), b"payload")
        assert store.flush(2.0)
        directory = os.path.dirname(
            next(
                os.path.join(root, name)
                for root, _, names in os.walk(tmp_path)
                for name in names
            )
        )
        stray = os.path.join(directory, ".writing-deadbeef.tkv")
        with open(stray, "wb") as handle:
            handle.write(b"half a record")
    finally:
        store.close()

    reopened = make_disk_store(tmp_path)
    try:
        assert not os.path.exists(stray)
        assert reopened.get_block(key("a")) == b"payload"
    finally:
        reopened.close()


def test_disk_tier_evicts_oldest_over_capacity(tmp_path):
    store = make_disk_store(tmp_path, ssd_capacity_bytes=900, hot_budget_bytes=1)
    try:
        for name in ("a", "b", "c"):
            store.put_block(key(name), b"x" * 300)
            assert store.flush(2.0)
        assert store.disk_bytes() <= 900
        assert store.stats.ssd_evictions >= 1
        assert store.get_block(key("c")) == b"x" * 300
    finally:
        store.close()


def test_snapshots_use_their_own_namespace(tmp_path):
    store = make_disk_store(tmp_path)
    try:
        store.put_snapshot("24576-abc", b"recurrent")
        assert store.flush(2.0)
        assert store.get_snapshot("24576-abc") == b"recurrent"
        assert store.get_block(key("24576-abc")) is None
        assert store.contains_snapshot("24576-abc")
    finally:
        store.close()


def test_record_written_by_hand_is_readable(tmp_path):
    """The format is the contract, not the writer that happens to produce it."""
    store = make_disk_store(tmp_path)
    try:
        path = store._path_for("b:" + bytes(key("a")).hex())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(
                encode_record(
                    kind="block",
                    key="b:" + bytes(key("a")).hex(),
                    signature=signature(),
                    tokens=8,
                    payload=b"by hand",
                )
            )
        store._scan_existing()
        assert store.get_block(key("a")) == b"by hand"
    finally:
        store.close()
