"""Chain hashing and the record wrapper."""

from __future__ import annotations

import pytest

from titan.adapters.cache.format import (
    RecordError,
    chain_hash,
    decode_record,
    encode_record,
    snapshot_id_for,
)

from tests.cache.fakes import signature


def test_chain_hash_identifies_a_prefix_not_a_window():
    sig = signature().digest()
    first = chain_hash(None, [1, 2, 3], sig)
    same_block_other_parent = chain_hash(first, [1, 2, 3], sig)
    assert first != same_block_other_parent

    left = chain_hash(chain_hash(None, [1, 2], sig), [3, 4], sig)
    right = chain_hash(chain_hash(None, [1, 2], sig), [3, 4], sig)
    assert left == right
    diverged = chain_hash(chain_hash(None, [1, 2], sig), [3, 5], sig)
    assert left != diverged


def test_chain_hash_depends_on_the_signature():
    ids = [7, 8, 9]
    assert chain_hash(None, ids, signature().digest()) != chain_hash(
        None, ids, signature(block_tokens=16).digest()
    )
    assert chain_hash(None, ids, signature().digest()) != chain_hash(
        None, ids, signature(layer_layout=("gdn", "qsa", "qsa")).digest()
    )
    assert chain_hash(None, ids, signature().digest()) != chain_hash(
        None, ids, signature(snapshot_dtype="bf16").digest()
    )


def test_short_block_cannot_collide_with_a_full_one():
    sig = signature().digest()
    assert chain_hash(None, [1, 2], sig) != chain_hash(None, [1, 2, 0], sig)


def test_record_round_trip():
    sig = signature()
    raw = encode_record(
        kind="block", key="b:abc", signature=sig, tokens=8, payload=b"payload"
    )
    header, payload = decode_record(raw, signature=sig, expect_key="b:abc")
    assert payload == b"payload"
    assert header.kind == "block"
    assert header.tokens == 8


def test_record_from_another_signature_is_rejected():
    raw = encode_record(
        kind="block", key="b:abc", signature=signature(), tokens=8, payload=b"x"
    )
    with pytest.raises(RecordError, match="signature"):
        decode_record(raw, signature=signature(snapshot_dtype="int8"))


def test_truncated_and_corrupt_records_are_rejected():
    sig = signature()
    raw = encode_record(
        kind="snapshot", key="s:1", signature=sig, tokens=8, payload=b"0123456789"
    )
    with pytest.raises(RecordError):
        decode_record(raw[:-3], signature=sig)
    flipped = bytearray(raw)
    flipped[-1] ^= 0xFF
    with pytest.raises(RecordError, match="checksum"):
        decode_record(bytes(flipped), signature=sig)
    with pytest.raises(RecordError, match="not a Titan"):
        decode_record(b"garbage" + raw, signature=sig)


def test_record_holding_another_key_is_rejected():
    sig = signature()
    raw = encode_record(kind="block", key="b:aa", signature=sig, tokens=0, payload=b"")
    with pytest.raises(RecordError, match="different key"):
        decode_record(raw, signature=sig, expect_key="b:bb")


def test_snapshot_id_is_content_addressed():
    digest = chain_hash(None, [1, 2, 3], signature().digest())
    assert snapshot_id_for(digest, 512) == snapshot_id_for(digest, 512)
    assert snapshot_id_for(digest, 512) != snapshot_id_for(digest, 1024)


def test_model_slug_is_filesystem_safe():
    slug = signature(model_name="qwen/3.8 flash:next").slug()
    assert "/" not in slug and ":" not in slug and " " not in slug
