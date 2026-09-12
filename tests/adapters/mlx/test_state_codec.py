# SPDX-License-Identifier: MIT
"""The state codec, round tripped on arrays small enough to read.

Three claims, all of them things the cache would otherwise discover in
production. Bytes written by ``export`` come back as the same arrays, including
the bfloat16 the recurrent state actually uses and numpy has no dtype for. A
block range covers exactly the tokens it says it covers, so a prefix
reassembled from blocks equals the prefix that produced them. And a restore
that fails partway leaves the state empty rather than half filled, because a
state carrying three of eight blocks answers from a context nobody asked for.

Everything here builds the vendored cache objects directly with a handful of
values in them. No weights, no checkpoint, no forward pass.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from titan.adapters.cache.format import CacheSignature
from titan.adapters.mlx.codec import MLXStateCodec
from titan.adapters.mlx.payload import PayloadError, pack_arrays, unpack_arrays
from titan.adapters.mlx.state import ModelState, StateError
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.cache import ArraysCache
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.language import QSAKVCache

HEADS = 2
DIM = 3
INDEX_DIM = 2


def signature() -> CacheSignature:
    return CacheSignature(
        model_name="tiny",
        layer_layout=("gdn", "qsa"),
        block_tokens=4,
        snapshot_dtype="bfloat16",
    )


def kv_for(tokens: range | list[int]) -> tuple[mx.array, mx.array]:
    """Positional by construction: token t's row depends only on t."""
    ids = mx.array(list(tokens), dtype=mx.float32)
    base = ids.reshape(1, 1, -1, 1)
    keys = mx.broadcast_to(base, (1, HEADS, len(list(tokens)), DIM)) + 0.5
    values = keys * -1.0
    return keys.astype(mx.bfloat16), values.astype(mx.bfloat16)


def make_state(tokens: list[int]) -> ModelState:
    """One recurrent layer and one attention layer, filled by hand."""
    recurrent = ArraysCache(size=2)
    recurrent.cache = [
        mx.array([[float(sum(tokens))]], dtype=mx.bfloat16),
        mx.array([[float(len(tokens))]], dtype=mx.float32),
    ]
    attention = QSAKVCache()
    if not tokens:
        return ModelState(layers=[recurrent, attention], length=0)
    keys, values = kv_for(tokens)
    index_keys = mx.array(
        [[[float(t), float(t) + 0.25] for t in tokens]], dtype=mx.bfloat16
    )
    positions = mx.array([list(range(len(tokens)))], dtype=mx.int32)
    attention.state = (keys, values, index_keys, positions)
    state = ModelState(layers=[recurrent, attention], length=len(tokens))
    return state


def equal(a, b) -> bool:
    return bool(mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)))


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [mx.bfloat16, mx.float16, mx.float32, mx.int32, mx.uint8, mx.int64]
)
def test_every_dtype_the_caches_hold_survives_the_container(dtype):
    original = mx.array([[1, 2, 3], [4, 5, 6]]).astype(dtype)
    _header, arrays = unpack_arrays(pack_arrays({"kind": "t"}, {"a": original}))
    assert arrays["a"].dtype == dtype
    assert arrays["a"].shape == (2, 3)
    assert equal(arrays["a"], original)


def test_the_header_travels_with_the_arrays():
    header, arrays = unpack_arrays(
        pack_arrays({"kind": "blocks", "start": 4, "end": 8}, {})
    )
    assert header == {"kind": "blocks", "start": 4, "end": 8}
    assert arrays == {}


def test_a_payload_from_another_build_is_refused():
    with pytest.raises(PayloadError):
        unpack_arrays(b"NOTTITAN" + b"\x00" * 8)
    good = pack_arrays({"kind": "t"}, {})
    with pytest.raises(PayloadError):
        unpack_arrays(good[:8] + bytes([99]) + good[9:])


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def test_a_snapshot_round_trips_through_the_codec():
    codec = MLXStateCodec(signature())
    state = make_state([1, 2, 3, 4])
    state.stage_snapshot()
    blob = codec.export_snapshot(state, 4)

    fresh = make_state([9, 9, 9, 9])
    codec.import_blocks(fresh, 0, 4, codec.export_blocks(state, 0, 4))
    codec.import_snapshot(fresh, 4, blob)
    for original, restored in zip(
        state.recurrent_caches()[0].cache, fresh.recurrent_caches()[0].cache
    ):
        assert equal(original, restored)
    assert fresh.length == 4


def test_exporting_a_snapshot_nobody_staged_is_an_error():
    codec = MLXStateCodec(signature())
    state = make_state([1, 2, 3, 4])
    with pytest.raises(StateError, match="no recurrent snapshot"):
        codec.export_snapshot(state, 4)


def test_a_snapshot_for_another_length_is_refused():
    codec = MLXStateCodec(signature())
    state = make_state([1, 2, 3, 4])
    state.stage_snapshot()
    blob = codec.export_snapshot(state, 4)
    fresh = make_state([1, 2, 3, 4])
    codec.import_blocks(fresh, 0, 4, codec.export_blocks(state, 0, 4))
    with pytest.raises(StateError, match="covers 4 tokens, not 8"):
        codec.import_snapshot(fresh, 8, blob)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


def test_a_block_carries_exactly_the_tokens_it_names():
    codec = MLXStateCodec(signature())
    state = make_state(list(range(8)))
    _header, arrays = unpack_arrays(codec.export_blocks(state, 4, 8))
    keys = arrays["l1.keys"]
    assert keys.shape == (1, HEADS, 4, DIM)
    assert equal(keys, kv_for(list(range(4, 8)))[0])
    assert arrays["l1.index_positions"].tolist() == [[4, 5, 6, 7]]


def test_a_prefix_reassembled_from_blocks_equals_the_one_that_produced_it():
    """The whole reason attention KV is stored per block: it is positional, so
    two blocks stuck back together are the eight tokens that made them."""
    codec = MLXStateCodec(signature())
    tokens = list(range(8))
    state = make_state(tokens)
    state.stage_snapshot()
    blocks = [codec.export_blocks(state, 0, 4), codec.export_blocks(state, 4, 8)]
    snapshot = codec.export_snapshot(state, 8)

    fresh = make_state([0])
    for index, payload in enumerate(blocks):
        codec.import_blocks(fresh, index * 4, index * 4 + 4, payload)
    codec.import_snapshot(fresh, 8, snapshot)

    original = state.attention_caches()[0]
    restored = fresh.attention_caches()[0]
    assert restored.offset == 8
    assert equal(restored.state[0], original.state[0])
    assert equal(restored.state[1], original.state[1])
    assert equal(restored.state[2], original.state[2])
    assert restored.state[3].tolist() == original.state[3].tolist()


def test_blocks_are_applied_only_when_the_snapshot_lands():
    """A restore that fails partway leaves nothing behind. The cache counts the
    failure and prefills from zero, which is only safe if zero is the truth."""
    codec = MLXStateCodec(signature())
    state = make_state(list(range(8)))
    state.stage_snapshot()

    fresh = make_state([])
    codec.import_blocks(fresh, 0, 4, codec.export_blocks(state, 0, 4))
    assert fresh.attention_caches()[0].offset == 0
    assert fresh.length == 0
    with pytest.raises(StateError, match="only 4 were staged"):
        codec.import_snapshot(fresh, 8, codec.export_snapshot(state, 8))
    assert fresh.attention_caches()[0].offset == 0
    assert fresh.length == 0


def test_a_block_that_skips_one_is_refused():
    """Blocks arrive in ascending order and each starts where the last ended.
    A gap in the middle is a prefix that never existed."""
    codec = MLXStateCodec(signature())
    state = make_state(list(range(12)))
    fresh = make_state([])
    codec.import_blocks(fresh, 0, 4, codec.export_blocks(state, 0, 4))
    with pytest.raises(StateError, match="does not follow the staged prefix"):
        codec.import_blocks(fresh, 8, 12, codec.export_blocks(state, 8, 12))


def test_a_block_payload_read_at_the_wrong_offset_is_refused():
    codec = MLXStateCodec(signature())
    state = make_state(list(range(8)))
    fresh = make_state([])
    with pytest.raises(StateError, match="covers"):
        codec.import_blocks(fresh, 0, 4, codec.export_blocks(state, 4, 8))


def test_exporting_past_what_the_state_covers_is_an_error():
    codec = MLXStateCodec(signature())
    state = make_state(list(range(4)))
    with pytest.raises(StateError, match="covers 4 tokens"):
        codec.export_blocks(state, 4, 8)


def test_the_codec_refuses_a_state_it_did_not_come_from():
    codec = MLXStateCodec(signature())
    with pytest.raises(StateError, match="needs a ModelState"):
        codec.export_blocks(object(), 0, 4)


def test_the_codec_resolves_the_handle_the_cache_hands_back():
    """The cache calls the codec with whatever the engine gave it, and the
    engine deals in opaque handles. Resolving one is the backend's table."""
    state = make_state(list(range(4)))
    codec = MLXStateCodec(signature(), {7: state}.get)
    _header, arrays = unpack_arrays(codec.export_blocks(7, 0, 4))
    assert arrays["l1.keys"].shape == (1, HEADS, 4, DIM)
    with pytest.raises(StateError, match="live handle"):
        codec.export_blocks(9, 0, 4)


def test_the_signature_is_what_the_store_stamps():
    codec = MLXStateCodec(signature())
    assert codec.signature.layer_layout == ("gdn", "qsa")
    assert codec.signature.digest() == signature().digest()
