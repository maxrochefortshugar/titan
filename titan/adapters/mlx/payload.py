"""The raw container every cache payload is written in.

One JSON header naming each array's dtype, shape and byte range, then the
arrays back to back. A zip file (which is what ``numpy.savez`` writes) spends
real time deflating a 110 MiB block of dense floats for no gain, and the
store's own record wrapper already carries the checksum a zip's directory would
duplicate.

The dtype handling is the part worth knowing. numpy has no bfloat16 and mlx
will not hand one to the buffer protocol, so every array crosses as its raw
bits under a same-width unsigned view and comes back through the same view.
The width is what is preserved; the name travels in the header.

Nothing in this module knows what a cache block or a recurrent snapshot is. It
is shared by the state codec and by :class:`~titan.adapters.mlx.state.ModelState`
so that the two cannot drift into writing different bytes for the same arrays.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from titan.core.errors import SnapshotError

__all__ = ["PayloadError", "PAYLOAD_MAGIC", "PAYLOAD_VERSION", "pack_arrays", "unpack_arrays"]


class PayloadError(SnapshotError, ValueError):
    """A payload could not be read. Always handled by the cache, never fatal.

    Inside the ``TitanError`` tree so that a payload from another build ends a
    restore rather than the scheduler thread, and a ``ValueError`` as well
    because that is what a caller reaching for the container directly would
    expect to catch.
    """


PAYLOAD_MAGIC = b"TITANARR"
PAYLOAD_VERSION = 1

_KIND_BLOCKS = "blocks"
_KIND_SNAPSHOT = "snapshot"

# Same-width unsigned view for every dtype the caches hold. numpy has no
# bfloat16, and mlx will not hand one to the buffer protocol, so an array
# crosses as its raw bits and comes back through the same view. The width is
# what matters, not the name.
_VIEW_BY_SIZE = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
_NUMPY_BY_SIZE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _dtype_from_name(name: str) -> Any:
    dtype = getattr(mx, name, None)
    if dtype is None:
        raise PayloadError(f"payload names a dtype this build does not have: {name}")
    return dtype


def pack_arrays(meta: Mapping[str, Any], arrays: Mapping[str, mx.array]) -> bytes:
    """One header plus one contiguous run of raw bytes per array.

    A JSON header rather than npz because npz is a zip file: a 110 MiB
    recurrent snapshot spends real time being deflated for a payload that is
    already dense floats, and the store's own record wrapper carries the
    checksum that a zip's would duplicate.
    """
    entries: list[dict[str, Any]] = []
    buffers: list[memoryview] = []
    offset = 0
    ordered = sorted(arrays.items())
    if ordered:
        mx.eval([value for _name, value in ordered])
    for name, value in ordered:
        size = value.dtype.size
        view = _VIEW_BY_SIZE.get(size)
        if view is None:  # pragma: no cover - defensive
            raise PayloadError(f"no raw view for a {size}-byte dtype")
        # One copy, not two. The host array wraps the same unified memory the
        # device array holds, and the byte view of it is handed straight to
        # ``join``, so the payload is materialised exactly once. The old
        # ``.tobytes()`` here copied a 110 MiB snapshot into a temporary that
        # ``join`` then copied again, on the scheduler thread both times.
        host = np.ascontiguousarray(np.array(value.view(view), copy=False))
        raw = memoryview(host).cast("B")
        entries.append(
            {
                "name": name,
                "dtype": _dtype_name(value.dtype),
                "shape": list(value.shape),
                "offset": offset,
                "nbytes": raw.nbytes,
            }
        )
        buffers.append(raw)
        offset += raw.nbytes
    header = json.dumps(
        {**dict(meta), "arrays": entries}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return b"".join(
        [
            PAYLOAD_MAGIC,
            bytes([PAYLOAD_VERSION]),
            len(header).to_bytes(4, "little"),
            header,
            *buffers,
        ]
    )


def unpack_arrays(payload: bytes) -> tuple[dict[str, Any], dict[str, mx.array]]:
    """Reverse of :func:`pack_arrays`, evaluated before it returns.

    Every array is materialised on the device here, so the caller may drop the
    payload the moment this returns. That is not an optimisation: an array left
    attached to the buffer it was read from outlives the buffer the first time
    the store evicts one.
    """
    head = len(PAYLOAD_MAGIC)
    if payload[:head] != PAYLOAD_MAGIC:
        raise PayloadError("not a Titan state payload")
    version = payload[head]
    if version != PAYLOAD_VERSION:
        raise PayloadError(f"state payload version {version} came from another build")
    start = head + 1
    header_len = int.from_bytes(payload[start : start + 4], "little")
    start += 4
    header = json.loads(payload[start : start + header_len].decode("utf-8"))
    body = start + header_len

    arrays: dict[str, mx.array] = {}
    for entry in header.pop("arrays", ()):
        dtype = _dtype_from_name(entry["dtype"])
        size = dtype.size
        begin = body + int(entry["offset"])
        raw = payload[begin : begin + int(entry["nbytes"])]
        flat = np.frombuffer(raw, dtype=_NUMPY_BY_SIZE[size])
        shape = tuple(int(n) for n in entry["shape"])
        arrays[entry["name"]] = mx.array(flat.reshape(shape)).view(dtype)
    if arrays:
        mx.eval(list(arrays.values()))
    return header, arrays


