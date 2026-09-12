"""Hashing and the on-disk record format for the cache tiers.

Two things live here, and nothing else. The first is the chain hash that names
a block: sha256 over the parent block's hash, a compatibility signature and the
block's token ids, so one digest identifies a whole prefix rather than a
512-token window. Two prompts that share their first 24576 tokens share the
first 48 digests and diverge at the 49th, which is the entire matching
algorithm.

The second is the record wrapper the SSD tier writes. A record is a small
header plus an opaque payload the state codec produced. The header carries the
compatibility signature and a crc32 of the payload, so a file written by
another build, another layer layout or another block size is rejected on load
rather than misread, and a file that was truncated by a crash mid-write is
skipped. Both cases count as a miss and shorten the match; neither raises.

crc32 rather than sha256 on the payload is deliberate. A recurrent snapshot is
around 110 MiB and the load happens on the scheduler thread, where sha256 would
cost about 200 ms and crc32 costs under 20.

Nothing in this module imports mlx or numpy. The payload is bytes by the time
it arrives, which is what lets the writer thread stay away from the GPU.
"""

from __future__ import annotations

import json
import sys
import zlib
from array import array
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "ROOT_SEED",
    "CacheSignature",
    "RecordHeader",
    "chain_hash",
    "encode_record",
    "decode_record",
    "snapshot_id_for",
]

MAGIC = b"TITANKV"
FORMAT_VERSION = 1
ROOT_SEED = b"titan-prefix-root"

_HEADER_LEN_BYTES = 4
_PREAMBLE = len(MAGIC) + 1 + _HEADER_LEN_BYTES


class RecordError(ValueError):
    """A stored record could not be trusted. Always handled, never propagated.

    The store catches this, counts it and returns ``None``. A caller that sees
    ``None`` recomputes; that is the whole recovery path.
    """


@dataclass(frozen=True, slots=True)
class CacheSignature:
    """Everything that must match for a stored byte string to be meaningful.

    A block payload is a serialisation of one model's KV layout. Reading it
    back into a different model, a different number of layers, a different
    block size or a different snapshot dtype produces silent nonsense, so the
    signature is hashed into every block name and written into every file
    header, and a load whose header disagrees is a miss.

    ``layer_layout`` is the per-layer cache kind in layer order, for example
    ``("gdn", "gdn", "qsa", ...)``. Layer count and ordering both matter: a
    build that moves one attention layer produces different bytes at the same
    offsets.
    """

    model_name: str
    layer_layout: tuple[str, ...]
    block_tokens: int
    snapshot_dtype: str
    format_version: int = FORMAT_VERSION

    def digest(self) -> bytes:
        """Stable 32-byte digest. Fed into every chain hash and every header."""
        canonical = json.dumps(
            {
                "model_name": self.model_name,
                "layer_layout": list(self.layer_layout),
                "block_tokens": self.block_tokens,
                "snapshot_dtype": self.snapshot_dtype,
                "format_version": self.format_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).digest()

    def hex(self) -> str:
        return self.digest().hex()

    def slug(self) -> str:
        """Directory name for this model. Filesystem safe, human readable."""
        safe = "".join(
            ch if ch.isalnum() or ch in "-_." else "-" for ch in self.model_name
        )
        return f"{safe or 'model'}-{self.hex()[:12]}"


def _ids_bytes(ids: Sequence[int]) -> bytes:
    """Token ids as little-endian int32, whatever the host is."""
    buffer = array("i", ids)
    if sys.byteorder != "little":  # pragma: no cover - Apple Silicon is little
        buffer.byteswap()
    return buffer.tobytes()


def chain_hash(
    parent: bytes | None,
    ids: Sequence[int],
    signature_digest: bytes,
) -> bytes:
    """Hash of the block covering ``ids`` whose predecessor hashed to ``parent``.

    ``parent`` is ``None`` for the first block of a sequence, which seeds with a
    constant so that block zero of every sequence starts from the same place.
    The token count goes into the digest separately from the ids, so a short
    terminal block can never collide with a full one that happens to start the
    same way.
    """
    hasher = sha256()
    hasher.update(parent if parent is not None else ROOT_SEED)
    hasher.update(signature_digest)
    hasher.update(len(ids).to_bytes(4, "little"))
    hasher.update(_ids_bytes(ids))
    return hasher.digest()


def snapshot_id_for(block_hash: bytes, length: int) -> str:
    """Name of the recurrent snapshot taken at the end of ``block_hash``.

    Keyed by content, not by request. Two requests that arrive at the same
    prefix name the same snapshot, so the second one stores nothing.
    """
    return f"{length}-{block_hash.hex()}"


@dataclass(frozen=True, slots=True)
class RecordHeader:
    kind: str
    """``"block"`` or ``"snapshot"``. Only used for diagnostics and fan-out."""
    key: str
    signature: str
    tokens: int
    payload_len: int
    payload_crc: int


def encode_record(
    *,
    kind: str,
    key: str,
    signature: CacheSignature,
    tokens: int,
    payload: bytes,
) -> bytes:
    """Wrap ``payload`` in a header. Pure bytes work, safe on any thread."""
    header = {
        "kind": kind,
        "key": key,
        "signature": signature.hex(),
        "tokens": tokens,
        "payload_len": len(payload),
        "payload_crc": zlib.crc32(payload) & 0xFFFFFFFF,
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"".join(
        [
            MAGIC,
            bytes([FORMAT_VERSION]),
            len(blob).to_bytes(_HEADER_LEN_BYTES, "little"),
            blob,
            payload,
        ]
    )


def decode_record(
    raw: bytes,
    *,
    signature: CacheSignature,
    expect_key: str | None = None,
) -> tuple[RecordHeader, bytes]:
    """Unwrap a record, or raise :class:`RecordError` if it cannot be trusted.

    Every check here has a failure mode behind it. A wrong magic or version is
    a file from another build. A signature mismatch is the same bytes under a
    different model or block size. A key mismatch is a hash collision or a
    misplaced file. A short payload is a crash between the write and the
    rename, and a bad crc is a corrupt block on disk.
    """
    if len(raw) < _PREAMBLE:
        raise RecordError("record shorter than its preamble")
    if raw[: len(MAGIC)] != MAGIC:
        raise RecordError("not a Titan cache record")
    version = raw[len(MAGIC)]
    if version != FORMAT_VERSION:
        raise RecordError(f"record format {version} was written by another build")
    header_len = int.from_bytes(raw[len(MAGIC) + 1 : _PREAMBLE], "little")
    start = _PREAMBLE + header_len
    if header_len <= 0 or start > len(raw):
        raise RecordError("record header length is out of range")
    try:
        fields: Mapping[str, Any] = json.loads(raw[_PREAMBLE:start].decode("utf-8"))
        header = RecordHeader(
            kind=str(fields["kind"]),
            key=str(fields["key"]),
            signature=str(fields["signature"]),
            tokens=int(fields["tokens"]),
            payload_len=int(fields["payload_len"]),
            payload_crc=int(fields["payload_crc"]),
        )
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise RecordError(f"unreadable record header: {exc}") from exc

    if header.signature != signature.hex():
        raise RecordError("record was written under a different cache signature")
    if expect_key is not None and header.key != expect_key:
        raise RecordError("record holds a different key than the one requested")

    payload = raw[start:]
    if len(payload) != header.payload_len:
        raise RecordError(
            f"payload is {len(payload)} bytes, header says {header.payload_len}"
        )
    if (zlib.crc32(payload) & 0xFFFFFFFF) != header.payload_crc:
        raise RecordError("payload checksum mismatch")
    return header, payload
