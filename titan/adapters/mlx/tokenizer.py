# SPDX-License-Identifier: MIT
"""Tokenizer adapter: the Rust ``tokenizers`` library plus a streaming detokeniser.

Three decisions are worth stating, because each one was a bug somewhere else
first.

**The fast tokenizer, loaded from ``tokenizer.json`` directly.** No transformers
import lives under ``titan/``. ``tokenizers.Tokenizer.from_file`` gives the same
Rust object transformers would wrap, without the slow-path fallback, without the
``AutoTokenizer`` registry lookup and without pulling a large dependency into the
serving process. Loading the 12 MB file takes about 0.15 s.

**Detokenisation goes through a byte table, not through repeated ``decode``.**
This checkpoint is byte-level BPE with a ByteLevel decoder, so a token id maps to
a fixed byte string: either the added token's literal UTF-8, or the vocabulary
entry run back through the GPT-2 byte alphabet. The table is built once at load.
Streaming then becomes byte concatenation plus an incremental UTF-8 decoder,
which is O(1) per token and cannot disagree with :meth:`decode` because both use
the same table. Re-decoding a growing id list on every step is the alternative
and it is quadratic.

**The held-back suffix is the decoder's own state.** An incremental UTF-8 decoder
holds an incomplete multi-byte sequence and releases it when the continuation
bytes arrive, which is exactly the rule the port asks for: an emoji split across
three tokens produces nothing, nothing, then the character. Byte-level BPE has no
leading-space or byte-fallback merge to revise on top of that, so nothing else
has to be withheld for correctness. Callers that must also withhold text for
another reason (a stop string that could still complete across an emit boundary)
pass ``hold``, and those characters stay in the stream until they are either
released by later text or dropped at flush.

Concurrency: one :class:`FastTokenizer` instance serves every sequence. The Rust
tokenizer is immutable once loaded, the byte table is a read-only tuple, and the
only mutable state is one small per-sequence stream object in a dict behind a
lock. Nothing is deep-copied per sequence, which is what the overlay did and what
made its detokenisation cost show up in the cycle profile.
"""

from __future__ import annotations

import codecs
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from titan.core.errors import ConfigError
from titan.core.types import SequenceId

__all__ = ["FastTokenizer", "load_tokenizer", "byte_decoder"]


def _byte_to_unicode() -> dict[int, str]:
    """The GPT-2 byte alphabet. Every byte gets one printable code point."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + spare)
            spare += 1
    return mapping


def byte_decoder() -> dict[str, int]:
    """Code point back to byte. The inverse of the ByteLevel encoder."""
    return {c: b for b, c in _byte_to_unicode().items()}


@dataclass(slots=True)
class _Stream:
    """Detokenisation state for one sequence. Touched by one thread at a time."""

    decoder: codecs.IncrementalDecoder
    pending: str = ""
    """Decoded characters not yet handed to the caller."""


@dataclass(slots=True)
class _Loaded:
    """Everything read off disk, resolved once."""

    tokenizer: object
    table: tuple[bytes, ...]
    special_ids: frozenset[int]
    eos_ids: frozenset[int]
    vocab_size: int


def _load(model_dir: Path) -> _Loaded:
    try:
        from tokenizers import Tokenizer as RustTokenizer
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ConfigError(
            "the 'tokenizers' package is required by titan.adapters.mlx.tokenizer"
        ) from exc

    model_dir = Path(model_dir).expanduser()
    if model_dir.is_file():
        tokenizer_file = model_dir
        model_dir = model_dir.parent
    else:
        tokenizer_file = model_dir / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise ConfigError(f"no tokenizer.json under {model_dir}")

    tokenizer = RustTokenizer.from_file(str(tokenizer_file))
    raw = json.loads(tokenizer_file.read_text(encoding="utf-8"))

    added = {int(a["id"]): a for a in raw.get("added_tokens", ())}
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    size = max(vocab.values()) + 1 if vocab else 0

    to_byte = byte_decoder()
    table: list[bytes] = [b""] * size
    for text, token_id in vocab.items():
        entry = added.get(token_id)
        if entry is not None:
            table[token_id] = str(entry["content"]).encode("utf-8")
            continue
        try:
            table[token_id] = bytes(to_byte[ch] for ch in text)
        except KeyError:
            # Not a byte-level piece. Nothing in this checkpoint hits this, but
            # a vocabulary with literal pieces should still round-trip.
            table[token_id] = text.encode("utf-8")

    special = frozenset(
        token_id for token_id, entry in added.items() if entry.get("special")
    )
    return _Loaded(
        tokenizer=tokenizer,
        table=tuple(table),
        special_ids=special,
        eos_ids=_eos_ids(model_dir, vocab),
        vocab_size=size,
    )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _eos_ids(model_dir: Path, vocab: dict[str, int]) -> frozenset[int]:
    """Every id that ends a turn, from the checkpoint rather than from a guess.

    ``config.json`` carries a list on this model (``[248046, 248044]``: the chat
    terminator and the raw end-of-text), and ``tokenizer_config.json`` names one
    by string. Both are taken. Missing one of them is how a server ends up
    generating past the end of a turn on a request that took the other path.
    """
    ids: set[int] = set()

    for source in ("config.json", "generation_config.json"):
        value = _read_json(model_dir / source).get("eos_token_id")
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list):
            ids.update(int(v) for v in value if isinstance(v, int))

    config = _read_json(model_dir / "tokenizer_config.json")
    token = config.get("eos_token")
    if isinstance(token, dict):
        token = token.get("content")
    if isinstance(token, str) and token in vocab:
        ids.add(int(vocab[token]))

    return frozenset(ids)


class FastTokenizer:
    """``Tokenizer`` port over the Rust tokenizer. One instance per process."""

    def __init__(self, loaded: _Loaded) -> None:
        self._t = loaded.tokenizer
        self._table = loaded.table
        self._special = loaded.special_ids
        self._eos = loaded.eos_ids
        self._vocab_size = loaded.vocab_size
        self._streams: dict[SequenceId, _Stream] = {}
        self._lock = threading.Lock()

    # -- ids ---------------------------------------------------------------
    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return self._t.encode(text, add_special_tokens=add_special).ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = False) -> str:
        """Whole-sequence detokenisation, through the same table streaming uses.

        ``skip_special`` drops the ``<|...|>`` control tokens. It is off by
        default: the engine's own round-trip checks want what the ids actually
        say, and the API layer is the place that decides a terminator is not
        content. Note that ``<think>`` is not a special token in this
        checkpoint, so the reasoning channel is never affected either way.
        """
        return self._bytes(ids, skip_special=skip_special).decode(
            "utf-8", errors="replace"
        )

    def _bytes(self, ids: Iterable[int], *, skip_special: bool = False) -> bytes:
        table = self._table
        size = len(table)
        out = bytearray()
        for token_id in ids:
            i = int(token_id)
            if not 0 <= i < size:
                raise ValueError(f"token id {i} is outside the vocabulary")
            if skip_special and i in self._special:
                continue
            out += table[i]
        return bytes(out)

    # -- streaming ---------------------------------------------------------
    def decode_incremental(
        self,
        seq: SequenceId,
        ids: Sequence[int],
        *,
        hold: int = 0,
        skip_special: bool = False,
    ) -> str:
        """Append ``ids`` to ``seq``'s stream and return what is safe to emit.

        Safe means two things. The text is complete UTF-8, so a character whose
        bytes span three tokens appears once, whole. And it is final: byte-level
        BPE never revises an earlier piece, so nothing already returned can
        change. ``hold`` keeps that many trailing characters back for a caller
        matching stop strings across emits; they are returned by a later call or
        by :meth:`flush_incremental`.
        """
        if hold < 0:
            raise ValueError("hold must not be negative")
        stream = self._stream(seq)
        stream.pending += stream.decoder.decode(
            self._bytes(ids, skip_special=skip_special)
        )
        if hold == 0:
            out, stream.pending = stream.pending, ""
            return out
        if len(stream.pending) <= hold:
            return ""
        cut = len(stream.pending) - hold
        out, stream.pending = stream.pending[:cut], stream.pending[cut:]
        return out

    def flush_incremental(self, seq: SequenceId) -> str:
        """Emit everything held back and forget the sequence.

        A truncated multi-byte sequence at the very end (the model stopped
        mid-character, or a stop condition cut the stream) becomes one
        replacement character rather than being dropped silently or raising.
        """
        with self._lock:
            stream = self._streams.pop(seq, None)
        if stream is None:
            return ""
        return stream.pending + stream.decoder.decode(b"", final=True)

    def _stream(self, seq: SequenceId) -> _Stream:
        with self._lock:
            stream = self._streams.get(seq)
            if stream is None:
                stream = _Stream(
                    decoder=codecs.getincrementaldecoder("utf-8")("replace")
                )
                self._streams[seq] = stream
            return stream

    @property
    def open_streams(self) -> int:
        """Sequences with live detokenisation state. Leak check for tests."""
        with self._lock:
            return len(self._streams)

    # -- properties --------------------------------------------------------
    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special


def load_tokenizer(model_dir: str | Path) -> FastTokenizer:
    """Open a checkpoint's tokenizer. Accepts the directory or the json file."""
    return FastTokenizer(_load(Path(model_dir)))
