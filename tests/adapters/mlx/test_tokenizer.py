"""Tokenizer adapter: ids, eos resolution and streaming detokenisation.

Most of this runs against a tokenizer built in the test rather than the
checkpoint's, and deliberately so. Its vocabulary is the 256 raw bytes, one
token each, which makes every multi-byte character a split across tokens by
construction: an emoji is four separate emits and a held-back partial has to
survive three of them. The real 248k-entry tokenizer would need a lucky string
to reach the same state, and it is not present on every machine.

The checkpoint's own tokenizer is then used, when it exists, for the two things
the synthetic one cannot check: that the byte table agrees with the Rust
library's ``decode`` over the whole vocabulary, and that the eos ids are the two
this model actually ends turns on.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import pytest

from titan.adapters.mlx.tokenizer import byte_decoder, load_tokenizer
from titan.core.errors import ConfigError
from titan.core.types import SequenceId

MODEL_DIR = Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"

IM_END = 256
ENDOFTEXT = 257
THINK = 258


@pytest.fixture(scope="module")
def byte_model(tmp_path_factory) -> Path:
    """A byte-level BPE checkpoint with no merges: one token per byte."""
    to_char = {b: c for c, b in byte_decoder().items()}
    vocab = {to_char[b]: b for b in range(256)}
    added = [
        {"id": IM_END, "content": "<|im_end|>", "special": True},
        {"id": ENDOFTEXT, "content": "<|endoftext|>", "special": True},
        {"id": THINK, "content": "<think>", "special": False},
    ]
    for entry in added:
        entry.update(single_word=False, lstrip=False, rstrip=False, normalized=False)
    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    directory = tmp_path_factory.mktemp("byte-model")
    (directory / "tokenizer.json").write_text(json.dumps(spec))
    (directory / "config.json").write_text(
        json.dumps({"eos_token_id": [IM_END, ENDOFTEXT]})
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"})
    )
    return directory


@pytest.fixture(scope="module")
def tok(byte_model):
    return load_tokenizer(byte_model)


def ids_of(tok, text: str) -> list[int]:
    return tok.encode(text)


# ---------------------------------------------------------------------------
# loading and properties
# ---------------------------------------------------------------------------


def test_missing_tokenizer_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_tokenizer(tmp_path)


def test_accepts_the_json_file_as_well_as_the_directory(byte_model):
    direct = load_tokenizer(byte_model / "tokenizer.json")
    assert direct.vocab_size == load_tokenizer(byte_model).vocab_size


def test_vocab_size_counts_added_tokens(tok):
    assert tok.vocab_size == THINK + 1


def test_eos_ids_come_from_config_and_tokenizer_config(tok):
    assert tok.eos_token_ids == frozenset({IM_END, ENDOFTEXT})


def test_special_ids_exclude_the_thinking_marker(tok):
    assert tok.special_token_ids == frozenset({IM_END, ENDOFTEXT})


# ---------------------------------------------------------------------------
# encode and decode
# ---------------------------------------------------------------------------


def test_round_trip_ascii(tok):
    assert tok.decode(ids_of(tok, "hello world")) == "hello world"


@pytest.mark.parametrize(
    "text",
    [
        "café",
        "日本語のテスト",
        "🌍🚀👩‍💻",
        "mixed ascii, café, 日本語 and 🌍 in one line",
        "\n\ttabs and newlines\r\n",
        "   leading and trailing spaces   ",
    ],
)
def test_round_trip_multibyte(tok, text):
    assert tok.decode(ids_of(tok, text)) == text


def test_one_token_per_byte(tok):
    # the premise the streaming tests rest on
    assert len(ids_of(tok, "🌍")) == 4


def test_decode_can_skip_special_tokens(tok):
    ids = ids_of(tok, "done") + [IM_END]
    assert tok.decode(ids) == "done<|im_end|>"
    assert tok.decode(ids, skip_special=True) == "done"


def test_decode_rejects_an_out_of_range_id(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_decode_empty(tok):
    assert tok.decode([]) == ""


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

SEQ = SequenceId(1)


def stream(tok, ids, *, seq=SEQ, hold=0, chunk=1):
    """Feed ``ids`` in chunks and return the list of emitted deltas."""
    out = []
    for i in range(0, len(ids), chunk):
        out.append(tok.decode_incremental(seq, ids[i : i + chunk], hold=hold))
    out.append(tok.flush_incremental(seq))
    return out


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "text",
    [
        "plain ascii text",
        "🌍 at the start",
        "at the end 🌍",
        "café 日本語 🚀 mixed",
        "👩‍💻 zero width joiner",
    ],
)
def test_incremental_concatenation_equals_decode(tok, text, chunk):
    ids = ids_of(tok, text)
    assert "".join(stream(tok, ids, chunk=chunk)) == text


def test_partial_utf8_is_never_emitted(tok):
    ids = ids_of(tok, "a🌍b")
    seq = SequenceId(2)
    deltas = [tok.decode_incremental(seq, [i]) for i in ids]
    tok.flush_incremental(seq)
    # one byte of the emoji at a time: nothing until the fourth
    assert deltas == ["a", "", "", "", "🌍", "b"]
    assert all("�" not in d for d in deltas)


def test_flush_replaces_a_truncated_character(tok):
    seq = SequenceId(3)
    truncated = ids_of(tok, "🌍")[:2]
    assert tok.decode_incremental(seq, truncated) == ""
    assert tok.flush_incremental(seq) == "�"


def test_flush_drops_the_stream_state(tok):
    seq = SequenceId(4)
    tok.decode_incremental(seq, ids_of(tok, "abc"))
    assert tok.open_streams >= 1
    tok.flush_incremental(seq)
    assert tok.flush_incremental(seq) == ""


def test_streams_are_independent(tok):
    a, b = SequenceId(10), SequenceId(11)
    ids = ids_of(tok, "🌍")
    assert tok.decode_incremental(a, ids[:2]) == ""
    assert tok.decode_incremental(b, ids_of(tok, "hi")) == "hi"
    assert tok.decode_incremental(a, ids[2:]) == "🌍"
    tok.flush_incremental(a)
    tok.flush_incremental(b)


def test_hold_keeps_a_stop_string_straddling_two_emits(tok):
    """Nothing already emitted may still grow into a stop string.

    That is the rule the scheduler needs, and ``hold = len(stop) - 1`` is the
    number that buys it: any suffix of the stream that is a proper prefix of the
    stop is at most that long, so it is still in the stream and can be revised
    by what comes next. Without it a client gets ``</tool_`` and then a
    correction, which is not a thing SSE can do.
    """
    stop = "</tool_call>"
    text = "before</tool_call>after"
    ids = ids_of(tok, text)
    seq = SequenceId(5)
    hold = len(stop) - 1

    emitted = ""
    saw_the_stop = False
    for i in range(0, len(ids), 4):
        emitted += tok.decode_incremental(seq, ids[i : i + 4], hold=hold)
        full = tok.decode(ids[: i + 4])
        assert full.startswith(emitted)
        assert len(full) - len(emitted) == min(hold, len(full))
        if stop in full:
            saw_the_stop = True
            continue
        for k in range(1, len(stop)):
            assert not emitted.endswith(stop[:k]), (emitted, stop[:k])
    emitted += tok.flush_incremental(seq)
    assert saw_the_stop
    assert emitted == text


def test_hold_never_emits_the_final_characters_early(tok):
    ids = ids_of(tok, "abcdefgh")
    seq = SequenceId(6)
    emitted = "".join(
        tok.decode_incremental(seq, [i], hold=3) for i in ids
    )
    assert emitted == "abcde"
    assert tok.flush_incremental(seq) == "fgh"


def test_hold_rejects_a_negative_value(tok):
    with pytest.raises(ValueError):
        tok.decode_incremental(SequenceId(7), [], hold=-1)


def test_concurrent_streams_do_not_share_state(tok):
    """One instance, many sequences, one detokeniser state each."""
    texts = [f"thread {i} 🌍 café" for i in range(16)]
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            seq = SequenceId(100 + index)
            ids = ids_of(tok, texts[index])
            got = "".join(
                tok.decode_incremental(seq, ids[i : i + 3])
                for i in range(0, len(ids), 3)
            )
            results[index] = got + tok.flush_incremental(seq)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == dict(enumerate(texts))
    assert tok.open_streams == 0


# ---------------------------------------------------------------------------
# the real checkpoint
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(), reason="checkpoint not present"
)


@pytest.fixture(scope="module")
def real_tok():
    if not (MODEL_DIR / "tokenizer.json").exists():
        pytest.skip("checkpoint not present")
    return load_tokenizer(MODEL_DIR)


@real
def test_real_eos_ids(real_tok):
    # config.json lists the chat terminator and the raw end-of-text
    assert real_tok.eos_token_ids == frozenset({248046, 248044})


@real
def test_real_vocab_size(real_tok):
    assert real_tok.vocab_size == 248077


@real
def test_real_byte_table_agrees_with_the_rust_decoder(real_tok):
    """Random ids from anywhere in the vocabulary, ours against the library's."""
    from tokenizers import Tokenizer

    rust = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rng = random.Random(20260912)
    for _ in range(200):
        ids = [rng.randrange(real_tok.vocab_size) for _ in range(24)]
        assert real_tok.decode(ids) == rust.decode(ids, skip_special_tokens=False)


@real
@pytest.mark.parametrize(
    "text",
    [
        "def triangular(n):\n    return n * (n + 1) // 2\n",
        "café 日本語 🌍 مرحبا Здравствуйте",
        "<|im_start|>user\nhi<|im_end|>\n",
        "<think>\nreasoning\n</think>\n\nanswer",
    ],
)
def test_real_round_trip(real_tok, text):
    assert real_tok.decode(real_tok.encode(text)) == text


@real
def test_real_incremental_matches_decode(real_tok):
    text = "The 🌍 is round; 日本語 works; tabs\tand\nnewlines too."
    ids = real_tok.encode(text)
    seq = SequenceId(900)
    got = "".join(
        real_tok.decode_incremental(seq, ids[i : i + 2]) for i in range(0, len(ids), 2)
    )
    assert got + real_tok.flush_incremental(seq) == text


def test_satisfies_the_port(tok):
    """``Tokenizer`` is not ``runtime_checkable``, so this checks it by hand."""
    from titan.core.ports import Tokenizer

    for name in (
        "encode",
        "decode",
        "decode_incremental",
        "flush_incremental",
        "eos_token_ids",
        "vocab_size",
    ):
        assert hasattr(Tokenizer, name) and hasattr(tok, name), name
