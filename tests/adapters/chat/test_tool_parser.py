"""Parser tests: recorded real streams, plus the cases that broke production.

The recorded half works by reconstruction. Each ``*.sse`` in ``tests/fixtures``
is a real oMLX stream for a real opencode or harness request, and the model text
behind it can be rebuilt from the deltas: reasoning before ``</think>``, content
after, and each parsed ``tool_calls`` delta re-rendered as the XML envelope that
produced it. Feeding that text back in at random split points and demanding the
same calls come out is a strong end-to-end check on both the boundary logic and
the schema coercion, against traffic nobody wrote for a test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tests.api.conftest import RAW, raw_paths, raw_stream_text, recorded_calls, recorded_tools
from titan.adapters.chat.tool_parser import Qwen3CoderToolParser, coerce_parameter

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "o": {"type": "object"},
                    "a": {"type": "array"},
                },
            },
        },
    }
]


def run(text: str, chunks: list[str] | None = None, **kwargs):
    """Feed ``text`` (or a pre-split ``chunks``) through a fresh parser."""
    ids = iter(f"call_{n}" for n in range(1000))
    parser = Qwen3CoderToolParser(id_factory=lambda: next(ids), **kwargs)
    content: list[str] = []
    reasoning: list[str] = []
    calls: list = []
    for chunk in chunks if chunks is not None else [text]:
        c, r, k = parser.feed(chunk)
        content.append(c)
        reasoning.append(r)
        calls.extend(k)
    c, r, k, error = parser.finish()
    content.append(c)
    reasoning.append(r)
    calls.extend(k)
    return "".join(content), "".join(reasoning), calls, error


def split_randomly(text: str, seed: int, pieces: int = 40) -> list[str]:
    rng = random.Random(seed)
    if len(text) <= 1:
        return [text]
    cuts = sorted(rng.sample(range(1, len(text)), min(pieces, len(text) - 1)))
    return [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]


# -- recorded streams --------------------------------------------------------


@pytest.mark.parametrize("path", raw_paths(), ids=lambda p: p.stem)
def test_recorded_stream_reproduces_omlx_tool_calls(path: Path):
    """Titan must parse the real streams into the calls oMLX parsed."""
    text = raw_stream_text(path)
    expected = recorded_calls(path)
    tools = recorded_tools(path)

    content, reasoning, calls, error = run(text, tools=tools)

    assert error is None, f"{path.name}: {error}"
    assert [c.index for c in calls] == [e["index"] for e in expected]
    assert [c.name for c in calls] == [e["name"] for e in expected]
    assert [json.loads(c.arguments) for c in calls] == [e["arguments"] for e in expected]
    assert "<tool_call>" not in content, "markup must never reach the content channel"


@pytest.mark.parametrize("path", raw_paths(), ids=lambda p: p.stem)
def test_recorded_stream_is_split_invariant(path: Path):
    """Token boundaries must not change the output. This is the whole game."""
    text = raw_stream_text(path)
    tools = recorded_tools(path)
    baseline = run(text, tools=tools)

    for seed in range(8):
        assert run(text, split_randomly(text, seed), tools=tools) == baseline, (
            f"{path.name}: split at seed {seed} changed the result"
        )
    assert run(text, list(text), tools=tools) == baseline, "character-wise feed differs"


@pytest.mark.parametrize("path", raw_paths(), ids=lambda p: p.stem)
def test_recorded_stream_reasoning_and_content_split(path: Path):
    """Everything before ``</think>`` is reasoning; nothing after it is."""
    text = raw_stream_text(path)
    content, reasoning, _, _ = run(text, tools=recorded_tools(path))
    if "</think>" in text:
        assert reasoning == text.split("</think>", 1)[0]
    assert "</think>" not in reasoning
    assert "</think>" not in content


# -- structure ---------------------------------------------------------------


def test_text_outside_envelopes_is_content():
    content, reasoning, calls, error = run(
        "</think>before <tool_call>\n<function=f>\n</function>\n</tool_call> after",
        tools=WEATHER_TOOLS,
    )
    assert content == "before  after"
    assert reasoning == ""
    assert [c.name for c in calls] == ["f"]
    assert calls[0].arguments == "{}"
    assert error is None


def test_multiple_calls_in_one_message():
    envelope = "<tool_call>\n<function=f>\n<parameter=s>\nv\n</parameter>\n</function>\n</tool_call>"
    _, _, calls, _ = run(f"</think>{envelope}\n{envelope}\n{envelope}", tools=WEATHER_TOOLS)
    assert [c.index for c in calls] == [0, 1, 2]
    assert [c.call_id for c in calls] == ["call_0", "call_1", "call_2"]


def test_value_containing_literal_close_tags_survives():
    """File-editing tools carry source that quotes the tool-call format."""
    value = "text with </parameter> and </function></tool_call> inside"
    _, _, calls, error = run(
        f"</think><tool_call>\n<function=f>\n<parameter=s>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>",
        tools=WEATHER_TOOLS,
    )
    assert error is None
    assert json.loads(calls[0].arguments) == {"s": value}


def test_value_containing_close_tags_is_split_invariant():
    value = "a </parameter> b </function></tool_call> c"
    text = (
        f"</think><tool_call>\n<function=f>\n<parameter=s>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>tail"
    )
    baseline = run(text, tools=WEATHER_TOOLS)
    for seed in range(20):
        assert run(text, split_randomly(text, seed), tools=WEATHER_TOOLS) == baseline
    assert run(text, list(text), tools=WEATHER_TOOLS) == baseline


def test_multiline_value_keeps_its_newlines():
    value = "line one\nline two\n\nline four"
    _, _, calls, _ = run(
        f"</think><tool_call>\n<function=f>\n<parameter=s>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>",
        tools=WEATHER_TOOLS,
    )
    assert json.loads(calls[0].arguments)["s"] == value


def test_json_dialect_envelope_is_accepted():
    _, _, calls, error = run(
        '</think><tool_call>{"name": "f", "arguments": {"s": "x"}}</tool_call>',
        tools=WEATHER_TOOLS,
    )
    assert error is None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"s": "x"}


# -- reasoning ---------------------------------------------------------------


def test_reasoning_splits_at_the_close_marker():
    content, reasoning, _, _ = run("\nthinking here</think>\n\nanswer")
    assert reasoning == "\nthinking here"
    assert content == "\n\nanswer"


def test_close_marker_split_across_deltas():
    content, reasoning, _, _ = run("", ["think", "</", "thi", "nk>", "out"])
    assert reasoning == "think"
    assert content == "out"


def test_no_thinking_mode_starts_in_content():
    content, reasoning, _, _ = run("straight to it", in_reasoning=False)
    assert content == "straight to it"
    assert reasoning == ""


def test_reopened_think_block_returns_to_reasoning():
    content, reasoning, _, _ = run(
        "</think>answer<think>more thought</think>rest"
    )
    assert content == "answerrest"
    assert reasoning == "more thought"


def test_tool_markup_inside_thinking_is_not_a_call():
    """The model quotes the call format while deciding. Do not fire on it."""
    text = (
        "I could call <tool_call>\n<function=f>\n<parameter=s>\nx\n</parameter>\n"
        "</function>\n</tool_call> but I will not.</think>plain answer"
    )
    content, reasoning, calls, error = run(text, tools=WEATHER_TOOLS)
    assert calls == []
    assert error is None
    assert content == "plain answer"
    assert "<tool_call>" in reasoning


# -- malformed input ---------------------------------------------------------


def test_truncated_envelope_becomes_content_and_an_error():
    """The regression: a cut-off call used to vanish, leaving a blank turn."""
    tail = "<tool_call>\n<function=f>\n<parameter=s>\nhalf a val"
    content, _, calls, error = run(f"</think>text {tail}", tools=WEATHER_TOOLS)
    assert calls == []
    assert content == f"text {tail}", "the truncated envelope must be visible"
    assert error == "unterminated tool-call envelope at end of stream"


def test_truncated_envelope_is_logged(caplog):
    with caplog.at_level("WARNING", logger="titan.adapters.chat.tool_parser"):
        run("</think><tool_call>\n<function=f>\n<parameter=s>\nx", tools=WEATHER_TOOLS)
    assert any("unterminated tool-call envelope" in r.getMessage() for r in caplog.records)


def test_envelope_with_no_function_element_becomes_content():
    text = "</think><tool_call>\nnot xml and not json\n</tool_call>"
    content, _, calls, error = run(text, tools=WEATHER_TOOLS)
    assert calls == []
    assert content == "<tool_call>\nnot xml and not json\n</tool_call>"
    assert error == "unparseable tool-call envelope emitted as content"


def test_a_bad_envelope_does_not_kill_the_good_one_after_it():
    good = "<tool_call>\n<function=f>\n<parameter=s>\nok\n</parameter>\n</function>\n</tool_call>"
    bad = "<tool_call>\ngarbage\n</tool_call>"
    content, _, calls, error = run(f"</think>{bad}{good}", tools=WEATHER_TOOLS)
    assert [c.name for c in calls] == ["f"]
    assert bad in content
    assert error is not None


def test_missing_tool_call_close_recovers_at_end_of_stream():
    text = "</think><tool_call>\n<function=f>\n<parameter=s>\nv\n</parameter>\n</function>\n"
    _, _, calls, error = run(text, tools=WEATHER_TOOLS)
    # No </tool_call> at all: unterminated, so it surfaces rather than guessing.
    assert calls == []
    assert error == "unterminated tool-call envelope at end of stream"


def test_stray_close_marker_is_content():
    content, _, calls, error = run("</think>done</tool_call>", tools=WEATHER_TOOLS)
    assert calls == []
    assert content == "done</tool_call>"


def test_partial_open_marker_at_end_of_stream_is_content():
    content, _, calls, _ = run("</think>text <tool_c", tools=WEATHER_TOOLS)
    assert calls == []
    assert content == "text <tool_c"


def test_empty_stream():
    content, reasoning, calls, error = run("")
    assert (content, reasoning, calls, error) == ("", "", [], None)


def test_feed_after_finish_raises():
    parser = Qwen3CoderToolParser()
    parser.finish()
    with pytest.raises(RuntimeError):
        parser.feed("x")


# -- schema coercion ---------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "raw", "expected"),
    [
        ("string", "42", "42"),
        ("string", "true", "true"),
        ("string", '{"a": 1}', '{"a": 1}'),
        ("string", '"quoted"', "quoted"),
        ("integer", "42", 42),
        ("integer", "  7 ", 7),
        ("integer", "notanint", "notanint"),
        ("number", "2.5", 2.5),
        ("number", "3.0", 3),
        ("boolean", "true", True),
        ("boolean", "false", False),
        ("boolean", "TRUE", True),
        ("boolean", "yes", "yes"),
        ("object", '{"a": 1}', {"a": 1}),
        ("array", "[1, 2]", [1, 2]),
        ("object", "{'a': 1}", {"a": 1}),
        ("string", "null", None),
        ("integer", "null", None),
    ],
)
def test_coercion_by_declared_type(declared, raw, expected):
    props = {"p": {"type": declared}}
    assert coerce_parameter(raw, "p", props, "f") == expected


def test_undeclared_parameter_falls_back_to_json_then_string():
    assert coerce_parameter("42", "p", {}, "f") == 42
    assert coerce_parameter("hello", "p", {}, "f") == "hello"


def test_container_with_broken_json_is_repaired():
    assert coerce_parameter('{"a": [1, 2}', "p", {"p": {"type": "object"}}, "f") == {"a": [1, 2]}


def test_unrepairable_container_keeps_its_raw_string():
    assert coerce_parameter("<<<", "p", {"p": {"type": "object"}}, "f") == "<<<"


def test_coercion_uses_the_right_tool_when_names_collide():
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": {"properties": {"p": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "b", "parameters": {"properties": {"p": {"type": "string"}}}}},
    ]
    _, _, calls, _ = run(
        "</think>"
        "<tool_call>\n<function=a>\n<parameter=p>\n5\n</parameter>\n</function>\n</tool_call>"
        "<tool_call>\n<function=b>\n<parameter=p>\n5\n</parameter>\n</function>\n</tool_call>",
        tools=tools,
    )
    assert json.loads(calls[0].arguments) == {"p": 5}
    assert json.loads(calls[1].arguments) == {"p": "5"}


def test_unicode_arguments_are_not_escaped():
    _, _, calls, _ = run(
        "</think><tool_call>\n<function=f>\n<parameter=s>\nnaïve café\n</parameter>\n"
        "</function>\n</tool_call>",
        tools=WEATHER_TOOLS,
    )
    assert calls[0].arguments == '{"s": "naïve café"}'
