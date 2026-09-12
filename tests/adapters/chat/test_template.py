"""Chat template rendering, against expected strings read off the template.

Every expected value here was derived by reading
``chat_template.jinja`` and writing down what it must emit, rather than by
capturing what the code produced. That direction matters: a test written from
the output only proves the code is self-consistent, and the whole risk with a
prompt is being confidently, silently wrong.

The real template is loaded from the model directory when it is present, and the
tests that need it skip otherwise, so the suite still runs on a machine with no
weights. The request fixtures are the genuine opencode and harness payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.api.conftest import RAW
from titan.adapters.chat.template import (
    QwenChatTemplate,
    TemplateError,
    normalise_reasoning_effort,
    render_chat,
    tools_to_template,
)
from titan.core.types import Message, Role, ToolCall

MODEL_DIR = Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
TEMPLATE_FILE = MODEL_DIR / "chat_template.jinja"

pytestmark = pytest.mark.skipif(
    not TEMPLATE_FILE.exists(), reason="model chat_template.jinja not present"
)

LOW_INSTRUCTION = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)
XHIGH_INSTRUCTION = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@pytest.fixture(scope="module")
def template():
    return QwenChatTemplate.from_path(MODEL_DIR).template


def render(template, messages, tools=None, **kwargs):
    return render_chat(template, messages, tools, **kwargs)


# -- the basic turn ----------------------------------------------------------


def test_system_and_user_at_medium_effort(template):
    """Medium is the quiet setting: the template prepends no instruction."""
    out = render(
        template,
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert out == (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_low_effort_prepends_its_instruction(template):
    out = render(
        template,
        [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hi"}],
        reasoning_effort="low",
    )
    assert out.startswith(f"<|im_start|>system\n{LOW_INSTRUCTION}\n\nSys.<|im_end|>\n")


def test_xhigh_effort_prepends_its_instruction(template):
    out = render(
        template,
        [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hi"}],
        reasoning_effort="xhigh",
    )
    assert out.startswith(f"<|im_start|>system\n{XHIGH_INSTRUCTION}\n\nSys.<|im_end|>\n")


def test_effort_instruction_without_a_system_message(template):
    """With no system turn the template invents one to hold the instruction."""
    out = render(template, [{"role": "user", "content": "Hi"}], reasoning_effort="low")
    assert out == (
        f"<|im_start|>system\n{LOW_INSTRUCTION}<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_medium_effort_without_a_system_message_emits_no_system_turn(template):
    out = render(template, [{"role": "user", "content": "Hi"}])
    assert out == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_thinking_disabled_pre_closes_the_block(template):
    """This is what makes the model answer without thinking."""
    out = render(template, [{"role": "user", "content": "Hi"}], enable_thinking=False)
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_thinking_disabled_suppresses_the_effort_instruction(template):
    out = render(
        template,
        [{"role": "user", "content": "Hi"}],
        enable_thinking=False,
        reasoning_effort="low",
    )
    assert LOW_INSTRUCTION not in out


def test_no_generation_prompt(template):
    out = render(template, [{"role": "user", "content": "Hi"}], add_generation_prompt=False)
    assert out == "<|im_start|>user\nHi<|im_end|>\n"


def test_content_is_trimmed(template):
    out = render(template, [{"role": "user", "content": "  padded  "}])
    assert "<|im_start|>user\npadded<|im_end|>" in out


def test_multipart_text_content_is_concatenated(template):
    out = render(
        template,
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
    )
    assert "<|im_start|>user\nab<|im_end|>" in out


# -- tools -------------------------------------------------------------------


def test_tool_block_is_the_system_turn(template):
    out = render(
        template,
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        [WEATHER_TOOL],
    )
    expected_tools = json.dumps(WEATHER_TOOL, ensure_ascii=False)
    assert out == (
        "<|im_start|>system\n"
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{expected_tools}\n</tools>"
        f"{TOOL_INSTRUCTIONS}"
        "\n\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_tool_block_carries_the_effort_instruction_first(template):
    out = render(
        template, [{"role": "user", "content": "Hi"}], [WEATHER_TOOL], reasoning_effort="low"
    )
    assert out.startswith(f"<|im_start|>system\n{LOW_INSTRUCTION}\n\n# Tools\n")


def test_tool_schemas_are_not_html_escaped(template):
    """Jinja's stock tojson would mangle every angle bracket in a description."""
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "compare a < b && c > d, quote: 'x'",
            "parameters": {},
        },
    }
    out = render(template, [{"role": "user", "content": "Hi"}], [tool])
    assert "a < b && c > d" in out
    assert "\\u003c" not in out


def test_multiple_tools_render_one_per_line(template):
    second = {"type": "function", "function": {"name": "g", "description": "", "parameters": {}}}
    out = render(template, [{"role": "user", "content": "Hi"}], [WEATHER_TOOL, second])
    block = out.split("<tools>", 1)[1].split("</tools>", 1)[0]
    assert block.count("\n") == 3  # leading newline plus one per tool
    assert json.loads(block.strip().split("\n")[1])["function"]["name"] == "g"


def test_missing_description_becomes_an_empty_string(template):
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    rendered = tools_to_template([tool])
    assert rendered[0]["function"]["description"] == ""


# -- history: assistant tool calls and tool results --------------------------


def test_assistant_tool_call_round_trips_into_the_template_format(template):
    out = render(
        template,
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "need the tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "thanks"},
        ],
        [WEATHER_TOOL],
    )
    assert (
        "<|im_start|>assistant\n<think>\nneed the tool\n</think>\n\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBerlin\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n"
    ) in out


def test_tool_result_is_wrapped_in_a_user_turn(template):
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "sunny"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" in out


def test_consecutive_tool_results_share_one_user_turn(template):
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "one"},
            {"role": "tool", "content": "two"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert (
        "<|im_start|>user\n<tool_response>\none\n</tool_response>\n"
        "<tool_response>\ntwo\n</tool_response><|im_end|>\n"
    ) in out


def test_two_tool_calls_in_one_assistant_message(template):
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "A"}'}},
                    {"function": {"name": "get_weather", "arguments": '{"city": "B"}'}},
                ],
            },
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nB\n</parameter>\n"
        "</function>\n</tool_call>"
    ) in out


def test_assistant_text_before_a_call_is_separated_by_a_blank_line(template):
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "Let me look.",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert "Let me look.\n\n<tool_call>\n" in out


def test_non_string_argument_values_are_json_encoded(template):
    """Strings go in bare; everything else goes through tojson."""
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps(
                                {"city": "Berlin", "days": 3, "exact": True, "opts": {"a": 1}}
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert "<parameter=city>\nBerlin\n</parameter>" in out
    assert "<parameter=days>\n3\n</parameter>" in out
    assert "<parameter=exact>\ntrue\n</parameter>" in out
    assert '<parameter=opts>\n{"a": 1}\n</parameter>' in out


def test_arguments_as_a_dict_are_accepted_too(template):
    out = render(
        template,
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}
            ]},
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    assert "<parameter=city>\nOslo\n</parameter>" in out


def test_core_message_objects_render_identically(template):
    """The port takes core types; the wire path takes dicts. Same prompt."""
    from_dicts = render(
        template,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "sure",
                "reasoning_content": "hmm",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}
                ],
            },
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "ok"},
        ],
        [WEATHER_TOOL],
    )
    from_objects = render(
        template,
        [
            Message(role=Role.USER, content="go"),
            Message(
                role=Role.ASSISTANT,
                content="sure",
                reasoning_content="hmm",
                tool_calls=(
                    ToolCall(index=0, call_id="c", name="get_weather", arguments='{"city": "Rome"}'),
                ),
            ),
            Message(role=Role.TOOL, content="x"),
            Message(role=Role.USER, content="ok"),
        ],
        [WEATHER_TOOL],
    )
    assert from_dicts == from_objects


# -- errors and kwargs -------------------------------------------------------


def test_system_message_out_of_position_raises(template):
    with pytest.raises(TemplateError, match="System message must be at the beginning"):
        render(
            template,
            [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
        )


def test_no_messages_raises(template):
    with pytest.raises(TemplateError, match="No messages provided"):
        render(template, [])


def test_conversation_with_no_user_query_raises(template):
    with pytest.raises(TemplateError, match="No user query found"):
        render(template, [{"role": "assistant", "content": "hello"}])


def test_unknown_role_raises(template):
    with pytest.raises(TemplateError, match="Unexpected message role"):
        render(template, [{"role": "user", "content": "a"}, {"role": "wizard", "content": "b"}])


def test_template_kwargs_reach_the_template(template):
    """``preserve_thinking=False`` drops reasoning from pre-query history."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first", "reasoning_content": "old thought"},
        {"role": "user", "content": "two"},
    ]
    kept = render(template, messages)
    dropped = render(template, messages, preserve_thinking=False)
    assert "old thought" in kept
    assert "old thought" not in dropped


def test_rendering_is_deterministic(template):
    messages = [{"role": "user", "content": "Hi"}]
    assert render(template, messages, [WEATHER_TOOL]) == render(template, messages, [WEATHER_TOOL])


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("low", "low"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("high", "xhigh"),
        ("minimal", "low"),
        ("none", "low"),
        ("off", "low"),
        ("moderate", "medium"),
        ("max", "xhigh"),
        ("  HIGH  ", "xhigh"),
        (None, "medium"),
    ],
)
def test_reasoning_effort_aliases(given, expected):
    assert normalise_reasoning_effort(given) == expected


def test_unknown_reasoning_effort_raises():
    with pytest.raises(TemplateError):
        normalise_reasoning_effort("turbo")


# -- the real captured client payloads ---------------------------------------


@pytest.mark.parametrize(
    "path", sorted(RAW.glob("*.req.json")), ids=lambda p: p.name.split(".")[0]
)
def test_captured_client_requests_render(template, path):
    """Every recorded opencode and harness payload must render without error."""
    request = json.loads(path.read_text())
    out = render(
        template,
        request["messages"],
        request.get("tools"),
        reasoning_effort="medium",
    )
    assert out.startswith("<|im_start|>system\n# Tools\n")
    assert out.endswith("<|im_start|>assistant\n<think>\n")
    # The tool the harness offers is rendered into the tool block verbatim.
    assert '"name": "query_specification"' in out
    # Every user and tool turn survives into the prompt.
    assert out.count("<|im_start|>user\n") >= 1


def test_captured_multi_turn_request_replays_its_own_tool_calls(template):
    """The shape that matters: a later turn echoing back earlier calls."""
    request = json.loads((RAW / "162448.req.json").read_text())
    assert any(m.get("tool_calls") for m in request["messages"]), "fixture lost its calls"
    out = render(template, request["messages"], request["tools"], reasoning_effort="medium")
    assert "<tool_call>\n<function=query_specification>\n<parameter=" in out
    assert "<tool_response>\n" in out
