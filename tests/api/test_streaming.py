"""The SSE wire contract, asserted event by event.

These tests are deliberately literal. The streaming shape is what opencode and
the harness were built against, and "roughly the same events" is not a property
either of them respects, so the assertions are on exact payloads in exact order.
"""

from __future__ import annotations

import json

import pytest

from tests.api.conftest import (
    MODEL_NAME,
    FakeEngine,
    make_client,  # noqa: F401  (fixture)
    make_config,
    sse_events,
)
from titan.core.types import FinishReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_specification",
            "description": "Navigate the specification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "variables": {"type": "object"},
                },
            },
        },
    }
]

# One scripted turn with everything in it: a thinking block, text, and a call.
THINK_AND_CALL = [
    "\nI should",
    " look this up.\n",
    "</think>",
    "\n\n",
    "<tool_call>\n<function=query_specification>\n<parameter=query>\n",
    "{ systems { id } }",
    "\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>",
]


async def _collect(client, payload: dict) -> list:
    async with client:
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return sse_events(response.text)


async def test_streaming_event_sequence_with_tool_call(make_client):
    engine = FakeEngine(deltas=THINK_AND_CALL, completion_tokens=42)
    events = await _collect(
        make_client(engine),
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "continue"}],
            "tools": TOOLS,
            "stream": True,
        },
    )

    # 1. keepalive, 2. role, 3..n reasoning/content/tool_calls, then finish, DONE.
    assert events[0] == {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "keepalive",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    assert events[1] == {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
    }

    deltas = [c["choices"][0]["delta"] for c in events[2:-2]]
    assert deltas == [
        {"reasoning_content": "\nI should"},
        {"reasoning_content": " look this up.\n"},
        {"content": "\n\n"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-0",
                    "type": "function",
                    "function": {
                        "name": "query_specification",
                        "arguments": '{"query": "{ systems { id } }", "limit": 10}',
                    },
                }
            ]
        },
    ]

    assert events[-2] == {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    assert events[-1] == "[DONE]"


async def test_role_chunk_omits_finish_reason(make_client):
    """``finish_reason: null`` and an absent key are different to some clients."""
    engine = FakeEngine(deltas=["</think>", "hi"])
    async with make_client(engine) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    role_line = [
        line for line in response.text.split("\n") if '"role":"assistant"' in line
    ][1]
    assert "finish_reason" not in role_line


async def test_no_thinking_alias_routes_all_text_to_content(make_client):
    engine = FakeEngine(deltas=["Hello ", "there."])
    events = await _collect(
        make_client(engine),
        {
            "model": f"{MODEL_NAME}:no-think",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    deltas = [c["choices"][0]["delta"] for c in events[2:-2]]
    assert deltas == [{"content": "Hello "}, {"content": "there."}]
    assert events[-2]["choices"][0]["finish_reason"] == "stop"


async def test_finish_reason_length(make_client):
    engine = FakeEngine(deltas=["</think>", "truncated"], finish_reason=FinishReason.LENGTH)
    events = await _collect(
        make_client(engine),
        {"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert events[-2]["choices"][0]["finish_reason"] == "length"


async def test_tool_calls_beat_length_as_finish_reason(make_client):
    """A call plus a token ceiling must still tell the client to run the call."""
    engine = FakeEngine(
        deltas=[
            "</think>",
            "<tool_call>\n<function=query_specification>\n"
            "<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>",
        ],
        finish_reason=FinishReason.LENGTH,
    )
    events = await _collect(
        make_client(engine),
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "x"}],
            "tools": TOOLS,
            "stream": True,
        },
    )
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"


async def test_two_calls_get_distinct_indices_and_ids(make_client):
    envelope = (
        "<tool_call>\n<function=query_specification>\n"
        "<parameter=query>\n{}\n</parameter>\n</function>\n</tool_call>"
    )
    engine = FakeEngine(deltas=["</think>", envelope, "\n", envelope])
    events = await _collect(
        make_client(engine),
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "x"}],
            "tools": TOOLS,
            "stream": True,
        },
    )
    calls = [
        call
        for event in events[:-1]
        if event != "[DONE]"
        for choice in event["choices"]
        for call in choice["delta"].get("tool_calls", [])
    ]
    assert [c["index"] for c in calls] == [0, 1]
    assert [c["id"] for c in calls] == ["call-0", "call-1"]


async def test_include_usage_appends_a_usage_frame(make_client):
    engine = FakeEngine(
        deltas=["</think>", "ok"], prompt_tokens=1234, cached_tokens=1000, completion_tokens=9
    )
    events = await _collect(
        make_client(engine),
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    usage_event = events[-2]
    assert usage_event["choices"] == []
    assert usage_event["usage"] == {
        "prompt_tokens": 1234,
        "completion_tokens": 9,
        "total_tokens": 1243,
        "prompt_tokens_details": {"cached_tokens": 1000},
    }
    assert events[-1] == "[DONE]"


async def test_usage_frame_absent_by_default(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    events = await _collect(
        make_client(engine),
        {"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert all(e == "[DONE]" or "usage" not in e for e in events)


async def test_truncated_envelope_reaches_the_client_as_content(make_client):
    """The regression this module exists for: a cut-off call must not vanish."""
    engine = FakeEngine(
        deltas=["</think>", "<tool_call>\n<function=query_specification>\n<parameter=query>\npar"],
        finish_reason=FinishReason.LENGTH,
    )
    events = await _collect(
        make_client(engine),
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "x"}],
            "tools": TOOLS,
            "stream": True,
        },
    )
    content = "".join(
        choice["delta"].get("content", "")
        for event in events
        if event != "[DONE]"
        for choice in event["choices"]
    )
    assert content == (
        "<tool_call>\n<function=query_specification>\n<parameter=query>\npar"
    )
    assert events[-2]["choices"][0]["finish_reason"] == "length"


async def test_engine_error_is_reported_not_swallowed(make_client):
    engine = FakeEngine(
        deltas=["</think>", "partial"], finish_reason=FinishReason.ERROR, error="kv store failed"
    )
    events = await _collect(
        make_client(engine),
        {"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert any(
        isinstance(e, dict) and e.get("error", {}).get("message") == "kv store failed"
        for e in events
    )


async def test_keepalive_comment_mode(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    config = make_config(keepalive_mode="comment")
    async with make_client(engine, config) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
    assert response.text.startswith(": ping\n\n")


async def test_keepalive_off_mode(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    config = make_config(keepalive_mode="off")
    async with make_client(engine, config) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
    assert "keepalive" not in response.text
    assert response.text.startswith("data: ")


async def test_keepalive_fires_while_the_engine_is_silent(make_client):
    """A slow prefill must produce filler rather than a client-side timeout."""
    import asyncio

    from titan.core.types import StreamEnd, TokenEvent

    class SlowEngine(FakeEngine):
        def generate(self, request):
            async def _stream():
                await asyncio.sleep(0.2)  # keepalive interval in tests is 0.05s
                yield TokenEvent(request_id=request.request_id, token_ids=(1,), text="</think>hi")
                yield StreamEnd(
                    request_id=request.request_id,
                    finish_reason=FinishReason.STOP,
                    prompt_tokens=1,
                    cached_tokens=0,
                    completion_tokens=1,
                )

            return _stream()

    events = await _collect(
        make_client(SlowEngine()),
        {"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    keepalives = [e for e in events if e != "[DONE]" and e.get("model") == "keepalive"]
    assert len(keepalives) > 1
    assert all(k["id"] == "chatcmpl-0" for k in keepalives), "keepalive must share the stream id"
