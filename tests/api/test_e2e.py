"""End to end over a real socket: uvicorn on an ephemeral port, httpx client.

The ASGI-transport tests cover the wire shapes; this one covers what only a real
server can break. Chunked transfer encoding, the streaming response actually
flushing per event rather than buffering to the end, and bearer auth surviving a
genuine HTTP round trip have all been wrong in ways an in-process transport
cannot see.

Binds 127.0.0.1 on port 0 so the OS picks a free port: nothing here goes near
8083 or 8084.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import httpx
import pytest
import uvicorn

from tests.api.conftest import MODEL_NAME, FakeEngine, make_config, make_deps
from titan.api.openai import build_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.asynccontextmanager
async def running(app, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.02)
        else:
            raise RuntimeError("server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_specification",
            "description": "Navigate the specification.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    }
]

SCRIPT = [
    "\nI will look",
    " it up.\n",
    "</think>",
    "\n\n",
    "<tool_call>\n<function=query_specification>\n<parameter=query>\n",
    "{ systems { id } }\n</parameter>\n<parameter=limit>\n5\n</parameter>\n",
    "</function>\n</tool_call>",
]


async def test_end_to_end_streaming_over_a_real_socket(tmp_path):
    key_file = tmp_path / "titan.key"
    key_file.write_text("e2e-key\n")
    engine = FakeEngine(deltas=SCRIPT, prompt_tokens=321, cached_tokens=300, completion_tokens=44)
    app = build_app(make_deps(engine, make_config(api_key_file=key_file)))

    async with running(app, free_port()) as base_url:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            unauthorised = await client.get("/v1/models")
            assert unauthorised.status_code == 401

            headers = {"Authorization": "Bearer e2e-key"}
            assert (await client.get("/health")).json()["status"] == "ok"
            assert (await client.get("/v1/models", headers=headers)).status_code == 200

            events: list[str] = []
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "continue"}],
                    "tools": TOOLS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line[6:])

    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]

    reasoning = "".join(
        c["delta"]["reasoning_content"]
        for p in payloads
        for c in p["choices"]
        if "reasoning_content" in c["delta"]
    )
    assert reasoning == "\nI will look it up.\n"

    calls = [
        call
        for p in payloads
        for c in p["choices"]
        for call in c["delta"].get("tool_calls", [])
    ]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "query_specification"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "{ systems { id } }",
        "limit": 5,
    }

    finishes = [c["finish_reason"] for p in payloads for c in p["choices"] if c.get("finish_reason")]
    assert finishes == ["tool_calls"]

    usage = [p["usage"] for p in payloads if p.get("usage")]
    assert usage == [
        {
            "prompt_tokens": 321,
            "completion_tokens": 44,
            "total_tokens": 365,
            "prompt_tokens_details": {"cached_tokens": 300},
        }
    ]


async def test_end_to_end_non_streaming_over_a_real_socket():
    engine = FakeEngine(deltas=["\nthink", "</think>", "The answer."])
    app = build_app(make_deps(engine))

    async with running(app, free_port()) as base_url:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            )
    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "The answer."
    assert body["choices"][0]["message"]["reasoning_content"] == "\nthink"
    assert body["choices"][0]["finish_reason"] == "stop"


async def test_streaming_response_is_not_buffered_to_the_end():
    """A keepalive is worthless if the whole body lands at once."""
    from titan.core.types import FinishReason, StreamEnd, TokenEvent

    released = asyncio.Event()

    class SlowEngine(FakeEngine):
        def generate(self, request):
            async def _stream():
                yield TokenEvent(request_id=request.request_id, token_ids=(1,), text="</think>first")
                await released.wait()
                yield TokenEvent(request_id=request.request_id, token_ids=(1,), text="second")
                yield StreamEnd(
                    request_id=request.request_id,
                    finish_reason=FinishReason.STOP,
                    prompt_tokens=1,
                    cached_tokens=0,
                    completion_tokens=2,
                )

            return _stream()

    app = build_app(make_deps(SlowEngine()))
    async with running(app, free_port()) as base_url:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
            ) as response:
                seen_first = False
                async for line in response.aiter_lines():
                    if '"first"' in line:
                        seen_first = True
                        # Arrived before the engine produced anything further,
                        # so the body is genuinely streaming.
                        released.set()
                    if line.strip() == "data: [DONE]":
                        break
    assert seen_first


@pytest.mark.anyio
async def test_metrics_reports_the_profile_and_the_loop(make_client):
    """A benchmark number that cannot name its configuration is not evidence,
    so the counters, the decode summary and the loop's own stats come from one
    endpoint."""
    client = make_client(FakeEngine(deltas=("hi",)))
    async with client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_NAME
    # The fake engine has no loop and the deps carry no profiler, so what comes
    # back is the identity of the instance and nothing invented on top of it.
    assert "loop" not in body
