"""The non-streaming body, the auxiliary endpoints, auth, and the limit checks."""

from __future__ import annotations

import json

import pytest

from tests.api.conftest import (
    MODEL_NAME,
    FakeEngine,
    make_client,  # noqa: F401  (fixture)
    make_config,
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
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "raw": {"type": "boolean"},
                },
            },
        },
    }
]


async def test_non_streaming_body(make_client):
    engine = FakeEngine(
        deltas=["\nthinking", "</think>", "Here you go."],
        prompt_tokens=100,
        cached_tokens=64,
        completion_tokens=12,
    )
    async with make_client(engine) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json() == {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here you go.",
                    "reasoning_content": "\nthinking",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "total_tokens": 112,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
    }


async def test_non_streaming_tool_call(make_client):
    engine = FakeEngine(
        deltas=[
            "</think>",
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>\nsrc/main.py\n</parameter>\n"
            "<parameter=start>\n40\n</parameter>\n"
            "<parameter=raw>\nfalse\n</parameter>\n"
            "</function>\n</tool_call>",
        ]
    )
    async with make_client(engine) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": TOOLS,
            },
        )
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"] == [
        {
            "index": 0,
            "id": "call-0",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "src/main.py", "start": 40, "raw": false}',
            },
        }
    ]


async def test_non_streaming_engine_error_is_a_500(make_client):
    engine = FakeEngine(deltas=[], finish_reason=FinishReason.ERROR, error="backend died")
    async with make_client(engine) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "backend died"


async def test_unknown_model_is_404(make_client):
    async with make_client(FakeEngine()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_max_tokens_over_the_limit_is_400(make_client):
    async with make_client(FakeEngine(), make_config(max_tokens=64)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 65,
            },
        )
    assert response.status_code == 400
    assert "exceeds the server limit" in response.json()["error"]["message"]


async def test_context_overflow_is_400(make_client):
    config = make_config(max_tokens=64, max_context=128)
    async with make_client(FakeEngine(), config) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "x" * 4000}],
                "max_tokens": 16,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


async def test_empty_messages_is_400(make_client):
    async with make_client(FakeEngine()) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": MODEL_NAME, "messages": []}
        )
    assert response.status_code == 400


async def test_bad_reasoning_effort_is_400(make_client):
    async with make_client(FakeEngine()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "turbo",
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "reasoning_effort"


async def test_sampling_defaults_come_from_config(make_client):
    """An omitted temperature must inherit 0.7, not OpenAI's 1.0."""
    engine = FakeEngine(deltas=["</think>", "ok"])
    async with make_client(engine) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
        )
    sampling = engine.last.sampling
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (0.7, 0.8, 20)


async def test_request_overrides_beat_config_defaults(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    async with make_client(engine) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 5,
                "seed": 11,
                "stop": ["<|im_end|>"],
            },
        )
    sampling = engine.last.sampling
    assert (sampling.temperature, sampling.top_p, sampling.top_k, sampling.seed) == (
        0.0,
        0.95,
        5,
        11,
    )
    assert engine.last.stop.stop_strings == ("<|im_end|>",)


async def test_alias_sampling_profile_applies(make_client):
    engine = FakeEngine(deltas=["ok"])
    async with make_client(engine) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": f"{MODEL_NAME}:no-think",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        prompt = client.deps.tokenizer.prompts[-1]
    assert engine.last.sampling.temperature == 0.0
    assert engine.last.sampling.is_greedy
    # The alias also switched the template off thinking mode.
    assert "<think>\n\n</think>" in prompt


async def test_tool_choice_none_drops_the_tool_block(make_client):
    """The template cannot say "tools exist, do not call one"; honour the intent."""
    engine = FakeEngine(deltas=["</think>", "ok"])
    async with make_client(engine) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": TOOLS,
                "tool_choice": "none",
            },
        )
        prompt = client.deps.tokenizer.prompts[-1]
    assert "<|tools|>" not in prompt


async def test_max_completion_tokens_is_accepted(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    async with make_client(engine) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 77,
            },
        )
    assert engine.last.stop.max_tokens == 77


async def test_unknown_request_fields_are_ignored(make_client):
    engine = FakeEngine(deltas=["</think>", "ok"])
    async with make_client(engine) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": 0.2,
                "user": "someone",
                "n": 1,
            },
        )
    assert response.status_code == 200


async def test_models_endpoint(make_client):
    async with make_client(FakeEngine()) as client:
        response = await client.get("/v1/models")
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [
        MODEL_NAME,
        f"{MODEL_NAME}:no-think",
        f"{MODEL_NAME}:xhigh",
    ]
    assert all(m["object"] == "model" and m["owned_by"] == "titan" for m in body["data"])
    assert body["data"][0]["max_model_len"] == 8192


async def test_health_endpoint(make_client):
    async with make_client(FakeEngine()) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_auth_required_when_a_key_file_is_configured(make_client, tmp_path):
    key_file = tmp_path / "titan.key"
    key_file.write_text("s3cret\n")
    config = make_config(api_key_file=key_file)

    async with make_client(FakeEngine(deltas=["</think>", "ok"]), config) as client:
        assert (await client.get("/v1/models")).status_code == 401
        assert (
            await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        good = await client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
        assert good.status_code == 200
        # Liveness stays open: a watchdog does not carry a key.
        assert (await client.get("/health")).status_code == 200
