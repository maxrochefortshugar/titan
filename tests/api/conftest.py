"""Fakes for the API layer: an engine, a tokenizer, and a config to hang them on.

Everything here is deterministic. The tests assert on exact SSE bytes, so a
random completion id, a random call id or a wall-clock timestamp would make them
useless; every source of variation is injected and pinned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

import pytest

from titan.adapters.chat.template import QwenChatTemplate
from titan.api.openai import ChatDeps, build_app
from titan.config.settings import (
    AliasConfig,
    LimitsConfig,
    ModelConfig,
    SamplingConfig,
    ServerConfig,
    TitanConfig,
)
from titan.core.types import (
    FinishReason,
    Request,
    RequestId,
    StreamEnd,
    TokenEvent,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RAW = FIXTURES / "m4-raw"

MODEL_NAME = "Qwen3.8-Flash-Next-oQ4e-mtp"

# A minimal stand-in for the model's chat template. The real one is exercised in
# tests/adapters/chat; here the prompt only has to be deterministic, because
# what is under test is the SSE contract rather than the prompt bytes.
FAKE_TEMPLATE = (
    "{%- for m in messages %}"
    "{{- '<|im_start|>' + m.role + '\\n' + (m.content or '') + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if tools %}{{- '<|tools|>' + (tools | tojson) + '\\n' }}{%- endif %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking %}{{- '<think>\\n' }}{%- else %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}"
)


class FakeTokenizer:
    """One token per four characters. Enough to make limit checks meaningful.

    Keeps every text it encoded, which is how a test gets at the rendered
    prompt: the API layer hands the prompt straight here and keeps only ids.
    """

    eos_token_ids = frozenset({151645})
    vocab_size = 151936

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        self.prompts.append(text)
        return [1] * max(1, (len(text) + 3) // 4)

    def decode(self, ids: Sequence[int]) -> str:
        return ""

    def decode_incremental(self, seq: Any, ids: Sequence[int]) -> str:
        return ""

    def flush_incremental(self, seq: Any) -> str:
        return ""


@dataclass
class FakeEngine:
    """Replays a scripted list of text deltas, then a scripted ``StreamEnd``.

    Records the requests it was handed so a test can assert on the prompt, the
    sampling parameters and the stop condition the API layer resolved.
    """

    deltas: Sequence[str] = ()
    finish_reason: FinishReason = FinishReason.STOP
    prompt_tokens: int = 11
    cached_tokens: int = 7
    completion_tokens: int = 5
    error: str | None = None
    seen: list[Request] = field(default_factory=list)

    def generate(self, request: Request) -> AsyncIterator[Any]:
        self.seen.append(request)

        async def _stream() -> AsyncIterator[Any]:
            for text in self.deltas:
                yield TokenEvent(
                    request_id=request.request_id, token_ids=(1,), text=text
                )
            yield StreamEnd(
                request_id=request.request_id,
                finish_reason=self.finish_reason,
                prompt_tokens=self.prompt_tokens,
                cached_tokens=self.cached_tokens,
                completion_tokens=self.completion_tokens,
                error=self.error,
            )

        return _stream()

    @property
    def last(self) -> Request:
        return self.seen[-1]


def make_config(
    *,
    api_key_file: Path | None = None,
    keepalive_mode: str = "chunk",
    max_tokens: int = 4096,
    max_context: int = 8192,
) -> TitanConfig:
    return TitanConfig(
        server=ServerConfig(
            port=8085,
            api_key_file=api_key_file,
            sse_keepalive_mode=keepalive_mode,
            sse_keepalive_seconds=0.05,
        ),
        model=ModelConfig(
            name=MODEL_NAME,
            path=Path("/nonexistent/model"),
            sampling=SamplingConfig(),
            aliases={
                f"{MODEL_NAME}:no-think": AliasConfig(
                    enable_thinking=False,
                    sampling=SamplingConfig(temperature=0.0, top_p=1.0, top_k=0),
                ),
                f"{MODEL_NAME}:xhigh": AliasConfig(reasoning_effort="xhigh"),
            },
        ),
        limits=LimitsConfig(max_tokens=max_tokens, max_context=max_context),
    )


def counter(prefix: str):
    """Deterministic id factory: ``prefix-0``, ``prefix-1``, ..."""
    state = {"n": -1}

    def next_id() -> str:
        state["n"] += 1
        return f"{prefix}-{state['n']}"

    return next_id


def make_deps(engine: FakeEngine, config: TitanConfig | None = None) -> ChatDeps:
    return ChatDeps(
        config=config or make_config(),
        engine=engine,
        renderer=QwenChatTemplate.from_source(FAKE_TEMPLATE),
        tokenizer=FakeTokenizer(),
        id_factory=counter("chatcmpl"),
        call_id_factory=counter("call"),
        created_factory=lambda: 1700000000,
        clock=lambda: 0.0,
    )


@pytest.fixture
def make_client():
    """Build an httpx client against an app wired to the given fake engine."""
    import httpx

    def _build(engine: FakeEngine, config: TitanConfig | None = None):
        deps = make_deps(engine, config)
        app = build_app(deps)
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://titan.test")
        client.deps = deps  # so a test can reach the tokenizer's recorded prompts
        return client

    return _build


def sse_events(body: str) -> list[Any]:
    """Parse an SSE body into ``[DONE]`` plus decoded JSON payloads."""
    out: list[Any] = []
    for line in body.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        out.append("[DONE]" if payload.strip() == "[DONE]" else json.loads(payload))
    return out


def raw_stream_text(path: Path) -> str:
    """Reconstruct the model's raw output text from a recorded oMLX SSE file.

    Runs the recorded deltas backwards through oMLX's own output mapping:
    ``reasoning_content`` was text before ``</think>``, ``content`` was text
    after it, and a ``tool_calls`` delta stood for an XML envelope that oMLX had
    already parsed away. Rebuilding the envelope from the parsed call is what
    makes the recorded streams usable as parser fixtures.
    """
    reasoning: list[str] = []
    content: list[str] = []
    for event in sse_events(path.read_text()):
        if event == "[DONE]" or event.get("model") == "keepalive":
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "reasoning_content" in delta:
                reasoning.append(delta["reasoning_content"])
            if "content" in delta:
                content.append(delta["content"])
            for call in delta.get("tool_calls", []) or []:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                parts = [f"<tool_call>\n<function={function['name']}>\n"]
                for key, value in arguments.items():
                    rendered = value if isinstance(value, str) else json.dumps(value)
                    parts.append(f"<parameter={key}>\n{rendered}\n</parameter>\n")
                parts.append("</function>\n</tool_call>")
                content.append("".join(parts))
    head = "".join(reasoning)
    return f"{head}</think>{''.join(content)}" if head else "".join(content)


def recorded_calls(path: Path) -> list[dict[str, Any]]:
    """The tool calls oMLX emitted for a recorded stream."""
    out: list[dict[str, Any]] = []
    for event in sse_events(path.read_text()):
        if event == "[DONE]" or event.get("model") == "keepalive":
            continue
        for choice in event.get("choices", []):
            for call in choice.get("delta", {}).get("tool_calls", []) or []:
                out.append(
                    {
                        "index": call["index"],
                        "name": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"]),
                    }
                )
    return out


def recorded_tools(path: Path) -> list[dict[str, Any]]:
    """The tool schemas from the request that produced a recorded stream."""
    request = json.loads(path.with_suffix(".req.json").read_text())
    return request.get("tools") or []


def raw_paths() -> list[Path]:
    return sorted(RAW.glob("*.sse"))
