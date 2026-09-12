"""Wire models for the OpenAI-compatible surface.

These exist only at the boundary: nothing below ``titan.api`` sees them, and
``titan.api.openai`` translates them into core types on the way in and back out.

Compatibility is a hard requirement, not a nice-to-have. opencode and the
owner's harness must work unchanged, which pins three details: streaming deltas
carry ``reasoning_content`` alongside ``content``; tool calls stream as indexed
deltas carrying the id, the name and the arguments; and ``usage`` reports
``prompt_tokens_details.cached_tokens`` so a prefix miss is visible from the
client side.

Every response model is serialised with ``exclude_none=True``. That is not a
cosmetic choice. A delta that carries ``"content": null`` alongside a
``tool_calls`` array makes some strict accumulators treat the chunk as a text
chunk and discard the calls, so an absent field and a null field are genuinely
different on this wire.

Unknown request fields are ignored rather than rejected: clients send
``frequency_penalty``, ``user``, ``n`` and a dozen other keys Titan has no
opinion about, and 400-ing on them would break every one of them for nothing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "new_completion_id",
    "new_call_id",
    "unix_now",
    "ToolFunction",
    "ToolDefinition",
    "StreamOptions",
    "ChatCompletionRequest",
    "FunctionCallPayload",
    "ToolCallPayload",
    "Delta",
    "StreamChoice",
    "PromptTokensDetails",
    "Usage",
    "ChatCompletionChunk",
    "AssistantMessage",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "ModelInfo",
    "ModelsResponse",
    "ErrorDetail",
    "ErrorResponse",
]


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def unix_now() -> int:
    return int(time.time())


class _Wire(BaseModel):
    """Base for outbound models. Frozen, and no surprise coercion."""

    model_config = ConfigDict(frozen=True)


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: ToolFunction


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """An inbound chat completion request.

    ``messages`` stays as raw mappings rather than a parsed model. The chat
    template understands multipart content, ``reasoning_content`` and echoed
    tool calls better than a wire schema would, and re-validating those shapes
    here would only add a second place for them to be wrong.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | Mapping[str, Any] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    """OpenAI's newer name for ``max_tokens``. Accepted as a synonym."""
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    @property
    def completion_budget(self) -> int | None:
        return self.max_tokens if self.max_tokens is not None else self.max_completion_tokens

    @property
    def stop_strings(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        if isinstance(self.stop, str):
            return (self.stop,)
        return tuple(s for s in self.stop if isinstance(s, str) and s)

    @property
    def wants_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class FunctionCallPayload(_Wire):
    name: str | None = None
    arguments: str | None = None


class ToolCallPayload(_Wire):
    index: int
    id: str | None = None
    type: str | None = "function"
    function: FunctionCallPayload


class Delta(_Wire):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallPayload] | None = None


class StreamChoice(_Wire):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class PromptTokensDetails(_Wire):
    cached_tokens: int = 0


class Usage(_Wire):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletionChunk(_Wire):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=unix_now)
    model: str
    choices: list[StreamChoice] = Field(default_factory=list)
    usage: Usage | None = None

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"


class AssistantMessage(_Wire):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallPayload] | None = None


class ChatCompletionChoice(_Wire):
    index: int = 0
    message: AssistantMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(_Wire):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=unix_now)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(_Wire):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=unix_now)
    owned_by: str = "titan"
    max_model_len: int | None = None
    """vLLM-compatible extension: lets a client discover the context window
    from the listing rather than guessing it."""


class ModelsResponse(_Wire):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorDetail(_Wire):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(_Wire):
    error: ErrorDetail
