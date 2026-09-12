"""Server-sent events: engine events to OpenAI streaming chunks.

The mapping is not quite mechanical, and the awkward parts are the ones clients
depend on:

- Content and reasoning are separate channels in the same chunk stream. The
  tool-call parser decides which channel a delta belongs to, and markup never
  reaches either.
- A tool call streams as an indexed delta carrying ``id``, ``type`` and
  ``function`` with both the name and the complete argument JSON. Clients
  accumulate by index.
- ``finish_reason`` is ``tool_calls`` when the turn ended in calls, ``stop`` or
  ``length`` otherwise. Getting this wrong makes an agent loop hang rather than
  fail.
- The final chunk carries ``usage`` including ``cached_tokens``, when the client
  asked for it.
- A client disconnect cancels the sequence at the next turn boundary.

**Why a whole tool call arrives in one delta.** OpenAI's cloud API dribbles
arguments out as fragments, and the shape here permits that, but Titan cannot:
the argument values are XML text that must be coerced against the tool's JSON
schema, and the type of ``42`` is not known until the closing tag says which
parameter it belonged to. Emitting fragments would mean emitting them before
they can be typed. oMLX made the same call, so this is also what the recorded
streams these clients were built against actually contain.

**Why the keepalive is a chunk and not a comment.** A 90k-token prefill produces
no output for tens of seconds and clients with read timeouts hang up. An SSE
comment (``: ping``) is the protocol-correct filler, and some clients in this
population cannot parse comment lines at all. So the default is a no-op *event*:
a chunk with an empty content delta, ``created: 0`` and ``model: "keepalive"``,
carrying the stream's own id. The id matters — strict accumulators latch the
first chunk's id and silently drop every later chunk that disagrees, which would
throw away the real tool calls. ``comment`` and ``off`` modes stay available in
config for clients that prefer them.
"""

from __future__ import annotations

from typing import Iterable

from titan.api.models import (
    ChatCompletionChunk,
    Delta,
    FunctionCallPayload,
    StreamChoice,
    ToolCallPayload,
    Usage,
)
from titan.core.types import FinishReason, ToolCall

__all__ = [
    "DONE_FRAME",
    "KEEPALIVE_COMMENT",
    "keepalive_frame",
    "role_chunk",
    "content_chunk",
    "reasoning_chunk",
    "tool_calls_chunk",
    "finish_chunk",
    "usage_chunk",
    "error_frame",
    "finish_reason_for",
    "tool_call_payloads",
]

DONE_FRAME = "data: [DONE]\n\n"
KEEPALIVE_COMMENT = ": ping\n\n"


def keepalive_frame(response_id: str, mode: str = "chunk") -> str | None:
    """A frame that holds the connection open without meaning anything.

    Hand-built rather than serialised from a model: it is a fixed byte string
    emitted on a timer, and its ``created: 0`` sentinel plus ``keepalive`` model
    name are how a client (or a recorded stream) tells it apart from real
    output.
    """
    if mode == "off":
        return None
    if mode == "comment":
        return KEEPALIVE_COMMENT
    return (
        'data: {"id":"' + response_id + '","object":"chat.completion.chunk",'
        '"created":0,"model":"keepalive",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
        '"finish_reason":null}]}\n\n'
    )


def _chunk(response_id: str, model: str, delta: Delta, created: int | None = None) -> ChatCompletionChunk:
    kwargs = {"id": response_id, "model": model, "choices": [StreamChoice(delta=delta)]}
    if created is not None:
        kwargs["created"] = created
    return ChatCompletionChunk(**kwargs)


def role_chunk(response_id: str, model: str, created: int | None = None) -> ChatCompletionChunk:
    """The opening chunk. Cloud APIs always send one and clients rely on it."""
    return _chunk(response_id, model, Delta(role="assistant"), created)


def content_chunk(response_id: str, model: str, text: str, created: int | None = None) -> ChatCompletionChunk:
    return _chunk(response_id, model, Delta(content=text), created)


def reasoning_chunk(response_id: str, model: str, text: str, created: int | None = None) -> ChatCompletionChunk:
    return _chunk(response_id, model, Delta(reasoning_content=text), created)


def tool_call_payloads(calls: Iterable[ToolCall]) -> list[ToolCallPayload]:
    return [
        ToolCallPayload(
            index=call.index,
            id=call.call_id,
            type="function",
            function=FunctionCallPayload(name=call.name, arguments=call.arguments),
        )
        for call in calls
    ]


def tool_calls_chunk(
    response_id: str, model: str, calls: Iterable[ToolCall], created: int | None = None
) -> ChatCompletionChunk:
    return _chunk(response_id, model, Delta(tool_calls=tool_call_payloads(calls)), created)


def finish_chunk(
    response_id: str, model: str, finish_reason: str, created: int | None = None
) -> ChatCompletionChunk:
    kwargs = {
        "id": response_id,
        "model": model,
        "choices": [StreamChoice(delta=Delta(), finish_reason=finish_reason)],
    }
    if created is not None:
        kwargs["created"] = created
    return ChatCompletionChunk(**kwargs)


def usage_chunk(
    response_id: str, model: str, usage: Usage, created: int | None = None
) -> ChatCompletionChunk:
    """Trailing usage frame. ``choices`` is empty, per the OpenAI convention."""
    kwargs = {"id": response_id, "model": model, "choices": [], "usage": usage}
    if created is not None:
        kwargs["created"] = created
    return ChatCompletionChunk(**kwargs)


def error_frame(message: str, error_type: str = "server_error") -> str:
    """Report a mid-stream failure.

    The response is already committed by the time an engine fault surfaces, so
    there is no status code left to use. Emitting an error object is the only
    way to tell the client something went wrong; the alternative, ending the
    stream with a clean ``stop``, is a lie that looks like an empty answer.
    """
    import json

    body = json.dumps({"error": {"message": message, "type": error_type}})
    return f"data: {body}\n\n"


def finish_reason_for(reason: FinishReason | None, *, had_tool_calls: bool) -> str:
    """Map the engine's reason onto the four values OpenAI clients handle.

    ``tool_calls`` wins over everything: a turn that produced calls and then hit
    its token ceiling still needs the agent loop to run those calls, and a
    ``length`` there would make the client discard them.
    """
    if had_tool_calls:
        return "tool_calls"
    if reason is FinishReason.LENGTH:
        return "length"
    return "stop"
