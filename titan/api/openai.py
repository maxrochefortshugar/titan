"""Route handlers and the translation between wire models and core types.

Endpoints: ``POST /v1/chat/completions``, ``GET /v1/models``, ``GET /health``.
Authentication is a bearer key read from a file and compared in constant time;
there is no key management, because there is one client population on one
tailnet.

The layer holds no state. Everything it needs arrives as a port on
:class:`ChatDeps`, which is what lets the whole surface be tested against a fake
engine with no model, no GPU and no tokenizer weights.

**Aliases are resolved here and nowhere else.** A request for
``<model>:no-think`` picks a profile that sets template kwargs and sampling
defaults, then renders and samples as if the client had asked for all of it
explicitly. The engine never learns that aliases exist.

**Every request is rendered before it is admitted.** Templating and tokenising
happen on the event loop, which is fine (both are microseconds against a
multi-second prefill) and buys the limit checks something real to check: the
context ceiling is enforced against the prompt that will actually be run, not
an estimate of it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from fastapi import Depends, FastAPI, Header, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from titan.adapters.chat.template import TemplateError, normalise_reasoning_effort
from titan.adapters.chat.tool_parser import Qwen3CoderToolParser
from titan.api import models as wire
from titan.api import sse
from titan.api.ports import Engine, TemplateRenderer, Tokenizer
from titan.config.settings import AliasConfig, TitanConfig
from titan.core.types import (
    FinishReason,
    Request,
    RequestId,
    SamplingParams,
    StopCondition,
    StreamEnd,
    TokenEvent,
    ToolCall,
)

__all__ = ["ChatDeps", "ApiError", "build_app", "resolve_generation"]

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """A request that is wrong in a way the client can fix."""

    def __init__(self, message: str, *, status: int = 400, error_type: str = "invalid_request_error", param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_response(self) -> JSONResponse:
        body = wire.ErrorResponse(
            error=wire.ErrorDetail(
                message=self.message, type=self.error_type, param=self.param, code=self.code
            )
        )
        return JSONResponse(status_code=self.status, content=body.model_dump(exclude_none=True))


@dataclass(frozen=True)
class ChatDeps:
    """Everything the HTTP surface needs, as ports.

    ``parser_factory`` and ``id_factory`` are injectable for one reason: tests
    assert on exact SSE bytes, and a random call id makes that impossible. They
    are not a plugin point.
    """

    config: TitanConfig
    engine: Engine
    renderer: TemplateRenderer
    tokenizer: Tokenizer
    parser_factory: Callable[..., Qwen3CoderToolParser] = Qwen3CoderToolParser
    id_factory: Callable[[], str] = wire.new_completion_id
    call_id_factory: Callable[[], str] = wire.new_call_id
    clock: Callable[[], float] = time.monotonic
    created_factory: Callable[[], int] = wire.unix_now
    profiler: Any = None
    """Optional. ``GET /metrics`` serves its snapshot when there is one."""
    resolved_config: Mapping[str, Any] | None = None
    """The resolved config, redacted, echoed at ``GET /metrics`` so that every
    measurement can name the configuration it came from."""


@dataclass(frozen=True)
class ResolvedGeneration:
    """One request, resolved against its alias and the server's limits."""

    core_request: Request
    alias: AliasConfig
    enable_thinking: bool
    reasoning_effort: str
    tools: tuple[dict[str, Any], ...]
    prompt: str


def _merge_sampling(alias: AliasConfig, request: wire.ChatCompletionRequest) -> SamplingParams:
    """Request values win over the alias profile, field by field.

    Absent means absent: a client that sends no ``temperature`` gets the
    profile's, not OpenAI's 1.0 default. That distinction is the whole reason
    the wire model uses ``None`` defaults, and getting it wrong is how a
    carefully tuned 0.7 silently becomes 1.0 for every client that omits it.
    """
    defaults = alias.sampling
    return SamplingParams(
        temperature=defaults.temperature if request.temperature is None else request.temperature,
        top_p=defaults.top_p if request.top_p is None else request.top_p,
        top_k=defaults.top_k if request.top_k is None else request.top_k,
        min_p=defaults.min_p if request.min_p is None else request.min_p,
        repetition_penalty=(
            defaults.repetition_penalty
            if request.repetition_penalty is None
            else request.repetition_penalty
        ),
        seed=request.seed,
    )


def _resolve_template_settings(
    alias: AliasConfig, request: wire.ChatCompletionRequest
) -> tuple[bool, str, dict[str, Any]]:
    """Settle ``enable_thinking``, the effort, and the leftover template kwargs.

    Precedence is explicit-beats-implicit twice over: an explicit
    ``chat_template_kwargs.enable_thinking`` beats the alias, and a top-level
    ``reasoning_effort`` beats both the kwargs and the alias. Clients disagree
    about which of the two places to use, and both are in active use here.
    """
    kwargs = dict(alias.template_kwargs)
    kwargs.update(dict(request.chat_template_kwargs or {}))

    enable_thinking = alias.enable_thinking
    if "enable_thinking" in kwargs:
        value = kwargs.pop("enable_thinking")
        if not isinstance(value, bool):
            raise ApiError(
                "chat_template_kwargs.enable_thinking must be a boolean",
                param="chat_template_kwargs.enable_thinking",
            )
        enable_thinking = value

    effort_source = request.reasoning_effort
    if effort_source is None:
        effort_source = kwargs.pop("reasoning_effort", None)
    else:
        kwargs.pop("reasoning_effort", None)
    if effort_source is None:
        effort_source = alias.reasoning_effort
    try:
        effort = normalise_reasoning_effort(effort_source)
    except TemplateError as exc:
        raise ApiError(str(exc), param="reasoning_effort") from exc

    kwargs.pop("add_generation_prompt", None)
    return enable_thinking, effort, kwargs


def resolve_generation(deps: ChatDeps, request: wire.ChatCompletionRequest) -> ResolvedGeneration:
    """Validate, render, tokenise. Raises :class:`ApiError` on client mistakes."""
    config = deps.config
    alias = config.model.resolve(request.model)
    if alias is None:
        raise ApiError(
            f"model {request.model!r} is not served by this instance; "
            f"available: {', '.join(config.model.served_names())}",
            status=404,
            error_type="not_found_error",
            param="model",
            code="model_not_found",
        )
    if not request.messages:
        raise ApiError("messages must not be empty", param="messages")

    enable_thinking, effort, template_kwargs = _resolve_template_settings(alias, request)
    tools = tuple(t.model_dump() for t in (request.tools or []))

    if request.tool_choice in ("none",) and tools:
        # The template has no way to express "tools exist but do not call one",
        # so honour it by not rendering the tool block at all. Dropping the
        # tools is the only reading that matches the client's intent.
        tools = ()

    try:
        prompt = deps.renderer.render(
            request.messages,
            tools,
            reasoning_effort=effort,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            template_kwargs=template_kwargs,
        )
    except TemplateError as exc:
        raise ApiError(f"chat template rejected the conversation: {exc}", param="messages") from exc

    prompt_tokens = tuple(deps.tokenizer.encode(prompt))

    limits = config.limits
    budget = request.completion_budget
    if budget is None:
        budget = alias.sampling.max_tokens or limits.max_tokens
    if budget > limits.max_tokens:
        raise ApiError(
            f"max_tokens {budget} exceeds the server limit of {limits.max_tokens}",
            param="max_tokens",
        )
    if len(prompt_tokens) + budget > limits.max_context:
        raise ApiError(
            f"prompt of {len(prompt_tokens)} tokens plus max_tokens {budget} "
            f"exceeds the context limit of {limits.max_context}",
            param="messages",
            code="context_length_exceeded",
        )

    core_request = Request(
        request_id=RequestId(deps.id_factory()),
        prompt_tokens=prompt_tokens,
        sampling=_merge_sampling(alias, request),
        stop=StopCondition(
            eos_token_ids=deps.tokenizer.eos_token_ids,
            stop_strings=request.stop_strings,
            max_tokens=budget,
            max_total_tokens=limits.max_context,
        ),
        stream=request.stream,
        arrival_time=deps.clock(),
    )
    return ResolvedGeneration(
        core_request=core_request,
        alias=alias,
        enable_thinking=enable_thinking,
        reasoning_effort=effort,
        tools=tools,
        prompt=prompt,
    )


@dataclass
class _TurnAccumulator:
    """Runs the tool parser over an engine stream and collects the result.

    One instance per turn. It is the only thing that knows how raw model text
    becomes the three output channels, and both the streaming and the
    non-streaming path drive it, so the two cannot drift apart.
    """

    parser: Qwen3CoderToolParser
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    parse_error: str | None = None

    def feed(self, text: str) -> tuple[str, str, tuple[ToolCall, ...]]:
        step = self.parser.step(text)
        if step.content:
            self.content.append(step.content)
        if step.reasoning:
            self.reasoning.append(step.reasoning)
        self.calls.extend(step.calls)
        return step.as_tuple()

    def finish(self) -> tuple[str, str, tuple[ToolCall, ...]]:
        content, reasoning, calls, error = self.parser.finish()
        if content:
            self.content.append(content)
        if reasoning:
            self.reasoning.append(reasoning)
        self.calls.extend(calls)
        if error:
            self.parse_error = error
            logger.warning("tool parser reported: %s", error)
        return (content, reasoning, calls)

    @property
    def had_tool_calls(self) -> bool:
        return bool(self.calls)

    def joined_content(self) -> str:
        return "".join(self.content)

    def joined_reasoning(self) -> str:
        return "".join(self.reasoning)


def _usage_from(end: StreamEnd | None) -> wire.Usage:
    if end is None:
        return wire.Usage()
    return wire.Usage(
        prompt_tokens=end.prompt_tokens,
        completion_tokens=end.completion_tokens,
        total_tokens=end.prompt_tokens + end.completion_tokens,
        prompt_tokens_details=wire.PromptTokensDetails(cached_tokens=end.cached_tokens),
    )


async def _keepalive_merge(
    source: AsyncIterator[Any], interval: float
) -> AsyncIterator[Any]:
    """Yield from ``source``, inserting ``None`` whenever it goes quiet.

    A ``None`` means "nothing happened for ``interval`` seconds", which the
    caller turns into a keepalive frame. Done with a task and a timeout rather
    than by polling so that a token arriving 1 ms after the deadline is still
    forwarded immediately.
    """
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield None
                continue
            task, pending = pending, None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            pending.cancel()
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def _stream_chat(
    deps: ChatDeps,
    resolved: ResolvedGeneration,
    request: wire.ChatCompletionRequest,
    http_request: FastAPIRequest | None,
) -> AsyncIterator[str]:
    """The SSE body. Ordering here is the wire contract; see tests/api."""
    response_id = str(resolved.core_request.request_id)
    model_name = request.model
    created = deps.created_factory()
    mode = deps.config.server.sse_keepalive_mode
    interval = deps.config.server.sse_keepalive_seconds

    keepalive = sse.keepalive_frame(response_id, mode)
    if keepalive is not None:
        # One immediately: prefill starts now and can outlast a client's read
        # timeout before the first token exists.
        yield keepalive
    yield sse.role_chunk(response_id, model_name, created).to_sse()

    accumulator = _TurnAccumulator(
        deps.parser_factory(
            tools=resolved.tools,
            in_reasoning=resolved.enable_thinking,
            id_factory=deps.call_id_factory,
        )
    )
    end: StreamEnd | None = None

    def frames(content: str, reasoning: str, calls: Sequence[ToolCall]) -> list[str]:
        out: list[str] = []
        if reasoning:
            out.append(sse.reasoning_chunk(response_id, model_name, reasoning, created).to_sse())
        if content:
            out.append(sse.content_chunk(response_id, model_name, content, created).to_sse())
        if calls:
            out.append(sse.tool_calls_chunk(response_id, model_name, calls, created).to_sse())
        return out

    stream = deps.engine.generate(resolved.core_request)
    async for event in _keepalive_merge(stream, interval):
        if event is None:
            if keepalive is not None:
                yield keepalive
            if http_request is not None and await http_request.is_disconnected():
                logger.info("client disconnected during %s; cancelling", response_id)
                return
            continue
        if isinstance(event, StreamEnd):
            end = event
            break
        assert isinstance(event, TokenEvent)
        if not event.text:
            continue
        for frame in frames(*accumulator.feed(event.text)):
            yield frame

    for frame in frames(*accumulator.finish()):
        yield frame

    if end is not None and end.finish_reason is FinishReason.ERROR:
        message = end.error or "engine error"
        logger.error("engine error on %s: %s", response_id, message)
        yield sse.error_frame(message)

    finish_reason = sse.finish_reason_for(
        end.finish_reason if end else None, had_tool_calls=accumulator.had_tool_calls
    )
    yield sse.finish_chunk(response_id, model_name, finish_reason, created).to_sse()

    if request.wants_usage:
        yield sse.usage_chunk(response_id, model_name, _usage_from(end), created).to_sse()

    yield sse.DONE_FRAME


async def _complete_chat(
    deps: ChatDeps, resolved: ResolvedGeneration, request: wire.ChatCompletionRequest
) -> JSONResponse:
    """The non-streaming path. Same accumulator, one JSON body at the end."""
    response_id = str(resolved.core_request.request_id)
    accumulator = _TurnAccumulator(
        deps.parser_factory(
            tools=resolved.tools,
            in_reasoning=resolved.enable_thinking,
            id_factory=deps.call_id_factory,
        )
    )
    end: StreamEnd | None = None

    async for event in deps.engine.generate(resolved.core_request):
        if isinstance(event, StreamEnd):
            end = event
            break
        if event.text:
            accumulator.feed(event.text)
    accumulator.finish()

    if end is not None and end.finish_reason is FinishReason.ERROR:
        message = end.error or "engine error"
        logger.error("engine error on %s: %s", response_id, message)
        raise ApiError(message, status=500, error_type="server_error")

    content = accumulator.joined_content()
    reasoning = accumulator.joined_reasoning()
    body = wire.ChatCompletionResponse(
        id=response_id,
        created=deps.created_factory(),
        model=request.model,
        choices=[
            wire.ChatCompletionChoice(
                index=0,
                message=wire.AssistantMessage(
                    role="assistant",
                    # An empty string, not null: a turn that was pure tool calls
                    # still has a content field, and clients that concatenate it
                    # blindly crash on null.
                    content=content,
                    reasoning_content=reasoning or None,
                    tool_calls=sse.tool_call_payloads(accumulator.calls) or None,
                ),
                finish_reason=sse.finish_reason_for(
                    end.finish_reason if end else None,
                    had_tool_calls=accumulator.had_tool_calls,
                ),
            )
        ],
        usage=_usage_from(end),
    )
    return JSONResponse(content=body.model_dump(exclude_none=True))


def build_app(deps: ChatDeps) -> FastAPI:
    """Construct the FastAPI application against an assembled set of ports."""
    app = FastAPI(title="Titan", version="0.0.1", docs_url=None, redoc_url=None)
    app.state.deps = deps

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        """Bearer auth, constant-time, with the key read from disk per check.

        ``secrets.compare_digest`` rather than ``==``: the comparison is against
        an attacker-supplied string and an early-exit compare leaks the prefix.
        Cheap insurance even on a tailnet.
        """
        expected = deps.config.server.read_api_key()
        if expected is None:
            return
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise ApiError(
                "missing bearer token", status=401, error_type="authentication_error"
            )
        if not secrets.compare_digest(authorization[len(prefix):].strip(), expected):
            raise ApiError(
                "invalid bearer token", status=401, error_type="authentication_error"
            )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: FastAPIRequest, exc: ApiError) -> JSONResponse:
        return exc.to_response()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness. Unauthenticated on purpose: watchdogs do not carry keys."""
        return {
            "status": "ok",
            "model": deps.config.model.name,
            "max_context": deps.config.limits.max_context,
        }

    @app.get("/metrics", dependencies=[Depends(require_auth)])
    async def metrics(window: int = 0) -> dict[str, Any]:
        """Counters, the decode summary, and the configuration that produced
        them. One endpoint, because a benchmark number that cannot name its
        configuration is not evidence.

        ``window`` limits the decode summary, the stage breakdown and the
        acceptance curve to the last *n* cycles held in the ring. Zero, the
        default, is every cycle since the process started. It exists so a
        sweep can measure several contexts in one process: the counters say
        how many cycles a request cost, and that count is the window its own
        numbers are read back at. Nothing is reset, so two readers cannot
        take each other's measurements away.
        """
        body: dict[str, Any] = {"model": deps.config.model.name}
        if deps.profiler is not None:
            snapshot = getattr(deps.profiler, "snapshot", None)
            if callable(snapshot):
                body.update(dict(snapshot(window) if window else snapshot()))
            recent = getattr(deps.profiler, "recent_events", None)
            if callable(recent):
                body["recent_events"] = list(recent(60))
            cycles = getattr(deps.profiler, "recent_cycles", None)
            if callable(cycles):
                body["recent_cycles"] = list(cycles(12))
        stats = getattr(deps.engine, "stats", None)
        if callable(stats):
            loop = stats()
            body["loop"] = (
                dict(loop) if isinstance(loop, dict) else dataclasses.asdict(loop)
            )
        cache = getattr(getattr(deps.engine, "loop", None), "cache", None)
        cache_stats = getattr(cache, "stats", None)
        if callable(cache_stats):
            body["cache"] = dict(cache_stats())
        if deps.resolved_config is not None:
            body["config"] = dict(deps.resolved_config)
        return body

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models() -> dict[str, Any]:
        created = deps.created_factory()
        listing = wire.ModelsResponse(
            data=[
                wire.ModelInfo(
                    id=name,
                    created=created,
                    max_model_len=deps.config.limits.max_context,
                )
                for name in deps.config.model.served_names()
            ]
        )
        return listing.model_dump(exclude_none=True)

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(
        body: wire.ChatCompletionRequest, http_request: FastAPIRequest
    ) -> Any:
        resolved = resolve_generation(deps, body)
        if not body.stream:
            return await _complete_chat(deps, resolved, body)
        return StreamingResponse(
            _stream_chat(deps, resolved, body, http_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Proxies that buffer an SSE body turn a keepalive into a lie.
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
