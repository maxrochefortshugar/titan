"""What the HTTP client sees when the engine fails.

Three failure modes and three answers, and the point of the file is that the
client is never left holding an open connection:

* the engine ends a stream with ``FinishReason.ERROR`` -- an SSE error frame
  carrying the engine's own message, then a normal finish and ``[DONE]``;
* the engine raises instead of yielding -- the same error frame on the
  streaming path, a 500 with the message on the non-streaming one;
* the engine reports itself unhealthy -- ``GET /health`` answers 503 and new
  requests are refused with 503 before anything is rendered.

The fakes here raise on purpose. Their exception types and texts are asserted
on, because a failure message that loses the original type is a failure message
that costs somebody a decode round.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

import pytest

from tests.api.conftest import (
    MODEL_NAME,
    FakeEngine,
    make_client,  # noqa: F401  (fixture)
    make_config,
    sse_events,
)
from titan.core.errors import EngineError, EngineUnhealthyError
from titan.core.types import FinishReason, Request, StreamEnd, TokenEvent

pytestmark = pytest.mark.anyio

#: Every request in this file is answered by a fake with no I/O in it, so any
#: wait longer than this is a hang rather than a slow machine.
BOUND_S = 5.0


@pytest.fixture
def anyio_backend():
    return "asyncio"


def body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


@dataclass
class Health:
    """What ``EngineLoop.health()`` returns, as the API layer reads it."""

    healthy: bool = True
    reason: str | None = None
    port_failures: int = 0
    failed_requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "reason": self.reason,
            "port_failures": self.port_failures,
            "failed_requests": self.failed_requests,
        }


@dataclass
class HealthyEngine(FakeEngine):
    """A fake engine that also publishes a health report."""

    report: Health = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.report is None:
            self.report = Health()

    def health(self) -> Health:
        return self.report


class RaisingEngine:
    """Raises out of ``generate`` rather than ending the stream.

    The shape of a bridge that could not submit, and of any bug that gets past
    the engine's own guards. The API layer owes the client an answer either
    way.
    """

    def __init__(self, exc: BaseException, *, after: Sequence[str] = ()) -> None:
        self.exc = exc
        self.after = tuple(after)
        self.seen: list[Request] = []

    def generate(self, request: Request) -> AsyncIterator[Any]:
        self.seen.append(request)

        async def _stream() -> AsyncIterator[Any]:
            for text in self.after:
                yield TokenEvent(
                    request_id=request.request_id, token_ids=(1,), text=text
                )
            raise self.exc

        return _stream()


def frames(text: str) -> list[Any]:
    return sse_events(text)


def error_frames(events: Sequence[Any]) -> list[dict[str, Any]]:
    return [e for e in events if isinstance(e, dict) and "error" in e]


# ---------------------------------------------------------------------------
# the engine ends the stream with an error
# ---------------------------------------------------------------------------


async def test_a_streaming_error_finish_becomes_an_error_frame(make_client):
    engine = FakeEngine(
        deltas=["par", "tial"],
        finish_reason=FinishReason.ERROR,
        error="backend.prefill failed: RuntimeError: prefill exploded",
    )
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post("/v1/chat/completions", json=body(stream=True))

    assert response.status_code == 200
    events = frames(response.text)
    errors = error_frames(events)
    assert len(errors) == 1
    message = errors[0]["error"]["message"]
    # The original exception type and text both survive the trip.
    assert "RuntimeError" in message
    assert "prefill exploded" in message
    assert errors[0]["error"]["type"] == "server_error"
    # And the stream still ends properly, so a client that reads to the end
    # gets there instead of waiting on a socket.
    assert events[-1] == "[DONE]"


async def test_a_non_streaming_error_finish_becomes_a_500_with_the_message(make_client):
    engine = FakeEngine(
        finish_reason=FinishReason.ERROR,
        error="cycle.run failed: KeyError: 'sequence'",
    )
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post(
                "/v1/chat/completions", json=body(stream=False)
            )

    assert response.status_code == 500
    detail = response.json()["error"]
    assert "KeyError" in detail["message"]
    assert detail["type"] == "server_error"


async def test_the_partial_text_before_an_error_still_reaches_the_client(make_client):
    engine = FakeEngine(
        deltas=["Hello", " world"],
        finish_reason=FinishReason.ERROR,
        error="cycle.run failed: RuntimeError: boom",
    )
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post("/v1/chat/completions", json=body(stream=True))

    # Reasoning, because the default alias thinks and nothing closed the
    # block. Which channel it lands in is the parser's business; what this
    # test is about is that it lands somewhere rather than being dropped when
    # the turn ends badly.
    text = "".join(
        choice["delta"].get("content", "") + choice["delta"].get("reasoning_content", "")
        for event in frames(response.text)
        if isinstance(event, dict) and "choices" in event
        for choice in event["choices"]
    )
    assert text == "Hello world"
    assert error_frames(frames(response.text))


# ---------------------------------------------------------------------------
# the engine raises
# ---------------------------------------------------------------------------


async def test_an_engine_that_raises_mid_stream_still_closes_the_stream(make_client):
    engine = RaisingEngine(RuntimeError("the loop went away"), after=["par"])
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post("/v1/chat/completions", json=body(stream=True))

    assert response.status_code == 200
    events = frames(response.text)
    errors = error_frames(events)
    assert len(errors) == 1
    assert "RuntimeError" in errors[0]["error"]["message"]
    assert "the loop went away" in errors[0]["error"]["message"]
    assert events[-1] == "[DONE]"


async def test_an_engine_that_raises_before_the_first_token_still_answers(make_client):
    engine = RaisingEngine(EngineError("could not submit to the engine loop: full"))
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post("/v1/chat/completions", json=body(stream=True))

    errors = error_frames(frames(response.text))
    assert len(errors) == 1
    assert "could not submit" in errors[0]["error"]["message"]


async def test_an_engine_that_raises_on_the_non_streaming_path_is_a_500(make_client):
    engine = RaisingEngine(EngineError("could not submit to the engine loop: full"))
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post(
                "/v1/chat/completions", json=body(stream=False)
            )

    assert response.status_code == 500
    detail = response.json()["error"]
    assert "EngineError" in detail["message"]
    assert "could not submit" in detail["message"]


async def test_an_engine_that_falls_over_after_the_readiness_check_is_503(make_client):
    """The gap between ``/health`` saying yes and the loop taking the request.

    The engine reports itself healthy when the endpoint asks and raises
    ``EngineUnhealthyError`` a moment later. The client is owed the 503 it
    would have got had the fault landed a moment earlier, not a 500.
    """
    engine = RaisingEngine(EngineUnhealthyError("engine unhealthy: device error"))
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.post(
                "/v1/chat/completions", json=body(stream=False)
            )

    assert response.status_code == 503
    detail = response.json()["error"]
    assert detail["type"] == "service_unavailable"
    assert "device error" in detail["message"]


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


async def test_health_is_200_and_says_so_while_the_engine_is_serving(make_client):
    engine = HealthyEngine(deltas=["hi"])
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"]["healthy"] is True


async def test_health_is_503_when_the_engine_reports_unhealthy(make_client):
    engine = HealthyEngine(
        report=Health(
            healthy=False,
            reason="loop step exceeded its 120s budget (running for 131.4s)",
            port_failures=3,
            failed_requests=2,
        )
    )
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert "120s budget" in payload["reason"]
    assert payload["engine"]["port_failures"] == 3


async def test_an_unhealthy_engine_refuses_new_requests_with_503(make_client):
    engine = HealthyEngine(
        deltas=["hi"], report=Health(healthy=False, reason="the device is gone")
    )
    async with make_client(engine) as client:
        for stream in (True, False):
            async with asyncio.timeout(BOUND_S):
                response = await client.post(
                    "/v1/chat/completions", json=body(stream=stream)
                )
            assert response.status_code == 503
            detail = response.json()["error"]
            assert detail["type"] == "service_unavailable"
            assert "the device is gone" in detail["message"]
    # Refused before admission: the engine was never asked to generate.
    assert engine.seen == []


async def test_an_engine_with_no_health_report_is_served_as_healthy(make_client):
    """The ``Engine`` protocol is one method wide. A fake without ``health``
    must not become a 503."""
    engine = FakeEngine(deltas=["hi"])
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            health = await client.get("/health")
            response = await client.post(
                "/v1/chat/completions", json=body(stream=False)
            )

    assert health.status_code == 200
    assert "engine" not in health.json()
    assert response.status_code == 200


async def test_a_health_probe_that_raises_does_not_take_the_endpoint_down(make_client):
    class ExplodingHealth(FakeEngine):
        def health(self):
            raise RuntimeError("the health check itself is broken")

    engine = ExplodingHealth(deltas=["hi"])
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            health = await client.get("/health")
            response = await client.post(
                "/v1/chat/completions", json=body(stream=False)
            )

    assert health.status_code == 200
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


async def test_metrics_carries_the_failure_counters(make_client):
    @dataclass
    class Stats:
        port_failures: int = 4
        failed_requests: int = 2
        healthy: bool = False
        unhealthy_reason: str | None = "device error in backend.prefill"

    class CountingEngine(FakeEngine):
        def stats(self) -> Stats:
            return Stats()

    engine = CountingEngine(deltas=["hi"])
    async with make_client(engine) as client:
        async with asyncio.timeout(BOUND_S):
            response = await client.get("/metrics")

    assert response.status_code == 200
    loop = response.json()["loop"]
    assert loop["port_failures"] == 4
    assert loop["failed_requests"] == 2
    assert loop["healthy"] is False
    assert "device error" in loop["unhealthy_reason"]
