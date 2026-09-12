"""``TitanEngine`` against the Engine protocol the HTTP layer drives.

The API layer does exactly two things with the engine: it iterates
``generate(request)`` until a ``StreamEnd``, and it closes the iterator when
the client hangs up. Both are checked here, on a real event loop, with the
scheduler driven by :class:`InlineRunner` so the test is deterministic rather
than timing-dependent. The runner is the only difference from production; the
loop, the cycle and the fakes are the same objects.
"""

from __future__ import annotations

import asyncio

import pytest

from titan.core.types import FinishReason, StreamEnd, TokenEvent
from titan.engine.admission import AdmissionConfig
from titan.engine.decode_cycle import MTPDecodeCycle, PlainDecodeCycle
from titan.engine.engine import InlineRunner, TitanEngine
from titan.engine.scheduler import EngineLoop

from tests.engine.conftest import (
    FakeBackend,
    FakeCache,
    FakeClock,
    FakeTokenizer,
    ScriptedDrafter,
    make_request,
)
from tests.engine.test_mtp_parity import perfect

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_engine(*, cache=None, drafter=None, backend=None, pieces=None, config=None):
    backend = backend or FakeBackend()
    tokenizer = FakeTokenizer(pieces or {})
    clock = FakeClock()
    config = config or AdmissionConfig()
    if drafter is None:
        cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)
    else:
        cycle = MTPDecodeCycle(
            backend=backend, tokenizer=tokenizer, drafter=drafter, clock=clock
        )
    loop = EngineLoop(
        backend=backend,
        tokenizer=tokenizer,
        cycle=cycle,
        config=config,
        cache=cache,
        clock=clock,
    )
    engine = TitanEngine(loop, runner=InlineRunner(loop))
    return engine, loop, backend


async def collect(engine, request):
    events = []
    async for event in engine.generate(request):
        events.append(event)
    return events


async def test_generate_streams_tokens_then_exactly_one_stream_end():
    engine, _loop, backend = build_engine()
    request = make_request((11, 12, 13), max_tokens=6)
    events = await collect(engine, request)

    assert isinstance(events[-1], StreamEnd)
    assert sum(isinstance(e, StreamEnd) for e in events) == 1
    tokens = [t for e in events if isinstance(e, TokenEvent) for t in e.token_ids]
    assert tokens == backend.greedy_continuation(request.prompt_tokens, 6)
    end = events[-1]
    assert end.finish_reason is FinishReason.LENGTH
    assert end.prompt_tokens == 3
    assert end.completion_tokens == 6
    assert end.cached_tokens == 0


async def test_usage_reports_cached_tokens_from_the_prefix_cache():
    config = AdmissionConfig()
    cache = FakeCache(matched=2048, config=config)
    engine, _loop, _backend = build_engine(cache=cache)
    events = await collect(engine, make_request(tuple(range(3000)), max_tokens=3))
    end = events[-1]
    assert end.cached_tokens == 2048
    assert end.prompt_tokens == 3000
    assert end.completion_tokens == 3


async def test_speculation_streams_the_same_tokens_as_the_plain_loop():
    plain_engine, _loop, plain_backend = build_engine()
    request = make_request((11, 12, 13), max_tokens=12)
    plain = await collect(plain_engine, request)

    engine, _loop, backend = build_engine(drafter=ScriptedDrafter(perfect(FakeBackend())))
    speculative = await collect(engine, make_request((11, 12, 13), max_tokens=12))

    def ids(events):
        return [t for e in events if isinstance(e, TokenEvent) for t in e.token_ids]

    assert ids(speculative) == ids(plain)
    # Same tokens, fewer events: that is the whole benefit and the whole
    # permitted difference.
    assert len(speculative) < len(plain)


async def test_closing_the_stream_early_frees_the_sequence():
    """The client hung up. The state handle closes at the next turn boundary."""
    engine, loop, backend = build_engine()
    request = make_request((11, 12, 13), max_tokens=1000)
    stream = engine.generate(request)
    first = await stream.__anext__()
    assert isinstance(first, TokenEvent)
    assert backend.open_handles

    await stream.aclose()
    for _ in range(4):
        loop.step()
    assert backend.open_handles == set()
    assert loop.live == ()


async def test_cancelling_the_consuming_task_frees_the_sequence():
    engine, loop, backend = build_engine()
    request = make_request((11, 12, 13), max_tokens=1000)

    async def consume():
        async for _event in engine.generate(request):
            pass

    task = asyncio.create_task(consume())
    for _ in range(6):
        await asyncio.sleep(0)
    assert backend.open_handles
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(4):
        loop.step()
    assert backend.open_handles == set()


async def test_two_requests_stream_concurrently():
    engine, _loop, _backend = build_engine()
    first, second = await asyncio.gather(
        collect(engine, make_request((11, 12, 13), request_id="a", max_tokens=5)),
        collect(engine, make_request((21, 22, 23), request_id="b", max_tokens=5)),
    )
    assert first[-1].request_id == "a"
    assert second[-1].request_id == "b"
    assert all(e.request_id == "a" for e in first)
    assert all(e.request_id == "b" for e in second)
    assert first[-1].completion_tokens == 5
    assert second[-1].completion_tokens == 5


async def test_a_refused_request_ends_the_stream_rather_than_raising():
    """The response headers are already on the wire, so a failure is an event."""
    engine, _loop, _backend = build_engine(config=AdmissionConfig(max_context=4))
    events = await collect(engine, make_request(tuple(range(100)), max_tokens=10))
    assert len(events) == 1
    assert isinstance(events[0], StreamEnd)
    assert events[0].finish_reason is FinishReason.ERROR
    assert "context window" in events[0].error


async def test_stop_strings_never_reach_the_client_through_the_stream():
    stream = [20, 23, 24, 25]

    def next_token(context, _vocab):
        index = len(context) - 3
        return stream[index] if 0 <= index < len(stream) else stream[-1]

    engine, _loop, _backend = build_engine(
        backend=FakeBackend(next_token=next_token),
        pieces={20: "Hello", 23: "<|im_", 24: "end|>", 25: "!"},
    )
    events = await collect(
        engine,
        make_request((1, 2, 3), max_tokens=10, stop_strings=("<|im_end|>",)),
    )
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert text == "Hello"
    assert events[-1].finish_reason is FinishReason.STOP
    assert events[-1].completion_tokens == 1
