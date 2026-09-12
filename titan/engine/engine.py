"""``TitanEngine``: the asyncio face of the scheduler thread.

The HTTP layer drives one method, ``generate(request) -> AsyncIterator``, and
knows nothing else about the engine. Everything below that call is the
scheduler loop on its own thread, so this module is a bridge and deliberately
nothing more: a queue, a sink, and the cancellation path.

    api (asyncio)                  TitanEngine                 EngineLoop
      async for event in  -->  asyncio.Queue  <-- sink <--  emit on the
      engine.generate(req)                                  loop thread

Three things it owns and the reason each one is here rather than in the loop.

**One exception boundary.** ``generate`` yields exactly one ``StreamEnd``, even
when admission refuses or the loop dies, because the response headers are on
the wire by the time anything can go wrong. A failure is a ``StreamEnd`` with
``FinishReason.ERROR``, never a raised exception mid-stream.

**Cancellation.** When the client hangs up, the API layer closes the async
generator. The ``finally`` here sends a cancel command, and the loop closes the
state handle at the next turn boundary, never mid-cycle. The generator does not
wait for the loop to confirm: the client is gone, and blocking a disconnect on
a decode cycle is how a dead connection holds a slot.

**The thread hand-off.** Events cross with ``call_soon_threadsafe``, which is
the one safe direction between a plain thread and a running event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Protocol, Union

from titan.core.types import Request, StreamEnd, TokenEvent

from titan.engine.scheduler import EngineLoop

__all__ = ["TitanEngine", "LoopRunner", "ThreadRunner", "InlineRunner", "EngineEvent"]

EngineEvent = Union[TokenEvent, StreamEnd]


class LoopRunner(Protocol):
    """Whatever owns the loop's thread. Started once, stopped once."""

    def start(self) -> None: ...

    def stop(self, drain_timeout_s: float) -> None: ...


class ThreadRunner:
    """The production runner: one daemon thread per engine, forever.

    One thread, not a pool. The loop owns the MLX stream and every cache-
    building op has to happen on it; a pool would be a correctness bug wearing
    a performance costume.
    """

    def __init__(self, loop: EngineLoop, *, name: str = "titan-engine") -> None:
        self.loop = loop
        self.name = name
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.loop.run_forever, name=self.name, daemon=True
        )
        self._thread.start()

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        thread, self._thread = self._thread, None
        self.loop.shutdown(drain_timeout_s)
        if thread is not None:
            thread.join(timeout=drain_timeout_s)


class InlineRunner:
    """Drives the loop from the event loop itself, one step at a time.

    For tests and for the bench harness, where a real thread turns a
    deterministic sequence of steps into a race. Same ``step`` as production:
    what changes is who calls it.
    """

    def __init__(self, loop: EngineLoop, *, idle_sleep: float = 0.0) -> None:
        self.loop = loop
        self.idle_sleep = idle_sleep
        self._task: asyncio.Task | None = None
        self._stop = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while not self._stop:
            worked = self.loop.step()
            await asyncio.sleep(0 if worked else self.idle_sleep)

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            self._task = None


class TitanEngine:
    """Satisfies ``titan.api.ports.Engine``.

    Structurally, not by inheritance: the Engine protocol lives in
    ``titan.api.ports`` and the engine layer may not import the API layer. The
    protocol is a ``typing.Protocol``, so shape is the whole contract.

    Wiring, in one line:

        engine = TitanEngine(EngineLoop(backend=..., tokenizer=..., cycle=...))

    ``titan.config.wiring`` builds the loop with the real adapters, calls
    ``engine.start()`` before the API binds its port, and ``engine.stop()`` on
    shutdown. Everything the HTTP layer needs is on ``ChatDeps.engine``.
    """

    def __init__(
        self,
        loop: EngineLoop,
        *,
        runner: LoopRunner | None = None,
        queue_maxsize: int = 0,
    ) -> None:
        self.loop = loop
        self.runner = runner if runner is not None else ThreadRunner(loop)
        self.queue_maxsize = queue_maxsize
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self._started:
            self.runner.start()
            self._started = True

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        if self._started:
            self.runner.stop(drain_timeout_s)
            self._started = False

    # -- the port ----------------------------------------------------------
    async def generate(self, request: Request) -> AsyncIterator[EngineEvent]:
        """Stream one request. Zero or more ``TokenEvent``, then one ``StreamEnd``.

        The engine starts itself on the first call so a test never has to
        remember to, and production still starts it explicitly before the port
        is bound, which is where warmup belongs.
        """
        self.start()
        event_loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)

        def sink(event: EngineEvent) -> None:
            # Called on the scheduler thread. This is the only cross-thread
            # touch in the engine, and it is the direction asyncio allows.
            event_loop.call_soon_threadsafe(events.put_nowait, event)

        finished = False
        try:
            self.loop.submit(request, sink)
            while True:
                event = await events.get()
                yield event
                if isinstance(event, StreamEnd):
                    finished = True
                    return
        finally:
            # Covers every way out that is not a StreamEnd: the consumer closed
            # the generator, its task was cancelled, or it raised. A
            # CancelledError passes straight through -- swallowing one leaves
            # the task tree in a state nobody can reason about -- and the cancel
            # command is sent on its way past.
            if not finished:
                self.loop.cancel(request.request_id)

    # -- observability -----------------------------------------------------
    def stats(self) -> Any:
        return self.loop.stats()
