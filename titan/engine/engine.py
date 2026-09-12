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
import time
from typing import Any, AsyncIterator, Protocol, Union

from titan.core.errors import EngineError, EngineUnhealthyError
from titan.core.types import FinishReason, Request, StreamEnd, TokenEvent

from titan.engine.scheduler import EngineHealth, EngineLoop

__all__ = ["TitanEngine", "LoopRunner", "ThreadRunner", "InlineRunner", "EngineEvent"]

EngineEvent = Union[TokenEvent, StreamEnd]


class LoopRunner(Protocol):
    """Whatever owns the loop's thread. Started once, stopped once."""

    def start(self) -> None: ...

    def stop(self, drain_timeout_s: float) -> None: ...

    def alive(self) -> bool:
        """Whether the loop is still turning.

        The bridge asks periodically while a request is outstanding. A runner
        that does not implement it is assumed alive, which is the old
        behaviour; a runner that answers False releases every waiter with an
        error rather than leaving them on a queue nobody feeds.
        """


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
        # Wait for the loop to claim the thread before returning. A ``stop``
        # that arrives in the window before it does would see an unowned loop
        # and drain it from the caller's thread, which is two threads in one
        # turn. Bounded, because a loop that cannot reach its own first line
        # is a problem the caller cannot fix by waiting longer.
        self.loop.wait_until_running(1.0)

    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        """Stop within ``drain_timeout_s``, wedged loop step or not.

        One budget, split between the drain and the join, rather than one each:
        a caller that asked for five seconds meant five, and a loop stuck
        inside a port call would otherwise take ten. The thread is a daemon, so
        a join that times out leaves nothing behind that can hold the process
        open.
        """
        thread, self._thread = self._thread, None
        deadline = time.monotonic() + max(0.0, drain_timeout_s)
        self.loop.shutdown(drain_timeout_s)
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


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

    def alive(self) -> bool:
        task = self._task
        return task is not None and not task.done()

    async def _run(self) -> None:
        while not self._stop:
            try:
                worked = self.loop.step()
            except Exception as exc:  # noqa: BLE001 - same contract as the thread
                # The inline runner is the production loop driven by hand, so
                # it survives a fault the same way: the engine is marked
                # unhealthy, the sequences it held are failed, and the driver
                # keeps stepping.
                worked = self.loop.on_step_error(exc)
            await asyncio.sleep(0 if worked else self.idle_sleep)

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        """Shut the loop down, then drop the driving task.

        The shutdown comes first and it is the same call the thread runner
        makes: every live sequence is answered and every queued one is told the
        engine is going away. Cancelling the task without it would leave the
        loop holding sequences that nobody will ever step again, which is the
        same hang as a dead thread wearing a tidier name.
        """
        self._stop = True
        self.loop.shutdown(drain_timeout_s)
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
        liveness_poll_s: float = 1.0,
    ) -> None:
        self.loop = loop
        self.runner = runner if runner is not None else ThreadRunner(loop)
        self.queue_maxsize = queue_maxsize
        self.liveness_poll_s = max(0.001, float(liveness_poll_s))
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

    def health(self) -> EngineHealth:
        """Whether the engine is serving. ``GET /health`` reads this, and the
        chat endpoint refuses with a 503 when it says no."""
        return self.loop.health()

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
            try:
                # Checked here as well as at the HTTP layer, because the engine
                # can fall over in the gap between the two. Before the first
                # yield, so it is a status code rather than an error frame.
                health = self.loop.health()
                if health is not None and not health.healthy:
                    raise EngineUnhealthyError(f"engine unhealthy: {health.reason}")
                self.loop.submit(request, sink)
            except EngineError:
                raise
            except Exception as exc:  # noqa: BLE001 - nothing is on the wire yet
                # Before the first yield, so this one is allowed to raise: the
                # API layer still has a status code to set.
                raise EngineError(f"could not submit to the engine loop: {exc}") from exc
            while True:
                event = await self._next_event(events, request)
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

    async def _next_event(self, events: asyncio.Queue, request: Request) -> EngineEvent:
        """Wait for the loop's next event, and notice if the loop is gone.

        The queue is fed by one thread and nothing else, so a loop that stops
        without emitting a ``StreamEnd`` is a client that waits forever. That
        is the shape of the hang this whole path was written against, and it is
        cheap enough to keep a guard for it even now that the loop survives its
        own faults: a poll a second, per outstanding request, and a synthetic
        error finish rather than an open connection.
        """
        while True:
            try:
                # ``asyncio.timeout`` rather than ``asyncio.wait_for``. They
                # look interchangeable and are not: ``wait_for`` on 3.11
                # returns the inner result when the outer task is cancelled at
                # the moment the inner one has already completed, so a
                # disconnect during a token delivery would be swallowed and
                # the stream would run on with nobody reading it. The context
                # manager converts only its own cancellation into a timeout
                # and re-raises anybody else's, which is the behaviour this
                # loop needs. No shield either: a cancelled ``get`` leaves its
                # item in the queue, and shielding would hold a cancellation
                # of the consuming task at arm's length, which is how a
                # disconnect stops being a disconnect.
                async with asyncio.timeout(self.liveness_poll_s):
                    return await events.get()
            except TimeoutError:
                if self._runner_alive():
                    continue
                if not events.empty():
                    return events.get_nowait()
                return StreamEnd(
                    request_id=request.request_id,
                    finish_reason=FinishReason.ERROR,
                    prompt_tokens=len(request.prompt_tokens),
                    cached_tokens=0,
                    completion_tokens=0,
                    error="the engine loop stopped before this request finished",
                )

    def _runner_alive(self) -> bool:
        probe = getattr(self.runner, "alive", None)
        return True if not callable(probe) else bool(probe())

    # -- observability -----------------------------------------------------
    def stats(self) -> Any:
        return self.loop.stats()
