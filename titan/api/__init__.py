"""HTTP surface: OpenAI-compatible chat completions over FastAPI.

The API layer owns exactly three things: request validation, chat templating
plus tokenisation into a core :class:`~titan.core.types.Request`, and turning
the engine's event stream back into SSE chunks. It holds no engine state and
makes no scheduling decision.

Threading: FastAPI and uvicorn run asyncio on the main thread; the scheduler
owns its own thread and communicates through two queues (submissions in, events
out). Nothing in the engine is async, and nothing in the API touches mlx. That
split is deliberate. An awaited scheduler would put the decode loop at the mercy
of the event loop's fairness, and a decode cycle that yields between draft and
verify has lost the point of the cycle.
"""

from titan.api import models, openai, sse

__all__ = ["models", "openai", "sse"]
