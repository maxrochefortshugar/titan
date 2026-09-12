"""Ports the API layer consumes.

``titan.core.ports`` owns ``Tokenizer``, ``TemplateRenderer`` and
``ToolCallParser``; this module re-exports them so the API layer has a single
import site, and adds the one port the core does not yet declare: the
:class:`Engine` the HTTP surface drives.

The Engine port is deliberately thin. Generation is an async iterator of
``TokenEvent`` terminated by exactly one ``StreamEnd``, which is the shape
``titan.core.types`` already documents, so a fake engine in a test is a
generator function and nothing more.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, Union, runtime_checkable

from titan.core.ports import TemplateRenderer, Tokenizer, ToolCallParser
from titan.core.types import Request, StreamEnd, TokenEvent

__all__ = [
    "Engine",
    "EngineEvent",
    "TemplateRenderer",
    "Tokenizer",
    "ToolCallParser",
]

EngineEvent = Union[TokenEvent, StreamEnd]
"""What a generation stream yields. Exactly one ``StreamEnd``, and it is last."""


@runtime_checkable
class Engine(Protocol):
    """Generation, as the HTTP layer sees it.

    Invariant: :meth:`generate` yields zero or more ``TokenEvent`` followed by
    exactly one ``StreamEnd``, even when the request fails. A failure is a
    ``StreamEnd`` with ``finish_reason=FinishReason.ERROR`` and a populated
    ``error``, never a raised exception mid-stream: the response headers are
    already on the wire by then, so there is nothing left to turn into a 500.

    Invariant: ``TokenEvent.text`` is raw model text. The engine does not strip
    ``<think>`` markers or tool-call markup; that is the tool parser's job, and
    keeping it out of the engine is what lets the parser be tested on recorded
    streams without an engine at all.

    Invariant: cancelling the iterator (the client hung up) must release the
    sequence. The API layer closes the async generator on disconnect and does
    not wait for a ``StreamEnd`` after that.
    """

    def generate(self, request: Request) -> AsyncIterator[EngineEvent]:
        """Start generating. Called once per admitted request.

        Token accounting is reported on the terminal ``StreamEnd``
        (``prompt_tokens``, ``cached_tokens``, ``completion_tokens``) rather
        than queried separately, so the numbers the client sees are the ones
        the scheduler actually charged.
        """
