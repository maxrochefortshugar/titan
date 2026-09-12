"""Incremental parser for the qwen3_coder XML tool-call dialect.

The chat template tells the model to answer in exactly this shape::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Berlin
    </parameter>
    </function>
    </tool_call>

and the model emits a ``<think>...</think>`` block before it. This module turns
that byte stream, arriving a token at a time, into three channels: reasoning
deltas, content deltas, and completed tool calls.

Reimplemented from oMLX's ``omlx/api/tool_calling.py`` (Apache-2.0), which is
the behaviour these clients are already calibrated against. Two of its hard-won
rules are reproduced deliberately rather than reinvented:

*Envelope boundaries are structural, not textual.* A parameter value is free to
contain the literal text ``</parameter>``, ``</function>`` or even a whole
``</function></tool_call>`` sequence, and file-editing tools do this constantly
because they carry source code that talks about the format. The envelope
therefore ends at the ``</function>`` that is followed by ``</tool_call>`` *and*
leaves the ``<parameter=``/``</parameter>`` counts balanced. Anything else is a
copy sitting inside a value.

*Values are coerced by declared schema type.* The XML carries no types, so a
``number`` parameter arrives as the characters ``42``. Without the tool's JSON
schema the parser has to guess, and guessing turns a ``"42"`` string argument
into an integer and breaks the tool. See :func:`coerce_parameter`.

The one place this parser deliberately departs from its reference is failure
handling, and it is the reason the module exists:

**An envelope that cannot be parsed is emitted as content.** Never dropped.
A truncated tool call (the model hit its token budget mid-envelope) used to
vanish silently, so the turn ended with empty content and ``finish_reason:
stop`` and the agent loop stalled with nothing on screen and nothing in the log
to explain it. Surfacing the raw text is ugly, and being ugly is the point: the
user sees markup, the log carries a warning, and the failure is diagnosable in
seconds instead of an evening. ``finish()`` also returns an error string, so the
API layer can attach it to the response.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from titan.core.types import ToolCall

__all__ = [
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "FUNCTION_CLOSE",
    "PARAMETER_OPEN",
    "PARAMETER_CLOSE",
    "THINK_OPEN",
    "THINK_CLOSE",
    "ParseStep",
    "coerce_parameter",
    "iter_xml_parameters",
    "Qwen3CoderToolParser",
]

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_FUNCTION_OPEN_RE = re.compile(r"<function=([^>\s]+)>")
_PARAMETER_OPEN_RE = re.compile(r"<parameter=([^>\s]+)>")

# Declared-type buckets, mirroring oMLX so a Titan-parsed call and an
# oMLX-parsed call produce identical argument JSON for the same model output.
_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})
_BOOL_TYPES = frozenset({"boolean", "bool", "binary"})
_CONTAINER_TYPES = frozenset({"object", "array", "arr"})
_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")

# Deeply nested model output breaks the decoders inconsistently: json.loads
# raises RecursionError on some builds and JSONDecodeError on others, and
# ast.literal_eval raises SyntaxError. Neither is a ValueError, so both escape
# excepts written for decode errors. Model output is untrusted; a breach has to
# be a clean parse failure, never a raised exception out of feed().
_DECODE_ERRORS = (json.JSONDecodeError, ValueError, RecursionError, SyntaxError)

# A value stuffed with fake terminators must not make the balance check
# quadratic. Past the cap the envelope is treated as unresolved, which is the
# safe direction: it surfaces as content rather than parsing as something else.
_MAX_END_CANDIDATES = 64


def _default_id_factory() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _repair_json_value(val: str) -> Any | None:
    """Best-effort repair of near-valid JSON with unbalanced brackets.

    Small models close an array with ``}``, or run out of budget mid-string.
    Rewrites mismatched closers to match the innermost opener, drops closers
    with no opener, terminates an unclosed string and appends what is missing.
    Returns ``None`` when the repaired text still will not parse.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in val:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
            out.append(ch)
        elif ch in "]}":
            if stack:
                out.append(stack.pop())
        else:
            out.append(ch)
    if in_string:
        out.append('"')
    while stack:
        out.append(stack.pop())
    try:
        return json.loads("".join(out), strict=False)
    except _DECODE_ERRORS:
        return None


def coerce_parameter(value: str, key: str, properties: Mapping[str, Any], func_name: str) -> Any:
    """Convert one XML-extracted parameter value using its declared type.

    The rules, in the order they are applied:

    - No declared type (undeclared parameter, a union list, an ``anyOf``): try
      JSON, keep the raw string if that fails. Guessing is all that is left.
    - The literal ``null`` is ``None`` for any declared type.
    - ``string``: keep the characters verbatim. The one exception is a value
      that is a JSON-quoted string literal, which is unwrapped. This asymmetry
      is deliberate: a string parameter must never come out as a number, so
      ``42`` stays ``"42"``, but a model that wrapped its string in quotes meant
      the inside, not the quotes.
    - ``boolean``: only the exact words ``true``/``false`` convert.
    - integer and number types: parsed, with a whole float narrowed to ``int``
      so ``3.0`` for an ``integer`` parameter does not arrive as a float.
    - Anything else, and any of the above that failed to convert: JSON, then a
      Python literal (models emit ``{'a': 1}``), then, for declared containers
      only, the bracket repair above. A container that still will not parse
      keeps its raw string and logs, because a wrong shape the tool can reject
      beats a silently emptied argument.
    """
    spec = properties.get(key)
    raw_type = spec.get("type") if isinstance(spec, Mapping) else None
    if not isinstance(raw_type, str):
        try:
            return json.loads(value)
        except _DECODE_ERRORS:
            return value

    if value.strip().lower() == "null":
        return None

    ptype = raw_type.strip().lower()

    if ptype in _STRING_TYPES:
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            try:
                decoded = json.loads(stripped)
            except _DECODE_ERRORS:
                decoded = None
            if isinstance(decoded, str):
                return decoded
        return value

    if ptype in _BOOL_TYPES:
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    elif ptype.startswith(_INT_PREFIXES):
        try:
            return int(value.strip())
        except ValueError:
            pass
    elif ptype.startswith(("num", "float", "double", "decimal")):
        try:
            num = float(value.strip())
        except (ValueError, OverflowError):
            pass
        else:
            return int(num) if num.is_integer() else num

    try:
        return json.loads(value, strict=False)
    except _DECODE_ERRORS:
        pass

    try:
        literal = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    else:
        if isinstance(literal, (dict, list, tuple)):
            return list(literal) if isinstance(literal, tuple) else literal

    if ptype in _CONTAINER_TYPES or ptype.startswith(("dict", "list")):
        repaired = _repair_json_value(value)
        if repaired is not None:
            logger.warning(
                "repaired malformed JSON for parameter %r of tool %r (declared %r)",
                key, func_name, ptype,
            )
            return repaired
        logger.warning(
            "parameter %r of tool %r failed to parse as declared type %r; keeping raw string",
            key, func_name, ptype,
        )
    return value


def _xml_element_value_end(text: str, start: int, close_tag: str, next_open: str) -> int:
    """End index of an XML value, tolerating ``close_tag`` inside the value.

    The value ends at the ``close_tag`` whose next non-space token is either
    ``next_open`` (a sibling element) or the end of the enclosing text. A copy
    inside the value is followed by neither. Falls back to the first
    ``close_tag``, then to the end of the text.
    """
    search = start
    while True:
        close = text.find(close_tag, search)
        if close < 0:
            first = text.find(close_tag, start)
            return first if first >= 0 else len(text)
        after = _skip_ws(text, close + len(close_tag))
        if after >= len(text) or text.startswith(next_open, after):
            return close
        search = close + len(close_tag)


def iter_xml_parameters(params_text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(key, value)`` for each ``<parameter=k>v</parameter>`` element.

    Scanning advances past each *value* rather than matching open tags with
    ``finditer``, so a literal ``<parameter=`` inside a value cannot start a
    spurious element and a literal ``</parameter>`` cannot end one early.
    """
    pos = 0
    while True:
        match = _PARAMETER_OPEN_RE.search(params_text, pos)
        if not match:
            return
        value_end = _xml_element_value_end(
            params_text, match.end(), PARAMETER_CLOSE, PARAMETER_OPEN
        )
        yield match.group(1), params_text[match.end():value_end].strip()
        pos = value_end + len(PARAMETER_CLOSE)


def _body_is_well_formed(body: str) -> bool:
    """Whether ``body`` is a complete ``<function=…>`` element and nothing more.

    "Well-formed" means the body walks cleanly with the same scanner that will
    extract the arguments (:func:`iter_xml_parameters`): every parameter element
    closes, and nothing but whitespace sits between them or after the last one.

    oMLX checks instead that the ``<parameter=``/``</parameter>`` tags *balance*
    in the candidate body. That check is wrong in both directions once a value
    may contain a literal close tag: it accepts the truncated body ending at a
    copy inside a value (counts 1 and 1), and it rejects the real body that
    contains the copy plus the genuine close (counts 1 and 2). Walking the
    elements decides both cases correctly, and costs a scan of text we are about
    to scan anyway.
    """
    stripped = body.strip()
    match = _FUNCTION_OPEN_RE.match(stripped)
    if not match:
        return False
    rest = stripped[match.end():]
    pos = 0
    while True:
        element = _PARAMETER_OPEN_RE.search(rest, pos)
        if element is None:
            break
        if rest[pos:element.start()].strip():
            return False
        value_end = _xml_element_value_end(
            rest, element.end(), PARAMETER_CLOSE, PARAMETER_OPEN
        )
        if not rest.startswith(PARAMETER_CLOSE, value_end):
            return False
        pos = value_end + len(PARAMETER_CLOSE)
    return not rest[pos:].strip()


def _envelope_body_end(text: str, body_start: int) -> int | None:
    """Index just past the ``</function>`` that really closes this envelope.

    Requires the structural pair: a ``</function>`` followed (after whitespace)
    by ``</tool_call>``, enclosing a body that is a well-formed function element
    (:func:`_body_is_well_formed`). Both conditions together skip copies
    embedded in values while still ending at the first call when several are
    concatenated.

    Returns ``None`` when this is not the XML dialect, or when the envelope has
    not finished arriving yet.
    """
    idx = _skip_ws(text, body_start)
    if not text.startswith(FUNCTION_OPEN, idx):
        return None
    search = idx
    for _ in range(_MAX_END_CANDIDATES):
        close = text.find(FUNCTION_CLOSE, search)
        if close < 0:
            return None
        after = close + len(FUNCTION_CLOSE)
        if text.startswith(TOOL_CALL_CLOSE, _skip_ws(text, after)):
            if _body_is_well_formed(text[idx:close]):
                return after
        search = after
    return None


def _partial_suffix_len(text: str, marker: str) -> int:
    """Length of the trailing run of ``text`` that could still become ``marker``.

    Withholding this much is what makes the parser safe to feed one character
    at a time: ``<tool`` at the end of a delta is not content yet, because the
    next delta may complete the marker.
    """
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


@dataclass(frozen=True, slots=True)
class ParseStep:
    """What one :meth:`Qwen3CoderToolParser.feed` produced."""

    content: str = ""
    reasoning: str = ""
    calls: tuple[ToolCall, ...] = ()

    def as_tuple(self) -> tuple[str, str, tuple[ToolCall, ...]]:
        """The ``ToolCallParser`` port's return shape."""
        return (self.content, self.reasoning, self.calls)


@dataclass
class Qwen3CoderToolParser:
    """Streaming parser for one assistant turn. Not reusable across turns.

    ``in_reasoning`` starts true because the generation prompt ends with an open
    ``<think>``: the model is already inside the thinking channel when its first
    token arrives, and there is no opening marker in the stream to detect. With
    ``enable_thinking`` false the template pre-closes the block instead, so the
    API layer constructs the parser with ``in_reasoning=False``.

    Tool-call markup is only recognised in the content channel. Inside the
    thinking block the model routinely writes about calling a tool, sometimes
    quoting the format, and treating that as a real call would fire a tool the
    model was only considering.
    """

    tools: Sequence[Mapping[str, Any]] = ()
    in_reasoning: bool = True
    id_factory: Callable[[], str] = _default_id_factory

    _buffer: str = field(default="", init=False)
    _envelope: str | None = field(default=None, init=False)
    _next_index: int = field(default=0, init=False)
    _properties: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _errors: list[str] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._properties = _index_tool_properties(self.tools)

    # -- public API ----------------------------------------------------------

    def feed(self, delta: str) -> tuple[str, str, tuple[ToolCall, ...]]:
        """Consume a text delta. Returns ``(content, reasoning, calls)``."""
        return self.step(delta).as_tuple()

    def step(self, delta: str) -> ParseStep:
        """:meth:`feed`, keeping the named fields."""
        if self._finished:
            raise RuntimeError("parser already finished")
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        self._buffer += delta
        self._drain(content, reasoning, calls, at_eof=False)
        return ParseStep("".join(content), "".join(reasoning), tuple(calls))

    def finish(self) -> tuple[str, str, tuple[ToolCall, ...], str | None]:
        """Flush the tail. Returns ``(content, reasoning, calls, error)``.

        An envelope still open at end of stream is flushed as *content*, and the
        returned error names it. Both, not either: the client needs the text to
        see what happened, and the API layer needs the error to log and to
        decide the finish reason.
        """
        if self._finished:
            raise RuntimeError("parser already finished")
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        self._drain(content, reasoning, calls, at_eof=True)

        if self._envelope is not None:
            # Truncated mid-envelope: the model ran out of budget, or the stream
            # was cut. This is the silent-drop case; surface it loudly.
            tail = self._envelope
            self._envelope = None
            logger.warning(
                "unterminated tool-call envelope at end of stream (%d chars); "
                "emitting as content: %.200r",
                len(tail), tail,
            )
            self._errors.append("unterminated tool-call envelope at end of stream")
            content.append(tail)

        if self._buffer:
            # A withheld partial marker that never completed is ordinary text.
            if self.in_reasoning:
                reasoning.append(self._buffer)
            else:
                content.append(self._buffer)
            self._buffer = ""

        self._finished = True
        error = "; ".join(self._errors) if self._errors else None
        return ("".join(content), "".join(reasoning), tuple(calls), error)

    @property
    def saw_tool_call(self) -> bool:
        return self._next_index > 0

    # -- internals -----------------------------------------------------------

    def _drain(
        self,
        content: list[str],
        reasoning: list[str],
        calls: list[ToolCall],
        *,
        at_eof: bool,
    ) -> None:
        """Consume as much of the buffer as can be resolved without more input."""
        while True:
            if self._envelope is not None:
                if not self._consume_envelope(content, calls, at_eof=at_eof):
                    return
                continue
            if self.in_reasoning:
                if not self._consume_reasoning(reasoning, at_eof=at_eof):
                    return
                continue
            if not self._consume_content(content, at_eof=at_eof):
                return

    def _consume_reasoning(self, reasoning: list[str], *, at_eof: bool) -> bool:
        """Emit reasoning up to ``</think>``. True when the mode changed."""
        buf = self._buffer
        close = buf.find(THINK_CLOSE)
        if close >= 0:
            if close:
                reasoning.append(buf[:close])
            self._buffer = buf[close + len(THINK_CLOSE):]
            self.in_reasoning = False
            return True
        if at_eof:
            return False
        hold = _partial_suffix_len(buf, THINK_CLOSE)
        emit = buf[: len(buf) - hold] if hold else buf
        if emit:
            reasoning.append(emit)
            self._buffer = buf[len(emit):]
        return False

    def _consume_content(self, content: list[str], *, at_eof: bool) -> bool:
        """Emit content up to the next marker. True when the mode changed."""
        buf = self._buffer
        open_at = buf.find(TOOL_CALL_OPEN)
        think_at = buf.find(THINK_OPEN)
        # A bare <think> after content means the model re-entered the thinking
        # channel; only honour it when it comes first.
        if think_at >= 0 and (open_at < 0 or think_at < open_at):
            if think_at:
                content.append(buf[:think_at])
            self._buffer = buf[think_at + len(THINK_OPEN):]
            self.in_reasoning = True
            return True
        if open_at >= 0:
            if open_at:
                content.append(buf[:open_at])
            self._envelope = TOOL_CALL_OPEN
            self._buffer = buf[open_at + len(TOOL_CALL_OPEN):]
            return True
        if at_eof:
            return False
        hold = max(
            _partial_suffix_len(buf, TOOL_CALL_OPEN),
            _partial_suffix_len(buf, THINK_OPEN),
        )
        emit = buf[: len(buf) - hold] if hold else buf
        if emit:
            content.append(emit)
            self._buffer = buf[len(emit):]
        return False

    def _consume_envelope(
        self, content: list[str], calls: list[ToolCall], *, at_eof: bool
    ) -> bool:
        """Try to close the open envelope. True when it closed (either way)."""
        assert self._envelope is not None
        pending = self._envelope[len(TOOL_CALL_OPEN):] + self._buffer
        self._buffer = ""
        self._envelope = TOOL_CALL_OPEN + pending

        body_end = _envelope_body_end(pending, 0)
        if body_end is not None:
            close_at = _skip_ws(pending, body_end)
            end = close_at + len(TOOL_CALL_CLOSE)
            self._emit_envelope(pending[:body_end], pending[:end], content, calls)
            self._envelope = None
            self._buffer = pending[end:]
            return True

        # Not the XML dialect, or not finished. A JSON-payload envelope ends at
        # its first close marker; only accept that reading when the payload does
        # not open a <function=, which the structural check above already owns.
        idx = _skip_ws(pending, 0)
        if not pending.startswith(FUNCTION_OPEN, idx):
            close_at = pending.find(TOOL_CALL_CLOSE)
            if close_at >= 0:
                end = close_at + len(TOOL_CALL_CLOSE)
                self._emit_envelope(pending[:close_at], pending[:end], content, calls)
                self._envelope = None
                self._buffer = pending[end:]
                return True

        if not at_eof:
            return False

        # End of stream with an unresolved XML envelope. Before giving up, take
        # the first close marker if there is one: a value containing a literal
        # terminator is far less likely than a model that simply forgot a tag,
        # and a recovered call beats markup on screen.
        close_at = pending.find(TOOL_CALL_CLOSE)
        if close_at >= 0:
            end = close_at + len(TOOL_CALL_CLOSE)
            logger.warning(
                "tool-call envelope did not close structurally; "
                "falling back to the first %s", TOOL_CALL_CLOSE,
            )
            self._errors.append("tool-call envelope recovered by first-close fallback")
            self._emit_envelope(pending[:close_at], pending[:end], content, calls)
            self._envelope = None
            self._buffer = pending[end:]
            return True
        return False

    def _emit_envelope(
        self, body: str, raw: str, content: list[str], calls: list[ToolCall]
    ) -> None:
        """Parse one closed envelope into a call, or surface it as content.

        ``raw`` is measured from just after the opening marker, which the caller
        has already consumed, so the marker is put back before the text goes out
        as content. Emitting the envelope without its opener would hand the user
        a fragment that looks like a parser bug rather than model output.
        """
        call = self._parse_envelope(body)
        if call is None:
            raw = TOOL_CALL_OPEN + raw
            logger.warning(
                "unparseable tool-call envelope; emitting as content: %.200r", raw
            )
            self._errors.append("unparseable tool-call envelope emitted as content")
            content.append(raw)
            return
        calls.append(call)

    def _parse_envelope(self, body: str) -> ToolCall | None:
        """Build a ``ToolCall`` from an envelope body, or ``None`` if malformed."""
        stripped = body.strip()
        if not stripped:
            return None

        # JSON dialect: {"name": ..., "arguments": {...}}. Some clients and some
        # fine-tunes emit it even under an XML template.
        if stripped[0] == "{":
            try:
                parsed = json.loads(stripped, strict=False)
            except _DECODE_ERRORS:
                parsed = None
            if isinstance(parsed, Mapping):
                name = parsed.get("name")
                if isinstance(name, str) and name:
                    arguments = parsed.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            decoded = json.loads(arguments)
                        except _DECODE_ERRORS:
                            decoded = None
                        arguments = decoded if isinstance(decoded, Mapping) else {}
                    if not isinstance(arguments, Mapping):
                        arguments = {}
                    return self._build_call(name, dict(arguments))
            return None

        match = _FUNCTION_OPEN_RE.match(stripped)
        if not match:
            return None
        # Bounded by rfind: the body is already delimited, so the element closes
        # at the LAST </function>, and a copy inside a value must not end it.
        close = stripped.rfind(FUNCTION_CLOSE)
        if close < match.end():
            return None
        name = match.group(1)
        params_text = stripped[match.end():close]
        properties = self._properties.get(name, {})
        arguments: dict[str, Any] = {}
        for key, value in iter_xml_parameters(params_text):
            arguments[key] = coerce_parameter(value, key, properties, name)
        return self._build_call(name, arguments)

    def _build_call(self, name: str, arguments: Mapping[str, Any]) -> ToolCall | None:
        """Serialise arguments and assign the call its stream index.

        Serialisation can fail on pathologically nested output. When it does the
        call is dropped rather than emitted with empty arguments: a runnable
        ``write_file`` with nothing to write is worse than a missing call.
        """
        try:
            serialised = json.dumps(dict(arguments), ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            logger.warning(
                "dropping tool call %.80r: arguments failed to serialise (%s: %s)",
                name, type(exc).__name__, exc,
            )
            self._errors.append(f"tool call {name!r} dropped: arguments unserialisable")
            return None
        index = self._next_index
        self._next_index += 1
        return ToolCall(
            index=index,
            call_id=self.id_factory(),
            name=name,
            arguments=serialised,
        )


def _index_tool_properties(tools: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool name to its declared parameter properties, for coercion."""
    out: dict[str, dict[str, Any]] = {}
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        source = function if isinstance(function, Mapping) else tool
        name = source.get("name")
        if not isinstance(name, str):
            continue
        parameters = source.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        out[name] = dict(properties) if isinstance(properties, Mapping) else {}
    return out
