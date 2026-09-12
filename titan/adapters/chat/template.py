"""Chat template rendering for Qwen3.8-Flash-Next.

The model ships a ``chat_template.jinja`` and it is the authority: Titan renders
that file rather than reimplementing its rules in Python. A hand-written prompt
builder drifts from the template the moment the model is updated, and the drift
is silent, showing up as a model that stops calling tools rather than as an
error.

Three things about this template drive the code below.

**The tool block is the system message.** When ``tools`` is non-empty the
template emits its own system turn holding the tool list and the XML calling
instructions, then appends the caller's system content to it. There is no
separate place to inject tools, and the XML dialect the model is told to use is
exactly what ``titan.adapters.chat.tool_parser`` parses back.

**Reasoning effort is prompt text, not a sampling knob.** ``low`` and ``xhigh``
prepend an instruction sentence; ``medium`` deliberately prepends nothing. With
``enable_thinking`` false the whole block is skipped and the generation prompt
carries a pre-closed ``<think>\\n\\n</think>\\n\\n``, which is what makes the
model answer without thinking.

**Assistant tool calls round-trip through ``arguments|items``.** The template
iterates the arguments as a mapping when it renders history, but OpenAI clients
send ``arguments`` as a JSON *string*. Handing the template that string raises
inside Jinja, so :func:`normalise_messages` parses it first. This is the one
place where being liberal in what we accept is mandatory rather than polite:
every multi-turn agent request replays its own previous tool calls.

Jinja environment: ``trim_blocks`` and ``lstrip_blocks`` on, sandboxed, with
``tojson`` bound to plain ``json.dumps`` rather than Jinja's HTML-escaping
default. That last detail matters. Jinja's stock ``tojson`` escapes ``<``, ``>``
and ``&`` to ``\\u003c`` and friends, which would corrupt every tool schema
containing a description with an angle bracket and change the prompt bytes
against what transformers produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from titan.core.types import Message, Role, ToolCall, ToolSpec

__all__ = [
    "TemplateError",
    "REASONING_EFFORTS",
    "THINK_OPEN",
    "THINK_CLOSE",
    "normalise_reasoning_effort",
    "normalise_messages",
    "tools_to_template",
    "compile_template",
    "render_chat",
    "QwenChatTemplate",
]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

REASONING_EFFORTS = ("low", "medium", "xhigh")
"""The only values this template accepts. Anything else raises inside Jinja."""

# Client-facing effort names mapped onto the three the template knows. OpenAI
# clients send "high" and "minimal"; oMLX absorbs them with a retry-on-raise
# ladder (omlx/reasoning_effort.py, Apache-2.0) whose fixed point for a
# low/medium/xhigh template is the table below. Resolving in one step instead of
# rendering twice keeps a bad value a 400 rather than a wasted render.
_EFFORT_ALIASES: dict[str, str] = {
    "off": "low",
    "none": "low",
    "minimal": "low",
    "low": "low",
    "moderate": "medium",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "maximum": "xhigh",
    "ultra": "xhigh",
}


class TemplateError(ValueError):
    """The template refused the conversation, or the conversation is malformed.

    Carries the template's own ``raise_exception`` message where there is one,
    because those messages ("System message must be at the beginning.") name the
    client's mistake better than anything this layer could invent.
    """


def normalise_reasoning_effort(value: Any) -> str:
    """Map a client's effort string onto ``low``/``medium``/``xhigh``."""
    if value is None:
        return "medium"
    if not isinstance(value, str):
        raise TemplateError(f"reasoning_effort must be a string, got {type(value).__name__}")
    resolved = _EFFORT_ALIASES.get(value.strip().lower())
    if resolved is None:
        raise TemplateError(
            f"unsupported reasoning_effort {value!r}; "
            f"supported values are {', '.join(REASONING_EFFORTS)}"
        )
    return resolved


def _tojson(value: Any, indent: int | None = None, sort_keys: bool = False) -> str:
    """``tojson`` as transformers defines it: json.dumps, no HTML escaping."""
    return json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=sort_keys)


def _raise_exception(message: str) -> None:
    raise TemplateError(message)


def compile_template(source: str) -> jinja2.Template:
    """Compile a chat template the way transformers does.

    Matching transformers exactly is the point: the same template text has to
    produce the same prompt bytes here as it does in the reference stack, or
    every parity measurement against oMLX is measuring the wrong thing.
    """
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
        undefined=jinja2.Undefined,
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(source)


def _content_to_template(content: Any) -> Any:
    """Pass content through, mapping ``None`` to ``''``.

    The template handles strings and multipart lists itself (its
    ``render_content`` macro), so anything structured is forwarded untouched.
    """
    return "" if content is None else content


def _arguments_to_mapping(arguments: Any, *, name: str) -> dict[str, Any]:
    """Coerce a tool call's ``arguments`` into the mapping the template wants.

    OpenAI sends a JSON string. A non-object (a bare list, a number, a string
    that does not parse) becomes ``{}`` rather than an error: the client is
    replaying history and cannot fix it now, and a crashed render loses the
    whole conversation to save one malformed argument set. The behaviour mirrors
    oMLX's ``_serialize_tool_call_arguments`` in reverse.
    """
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _tool_call_to_template(call: Any) -> dict[str, Any]:
    """Normalise one tool call, from either a wire dict or a core ``ToolCall``."""
    if isinstance(call, ToolCall):
        name, arguments = call.name, call.arguments
    else:
        function = call.get("function") if isinstance(call, Mapping) else None
        source = function if isinstance(function, Mapping) else call
        name = source.get("name", "") if isinstance(source, Mapping) else ""
        arguments = source.get("arguments") if isinstance(source, Mapping) else None
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": _arguments_to_mapping(arguments, name=name),
        },
    }


def normalise_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Turn wire dicts or core ``Message`` objects into template-ready dicts.

    Only keys the template actually reads survive, which keeps the rendered
    prompt independent of whatever extra fields a client decides to send. That
    is a cache-hit-rate property, not tidiness: an echoed-back field that varies
    per turn would change the prompt prefix on every request.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Message):
            role = message.role.value if isinstance(message.role, Role) else str(message.role)
            item: dict[str, Any] = {"role": role, "content": message.content or ""}
            if message.reasoning_content:
                item["reasoning_content"] = message.reasoning_content
            if message.tool_calls:
                item["tool_calls"] = [_tool_call_to_template(c) for c in message.tool_calls]
        elif isinstance(message, Mapping):
            role = message.get("role")
            if not isinstance(role, str):
                raise TemplateError("every message needs a string role")
            item = {"role": role, "content": _content_to_template(message.get("content"))}
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                item["reasoning_content"] = reasoning
            calls = message.get("tool_calls")
            if calls:
                item["tool_calls"] = [_tool_call_to_template(c) for c in calls]
        else:
            raise TemplateError(f"unsupported message type {type(message).__name__}")
        out.append(item)
    return out


def tools_to_template(tools: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Normalise tools to the ``{"type","function"}`` shape the template dumps.

    The template writes each entry with ``tool | tojson``, so this shape is
    literally what the model reads. Key order is preserved as given, and a
    missing description becomes ``""`` rather than being omitted: an absent key
    and an empty one render differently, and the difference would move the
    prompt prefix for no reason.
    """
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, ToolSpec):
            function: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": dict(tool.parameters or {}),
            }
        elif isinstance(tool, Mapping):
            source = tool.get("function")
            if not isinstance(source, Mapping):
                raise TemplateError("each tool needs a 'function' object")
            function = {
                "name": source.get("name", ""),
                "description": source.get("description") or "",
                "parameters": source.get("parameters") or {},
            }
        else:
            raise TemplateError(f"unsupported tool type {type(tool).__name__}")
        out.append({"type": "function", "function": function})
    return out


def render_chat(
    template: jinja2.Template,
    messages: Iterable[Any],
    tools: Iterable[Any] | None = None,
    *,
    enable_thinking: bool = True,
    reasoning_effort: str | None = "medium",
    add_generation_prompt: bool = True,
    **extra: Any,
) -> str:
    """Render one conversation. The low-level entry point the tests drive.

    ``extra`` is passed through to the template as-is, which is how
    ``chat_template_kwargs`` from a request reaches template variables this code
    does not know about (``preserve_thinking``, ``add_vision_id``).
    """
    rendered_tools = tools_to_template(tools)
    context: dict[str, Any] = {
        "messages": normalise_messages(messages),
        "tools": rendered_tools or None,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        **extra,
    }
    if enable_thinking:
        context["reasoning_effort"] = normalise_reasoning_effort(reasoning_effort)
    try:
        return template.render(**context)
    except TemplateError:
        raise
    except jinja2.TemplateError as exc:
        raise TemplateError(str(exc)) from exc


@dataclass(frozen=True)
class QwenChatTemplate:
    """``TemplateRenderer`` for Qwen3.8-Flash-Next. Compiled once, at startup.

    Deterministic by construction: the only inputs are the messages, the tools
    and the keyword arguments, and nothing here consults a clock or a random
    source. That is what the port's stability invariant demands, and it is worth
    saying out loud because ``strftime_now`` is a standard template global in
    other model families and is deliberately not bound here.
    """

    template: jinja2.Template
    source: str

    @classmethod
    def from_path(cls, path: str | Path) -> "QwenChatTemplate":
        """Load ``chat_template.jinja``, or a model directory containing one."""
        p = Path(path)
        if p.is_dir():
            p = p / "chat_template.jinja"
        source = p.read_text(encoding="utf-8")
        return cls(template=compile_template(source), source=source)

    @classmethod
    def from_source(cls, source: str) -> "QwenChatTemplate":
        return cls(template=compile_template(source), source=source)

    def render(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        reasoning_effort: str = "medium",
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        return render_chat(
            self.template,
            messages,
            tools,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            **dict(template_kwargs or {}),
        )

    def reasoning_markers(self) -> tuple[str, str]:
        return (THINK_OPEN, THINK_CLOSE)
