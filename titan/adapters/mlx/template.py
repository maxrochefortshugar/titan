# SPDX-License-Identifier: MIT
"""``TemplateRenderer`` port over the checkpoint's own chat template.

All of the rendering lives in :mod:`titan.adapters.chat.template`, which
compiles ``chat_template.jinja`` the way transformers does and knows the
template's three quirks (the tool block is the system message, reasoning effort
is prompt text, tool-call arguments arrive as JSON strings). This module adds
nothing to that and deliberately duplicates none of it. What it does add is the
port's shape and the model's configured defaults, so a caller that holds a
``TemplateRenderer`` can render a turn without knowing which effort level or
thinking mode this deployment runs.

Why the defaults belong here rather than at every call site: the port's
stability invariant is about prompt bytes, and the bytes move if two call sites
disagree about ``enable_thinking``. Resolving the default once, from the
configured alias, is what keeps a warm prefix warm across turns that came in
through different paths.

The renderer is frozen and holds no per-request state, so one instance serves
every request and the whole thing is thread-safe by having nothing to race on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from titan.adapters.chat.template import (
    REASONING_EFFORTS,
    THINK_CLOSE,
    THINK_OPEN,
    QwenChatTemplate,
    TemplateError,
    normalise_reasoning_effort,
)
from titan.core.types import Message, ToolSpec

__all__ = [
    "MlxTemplateRenderer",
    "load_renderer",
    "REASONING_EFFORTS",
    "TemplateError",
]


@dataclass(frozen=True)
class MlxTemplateRenderer:
    """The chat template as the core sees it.

    ``enable_thinking`` and ``reasoning_effort`` are the deployment defaults;
    both can be overridden per call, which is what an alias like
    ``<model>:no-think`` and an OpenAI ``reasoning_effort`` field need.
    ``template_kwargs`` are extra template variables (``preserve_thinking``,
    ``add_vision_id``) that a profile pins for every render.
    """

    template: QwenChatTemplate
    enable_thinking: bool = True
    reasoning_effort: str = "medium"
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail at construction, not on the first request. An unusable effort
        # string in a config file should stop startup.
        object.__setattr__(
            self, "reasoning_effort", normalise_reasoning_effort(self.reasoning_effort)
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str = "medium",
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> "MlxTemplateRenderer":
        """Compile ``chat_template.jinja`` from a checkpoint directory."""
        return cls(
            template=QwenChatTemplate.from_path(model_dir),
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            template_kwargs=dict(template_kwargs or {}),
        )

    @classmethod
    def from_config(cls, model: Any, alias: str | None = None) -> "MlxTemplateRenderer":
        """Build from a ``titan.config.settings.ModelConfig``.

        ``alias`` picks one of the model's profiles; without it the canonical
        model's own defaults are used. Duck-typed rather than imported so this
        adapter does not depend on pydantic being loaded to render a prompt in a
        test.
        """
        profile = model.resolve(alias) if alias is not None else model.base_alias
        if profile is None:
            raise TemplateError(f"unknown model alias {alias!r}")
        return cls.from_model_dir(
            Path(model.path),
            enable_thinking=bool(profile.enable_thinking),
            reasoning_effort=str(profile.reasoning_effort),
            template_kwargs=dict(getattr(profile, "template_kwargs", {}) or {}),
        )

    # -- port --------------------------------------------------------------
    def render(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        reasoning_effort: str | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """Render one turn. ``None`` on either mode argument means the default.

        Extra ``template_kwargs`` are merged over the configured ones rather
        than replacing them, so a per-request ``preserve_thinking`` can be set
        without losing whatever the profile pinned.
        """
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        extra = dict(self.template_kwargs)
        if template_kwargs:
            extra.update(template_kwargs)
        return self.template.render(
            messages,
            tools,
            reasoning_effort=effort,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
            template_kwargs=extra,
        )

    def reasoning_markers(self) -> tuple[str, str]:
        return (THINK_OPEN, THINK_CLOSE)

    @property
    def source(self) -> str:
        """The template text, for the startup log and the config echo."""
        return self.template.source


def load_renderer(
    model_dir: str | Path,
    *,
    enable_thinking: bool = True,
    reasoning_effort: str = "medium",
    template_kwargs: Mapping[str, Any] | None = None,
) -> MlxTemplateRenderer:
    """Module-level entry point, matching ``load_tokenizer``'s shape."""
    return MlxTemplateRenderer.from_model_dir(
        model_dir,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        template_kwargs=template_kwargs,
    )
