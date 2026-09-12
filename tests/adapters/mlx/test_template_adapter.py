"""The ``TemplateRenderer`` adapter: defaults, overrides and byte stability.

The rendering itself is tested in ``tests/adapters/chat/test_template.py``
against the real template. What is left for this file is the thin part: that the
deployment defaults reach the template, that a per-request override beats them,
that ``template_kwargs`` merge rather than replace, and that the same
conversation renders to the same bytes every time. That last one is the port's
whole invariant, and it is a cache-hit-rate property rather than a cosmetic one.

A two-line probe template stands in wherever the assertion is about which
variables arrive, because reading a variable back out of the output is a
stronger check than matching a paragraph of the model's prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from titan.adapters.chat.template import QwenChatTemplate, TemplateError
from titan.adapters.mlx.template import MlxTemplateRenderer, load_renderer
from titan.core.types import Message, Role, ToolSpec

MODEL_DIR = Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
TEMPLATE_FILE = MODEL_DIR / "chat_template.jinja"

PROBE = (
    "thinking={{ enable_thinking }} "
    "effort={{ reasoning_effort | default('-') }} "
    "gen={{ add_generation_prompt }} "
    "preserve={{ preserve_thinking | default('-') }} "
    "vision={{ add_vision_id | default('-') }} "
    "n={{ messages | length }} "
    "tools={{ (tools or []) | length }}"
)

USER = Message(role=Role.USER, content="hello")


def probe(**kwargs) -> MlxTemplateRenderer:
    return MlxTemplateRenderer(template=QwenChatTemplate.from_source(PROBE), **kwargs)


# ---------------------------------------------------------------------------
# defaults and overrides
# ---------------------------------------------------------------------------


def test_defaults_reach_the_template():
    out = probe(enable_thinking=True, reasoning_effort="low").render([USER])
    assert "thinking=True" in out
    assert "effort=low" in out
    assert "gen=True" in out


def test_a_per_call_effort_beats_the_default():
    renderer = probe(reasoning_effort="low")
    assert "effort=xhigh" in renderer.render([USER], reasoning_effort="xhigh")
    # and the renderer is unchanged by it
    assert "effort=low" in renderer.render([USER])


def test_a_per_call_thinking_flag_beats_the_default():
    renderer = probe(enable_thinking=True)
    out = renderer.render([USER], enable_thinking=False)
    assert "thinking=False" in out
    # with thinking off the template is not given an effort at all
    assert "effort=-" in out


def test_add_generation_prompt_is_forwarded():
    assert "gen=False" in probe().render([USER], add_generation_prompt=False)


def test_client_effort_aliases_are_resolved():
    # "high" is what OpenAI clients send; this template only knows xhigh
    assert "effort=xhigh" in probe().render([USER], reasoning_effort="high")
    assert probe(reasoning_effort="minimal").reasoning_effort == "low"


def test_an_unusable_effort_fails_at_construction():
    with pytest.raises(TemplateError):
        probe(reasoning_effort="turbo")


def test_an_unusable_per_call_effort_fails_at_render():
    with pytest.raises(TemplateError):
        probe().render([USER], reasoning_effort="turbo")


# ---------------------------------------------------------------------------
# template kwargs
# ---------------------------------------------------------------------------


def test_configured_template_kwargs_reach_the_template():
    renderer = probe(template_kwargs={"preserve_thinking": False})
    assert "preserve=False" in renderer.render([USER])


def test_per_call_kwargs_merge_over_the_configured_ones():
    renderer = probe(template_kwargs={"preserve_thinking": False})
    out = renderer.render([USER], template_kwargs={"add_vision_id": True})
    assert "preserve=False" in out
    assert "vision=True" in out


def test_per_call_kwargs_can_override_a_configured_one():
    renderer = probe(template_kwargs={"preserve_thinking": False})
    assert "preserve=True" in renderer.render(
        [USER], template_kwargs={"preserve_thinking": True}
    )


def test_a_render_does_not_mutate_the_configured_kwargs():
    configured = {"preserve_thinking": False}
    renderer = probe(template_kwargs=configured)
    renderer.render([USER], template_kwargs={"add_vision_id": True})
    assert configured == {"preserve_thinking": False}


# ---------------------------------------------------------------------------
# construction from config
# ---------------------------------------------------------------------------


class _Profile:
    def __init__(self, thinking: bool, effort: str, kwargs: dict | None = None):
        self.enable_thinking = thinking
        self.reasoning_effort = effort
        self.template_kwargs = kwargs or {}


class _ModelConfig:
    def __init__(self, path: Path, profiles: dict[str | None, _Profile]):
        self.path = path
        self._profiles = profiles

    @property
    def base_alias(self) -> _Profile:
        return self._profiles[None]

    def resolve(self, name: str) -> _Profile | None:
        return self._profiles.get(name)


@pytest.fixture
def template_dir(tmp_path) -> Path:
    (tmp_path / "chat_template.jinja").write_text(PROBE)
    return tmp_path


def test_from_config_uses_the_canonical_profile(template_dir):
    config = _ModelConfig(
        template_dir,
        {
            None: _Profile(True, "xhigh"),
            "m:no-think": _Profile(False, "medium", {"preserve_thinking": False}),
        },
    )
    assert "effort=xhigh" in MlxTemplateRenderer.from_config(config).render([USER])


def test_from_config_uses_an_alias_profile(template_dir):
    config = _ModelConfig(
        template_dir,
        {
            None: _Profile(True, "xhigh"),
            "m:no-think": _Profile(False, "medium", {"preserve_thinking": False}),
        },
    )
    out = MlxTemplateRenderer.from_config(config, "m:no-think").render([USER])
    assert "thinking=False" in out
    assert "preserve=False" in out


def test_from_config_rejects_an_unknown_alias(template_dir):
    config = _ModelConfig(template_dir, {None: _Profile(True, "medium")})
    with pytest.raises(TemplateError):
        MlxTemplateRenderer.from_config(config, "nope")


def test_from_config_accepts_the_real_settings_object(template_dir):
    """The duck-typed constructor has to fit the pydantic model it is for."""
    from titan.config.settings import ModelConfig

    config = ModelConfig(name="m", path=template_dir, reasoning_effort="low")
    assert "effort=low" in MlxTemplateRenderer.from_config(config).render([USER])


def test_load_renderer_reads_the_directory(template_dir):
    assert "effort=medium" in load_renderer(template_dir).render([USER])


# ---------------------------------------------------------------------------
# the port's own contract
# ---------------------------------------------------------------------------


def test_reasoning_markers():
    assert probe().reasoning_markers() == ("<think>", "</think>")


def test_satisfies_the_port():
    from titan.core.ports import TemplateRenderer

    assert isinstance(probe(), TemplateRenderer)


def test_source_is_exposed():
    assert probe().source == PROBE


# ---------------------------------------------------------------------------
# the real template
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(
    not TEMPLATE_FILE.exists(), reason="model chat_template.jinja not present"
)


@pytest.fixture
def real_renderer() -> MlxTemplateRenderer:
    return load_renderer(MODEL_DIR, reasoning_effort="medium")


@real
def test_real_render_is_byte_stable(real_renderer):
    messages = [
        Message(role=Role.SYSTEM, content="You are a coding agent."),
        Message(role=Role.USER, content="list the files"),
    ]
    tools = [
        ToolSpec(
            name="bash",
            description="run a shell command <safely>",
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )
    ]
    first = real_renderer.render(messages, tools)
    for _ in range(3):
        assert real_renderer.render(messages, tools) == first


@real
def test_real_render_prefix_is_stable_as_a_turn_is_added(real_renderer):
    """Adding a turn must extend the prompt, not rewrite its head.

    This is the property the prefix cache lives on. If the template put anything
    per-turn near the front, a second turn would miss the cache entirely.
    """
    base = [
        Message(role=Role.SYSTEM, content="You are a coding agent."),
        Message(role=Role.USER, content="first question"),
    ]
    first = real_renderer.render(base, ())
    extended = real_renderer.render(
        [
            *base,
            Message(role=Role.ASSISTANT, content="first answer"),
            Message(role=Role.USER, content="second question"),
        ],
        (),
    )
    head = first[: first.index("<|im_start|>assistant")]
    assert extended.startswith(head)


@real
def test_real_no_think_closes_the_thinking_block(real_renderer):
    out = real_renderer.render([USER], (), enable_thinking=False)
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@real
def test_real_thinking_opens_the_block(real_renderer):
    assert real_renderer.render([USER], ()).endswith("<|im_start|>assistant\n<think>\n")


@real
@pytest.mark.parametrize(
    "effort,marker",
    [("low", "Reasoning effort is set to low."),
     ("xhigh", "Reasoning effort is set to xhigh."),
     ("medium", None)],
)
def test_real_effort_text(real_renderer, effort, marker):
    out = real_renderer.render([USER], (), reasoning_effort=effort)
    if marker is None:
        assert "Reasoning effort is set to" not in out
    else:
        assert marker in out


@real
def test_real_tool_schema_is_not_html_escaped(real_renderer):
    tools = [
        ToolSpec(
            name="edit",
            description="replace <old> with <new>",
            parameters={"type": "object"},
        )
    ]
    out = real_renderer.render([USER], tools)
    assert "replace <old> with <new>" in out
    assert "\\u003c" not in out
