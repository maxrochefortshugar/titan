"""Chat adapters: the model's prompt format and its output markup.

Two ports live here, and they are inverses of each other:

- :class:`titan.adapters.chat.template.QwenChatTemplate` implements
  ``TemplateRenderer``: messages and tools in, prompt string out.
- :class:`titan.adapters.chat.tool_parser.Qwen3CoderToolParser` implements
  ``ToolCallParser``: the model's generated text in, content, reasoning and
  structured tool calls out.

Neither imports mlx, fastapi or the engine.
"""
