"""A minimal valid config, and the fakes the wiring tests build against."""

from __future__ import annotations

from typing import Any

import pytest

MINIMAL = """
[model]
path = "/models/qwen"
name = "Qwen3.8-Flash-Next-oQ4e-mtp"
"""

FULL = """
[model]
path = "/models/qwen"
name = "Qwen3.8-Flash-Next-oQ4e-mtp"
max_context = 262144
weights_gb = 62.5
ngram_table_path = "/models/ngram.packed"

[model.dtype]
compute = "bf16"
kv = "bf16"

[model.sampling]
temperature = 0.7
top_p = 0.8
top_k = 20

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think"]
enable_thinking = false
reasoning_effort = "low"

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think".sampling]
temperature = 0.0
top_p = 1.0
top_k = 0

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think".template_kwargs]
tool_choice = "auto"

[server]
host = "127.0.0.1"
port = 8085
api_key_file = "/etc/titan/titan.key"
sse_keepalive_mode = "chunk"

[sampling]
temperature = 0.7

[scheduler]
max_seqs = 8
prefill_chunk = 2048
memory_guard_gb = 110.0
memory_guard_soft_fraction = 0.85

[speculation]
mtp_depth_max = 3
adaptive_depth = true
shortlist_draft = false

[cache]
block_tokens = 512
snapshot_grid = 2048
ram_tier_mb = 4096.0
fine_tail = true
ssd_dir = "/var/titan/cache"

[kernels]
disabled = ["topk_radix"]

[limits]
max_tokens = 32768
max_context = 262144

[observability]
cycle_profile = true
log_path = "/var/log/titan.jsonl"
"""


@pytest.fixture()
def write_config(tmp_path):
    """Write a config file and return its path."""

    def _write(text: str = MINIMAL, name: str = "titan.toml"):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    return _write


# ---------------------------------------------------------------------------
# fakes for the wiring, one per port the composition root builds
# ---------------------------------------------------------------------------


class FakeBackend:
    """Enough of a backend for the wiring: it can be warmed, and nothing else."""

    def __init__(self) -> None:
        self.warmed = 0

    def warmup(self) -> None:
        self.warmed += 1


class FakeCodec:
    """Stands in for the state codec the MLX adapter still owes."""

    signature = None


class FakeEngine:
    def __init__(self) -> None:
        self.started = 0
        self.stopped: list[float] = []

    def start(self) -> None:
        self.started += 1

    def stop(self, drain_timeout_s: float) -> None:
        self.stopped.append(drain_timeout_s)


class FakeStore:
    def __init__(self) -> None:
        self.flushed: list[float] = []

    def flush(self, timeout_s: float) -> bool:
        self.flushed.append(timeout_s)
        return True


class FakeTokenizer:
    eos_token_ids = frozenset({151645})
    vocab_size = 151936

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return [1] * max(1, len(text) // 4)

    def decode(self, ids) -> str:
        return ""

    def decode_incremental(self, seq: Any, ids) -> str:
        return ""

    def flush_incremental(self, seq: Any) -> str:
        return ""


class FakeTemplate:
    def render(self, messages, tools=(), **kwargs) -> str:
        return "prompt"

    def reasoning_markers(self) -> tuple[str, str]:
        return ("<think>", "</think>")


@pytest.fixture()
def fake_parts():
    from titan.config.wiring import Parts

    return Parts(
        profiler=object(),
        registry=object(),
        backend=FakeBackend(),
        tokenizer=FakeTokenizer(),
        template=FakeTemplate(),
        codec=FakeCodec(),
        store=FakeStore(),
        cache=object(),
        engine=FakeEngine(),
        app=object(),
    )
