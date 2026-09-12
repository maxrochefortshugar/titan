"""Configuration schema. One file, validated once, no environment variables.

The overlay was driven by about twenty env vars, which meant the running
configuration existed only in a launchd plist and could not be diffed against a
benchmark result. Titan takes one TOML file, validates it at startup, refuses to
start on anything invalid, and echoes the resolved values into the log and into
``GET /metrics`` so every measurement can name the config it came from.

Two escape hatches, both explicit: ``--set path.key=value`` on the command line
for a one-off experiment, and ``kernels.disabled`` for bisecting a bad kernel.
Both are recorded in the resolved config, so a run started with an override is
never mistaken for a stock run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "SpeculationConfig",
    "KernelConfig",
    "ServerConfig",
    "ObservabilityConfig",
    "TitanConfig",
]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    path: str
    max_context: int = 262144
    ngram_table_path: str = ""
    """Packed contiguous n-gram table built by the repack tool. Streamed from
    SSD. A resident copy is not an option on a 128 GB machine: 32 GB on top of
    the model panicked the kernel."""
    ngram_reader_workers: int = 16
    """Parallel preads. The SSD does 13.6 GB/s and about 84% of the cost of a
    KV-style read is filesystem overhead, so worker count matters more than
    bandwidth."""


@dataclass(frozen=True, slots=True)
class CacheConfig:
    block_tokens: int = 512
    """KV persistence and reconstruction unit. Decoupled from the snapshot
    grid. Reconstruction is bytes-bound at about 2.77e-3 ms per token, so 48
    blocks of 512 cost what 12 of 2048 cost: block count is free, resolution is
    not."""
    snapshot_grid: int = 2048
    """Coarse grid where recurrent snapshots are always staged. Must be a
    multiple of ``block_tokens``, and every grid multiple inside a prefill
    suffix must end a chunk."""
    snapshot_at_prompt_end: bool = True
    """Always snapshot at the natural end of a prompt. The last chunk already
    ends there, so the extra cost is one write, about 210 ms of writer time, and
    no forward pass. This is what makes a follow-up turn resume at the exact
    prompt end instead of the previous grid multiple."""
    ram_tier_gb: float = 4.0
    """Measured to give the same hit rate as 16 GB with far less pressure."""
    ssd_dir: str = ""
    ssd_capacity_gb: float = 200.0
    max_stall_ms: float = 50.0
    """Hard cap on how long a store may block the scheduler thread."""
    pending_write_budget_mb: float = 512.0


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_concurrent: int = 8
    queue_depth: int = 64
    prefill_chunk_tokens: int = 2048
    """2048 is measured right on this GPU: 512 is 22% worse per token and 4096
    is 20% worse."""
    serialise_prefill: bool = True
    """Only one sequence prefills at a time, and never during a decode cycle.
    Batched prefill loses the gathered sparse-attention arm and materialises a
    134 MB mask per QSA layer at 65k."""
    memory_guard_gb: float = 110.0
    """Admission gate only, never a throttle on running work."""
    decode_rows_budget: int = 32
    """Total verify rows per cycle across all sequences."""


@dataclass(frozen=True, slots=True)
class SpeculationConfig:
    enabled: bool = True
    """Off costs 1.5x: 61.5 tok/s becomes 41."""
    max_depth: int = 3
    """Depth 4 measured 6% worse under a fixed policy."""
    min_depth: int = 1
    adaptive: bool = True
    acceptance_window: int = 64
    ngram_copy_lane: bool = False
    """Prompt-sliced copy blocks. Off until it beats the chain on this machine;
    two rules if it is enabled: slice from the prompt only, never from generated
    text, and never accept a copy block past a stop token."""
    copy_max_block: int = 14


@dataclass(frozen=True, slots=True)
class KernelConfig:
    enabled: tuple[str, ...] = ()
    """Empty means every registered op uses its fast path where supported."""
    disabled: tuple[str, ...] = ()
    """Named ops forced to their reference implementation. For bisecting."""
    fail_open: bool = True
    """A fast kernel that raises falls back and counts, rather than killing the
    sequence. Safe because the exactness tests make output independent of which
    implementation ran."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8085
    """Not 8083 or 8084: production and the workbench own those."""
    api_key_file: str = ""
    request_timeout_s: float = 600.0
    max_body_mb: float = 64.0


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    cycle_profile: bool = True
    """Always on. The per-cycle profile is a product feature."""
    cycle_profile_ring: int = 4096
    log_path: str = ""
    metrics_enabled: bool = True
    trace_sample_every: int = 0
    """0 disables the detailed sampled trace, which does sync the device."""


@dataclass(frozen=True, slots=True)
class TitanConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculation: SpeculationConfig = field(default_factory=SpeculationConfig)
    kernels: KernelConfig = field(default_factory=KernelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def validate(self) -> None:
        """Raise :class:`~titan.core.errors.ConfigError` on anything invalid.

        Checks that are not optional: the snapshot grid is a multiple of the
        block size; the prefill chunk is a multiple of the block size; the
        memory guard leaves headroom above the resident weight size; the port is
        not 8083 or 8084; ``max_depth >= min_depth``; the n-gram table exists and
        is in packed layout; every name in ``kernels.disabled`` is registered.
        """
