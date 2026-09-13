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

This module is the single source of truth for what a Titan configuration is.
The dataclasses here are frozen, they hold every section, and they are the type
the engine and the wiring consume. ``titan.config.settings`` is the pydantic
front end the HTTP layer reads; it mirrors these sections field for field and
``tests/config/test_parity.py`` fails the build if the two ever drift.

Nothing here imports anything but the standard library, and nothing here reads
the environment. Parsing is :meth:`TitanConfig.from_mapping`, which takes the
already-decoded TOML table and raises :class:`~titan.core.errors.ConfigError`
naming the full dotted path of the first key that is wrong.
"""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

from titan.core.errors import ConfigError

__all__ = [
    "AliasConfig",
    "CacheConfig",
    "DtypePolicy",
    "KernelConfig",
    "LimitsConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "SamplingConfig",
    "SchedulerConfig",
    "ServerConfig",
    "SpeculationConfig",
    "TitanConfig",
    "REASONING_EFFORTS",
    "KEEPALIVE_MODES",
    "RESERVED_PORTS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_REASONING_EFFORT",
    "apply_overrides",
    "parse_override",
    "dumps_toml",
]


# ---------------------------------------------------------------------------
# constants the pydantic front end also uses, so there is one place to change
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_REASONING_EFFORT = "medium"
"""Production sampling defaults for Qwen3.8-Flash-Next, from the model card's
thinking-mode row. Temperature 1.0 in particular produces the repetition and
tool-argument corruption that made the first week of agent runs unusable, so
these are a decision rather than a preference."""

REASONING_EFFORTS = ("low", "medium", "xhigh")
KEEPALIVE_MODES = ("chunk", "comment", "off")
DTYPES = ("auto", "bf16", "fp16", "fp32")
RESERVED_PORTS = (8083, 8084)
"""Production and the workbench own these. Binding one of them from a Titan
config is a misconfiguration that would take down the thing being compared
against, so it is refused at parse time rather than at bind time."""

DEFAULT_PORT = 8085


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Sampling defaults. A request may override any of these field by field."""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int | None = None
    """Default completion budget when the request does not set one."""


@dataclass(frozen=True, slots=True)
class AliasConfig:
    """A profile: one model, one set of template kwargs, one set of defaults.

    Aliases are how a client picks a behaviour without knowing anything about
    the template. ``<model>:no-think`` is the canonical example: same weights,
    ``enable_thinking=false``, and greedy-ish sampling because a non-thinking
    turn that wanders is just a slow wrong answer.
    """

    name: str = ""
    enable_thinking: bool = True
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments passed straight to the chat template."""


RESERVED_TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "add_generation_prompt")


@dataclass(frozen=True, slots=True)
class DtypePolicy:
    """What runs in which precision. One place, so a bench run can name it.

    ``compute`` is the activation dtype of the forward pass, ``kv`` is what the
    attention cache stores, and ``accumulate`` is what reductions widen to.
    ``quantization`` is normally ``checkpoint``, meaning the per-module bits and
    group size recorded in the checkpoint win; the loader never guesses.
    """

    compute: str = "bf16"
    kv: str = "bf16"
    accumulate: str = "fp32"
    quantization: str = "checkpoint"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The one model this process serves, plus its aliases."""

    path: str
    """Directory holding chat_template.jinja, tokenizer.json and the weights."""
    name: str = ""
    """Canonical id, echoed in every response and listed by /v1/models. Empty
    means the directory name, resolved by :meth:`resolved_name`."""
    max_context: int = 262144
    aliases: tuple[AliasConfig, ...] = ()
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    enable_thinking: bool = True
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    dtype: DtypePolicy = field(default_factory=DtypePolicy)
    weights_gb: float = 0.0
    """Resident size of the weights. 0 means unknown, which switches off the
    memory-guard headroom check rather than guessing at it."""
    fuse_gate_up: bool = True
    ngram_table_path: str = ""
    """Packed contiguous n-gram table built by the repack tool. Streamed from
    SSD. A resident copy is not an option on a 128 GB machine: 32 GB on top of
    the model panicked the kernel."""
    ngram_reader_workers: int = 16
    """Parallel preads. The SSD does 13.6 GB/s and about 84% of the cost of a
    KV-style read is filesystem overhead, so worker count matters more than
    bandwidth."""

    def resolved_name(self) -> str:
        if self.name:
            return self.name
        return self.path.rstrip("/").rsplit("/", 1)[-1]

    def served_names(self) -> list[str]:
        """Everything /v1/models advertises: the model, then its aliases."""
        return [self.resolved_name(), *sorted(a.name for a in self.aliases)]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Prefix cache grids and the two storage tiers.

    The block size and the snapshot grid live here rather than under
    ``[scheduler]`` because they are properties of the cache; the scheduler
    reads them from this section when it plans chunks. Keeping them decoupled
    from each other is the point: the overlay tied them together and could only
    resume on a 2048-token grid.
    """

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
    fine_tail: bool = True
    """Cut the uncached suffix on the fine grid near its end, so the last
    resumable point sits close to the prompt end instead of a grid multiple
    back. Round 4a of the cache-boundary work; off restores the coarse
    behaviour for an A/B."""
    fine_min_gain_tokens: int = 384
    """How many tokens the prompt-end cut has to buy back before it is taken.

    The gate the fine tail is actually decided by, in ``_fine_cut_allowed``.
    An extra chunk launch plus a snapshot is about 283 ms on a quiet writer,
    so a cut that saves less than this much recompute is a cut that costs
    more than it returns.

    It is a token count and not a block count on purpose. The field used to be
    ``fine_tail_blocks``, four blocks, and nothing read it: derived as a
    threshold it came out at 2048 and refused every fine cut on a prompt
    shorter than the snapshot grid, which is most of them.
    """
    ram_tier_mb: float = 4096.0
    """Hot tier size. 4 GB measured the same hit rate as 16 GB with far less
    memory pressure."""
    ssd_dir: str = ""
    ssd_capacity_gb: float = 200.0
    max_stall_ms: float = 50.0
    """Hard cap on loop-thread milliseconds one sequence's store may spend in
    a single cycle.

    Two things read it. The prefix cache spends it as a per-cycle budget: a
    boundary is only serialised when the measured cost of the last one still
    fits, and the rest waits for the next cycle. The store checks its own put
    path against it and counts a breach, which is how the report's 1.50 s
    against a 50 ms cap would be caught next time."""
    pending_write_budget_mb: float = 512.0

    @property
    def ram_tier_gb(self) -> float:
        return self.ram_tier_mb / 1024.0


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_seqs: int = 8
    """Sequences decoding at once. The loop is one thread; this is how many
    sequences share a cycle, not how many threads run."""
    queue_depth: int = 64
    prefill_chunk: int = 2048
    """2048 is measured right on this GPU: 512 is 22% worse per token and 4096
    is 20% worse."""
    serialise_prefill: bool = True
    """Only one sequence prefills at a time, and never during a decode cycle.
    Batched prefill loses the gathered sparse-attention arm and materialises a
    134 MB mask per QSA layer at 65k."""
    memory_guard_gb: float = 110.0
    """Admission gate only, never a throttle on running work."""
    memory_guard_soft_fraction: float = 0.85
    """Above this fraction of the guard, admission stops taking new work but
    running sequences are untouched. Set too low it silently serialises
    concurrent requests, which cost the overlay its whole concurrency win."""
    decode_rows_budget: int = 32
    """Total verify rows per cycle across all sequences."""
    release_after_prefill: bool = False
    """Drop MLX's buffer cache when the last prefill chunk lands.

    ROUND5 step 2's second arm. The cache is a free-list rather than live data,
    so releasing it cannot lose anything the decode needs; it trades the
    reallocation of whatever the decode would have reused against a decode that
    is not allocating underneath a 64k prefill's leftovers."""

    @property
    def memory_guard_soft_gb(self) -> float:
        return self.memory_guard_gb * self.memory_guard_soft_fraction


@dataclass(frozen=True, slots=True)
class SpeculationConfig:
    enabled: bool = True
    """Off costs 1.5x: 61.5 tok/s becomes 41."""
    mtp_depth_max: int = 3
    """Largest MTP chain the drafter may propose. Depth 4 measured 6% worse
    under a fixed policy."""
    mtp_depth_min: int = 1
    adaptive_depth: bool = True
    acceptance_window: int = 64
    depth_policy: str = "expected_value"
    """Which adaptive policy runs when ``adaptive_depth`` is on.

    ``expected_value`` is the converging policy: one decayed cost mean per
    width, per-position acceptance, hysteresis, dwell and probe runs.
    ``mean_accepted`` is the ROUND4 policy, ``round(mean_accepted) + 1`` with
    two clamps, kept so the two can be measured against each other on the same
    machine in the same hour. Ignored when ``adaptive_depth`` is false."""
    depth_probe_every: int = 48
    """Decisions between probe runs in the expected-value policy. Zero is no
    probing, which is the policy without its route out of a wrong resting
    point."""
    depth_probe_cycles: int = 8
    """Consecutive decisions one probe run spends at the neighbouring depth.
    With ``depth_probe_every`` this is the duty cycle: 8 in 48 is 17%."""
    depth_hysteresis: float = 0.06
    """How much better in expected committed tokens per millisecond a
    candidate depth has to be before it displaces the incumbent."""
    mtp_chain: str = "head_output"
    """How draft step ``i+1`` is fed. ``head_output`` re-enters the head on its
    own post-norm output, which is vLLM's form and what EAGLE 3.1 credits for
    long-context acceptance; ``omlx`` re-enters on the head layer's pre-mixer
    streams, which is what oMLX does. Draft numerics cannot change output, only
    acceptance, so this is an A/B knob and not a correctness one."""
    mtp_chain_cache: str = "clone"
    """How the draft chain keeps its tail out of the head's committed history.

    ``clone`` copies the head KV per cycle, which is free on a short head and
    is 65 MB plus a re-pooled sparse index on a head that holds a 64k prompt.
    ``trim`` appends to the real cache and rewinds it, which is oMLX's form.
    Identical drafts either way."""
    mtp_prime_prompt: bool = True
    """Fold the prompt into the MTP head's KV cache during prefill.

    The head is one sparse-attention layer out of 49, so a fold over the prompt
    costs about 2% of a prefill; what it buys is a drafter whose attention has
    seen the prompt at all. Without it the head's cache is empty at the first
    decode cycle and fills only with committed tokens, which is where the 64k
    acceptance defect lives. Draft numerics cannot change output."""
    mtp_prime_window: int = 1024
    """Prime only the last this-many tokens of the prompt. 0 primes all of it.

    Priming the whole prompt leaves the head holding a 64k KV cache that it
    re-attends over once per drafted token, which was measured at 10.3 ms a
    cycle of draft against 7.1 unprimed. A window is the middle: the head sees
    the part of the prompt a next-token draft is conditioned on and pays
    attention over that much only."""
    mtp_head_align_positions: bool = False
    """Give the MTP head the trunk's sequence positions instead of its own.

    The head's KV cache starts empty at the first decode cycle, so its RoPE
    offset is the count of decoded tokens while the trunk attends at the
    prompt's length plus that count. On this checkpoint the gap is the prompt,
    which at 64k is the whole position range the head was trained on. Draft
    numerics cannot change output, so this is an acceptance knob."""
    overlap_draft: bool = False
    """Dispatch the next cycle's draft chain at the end of this one.

    The chain's GPU work is enqueued behind this cycle's verify instead of in
    front of the next cycle's, so the host's wait for it overlaps the commit,
    the detokenisation and the next cycle's depth planning. It cannot change
    what is drafted -- the fold's inputs are the tokens the commit just
    produced -- and a dispatch whose batch has moved by the next cycle is
    dropped rather than read."""
    draft_p_min: float = 0.0
    """Stop the chain past a draft whose top-token probability is below this.
    llama.cpp measured (16, 0.8) beating (4, 0.0) by 20.4% with acceptance
    falling and mean run length rising. Zero disables the gate."""
    shortlist_draft: bool = False
    """Prompt-sliced copy blocks drafted from a shortlist of recent n-grams.
    Off until it beats the chain on this machine; two rules if it is enabled:
    slice from the prompt only, never from generated text, and never accept a
    block past a stop token."""
    shortlist_max_block: int = 14


@dataclass(frozen=True, slots=True)
class KernelConfig:
    enabled: tuple[str, ...] = ()
    """Empty means every registered op uses its fast path where supported."""
    disabled: tuple[str, ...] = ()
    """Named ops forced to their reference implementation. For bisecting."""
    reference_only: bool = False
    """Every fast path off, fail-open off. The M1 parity baseline and the
    control arm of every kernel A/B."""
    fail_open: bool = True
    """A fast kernel that raises falls back and counts, rather than killing the
    sequence. Safe because the exactness tests make output independent of which
    implementation ran."""
    prefill_only: tuple[str, ...] = ()
    """Ops whose fast path is taken during prefill and not during decode.

    ROUND4 measured ``moe_gather_int8`` at -3.7% on a 64k decode, which is a
    reason to keep it off a decode and not in itself a reason to keep it off a
    prefill: the two phases run the same op at shapes three orders of
    magnitude apart. Naming an op here selects its fast path only while a
    prompt is going in. An op named here still has to be named in ``enabled``
    (or be on by default) to run at all."""
    forward_paths_on: tuple[str, ...] = ()
    """Vendored forward arms switched on over their defaults, by name.

    A *path* is not an op: it is an arm of the vendored forward that changes
    how much host work and how many launches a step costs without changing
    what the step computes (``models/forward_paths.py``). The names are that
    module's. Two tuples rather than one mapping so that a ``--set`` override
    on the command line looks exactly like the kernel lists next to it."""
    forward_paths_off: tuple[str, ...] = ()
    """Vendored forward arms switched off over their defaults, by name."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    """Not 8083 or 8084: production and the workbench own those."""
    api_key_file: str = ""
    """File holding the bearer token. Empty disables auth entirely. The key is
    never a config value, only a path, so the secret stays out of anything that
    gets pasted into an issue."""
    request_timeout_s: float = 600.0
    max_body_mb: float = 64.0
    sse_keepalive_seconds: float = 10.0
    """Idle gap after which a keepalive frame goes out during long prefill."""
    sse_keepalive_mode: str = "chunk"
    """``chunk`` (oMLX-compatible no-op event), ``comment`` (``: ping``) or
    ``off``."""


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Hard ceilings. A request asking for more is a 400, not a silent clamp."""

    max_tokens: int = 32768
    max_context: int = 262144


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
    server: ServerConfig = field(default_factory=ServerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculation: SpeculationConfig = field(default_factory=SpeculationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    kernels: KernelConfig = field(default_factory=KernelConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    overrides: tuple[str, ...] = ()
    """The ``--set`` arguments this config was built with, recorded so a run
    started with an override is never mistaken for a stock run."""
    source: str = ""
    """Where the file came from. Echoed into the log and ``GET /metrics``."""

    # -- parsing -----------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TitanConfig":
        """Build from a decoded TOML or JSON table. Does not validate."""
        return _build(cls, data, "")

    @classmethod
    def from_toml(
        cls,
        text: str,
        *,
        overrides: Sequence[str] = (),
        source: str = "",
    ) -> "TitanConfig":
        """Parse ``text``, apply ``--set`` overrides, build, validate."""
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{source or 'config'} is not valid TOML: {exc}") from None
        data = apply_overrides(data, overrides)
        config = _build(cls, data, "")
        config = dataclasses.replace(
            config,
            overrides=config.overrides + tuple(overrides),
            source=source or config.source,
        )
        config.validate()
        return config

    # -- dumping -----------------------------------------------------------
    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        """Nested plain-data view, suitable for TOML, JSON or ``/metrics``.

        ``redact`` masks the API key file path. The key itself is never held in
        memory here, but the path is the one field an operator would not want in
        a pasted bug report.
        """
        out = _to_plain(self)
        if redact and out["server"].get("api_key_file"):
            out["server"]["api_key_file"] = "***redacted***"
        return out

    def dumps(self, *, redact: bool = False) -> str:
        """The resolved config as TOML. Round trips through :meth:`from_toml`."""
        return dumps_toml(self.to_dict(redact=redact))

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`~titan.core.errors.ConfigError` on anything invalid.

        Every message names the dotted path of the offending key, because the
        operator reading it is looking at a file and needs to know which line to
        change. The first failure stops the walk: a config with two problems is
        fixed one at a time anyway.
        """
        m, s, c, sc, sp, k, lim, obs = (
            self.model,
            self.server,
            self.cache,
            self.scheduler,
            self.speculation,
            self.kernels,
            self.limits,
            self.observability,
        )

        # model
        if not m.path:
            raise ConfigError("model.path is required and may not be empty")
        _positive("model.max_context", m.max_context)
        _positive("model.ngram_reader_workers", m.ngram_reader_workers)
        _at_least("model.weights_gb", m.weights_gb, 0.0)
        _one_of("model.reasoning_effort", m.reasoning_effort, REASONING_EFFORTS)
        _one_of("model.dtype.compute", m.dtype.compute, DTYPES)
        _one_of("model.dtype.kv", m.dtype.kv, DTYPES)
        _one_of("model.dtype.accumulate", m.dtype.accumulate, DTYPES)
        if m.dtype.quantization not in ("checkpoint", "none"):
            raise ConfigError(
                "model.dtype.quantization must be checkpoint or none, got "
                f"{m.dtype.quantization!r}"
            )
        _sampling("model.sampling", m.sampling)
        _sampling("sampling", self.sampling)

        seen: set[str] = set()
        for i, alias in enumerate(m.aliases):
            path = f"model.aliases.{alias.name or i}"
            if not alias.name:
                raise ConfigError(f"{path}: an alias must have a name")
            if alias.name == m.resolved_name():
                raise ConfigError(
                    f"{path} collides with the canonical model name "
                    f"{m.resolved_name()!r}"
                )
            if alias.name in seen:
                raise ConfigError(f"{path} is declared twice")
            seen.add(alias.name)
            _one_of(f"{path}.reasoning_effort", alias.reasoning_effort, REASONING_EFFORTS)
            _sampling(f"{path}.sampling", alias.sampling)
            for reserved in RESERVED_TEMPLATE_KWARGS:
                if reserved in alias.template_kwargs:
                    raise ConfigError(
                        f"{path}.template_kwargs may not set {reserved!r}; "
                        "it has its own field"
                    )

        # server
        if not 1 <= s.port <= 65535:
            raise ConfigError(f"server.port must be in 1..65535, got {s.port}")
        if s.port in RESERVED_PORTS:
            raise ConfigError(
                f"server.port {s.port} is reserved: production and the workbench "
                f"own {RESERVED_PORTS}; Titan serves on {DEFAULT_PORT}"
            )
        if not s.host:
            raise ConfigError("server.host may not be empty")
        _one_of("server.sse_keepalive_mode", s.sse_keepalive_mode, KEEPALIVE_MODES)
        _positive("server.sse_keepalive_seconds", s.sse_keepalive_seconds)
        _positive("server.request_timeout_s", s.request_timeout_s)
        _positive("server.max_body_mb", s.max_body_mb)

        # cache grids
        _positive("cache.block_tokens", c.block_tokens)
        _positive("cache.snapshot_grid", c.snapshot_grid)
        if c.snapshot_grid % c.block_tokens:
            raise ConfigError(
                f"cache.snapshot_grid ({c.snapshot_grid}) must be a multiple of "
                f"cache.block_tokens ({c.block_tokens}); a grid point that is "
                "not a block end can never be restored"
            )
        _at_least("cache.fine_min_gain_tokens", c.fine_min_gain_tokens, 1)
        _positive("cache.ram_tier_mb", c.ram_tier_mb)
        _positive("cache.ssd_capacity_gb", c.ssd_capacity_gb)
        _positive("cache.max_stall_ms", c.max_stall_ms)
        _positive("cache.pending_write_budget_mb", c.pending_write_budget_mb)

        # scheduler
        _positive("scheduler.max_seqs", sc.max_seqs)
        _positive("scheduler.queue_depth", sc.queue_depth)
        if sc.queue_depth < sc.max_seqs:
            raise ConfigError(
                f"scheduler.queue_depth ({sc.queue_depth}) is below "
                f"scheduler.max_seqs ({sc.max_seqs}); the queue could never fill "
                "the loop"
            )
        _positive("scheduler.prefill_chunk", sc.prefill_chunk)
        if sc.prefill_chunk % c.block_tokens:
            raise ConfigError(
                f"scheduler.prefill_chunk ({sc.prefill_chunk}) must be a multiple "
                f"of cache.block_tokens ({c.block_tokens}); a chunk that ends "
                "mid-block stages no snapshot there"
            )
        _positive("scheduler.decode_rows_budget", sc.decode_rows_budget)
        _positive("scheduler.memory_guard_gb", sc.memory_guard_gb)
        if not 0.0 < sc.memory_guard_soft_fraction <= 1.0:
            raise ConfigError(
                "scheduler.memory_guard_soft_fraction must be in (0, 1], got "
                f"{sc.memory_guard_soft_fraction}"
            )
        if m.weights_gb and sc.memory_guard_gb <= m.weights_gb:
            raise ConfigError(
                f"scheduler.memory_guard_gb ({sc.memory_guard_gb}) leaves no "
                f"headroom over model.weights_gb ({m.weights_gb}); the guard "
                "would refuse the first request"
            )

        # speculation
        _at_least("speculation.mtp_depth_min", sp.mtp_depth_min, 1)
        if sp.mtp_depth_max < sp.mtp_depth_min:
            raise ConfigError(
                f"speculation.mtp_depth_max ({sp.mtp_depth_max}) is below "
                f"speculation.mtp_depth_min ({sp.mtp_depth_min})"
            )
        _positive("speculation.acceptance_window", sp.acceptance_window)
        _at_least("speculation.mtp_prime_window", sp.mtp_prime_window, 0)
        _at_least("speculation.shortlist_max_block", sp.shortlist_max_block, 1)
        if sp.mtp_chain_cache not in ("clone", "trim"):
            raise ConfigError(
                f"speculation.mtp_chain_cache must be 'clone' or 'trim', got "
                f"{sp.mtp_chain_cache!r}"
            )
        if set(k.prefill_only) & set(k.disabled):
            both = sorted(set(k.prefill_only) & set(k.disabled))
            raise ConfigError(
                "kernels.prefill_only and kernels.disabled name the same op: "
                + ", ".join(both)
            )
        if set(k.forward_paths_on) & set(k.forward_paths_off):
            both = sorted(set(k.forward_paths_on) & set(k.forward_paths_off))
            raise ConfigError(
                "kernels.forward_paths_on and kernels.forward_paths_off name "
                f"the same path: {', '.join(both)}"
            )
        if sp.depth_policy not in ("expected_value", "mean_accepted"):
            raise ConfigError(
                "speculation.depth_policy must be 'expected_value' or "
                f"'mean_accepted', got {sp.depth_policy!r}"
            )
        _at_least("speculation.depth_probe_every", sp.depth_probe_every, 0)
        _at_least("speculation.depth_probe_cycles", sp.depth_probe_cycles, 0)
        if sp.depth_hysteresis < 0.0:
            raise ConfigError(
                "speculation.depth_hysteresis must not be negative, got "
                f"{sp.depth_hysteresis}"
            )
        if sp.mtp_chain not in ("head_output", "omlx"):
            raise ConfigError(
                f"speculation.mtp_chain must be 'head_output' or 'omlx', got "
                f"{sp.mtp_chain!r}"
            )
        if not 0.0 <= sp.draft_p_min < 1.0:
            raise ConfigError(
                f"speculation.draft_p_min must be in [0, 1), got {sp.draft_p_min}"
            )

        # kernels
        both = sorted(set(k.enabled) & set(k.disabled))
        if both:
            raise ConfigError(
                f"kernels.enabled and kernels.disabled both name {both[0]!r}"
            )
        if k.reference_only and k.enabled:
            raise ConfigError(
                "kernels.reference_only is set, so kernels.enabled can only be "
                "empty; naming an op there asks for a fast path that is off"
            )

        # limits
        _positive("limits.max_tokens", lim.max_tokens)
        _positive("limits.max_context", lim.max_context)
        if lim.max_tokens > lim.max_context:
            raise ConfigError(
                f"limits.max_tokens ({lim.max_tokens}) exceeds "
                f"limits.max_context ({lim.max_context})"
            )
        if lim.max_context > m.max_context:
            raise ConfigError(
                f"limits.max_context ({lim.max_context}) exceeds "
                f"model.max_context ({m.max_context})"
            )

        # observability
        _positive("observability.cycle_profile_ring", obs.cycle_profile_ring)
        _at_least("observability.trace_sample_every", obs.trace_sample_every, 0)

    # -- convenience -------------------------------------------------------
    def alias(self, name: str) -> AliasConfig | None:
        """Profile for a requested model string, or ``None`` if unknown."""
        if name == self.model.resolved_name():
            return AliasConfig(
                name=name,
                enable_thinking=self.model.enable_thinking,
                reasoning_effort=self.model.reasoning_effort,
                sampling=self.model.sampling,
            )
        for a in self.model.aliases:
            if a.name == name:
                return a
        return None


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _positive(path: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{path} must be greater than 0, got {value}")


def _at_least(path: str, value: float, floor: float) -> None:
    if value < floor:
        raise ConfigError(f"{path} must be at least {floor}, got {value}")


def _one_of(path: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ConfigError(
            f"{path} must be one of {', '.join(allowed)}, got {value!r}"
        )


def _sampling(path: str, s: SamplingConfig) -> None:
    if not 0.0 <= s.temperature <= 2.0:
        raise ConfigError(
            f"{path}.temperature must be in [0, 2], got {s.temperature}"
        )
    if not 0.0 < s.top_p <= 1.0:
        raise ConfigError(f"{path}.top_p must be in (0, 1], got {s.top_p}")
    _at_least(f"{path}.top_k", s.top_k, 0)
    if not 0.0 <= s.min_p <= 1.0:
        raise ConfigError(f"{path}.min_p must be in [0, 1], got {s.min_p}")
    _positive(f"{path}.repetition_penalty", s.repetition_penalty)
    if s.max_tokens is not None:
        _positive(f"{path}.max_tokens", s.max_tokens)


# ---------------------------------------------------------------------------
# mapping to dataclass
# ---------------------------------------------------------------------------


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls, include_extras=False)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path or 'config'} must be a table, got {type(data).__name__}"
        )
    known = {f.name for f in fields(cls)}
    for key in data:
        if key not in known:
            raise ConfigError(
                f"unknown key {_join(path, str(key))!r}; "
                f"{path or 'the config'} accepts {', '.join(sorted(known))}"
            )
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name], _join(path, f.name))
    missing = [
        f.name
        for f in fields(cls)
        if f.name not in kwargs
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    ]
    if missing:
        raise ConfigError(f"{_join(path, missing[0])} is required")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = get_origin(hint)

    # X | None
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(args[0], value, path)

    if is_dataclass(hint):
        return _build(hint, value, path)

    if origin is tuple:
        (item, _ellipsis) = get_args(hint)
        if is_dataclass(item):
            return tuple(_named_items(item, value, path))
        if isinstance(value, (list, tuple)):
            return tuple(_coerce(item, v, f"{path}[{i}]") for i, v in enumerate(value))
        raise ConfigError(f"{path} must be an array, got {type(value).__name__}")

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a table, got {type(value).__name__}")
        return dict(value)

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {value!r}")
        return value
    if hint is Any:
        return value
    raise ConfigError(f"{path}: unsupported config type {hint!r}")


def _named_items(item: type, value: Any, path: str) -> list[Any]:
    """A table of name -> table, or an array of tables that carry their name.

    TOML writes aliases as ``[model.aliases."x:no-think"]``, which decodes to a
    mapping. The dataclass holds a tuple so the config stays hashable and
    ordered, and the mapping key becomes the entry's ``name``.
    """
    if isinstance(value, Mapping):
        out = []
        for key, entry in value.items():
            if not isinstance(entry, Mapping):
                raise ConfigError(f"{_join(path, str(key))} must be a table")
            if "name" in entry and entry["name"] != key:
                raise ConfigError(
                    f"{_join(path, str(key))}.name is {entry['name']!r} but the "
                    f"table is keyed {key!r}; drop the name field"
                )
            out.append(_build(item, {**entry, "name": key}, _join(path, str(key))))
        return out
    if isinstance(value, (list, tuple)):
        return [_build(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise ConfigError(f"{path} must be a table of tables, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# dumping
# ---------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            if v is None or (isinstance(v, Mapping) and not v):
                continue
            out[f.name] = _to_plain(v)
        return out
    if isinstance(obj, tuple):
        items = list(obj)
        if items and is_dataclass(items[0]):
            table: dict[str, Any] = {}
            for entry in items:
                plain = _to_plain(entry)
                table[plain.pop("name")] = plain
            return table
        return items
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"cannot write {value!r} as TOML")


def _toml_key(key: str) -> str:
    ok = key and all(ch.isalnum() or ch in "_-" for ch in key)
    return key if ok else _toml_value(key)


def dumps_toml(data: Mapping[str, Any], prefix: str = "") -> str:
    """Write a nested table of scalars, arrays and tables as TOML.

    Small on purpose: the standard library reads TOML and does not write it, and
    the only thing that needs writing here is a resolved config whose shape we
    already know.
    """
    scalars = {k: v for k, v in data.items() if not isinstance(v, Mapping)}
    tables = {k: v for k, v in data.items() if isinstance(v, Mapping)}
    lines: list[str] = []
    if prefix and (scalars or not tables):
        lines.append(f"[{prefix}]")
    for k, v in scalars.items():
        lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
    if scalars and tables:
        lines.append("")
    for i, (k, v) in enumerate(tables.items()):
        child = f"{prefix}.{_toml_key(k)}" if prefix else _toml_key(k)
        lines.append(dumps_toml(v, child).rstrip("\n"))
        if i != len(tables) - 1:
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# --set overrides
# ---------------------------------------------------------------------------


def parse_override(text: str) -> tuple[tuple[str, ...], Any]:
    """Split ``a.b=c`` into a key path and a TOML-parsed value.

    The value goes through the TOML parser, so ``--set server.port=8085`` is an
    integer, ``--set kernels.disabled='["moe_gather_int8"]'`` is an array and
    ``--set model.path=/models/qwen`` is a bare string. Getting a port as the
    string "8085" and finding out at bind time is exactly the class of error
    this whole module exists to prevent.
    """
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(
            f"override {text!r} is not of the form key.path=value"
        )
    path = tuple(part.strip() for part in key.strip().split("."))
    if any(not part for part in path):
        raise ConfigError(f"override {text!r} has an empty key segment")
    raw = raw.strip()
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except Exception:  # noqa: BLE001 - a bare word is a string, not an error
        value = raw
    return path, value


def apply_overrides(
    data: Mapping[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """Return a copy of ``data`` with every ``a.b=c`` override applied."""
    out = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in data.items()}
    for text in overrides:
        path, value = parse_override(text)
        cursor: dict[str, Any] = out
        for i, part in enumerate(path[:-1]):
            nxt = cursor.get(part)
            if nxt is None:
                nxt = {}
            elif isinstance(nxt, Mapping):
                nxt = dict(nxt)
            else:
                raise ConfigError(
                    f"override {text!r} descends into "
                    f"{'.'.join(path[: i + 1])}, which is not a table"
                )
            cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = value
    return out
