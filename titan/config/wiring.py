"""Composition root. The only module allowed to import across layers.

Everything is constructed here, once, in a known order, and handed to whoever
needs it as a constructor argument. There is no service locator, no global
registry lookup at call time and no import-time side effect anywhere else in the
package: an adapter is used because it was passed in, which is what makes a test
able to swap any one of them for a fake.

Construction order is fixed by dependency, not by taste:

    config -> profiler -> op registry -> kernels -> model backend
           -> tokenizer, template renderer, tool parser
           -> kv store -> prefix cache
           -> drafter, verifier -> decode cycle -> scheduler -> api app

The backend's ``warmup()`` runs after the registry is populated and before the
API binds its port, so no client ever pays for Metal compilation.

Every cross-layer import happens inside a function, not at module scope. Two
reasons, and the second is the one that matters day to day. Importing this
module must not pull MLX, Metal sources or a checkpoint reader into a process
that only wants to check a config file. And half the adapters are still being
written, so a module-scope import of one of them would make the whole wiring
unimportable; instead each builder raises a :class:`NotImplementedError` that
names the module still owed, and the parts that do exist can be built and tested
today against fakes for the parts that do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from titan.config import settings
from titan.config.schema import TitanConfig
from titan.core.errors import ConfigError

_FINE_CUT_DISABLED = 1 << 30
"""A gain no prompt can reach, which is how ``fine_tail = false`` is spelled."""

__all__ = [
    "Runtime",
    "TitanRuntime",
    "Parts",
    "build_runtime",
    "load_config",
    "read_config",
]


class Runtime(Protocol):
    """The assembled object graph. Held by the process entry point."""

    config: TitanConfig
    scheduler: Any
    profiler: Any
    registry: Any

    def start(self) -> None: ...

    def stop(self, drain_timeout_s: float) -> None: ...


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


def read_config(path: str | None = None) -> str:
    """Text of the config file, resolving ``$TITAN_CONFIG`` when path is None."""
    resolved = settings.resolve_config_path(path)
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {resolved}: {exc}") from None


def load_config(path: str | None = None, overrides: Sequence[str] = ()) -> TitanConfig:
    """Read the config file, apply ``key.path=value`` overrides, validate, return.

    Raises :class:`~titan.core.errors.ConfigError` on the first problem, with
    the offending key named. Never falls back to a default for a key the file
    got wrong: a typo must stop the process, not silently change behaviour.

    The parse itself lives in :func:`titan.config.settings.load_core`, because
    that module owns the one environment read Titan allows and there is no
    reason for two functions to know how to turn a path into a config.
    """
    return settings.load_core(path, tuple(overrides))


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


@dataclass
class Parts:
    """Pre-built components, for tests and for partial runtimes.

    Anything set here is used as is and its builder never runs. That is the
    whole seam: a wiring test hands in a fake backend, a fake store and a fake
    engine, and exercises the real construction order against them with no MLX
    import, no checkpoint and no GPU.
    """

    profiler: Any = None
    registry: Any = None
    backend: Any = None
    tokenizer: Any = None
    template: Any = None
    codec: Any = None
    store: Any = None
    cache: Any = None
    cycle: Any = None
    engine: Any = None
    app: Any = None


@dataclass
class TitanRuntime:
    """The assembled graph. Satisfies :class:`Runtime`."""

    config: TitanConfig
    profiler: Any
    registry: Any
    backend: Any
    tokenizer: Any
    template: Any
    codec: Any
    store: Any
    cache: Any
    engine: Any
    app: Any
    _started: bool = field(default=False, repr=False)

    @property
    def scheduler(self) -> Any:
        """The engine is the scheduler as far as the entry point is concerned."""
        return self.engine

    def start(self) -> None:
        """Warm the backend and start the loop. Does not bind a port.

        Binding is uvicorn's job and it happens in :func:`titan.cli.serve`, so
        that building a runtime in a test can never open a socket.
        """
        if self._started:
            return
        warmup = getattr(self.backend, "warmup", None)
        if callable(warmup):
            warmup()
        start = getattr(self.engine, "start", None)
        if callable(start):
            start()
        self._started = True

    def stop(self, drain_timeout_s: float = 30.0) -> None:
        stop = getattr(self.engine, "stop", None)
        if callable(stop):
            stop(drain_timeout_s)
        flush = getattr(self.store, "flush", None)
        if callable(flush):
            flush(drain_timeout_s)
        self._started = False


def _owed(module: str, what: str) -> NotImplementedError:
    """The one shape of failure for a component that does not exist yet."""
    return NotImplementedError(
        f"{what} is not implemented yet: {module} is still a stub. "
        f"Pass a prebuilt one through wiring.Parts to build a runtime without it."
    )


# -- builders ---------------------------------------------------------------


def _resolve(module_name: str, attr: str, what: str) -> Any:
    """Import ``module_name`` and pull ``attr`` off it, or say what is owed."""
    import importlib  # noqa: PLC0415

    module = importlib.import_module(module_name)
    found = getattr(module, attr, None)
    if found is None:
        raise _owed(module_name, what)
    return found


def build_profiler(config: TitanConfig) -> Any:
    import time  # noqa: PLC0415

    from titan.observability.profiler import build_profiler as _build  # noqa: PLC0415

    class _Clock:
        def now(self) -> float:
            return time.monotonic()

    try:
        return _build(config.observability, _Clock())
    except NotImplementedError:
        raise _owed("titan.observability.profiler", "the profiler") from None


def build_registry(config: TitanConfig) -> Any:
    from titan.kernels.registry import (  # noqa: PLC0415
        build_registry as _build,
        reference_only,
    )

    if config.kernels.reference_only:
        return reference_only()
    return _build(config.kernels)


def build_backend(config: TitanConfig, registry: Any) -> Any:
    """Load the checkpoint and wrap it. The one call that costs minutes.

    Two wrappers, and the order is not cosmetic. ``loader.load_model`` returns
    the vendored module tree; ``TitanQwenFlashNext`` is the forward API over it,
    which is what the parity harness drives directly and what the backend
    forwards to; ``MLXModelBackend`` adds the handle table and the acceptance
    reduction the engine's port describes. Handing the raw module to the
    backend gets you an object without ``new_state``, which fails at the first
    request rather than at startup.
    """
    from titan.adapters.mlx import loader  # noqa: PLC0415
    from titan.adapters.mlx.backend import MLXModelBackend  # noqa: PLC0415
    from titan.adapters.mlx.model import TitanQwenFlashNext  # noqa: PLC0415

    model, _plan = loader.load_model(
        config.model.path,
        mtp_enabled=config.speculation.enabled,
        mtp_depth=config.speculation.mtp_depth_max,
        fuse_gate_up=config.model.fuse_gate_up,
    )
    return MLXModelBackend(
        TitanQwenFlashNext(model, prefill_chunk=config.scheduler.prefill_chunk),
        prime_mtp=config.speculation.enabled and config.speculation.mtp_prime_prompt,
        prime_window=config.speculation.mtp_prime_window,
        release_after_prefill=config.scheduler.release_after_prefill,
    )


def build_tokenizer(config: TitanConfig) -> Any:
    load = _resolve(
        "titan.adapters.mlx.tokenizer", "load_tokenizer", "the tokenizer adapter"
    )
    return load(config.model.path)


def build_template(config: TitanConfig) -> Any:
    load = _resolve(
        "titan.adapters.mlx.template", "load_renderer", "the template renderer"
    )
    return load(
        config.model.path,
        enable_thinking=config.model.enable_thinking,
        reasoning_effort=config.model.reasoning_effort,
    )


def build_codec(config: TitanConfig, backend: Any, signature: Any = None) -> Any:
    """The ``StateCodec`` the store serialises through.

    The backend builds it, because only the adapter knows how a state handle
    turns into bytes, and it needs the signature because the codec is what
    stamps it into every payload the store writes.
    """
    codec = getattr(backend, "codec", None)
    if codec is not None:
        return codec
    factory = getattr(backend, "state_codec", None)
    if not callable(factory):
        raise _owed("titan.adapters.mlx.codec", "the state codec")
    return factory(signature if signature is not None else build_signature(config, backend))


def build_signature(config: TitanConfig, backend: Any) -> Any:
    """What must match for a stored byte string to mean anything."""
    from titan.adapters.cache.format import CacheSignature  # noqa: PLC0415

    layout = getattr(backend, "layer_layout", None)
    if layout is None:
        raise _owed("titan.adapters.mlx.backend", "the backend's layer layout")
    return CacheSignature(
        model_name=config.model.resolved_name(),
        layer_layout=tuple(layout),
        block_tokens=config.cache.block_tokens,
        snapshot_dtype=config.model.dtype.kv,
    )


def build_store(config: TitanConfig, signature: Any) -> Any:
    factory = _resolve(
        "titan.adapters.cache.store", "TwoTierStateStore", "the KV state store"
    )
    c = config.cache
    return factory(
        signature,
        ssd_dir=c.ssd_dir,
        hot_budget_bytes=int(c.ram_tier_mb * 1024**2),
        ssd_capacity_bytes=int(c.ssd_capacity_gb * 1024**3),
        pending_budget_bytes=int(c.pending_write_budget_mb * 1024**2),
        max_stall_s=c.max_stall_ms / 1000.0,
    )


def build_cache(config: TitanConfig, store: Any, codec: Any) -> Any:
    factory = _resolve(
        "titan.adapters.cache.prefix", "BlockPrefixCache", "the prefix cache"
    )
    c = config.cache
    return factory(
        store,
        codec,
        block_tokens=c.block_tokens,
        snapshot_grid=c.snapshot_grid,
        snapshot_at_prompt_end=c.snapshot_at_prompt_end,
        chunk_tokens=config.scheduler.prefill_chunk,
        contended_chunk_tokens=c.block_tokens,
        # The gate is a measured cost, not a count of blocks: the extra chunk
        # launch plus the snapshot is about 283 ms, so a fine cut has to buy at
        # least 384 tokens back to be worth taking. ``fine_tail = false``
        # raises the threshold out of reach rather than branching downstream.
        fine_min_gain_tokens=(
            c.fine_min_gain_tokens if c.fine_tail else _FINE_CUT_DISABLED
        ),
        fine_max_pending_bytes=int(c.pending_write_budget_mb * 1024**2),
        # The stall cap, spent as a per-cycle serialisation budget. The
        # scheduler reads it back off the cache, because the engine may not
        # import the config.
        store_budget_s=c.max_stall_ms / 1000.0,
    )


def build_admission_config(config: TitanConfig, backend: Any = None) -> Any:
    """The flat numbers the engine needs, since the engine may not import config."""
    factory = _resolve(
        "titan.engine.admission", "AdmissionConfig", "the admission config"
    )
    fields = dict(
        max_sequences=config.scheduler.max_seqs,
        queue_depth=config.scheduler.queue_depth,
        max_context=config.limits.max_context,
        prefill_chunk_tokens=config.scheduler.prefill_chunk,
        block_tokens=config.cache.block_tokens,
        snapshot_grid=config.cache.snapshot_grid,
        memory_guard_gb=config.scheduler.memory_guard_gb
        * config.scheduler.memory_guard_soft_fraction,
    )
    per_token = getattr(backend, "state_bytes_per_token", None)
    if per_token:
        # The backend knows the layer shapes; the default here is a number from
        # another checkpoint. A guard that overestimates skips a long prompt
        # rather than refusing it, and a skipped entry keeps its place in the
        # queue, so the failure is a request that never runs and no error
        # anywhere.
        fields["state_bytes_per_token"] = float(per_token)
    if config.model.weights_gb:
        # 0 means the config did not say. Passing it would tell the guard the
        # weights are free, which is worse than letting admission keep its own
        # measured default.
        fields["weights_gb"] = config.model.weights_gb
    return factory(**fields)


def build_drafter(config: TitanConfig, *, backend: Any, profiler: Any) -> Any:
    """The MTP drafter, or ``None`` when the backend has no head to draft with.

    Speculation on means the drafter is present. A backend that reports
    ``draft_depth_max == 0`` was loaded from a checkpoint without the MTP block,
    and there is nothing to build; the cycle then runs its one-row path, which
    is the same tokens at the same acceptance-free cost.
    """
    if int(getattr(backend, "draft_depth_max", 0)) <= 0:
        return None
    factory = _resolve("titan.adapters.mlx.drafter", "MTPDrafter", "the MTP drafter")
    return factory(
        backend,
        chain=config.speculation.mtp_chain,
        chain_cache=config.speculation.mtp_chain_cache,
        p_min=config.speculation.draft_p_min,
        max_depth=config.speculation.mtp_depth_max,
        align_positions=config.speculation.mtp_head_align_positions,
        profiler=profiler,
    )


def build_verifier(config: TitanConfig) -> Any:
    """The depth policy. Expected value when adaptive, a fixed depth when not."""
    factory = _resolve(
        "titan.engine.decode_cycle",
        "ExpectedValueDepthController",
        "the depth controller",
    )
    speculation = config.speculation
    return factory(
        max_depth=speculation.mtp_depth_max,
        # Zero when the policy is adaptive, because not drafting is one of the
        # options it has to be able to price: at 64k with an accepted median of
        # 1, a floor of one draft is two rejected columns a cycle that no
        # measurement can talk the policy out of. ``mtp_depth_min`` is the
        # fixed policy's floor, where nothing measures anything.
        min_depth=0 if speculation.adaptive_depth else speculation.mtp_depth_min,
        adaptive=speculation.adaptive_depth,
        window=speculation.acceptance_window,
        rows_budget=config.scheduler.decode_rows_budget,
    )


def build_cycle(
    config: TitanConfig, *, backend: Any, tokenizer: Any, profiler: Any
) -> Any:
    """MTP when speculation is on, the plain single-row loop when it is not."""
    if config.speculation.enabled:
        factory = _resolve(
            "titan.engine.decode_cycle", "MTPDecodeCycle", "the MTP decode cycle"
        )
        return factory(
            backend=backend,
            tokenizer=tokenizer,
            profiler=profiler,
            drafter=build_drafter(config, backend=backend, profiler=profiler),
            verifier=build_verifier(config),
            max_depth=config.speculation.mtp_depth_max,
            rows_budget=config.scheduler.decode_rows_budget,
            overlap_draft=config.speculation.overlap_draft,
        )
    factory = _resolve(
        "titan.engine.decode_cycle", "PlainDecodeCycle", "the plain decode cycle"
    )
    return factory(backend=backend, tokenizer=tokenizer, profiler=profiler)


def build_engine(
    config: TitanConfig,
    *,
    backend: Any,
    tokenizer: Any,
    cache: Any,
    profiler: Any,
    cycle: Any = None,
) -> Any:
    loop_factory = _resolve("titan.engine.scheduler", "EngineLoop", "the engine loop")
    engine_factory = _resolve("titan.engine.engine", "TitanEngine", "the engine")
    cycle = cycle or build_cycle(
        config, backend=backend, tokenizer=tokenizer, profiler=profiler
    )
    loop = loop_factory(
        backend=backend,
        tokenizer=tokenizer,
        cycle=cycle,
        config=build_admission_config(config, backend),
        cache=cache,
        profiler=profiler,
        rows_budget=config.scheduler.decode_rows_budget,
    )
    return engine_factory(loop)


def build_app(
    config: TitanConfig,
    *,
    engine: Any,
    tokenizer: Any,
    template: Any,
    profiler: Any = None,
) -> Any:
    """The FastAPI application, against the API's projection of the config."""
    from titan.api.openai import ChatDeps, build_app as _build  # noqa: PLC0415

    return _build(
        ChatDeps(
            config=settings.TitanConfig.from_schema(config),
            engine=engine,
            renderer=template,
            tokenizer=tokenizer,
            profiler=profiler,
            resolved_config=resolved_config_view(config),
        )
    )


def build_runtime(config: TitanConfig, parts: Parts | None = None) -> TitanRuntime:
    """Build the object graph. Loads the model; does not bind the port.

    Construction order is the dependency order in the module docstring, and it
    is not negotiable: the registry exists before the backend because the
    backend resolves its kernels through it, the store exists after the backend
    because its cache signature names the backend's layer layout, and the app is
    last because it is the only part that can be handed to a client.
    """
    config.validate()
    parts = parts or Parts()

    profiler = parts.profiler or build_profiler(config)
    registry = parts.registry or build_registry(config)
    backend = parts.backend or build_backend(config, registry)
    tokenizer = parts.tokenizer or build_tokenizer(config)
    template = parts.template or build_template(config)
    signature = None if parts.codec and parts.store else build_signature(config, backend)
    codec = parts.codec or build_codec(config, backend, signature)
    store = parts.store or build_store(config, signature)
    cache = parts.cache or build_cache(config, store, codec)
    engine = parts.engine or build_engine(
        config,
        backend=backend,
        tokenizer=tokenizer,
        cache=cache,
        profiler=profiler,
        cycle=parts.cycle,
    )
    app = parts.app or build_app(
        config,
        engine=engine,
        tokenizer=tokenizer,
        template=template,
        profiler=profiler,
    )
    return TitanRuntime(
        config=config,
        profiler=profiler,
        registry=registry,
        backend=backend,
        tokenizer=tokenizer,
        template=template,
        codec=codec,
        store=store,
        cache=cache,
        engine=engine,
        app=app,
    )


def resolved_config_view(config: TitanConfig) -> Mapping[str, Any]:
    """What ``GET /metrics`` echoes: the resolved config with the key redacted."""
    return config.to_dict(redact=True)
