# SPDX-License-Identifier: MIT
"""Op registry: the concrete implementation of :class:`titan.core.ports.OpRegistry`.

Every op is a :class:`KernelOp`: a name, a reference implementation in plain MLX
ops, an optional accelerated implementation (a ``mx.fast.metal_kernel`` or a
specialised MLX call sequence), a ``supports`` predicate over the call's shape
class, and the exactness class the pair was measured at.

Selection happens once per op per shape class, not per call. A shape class is
what the op's ``key`` function derives from the call arguments: operand shapes,
operand dtypes, the device, and any non-array discriminator that changes which
implementation is legal (a mode string, a bit width, a top_k). The resolved
implementation is memoised on that class, so the hot path is a dict lookup.

The registry is constructed once in ``titan.config.wiring`` and passed down;
nothing here registers itself at import time and nothing looks an op up by
importing a module. Modules expose a module-level ``OP`` object, and
:func:`build_registry` collects them.

State is never hidden. Weight tables, reader pools and compiled-kernel handles
belong to objects the caller constructs and passes in; the only cache the
registry owns is the selection memo, which is bookkeeping and can be dropped at
any time with :meth:`KernelRegistry.reset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx

from titan.config.schema import KernelConfig
from titan.core.errors import ConfigError, KernelError

__all__ = [
    "KernelOp",
    "KernelRegistry",
    "ShapeClass",
    "build_registry",
    "current",
    "reference_only",
    "set_current",
    "shape_class",
]


# ---------------------------------------------------------------------------
# shape classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeClass:
    """What selection is allowed to depend on.

    ``shapes`` and ``dtypes`` are parallel tuples over the op's array operands,
    ``device`` is ``"gpu"`` or ``"cpu"``, and ``extra`` carries any non-array
    discriminator. Two calls with the same shape class always select the same
    implementation, which is what makes selecting once per class sound.
    """

    shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[Any, ...] = ()
    device: str = "gpu"
    extra: tuple[Any, ...] = ()


def _device_name(device=None) -> str:
    if device is not None:
        return str(device)
    try:
        return "gpu" if mx.metal.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - CPU-only build
        return "cpu"


def shape_class(*arrays, extra: Sequence[Any] = (), device=None) -> ShapeClass:
    """Build a :class:`ShapeClass` from array operands (``None`` entries skipped)."""
    live = [a for a in arrays if a is not None]
    return ShapeClass(
        shapes=tuple(tuple(a.shape) for a in live),
        dtypes=tuple(a.dtype for a in live),
        device=_device_name(device),
        extra=tuple(extra),
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


@dataclass
class KernelOp:
    """One op. Satisfies :class:`titan.core.ports.Op`.

    ``tolerance`` is the exactness bar the pair is held to: ``0.0`` means
    bit-identical, otherwise it is a relative bound the op's test interprets
    (bf16 ULPs for elementwise ops, rrmse for the chunked scan). ``shapes`` is
    the shape list the exactness test must cover, so the test runs over shapes
    the op declares rather than shapes the author liked.
    """

    name: str
    reference_fn: Callable[..., Any]
    key: Callable[..., ShapeClass]
    fast_fn: Callable[..., Any] | None = None
    supports_key: Callable[[ShapeClass], bool] = lambda _k: False
    tolerance: float = 0.0
    shapes: tuple[Mapping[str, int], ...] = ()
    exactness: str = "unspecified"
    source: str = ""
    aliases: tuple[str, ...] = ()
    default_off: bool = False
    """Is this op's fast path off unless a configuration asks for it by name?

    The per-op default, in code, next to the op. An op is ``default_off``
    because it was *measured* to cost throughput on this machine at the shapes
    the engine actually calls it with, not because it is wrong: every one of
    them still passes its exactness test and can be turned back on by naming it
    in ``kernels.enabled``. ``bench/decode/ROUND4.md`` carries the measurement
    for each, and an op that is turned off here should carry the reason on its
    own line.
    """
    default_off_reason: str = ""

    # -- the Op protocol ---------------------------------------------------
    def reference(self, *args: Any, **kwargs: Any) -> Any:
        return self.reference_fn(*args, **kwargs)

    def fast(self, *args: Any, **kwargs: Any) -> Any:
        if self.fast_fn is None:
            raise KernelError(f"op {self.name!r} has no fast implementation")
        return self.fast_fn(*args, **kwargs)

    def supports(self, *args: Any, **kwargs: Any) -> bool:
        if self.fast_fn is None:
            return False
        return self.supports_key(self.key(*args, **kwargs))

    @property
    def has_fast(self) -> bool:
        return self.fast_fn is not None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_FAST = "fast"
_REFERENCE = "reference"


class KernelRegistry:
    """Name -> :class:`KernelOp`, plus the per-shape-class selection memo."""

    def __init__(self, config: KernelConfig | None = None) -> None:
        self.config = config or KernelConfig()
        self._ops: dict[str, KernelOp] = {}
        self._canonical: dict[str, str] = {}
        self._memo: dict[tuple[str, ShapeClass], str] = {}
        self._calls: dict[str, int] = {}
        self._fallbacks: dict[str, int] = {}

    # -- registration ------------------------------------------------------
    def register(self, op: KernelOp) -> None:
        for name in (op.name, *op.aliases):
            existing = self._ops.get(name)
            if existing is not None and existing is not op:
                raise ConfigError(f"op name {name!r} is registered twice")
            self._ops[name] = op
            self._canonical[name] = op.name

    def get(self, name: str) -> KernelOp:
        try:
            return self._ops[name]
        except KeyError:
            raise ConfigError(
                f"no such op {name!r}; registered: {', '.join(self.names())}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({op.name for op in self._ops.values()}))

    def validate(self) -> None:
        """Reject a stale bisect flag rather than letting it quietly do nothing."""
        for name in (*self.config.disabled, *self.config.enabled):
            if name not in self._ops:
                raise ConfigError(
                    f"kernels config names {name!r}, which is not a registered op"
                )

    # -- policy ------------------------------------------------------------
    def _fast_allowed(self, op: KernelOp) -> bool:
        if not op.has_fast:
            return False
        names = {op.name, *op.aliases}
        if names & set(self.config.disabled):
            return False
        asked_for = bool(names & set(self.config.enabled))
        if self.config.enabled and not asked_for:
            return False
        # A measured-off op takes its fast path only when a configuration names
        # it. An empty ``kernels.enabled`` means "every op's default", not
        # "every op's fast path", which is the distinction ROUND4 needed: the
        # production configuration names nothing, and the ops that cost
        # throughput there should be off in it without anyone having to
        # remember to list them.
        if op.default_off and not asked_for:
            return False
        return True

    def defaults(self) -> Mapping[str, bool]:
        """Name -> is this op's fast path on under an empty configuration."""
        return {name: not self._ops[name].default_off for name in self.names()}

    def fast_disabled(self, name: str) -> bool:
        """Is this op's fast path off by *policy* rather than by shape?

        The shape-class test is the caller's business; this answers only "did
        the configuration turn this op off", which is what a call site needs in
        order to decide between the registry and its own stock MLX path.
        """
        return not self._fast_allowed(self.get(name))

    def selection(self, name: str, key: ShapeClass) -> str:
        """``"fast"`` or ``"reference"`` for this op at this shape class, memoised."""
        op = self.get(name)
        memo_key = (op.name, key)
        hit = self._memo.get(memo_key)
        if hit is None:
            hit = (
                _FAST
                if self._fast_allowed(op) and op.supports_key(key)
                else _REFERENCE
            )
            self._memo[memo_key] = hit
        return hit

    def selections(self) -> Mapping[tuple[str, ShapeClass], str]:
        """Everything selected so far. For logging and tests."""
        return dict(self._memo)

    def reset(self) -> None:
        """Drop the selection memo and the counters. Registrations survive."""
        self._memo.clear()
        self._calls.clear()
        self._fallbacks.clear()

    # -- dispatch ----------------------------------------------------------
    def resolve(self, name: str) -> Callable[..., Any]:
        """The callable to use for ``name`` under the current config.

        The returned dispatcher does the per-shape-class selection (memoised) on
        the way through, because which implementation is legal depends on the
        shapes and the caller has them, not the registry.
        """
        op = self.get(name)

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            self._calls[op.name] = self._calls.get(op.name, 0) + 1
            if self.selection(op.name, op.key(*args, **kwargs)) == _REFERENCE:
                return op.reference_fn(*args, **kwargs)
            try:
                return op.fast_fn(*args, **kwargs)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the fail-open contract
                if not self.config.fail_open:
                    raise KernelError(f"{op.name} fast path failed") from exc
                self._fallbacks[op.name] = self._fallbacks.get(op.name, 0) + 1
                return op.reference_fn(*args, **kwargs)

        dispatch.__name__ = f"titan_kernel_{op.name}"
        dispatch.__doc__ = op.reference_fn.__doc__
        return dispatch

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.resolve(name)(*args, **kwargs)

    def counters(self) -> Mapping[str, int]:
        out: dict[str, int] = {}
        for name, n in self._calls.items():
            out[f"{name}.calls"] = n
        for name, n in self._fallbacks.items():
            out[f"{name}.fallbacks"] = n
        return out


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _op_modules():
    """Import the op modules here, not at package import, so that constructing a
    registry is the only thing that pulls Metal sources into memory."""
    from titan.kernels import (  # noqa: PLC0415
        gdn_chunk_scan,
        gdn_norm_gate,
        grouped_rmsnorm_bf16,
        hc_prefill,
        moe_gather_int8,
        moe_gather_ws,
        moe_weighted_sum,
        ple_packed_lookup,
        qsa_gathered_attention,
        topk_radix,
        verify_accept,
    )

    return (
        gdn_norm_gate,
        moe_weighted_sum,
        hc_prefill,
        gdn_chunk_scan,
        moe_gather_ws,
        moe_gather_int8,
        grouped_rmsnorm_bf16,
        topk_radix,
        ple_packed_lookup,
        verify_accept,
        qsa_gathered_attention,
    )


def build_registry(config: KernelConfig | None = None) -> KernelRegistry:
    """Register every op and apply the enable/disable policy.

    Raises :class:`~titan.core.errors.ConfigError` if a name in
    ``config.disabled`` is not a registered op, so a stale bisect flag cannot
    quietly do nothing.
    """
    registry = KernelRegistry(config)
    for module in _op_modules():
        registry.register(module.OP)
    registry.validate()
    set_current(registry)
    return registry


def reference_only() -> KernelRegistry:
    """A registry with every fast path off. The M1 parity baseline and the
    control arm of every kernel A/B."""
    registry = build_registry(KernelConfig())
    names = registry.names()
    registry.config = KernelConfig(disabled=names, fail_open=False)
    registry.reset()
    set_current(registry)
    return registry


# ---------------------------------------------------------------------------
# the process's configured registry
# ---------------------------------------------------------------------------
#
# ``titan.config.wiring`` builds a registry from ``kernels`` and hands it to
# the engine. The MLX adapter is not on that path: ``build_backend`` never
# passes the registry down, and ``titan/adapters/mlx/kernels.py`` used to build
# its own with default settings. The consequence was quiet and bad --
# ``kernels.reference_only = true`` disabled nothing inside the forward, so the
# control arm of every kernel A/B was not a control, and INCIDENTS rule 2 could
# not be followed as written.
#
# Publishing the last registry built is the smallest fix that stays out of the
# wiring: the wiring builds one at startup, before the checkpoint is loaded and
# long before the first forward, so the adapter's first lookup finds it. A
# process that never builds one (a unit test, a bench) gets ``None`` and the
# adapter falls back to building a default, which is what it did before.

_current: KernelRegistry | None = None


def set_current(registry: KernelRegistry | None) -> None:
    """Publish *registry* as the process's configured one."""
    global _current
    _current = registry


def current() -> KernelRegistry | None:
    """The registry the process was configured with, or ``None``."""
    return _current
