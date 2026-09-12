"""Ports: the interfaces the core owns and adapters implement.

Every one of these is a ``typing.Protocol``, so adapters do not import the core
to satisfy them and the core never imports an adapter. Composition happens once,
in ``titan.config.wiring``.

Rules that apply to every port here:

- No method takes or returns an mlx array, a numpy array or a tensor of any
  kind. Device data crosses a boundary as an opaque handle. The one deliberate
  exception is :class:`ModelBackend`, whose logits stay on device and are only
  described by shape.
- No method blocks on device work unless its docstring says so. Exactly one
  method per decode cycle is allowed to sync the host, and it is
  :meth:`ModelBackend.verify`.
- Failure is an exception, never a sentinel. Ports raise
  :class:`titan.core.errors.TitanError` subclasses.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from titan.core.types import (
    BlockHash,
    CycleProfile,
    DraftCandidate,
    Message,
    PrefixMatch,
    Request,
    SamplingParams,
    SequenceId,
    StateHandle,
    ToolCall,
    ToolSpec,
    VerifyOutcome,
)

__all__ = [
    "LogitsRef",
    "ModelBackend",
    "KVStateStore",
    "PrefixCache",
    "Tokenizer",
    "TemplateRenderer",
    "ToolCallParser",
    "Drafter",
    "Verifier",
    "NgramReader",
    "Clock",
    "Profiler",
    "OpRegistry",
    "Op",
]


class LogitsRef(Protocol):
    """An opaque reference to logits that are still on the GPU.

    The core passes these between backend calls and never reads them. Reading
    means a host sync, and the decode cycle budgets exactly one of those.
    """

    @property
    def rows(self) -> int: ...

    @property
    def vocab(self) -> int: ...


@runtime_checkable
class ModelBackend(Protocol):
    """The model. One process, one model, one backend instance.

    State handles are the whole contract. The backend owns all device state for
    a sequence behind a :class:`~titan.core.types.StateHandle`; the core decides
    when state is created, forked, truncated and destroyed, and never touches
    the arrays. That is what makes replay-free rollback expressible: the core
    asks for a truncation, and the backend either does it in place or fails
    loudly.
    """

    @property
    def n_layers(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_context(self) -> int: ...

    @property
    def draft_depth_max(self) -> int:
        """Largest chain the MTP block can produce. 0 if the model has none."""

    def open_state(self, seq: SequenceId, capacity_hint: int) -> StateHandle:
        """Allocate state for a new sequence.

        ``capacity_hint`` is the expected final length in tokens. The backend
        may use it to size KV allocations; it must not fail when the sequence
        exceeds it, only reallocate.
        """

    def close_state(self, state: StateHandle) -> None:
        """Release everything behind ``state``. Idempotent."""

    def state_length(self, state: StateHandle) -> int:
        """Tokens the state currently covers. Host-side counter, no sync."""

    def truncate_state(self, state: StateHandle, length: int) -> None:
        """Roll state back to exactly ``length`` tokens, without recomputation.

        Invariant: after this call the state is bit-identical to what it would
        have been had only those ``length`` tokens ever been processed. For the
        12 QSA layers that is an offset move. For the 36 Gated DeltaNet layers
        it is not, because the recurrence has no inverse, which is why
        :meth:`verify` must snapshot the pre-verify recurrent state and this
        method restores it. Truncating below the last snapshot point is an
        error, not a slow path.
        """

    def prefill(
        self,
        state: StateHandle,
        tokens: Sequence[int],
        *,
        want_logits: bool = False,
        snapshot: bool = False,
    ) -> LogitsRef | None:
        """Process ``tokens`` into ``state``, appending to what it holds.

        Does not sync. Returns a logits reference for the final position only
        when ``want_logits`` (the last chunk of a prompt), otherwise ``None``:
        this model declares ``supports_skip_lm_head``, and skipping the head is
        worth 48 to 95 ms per chunk.

        When ``snapshot`` is set the backend stages a recurrent-state snapshot
        at the end position, retrievable through :meth:`export_snapshot`. It is
        staged, not written: the store decides when it hits disk.
        """

    def decode(
        self,
        states: Sequence[StateHandle],
        tokens: Sequence[int],
    ) -> LogitsRef:
        """One row per state, one token each. Returns ``[len(states), vocab]``.

        Used when speculation is off and by the M1 reference path. Does not
        sync.
        """

    def verify(
        self,
        states: Sequence[StateHandle],
        drafts: Sequence[DraftCandidate],
        sampling: Sequence[SamplingParams],
    ) -> tuple[list[VerifyOutcome], CycleProfile]:
        """Lockstep batched verify. The one call per cycle that syncs the host.

        All sequences advance together over one padded row block of width
        ``max(len(d.tokens) for d in drafts) + 1``. Acceptance is computed in
        the graph (compare drafted ids against the target argmax, take a
        cumulative product, sum) so that only the small integer acceptance
        vector crosses to the host, and it crosses once.

        Invariants:

        1. Greedy exactness. For every sequence, ``accepted + (bonus,)`` equals
           the tokens plain greedy decode would have produced. Batch
           composition, padding width and draft length may not change output.
        2. Replay-free rollback. On return, each state covers exactly
           ``len(accepted) + 1`` more tokens than it did on entry. Rejected
           rows leave no trace, and no extra forward pass runs.
        3. One host sync, reported as ``CycleProfile.host_syncs == 1``.
        4. Sequences whose drafts are shorter than the block width are padded,
           and padding may not affect their own outputs.
        """

    def export_snapshot(self, state: StateHandle, length: int) -> bytes:
        """Serialise the recurrent state staged at ``length``.

        Raises if nothing was staged there. The bytes are opaque to the core and
        to the store; only the backend interprets them. They carry a version tag
        so a store written by an older build is rejected rather than misread.
        """

    def import_snapshot(self, state: StateHandle, length: int, blob: bytes) -> None:
        """Restore recurrent state, making ``state`` consistent with ``length``
        tokens. The caller has already restored the KV blocks."""

    def warmup(self) -> None:
        """Compile and run every kernel shape the engine will use, once.

        Called before the API starts listening, so first-request latency does
        not include Metal compilation.
        """


@runtime_checkable
class KVStateStore(Protocol):
    """Byte-level persistence for cache blocks and snapshots.

    This port knows nothing about prefixes or hashes as concepts; it is a
    two-tier key-value store with a budget. :class:`PrefixCache` is the policy
    on top of it. Splitting them is what makes the hot RAM tier, the SSD tier
    and an in-memory test double interchangeable.
    """

    def get_block(self, key: BlockHash) -> bytes | None:
        """Blocking read. Called on the prefill path, off the decode loop."""

    def put_block(self, key: BlockHash, payload: bytes) -> None:
        """Queue a write. Returns as soon as the payload is owned by the store.

        Invariant: this never blocks the inference thread for longer than
        ``config.store.max_stall_ms``. The overlay's version could wait up to
        two seconds for a pending-bytes budget, and that showed up directly as
        a slow turn after a large store.
        """

    def get_snapshot(self, snapshot_id: str) -> bytes | None: ...

    def put_snapshot(self, snapshot_id: str, payload: bytes) -> None: ...

    def pending_bytes(self) -> int:
        """Bytes queued but not yet durable. The scheduler reads this to decide
        whether an extra fine snapshot is affordable this turn."""

    def flush(self, timeout_s: float) -> bool:
        """Drain the write queue. Used at shutdown and in tests."""


@runtime_checkable
class PrefixCache(Protocol):
    """Prefix lookup and store policy.

    Two grids, decoupled from day one:

    - the *block size*, which is the unit of KV persistence and reconstruction;
    - the *snapshot grid*, which is where recurrent state can be resumed.

    A prefix is restorable only at a position that is both a block end and a
    snapshot point. The overlay tied these together and could only resume on a
    2048-token grid, so a 6-turn conversation recomputed 11.6k tokens it had
    already seen. Titan sets block size below the snapshot grid and always
    snapshots at the end of a prompt, where the last prefill chunk ends anyway,
    so the extra snapshot costs a write and no forward pass.
    """

    def lookup(self, tokens: Sequence[int]) -> PrefixMatch:
        """Longest restorable prefix of ``tokens``.

        Invariant: the returned length is restorable in full. A lookup that
        finds KV blocks but no usable snapshot must report the shorter length
        that does have one, never the longer one.
        """

    def restore(self, match: PrefixMatch, state: StateHandle) -> int:
        """Populate ``state`` from a match. Returns tokens restored.

        Blocking, bytes-bound: measured at 2.77e-3 ms per token in the overlay,
        so 30k tokens is about 85 ms and block count does not matter.
        """

    def store(
        self,
        tokens: Sequence[int],
        state: StateHandle,
        boundaries: Sequence[int],
    ) -> None:
        """Persist ``tokens`` with resumable points at ``boundaries``.

        ``boundaries`` must be ascending, must end at ``len(tokens)`` and each
        must be a position at which :meth:`ModelBackend.export_snapshot`
        succeeds. Store is all-or-nothing per boundary: a boundary whose
        snapshot did not commit is dropped, and the chain truncates there rather
        than recording a length it cannot restore.
        """

    def plan_chunks(
        self,
        matched: int,
        total: int,
        contended: bool,
    ) -> tuple[int, ...]:
        """Chunk end positions for the uncached suffix.

        Invariant: every snapshot-grid multiple strictly inside the suffix is
        itself a chunk end. Stepping over one and never landing on it is what
        broke the overlay's first fine-boundary attempt: nothing staged a
        snapshot there, so the store chain truncated.
        """

    def stats(self) -> Mapping[str, float]:
        """Hit rate, restored tokens, recomputed tokens, evictions."""


@runtime_checkable
class Tokenizer(Protocol):
    """Text to ids and back, with streaming-safe detokenisation.

    Invariant: :meth:`decode_incremental` never emits a partial UTF-8 sequence
    and never emits text that a later token could change (leading-space and
    byte-fallback merges). It holds back what it must and returns it on the next
    call or at flush.
    """

    def encode(self, text: str, *, add_special: bool = False) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def decode_incremental(self, seq: SequenceId, ids: Sequence[int]) -> str:
        """Append ``ids`` to the sequence's detokenisation stream."""

    def flush_incremental(self, seq: SequenceId) -> str:
        """Emit anything held back and drop the stream's state."""

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    @property
    def vocab_size(self) -> int: ...


@runtime_checkable
class TemplateRenderer(Protocol):
    """Chat template. Owns the prompt string, and therefore cache hit rate.

    Invariant: rendering is deterministic and stable across turns. Any per-turn
    nondeterminism (a timestamp, a reordered tool list, a client attribution
    header) changes the prefix and costs a full recompute; that was worth the
    difference between 9-15 tok/s and 68 tok/s effective on agent turns.
    """

    def render(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        reasoning_effort: str,
        add_generation_prompt: bool = True,
    ) -> str: ...

    def reasoning_markers(self) -> tuple[str, str]:
        """Open and close markers for the thinking channel."""


@runtime_checkable
class ToolCallParser(Protocol):
    """Incremental parser for the model's tool-call and reasoning markup.

    Feeds on text deltas and emits clean text plus structured calls. It must
    never leak markup into the content channel: an unterminated call at end of
    stream is reported as an error through :meth:`finish`, not flushed as prose.
    """

    def feed(self, delta: str) -> tuple[str, str, tuple[ToolCall, ...]]:
        """Return (content_delta, reasoning_delta, completed_calls)."""

    def finish(self) -> tuple[str, str, tuple[ToolCall, ...], str | None]:
        """Return the tail plus an error string if parsing ended mid-construct."""


@runtime_checkable
class Drafter(Protocol):
    """Proposes candidate continuations. Never affects output, only speed."""

    @property
    def name(self) -> str: ...

    def propose(
        self,
        states: Sequence[StateHandle],
        contexts: Sequence[Sequence[int]],
        depth: Sequence[int],
    ) -> list[DraftCandidate]:
        """One candidate per state. Does not sync the host.

        A drafter that cannot propose for a sequence returns an empty token
        tuple for it; the cycle then verifies one row for that sequence.
        """

    def observe(self, outcomes: Sequence[VerifyOutcome]) -> None:
        """Feedback for adaptive depth. Called once per cycle after verify."""


@runtime_checkable
class Verifier(Protocol):
    """Depth and admission policy for speculation.

    Separated from :class:`Drafter` because the policy question (how many rows
    to spend this cycle, given the batch and the recent acceptance rate) is
    independent of who proposes the tokens. The overlay's fixed depth of 3 and
    its confidence gate both lived inside the drafter, which is why neither
    could be measured against batch composition.
    """

    def plan_depth(
        self,
        n_sequences: int,
        recent_acceptance: float,
        rows_budget: int,
    ) -> list[int]:
        """Per-sequence draft depth for the coming cycle."""

    def record(self, profile: CycleProfile, outcomes: Sequence[VerifyOutcome]) -> None: ...


@runtime_checkable
class NgramReader(Protocol):
    """Reader for the 51B-parameter n-gram table on SSD.

    The table is a packed contiguous row layout: one pread per row instead of
    three, which cut a 2048-token chunk from 93,046 page reads to 31,855. It
    stays on SSD on a 128 GB machine; a resident copy panicked the kernel.

    Invariant: :meth:`prefetch` is the only way the decode loop should ever
    touch this. Blocking on a row read cost about 2.3 ms per forward, roughly
    9% of a decode step, and it is host and SSD work that has no reason to be
    on the critical path.
    """

    def prefetch(self, rows: Iterable[int]) -> None:
        """Queue row ids. Returns immediately. Duplicate ids are cheap."""

    def gather(self, rows: Sequence[int]) -> Any:
        """Return the rows as a device array, blocking only on rows that were
        not prefetched. The return type is deliberately opaque to the core; only
        the MLX adapter calls this."""

    def stats(self) -> Mapping[str, float]:
        """Hits, misses, bytes read, mean wait."""


@runtime_checkable
class Clock(Protocol):
    """Monotonic time. Injected so tests can be deterministic."""

    def now(self) -> float: ...


@runtime_checkable
class Profiler(Protocol):
    """Observability sink. Always on; the profile is a feature, not a flag.

    Invariant: no method here may sync the device, allocate per call, or block.
    A profiler that cannot keep up drops records and counts the drops.
    """

    def cycle(self, profile: CycleProfile) -> None: ...

    def event(self, name: str, **fields: float | int | str) -> None: ...

    def span(self, name: str) -> "Span":
        """Context manager measuring host wall time."""

    def snapshot(self) -> Mapping[str, Any]:
        """Current counters, for ``GET /metrics`` and the bench harness."""


class Span(Protocol):
    def __enter__(self) -> "Span": ...
    def __exit__(self, *exc: object) -> None: ...


@runtime_checkable
class Op(Protocol):
    """One kernel-level operation with a reference and an optional fast path.

    Every op ships two implementations. ``reference`` is plain mlx ops, obvious
    and slow, and is the definition of correct. ``fast`` is optional and is
    whatever ``mx.fast.metal_kernel`` we wrote. The exactness test for the op
    compares the two over the shapes in ``shapes``, at the tolerance in
    ``tolerance`` (0.0 means bit-identical; the house standard is bit-identical
    or one bf16 ULP).

    Invariant: if ``fast`` raises or its shape is unsupported, the registry
    falls back to ``reference`` and counts it. Output never depends on which one
    ran, which is what makes the fallback safe to leave enabled.
    """

    @property
    def name(self) -> str: ...

    @property
    def tolerance(self) -> float: ...

    @property
    def shapes(self) -> tuple[Mapping[str, int], ...]:
        """Shape dictionaries the exactness test must cover."""

    def reference(self, *args: Any, **kwargs: Any) -> Any: ...

    def fast(self, *args: Any, **kwargs: Any) -> Any: ...

    def supports(self, *args: Any, **kwargs: Any) -> bool:
        """True when ``fast`` can handle this call. Cheap, host-side only."""


@runtime_checkable
class OpRegistry(Protocol):
    """Lookup and policy for kernels.

    The registry is the only place that decides between reference and fast, so
    a bisect over a bad kernel is a config change, not a code change.
    """

    def register(self, op: Op) -> None: ...

    def get(self, name: str) -> Op: ...

    def resolve(self, name: str) -> Any:
        """Return the callable to use for ``name`` under current config."""

    def names(self) -> tuple[str, ...]: ...

    def counters(self) -> Mapping[str, int]:
        """Calls and fallbacks per op."""
