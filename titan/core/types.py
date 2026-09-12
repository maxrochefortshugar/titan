"""Domain types for Titan.

Everything here is a plain dataclass or enum. Nothing in this module imports
mlx, fastapi, numpy or any other framework: the core depends on nothing outside
itself, and these types are the vocabulary every port speaks.

Conventions used throughout:

- Token ids are plain ints. A "position" is an index into the sequence's token
  list, counting the prompt from zero.
- Anything named ``*Handle`` is an opaque identifier owned by an adapter. The
  core creates, compares and passes handles; it never inspects their contents.
- Times are monotonic seconds as floats, taken from the Clock port, never from
  ``time.time``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, NewType, Sequence

__all__ = [
    "RequestId",
    "SequenceId",
    "StateHandle",
    "BlockHash",
    "Role",
    "FinishReason",
    "StopCondition",
    "SamplingParams",
    "ToolSpec",
    "ToolCall",
    "Message",
    "Request",
    "SequencePhase",
    "SequenceState",
    "PrefixMatch",
    "PrefillChunk",
    "DraftCandidate",
    "VerifyOutcome",
    "TokenEvent",
    "StreamEnd",
    "CycleProfile",
]

RequestId = NewType("RequestId", str)
SequenceId = NewType("SequenceId", int)

StateHandle = NewType("StateHandle", int)
"""Opaque per-sequence handle for the model backend's mutable state.

One handle covers everything the backend keeps for one sequence: the 12 QSA KV
caches, the 36 Gated DeltaNet recurrent states, the hyper-connection residual
carriers and the MTP block's own state. The core never sees the arrays. A
handle is valid from :meth:`ModelBackend.open_state` until
:meth:`ModelBackend.close_state`; handle values are never reused inside a
process, so a stale handle is always an error rather than a silent aliasing bug.
"""

BlockHash = NewType("BlockHash", bytes)
"""Content hash of one cache block: the tokens it covers plus the hash of its
predecessor, so a hash identifies a whole prefix, not just a block."""


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    """Natural end of turn: an EOS token or a stop string matched."""
    LENGTH = "length"
    """Hit max_tokens or the context window."""
    TOOL_CALLS = "tool_calls"
    """The turn ended with one or more parsed tool calls."""
    ABORT = "abort"
    """Client disconnected or cancelled."""
    ERROR = "error"
    """Engine fault. The event carries a message; nothing partial is committed."""


@dataclass(frozen=True, slots=True)
class StopCondition:
    """Everything that can end a sequence, resolved once at admission.

    ``stop_strings`` are matched on decoded text, not on token ids, so the
    scheduler must hold back the longest suffix that could still become a stop
    string before it emits text. ``max_tokens`` counts accepted tokens, so an
    MTP cycle that accepts three tokens counts three.
    """

    eos_token_ids: frozenset[int]
    stop_strings: tuple[str, ...] = ()
    max_tokens: int = 4096
    max_total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Sampling configuration for one request.

    Invariant: ``temperature == 0.0`` means greedy, and greedy must be exactly
    reproducible. Every parity milestone is measured with greedy sampling, so
    the greedy path may not depend on any RNG, on batch composition, or on how
    many tokens an MTP cycle happened to accept.

    Invariant: ``seed`` is only consulted when ``temperature > 0``. Two
    sequences in the same lockstep batch must not share an RNG stream, or batch
    composition would leak into output.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the client offered, as given in the OpenAI request."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    """JSON Schema object. Passed to the template renderer verbatim."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One parsed tool call.

    ``arguments`` holds the raw string the model produced, not a parsed object:
    the API layer decides whether to validate it against the schema, and a
    malformed call must still be reportable to the client. ``index`` is stable
    across streaming deltas for the same call.
    """

    index: int
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Request:
    """An admitted unit of work. Immutable once created.

    ``prompt_tokens`` is the rendered, tokenised prompt: templating and tool
    rendering happen in the API layer before admission, so the engine never
    sees chat structure. The engine's only reason to keep ``messages`` is
    observability.
    """

    request_id: RequestId
    prompt_tokens: tuple[int, ...]
    sampling: SamplingParams
    stop: StopCondition
    tools: tuple[ToolSpec, ...] = ()
    messages: tuple[Message, ...] = ()
    stream: bool = True
    arrival_time: float = 0.0
    priority: int = 0
    """Lower is served first. Reserved; the M5 scheduler is FIFO with a guard."""


class SequencePhase(str, Enum):
    WAITING = "waiting"
    """Admitted, no state handle yet."""
    PREFILLING = "prefilling"
    """Owns a state handle; has prefilled ``prefill_position`` of the prompt."""
    DECODING = "decoding"
    DRAINING = "draining"
    """Finished; state handle still open until the store completes."""
    DONE = "done"


@dataclass(slots=True)
class SequenceState:
    """Mutable per-sequence bookkeeping owned by the scheduler.

    This is the only mutable domain object. Invariants the scheduler must keep:

    1. ``len(tokens) == prompt_len + len(generated)``, always.
    2. ``prefill_position <= len(tokens)``. The backend's state handle is
       consistent with exactly ``prefill_position`` tokens plus ``committed``
       decoded tokens; nothing else has touched it.
    3. ``committed`` never decreases. Rollback of a rejected MTP draft happens
       before the tokens are committed, never after (see ``VerifyOutcome``).
    4. ``restored_from`` records the prefix length recovered from cache, so the
       profile can attribute time between reconstruct and recompute.
    """

    sequence_id: SequenceId
    request: Request
    phase: SequencePhase = SequencePhase.WAITING
    state: StateHandle | None = None
    tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0
    prefill_position: int = 0
    committed: int = 0
    """Number of generated tokens accepted and emitted."""
    restored_from: int = 0
    text_emitted: int = 0
    """Characters of detokenised text already sent to the client."""
    finish_reason: FinishReason | None = None
    admitted_at: float = 0.0
    first_token_at: float | None = None


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """Result of a prefix cache lookup.

    Invariant: ``matched_tokens`` is a prefix length in tokens, and the store
    can restore backend state consistent with exactly that many tokens. It is
    not required to be a multiple of the block size: the terminal block of a
    stored prompt may be short, which is the point of fine boundaries.

    Invariant: a match is only usable if the store also holds a Gated DeltaNet
    snapshot at ``matched_tokens``. There is no partial restore of a hybrid
    model: KV can be sliced, recurrent state cannot.
    """

    matched_tokens: int
    block_hashes: tuple[BlockHash, ...]
    snapshot_id: str | None
    tier: str
    """"ram" or "ssd". Observability only; the core does not branch on it."""


@dataclass(frozen=True, slots=True)
class PrefillChunk:
    """One chunked-prefill unit of work.

    Invariant: ``end - start`` is at most the configured chunk size, and a chunk
    always ends either at a chunk-size multiple, at a snapshot-grid multiple, or
    at the end of the prompt. That third case is why every prompt gets a
    snapshot at its natural end at no extra forward pass.
    """

    sequence_id: SequenceId
    start: int
    end: int
    emit_snapshot: bool
    is_last: bool


@dataclass(frozen=True, slots=True)
class DraftCandidate:
    """Tokens proposed for one sequence in one cycle.

    ``tokens`` is a chain, not a tree: mlx cache objects carry a single offset,
    so branching drafts are out of scope. ``source`` distinguishes the MTP block
    from the n-gram copy lane so acceptance can be attributed per source.
    """

    sequence_id: SequenceId
    tokens: tuple[int, ...]
    source: str
    draft_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    """Result of one verify forward for one sequence.

    Invariant: ``accepted`` are the draft tokens the target model confirms, and
    ``bonus`` is the one extra token the verify forward produces for free at the
    first rejection point (or past the last accepted token). Committing
    ``accepted + (bonus,)`` yields exactly the tokens greedy decoding would have
    produced without any drafting. This is the property the M4 chain-parity test
    checks.

    Invariant: rollback is replay-free. The backend must leave its state
    consistent with ``len(accepted) + 1`` new tokens without re-running a
    forward pass over them, so the verify call itself has to write state that
    can be truncated rather than state that has to be rebuilt.
    """

    sequence_id: SequenceId
    accepted: tuple[int, ...]
    bonus: int
    n_drafted: int

    @property
    def n_committed(self) -> int:
        return len(self.accepted) + 1


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One streaming output event.

    ``text`` is the detokenised delta, already safe to send: the scheduler has
    resolved any partial UTF-8 and any held-back stop-string suffix.
    ``reasoning`` is true while the model is inside its thinking channel, which
    the API layer maps to ``reasoning_content``.
    """

    request_id: RequestId
    token_ids: tuple[int, ...]
    text: str
    reasoning: bool = False
    tool_call_delta: ToolCall | None = None
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Terminal event for a request. Exactly one per admitted request."""

    request_id: RequestId
    finish_reason: FinishReason
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    tool_calls: tuple[ToolCall, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CycleProfile:
    """Per-cycle decode profile. A first-class output, not a debug facility.

    Invariant: producing this costs no extra host sync. Every field is either a
    host-side wall time or a counter the cycle already had to compute; anything
    that would need a device readback belongs in a sampled profile instead.
    """

    cycle: int
    n_sequences: int
    n_rows: int
    """Rows presented to the verify forward across all sequences. This is the
    number the whole decode design exists to raise: the expert gather runs at
    300 GB/s at one row and 549 at eight."""
    draft_ms: float
    verify_ms: float
    accept_ms: float
    sample_ms: float
    detok_ms: float
    ngram_wait_ms: float
    """Host time blocked on the n-gram row reader. Should be ~0 once prefetch
    is working; it was ~2.3 ms per forward in the overlay."""
    host_syncs: int
    """Must be 1 in a healthy cycle."""
    tokens_committed: int
    tokens_drafted: int
    wall_ms: float
