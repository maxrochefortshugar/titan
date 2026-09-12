# The engine core

2026-09-12. What `titan/engine/` contains, the contracts it holds the adapters
to, and the two seams it is built to be changed at. Everything here imports
`titan.core` and the standard library, nothing else, so all of it runs in unit
tests with no model, no GPU and no server.

| file | responsibility |
|---|---|
| `engine.py` | `TitanEngine`, the async iterator the HTTP layer drives |
| `scheduler.py` | `EngineLoop`, the single engine thread, and `FifoPolicy` |
| `admission.py` | guard, wait queue, chunk planner, `PortAdmitter` |
| `decode_cycle.py` | `PlainDecodeCycle`, `MTPDecodeCycle`, depth control, stops |

## 1. How the API layer wires it

```python
loop = EngineLoop(
    backend=mlx_backend,          # ModelBackend
    tokenizer=tokenizer,          # Tokenizer
    cycle=MTPDecodeCycle(
        backend=mlx_backend,
        tokenizer=tokenizer,
        drafter=mtp_drafter,      # Drafter, or None for the M1 plain loop
        ngram=ngram_reader,       # NgramReader, optional
        verifier=DepthController(max_depth=3, rows_budget=32),
        profiler=profiler,
    ),
    config=AdmissionConfig(...),  # flattened from TitanConfig
    cache=prefix_cache,           # PrefixCache, None until M3
    clock=clock,
    profiler=profiler,
)
engine = TitanEngine(loop)        # ThreadRunner by default
engine.start()                    # before uvicorn binds the port
deps = ChatDeps(config=..., engine=engine, renderer=..., tokenizer=...)
```

`TitanEngine` satisfies `titan.api.ports.Engine` structurally. It does not
import the API layer, because the dependency rule points inwards and the port
is a `typing.Protocol`, so shape is the whole contract.

`AdmissionConfig` is a flat dataclass of plain numbers rather than a slice of
`TitanConfig`, for the same reason: the engine may not import `titan.config`.
`wiring` builds one from the resolved TOML.

Swap `ThreadRunner` for `InlineRunner` and the loop is stepped from the event
loop instead of its own thread. The bench harness and the tests use it; nothing
else about the engine changes, and there is no test-only path through `step`.

## 2. The decode invariant, stated once

A decoding sequence always has exactly one token the backend has not consumed:
the token the previous cycle produced last. Call it the pending token.

    backend.state_length(state) == len(sequence.tokens) - 1

Everything follows from that. Prefill covers `len(prompt) - 1` tokens, not the
whole prompt, and the last prompt token is the first decode input. The reason is
the dependency rule rather than a performance trick: the core may not read
logits, so the only way to turn a position into a token id is to run it through
the backend's `verify`, which does its argmax in the graph. oMLX arrives at the
same split from the other direction, by handing only the final prompt token to
its generator.

Two consequences worth knowing before reading the code:

- prefill never asks for logits, on any chunk. `want_logits=False` everywhere.
  The head would be run for an answer nobody consumes, at 48 to 95 ms a chunk.
- what the prefix cache is offered is `tokens[:-1]`, and the scheduler checks
  the state length against that before it offers anything.

## 3. The row block

One verify row for one sequence is

    (pending, draft_1, ..., draft_k)

of width `k+1`. The forward consumes all `k+1` positions, acceptance keeps
`1 + n` of them where `n` is the number of confirmed drafts, and the bonus token
the forward produced at the first rejection point becomes the next cycle's
pending token. Committing `drafts[:n] + (bonus,)` is exactly what plain greedy
decoding would have produced.

The drafter proposes the continuation only. It is handed states and contexts, it
has no authoritative sequence id, and it does not know about the pending token,
which is scheduler state. Prepending it is done in `_BaseCycle.verify_candidates`
and nowhere else, so the row layout is decided in one place.

The plain M1 loop is the same thing at `k = 0`: one row per sequence, no
drafter, one host sync, argmax in the graph. It is a separate class rather than
a branch, because M1 parity is measured against it and a branch inside the
speculative path is a branch that can drift. The two share `commit`, which is
what makes the parity test between them mean something.

## 4. One cycle

    depth   -> DepthController.plan_depth from the rolling acceptance estimate,
               clamped by the row budget and by what is left of max_tokens
    draft   -> Drafter.propose, no host sync
    prefetch-> NgramReader.prefetch for the draft rows, before the forward
    verify  -> ModelBackend.verify, one padded row block, one host sync
    commit  -> accepted + bonus per sequence; stops; truncation
    emit    -> TokenEvent per sequence
    profile -> one CycleProfile, always, host_syncs straight from the backend

Rollback is not a step. The port's contract is that `verify` leaves each state
covering exactly `len(accepted) + 1` more tokens than it did on entry, so a
rejected draft leaves no trace and no forward pass runs. The engine's own call
to `truncate_state` is on the stop path, described below.

**The budget clamp.** Depth is clamped to `max_tokens - committed - 1` before
the drafter is asked. The verify always produces a bonus token, so a sequence
with one token of budget left must draft nothing. Clamping here rather than
discarding the overrun afterwards is what keeps the state consistent: a token
committed past the budget has to be truncated away, and a truncation inside the
verify block lands below the snapshot that block staged, which the backend is
right to refuse. oMLX clamps in the same place for the same reason.

**Adaptive depth.** `depth = round(mean_accepted) + 1`, clamped to
`[min_depth, max_depth]` and to `rows_budget // n_sequences - 1`. The `+1` makes
it self-correcting: a cycle that accepted everything has no evidence about the
depth above it, so it spends one row to find out, and a cycle that accepts
nothing falls to the floor within a window. The estimator is a window rather
than an EWMA because the question is "how many of the last N drafts stuck",
which a window answers without a time constant nobody can name. At 64k the
measured accepted median is 1, so the estimator has to fall as fast as the
context grows.

## 5. Stop conditions

EOS ids, stop strings and `max_tokens` are all resolved inside `commit`, one
committed token at a time, by `TextEmitter`.

A stop string is matched on text, so it can straddle any number of tokens and
can start in the middle of one. `TextEmitter` holds back the longest suffix that
could still become a stop string, and it records the accumulated text length
after every token so a character offset can be mapped back to a token count.
Two rules come out of that and they disagree on purpose:

- **text is cut at the stop string.** The client gets every character before it,
  which is what asking for a stop string means.
- **tokens are cut at the token boundary.** Half a token is not something a KV
  cache can hold, so a stop string starting mid-token drops that token whole.

A stop string can also span cycles: its first half may already be in the token
list, held back as text. The emitter's absolute survivor count is the authority,
so the trim can reach back past the current cycle. Nothing the client saw is
retracted, because held-back text was never sent; what changes is the prefix the
cache is offered.

**Truncation on stop.** When a stop discards tokens, the engine calls
`truncate_state(state, len(tokens) - 1)`. If the point is below the snapshot the
verify staged, the backend refuses, which is correct (D5: truncating below the
last snapshot is an error, not a slow path). The sequence is still right on the
wire. What is lost is the right to store this prefix, and the loop declines the
store rather than recording a length the cache could not restore (D9). Both
paths emit a structured event: `truncate_refused`, then `store_skipped`.

## 6. Admission

    plan: lookup -> plan_chunks -> estimate      (no allocation, refusable)
    start: guard -> open_state -> restore        (allocation, one turn)

**The guard gates admission and nothing else.** `MemoryGuard` has one question
and no method that shrinks, pauses, evicts or preempts. A test asserts those
methods are absent, because their absence is the design: a guard that throttles
running work serialises concurrency, and that held the overlay flat at 76 tok/s
aggregate across one, two and four streams.

**A request that cannot fit is skipped, not blocked on.** `WaitQueue.select`
walks the queue in arrival order and returns the first entry that fits. A
skipped entry keeps its place and its arrival order, so a 60k prompt waiting for
room does not stop the eight short ones behind it. oMLX breaks out of its
admission loop at the first blocked request, which is head-of-line blocking by
design. Per-entry `skips` are counted so starvation is visible in the metrics
rather than inferred from a latency graph.

**Chunk planning.** `plan_chunks(matched, total)` returns ends that are at most
`prefill_chunk_tokens` apart, never step over a snapshot-grid multiple, and sit
on the block grid unless they are the last one. Snapshots are staged at every
grid multiple and at the end of the prefill, which is the boundary that stops a
six-turn conversation recomputing 11,617 tokens it has already seen.

The cache adapter owns chunk planning because it owns the grids. It does not own
the invariant: `PortAdmitter` calls `PrefixCache.plan_chunks`, checks the result
against the grid rule, and replaces it with the local planner if it is wrong.
Three comparisons per chunk against a 2048-token forward pass.

## 7. The loop

One turn:

1. drain the command queue (submit, cancel), bounded
2. admit up to `max_admissions_per_turn` waiting requests
3. **either** one prefill chunk **or** one decode cycle, never both
4. retire everything DRAINING

Prefill and decode never share a turn (D14). Cancellation is a command, so a
state handle closes at a turn boundary and never inside a cycle: a cancelled
sequence cannot take the rest of the batch with it.

Refusal is asynchronous. A full queue or a guard refusal arrives as a
`StreamEnd` carrying `FinishReason.ERROR`, not as an exception on the caller's
thread, because the loop is the single owner of the queue and the response
headers are already on the wire by then.

`resident_gb` asks the backend when the backend offers it, and otherwise models
it as the weights plus every live sequence's estimate. The measurement that
matters -- `mx.get_active_memory` plus the Mach footprint -- is MLX-thread-only
and belongs in the adapter.

## 8. Two seams

### Lockstep batched verify (M4b)

`MTPDecodeCycle` is already written for a batch: it plans a depth per sequence,
builds one row per sequence, and makes exactly one `verify` call for the whole
batch. What is not there yet is ragged width. `plan_depth` hands out a uniform
depth, so the block is rectangular by construction and the backend's padding
never engages.

Taking the seam is a change in two places and nowhere else:

1. `DepthController.plan_depth` may return different depths per sequence, from
   per-sequence acceptance rather than the batch mean;
2. the backend pads short rows to the common width, which its `verify` contract
   already promises ("padding may not affect their own outputs").

Nothing in `commit` knows the width. It reads `len(outcome.accepted)` and the
sequence's own proposal, which is what keeps the seam local. The
`verify_accept` op's reference already handles `R > 1` rows and is tested at
`R = 8`.

### Mixed prefill and decode batches (D14, revisit)

`EngineLoop.step` chooses between a prefill chunk and a decode cycle with one
`if`, marked `MIXED BATCH SEAM`. One call carrying both rows would replace that
branch and nothing above it. It is the obvious thing an engine that owns its
forward can do, and it is not obviously right on this model: batched prefill
loses the gathered sparse-attention arm and materialises a 134 MB mask per QSA
layer at 65k. Revisit on a measurement, not on principle.

## 9. `verify_accept`

`titan/kernels/verify_accept.py`. Given the target logits over one verify row
block `[R, k+1, V]` and the drafted ids `[R, k]`, it returns **one** `int32`
array of length `2R`: the accepted counts, then the bonus tokens. That array is
what the single host sync carries, and `split_host` defines the packing order in
one place.

Greedy is argmax, compare, cumulative product, sum, gather. Stochastic is exact
rejection sampling with the residual computed for *every* position, not only the
one that rejected, because a residual computed after the host learned where the
rejection was would need a second sync. Sampling is inverse-CDF against caller-
supplied uniforms, so the whole op is deterministic given its inputs and is
tested against a numpy transcription of the same arithmetic.

The fast path is currently the reference. The seam exists so that landing the
fused `mx.fast.metal_kernel` is a registry selection rather than an edit to the
decode cycle, and the exactness test asserts the pair agrees either way.

**Registry entry, not yet made.** `build_registry` collects op modules in
`registry.py`'s `_op_modules`, and adding `verify_accept` there is two lines: the
import and the tuple entry. It was left out to avoid a concurrent edit to a file
another workstream owns. Nothing in the engine needs it -- the engine holds no
mlx and the backend calls the op directly -- but `ARCHITECTURE.md` section 6
lists `verify_accept` among the ops the engine expects at M1, so the two lines
should land with the next registry change.

## 10. Tests

`tests/engine/` runs against fakes for every port: a `FakeBackend` with a
deterministic next-token function and real snapshot and truncation rules, a
`FakeTokenizer` with a piece table, a `FakeCache`, a `ScriptedDrafter` and a
`FakeClock`.

| file | what it pins down |
|---|---|
| `test_greedy_loop.py` | the plain loop reproduces the backend's argmax stream; batch composition changes nothing; the state invariant |
| `test_mtp_parity.py` | lossless speculation over every acceptance pattern, depths 1 to 8, random draft/target pairs; state identical to never having drafted; one host sync a cycle; depth control |
| `test_stop_conditions.py` | EOS, stop strings across emits and inside a token, both budgets, the emitter alone |
| `test_admission.py` | the grid rule over a sweep of prompt and match lengths, the guard, skip-not-block, cache plan validation |
| `test_scheduler.py` | turn ordering, chunked prefill and snapshot positions, retirement, cancellation, usage, refusals |
| `test_engine.py` | the async surface: one StreamEnd, cached tokens, concurrent streams, disconnect frees state |

`tests/kernels/test_verify_accept.py` covers k = 1 to 8, full accept, zero
accept, rejection at every position, row independence, and the stochastic path
against numpy.
