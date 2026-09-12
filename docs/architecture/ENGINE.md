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
        drafter=MTPDrafter(backend, chain="head_output"),   # or None
        ngram=ngram_reader,       # NgramReader, optional
        verifier=ExpectedValueDepthController(max_depth=6, rows_budget=32),
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

    depth   -> the depth policy, per sequence, from its acceptance curve and
               the cost table, clamped by the row budget and by max_tokens
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

### The drafter

`titan/adapters/mlx/drafter.py`. The Lightning MTP head is one sparse-attention
layer that predicts token *t+2* from the backbone's residual stream at position
*t* and the embedding of token *t+1*. One proposal is:

    fold   -> one head call over every token the last cycle committed, against
              the persistent head cache, logits kept for the last position only
    chain  -> depth-1 further calls, each re-entering the head on its own
              output hidden, against a clone of the head cache
    gate   -> drop the tail past the first draft below p_min
    read   -> one mx.async_eval and one .tolist() for the whole batch

**The fold is the whole alignment story.** The obvious implementation folds one
position, the pending token and the hidden beside it. Then the head's KV never
sees the tokens the cycle accepted, its positions drift from the sequence's by
the accepted count every cycle, and acceptance decays with nothing in the
profile naming why. So the fold covers the entire committed run: the verify
left a hidden state for every column of its block, the accepted prefix of that
block is exactly the run, and folding all of it appends one head entry per
committed token. The head cache stays a committed-only mirror of the sequence.

That is also why rollback of the head is a non-question. Steps 2..k run against
a clone (`QSAKVCache.extract`, one layer, two kv heads, one row), so nothing a
rejected draft produced is ever history. The drafter checks the head's offset
against its own count of folded tokens each cycle and drops the head cache when
they disagree, which happens when the stop path's `truncate_state` trims the
head by the trunk's delta. Dropping it costs acceptance for a cycle and cannot
cost correctness: only `verify` decides what is committed.

**Two chain forms, one switch.** `speculation.mtp_chain`. `head_output` is the
default and is vLLM's form: step *i+1* is fed the head's own output hidden
after the head's final norm, which in this architecture is
`hyper_connection_mixer`, lifted back to hyper-connection width the same way
the trunk lifts a token embedding, and `pre_fc_norm_hidden` is re-applied on
the way in. `omlx` re-enters on the head layer's pre-mixer streams instead.
EAGLE 3.1 credits the first form with long-context acceptance, so the two are
measured against each other rather than argued about. On this checkpoint the
`omlx` form wins, twice: 2.24 and 2.26 accepted tokens a cycle against 2.12 and
2.12, 75.5% acceptance against 71.4%. `bench/decode/README.md` holds the table
and the caveat.

**One sync.** Every sequence's chain lands in one float32 array -- draft ids are
exact below 2^24 and the vocabulary is 248,320 -- and crosses to the host once
per cycle, in `_sync_and_read`. The chain itself never syncs: step *i+1* is fed
the previous step's argmax as a device array.

### Depth by expected value

`ExpectedValueDepthController` replaces `round(mean_accepted) + 1`:

    k* = argmax_k  E[committed | k] / cycle_ms(k + 1)

`E` comes from `PositionAcceptance`, a decayed estimate of P(draft *i* accepted
| the chain reached *i*), kept **per sequence**: a 64k session and a 200-token
one share a process and a machine, not an acceptance curve, and averaging them
gives both the wrong depth. Positions are not alike -- DSpark measured the
seventh drafted token surviving under 10% of the time against over 70% for the
first -- so a single mean cannot express the thing the policy is choosing over.

`cycle_ms` comes from `CycleCostModel`, a decayed least-squares line in row
width, seeded with the measured 16.2/19.6/27.6/36.2 ms at widths 1/2/4/6 and
refitted from every batch-of-one `CycleProfile`. A line rather than a table
because the policy has to price widths it has never run, and because the shape
really is a line here: the intercept is the host graph build and the slope is
the marginal row. Cycles costing more than four times their prediction are
dropped rather than smoothed; that is a prefill sharing the thread, not a row.

Neither side reads the drafter's own confidence, which is the difference
between this and the confidence-gated depth the overlay measured at -6%. That
gate drafted *deeper* when the head was sure; this policy's usual effect is to
draft shallower, and at an accepted median of 1 it chooses depth 0 and stops
paying for speculation at all. `mtp_depth_min` is the fixed policy's floor;
the adaptive policy's floor is zero, because not drafting has to be one of the
options it can price.

**The probability floor.** `speculation.draft_p_min`, llama.cpp's gate: stop
the chain past a draft whose top-token probability is below the floor. Measured
there at +20.4% with acceptance *falling* and mean run length rising, which is
the right metric because run length is what the cycle spends rows on. It cannot
save the draft compute -- a device-side early exit would mean a ragged row
block, which fights lockstep verify -- but it saves the verify column, which is
the expensive half. Zero disables it and also skips the softmax that reads it.

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

What the loop does when a port call raises is section 11.

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
| `test_engine_failures.py` | a port that raises outside the taxonomy: prefill, decode, tokenizer, cache store; blast radius, health, the probe, the watchdog, shutdown and cancellation mid-failure |

`tests/api/test_engine_errors.py` covers the same failures from the client's
side: the SSE error frame, the 500 body, and the 503s.

`tests/kernels/test_verify_accept.py` covers k = 1 to 8, full accept, zero
accept, rejection at every position, row independence, and the stochastic path
against numpy.

## 11. Failure handling

### The bug this section is written against

A fake backend's `prefill` raised `TypeError: prefill() got an unexpected
keyword argument 'next_token'`, and the request never finished. Not slowly: at
all. Two decode rounds lost time to it before anybody pinned it down, so the
diagnosis is written here rather than in a commit message.

`Scheduler._run_prefill` caught `TitanError` and nothing else. A `TypeError` is
not a `TitanError`, so it left `_run_prefill`, left `step`, and left
`run_forever`. Under the production `ThreadRunner` that is the end of the loop
thread: Python's default thread hook prints the traceback to stderr and the
thread exits. Under `InlineRunner` it is quieter and worse, because the
exception lands in an asyncio task nobody awaits and nothing is printed until
the garbage collector gets around to it.

Either way the sequence is still on `_live`, still `PREFILLING`, and nothing
will ever retire it. `TitanEngine.generate` is parked on `await events.get()`
against a queue whose only producer was the thread that just died. The client
holds an open connection to a process that is up, healthy-looking and never
going to answer.

So: **the hang was the exception escaping into the runner, not the loop failing
to advance the sequence out of `PREFILLING`.** The stuck phase is a symptom of
the dead thread, and a fix that only re-drove the phase would have left the
next unguarded port call to do the same thing. The same hole existed at decode,
verify, draft, cache store, tokenizer and codec, all for the same reason: the
error taxonomy in `titan/core/errors.py` says what a fault *means*, and it had
quietly been read as saying which faults the loop *survives*. It never said
that.

### The contract

Every call the loop thread makes out of the engine goes through
`EngineLoop._guarded(op, sequences, fn, ...)`, which returns the call's value or
the `FAILED` sentinel. `TitanError` and a bare `RuntimeError` are treated
identically: the taxonomy is for the message, not for the blast radius.

When a guarded call raises:

1. the sequences named in `sequences` finish with `FinishReason.ERROR` and an
   error string of the form `"{op} failed: {ExceptionType}: {text}"`, so the
   original type and message survive all the way to the client;
2. retirement releases their state handles, their store sessions and their
   cache leases on the ordinary path, because a failing sequence retires the
   same way a finished one does;
3. the failure is counted -- `engine.port_failures`, `engine.port_failures.{op}`
   and a `port_failed` profiler event carrying the op, the exception type and
   the sequence count;
4. every other sequence keeps its seat and the turn carries on.

A batched call fails its batch and only its batch. `cycle.run` is one guarded
call for the whole forward, so a fault anywhere under it -- drafter, n-gram
index, verify, sampler, detokeniser, codec -- fails the sequences that shared
that forward. A batch of one fails one request; a batch of eight fails eight.
That is the price of sharing a forward, and it is stated here rather than
discovered.

Two calls are deliberately outside the blast-radius rule. A cache store that
raises costs the sequence its cache entry and not its answer, so the request
finishes on its real reason and the failure is counted with no sequence named.
Retirement's detokeniser flush is the same: the answer is already streamed.

`run_forever` also wraps `step` itself. Nothing inside `step` is supposed to
reach `on_step_error` -- every port call is guarded and the engine's own code is
not allowed to raise -- but the cost of being wrong about that is a dead thread
and a client on a queue nobody feeds, and the cost of the belt is one `try`.
`on_step_error` marks the engine unhealthy, fails every live sequence with the
same message, retires them and returns.

### Health

`EngineLoop.health()` returns an `EngineHealth`. `healthy` is False only for a
fault the engine cannot serve through:

- a **warm-up probe** that failed twice in a row. After any port failure the
  loop calls `backend.warmup()`, the cheapest call that still touches the
  device, and uses it as a probe. One failure is a bad moment; two with no
  success between them is a backend that is not coming back;
- a **device error**, which is marked on the first occurrence. MLX exports no
  exception type to catch, so a Metal fault arrives as a `RuntimeError` with the
  driver's text in it. `_is_device_error` matches on the exception's own module
  (`mlx`, `metal`) and on a short marker list. The markers are narrow on
  purpose: a false positive costs an engine that refuses work until it is
  restarted;
- the **watchdog** catching a step over its budget;
- a shutdown that timed out with a step still running.

A request that dies on its own bad luck leaves the engine healthy and takes only
itself down.

An unhealthy engine stops admitting. New submits are answered immediately with
an `ERROR` `StreamEnd` naming the reason, and the first turn afterwards drains
the wait queue the same way: a request queued a moment before the fault is owed
the same answer as one that arrives a moment after it. Draining happens on the
loop thread rather than inside `mark_unhealthy`, which the watchdog thread also
calls. `mark_healthy()` clears the flag and exists for an operator, not for the
loop.

### The watchdog

A step that never returns cannot report itself, so `step` records
`_step_started` against `time.monotonic` and a second daemon thread polls
`check_step_deadline` every `watchdog_poll_s` (default 0.5 s). Past
`step_budget_s` (default 120 s) it fires once per step and:

- captures the loop thread's stack with `sys._current_frames` and logs it. Not
  `faulthandler.dump_traceback`: that writes every thread to a file descriptor,
  and what is wanted is one thread as a string a log line and a test can both
  hold. The captured stack stays on `last_stall_stack`;
- emits `loop_stalled` with the elapsed time and the budget, counts
  `engine.loop_stalls`, and marks the engine unhealthy.

It does not kill the thread. There is no safe way to interrupt a thread inside a
Metal call, and an engine that kills its own loop is the hang this whole section
exists to prevent. The budget is wall-clock and reads a real clock even when the
loop was built with an injected one: a test clock that ticks per read would make
the budget a count of reads rather than a duration. A budget of zero disables
the watchdog.

### Shutdown

`shutdown(drain_timeout_s)` is bounded by a wall clock on every path, including
the one where a step is wedged inside a port call. When the loop owns a thread
of its own, the caller waits for it to notice rather than stepping in parallel
with it. If the drain window closes with the thread still inside a step, every
live sequence gets an `ABORT` `StreamEnd` and every queued one gets the same,
and the loop thread is left to whatever it is doing: it is a daemon, so it
cannot hold the process open, and its state handles are not closed underneath
it, because closing a handle under a thread still reading it trades a hang for a
crash. `ThreadRunner.stop` spends one budget across the drain and the join
rather than one each.

`InlineRunner.stop` calls the same `shutdown` before it cancels its driving
task. Cancelling without it would leave the loop holding sequences nobody will
ever step again, which is the same hang wearing a tidier name.

### The asyncio bridge

`TitanEngine.generate` waits on the queue in `liveness_poll_s` slices (default
1 s) and asks the runner whether it is still turning. A runner that answers
False releases the waiter with an `ERROR` `StreamEnd` rather than leaving it on
a queue nobody feeds. Belt and braces now that the loop survives its own faults,
and cheap enough to keep for the faults it cannot survive.

The poll is `asyncio.timeout`, not `asyncio.wait_for`. They look
interchangeable and are not: on 3.11 `wait_for` returns the inner result when
the outer task is cancelled at the moment the inner one has already completed,
so a client disconnecting during a token delivery would have its cancellation
swallowed and the stream would run on with nobody reading it. That is a second
hang, discovered while fixing the first, and it is why
`test_cancelling_the_consuming_task_frees_the_sequence` exists.

Anything that goes wrong before the first yield is allowed to raise, because the
API layer still has a status code to set: `EngineUnhealthyError` if the loop
reports itself unhealthy in the gap between the readiness check and the submit
(the API maps it to 503), `EngineError` if the submit itself fails (500). After
the first yield there is no status code left and every failure is a `StreamEnd`.

### What the client sees

| failure | streaming | non-streaming |
|---|---|---|
| a port call raised (prefill, decode, verify, tokenizer) | the partial text, then an SSE error frame carrying `"{op} failed: {Type}: {text}"`, then a normal finish chunk and `[DONE]` | HTTP 500, OpenAI error envelope, same message |
| the cache store raised | nothing: the turn finishes on its real reason | nothing |
| the engine raised instead of yielding | the same error frame and a closed stream | HTTP 500 with the exception type and text |
| the engine is unhealthy | HTTP 503 before anything is rendered | HTTP 503 |
| already queued when the engine went unhealthy | an error frame naming the reason | HTTP 500 with the reason |
| shutdown while in flight | an abort finish and a closed stream | HTTP 200 with whatever was generated |

`GET /health` is liveness and readiness at once, because here they are the same
question: a process that is up and cannot serve should not be sent work. It
answers 503 with `status: "unhealthy"`, the reason, and the failure counters
when the engine reports itself unhealthy, and 200 otherwise. An engine that
publishes no health report at all is served as healthy, because the `Engine`
protocol the API layer depends on is one method wide and stays that way.
`GET /metrics` carries `port_failures`, `failed_requests`, `healthy` and
`unhealthy_reason` on the `loop` object.
