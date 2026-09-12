# Titan decision records

Numbered, dated, and written so a later reader can tell what would have to
change for the decision to be wrong. Amending one means adding a record that
supersedes it, not editing the old text.

## D1. Titan is its own engine on MLX, not a fork and not a runtime

2026-09-12. Accepted, taken before this document.

MLX supplies arrays, quantised matmuls, Metal scheduling and
`mx.fast.metal_kernel`. Everything above that is Titan's. Not a fork of oMLX,
because the five things the overlay could not do (lockstep verify, replay-free
rollback, in-graph acceptance, n-gram prefetch, decoupled cache grids) are all
structural. Not from scratch, because MLX's quantised kernels and Metal
scheduling are the parts that are already at 76 to 87% of the hardware ceiling.

## D2. Clean architecture with a hard dependency rule

The core (types, ports, generation and scheduling logic) imports only the
standard library. Adapters implement Protocols; nothing in the core knows they
exist. Enforced by a test that walks the import graph.

The payoff is concrete rather than aesthetic: the scheduler and decode cycle
are testable with no model and no GPU, and any adapter can be swapped for a
reference implementation per run. The cost is a layer of indirection at every
boundary, which is real but small at the call rates involved (per cycle, not
per token).

## D3. State handles are opaque integers

A handle is an int key into a slot table the backend owns. Not an object,
because an object lets the engine reach into device state by accident. Ints are
cheap to log, impossible to dereference by mistake, and never reused, so a stale
handle is always an error.

Alternative rejected: passing cache lists around, as mlx-lm and oMLX do. That is
what makes rollback, forking and snapshotting show up in five places.

## D4. Asyncio for the API, one thread for the scheduler

FastAPI and uvicorn run asyncio on the main thread. The scheduler owns its own
thread and communicates through two queues. Nothing in the engine is async.

An awaited scheduler puts the decode loop at the mercy of event-loop fairness,
and a cycle that can yield between draft and verify has given up the property
the cycle exists for. oMLX arrived at the same shape from the other direction:
one `mlx-engine-<id>` worker thread running every forward, with the event loop
hopping into it via `run_in_executor` in bursts to avoid GIL ping-pong. Titan
copies the burst hand-off.

## D5. Rollback is a truncation, not a replay

`ModelBackend.truncate_state(state, length)` restores state to exactly `length`
tokens with no forward pass. The 12 QSA layers rewind by moving an offset. The
36 recurrent layers cannot be inverted, so the verify path stages a snapshot of
the recurrent state before it runs, and truncation restores it. Truncating below
the last snapshot raises rather than silently recomputing, because a hidden
recompute inside a decode cycle is a throughput cliff nobody can see.

oMLX does this differently: it keeps a pre-forward snapshot and replays the
accepted prefix through `_process_chunk`, all-or-nothing across layers. That
works and it costs a partial forward on every rejection, at an acceptance rate
around 80%. Restoring plus appending the accepted rows is the same information
without the replay.

## D6. Acceptance is computed in the graph, one host sync per cycle

Compare drafted ids against the target argmax, cumulative product, sum, and
concatenate the result with the target ids into one small array that crosses to
the host once. `CycleProfile.host_syncs` reports it, and a cycle that reads
anything but 1 is a bug the profile catches.

Lifted from oMLX's chain verify, which already does exactly this for both the
greedy and the rejection-sampling paths. Its legacy depth-1 path was never
converted and still syncs per token; Titan has no such path.

## D7. Lockstep batching, padded, never split

Every decoding sequence gets a row in the same verify forward, padded to a
common width. Ragged widths are padded, not decomposed.

oMLX decomposes: it extracts a single-row cache view per sequence, runs a
singleton cycle, and merges back, and its own multi-row path is off by default
because plain batched decode beats it at batch two and above. That matches our
measurement, where batched MTP lost to plain batching at every stream count.
Decomposition is why. Rows are the currency: the expert gather runs at 300 GB/s
at one row and 549 at eight.

The risk this takes on: padding waste when draft depths differ. Mitigated by the
depth policy handing out a common depth per cycle unless acceptance history
says otherwise, and measured directly as rows per cycle in the profile.

## D8. Block size 512, snapshot grid 2048, plus a snapshot at every prompt end

The two grids are separate. In oMLX they are the same thing: the snapshot grid
is the block grid, chunks are clamped to block boundaries so snapshots fire, and
the block size is forced up to at least 2048 for recurrent models precisely to
keep snapshot overhead down. That coupling is why a 6-turn conversation
recomputed 11,617 tokens it had already processed.

Block size 512 costs nothing: reconstruction is bytes-bound at about 2.77e-3 ms
per token, so 48 blocks of 512 cost what 12 of 2048 cost. Snapshots are what
cost, at around 110 MiB and 210 ms of writer time each, so they stay on the
2048 grid, with one exception: the end of every prompt. The last prefill chunk
already stops there, so that snapshot needs one write and no forward pass, and
it is the one that makes a follow-up turn resume where the previous turn
actually ended.

Two rules that come from the failed attempts. Every grid multiple strictly
inside a prefill suffix must itself end a chunk, or nothing stages a snapshot
there and the store chain truncates. And emission is restricted to the grid plus
the planner's own cuts, or a short contended chunk emits a snapshot every time.

## D9. A boundary that did not commit is dropped, never recorded

A lookup must never return a length it cannot restore. If a snapshot fails to
commit, the chain truncates at the previous good boundary. Lifted from oMLX,
where every cache failure path degrades to a shorter valid prefix rather than
raising, and it is the reason that stack is trustworthy despite its size.

## D10. The memory guard gates admission and nothing else

110 GB resident. It refuses new work; it never throttles or preempts running
work. A guard that throttles serialises concurrency, which held the overlay flat
at 76 tok/s aggregate across one, two and four streams. oMLX reaches the same
conclusion by a different route: its guard shrinks chunks and refuses
admissions, and its pressure-driven eviction hook is an empty function.

## D11. The n-gram reader is prefetched, never blocking

The table stays on SSD in the packed contiguous row layout: one pread per row
instead of three, 93,046 page reads per prefill chunk down to 31,855. A resident
copy is banned on 128 GB after two freezes and a kernel panic.

Scheduling is Titan's addition. The decode cycle queues the rows the draft
candidates need before it issues the verify forward, so the read overlaps GPU
work. A blocking read costs about 2.3 ms per forward, roughly 9% of a step, and
`ngram_wait_ms` in the cycle profile is how we know whether prefetch is working.

## D12. Kernels live behind an op registry, with a reference implementation each

Every op has a reference implementation in plain mlx ops, an optional
`mx.fast.metal_kernel` fast path, a declared tolerance and a declared shape
list. The registry chooses; the exactness test decides whether the fast path may
exist. Bit-identical or one bf16 ULP; anything looser is a different model.

This replaces monkey-patching a loaded module with env-gated overlays. The
overlay's approach worked but made the running kernel set invisible, and two
false starts came from a patch that silently did not route.

Also settled by measurement, recorded so nobody rebuilds them: judge a kernel in
situ on a paired A/B with a cooldown, because isolated microbenchmarks
overstated small-kernel launch cost by about 40x.

## D13. Configuration is one validated file, no environment variables

The overlay's configuration existed only in a launchd plist across twenty-odd
env vars, so a benchmark result could not name the configuration that produced
it. Titan reads one TOML file, validates it at startup, refuses to start on
anything invalid, and echoes the resolved values into the log and `/metrics`.
Two escape hatches, both recorded: `--set key=value` and `kernels.disabled`.

Note the failure this avoids. oMLX's `prefill_step_size = 2048`, the single most
consequential number in its scheduler, is a dataclass default with no path from
user settings, while `max_num_batched_tokens` is declared and never read.

## D14. Prefill is serialised and never shares a turn with decode

One sequence prefills at a time, and no prefill runs during a decode cycle.
Batched prefill loses the gathered sparse-attention arm and materialises a
134 MB mask per QSA layer at 65k, and a chunk shortened under contention costs
22% per token. Chunk size stays 2048: 512 is 22% worse per token and 4096 is
20% worse.

Revisit if a measurement shows a mixed prefill-and-decode forward beating this.
It is the obvious thing an engine that owns its own forward can do, and it is
not obviously right on this model.

## D15. The decode profile is a product feature

Every cycle emits one `CycleProfile`, always, with no extra host sync, because
every field is host wall time or a counter the cycle already had. Anything
needing a device readback goes in the sampled trace, which is off by default.
A decode regression has to be attributable without a rerun: fewer rows, worse
acceptance, more syncs, or n-gram wait.

## D16. Greedy is exactly reproducible and is the parity currency

Greedy uses no RNG and no batch-dependent state, and its output may not depend
on padding width, batch composition or how many tokens a cycle accepted. Every
parity milestone is greedy, token for token, against recorded oMLX output.
Sampled acceptance preserves the target distribution but not a particular
sample, so it is never compared token for token.

## D17. Model code is vendored, not depended on

The qwen4_exp and qwen3_5 sources are copied into `adapters/mlx/vendor/` with
their MIT headers and the upstream revision recorded. They will be edited: the
whole point of Titan is to change how the model's forward is scheduled. A pinned
dependency that has to be patched at import time is the arrangement being left
behind. Cost accepted: upstream fixes have to be merged by hand, and
`VENDORED.md` records what changed and why.

## D18. Ports are frozen at the start of M1

Parallel workstreams are written against the port definitions, so the ports
stop changing when the first one starts. An amendment is a new record here plus
a single commit that updates every implementation. This is a process decision,
and it is the one most likely to be tested by the first week.

## D19. Titan develops on port 8085

Production keeps 8083, the workbench keeps 8084, and config validation refuses
either. Nothing before M6 touches the production daemon except read-only
recording of parity fixtures.
