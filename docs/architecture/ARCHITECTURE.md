# Titan architecture

2026-09-12. Titan is a serving engine written in Python on MLX, for one model
family on one machine: Qwen3.8-Flash-Next (`qwen4_exp`, 125B-A6B MoE, 48 layers
= 36 Gated DeltaNet + 12 Qwen Sparse Attention, hyper-connections, a 51B n-gram
table on SSD, an MTP draft block). It is not a fork of oMLX and not a runtime
from scratch. MLX supplies arrays, quantised matmuls, Metal scheduling and
`mx.fast.metal_kernel`; everything above that is ours.

Production today is oMLX 0.7.0.dev2 plus our overlay. It keeps serving until
Titan beats it on `bench/`.

## 1. Layers and the dependency rule

```
                +-------------------------------------------+
                |                titan.api                  |  fastapi, sse
                +-------------------------------------------+
                       |                          ^
                       v (Request)                | (TokenEvent)
    +--------------------------------------------------------------+
    |                        titan.engine                           |
    |   admission | scheduler loop | decode cycle                   |
    +--------------------------------------------------------------+
                       |  ports only
                       v
    +--------------------------------------------------------------+
    |                         titan.core                            |
    |   types (Request, SequenceState, TokenEvent, ...)             |
    |   ports (ModelBackend, PrefixCache, Tokenizer, Drafter, ...)  |
    |   errors                          imports: stdlib only        |
    +--------------------------------------------------------------+
                       ^                    ^                ^
                       |                    |                |
    +------------------+--+  +--------------+---+  +---------+--------+
    | adapters.mlx        |  | adapters.cache   |  | kernels          |
    | backend, state,     |  | store (RAM+SSD), |  | op registry,     |
    | tokenizer, template,|  | prefix policy    |  | reference + fast |
    | ngram reader        |  |                  |  |                  |
    +---------------------+  +------------------+  +------------------+

                    titan.config.wiring builds all of it, once
```

The rule: imports point inwards. `titan.core` imports nothing but the standard
library. `titan.engine` imports `titan.core`. Adapters import `titan.core` and
implement its Protocols. `titan.api` imports core and engine. Only
`titan.config.wiring` sees more than one outer layer, because building the
object graph is the one job that needs to.

Two consequences worth the discipline. The scheduler and the decode cycle run
in unit tests against fake adapters, with no model and no GPU, at full speed.
And a kernel, a store or the whole backend can be swapped for a reference
implementation per run, which is what makes bisecting a bad kernel a config
change.

`tests/test_dependency_rule.py` walks the import graph and fails on a violation.

## 2. Module map

| module | responsibility |
|---|---|
| `core/types.py` | Request, SamplingParams, StopCondition, SequenceState, PrefixMatch, PrefillChunk, DraftCandidate, VerifyOutcome, TokenEvent, StreamEnd, ToolCall, FinishReason, CycleProfile |
| `core/ports.py` | ModelBackend, KVStateStore, PrefixCache, Tokenizer, TemplateRenderer, ToolCallParser, Drafter, Verifier, NgramReader, Clock, Profiler, Op, OpRegistry |
| `core/errors.py` | TitanError tree: Config, Capacity, MemoryGuard, State, Snapshot, Backend, Kernel, Template, Parse |
| `engine/admission.py` | render, encode, prefix lookup, guard, open state, plan chunks |
| `engine/scheduler.py` | the single loop, policy split out for testing, LoopStats |
| `engine/decode_cycle.py` | draft, prefetch, verify, commit, emit, profile |
| `adapters/mlx/backend.py` | ModelBackend over the vendored qwen4_exp code |
| `adapters/mlx/state.py` | what a StateHandle is: slot table, KV, recurrent state, staged snapshots |
| `adapters/mlx/ngram.py` | packed SSD row reader with prefetch |
| `adapters/mlx/tokenizer.py`, `template.py` | tokenisation, incremental detokenisation, chat template |
| `adapters/cache/store.py` | KVStateStore: 4 GB RAM tier, SSD tier, background writer, bounded stall |
| `adapters/cache/prefix.py` | PrefixCache policy: block hashing, two grids, chunk planning |
| `api/models.py`, `openai.py`, `sse.py` | wire models, routes, streaming |
| `kernels/registry.py` + one file per op | reference and fast implementations, exactness tests |
| `observability/profiler.py` | ring buffer of CycleProfile, counters, sampled traces |
| `config/schema.py`, `wiring.py` | validated config, composition root |

## 3. Data flow of a request

```
POST /v1/chat/completions
  |
  | api: validate, render template, tokenise, build Request        (asyncio)
  v
submit() -> queue -------------------------------------------------+
                                                                   |
  scheduler thread, one turn of the loop:                          v
  1. admission     PrefixCache.lookup(tokens) -> matched=N
                   guard check on resident GB, else 429
                   ModelBackend.open_state -> StateHandle
                   PrefixCache.restore(match, state)   ~2.8 us/tok
                   plan_chunks(N, total) -> chunk ends
  2. prefill       one chunk per turn, 2048 tokens, serialised
                   last chunk wants logits and stages a snapshot
  3. decode cycle  Drafter.propose (no sync)
                   NgramReader.prefetch(draft rows)
                   ModelBackend.verify(states, drafts)  <- one host sync
                   commit accepted + bonus, truncate on reject
                   Tokenizer.decode_incremental -> ToolCallParser.feed
  4. emit          TokenEvent per sequence -> out queue -> SSE
  5. finish        stop matched: truncate past stop, StreamEnd
                   PrefixCache.store(tokens, state, boundaries)
                   close_state
```

Prefill and decode never share a turn. Batched prefill loses the gathered
sparse-attention arm and builds a 134 MB mask per QSA layer at 65k, and a
contended chunk that drops to 512 tokens costs 22% per token.

## 4. Memory model

Resident: about 78 GB of weights, pinned for the process lifetime. Titan does
not stream experts and does not hold the n-gram table in RAM; 32 GB of table on
top of the model produced two freezes and a kernel panic, and the streamed
packed layout gets the same numbers at zero resident cost.

Per sequence, the backend owns one slot: 12 QSA KV caches, 36 recurrent GDN
states, hyper-connection carriers, conv windows, plus up to a few staged
snapshots. The engine owns none of it and reaches it only through the handle.

Snapshot budget. A snapshot of the recurrent state is around 110 MiB. Three
places hold them: staged in RAM inside the slot (bounded, two plus whatever the
current prompt staged, oldest evicted), the 4 GB hot store tier, and SSD. The
store reports `pending_bytes`, and the chunk planner drops an optional fine
snapshot rather than let a store stall the loop; the bound is 50 ms.

The guard is an admission gate at 110 GB resident and nothing else. A guard that
throttles running work serialises concurrency, which held the overlay flat at 76
tok/s aggregate for one, two and four streams until it was found.

## 5. Concurrency model

One process, one model, one scheduler thread. The API runs asyncio on the main
thread and talks to the scheduler through two queues. Nothing in the engine is
async: a decode cycle that can yield between draft and verify has given up the
property the cycle exists for.

```
main thread (asyncio)        scheduler thread            background
  fastapi/uvicorn    ---->   admission queue
  SSE encoders       <----   event queue
                             loop: prefill | decode cycle
                                    |            |
                                    v            v
                              MLX command queue (GPU)
                                                          store writer thread
                                                          n-gram reader pool
```

On the CPU thread: templating, tokenisation, detokenisation, tool parsing,
admission, chunk planning, acceptance bookkeeping, profiling. On the GPU queue:
everything the backend submits. Off both: store writes and n-gram preads.

Lockstep is the batching rule. Every decoding sequence gets a row in the same
verify forward, padded to a common width, never split. Rows are the currency:
the expert gather runs at 300 GB/s at one row and 549 at eight, and the
overlay's batched MTP lost to plain batching precisely because it split.

## 6. How kernels plug in

An op is a name, a reference implementation in plain mlx ops, an optional
`mx.fast.metal_kernel` fast path, a tolerance and a shape list. The registry is
built once from config and passed to the backend; the vendored model code asks
the registry for an op by name and uses the stock path when it gets nothing.
No monkey-patching, no import-time rebinding, no env var per kernel.

```
    call site --> OpRegistry.resolve("gdn_norm_gate")
                     |            |
              config disabled?    supports(shape)?
                     |                |         \
                  reference        fast       reference + count
```

Every op keeps `tests/kernels/test_<op>.py`, which runs the fast path against
the reference over the op's declared shapes at its declared tolerance. The house
standard is bit-identical or one bf16 ULP; anything looser is a different model
and is not deployed. Every op keeps a microbenchmark, and a kernel is judged in
situ on a paired A/B with a cooldown, because isolated timing overstated small
kernel launch cost by roughly 40x once already.

The ops the engine expects at M1: `ngram_gather`, `rms_norm_grouped`,
`gdn_norm_gate`, `gdn_scan_chunked`, `moe_weighted_sum`, `moe_gather_gate_up`,
`hyper_connection_block`, `qsa_sparse_decode`, `mtp_shortlist`, `verify_accept`.

## 7. Observability

Three layers, each cheaper than the one below.

1. Counters and the per-cycle decode profile. Always on. Every cycle emits one
   `CycleProfile` into a ring: rows, draft/verify/accept/sample/detok times,
   n-gram wait, host syncs, tokens drafted and committed. No field needs a
   device readback, so the profile costs nothing and `host_syncs` reading
   anything but 1 is a bug the profile itself catches.
2. Structured events: admission, prefix hit and miss with lengths, chunk
   boundaries, snapshot writes with their backlog, guard refusals, kernel
   fallbacks. One machine-readable line each, no sampling.
3. Sampled traces, off by default. Every Nth cycle, a per-stage breakdown that
   does sync the device.

`GET /metrics` serves layers 1 and 2 and the resolved config, so every benchmark
result can name the configuration it came from.

## 8. Configuration

One TOML file, validated at startup, echoed into the log and `/metrics`. No
environment variables. The overlay's twenty-odd env vars meant the running
configuration existed only in a launchd plist. Two escape hatches, both
recorded in the resolved config: `--set path.key=value`, and `kernels.disabled`
for bisecting.

Validation refuses to start on: a snapshot grid that is not a multiple of the
block size, a prefill chunk that is not, a guard with no headroom over the
weights, port 8083 or 8084, an unknown name in `kernels.disabled`, a missing or
unpacked n-gram table.

## 9. What is lifted from where

| taken | from | licence | how it is used |
|---|---|---|---|
| qwen4_exp and qwen3_5 model code | mlx-vlm (`.../vendor/mlx_vlm/models/qwen4_exp/`, upstream `78b96eb5`) | MIT | vendored under `adapters/mlx/vendor/`, headers intact, revision recorded, edited freely |
| cache classes, sampling, the qwen3_coder parser core | mlx-lm (`ab1806e8`; `tool_parsers/qwen3_coder.py`) | MIT | the parser's grammar handling is the base we port; its `ast.literal_eval` coercion and `$`-anchored function regex are replaced |
| paged prefix cache, GDN boundary snapshots, SSD and hot tiers | oMLX `cache/prefix_cache.py` (`BlockAwarePrefixCache`:172), `paged_ssd_cache.py`, `boundary_snapshot_store.py` (`save`:220, `commit_gdn_checkpoint_file`:2369), `prefill_boundaries.py` | Apache-2.0 | design reimplemented with the two grids decoupled; chain block hashing, the write-behind tier and the truncate-rather-than-lie rule are lifted intact |
| MTP chain: depth-k draft, one verify of k+1 rows, one `.tolist()` for acceptance | oMLX `patches/mlx_lm_mtp/batch_generator.py` (`_chain_next_drafts`:2303, `_run_verify_cycle_chain`:2908, sync at :2988) | Apache-2.0 | the cycle shape and the in-graph cumprod acceptance are lifted; lockstep batching and replay-free rollback are ours |
| tool-call hardening: payload bounding, non-raising coercion, the streaming suppression filter | oMLX `api/tool_calling.py` (`parse_tool_calls`:1542, `ToolCallStreamFilter`:1904, `_find_marker_span_end`:497), `api/thinking.py` | Apache-2.0 | ported to the `ToolCallParser` port; NOTICE entry |
| chunked prefill and the memory guard | oMLX `scheduler.py` (`_contended_prefill_cap`:5177, `_guard_prefill_chunk`:4258, `_current_usage_bytes`:4715) | Apache-2.0 | chunk sizing and guard semantics reused; the guard stays admission-only, and the chunk size becomes real config |
| our kernels | `engine/patches/` | MIT (ours) | moved behind the op registry |
| mlx-serve | no licence | none | ideas only, nothing read into the code |

Apache-2.0 obligations: a `NOTICE` file naming oMLX, the files derived from it,
and the changes made. MIT-derived vendored code keeps its own header. Titan
itself stays MIT.

## 10. What this buys over the overlay

The overlay could not reach five things, and each is a structural property here
rather than a patch: lockstep batched verify (one padded row block for the whole
batch), replay-free rollback (`truncate_state` against a staged snapshot),
in-graph acceptance with one host sync, n-gram prefetch for draft rows off the
critical path, and a prefix cache whose block size and snapshot grid are
separate, with a snapshot at the end of every prompt.
