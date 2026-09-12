# Titan plan

Milestones, acceptance criteria, and the workstreams that can run in parallel.
Each workstream is a unit of work for one agent: it names its inputs, its
outputs, the interface it must satisfy, and a size (S = under a day, M = a few
days, L = a week or more).

Baselines to beat, all measured on the M5 Max 128 GB with the fan floor on,
oMLX 0.7.0.dev2 plus the round-2/round-4 overlay:

| measure | oMLX + overlay |
|---|---|
| cold 65k prefill | 1642 tok/s |
| decode after a long prefill (64k) | 62 tok/s (71 best) |
| short-context decode | 91 tok/s |
| 6-turn cache probe, median warm turn | 2.17 s |
| 8 concurrent short streams | 130 tok/s aggregate |

Rule for every A/B in every milestone: paired baseline, 45 s cooldown, minimum
of three, fan floor on. Unpaired numbers are not evidence.

## M1: single-sequence greedy parity

Titan generates, single sequence, greedy, no speculation, no prefix cache, using
the vendored model code and our kernels.

Acceptance: for 20 fixed prompts spanning short, 8k, 32k and 64k, Titan's greedy
token ids match oMLX's recorded ids exactly, to 256 tokens each.

Tests: `tests/parity/test_m1_greedy.py` against `tests/fixtures/m1-oMLX.jsonl`
(recorded from production, ids plus prompt hash plus the config that produced
them). `tests/kernels/test_*.py` for every op. `tests/test_dependency_rule.py`.

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W1.1 | Vendor and licence the model code | mlx-vlm qwen4_exp and qwen3_5 sources | `adapters/mlx/vendor/`, `VENDORED.md`, NOTICE | files import and instantiate under mlx 0.32.2; no oMLX imports | S |
| W1.2 | Checkpoint loader and weight plan | checkpoint path, safetensors headers | `adapters/mlx/checkpoint.py`, `loader.py` | plan covers every tensor exactly once; header reading only, no tensor materialised in the test | M |
| W1.3 | State handles and the backend skeleton | `core.ports.ModelBackend`, W1.1 | `adapters/mlx/state.py`, `backend.py`: open/close/length/prefill/decode | `state_length` never syncs; handles never reused; `prefill` returns logits only on request | M |
| W1.4 | Op registry and the ten ops | `engine/patches/` kernels, `core.ports.Op` | `kernels/registry.py` plus one file and one test per op | fast vs reference within declared tolerance over declared shapes; fallback counted | M |
| W1.5 | Packed n-gram reader | packed table, repack tool | `adapters/mlx/ngram.py` | `gather` bit-exact against the three-tensor layout; `prefetch` returns without blocking | M |
| W1.6 | Tokenizer and template adapters | checkpoint tokenizer and template | `adapters/mlx/tokenizer.py`, `template.py` | incremental detokenisation never emits a partial or revisable string; render is byte-stable across turns | S |
| W1.7 | Config, wiring, CLI | `config/schema.py` | `config/settings.py`, `wiring.py`, `titan` entry point | invalid config raises `ConfigError` naming the key; resolved config echoed | S |
| W1.8 | Parity harness and fixtures | production oMLX on 8083, `bench/` | `tests/fixtures/m1-oMLX.jsonl`, `tests/parity/` | records ids and config; read-only against production | S |
| W1.9 | Greedy sampler and single-sequence loop | core types | `engine/` minimal loop, no speculation | greedy path uses no RNG and no batch-dependent state | S |

Order: W1.1, W1.7 and W1.8 start immediately with no dependencies. W1.2 and
W1.6 need W1.1. W1.3 needs W1.1 and W1.2. W1.4 and W1.5 are independent of
everything except the port definitions, which already exist. W1.9 needs W1.3.
That is seven of nine startable on day one.

## M2: OpenAI API, streaming, reasoning, tools

Acceptance: opencode and the owner's harness work against Titan unchanged, with
only the base URL changed. Streaming deltas carry `reasoning_content`; tool
calls stream as indexed deltas in qwen3_coder format; `finish_reason` is
`tool_calls` when the turn ends in calls; `usage` reports `cached_tokens`.

Tests: `tests/api/test_openai_contract.py` (golden SSE transcripts, byte-compared
against oMLX's for the same request), `tests/api/test_tool_calling.py` (the
oMLX parser's own cases plus malformed and truncated markup), a live opencode
smoke run and the agentic suite's tool probes at short and 25k context with no
leaked markup.

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W2.1 | Wire models and routes | `api/models.py` | `api/openai.py`, `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics` | pydantic validation; bearer key from file, constant-time compare | S |
| W2.2 | SSE encoder | `TokenEvent`, `StreamEnd` | `api/sse.py` | frames byte-identical in shape to oMLX's for the same events | S |
| W2.3 | Tool-call parser port | oMLX `api/tool_calling.py` | `adapters/chat/qwen3_coder.py` | incremental; never leaks markup; unterminated construct is a `ParseError`, not prose | M |
| W2.4 | Reasoning channel | template markers | parser split of content and reasoning | reasoning text never appears in `content` and vice versa | S |
| W2.5 | Cancellation and backpressure | scheduler | disconnect handling | cancel closes state at a turn boundary, never mid-cycle | S |

W2.1 to W2.4 are parallel. W2.5 needs the scheduler from M1.

## M3: prefix cache with fine boundaries

Acceptance: multi-turn greedy output identical to a cold run of the same final
prompt (caching changes nothing); the 6-turn probe's median warm turn beats
2.17 s; warm turns recompute fewer than 9,000 tokens against the overlay's
11,617.

Tests: `tests/cache/test_restore_exact.py` (KV and GDN restore bit-identical),
`tests/cache/test_boundaries.py` (every grid multiple inside a suffix ends a
chunk; a boundary whose snapshot did not commit is dropped, not recorded),
`tests/cache/test_cost_model.py` (the fitted model predicts each probe turn
within 0.35 s), `bench/multiturn.py`.

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W3.1 | Two-tier store | `core.ports.KVStateStore` | `adapters/cache/store.py` | writer never stalls the loop beyond `max_stall_ms`; `pending_bytes` accurate | M |
| W3.2 | Prefix policy and chunk planner | store, `PrefixCache` | `adapters/cache/prefix.py` | lookup never returns a length it cannot restore; grid rule enforced | M |
| W3.3 | Snapshot export and import | backend state slots | `export_snapshot`, `import_snapshot` | round trip bit-identical; version tag rejects stale blobs | M |
| W3.4 | Snapshot at prompt end | W3.2, W3.3 | terminal-boundary path | costs one write and no extra forward pass | S |
| W3.5 | Multi-turn bench and cost model | `bench/` | `bench/multiturn.py`, fitted constants | reproduces the six probe turns in three arms | S |

W3.1 and W3.3 are parallel. W3.2 needs both. W3.4 needs W3.2. W3.5 is parallel
throughout.

## M4: MTP, batched verify, replay-free rollback

Staged, because chain parity has to hold before batching is allowed to touch it.

- M4a chain: depth-3 chain, one verify of four rows, single sequence. Output
  identical to M1 greedy for all 20 prompts.
- M4b lockstep: one padded row block for the whole batch. Output per sequence
  unchanged by batch composition or padding width.
- M4c rollback and acceptance: rejected drafts undone by `truncate_state`, no
  replay; acceptance computed in the graph; `host_syncs == 1` in every cycle.
- M4d n-gram prefetch: `ngram_wait_ms` near zero in the cycle profile.

Acceptance: decode 62 -> 75+ tok/s at 64k, 91 -> 100+ short, output identical
to greedy throughout.

Tests: `tests/parity/test_m4_chain.py` (speculative output == greedy output),
`tests/engine/test_rollback.py` (state after a rejection is bit-identical to
never having drafted), `tests/engine/test_lockstep.py` (per-sequence output
invariant to batch composition), `tests/engine/test_one_sync.py` (asserts
`CycleProfile.host_syncs == 1`), `bench/decode_bench.py`.

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W4.1 | MTP draft block and shortlist drafter | vendored MTP block, `Drafter` | `adapters/mlx/drafter.py` | proposes without syncing; empty tuple means one row | M |
| W4.2 | Lockstep verify forward | `ModelBackend.verify` | padded batched verify | padding cannot change a sequence's own output | L |
| W4.3 | In-graph acceptance | W4.2, `verify_accept` op | acceptance vector on device, one readback | greedy acceptance exact by construction | M |
| W4.4 | Replay-free rollback | state slots, staged snapshots | `truncate_state` on the decode path | no forward pass on the rollback path; below-snapshot truncation raises | M |
| W4.5 | Depth policy | `Verifier` | adaptive depth from acceptance and row budget | policy is pure; depth 3 default, ceiling from config | S |
| W4.6 | N-gram prefetch scheduling | `NgramReader`, cycle | prefetch before verify | `ngram_wait_ms` under 0.2 ms at steady state | M |
| W4.7 | Copy lane, gated off by default | prompt slices | optional drafter source | slices from the prompt only; never accepts past a stop token | M |

W4.1 and W4.6 are parallel with W4.2. W4.3 and W4.4 need W4.2. W4.5 and W4.7
are last and optional.

## M5: concurrency and the memory guard

Acceptance: 130 tok/s aggregate at 8 short streams, no regression; per-stream
gain at 68k with 2 to 4 streams at least matches the overlay's +22 to 28%; no
serialisation from the guard at any stream count; mean rows per cycle rises with
stream count in the profile.

Tests: `tests/engine/test_guard.py` (guard gates admission, never running work),
`bench/concurrency_sweep.py` at 1/2/4/8, `tests/engine/test_fairness.py` (no
sequence starves under a full queue).

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W5.1 | Admission and guard policy | `SchedulerPolicy` | guard, queue, 429s | pure policy, tested with no model | S |
| W5.2 | Batched sparse QSA decode | `qsa_sparse_decode` op | batched decode arm | within 1 ULP of dense; engages above one stream | M |
| W5.3 | Prefill serialisation under load | scheduler | one prefill at a time, never during a cycle | chunk stays 2048 under contention | S |
| W5.4 | Concurrency bench and profile assertions | `bench/` | sweep plus profile checks | rows per cycle reported per stream count | S |

All four are parallel.

## M6: production switch

Acceptance: Titan serves the tailnet through the launchd wrapper with the fan
floor, passes the full `bench/` suite at or above the overlay on every measure,
runs the agentic suite at 5/5 with clean tool calls, and survives 24 hours with
flat swap and no guard refusals under normal load.

Tests: `prod/verify-and-bench.sh` against Titan, a 24-hour soak with the
memory watchdog, and a documented rollback to the oMLX daemon in one command.

| # | workstream | inputs | outputs | contract | size |
|---|---|---|---|---|---|
| W6.1 | launchd wrapper and config file | `prod/` | `prod/run-titan.sh`, plist, config TOML | starts at boot, KeepAlive, log rotation, no sudo to restart | S |
| W6.2 | Bench parity across the suite | `bench/` | one report comparing both engines | same harness, paired, fan floor on | S |
| W6.3 | Rollback path | both daemons | documented one-command revert | oMLX daemon returns to 8083 unchanged | S |
| W6.4 | Soak and watchdog | `bench/memwatch.sh` | 24-hour report | swap flat, no leaks, no guard refusals | S |

## Cross-cutting

Ports are frozen at the start of M1 and change only by an amendment recorded in
DECISIONS.md, because they are the contract every parallel workstream is written
against. Ports 8083 and 8084 stay with production and the workbench; Titan
develops on 8085. Nothing in M1 to M5 touches the production daemon except
read-only recording for fixtures.
