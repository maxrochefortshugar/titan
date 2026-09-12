# Configuration

One TOML file. Titan reads it once at startup, validates every field, refuses to
start on anything invalid, and echoes the resolved values into the log and into
`GET /metrics` so that any measurement can name the configuration it came from.

There are no environment variables. The single exception is `TITAN_CONFIG`,
which holds a path to the file and nothing else, and it is read in exactly one
function, `titan.config.settings.resolve_config_path`. A test walks the AST of
that module and fails if a second read appears.

Two escape hatches, both recorded in the resolved config so a run started with
one is never mistaken for a stock run:

- `--set path.key=value` on the command line, repeatable.
- `kernels.disabled`, for bisecting a bad kernel without a code change.

## Where the definition lives

`titan/config/schema.py` is the definition: frozen dataclasses, standard library
only, every section, plus the parse and the validation. This is the type the
engine and the composition root consume, and it is what `TitanConfig` means
anywhere in the engine.

`titan/config/settings.py` is the pydantic projection the HTTP surface reads. It
covers the three sections the API touches (server, model with its aliases, and
limits) and gives them request-shaped behaviour: resolve an alias, merge sampling
field by field, read the bearer key off disk on demand. Every default it uses
comes from the schema module rather than being typed again, and
`tests/config/test_parity.py` compares the two field sets so a section added on
one side and forgotten on the other fails the build.

There is one parse. `titan.config.settings.load_core` reads the file, applies the
overrides, builds the dataclass and validates it; `wiring.load_config` returns
that, and `settings.load_config` projects it. A JSON file with the same content
means exactly what the TOML one means, because both go through that function.

## Sections

### `[model]`

| key | default | what it is |
|---|---|---|
| `path` | required | directory holding the weights, `tokenizer.json` and `chat_template.jinja` |
| `name` | the directory name | canonical id, echoed in every response and listed by `/v1/models` |
| `max_context` | 262144 | the model's own window |
| `sampling` | see below | this model's defaults |
| `enable_thinking` | true | the canonical name's thinking mode |
| `reasoning_effort` | `medium` | one of `low`, `medium`, `xhigh` |
| `weights_gb` | 0 | resident weight size; 0 means unknown and turns off the guard headroom check |
| `fuse_gate_up` | true | loader plan option |
| `ngram_table_path` | `""` | packed contiguous n-gram table, streamed from SSD |
| `ngram_reader_workers` | 16 | parallel preads; about 84% of a row read is filesystem overhead, so worker count matters more than bandwidth |

`[model.dtype]` is the precision policy in one place, so a bench result can name
it: `compute` (bf16), `kv` (bf16), `accumulate` (fp32) and `quantization`, which
is `checkpoint` (the per-module bits and group size recorded in the checkpoint
win) or `none`. The loader never guesses.

`[model.aliases."<name>"]` declares a profile. Same weights, different template
kwargs and different sampling. The canonical example:

```toml
[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think"]
enable_thinking = false

[model.aliases."Qwen3.8-Flash-Next-oQ4e-mtp:no-think".sampling]
temperature = 0.0
top_p = 1.0
top_k = 0
```

An alias may not collide with the canonical name, and its `template_kwargs` may
not set `enable_thinking`, `reasoning_effort` or `add_generation_prompt`: those
have their own fields, and a kwarg that shadowed one would silently win.

### `[server]`

`host` (127.0.0.1), `port` (8085), `api_key_file`, `request_timeout_s`,
`max_body_mb`, `sse_keepalive_seconds` (10) and `sse_keepalive_mode`, which is
`chunk` (an oMLX-compatible no-op event), `comment` (`: ping`) or `off`.

Port 8085, not 8083 or 8084. Production and the workbench own those two, and a
config that names one is refused at parse time rather than at bind time, where
the failure would already have disturbed the thing being compared against.

The API key is never a config value, only a path to a file holding it. The file
is read per unauthenticated request rather than cached, so rotating the key does
not need a restart. `titan print-config` masks the path.

### `[sampling]`

The process-wide defaults: `temperature` 0.7, `top_p` 0.8, `top_k` 20, `min_p`
0.0, `repetition_penalty` 1.0, `max_tokens` unset. These are the numbers the
Qwen3.8-Flash-Next model card gives for thinking mode. Temperature 1.0 produces
the repetition and tool-argument corruption that made the first week of agent
runs unusable, so changing them is a decision, not a preference. A request
overrides them field by field: absent means absent, never OpenAI's 1.0.

### `[scheduler]`

| key | default | what it is |
|---|---|---|
| `max_seqs` | 8 | sequences sharing a decode cycle |
| `queue_depth` | 64 | admission queue, never below `max_seqs` |
| `prefill_chunk` | 2048 | measured on this GPU: 512 is 22% worse per token, 4096 is 20% worse |
| `serialise_prefill` | true | one sequence prefills at a time, never during a decode cycle |
| `memory_guard_gb` | 110 | admission gate, never a throttle on running work |
| `memory_guard_soft_fraction` | 0.85 | above this fraction of the guard, admission stops; running sequences are untouched |
| `decode_rows_budget` | 32 | total verify rows per cycle across all sequences |

The block size and the snapshot grid are not here. They are properties of the
cache and they live in `[cache]`; the scheduler reads them from there when it
plans chunks. Keeping them apart is the point of the design, so putting them in
the same table as the chunk size would be the wrong kind of tidy.

### `[speculation]`

`enabled` (true; off costs 1.5x, 61.5 tok/s becomes 41), `mtp_depth_max` (3;
depth 4 measured 6% worse under a fixed policy), `mtp_depth_min` (1),
`adaptive_depth` (true), `acceptance_window` (64), and the shortlist drafter:
`shortlist_draft` (false) with `shortlist_max_block` (14). The shortlist arm
stays off until it beats the chain on this machine. Two rules if it is turned on:
slice from the prompt only, never from generated text, and never accept a block
past a stop token.

### `[cache]`

| key | default | what it is |
|---|---|---|
| `block_tokens` | 512 | the KV persistence and reconstruction unit |
| `snapshot_grid` | 2048 | where recurrent snapshots are always staged; must be a multiple of `block_tokens` |
| `snapshot_at_prompt_end` | true | always snapshot where the prompt ends |
| `fine_tail` | true | cut the uncached suffix on the fine grid near its end |
| `fine_tail_blocks` | 4 | how many blocks the fine cut covers |
| `ram_tier_mb` | 4096 | hot tier; 4 GB measured the same hit rate as 16 GB with far less pressure |
| `ssd_dir` | `""` | empty means RAM only, which is a legitimate choice on a machine whose disk is busy with the n-gram table |
| `ssd_capacity_gb` | 200 | |
| `max_stall_ms` | 50 | hard cap on how long a store may block the scheduler thread |
| `pending_write_budget_mb` | 512 | |

The two grids are decoupled deliberately. Reconstruction is bytes-bound at about
2.77e-3 ms per token, so 48 blocks of 512 cost what 12 of 2048 cost: block count
is free, resolution is not. The overlay tied them together, could only resume on
a 2048-token grid, and recomputed 11.6k tokens across a six-turn conversation it
had already seen.

### `[kernels]`

`enabled` (empty means every registered op takes its fast path where supported),
`disabled` (named ops forced to reference, for bisecting), `reference_only`
(every fast path off and fail-open off: the M1 parity baseline and the control
arm of every kernel A/B) and `fail_open` (true).

A name in `enabled` or `disabled` that is not a registered op is an error, so a
stale bisect flag cannot quietly do nothing. `reference_only` together with a
non-empty `enabled` is a contradiction and is refused.

### `[limits]`

`max_tokens` (32768) and `max_context` (262144). A request asking for more is a
400, not a silent clamp. The context limit may not exceed `model.max_context`.

### `[observability]`

`cycle_profile` (true; the per-cycle profile is a product feature, not a flag),
`cycle_profile_ring` (4096), `log_path`, `metrics_enabled` (true) and
`trace_sample_every` (0, which disables the detailed sampled trace because it
does sync the device).

## Validation

Every failure raises `titan.core.errors.ConfigError` naming the dotted path of
the offending key, because the person reading the message is looking at a file
and needs to know which line to change. Unknown keys are errors at every depth: a
typo must stop the process rather than silently take a default.

The rules that are not negotiable:

- the snapshot grid is a multiple of the block size, and so is the prefill chunk;
- the port is not 8083 or 8084, and is in 1..65535;
- the memory guard leaves headroom over `model.weights_gb` when that is known;
- `mtp_depth_max >= mtp_depth_min >= 1`;
- `limits.max_tokens <= limits.max_context <= model.max_context`;
- no op is both enabled and disabled;
- an alias does not collide with the canonical name and does not shadow a
  template field;
- every sampling number is inside its range, in the process defaults, the model
  defaults and each alias, with the path in the message saying which.

## Overrides

`--set key.path=value`, repeatable, applied to the decoded table before the
config is built, so an override is validated exactly like a file value.

The value is parsed as TOML, which is the part worth knowing:

```
--set server.port=8085                    integer
--set scheduler.memory_guard_gb=96.5      float
--set speculation.enabled=false           boolean
--set kernels.disabled='["topk_radix"]'   array
--set model.path=/models/qwen             bare word, so a string
--set server.host="0.0.0.0"               quoted string
```

An override that names a key the schema does not have fails the same way a file
typo does. Later overrides of the same key win. The full list is recorded in the
resolved config and printed by `print-config`, so a captured config says how it
was produced.

## The command line

```
titan serve --config titan.toml [--set a.b=c]
titan check-config --config titan.toml
titan print-config --config titan.toml [--format toml|json] [--no-redact]
titan parity -- --only tools --reference-only
titan bench
```

`--config` may be omitted, in which case `$TITAN_CONFIG` is used; with neither,
the command fails rather than starting on defaults, because a server nobody
configured serves a model nobody chose.

`check-config` and `print-config` construct no runtime, import no MLX and do not
touch the model directory, so they run on a laptop against a config for a machine
that is not this one. `print-config` emits TOML that can be fed back in.

`parity` passes its arguments through to `bench/parity/greedy_parity.py`. `bench`
lists the benchmark scripts; each takes its own flags and several want a loaded
model, so wrapping them behind one command would either lose their arguments or
grow a second CLI.

## The composition root

`titan/config/wiring.py` is the only module allowed to import across layers, and
every one of those imports happens inside a function. Importing the wiring must
not pull MLX, Metal sources or a checkpoint reader into a process that only wants
to check a config file, and a component that has not been written yet must not
make the whole module unimportable.

Construction order is fixed by dependency:

```
config -> profiler -> op registry -> model backend
       -> tokenizer, template renderer
       -> state codec -> kv store -> prefix cache
       -> decode cycle -> engine loop -> engine -> api app
```

The store comes after the backend because its cache signature names the
backend's layer layout, and a payload written under one layout read back under
another is silent nonsense. The app is last because it is the only part that can
be handed to a client.

`build_runtime(config, parts)` takes a `Parts` object; anything set on it is used
as is and its builder never runs. That is the seam the wiring tests use: a fake
backend, a fake store and a fake engine, exercising the real construction order
with no MLX import, no checkpoint and no GPU. Anything still owed raises a
`NotImplementedError` naming the module that owes it.

`runtime.start()` warms the backend and starts the loop. It does not bind a port:
binding is uvicorn's job and it happens in `titan serve`, so building a runtime in
a test can never open a socket.
