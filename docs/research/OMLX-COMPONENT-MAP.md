# oMLX 0.7.0.dev2 : component map for Titan

Root: `/Applications/oMLX.app/Contents/Resources/omlx/` (referred to as **O/**). Vendored MLX packages live at `/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/` (**P/**). Apache-2.0. Pinned upstreams in `O/_engine_commits.json` (mlx-lm `ab1806e8`, mlx-vlm `78b96eb5`).

One correction to the brief up front: **the qwen3_coder tool-call parser is not implemented in oMLX.** `O/api/tool_calling.py` is a hardening/recovery/streaming layer around `P/mlx_lm/tool_parsers/qwen3_coder.py` (115 lines). Details in §3.

---

## 1. Paged prefix cache, GDN boundary snapshots, SSD tier

### Modules (`O/cache/`)

| Module | Lines | Role |
|---|---|---|
| `prefix_cache.py` | 5141 | **Orchestrator.** `BlockAwarePrefixCache:172` |
| `paged_ssd_cache.py` | 5154 | SSD tier + hot RAM tier + GDN sidecar namespace |
| `boundary_snapshot_store.py` | 2030 | Ephemeral per-request SSD staging of recurrent snapshots |
| `type_handlers.py` | 1887 | Per-cache-type slice/serialize strategy |
| `paged_cache.py` | 1742 | Block *metadata* pool (vLLM BlockPool port). Holds no tensors |
| `type_registry.py` | 278 | class-name → handler |
| `hybrid_cache.py` | 338 | `ModelCacheConfig:42` per-layer cache-type layout |
| `factory.py` | 247 | `CacheConfig:24`, `CacheFactory:51` |
| `interface.py` | 118 | `CacheManager(ABC):15` |
| `observability.py` / `stats.py` / `recovery.py` | 299/267/128 | `BoundarySnapshotDiagnostics:79`, stats dataclasses, `CacheRecoveryManager:22` |
| `vision_feature_cache.py` | 509 | Independent VLM feature cache, not on the KV path |
| `pooling_delta.py`, `deepseek_v41_delta.py`, `_rotating_subclass.py` | 120/141/46 | Model-specific compaction; `PrefillReadyRotatingKVCache:25` |

**Entry gate**: everything is `None` unless `config.paged_ssd_cache_dir` is set. The real server does *not* use `CacheFactory` : `O/scheduler.py:13450-13560` constructs `PagedSSDCacheManager` + `BoundarySnapshotSSDStore` directly with hot-cache/GDN knobs the factory doesn't expose.

### Block size and snapshot grid

There is **no separate snapshot interval : the boundary grid is the block grid.**

- `O/scheduler.py:1638 paged_cache_block_size: int = 256` (factory default is 64, test path only).
- Auto-adjusted at init: aligned to RotatingKVCache window (`scheduler.py:2802`); raised for GDN/ArraysCache hybrids by `_enlarge_block_size_for_arrays_cache:2850` to `max(2048, prefill_step_size, qwen35_floor)` where `_ARRAYS_CACHE_BLOCK_SIZE = 2048:2848` and the Qwen3.5 floor can be 4096.
- The two rules, both in `O/prefill_boundaries.py` (24 lines, worth reading in full):
  - `clamp_prefill_chunk_to_boundary(chunk_tokens, *, cache_tokens, block_size):4` : caps each prefill chunk at the next block boundary.
  - `should_emit_prefill_boundary(*, total_tokens, block_size, last_emitted_tokens):17` : `total_tokens % block_size == 0`.
- Enforced again on ingest: `scheduler.py:6721` rejects any snapshot with `token_count % block_size != 0` (reason `"unaligned_token_count"`).

### Prefix lookup

Not a radix trie : a **chain-hashed dict**, vLLM-style.

```python
# O/cache/paged_cache.py:78
compute_block_hash(parent_hash, token_ids, extra_keys=None, model_name=None) -> BlockHash
```
SHA-256 over `model_name` ‖ (`parent_hash` or seed `b"omlx-root"`) ‖ `bytes(str(tuple(token_ids)))` ‖ `extra_keys`. `BlockHash = NewType(..., bytes)` at `:40`.

Structures: `CacheBlock:127` (`block_id, ref_count, block_hash, prev/next_free_block, is_null, token_count, last_access`), `BlockHashToBlockMap:378`, `FreeKVCacheBlockQueue:194` (O(1) doubly-linked LRU), `BlockTable:446`, `BlockCacheEntry` (`prefix_cache.py:165`).

Lookup: `get_computed_blocks(token_ids, extra_keys, extra_key_token_start, extra_key_ranges) -> (List[CacheBlock], int)` at `paged_cache.py:1012`, wrapped by `find_shared_prefix:1200`. Second index: `prefix_cache.py:230 _prefix_index: dict[bytes, tuple[int, tuple[int,...], int]]`, matched by `_find_best_prefix_match:4722`. TOCTOU-hardened via `acquire_cached_block(block_id, expected_hash):867`.

### GDN snapshots

Capture: `scheduler.py:6131 _emit_prefill_boundary_snapshot(request, prompt_cache, total_tokens)` nulls sliceable layers → `scheduler.py:6690 _on_prefill_boundary_snapshot(...)` → `boundary_snapshot_store.py:220 save(request_id, token_count, snapshot_cache, extract_cache_states_fn, *, block_size=None) -> bool`. Serializes on the caller thread, hands raw bytes to a `boundary-snapshot-writer` daemon (`_writer_loop:1044`).

Format: safetensors written **without touching MLX** (`paged_ssd_cache.py:881 _write_safetensors_no_mx`), tensors as `layer_{i}_state_{k}`. Quantization opt-in via `gdn_sidecar_state_dtype ∈ {fp32,bf16,int8,rht_int8,rht_int16}` (scheduler default `fp32`), applied only to **state index 1 of Arrays-family float32 layers** (`_should_quantize_gdn_state:1352`). Encoder `_encode_gdn_state:1408`: symmetric per-row scale on last axis; RHT variants need a power-of-two last dim.

Restore: `load(request_id, token_count):377`, `load_file(path):448`, `_reconstruct_from_safetensors:1892` + `mx.eval` to detach file-backed arrays. Promotion to durable: `take_staged_file(...):482` → `scheduler.py:1743 commit_gdn_checkpoint(...)` → `paged_ssd_cache.py:2369 commit_gdn_checkpoint_file(...)` (atomic `os.replace`). Selection walks blocks **newest→oldest** (`prefix_cache.py:3153-3330`) and on hit **truncates the block table to that endpoint**.

**The invariant**: a snapshot is keyed `(request_id, token_count)`, accepted only at `token_count % block_size == 0`, and the durable sidecar is keyed by the content hash of the block *ending* at that boundary plus the cache signature. Missing intermediate snapshot ⇒ `store_cache` truncates the prefix rather than writing a placeholder (`prefix_cache.py:948/970/984`).

### SSD tier

Layout `<dir>/<model>/<first-hex-char>/<sha256>.safetensors` (16-way fan-out, `_get_file_path:2092`); sidecars in a parallel `_GDN_SIDECAR_DIRNAME` namespace (`:2114`); ephemeral staging under `<base>/_boundary_snapshots/<pid>-<uuid4>/`, wiped at server start.

Write path is **write-behind**: `save_block(block_hash, cache_data, token_count, model_name="", layer_cache_types=None, layer_meta_states=None, hot_cache_write_back=True, replace_existing=False) -> bool` at `:3118` serializes inline, enqueues, returns; one `ssd-cache-writer` daemon drains (`_writer_loop:3078`). Queue depth is bytes-aware (`_compute_max_pending_writes:96`, RAM fractions 0.10/0.30, clamped [32,256]).

Eviction: **size-capped global LRU** over three indexes sharing one budget (`PagedSSDCacheIndex:1061`, an incompatible-block twin, `GDNCheckpointIndex:1304`), walked by `_evict_tracked_until_size:4521`. Dynamic ceiling `_get_effective_max_size:4483` = `min(configured, _DISK_SAFE_RATIO × (tracked + disk_free))`.

Read/promote: `load_block:3744`, `load_block_with_metadata(block_hash, promote_to_hot_cache=True):3898`, bulk `preload_matched_blocks:4137`.

Compat: `_CACHE_FORMAT_VERSION = "3"`; signature `_cache_compat_signature:252` over `{model_name, num_layers, block_size, layer_cache_types}` + optional `turboquant_kv_bits, cachelist_subtypes, payload_layout, gdn_sidecar_state_dtype`.

### Hot RAM tier

`_hot_cache: OrderedDict[bytes, dict]` at `:1746`, holding raw bytes (saves) or `mx.array`s (SSD promotions). Accounting in **bytes** (`_hot_cache_entry_size:1818`), cap `hot_cache_max_size` (**default 0 = disabled**). `SharedHotCacheBudget:1416` allows a process-wide pool across models. RAM→SSD eviction: `_hot_cache_put:1851` pops LRU; clean entries dropped, dirty ones `_enqueue_ssd_write:1896`. Lock order documented at `:1777`: `_hot_cache_lock → _pending_write_hashes_lock`.

### Public API

`interface.py` ABC: `fetch(key)->(val,hit):27`, `store(key,value)->bool:40`, `evict:54`, `clear->int:67`, `get_stats:77`, `size:88`, `max_size:99`, `utilization:109`. Thin adapters : the real surface is on `BlockAwarePrefixCache`:

```python
fetch_cache(request_id, tokens, extra_keys=None, extra_key_token_start=None,
            extra_key_ranges=None) -> (BlockTable|None, list[int])            # :618
store_cache(request_id, tokens, cache_data, model_cache_config=None,
            boundary_snapshots=None, extra_keys=None, ..., hot_cache_write_back=True,
            _store_exact_terminal=False) -> BlockTable|None                   # :746
store_exact_prefix:1433 / fetch_exact_prefix:1531 / restore_exact_prefix:1582
get_cache_for_generation(request_id) -> (list|None, bool)                     # :2728
release_cache:2759 / clear_request_entry:2771 / fork_cache:2786
preload_blocks(block_table) -> int                                            # :2824
reconstruct_cache(block_table, promote_to_hot_cache=True) -> list|None        # :2854
store_mtp_prefix_snapshot:4865 / restore_mtp_prefix_snapshot:4898
set_gdn_checkpoint_loader:299 / set_paged_ssd_cache_manager:482
```
Typical order: `fetch_cache` → `preload_blocks` → `reconstruct_cache` → generate → `store_cache(..., boundary_snapshots=_BoundarySnapshotProvider)` → `release_cache`.

### Invariants

Token-id hashing with no tokenizer version in the digest (only `model_name`; adapter identity must be threaded as `extra_keys`) · full-block matching only · snapshots only at block multiples · single-process (`pid-uuid4` session, index rebuilt by scanning, no cross-process lock) · exact layer-count/layout match enforced by signature · GDN quantization only touches float32 Arrays state slot 1 · `PagedCacheManager._lock`/`PagedSSDCacheManager._lock` are RLocks but `BlockAwarePrefixCache` is largely unlocked (relies on GIL-atomic dict ops) · **all `mx.*` must run on the inference thread**, which is why serialization is inline and writers use `_write_safetensors_no_mx` · every failure degrades to a shorter prefix, never raises.

---

## 2. MTP chain : `O/patches/mlx_lm_mtp/batch_generator.py` (3578 lines)

The file defines **no generator class**; `apply():123` monkey-patches upstream `GenerationBatch` (`patched_init:143`, `patched_next:155`, `patched_extend:219`, `patched_filter:249`) and `BatchGenerator._next` (`patched_bg_next:271`, which forces `completion_batch_size = 0` while MTP is active).

The de-facto state object is built by `_make_row_batch:1049` : a `SimpleNamespace` with `model, prefill_step_size, uids, prompt_cache, tokens, samplers, fallback_sampler, logits_processors, state_machines, max_tokens, _next_tokens, _next_logprobs, _token_context, _num_tokens, _matcher_states` (`:1063-1085`). That field list is the real interface contract.

Classes: `_MtpStepFallback(RuntimeError):651`, `_MtpStats:661`, `_MtpState:692`, `_MtpBatchState:767`, `_MtpParkState:781`, `_DepthController:1823` (`__init__(max_depth, marginal_ms=None, exit_margin=None):1915`).

### Draft chain of depth k : `_chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf):2303`

Called at the *end* of each verify cycle (`:3131`) and once at seed (`:2534`).

1. One batched fold gives draft #1 free (`:2385`):
   ```python
   logits, head_hidden = model.mtp_forward(hidden_rows, committed.reshape(1, n),
                                           state.mtp_cache, return_hidden=True, logits_keep=1)
   ```
   `hidden_rows = hidden[:, : m + 1]` sliced from the backbone at `:3129`; trunk RMSNorm applied at `:2373` unless `_omlx_mtp_head_prenorm`.
2. Chain loop `:2402-2423`: step i+1 re-enters the head on **its own** output hidden (`h = head_hidden[:, -1:]` at `:2422`), not the backbone's. Only the fold consumes backbone hidden.
3. `chain_cache = _clone_mtp_head_cache(state.mtp_cache):2391` when `head_clone and depth > 1`, so the persistent head cache stays committed-only.
4. Dispatched lazily via `mx.async_eval(state.drafts):2431`, resolved by the *next* cycle's sync.

Depth knob: `_MTP_DEPTH = 1` (`__init__.py:70`), `set_mtp_depth:73` clamps to `MAX_LIGHTNING_MTP_DRAFT_TOKENS = 8` (`O/model_settings.py:34`); stamped on the model at load (`qwen35_model.py:490-493`), re-clamped in `_resolve_mtp_chain_depth:1609`, then made adaptive by `_DepthController`.

DeepSeek-V4 takes a different arm: `_dspark_next_drafts:2227` (one block-parallel `dspark_forward(..., draft_length=depth)` + rank-R Markov head).

### Verify forward with M = k+1 : `_run_verify_cycle_chain:2908`

```python
k = int(state.drafts.shape[0])                                    # :2932
inputs = mx.concatenate([state.next_main, state.drafts])          # :2935  (k+1,)
logits, hidden, gdn_states = _call_backbone(model, inputs[None, :],
                                            gen_batch.prompt_cache, n_confirmed=1)   # :2949
```
Input `(1, k+1)`; `rows = logits[0]` is `(k+1, vocab)`. **No explicit position ids, no tree/spec mask** : the chain is linear, so the plain causal mask built inside the model suffices (`qwen35_model.py:436-437`). Positions come from each layer cache's `offset`. All k+1 rows append in one `update_and_fetch`. `n_confirmed=1` is threaded to `GatedDeltaNet.__call__` (`qwen35_model.py:303`, split at `:332`) so the SSM snapshots pre-draft state. `_call_backbone:1520` arms `set_undo_armed(True):1552` and the verify-qmm kernel `:1555`.

### The single host sync

Exactly one per cycle, a `.tolist()` on a small int array (not `mx.eval`).

Greedy (`:2984-2996`):
```python
targets = mx.argmax(rows, axis=-1).astype(mx.int32)                        # :2985
matches = (targets[:k] == state.drafts.astype(mx.int32)).astype(mx.int32)  # :2986
m_arr   = mx.cumprod(matches).sum().reshape(1)                             # :2987
host = mx.concatenate([m_arr, targets, state.drafts.astype(mx.int32)]).tolist()  # :2988  <-- SYNC
```
Acceptance = in-graph cumprod-prefix match; ~2k+2 ints crossed.

Stochastic (`:2997-3049`): batched Leviathan/Chen rejection sampling, also single-sync at `:3022-3029` (`[m_arr, drafts, res_samples, bonus_tok].tolist()`); residuals computed for all k positions so the sync stays single. **Greedy is not required.** Legacy depth-1 (`_run_verify_cycle_legacy:3250`, `_step_mtp:3436`) is *not* single-sync-optimized.

Emitted per cycle = m + 1.

### Rollback : in-place trim, plus a small GDN replay

`_chain_rollback(model, prompt_cache, m, k, gdn_states):3205`, called only when `m < k` (`:3113`); full accept calls `_clear_rollback:3108`.

- **KV layers**: `c.trim(trim_n)` : an offset rewind. No backbone replay forward.
- **GDN/linear layers** (`qwen35_model.py:620-633`): the cache carries `rollback_state = (conv_0, ssm_0)` and `_mtp_draft_stash = (qkv_s, a_s, b_s)`. Rollback **replays the kept prefix** through `layer.linear_attn._process_chunk(qkv_s[:, :keep], a_s[:, :keep], b_s[:, :keep], conv_0, ssm_0, None)` at `:624`, where `keep = 1 + accepted`. So GDN state is recomputed from the pre-forward snapshot by a small kernel per linear layer, paid only on rejection.
- **All-or-nothing**: every layer is validated before anything mutates (`:610-618`); otherwise per-layer KV lengths desync and the shared mask breaks.
- **Rotating KV** needs `O/patches/mlx_lm_mtp/cache_rollback.py` (211 lines) : a rotated ring is not trimmable, so `_wrap_rotating:84` stashes an undo log on `update_and_fetch:95` (armed only around `_call_backbone`), and the patched `trim:135` restores the snapshot then **replays the accepted prefix** (`orig_update` per-token when decode-consistent, else one batched update).
- MTP head cache: `_mtp_head_trim_to(state.mtp_cache, state.hist_offset):3122`.
- Two clamps on m before rollback: `mtp_clamp_accept:3050` and commit-boundary alignment `:3062`; processor snapshots rewound at `:3085`.

### Per-sequence batch state

`_MtpState:692` per row (`queue: Deque[(token_id, logprobs, source)]:706`, `mtp_cache:709`, `next_main:713`, `drafts:738`, `hist_offset:749`, `controller:761`, …); `_MtpBatchState:767` is `Dict[uid, _MtpState]` : no shared arrays.

Differing acceptance lengths are handled by **decomposition, not padding**: `_mtp_batch_next:2590` extracts a single-row cache per row (`gen_batch.extract_cache(idx)`), runs the singleton cycle (`:2622`), and merges back via `_replace_cache_rows` → `_merge_row_caches:1089`, which requires a `merge(per_row)` classmethod on each cache class. Rows are ragged by construction; each row emits exactly one queued token per `next()`.

Finished mid-chain: `_emit_batch_responses:2632` builds `keep`/`finished_uids`, emits a terminal Response with the extracted cache, pops state (`:2694`), calls `gen_batch.filter(keep):2697`.

### Invariants

Singleton path requires `len(uids)==1`; multi-row is **env-gated off by default** (`_rowwise_batch_mtp_enabled:432`, reason at `:640`: "standard batched decode is faster at batch >= 2") · draft sampler deliberately sharper than target (`_resolve_draft_sampler:2190`, temp 0.6/top_p 0.95/top_k 20) with exactness preserved via true q in `_accept_lp_for:1398` · grammar-constrained decoding disables MTP (`_has_grammar_processors:882`) · every failure raises `_MtpStepFallback`, caught in `patched_next`, which calls `_reconcile_mtp_to_standard:1276` : the load-bearing correctness boundary, since the MTP path never maintains `_next_tokens` · `_DepthController.should_exit:2109` parks to standard decode with exponential 128→4096-token cooldown.

---

## 3. Tool-call parsing

### Where the qwen3_coder parser actually lives

`P/mlx_lm/tool_parsers/qwen3_coder.py` (115 lines) : this is the piece to lift.

```python
tool_call_start = "<tool_call>"     # :103
tool_call_end   = "</tool_call>"    # :104
_function_regex  = re.compile(r"<function=(.*?)</function>$", re.DOTALL)   # :14
_parameter_regex = re.compile(r"<parameter=(.*?)</parameter>", re.DOTALL)  # :15
def parse_tool_call(model_output: str, tools: Optional[Any] = None)        # :108 -> {"name":..., "arguments":{...}}
```
Grammar: `<tool_call><function=name><parameter=k>value</parameter></function></tool_call>`. Names extracted by `index(">")` (`:83-84, :89-90`), so anything but `>` is legal. One leading and one trailing `\n` stripped per value (`:92-95`).

**Dialect selection is chat-template sniffing, not model name, and it is an if/elif chain, not a registry** : `P/mlx_lm/tokenizer_utils.py:546 _infer_tool_parser(chat_template)`. qwen3_coder is selected on the literal `"<tool_call>\n<function="` at `:563-567`. Overridable by `tokenizer_config["tool_parser_type"]:623`; loaded via `importlib` at `:628`, exposing `.tool_parser` / `.tool_call_start` / `.tool_call_end` on the wrapper. oMLX's only qwen3_coder-specific branch is `O/server.py:5127 _chat_can_stream_qwen_tool_envelopes(engine)`, comparing parser identity at `:5140-5150`.

### oMLX layer : `O/api/tool_calling.py` (3143 lines)

Public entry:
```python
parse_tool_calls(text, tokenizer, tools=None) -> (cleaned_text, Optional[List[ToolCall]])   # :1542
_parse_tool_calls_impl(...)                                                                  # :1583
extract_tool_calls_with_thinking(thinking_content, regular_content, tokenizer, tools=None)
    -> ToolCallExtraction                                                                    # :1797
sanitize_tool_call_markup(text, tokenizer) -> str                                            # :1767
```
Flow: strip `<think>` → split payloads with `_marker_payloads:557` → call `tokenizer.tool_parser` per payload → on `ValueError/JSONDecodeError/AttributeError/KeyError/SyntaxError/TypeError/RecursionError`, fall back per-match to `_parse_xml_tool_calls:580` → strip spans → `_remap_tool_call_names:1478`.

The genuinely valuable hardening (upstream is fragile here): `_find_marker_span_end:497` picks the *real* `</tool_call>` when one appears inside a parameter value, trying `_json_value_end:362` then `_xml_function_payload_end:410` (requires `</function>` immediately before the close and balanced parameter counts, bounded by `_XML_MAX_END_CANDIDATES = 32:397`). Plus `_iter_xml_parameters:477`, `_xml_element_value_end:453`.

Other dialects present: `_parse_namespaced_tool_calls:659` (MiniMax), `_parse_hermes_tool_calls:707`, `_parse_bracket_tool_calls:807`, a large Gemma-4 cluster (`:892-1409`), `_parse_k2_tool_calls:1521`.

### Streaming state machine

`ToolCallStreamFilter:1904` : a **suppression filter, not an incremental argument parser.** It hides markup from the content channel and emits whole validated envelopes; it never emits partial `arguments`.

```python
__init__(self, tokenizer, *, consume_dsml_separator=True, capture_ordered_segments=False)  # :1934
feed(self, text: str) -> str        # :2642   (returns only content-safe text)
finish(self) -> str                 # :2734
take_completed_envelopes() -> List[str]                 # :2030
take_ordered_segments() -> List[ToolCallStreamSegment]  # :2044
take_recovery_candidate() -> str                        # :2019
```
No enum : a string field `_json_state` set in `_reset_json_scan:2121`, values `"undecided" | "array_head" | "scanning" | "xml" | "not_json" | "complete"`. Envelope state is boolean (`_suppressing`, `_suppressing_until`, `_pending_envelope_parts`).

Split-tag handling: opening tags via `_partial_prefix_len:2403` / `_partial_suffix_len:2433` (withhold the trailing partial); closing tags by draining into `_pending_envelope_parts` and rebasing offsets with `_shift_json_scan:2135`, which retains `_XML_TAIL_KEEP = 64:400` chars as `_xml_prev_tail` so a `</function>` straddling a chunk boundary stays visible. Dialect detection accumulates `_payload_head` across drains (`_advance_json_scan:2142`) because `<function=` needs ten chars. Close detection `_find_suppression_end:2227`. Bounds: 16 envelopes / 2 MB (`:1929-1930`).

Consumer `O/server.py:5344-5419` runs the **full non-streaming** `parse_tool_calls` on each completed envelope (`:5369`); a `stream_tool_sequence_safe` latch (`:5231`) disables early emission permanently on any parse failure or unregistered name, so ordering can't break.

### reasoning_content

Separate module, applied **before** tool parsing: `O/api/thinking.py`. `_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL):27`; MiniMax `<mm:think>:21` and HY3 `<think:opensource>:23` normalized to `<think>` at `:175-178`.

```python
extract_thinking(text) -> (thinking, content)          # :145
class ThinkingParser:215
    __init__(self, start_in_thinking: bool = False)    # :237
    feed(self, text) -> (thinking_delta, content_delta) # :250
    finish() -> (thinking_delta, content_delta)         # :326
```
Streamed as its own field: `ChatCompletionChunkDelta.reasoning_content` (`O/api/openai_models.py:606`); split at `O/server.py:5281`. Each channel gets its **own** `ToolCallStreamFilter` (`:5243` content, `:5250` thinking with `consume_dsml_separator=False`); thinking-channel envelopes are discarded mid-stream (`:5314`) : thinking-channel tool calls are a terminal fallback only, which exists because Qwen3-Coder genuinely emits them there (comment at `tool_calling.py:1842`).

### OpenAI output shapes

```python
# O/api/openai_models.py
FunctionCall:189   name: str;  arguments: str       # JSON string
ToolCall:237       id: str;  type: str = "function";  function: FunctionCall
AssistantMessage:401  role, content: Optional[str], reasoning_content, tool_calls
ChatCompletionChunkDelta:601  role, content, reasoning_content, tool_calls: Optional[List[dict]]
```
Non-streaming construction `tool_calling.py:144 _build_tool_call(name, arguments)`: **id = `call_` + `uuid4().hex[:8]`**; `arguments` serialized by `_serialize_tool_call_arguments:94` (dict → `json.dumps(ensure_ascii=False)`; str round-tripped). `RecursionError`/`SyntaxError` deliberately propagate so `_build_tool_call` drops the whole call (`:162-170`) rather than shipping empty arguments.

Streaming deltas are **raw dicts, name and arguments together in one chunk** : there is no name-then-fragments split:
```python
{"index": index, "id": tc.id, "type": "function",
 "function": {"name": ..., "arguments": ...}}     # O/server.py:5396-5412 (early), :5667-5687 (terminal)
```
`index = len(streamed_tool_calls)` early, `enumerate` index terminally; reconciled by `_merge_streamed_tool_call_prefix:5157` keyed on `_tool_call_semantic_key:5100`.

`finish_reason = "tool_calls" if tool_calls else output.finish_reason` : `O/server.py:4369` (non-streaming), `:5696-5699` (streaming).

### Coercion and robustness

Upstream `_convert_param_value:36` uses schema types from `_get_arguments_config:22`, falling through to bare `ast.literal_eval:79` : the main source of `SyntaxError` on real output. oMLX's `_coerce_param_value:258` mirrors it but never raises, decodes JSON-quoted strings back, and adds `_repair_json_value:214` (rebalances closers, terminates unclosed strings).

Unknown names: `_remap_tool_call_names:1478` remaps only when exactly one registered tool is a `:`-boundary suffix : deliberately not `str.endswith`, with the injection rationale spelled out at `:1493-1497`. Unterminated at EOS: `_unwind_withheld_at_eof:2561` (single forward pass), stashing to `_recovery_candidate:2600`, surfaced only if final parsing found nothing (`O/server.py:5591`). DoS bounds throughout: `_TOOL_CALL_MAX_BOUNDARY_SCAN = 262_144:359`, etc.

Two gotchas before copying: nothing streams partial arguments; and `_parse_xml_tool_calls` restricts names to `\w+` (`:474`, `:618`), stricter than upstream's `index(">")`, so hyphenated/namespaced tool names parse natively but fail the fallback.

---

## 4. Scheduler : `O/scheduler.py` (13,934 lines)

### Structure

Lines 95-1590 are monkey-patches onto mlx-lm's `GenerationBatch`/`KVCache` families, not scheduler logic. Dataclasses: `_PreflightRejection:176`, `_AdmissionEstimate:194`, `_VLMMTPDecodeState:210`, `_StopOutputState:250`, `_StoreCacheGate:310`, `PrefillEvictionRequest:432`, `_PrefillState:487`, `_InflightStoreInfo:513`, `_CacheFreshnessWait:521`, `_RegisteredRow:548`, `SchedulingPolicy:1589`, `SchedulerConfig:1607`, `SchedulerOutput:1674`, `_BoundarySnapshotProvider:1695`, **`Scheduler:1789`** (~12,000 lines, one class).

```python
def __init__(self, model, tokenizer, config: SchedulerConfig | None = None, stream: Any | None = None)  # :1817
```
`tokenizer = copy.deepcopy(tokenizer):1841` (HF Rust "Already borrowed" races); `config = copy.copy(config):1842` because the scheduler mutates its own copy. Construction sites: `engine_core.py:270`, `settings.py:1741`, `engine_pool.py:290`.

### Chunked prefill and the 2048 pin

**`O/scheduler.py:1620  prefill_step_size: int = 2048`** : a plain dataclass default and nothing else. `settings.py:1741-1760` builds `SchedulerConfig` and never passes it; there is no env var, settings key, CLI flag, or clamp anywhere in the package. It is reachable only by patching source or constructing `SchedulerConfig` directly. Two unrelated 2048s exist as block sizes: `_POOLING_ROTATING_BLOCK_SIZE:2749` and `_ARRAYS_CACHE_BLOCK_SIZE:2848`.

Sizing pipeline, in order, in both prefill loops:
1. `_base_prefill_step_size(processed, remaining):5339` : model hooks (GLM-DSA, MiniMax-M3) may override.
2. `_prefill_step_size_for_progress:5323`.
3. `_contended_prefill_cap():5177` : **adaptation is by decode contention, not memory**: `cap = _DECODE_STALL_TARGET_MS/1000 * _prefill_tps_best`, quantized to `_CONTENDED_CHUNK_GRID = 64:1550`, floored at `_CONTENDED_CHUNK_FLOOR = 256:1545`. Returns 0 when nothing decodes.
4. `clamp_prefill_chunk_to_boundary` (`O/prefill_boundaries.py:4`).
5. `_adaptive_chunk_size(requested, *, request_id, loop_label, kv_len, gathered_core):4467` : memory shrink.
6. `_guard_prefill_chunk(n_tokens, *, kv_len, progress, loop_label, request_id, gathered_core):4258` : hard clamp/abort.
7. `_snap_chunk_size(n, requested):4440` : quantize to a multiple of `_prefill_min_chunk_tokens` for MLX buffer-pool reuse (measured-miss-rate comment at `:4442-4459`).

Resume state: `_PrefillState:487-510` : `request, cache, tokens_remaining (mx.array (1,N)), last_token, tokens_processed, base_size, emitted_boundaries, boundary_enabled, block_size, total_length, sampler, sm, per_row_lps, qwen4_gathered_core`. Held in `_prefill_states: dict[str,_PrefillState]:1929` alongside `prefilling: deque:1928`. Lifecycle `_begin_prefill:5373` → `_step_prefill_chunk:5460` → `_advance_chunked_prefills:5860` → `_insert_prefilled_request:5759`. `tokens_remaining` is sliced forward as a view (`:5546`).

Entry decision `:11020-11037`: chunked when `(chunked_prefill or force_chunk) and vlm_embeds is None and len(tokens) > threshold+1`, where `force_chunk:11021` is true under decode contention : so long prompts chunk even with `chunked_prefill=False`. VLM prompts never chunk.

### Memory guard

Measurement `_current_usage_bytes(*, refresh_mlx_active=True):4715`:
```python
active = max(0, int(mx.get_active_memory()))
phys   = max(0, int(get_phys_footprint()) - hot_cache_bytes)
return max(active, phys)
```
`get_phys_footprint` is Mach `phys_footprint` from `O/utils/proc_memory.py`, not psutil RSS. `mx.get_active_memory()` is MLX-thread-only; other threads read `get_cached_mlx_active_memory_bytes():4668`.

Thresholds are written onto scheduler attributes from outside by `O/process_memory_enforcer.py:1150 _propagate_memory_limit` (`:1244-1281`): `_memory_limit_bytes:1965` (soft), `_memory_hard_limit_bytes:1966`, `_memory_hard_watermark_bytes:1970` (0.95), `_memory_abort_limit_bytes:1980`, `_memory_guard_tier:1993`, `_admission_paused:2006`, `_prefill_headroom_safety:2009` (0.90), `_prefill_safe_zone_ratio:2010` (0.80), `_prefill_min_chunk_tokens:2011`, `_prefill_abort_margin:2012` (0.90). Tiers `_PREFILL_STEP_TIERS = (1024, 512):4014`; `_PREFILL_TRANSIENT_SAFETY = 1.3:4032`.

Prediction: `_predicted_chunk_transient(n, kv_len, *, gathered_core):4036` = `max(static SDPA+KV estimate, EWMA bytes/token, last measured) × 1.3`, EWMA in `O/prefill_transient_tracker.py:227`; fed by `_record_chunk_transient:4941`.

Actions when over: shrink chunk (`_adaptive_chunk_size:4467`), binary-search largest fit (`_largest_fitting_prefill_chunk:4415`), clamp or abort with `PrefillMemoryExceededError` (`_guard_prefill_chunk:4258`), `mx.clear_cache()` + re-measure (`_reclaim_prefill_headroom:5136`), **evict an idle model** (`_raise_prefill_eviction_if_available:4212` → `_pause_for_prefill_eviction:12452`), reject at admission (`_preflight_memory_check:10126`, `_admission_estimate:10214`, `preflight_or_raise:10331`).

Two notable negatives: **`_check_memory_pressure:13598` is a `pass`** (paged-block eviction is dead code in SSD-only mode), and **running sequences are never preempted for memory**. The guard is a prediction-and-shrink system; the only levers are chunk size and admission. `prefill_speed_priority` inverts it (`:4576`, `:10257`).

### Admission and decode batching

Structures `:1922-1934`: `waiting: deque`, `running: dict`, `prefilling: deque`, `_prefill_states: dict`, `requests: dict`, `finished_req_ids: set`, `_pending_abort_ids: set`, plus uid↔request_id maps.

`_schedule_waiting():10543` loops while `waiting and _num_admitted_requests() < _effective_max_num_seqs()` (`:10122`, `:9970` : forced to 1 for Llama-4). Gates, each `break`ing: admission paused (`:10578`), `_prefill_gate_open()` (`:10593`), store-cache backpressure (`:10604`), generation memory guard (`:10653`), cache-freshness defer (`:10665`/`:8485`). Batch homogeneity trackers at `:10564-10567` keep VLM/text apart and run SpecPrefill solo.

**There is no KV budget and no token budget.** `max_num_batched_tokens = 8192:1613` is declared and never read. Admission is seq-count + byte estimates + backpressure only. `max_num_seqs = 256:1611` is overridden by `settings.py:1742` from `max_concurrent_requests` (default **8**).

`step():12475` runs prefill and decode as **strictly separate phases, never a mixed forward**: aborts/reclaim → `_advance_chunked_prefills:12534` → `_schedule_waiting:12540` → `batch_generator.next_generated():12570` → `_process_batch_responses:11330` → `_cleanup_finished:11640`. Prefill runs entirely outside BatchGenerator (`_do_external_prefill:3520`, ~500 lines); only the final prompt token goes to `insert()`.

Ordering is **pure FCFS** (`popleft`, gates break rather than skip : head-of-line blocking by design). `SchedulingPolicy.PRIORITY:1589` is unimplemented. The real fairness mechanism is prefill-vs-decode (`:5162-5303`): `_others_decoding:5162` via a **process-wide** registry (`O/decode_activity.py:21`, TTL 2.5 s), `_prefill_gate_open:5211`, `_accrue_decode_debt:5232` charging `chunk_seconds × _DECODE_FAIR_SHARE` (default 0.5), `_repay_decode_debt:5246`.

### Loop structure and host syncs

Threads: asyncio loop (`engine_core.py:314`); **`mlx-engine-<id8>`, one worker per engine** (`engine_core.py:264`) running `scheduler.step()` and owning `mx.new_thread_local_stream():263`; a global `mlx-global` fallback (`:138`); `omlx-store-cache` (`scheduler.py:2248`); the enforcer poller. The hop is `await loop.run_in_executor(self._mlx_executor, self._step_burst)` at `engine_core.py:418`, which runs several `step()`s per hand-off under a time budget to avoid GIL ping-pong with uvicorn (`engine_core.py:183`).

Synchronization is deliberately lock-light: cross-thread state uses GIL-atomic primitives (`_pending_abort_ids:1936` with the explicit comment at `:1935`, `_pending_reclaim_request:1943`, `_pending_pressure_clear:1957`, `_admin_snapshot:1960` published whole at `_publish_admin_snapshot:12768`). Real locks: `_StoreCacheGate._lock:331`, `_uid_row_registry_lock:563`.

Host syncs:

| Line | Sync |
|---|---|
| **5566** | `mx.eval([c.state for c in state.cache])` : one per chunked-prefill chunk |
| **3778** | same, external prefill loop |
| 739-740 | `mx.eval(next_tokens)` + `.tolist()` : grammar path only; moved to the top of the *following* step (`_omlx_advance_grammar_rows:702`, rationale `:705-717`) |
| 12594 | `_eval_generation_batch_cache` every `_decode_eval_kv_cache_interval` (256 for ArraysCache hybrids) : Metal's 499,000-buffer limit |
| 12624 | `_clear_cache()` every 1024 decode tokens : `mx.random.categorical` scalars in the IOGPU residency set |
| 9219-9249 | `mx.eval` + two `.item()` : SpecPrefill/MTP only |

**In steady-state decode the scheduler performs zero host syncs**: `_process_batch_responses:11330-11605` has no `mx.eval`, no `.item()`, no `.tolist()`, no numpy : `response.token` is already an int. Pipelining lives in mlx-lm's `GenerationBatch._step` via `mx.async_eval`.

### Invariants

One model per Scheduler, immutable · one `mx.Stream` per engine; **every model call and cache-building op must be inside `with mx.stream(self._stream)`** (violations are a documented bug class : see `:10680-10689`, issue #2330) · `mx.async_eval` assumed available, no probe · sliceable cache types enumerated in a frozenset at `:1242` · mlx-lm internals monkey-patched wholesale (`GenerationBatch._step:748`, `.filter:811`, `make_cache:948`, `merge_caches:1036`, `extend_cache:1054`, `PromptPrefillBatch.split/prompt:1062/1226`) : **tightly coupled to one mlx-lm version** · `paged_cache_block_size` is mutated at init, so the config value is a request not a guarantee · tokenizer must be deep-copyable and expose a streaming detokenizer.

---

## 5. Model code : qwen4_exp and qwen3_5

**V/** = `O/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/` (8374 lines).
**Q/** = `P/mlx_vlm/models/qwen3_5/` (3630 lines) : resolved from the relative import `..qwen3_5.language` at `V/language.py:23`; the vendor tree contains only `qwen4_exp`, so the base comes from the installed package.

| V/ file | Lines | | Q/ file | Lines |
|---|---|---|---|---|
| `language.py` | 3221 | | `language.py` | 2664 |
| `cache.py` | 2826 | | `gated_delta.py` | 647 |
| `qsa_fast.py` | 757 | | `config.py` | 153 |
| `hc_fused.py` | 525 | | `qwen3_5.py` | 155 |
| `hc_projection.py` | 420 | | `vision.py`/`__init__.py` | 11 |
| `qwen4_exp.py` | 303 | | | |
| `config.py` | 206 | | | |
| `vision.py` / `__init__.py` | 116 | | | |

### Classes : `V/language.py`

`Qwen4ExpMTPRuntime:128`, `_PLESpeculativeState:139`, `_QSAIndexerCache:232`, `QSAKVCache:489`, `BatchQSAKVCache:623`, `QSAQuantizedKVCache:958`, `Qwen4ExpRMSNorm:1049`, `Qwen4ExpRMSNormGated:1073`, `Qwen4ExpGatedDeltaNet:1091`, `Qwen4ExpQSAIndexer:1109`, `Qwen4ExpAttention:1256`, `Qwen4ExpGatedResidual:1641`, `ShardedEmbedding:2394`, `Qwen4ExpNGramEmbedding:2541`, `Qwen4ExpPLELayer:2679`, `Qwen4ExpDecoderLayer:2776`, `Qwen4ExpModel:2846`, `Qwen4ExpMTPModule:2939`, `LanguageModel:3037`.

`TextConfig` is in `V/config.py:24`. Serving-relevant fields: `num_hidden_layers:27`, `layer_types: Optional[List[str]]:44`, `full_attention_interval: int = 4:45` : `__post_init__:76` fills layer_types as `"linear_attention"` unless `(i+1) % interval == 0`, else `"qwen_sparse_attention"`; only those two are legal (`:103-110`). GDN dims `:31-35` (`linear_num_value_heads`, `linear_num_key_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`; `conv_dim = 2*key_dim + value_dim`). QSA `:56-60` (`indexer_n_heads=4`, `indexer_kv_heads=1`, `indexer_head_dim=128`, `indexer_budget=2048`, `indexer_compress_ratio=4`; `block_topk = budget // compress_ratio`). MTP `:64-66` (`mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings`, `mtp: Optional[Dict]`). Hyper-connections `hc_count=4`, `hc_lowrank=320` : the residual stream is `hc_count * hidden_size` wide.

### Forward signatures

```python
LanguageModel.__call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs)   # V:3071
# kwargs popped by the base (Q:2423): position_ids, pixel_values, image_grid_thw,
#   video_grid_thw, attention_mask, capture_layer_ids, return_hidden,
#   return_shared_kv, skip_logits, rope_deltas
# returns .logits, .hidden_states, .shared_kv_states, .gdn_states

Qwen4ExpModel.__call__(self, inputs, inputs_embeds=None, mask=None, cache=None,
                       position_ids=None, capture_layer_ids=None,
                       hidden_sink=None, gdn_sink=None, **kwargs)                            # V:2863
Qwen4ExpDecoderLayer.__call__(self, hidden_states, input_ids, mask, cache, position_ids,
                              gdn_sink=None, target_verify=False)                            # V:2795
Qwen4ExpAttention.__call__(self, x, mask=None, cache=None, position_ids=None,
                           position_embeddings=None, target_verify=False) -> mx.array        # V:1568
Qwen3_5GatedDeltaNet.__call__(self, inputs, mask=None, cache=None,
                              gdn_sink=None, target_verify=False) -> mx.array                # Q:1593
```
Masks are built once per forward from representative caches: `_create_qwen3_5_attention_mask(h, cache[self.fa_idx])` and `_create_qwen3_5_ssm_mask(h, cache[self.ssm_idx])` at `V:2883-2884`; `ssm_idx`/`fa_idx` discovered by scanning at `V:2856-2861`. `target_verify = gdn_sink is not None` propagates to every layer.

### Cache list contract

One entry per decoder layer, indexed positionally by `zip(self.layers, cache)` (`V:2889`), type chosen by `layer.is_linear`:

```python
def make_cache(self):                          # V:3135
    return [ArraysCache(size=4 if "ple" in layer else 2) if layer.is_linear
            else QSAKVCache()
            for layer in self.layers]

def make_mtp_cache(self):                      # V:3131
    return [QSAKVCache() for _ in mtp.layers]  # one entry
```
Baseline for comparison: `Q:2633` returns `[ArraysCache(size=2) if l.is_linear else KVCache() ...]`.

### GDN state

Container is **`ArraysCache`** (`V/cache.py:661`, subclass of `_BaseCache:86`) : a plain indexable slot list (`__setitem__:711`, `__getitem__:714`, `state:717/721`, `filter:725`, `extend:734`). No `update_and_fetch`; the GDN module writes slots directly.

- `cache[0]` conv state `(B, conv_kernel_size-1, conv_dim)`, input dtype (bf16) : `Q:1612-1620`, written `Q:1636/1638`
- `cache[1]` recurrent state `(B, num_v_heads, head_k_dim, head_v_dim)` : read `Q:1659`, written `Q:1714`
- `cache[2]` PLE short-conv state `(B, short_conv_state_len, ple_embed_dim)` : `V:2718-2729`
- `cache[3]` PLE n-gram token history (int64)

Serialization surface (the snapshot hook): `_BaseCache.state` / `meta_state` / `from_state(cls, state, meta_state):127`, plus **`prefix_cache_snapshot():134`** → `{"state":..., "meta_state":...}`, **`prefix_cache_restore(snapshot):145`**, `prefix_cache_merge(rows, prefix_lens):149`. For `ArraysCache`, `state` is literally the 2- or 4-element array list.

Speculative rollback needs the per-layer `gdn_sink` 12-tuple appended at `Q:1696-1711`: `(q, k, v, a, b, A_log, dt_bias, initial_state, mask, conv_input, conv_kernel_size, intermediate_states)`. Kernels in `Q/gated_delta.py`: `gated_delta_update:126`, `..._with_states:292`, `gated_delta_accept_states:540`, `gated_delta_state_update:597`.

### QSA

**QSA = Qwen Sparse Attention** : layer-type string `"qwen_sparse_attention"` (`V/config.py:87`); `Qwen4ExpQSAIndexer:1109` selects compressed key blocks by top-k over pooled index keys. Full-attention layers are replaced by block-sparse attention.

`_QSAIndexerCache:232` (mixin, `index_step = 8192`): fields `_index_keys [B, capacity, indexer_head_dim]`, `_index_position_ids [B,S]` or MRoPE `[3,B,S]`, `_index_offset`, `_index_capacity_managed`, `_index_reserved_tokens`, plus an ephemeral pooled bank. Methods `reserve_index_capacity:251`, `update_indexer(keys, position_ids):358`, `pooled_indexer_keys:399`, `_trim_indexer(length):465`, `indexer_nbytes:477`.

`QSAKVCache(_QSAIndexerCache, KVCache):489` : `preserve_auxiliary_kv_state = True`, `step = 8192`, `geometric_growth = True`. **`state` is a 4-tuple** `(keys[..., :offset, :], values[..., :offset, :], index_keys, index_position_ids)` (`:504/515`). `trim(n):521` trims KV then `_trim_indexer(self.offset)`. Also `extract:527`, `filter:551`, `to_batch:562`, `merge:605`, `to_quantized:608`. The pooled bank is deliberately not serialized (docstring `:234-239`).

`BatchQSAKVCache:623` adds `prepare(**kwargs):655` / `finalize():658` (right-padding trim used by rollback), `make_mask:673`, `extend:748`, `trim:914`, `state:929/945`.

Plain `KVCache` (`V/cache.py:354`): `step = 256`, `update_and_fetch:364`, `state:407/417` (setter sets `offset = keys.shape[2]`), `is_trimmable:422`, `trim(n):425`, `extract:430`, `to_quantized:450`, `merge:464`.

Rollback: `LanguageModel.rollback_speculative_cache(self, caches, gdn_states, accepted, block_size)` : `V:3202` override wrapping `Q:1963`. The base classifies SSM caches as `not c.is_trimmable() and not hasattr(c, "zero_row_tail")` (`Q:1997`), trims trimmable ones, and rebuilds GDN state via `gated_delta_accept_states`.

### MTP draft block

`Qwen4ExpMTPModule(nn.Module):2939` : "Embedded one-layer draft head for Qwen4 Lightning MTP". Instantiated on the **top-level `Model`**, not `LanguageModel`: `V/qwen4_exp.py:140-142` creates it when `get_mtp_runtime().enabled`, then calls `self.language_model.bind_mtp_owner(self)`.

Structure (`__init__:2942`): `pre_fc_norm_embedding`, `pre_fc_norm_hidden` (`hc_count*hidden_size`), `fc_embedding`/`fc_hidden` (`hidden_size→hidden_size`, no bias), one `Qwen4ExpDecoderLayer` forced to `qwen_sparse_attention` with `ple_layer_ids=[]`, and a `hyper_connection_mixer`. **Depth = 1 draft layer / 1 head**; per-step draft depth comes from `omlx.patches.mlx_lm_mtp.get_mtp_depth()` (`V:3056-3060`).

```python
fuse_inputs(self, token_embeddings, hidden_states) -> mx.array                  # :2979
__call__(self, hidden_states, next_token_ids, embed_tokens, cache=None)
    -> (mixed_output, hc_hidden)                                                 # :3009
# engine entry point:
LanguageModel.mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                          return_hidden: bool = False, logits_keep: int = 0)     # :3101
```
`fuse_inputs` requires `[B, T, hc_count*hidden_size]` and adds the projected accepted-token embedding to each stream. Target hidden states come from calling the model with `return_hidden=True`, which forces `capture_layer_ids=[]` and yields the pre-mixer residual stream (`V:2928-2931`).

### qwen4_exp vs qwen3_5

Imported unchanged (`V:23-32`): `LanguageModel` base, `Qwen3_5Attention`, `Qwen3_5GatedDeltaNet`, both mask builders, `_target_verify_linear(s)`; `Qwen3_5MoeSparseMoeBlock` from `qwen3_5_moe` used verbatim as the MLP; all of `Q/gated_delta.py`; `resolve_qwen_eos_token_id` and `sanitize_quantization_config`. Inherited on `LanguageModel`: `chunked_prefill_policy`, `get_rope_index`, `__call__` body, the four `speculative_*` methods, `layers`, `head_dim`, `n_kv_heads`, `quant_predicate`, `cast_predicate`, most of `rollback_speculative_cache`.

Deltas:
1. **Hyper-connections replace the plain residual.** qwen3_5 (`Q:1741`) does `h = x + r`; qwen4_exp carries an `hc_count`-wide stream through `Qwen4ExpGatedResidual:1641` plus a final mixer. No `input_layernorm`/`post_attention_layernorm` : norms move inside the gated residual.
2. **Attention** adds the QSA indexer + four gathered fast paths backed by `V/qsa_fast.py`.
3. **Cache types**: QSA variants replace `KVCache`/`QuantizedKVCache`, adding the index pair to `state`; `ArraysCache` sized 4 on PLE layers.
4. **GDN**: only the gated output norm (`Qwen4ExpRMSNormGated:1073`) and L2 `_normalize_qk:1100` differ.
5. **MLP**: MoE instead of dense.
6. **New subsystems with no qwen3_5 counterpart**: PLE (`Qwen4ExpPLELayer:2679`, sharded/disk-backed embeddings), Lightning MTP (`:2939`, `mtp_forward:3101`, `make_mtp_cache:3131`), hyper-connection fusion (`hc_fused.py`, `hc_projection.py`).
7. **Layer layout** read from validated `config.layer_types` (`V:2779`) rather than derived arithmetically (`Q:1728`); `ssm_idx`/`fa_idx` discovered by scan.
8. **Rollback** adds PLE conv-state/history restoration via `_PLESpeculativeState` stashed as `cache._qwen4_exp_ple_speculative_state`.

---

## Design notes for Titan

A few things the source makes explicit that are worth deciding on deliberately rather than inheriting:

- **`prefill_step_size = 2048` is unreachable config.** It is the single most consequential number in the scheduler and has no path from user settings. If Titan wants a tunable chunk, wire it from the start.
- **`max_num_batched_tokens` is dead.** oMLX has no token-budget batching at all; admission is seq-count plus byte estimates. If you want continuous batching against a token budget, that is new work, not a port.
- **Prefill lives outside the batch generator.** `_do_external_prefill:3520` and its twin `_step_prefill_chunk:5460` duplicate the whole sizing/guard/boundary sequence ("mirrors the external loop" comments). An engine that owns its own forward can mix prefill and decode in one batched call and delete roughly 2,000 lines. Unify the two loops up front if you port them.
- **The memory guard predicts and shrinks; it never evicts or preempts.** `_check_memory_pressure` is a `pass`. The escape hatch is evicting an entire idle *model*.
- **The MTP single-sync pattern is the genuinely reusable idea**: compute acceptance in-graph with `cumprod`, concatenate everything the host needs into one small int array, and cross the boundary exactly once. It works for both greedy and exact rejection sampling.
- **GDN rollback replays rather than trims**, from a pre-forward snapshot, and validates every layer before mutating any. That all-or-nothing rule is load-bearing: partial rollback desyncs per-layer KV lengths and breaks the shared mask.
- **The boundary-snapshot grid is just the block grid.** Two 12-line functions in `prefill_boundaries.py` encode the whole contract; everything else is bookkeeping around them.
- **Comments carry measured results** at `_snap_chunk_size:4440`, `_reclaim_prefill_headroom:5136`, `_omlx_advance_grammar_rows:705-717`, and `engine_core.py:183` : each encodes a benchmark that would be expensive to rediscover.
