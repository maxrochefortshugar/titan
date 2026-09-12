# QSA verify routing: does MTP verify take the sparse arm?

Workstream `kernels/round3/qsa-verify/`, 2026-09-12. No model loaded, no safetensors opened, peak
1.6 GB GPU, 8083/8084 untouched. `LANG` =
`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`, `QSA` =
`.../qwen4_exp/qsa_fast.py`, `Q35` = bundled `mlx_vlm/models/qwen3_5/language.py`, `BG` =
`omlx/patches/mlx_lm_mtp/batch_generator.py`. Config: 12 QSA layers (48 / `full_attention_interval`
4), 2 KV heads, head_dim 256, bf16 KV, `indexer_budget` 2048, `indexer_compress_ratio` 4, so
top-512 blocks of 4 tokens plus a 0-3 token tail.

## 1. Answer: verify already takes the sparse arm

`LANG:1356-1390` `_gathered_text_verify_eligible` is on by default: `LANG:1044-1046` reads
`OMLX_QWEN4_QSA_GATHERED_VERIFY`, and only `0/false/no/off` disables it. `LANG:1576-1610` tries
decode, prefill, then verify before the dense fallback at `LANG:1612`. Every precondition holds on
the live verify call.

| precondition | file:line | why it holds |
|---|---|---|
| `target_verify` | `Q35:2555-2556` | `gdn_sink` is a list whenever `capture_layer_ids` is set; `LANG:3072-3076` sets it to `[]` for every `return_hidden=True` call, which is what `BG:2946-2951` `_call_backbone` does |
| `x.shape == (1, M)`, `M > 1` | `BG:2937, 2946` | `mx.concatenate([next_main, drafts])[None, :]`, M = k+1 |
| mask is `"causal"` | `Q35:806-826` -> `mlx_lm/models/cache.py:114-124` | `KVCache.make_mask` returns the string for N>1, no window |
| `type(cache) is QSAKVCache` | `LANG:3135-3142` | `make_cache` builds plain `QSAKVCache` per full-attention layer |
| `position_embeddings is None` | `LANG:2891-2898` | the layer loop never passes them |
| `_rank_two_text_position_ids` | `Q35:2548-2550` | with a warm cache and cached `_rope_deltas`, position ids are built as `arange(M).reshape(1,-1) + delta`, exactly `(1, M)` |
| `cache.offset + M > 2048` | `LANG:1390` | true past the QSA budget |

`probe_routing.py` calls the three real predicates over every regime. Result, identical at 8k, 65k
and 130k:

| regime | arm |
|---|---|
| decode L=1, MTP off (`target_verify=False`) | gathered decode, sparse |
| decode L=1, MTP depth-0 cycle (`target_verify=True`) | **dense** |
| verify L=2,3,4,5,6 | gathered verify, sparse |
| prefill L=2048 or 512, batch 1 | gathered prefill, sparse |
| prefill L=2048, batch 2 | **dense** |
| verify L=4 with 3-plane mRoPE ids | **dense** |

Inside the sparse arm, `QSA:321` keeps the native block-sparse GQA kernel off below 24 query rows,
so verify runs the portable gathered SDPA at `QSA:691-745` with fp32 scores; prefill chunks clear 24
rows and get the native kernel. TurboQuant never engages: `turboquant_attention.py` dispatches only
on a `TurboQuantKVCache`. If KV quantisation were switched on, `QSAQuantizedKVCache` (`LANG:958`)
would fail the strict `type(...) is QSAKVCache` test at `LANG:1292, 1327, 1376` and silently drop
all three sparse arms.

## 2. Cost of each arm

KV per token per QSA layer = 2 heads x 256 x 2 (K and V) x 2 B = 2048 B, so 24 KiB per token over
12 layers. `bench_arms.py`, median of 15, CHAIN=3, `mx.synchronize`, GPU contended by the live
daemon, so ratios are reliable and absolutes run high.

| regime | KV read at 65k | KV read at 130k | ms/layer 65k | ms/forward 65k | ms/forward 130k |
|---|---|---|---|---|---|
| sparse decode L=1 | 4.2 MB | 4.2 MB | 0.118 | **1.42** | **1.68** |
| sparse verify L=4 | 4.2 MB (x4 gathered) | same | 0.230 | **2.83** | **3.03** |
| sparse verify L=6 | same | same | 0.266 | 3.19 | 3.39 |
| dense decode L=1 | 134 MB | 268 MB | 0.235 | 2.82 | 4.29 |
| dense verify L=4 | 134 MB | 268 MB | 1.242 | **14.9** | **28.5** |

The sparse arms are flat in context; the dense arms are not. The dense read roofline is 0.187 ms
per layer at 65k, so dense SDPA at M=4 sits 6.6x off it: the GQA broadcast, not bandwidth. A dense
verify would add ~12 ms to the 31 ms MTP cycle at 65k and ~26 ms at 130k, dropping 61.5 tok/s to
~44 and ~33. That brackets MTPLX's paired 36.0 vs 64.5.

Crossover (`bench_crossover.py`): the gathered arm loses below ~8k at L=1 (0.117 vs 0.083 ms) and
wins everywhere at L=4 past 8k. The gate is `cache.offset + L > 2048`, earlier than the measured
crossover, so decode between 2k and 8k pays a small penalty. Not patched here.

## 3. Exactness

`test_verify_sparse.py`, synthetic tensors at the real shapes. [A] verify output against dense SDPA
over the whole cache masked to exactly the tokens QSA picked, with the selection reproduced
independently from `QSA._portable_indexer_scores`. [B] each verify row against the gathered sparse
**decode** arm on a cache truncated to that row's visible prefix.

| N | M | [A] max abs | [A] rel | [B] max abs | cache fraction read |
|---|---|---|---|---|---|
| 4096 | 4 | 1.465e-03 | 8.5e-03 | 9.766e-04 | 50.0% |
| 8192 | 2 | 6.104e-04 | 4.2e-03 | 9.766e-04 | 25.0% |
| 8192 | 4 | 9.766e-04 | 5.5e-03 | 9.766e-04 | 25.0% |
| 8192 | 6 | 1.038e-03 | 5.9e-03 | 9.766e-04 | 25.0% |
| 16384 | 4 | 1.038e-03 | 7.0e-03 | 9.766e-04 | 12.5% |
| 65536 | 4 | 1.953e-03 | 1.1e-02 | 4.883e-04 | 3.1% |
| 2052 | 4 | 9.766e-04 | | | 99.9%, the all-blocks-selected case, equals plain dense |

Every error is one bf16 ULP at the output magnitude. The arms agree.

## 4. The batched-prefill finding is stale

The audit's "batched or padded prefill materialises a 134 MB mask per layer" does not happen on
this build. `scheduler.py:3179` passes `prefill_batch_size=1`, and `scheduler.py:10549-10551` states
that every request is prefilled externally before insertion; `_do_external_prefill`
(`scheduler.py:3520`) calls `make_prompt_cache(self.model)`, which yields a singleton `QSAKVCache`.
Prefill is already serialised, so the gathered arm survives concurrency. Contention only shrinks the
chunk to 512 (`scheduler.py:5323-5337`), which still clears the 16-row gate at `LANG:1290`.

The real concurrency cost is batched **decode**. Joining the batch converts caches through
`to_batch` (`scheduler.py:1015-1017`) into `BatchQSAKVCache` (`LANG:623`), which fails
`type(cache) is QSAKVCache`, so decode falls to the indexer plus dense SDPA. When rows are
left-padded the mask is the string `"left_padded_decode"` and `LANG:1628` drops the QSA mask
entirely; even unpadded, a mask reduces scores, not reads. Measured dense masked decode at 65k:
2.10 ms/forward at batch 1, 2.34 at batch 2, 3.40 at batch 4, against 1.42 ms for one sparse row.
At 2 streams that is roughly a wash against two sparse rows; at 4 streams and at 130k it loses. A
sparse batched arm means per-row selection over a left-padded cache, a real kernel project, out of
scope here.

## 5. What was built

`patch.py`, one env-gated install, idempotent, returns False when preconditions fail. Load by path
with `importlib.util.spec_from_file_location`, **after** the model is loaded: it swaps a class
method on `Qwen4ExpAttention`.

| install | env var | what it fixes |
|---|---|---|
| `install_verify_sparse(model)` | `OMLX_QSA_VERIFY_SPARSE=1` | routes M=1 `target_verify` rows (the depth-0 MTP cycle) through the gathered sparse decode arm instead of dense |
| same, plus `OMLX_QSA_VERIFY_SPARSE_MROPE=1` | | also accepts broadcast-identical 3-plane mRoPE ids at M=1, matching what prefill already accepts at `LANG:1290` |

The depth-0 cycle is the controller's escape hatch (`BG:2971-2980`), so this is worth 2.8 ms at 65k
and 4.3 at 130k on the cycles that use it and nothing on the rest. Routing only: the gathered decode
arm reads no verify-only state, and section 3 [B] shows it reproduces the verify arm to one bf16
ULP.

## 6. Workbench test plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, paired within a round.

| step | change | prompt | measure | expect |
|---|---|---|---|---|
| 0 | none | 8k, 16k, 32k, 65k, 130k contexts, 300 decoded tokens each | decode tok/s | **the pattern that confirms the sparse arm**: tok/s roughly flat from 8k to 130k, at most 10-15% total decay from PLE and GDN, with no knee. A knee past 16k, or 65k landing near 60% of 8k, means the verify arm is dense |
| 1 | `OMLX_QWEN4_QSA_GATHERED_VERIFY=0` | same ladder | decode tok/s | the negative control. Expect ~61 -> ~44 tok/s at 65k and ~33 at 130k, unchanged at 8k. If step 1 looks like step 0, the arm was never engaging and section 1 is wrong |
| 2 | `OMLX_QWEN4_QSA_NATIVE_MAIN_MIN_ROWS=4` | 65k, 300 tokens | decode tok/s | forces the native block-sparse kernel down to verify width. `QSA:65-68` says it loses there (0.7 vs 1.6 ms/layer); expect a regression, which confirms the default 24 is right |
| 3 | `OMLX_QSA_VERIFY_SPARSE=1` plus `OMLX_MTP_DEPTH_TRACE=1` | 65k and 130k agentic turn | `MTP depth:` lines, tok/s | only helps if the trace shows depth-0 cycles. If `zero_cycles` is under 1% of cycles, drop the patch |
| 4 | 2 and 4 concurrent streams at 65k | agentic suite | aggregate and per-stream tok/s | per-stream should fall harder than 1/N once batched decode goes dense; that is the size of the batched-sparse prize |
| 5 | greedy, temperature 0 | all of the above | output bytes | byte-identical to the unpatched run at every context. Any divergence means the M=1 routing change is not neutral |

## 7. Commands

    cd ~/inference-server/kernels/round3/qsa-verify
    ~/inference-server/kdev/bin/python probe_routing.py
    ~/inference-server/kdev/bin/python test_verify_sparse.py
    ~/inference-server/kdev/bin/python bench_arms.py
    ~/inference-server/kdev/bin/python bench_crossover.py

## 8. Limitations

Benches ran against a contended GPU. The routing proof calls the real predicates on stub attention
objects, so it proves the branch conditions, not a full forward. The dense numbers are plain
`mx.fast.scaled_dot_product_attention`; the production fallback also builds the indexer mask, so a
real dense verify would be worse. Nothing here has seen the model.
