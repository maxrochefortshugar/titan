# Vendored model code

`titan/adapters/mlx/vendor/` holds the Qwen3.8-Flash-Next language model as
Titan's own source. Nothing here imports oMLX, nothing is monkeypatched at
runtime, and every file is under a permissive licence with its notice intact.

## Origin

| source | version | licence | copied from |
|---|---|---|---|
| mlx-vlm `models/qwen4_exp/` | mlx-vlm 0.6.3 vendored copy, files dated 2026-09-10 | MIT | `/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/` |
| mlx-vlm `models/qwen3_5/`, `qwen3_5_moe/`, `base.py`, `cache.py`, `rope_utils.py`, `qwen3_vl/config.py` | mlx-vlm 0.6.3 | MIT | `.../framework-mlx-base/lib/python3.11/site-packages/mlx_vlm/` |
| mlx-lm `models/{base,cache,activations,switch_layers,gated_delta}.py` | mlx-lm 0.31.3 | MIT | `.../site-packages/mlx_lm/` |
| block-FP8 dequantisation (`vendor/fp8.py`) | oMLX 0.7.0.dev2 | Apache-2.0 | `omlx/patches/mlx_vlm_mtp/qwen38_fp8.py`, SPDX header preserved |

Licence texts: `LICENSE-mlx-vlm-MIT`, `LICENSE-mlx-lm-MIT`, `LICENSE-omlx-fp8-Apache-2.0`
next to this file. Titan itself is MIT (repository root `LICENSE`).

The package layout mirrors upstream (`vendor/mlx_vlm/models/...`,
`vendor/mlx_lm/models/...`) so a diff against a fresh install stays readable.
Absolute imports became relative; that rewrite is not listed per file below.

## File list and modifications

### `vendor/mlx_vlm/models/qwen4_exp/`

| file | lines | modifications |
|---|---:|---|
| `language.py` | 3055 | oMLX MTP checkpoint-prefix probe replaced with `titan.adapters.mlx.checkpoint.checkpoint_mtp_weight_prefix`. oMLX's `prompt_priming` host hook removed (Titan's `ModelState` owns the MTP hidden). `get_mtp_depth()` from oMLX's process global replaced by a `depth` argument. Process globals `_MTP_RUNTIME`, `_PLE_RUNTIME_MODE`, `_PLE_RUNTIME_MODEL_PATH` and their `configure_*` setters deleted; the MTP decision is now `mtp_runtime(config)` and the model path arrives on `TextConfig`. Resident PLE removed: `ShardedEmbedding` and `fuse_resident_ple_embeddings` deleted, `resolve_ple_runtime_mode` deleted, `get_ple_runtime_mode()` is the constant `"mmap"`. The MTP gate on the compiled and exact-hybrid decode paths reads a per-module flag stamped by `compile_hyper_connections(model, mtp_enabled)` instead of a global. `Qwen4ExpRMSNormGated.__call__` calls `gdn.norm_gate_fused`. `_omlx_*` attributes renamed `_titan_*`; `OMLX_QWEN4_*` env vars renamed `TITAN_QWEN4_*`. |
| `cache.py` | 2826 | unmodified apart from imports |
| `qsa_fast.py` | 734 | the five `omlx.custom_kernels` entry points (`glm_moe_dsa` indexer scores, top-k, sparse GQA; `decode_fast` SDPA; the `nax` availability probe) replaced by registry lookups `qsa.indexer_scores`, `qsa.topk_indices`, `qsa.sparse_gqa`, `qsa.decode_sdpa`. The guard structure, shape gates and fail-closed behaviour are unchanged. Env vars renamed. |
| `hc_fused.py` | 550 | `prefill_forward` gained two registry hooks: `hc.prefill_block_fused` for the whole block, `norm.grouped_rms_bf16` for the grouped norm alone. Metal kernel names `omlx_qwen4_*` renamed `titan_qwen4_*`. |
| `hc_projection.py` | 420 | kernel name renamed |
| `config.py` | 213 | four Titan fields added to `TextConfig`: `titan_model_path`, `titan_mtp_enabled`, `titan_mtp_checkpoint_prefix`, `titan_mtp_depth` |
| `qwen4_exp.py` | 312 | `Model` no longer subclasses the Qwen3.5 VLM model and no longer builds a vision tower; it is a plain `nn.Module` holding `language_model` and the optional `mtp`. FP8 dequant imported from `vendor/fp8.py`. MTP and PLE runtime read from the config. Resident-PLE fusion call removed. |
| `vision.py` | -- | not vendored |

### `vendor/mlx_vlm/models/qwen3_5/`, `qwen3_5_moe/`

| file | lines | modifications |
|---|---:|---|
| `qwen3_5/language.py` | 2664 | imports only |
| `qwen3_5/gated_delta.py` | 659 | `gated_delta_update` tries `gdn.chunk_scan` for prefill-shaped calls with scalar gating and no mask |
| `qwen3_5/config.py` | 153 | imports only |
| `qwen3_5/qwen3_5.py` | 18 | trimmed to `sanitize_key`; the `Model` wrapper builds a vision tower |
| `qwen3_5_moe/language.py` | 177 | the MoE tail routes through `moe.switch_weighted_sum` (ordinary layout) or `moe.weighted_sum_verify` (MTP target-verify layout); `_target_verify_switch_glu` understands the fused `gate_up_proj` |
| `qwen3_5_moe/config.py` | 125 | imports only |
| `qwen3_5/vision.py`, `qwen3_5_moe/vision.py`, `qwen3_5_moe/qwen3_5_moe.py` | -- | not vendored |

### `vendor/mlx_vlm/models/` (shared)

| file | lines | modifications |
|---|---:|---|
| `base.py` | 81 | trimmed to `BaseModelConfig`, `LanguageModelOutput`, `InputEmbeddingsFeatures`, `scaled_dot_product_attention` and the two mask helpers. PIL, transformers and the TurboQuant KV-cache branches dropped -- Titan constructs no TurboQuant cache, so the remaining body is the `mlx_lm` delegation the stock code fell through to. |
| `qwen3_vl_config.py` | 96 | trimmed copy of `qwen3_vl/config.py`: helpers plus the base `VisionConfig` and `TextConfig` the Qwen3.5 and Qwen4-Exp configs subclass |
| `cache.py` | 1271 | imports only |
| `rope_utils.py` | 857 | unmodified |

### `vendor/mlx_lm/models/`

| file | lines | modifications |
|---|---:|---|
| `switch_layers.py` | 277 | `QuantizedSwitchLinear.__call__` offers the routed gather to `moe.gather_qmm_int8` then `moe.gather_qmm_ws`, both with the stock `mx.gather_qmm` as their fallback. `SwitchGLU.__call__` uses a fused `gate_up_proj` when the loader produced one. |
| `gated_delta.py` | 283 | unmodified |
| `cache.py` | 1763 | unmodified |
| `base.py` | 137 | unmodified |
| `activations.py` | 43 | unmodified |

### `vendor/fp8.py`

63 lines, unmodified, Apache-2.0 header preserved. Not exercised by the oQ4e
checkpoint, which carries no `weight_scale_inv` tensors; kept so an FP8 export
of the same model loads.

## Ops the adapter asks the registry for

Each call site keeps its stock MLX path and uses the op only if the registry
returns one. `titan/adapters/mlx/kernels.py` is the only import site, and it
maps the adapter's site names to the names `titan.kernels` registers.

| adapter op | registry op | call site | wired | exactness (per the overlay report) |
|---|---|---|---|---|
| `hc.prefill_block_fused` | `hc_prefill` | `hc_fused.prefill_forward` | yes | bit-identical |
| `norm.grouped_rms_bf16` | `grouped_rmsnorm_bf16` | same site, norm only | yes | within 1 bf16 ULP |
| `gdn.norm_gate_fused` | `gdn_norm_gate` | `Qwen4ExpRMSNormGated.__call__`, T>1 | yes | bit-identical |
| `gdn.chunk_scan` | `gdn_chunk_scan` | `gated_delta_update`, prefill | yes | approximate; state rrmse 6.1e-7 |
| `moe.weighted_sum` | `moe_weighted_sum` | `Qwen3_5MoeSparseMoeBlock.__call__`, both layouts | yes | bit-identical in clone mode |
| `moe.gather_qmm_int8` | `moe_gather_int8` | `QuantizedSwitchLinear.__call__`, tried first | yes | approximate; 0.64-1.4% of output RMS, +11.3 GB |
| `moe.gather_qmm_ws` | `moe_gather_ws` | same site, second | yes | bit-identical |
| `ple.packed_rows_lookup` | `ple_packed_lookup` | `DiskBackedShardedEmbedding.__call__` | yes | bit-exact |
| `sample.fast_topk` | `topk_radix` | the drafter (not a vendored site) | yes | value multiset identical to `mx.topk` |
| `qsa.gathered_batched` | `qsa_gathered_attention` | `Qwen4ExpAttention.__call__`, batched decode | **no** | padded route within 1 ULP, loop route exact |
| `ple.packed_rows_prefetch` | -- | `DiskBackedShardedEmbedding.prefetch` | no | -- |
| `qsa.indexer_scores`, `qsa.topk_indices`, `qsa.sparse_gqa`, `qsa.decode_sdpa` | -- | `qsa_fast.py` | no | -- |

The unwired rows are deliberate, and each is a different kind of gap.

The four `qsa_fast.py` seams are the narrow ABIs oMLX filled from its own Metal
extension. Titan has no equivalent yet, so the guards fail closed and those
paths run the MLX ops -- slower, and identical.

`qsa.gathered_batched` is the one that is worth finishing. The kernel exists,
but it replaces the whole attention block and wants the pooled index bank in
phased slot space (`QSAGeometry`, `QSAConfig`), which the vendored attention
does not build -- round3/qsa-batched built it in its patch. Wiring it means
adding that bank construction to `Qwen4ExpAttention.__call__` for a
`BatchQSAKVCache`, and it should not be guessed at: the padded route is only
within one ULP when every row clears the gather crossover, and a wrong phase
offset would be silently wrong rather than loud. Until then batched decode
takes the dense path.

## What is deliberately absent

* No vision tower. The checkpoint's 333 `vision_tower.*` tensors are dropped by
  the loader plan, which says so per tensor.
* No resident PLE mode. The 320,001,536-row n-gram table is read from SSD in
  rows layout only; the 32 GB device-resident variant does not exist in this
  tree and cannot be selected.
* No process-global runtime configuration. The MTP decision, the MTP depth and
  the checkpoint path travel on `TextConfig`.
