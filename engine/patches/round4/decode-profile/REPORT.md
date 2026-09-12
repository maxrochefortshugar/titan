# Decode profile: per-cycle instrumentation for the MTP path

Workstream `kernels/round4/decode-profile/`, 2026-09-12. No model loaded, no safetensors opened,
no server started, ports 8083/8084 untouched. The only GPU work is the synthetic harness below, a
few MB of bf16 toy tensors. `staging/GPU_FREE` never existed during the work, so there are no
machine timings here beyond the toy harness and the audit's existing numbers.

## 1. What was built

`patch.py`, one idempotent install, **import time**, loaded by path. With `OMLX_DECODE_PROFILE`
unset, `install()` returns False and touches nothing: no wrapper bound, `mx.eval` and friends keep
their identity, no directory created. The test asserts that. It wraps the functions bracketing one
chain cycle in `omlx/patches/mlx_lm_mtp/batch_generator.py` (`BG`) and the classes the cycle calls.

| wrapped | file:line of the stock code | what it yields |
|---|---|---|
| `BG._run_verify_cycle_chain` | `BG:2917-3149` | cycle wall, one record per cycle |
| `BG._call_backbone` | `BG:1520-1572`, called at `BG:2946` | verify forward dispatch, M, context length, KV bytes |
| `BG._chain_rollback` / `_clear_rollback` | `BG:3200-3260` / `BG:1575` | commit phase, rollback taken or not |
| `BG._chain_next_drafts` | `BG:2303-2434` | draft chain |
| `BG._log_mtp_stats` | `BG:2831-2870` | per-request dump, next to the `MTP[...]` line |
| `Qwen4ExpDecoderLayer.__call__` | `LANG:2795` | `bb_gdn` vs `bb_attn` by `self.is_linear` |
| `Qwen3_5MoeSparseMoeBlock.__call__` | `qwen3_5_moe/language.py` via `LANG:2785` | `bb_moe`, nested inside the layer bucket |
| `Qwen4ExpPLELayer.__call__`, `Qwen4ExpNGramEmbedding.__call__` | `LANG:2731`, `LANG:2666` | `bb_ple`, `ngram_lookup` plus rows read |
| packed `assemble_host` / `dequantize_host` (`ple-fix/patch.py:178` / `:207`), or stock `_assemble` / `_host_indices` (`LANG:2248` / `LANG:2265`) | | `ngram_pread` (host page reads) vs `ngram_upload` (device) |
| `lm_head`, instance-keyed dispatcher on its class | `LANG:3048` | `lm_head`, rows per pass |
| `LanguageModel.mtp_forward` | `LANG:3101` | `mtp_head1` (fold + full head) vs `mtp_headk` |
| `rollback_speculative_cache`, `_restore_ple_state` | `LANG:3202`, `LANG:3171` | GDN replay + KV trim vs PLE snapshot restore |
| `mx.eval`, `mx.async_eval`, `mx.synchronize`, `mx.array.tolist`, `mx.array.item` | | every host sync, counted and timed, attributed to the open phase |

`mx.array.tolist` and `.item` are settable on the nanobind type, so the acceptance sync at
`BG:2988` is timed directly rather than inferred.

Each cycle is partitioned into `pre`, `verify_dispatch`, `accept`, `commit`, `head_gap`, `draft`,
`post` from timestamps taken in those wrappers. The partition is exact by construction, so the
stage totals equal the cycle wall to the microsecond, and the sub-stages sit inside it as a
separate, nested breakdown with its own coverage ratio. A sub-stage entered while the phase is
`draft` or `commit` gets that prefix, so the MTP head's own decoder layer, MoE block, embedding
and head land in `draft.*` instead of inflating the backbone.

**Sync mode.** `OMLX_DECODE_PROFILE_SYNC=1` adds `mx.synchronize()` at the end of every sub-stage
and at the end of the verify forward, the commit and the draft chain. That turns dispatch time
into device time, and it serialises a pipeline built to overlap: on the toy it inflates the run
1.5 to 2.1x. Use it for attribution, never for a throughput number.

**Bytes.** Per cycle the record carries modelled bytes by category: dense weights (2.34 GB from
audit section A), distinct experts touched, `512 * (1 - (1 - 10/512)^M) * 48 * 2.765 MB`,
`lm_head` passes x 0.63 GB, KV plus GDN state read from `cache.nbytes()` (no sync), and n-gram
rows read x row stride. Every constant is env-overridable. The draft head's own weights are not
modelled; only its head passes are.

**Output.** One JSON per request under `~/inference-server/staging/decode-profile/` with every
cycle record, plus a `DPROF[uid] cycles=.. cycle=..ms verify=.. accept=..(sync ..) commit=.. draft=..
| gdn=.. attn=.. moe=.. ple=.. ngram=.. head=.. draft1=.. draftk=.. rb=.. syncs/cycle=.. ctx=..`
line beside the `MTP[...]` line.

| env | default | effect |
|---|---|---|
| `OMLX_DECODE_PROFILE` | 0 | the whole install |
| `OMLX_DECODE_PROFILE_SYNC` | 0 | device time instead of wall, slower |
| `OMLX_DECODE_PROFILE_DIR` | `~/inference-server/staging/decode-profile` | JSON destination |
| `OMLX_DECODE_PROFILE_MAX_CYCLES` | 4096 | per-request record cap |
| `OMLX_DPROF_B_*`, `OMLX_DPROF_{MOE_LAYERS,EXPERTS,TOPK}` | audit values | byte model |

## 2. Test

`test_profile.py` builds a miniature qwen4_exp-shaped model (6 decoder layers, 4 linear and 2
attention, a MoE block each, one PLE layer with an n-gram table, a 512-row head, an MTP head with
rollback and a PLE snapshot restore) and registers a miniature `batch_generator` under the real
module name, its body `exec`'d into the module namespace so every intra-module call is a global
lookup, as in the shipped file. Three subprocess modes: flag off, on, sync.

| check | result |
|---|---|
| [0] flag off: `install()` False, no wrapper bound, no file written | pass |
| [1] on/sync: True, idempotent, wrappers bound | pass |
| [3] phases sum to the cycle wall | median and max residual **0.0000%** |
| [4] layers + head + embed cover the verify forward | 91.5% (wall), 95.0% (sync) |
| [5] all 6 layers timed, MoE/PLE/n-gram split present, rows counted | pass |
| [6] draft head split 1 + (k-1) | 1 / 2 at depth 3 |
| [7] acceptance host sync timed as `accept.tolist`; bytes populated | pass |
| [8] sync mode slower | 1.48 to 2.08x |

26/26. Instrumentation overhead is 0.21 us per wrapped call measured directly; at roughly 130
wrapped calls per real cycle that is ~0.03 ms, under 0.1% of a 34 ms cycle. The toy's own off/on
ratio (0.95 to 1.10x over three runs) is noise on a 1.8 ms cycle and proves nothing better.

`analyze.py RUN [--ctx-min N] [--ctx-max N] [--k K] [--all] [--csv]` prints the stage budget
(median ms, % of cycle, modelled GB/s and % of the 718 GB/s ceiling, host syncs per cycle, tokens
per cycle and the implied rate); `analyze.py --compare A B` prints both tables and the per-stage
delta, sorted by absolute change.

## 3. Workbench plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, greedy, two rounds. Every row
carries the deployed decode patches (`OMLX_MTP_SHORTLIST_DRAFT=1`, `OMLX_PLE_PACKED=1`, and
whichever round-3 items are live). Add to `OMLX_ROUND2_IMPORT_PATCHES` after the other
`_chain_next_drafts` wrappers and **before copy-lane**, which must stay outermost:

    OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft,\
    $R3/mtp-depth/patch.py:install_conf_depth,$R3/mtp-park/patch.py:install_park_policy,\
    $R4/decode-profile/patch.py:install,$R3/copy-lane/patch.py:install_copy_lane"
    export OMLX_DECODE_PROFILE=1

| step | env | prompt | record |
|---|---|---|---|
| 0 | profile off | `decode_bench.py --tag base` | tok/s per prompt: the reference the profiled runs must not be far from |
| 1 | `OMLX_DECODE_PROFILE=1` | `decode_bench.py --tag prof` | tok/s again (overhead must be under 1%), then `analyze.py <dir>` per prompt |
| 2 | as 1 | `e2e_cold.py --tag prof64k` (the CAP-theorem tail after a 64k cold prefill) | the 64k stage table, `--ctx-min 60000` |
| 3 | `..._SYNC=1` | the `prose` prompt and the 64k tail | device time per stage; compare with step 1/2 via `--compare` |
| 4 | as 1, `set_mtp_depth(0)` or a parked request | 64k tail | the plain-step budget, `--k 0`, against the same stages |
| 5 | as 1 | 8k, 16k, 32k, 64k tails | the context slope of `bb_attn`, `accept` and `ngram_lookup` |

The numbers must answer: where do the 34.1 ms of a 64k cycle go against 17.7 ms for a plain step,
and how much of the 9.3 ms marginal verify row is the backbone rather than the head; what fraction
of `accept` is the host sync wait, and how many syncs a cycle really costs; what rollback costs,
split GDN replay against PLE snapshot restore; what the n-gram lookup costs, split host page reads
against upload, and whether it scales with M; what fraction of the backbone is GDN against
attention against MoE; and the achieved GB/s per stage against 718.

## 4. Limitations

Nothing here has seen the real model. Wall mode measures dispatch, and `LANG:2905`'s per-layer
`mx.async_eval` means a layer bucket absorbs whatever queue backpressure it happens to meet, so
only sync mode attributes device time, and sync mode is a different machine. Expert bytes are
modelled from an independence assumption that real routing violates; measuring them exactly needs
a sync per cycle and was left out. The `accept` phase is the whole span from the forward returning
to the commit, so it carries the logits processors and the in-graph acceptance math as well as the
sync; the `accept.*` sync rows separate the wait itself. Sub-stage coverage of the verify forward
is 91 to 95% on the toy, and the remainder (masks, the hyper-connection mixer, the trunk norm) is
unattributed. Only the chain cycle is instrumented: `_run_verify_cycle_legacy` and
the standard non-MTP step are not, so step 4's plain-step budget must come from a depth-0 cycle.

## 5. Commands

    cd ~/inference-server/kernels/round4/decode-profile
    ~/inference-server/kdev/bin/python test_profile.py
    ~/inference-server/kdev/bin/python analyze.py ~/inference-server/staging/decode-profile
    ~/inference-server/kdev/bin/python analyze.py --compare RUN_A.json RUN_B.json
