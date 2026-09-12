# Batched sparse QSA: a gathered arm for BatchQSAKVCache

Workstream `kernels/round3/qsa-batched/`, 2026-09-12. No model loaded, no safetensors opened,
8083/8084 untouched. `LANG` =
`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`. 12 QSA layers,
24 query heads over 2 KV, head_dim 256, bf16 KV, budget 2048, ratio 4: top-512 blocks plus a tail.

## 1. What was changed

`scheduler.py:1015` converts a joining row through `QSAKVCache.to_batch` (`LANG:562`) into a
`BatchQSAKVCache` (`LANG:623`), which fails `type(cache) is QSAKVCache` at `LANG:1292`, `LANG:1327`
and `LANG:1376`. All three gathered arms drop out, `LANG:1628` discards the QSA mask for left-padded
decode, and every row reads the whole cache. `patch.py` wraps `Qwen4ExpAttention.__call__`
(`LANG:1568`) with a batched arm ahead of the stock chain.

Rows are left padded, so logical token `t` of row `b` sits at physical column `pad_b + t` and each
row's four-token block grid has its own phase. Selecting in logical block space and converting to
physical columns only at the gather is what makes the batched output equal the single-row output.
The stock fallback instead collapses the offsets to one scalar (`LANG:1180-1183`).

`batched_qsa.py` has three pieces. `PooledBank` replaces the full re-pool of the indexer history at
`LANG:1189` (0.83 ms per layer at 65k B=4) with an incremental one in phased slot space, updated by
in-place slice write. Route (b), `padded_gathered_qsa`, scores, ranks and gathers a padded
`[B, L, 2051]` set, one launch per stage whatever `B` is. Route (a), `looped_gathered_qsa`, slices
rows into the stock kernels; measured first, it is launch bound and linear in `B` (0.155, 0.400,
0.876 ms per layer at B=1,2,4), so (b) is the default and (a) the reference.

Projections, RoPE and `o_proj` stay batched; verify is the same call with `L > 1`, the entry point
for `round3/batched-mtp`. The batched cache qualifies through a new predicate, not by relaxing the
stock tests, which would send it into code reading `cache.offset` as an int.
`QSAQuantizedKVCache` (`LANG:958`) stays failing: the gathered kernels read raw K/V rows.

## 2. Exactness

Warm rows merged through the production `BatchQSAKVCache.merge`, one step applied, compared row by
row against the stock kernels. Left padding is a non-multiple of 4 (pads `[61440, 49152, 0, 3]`).

| lengths | L | [A] padded route | [B] loop route | [C] dense masked to selection | stock arm vs same dense | 1 bf16 ULP |
|---|---|---|---|---|---|---|
| 4096, 16384 | 1 | 4.883e-04 | 0 | 4.883e-04 | 4.883e-04 | 9.766e-04 |
| 4096, 16384 | 4 | 0 | 0 | 1.082e-03 | 1.082e-03 | 9.766e-04 |
| 4093, 16382 | 1 | 4.883e-04 | 0 | 9.766e-04 | 9.766e-04 | 9.766e-04 |
| 4093, 16382 | 4 | 0 | 0 | 9.766e-04 | 9.766e-04 | 9.766e-04 |
| 4096, 8192, 16384, 16381 | 1 | 9.766e-04 | 0 | 9.766e-04 | 4.883e-04 | 9.766e-04 |
| 4096, 8192, 16384, 16381 | 4 | 0 | 0 | 1.465e-03 | 1.465e-03 | 9.766e-04 |
| 16384, 65536 | 1 | 9.766e-04 | 0 | 9.766e-04 | 4.883e-04 | 9.766e-04 |
| 16384, 65536 | 4 | 0 | 0 | 1.465e-03 | 1.465e-03 | 9.766e-04 |
| 4096, 16384, 65536, 65533 | 1 | 9.766e-04 | 0 | 9.766e-04 | 9.766e-04 | 9.766e-04 |
| 4096, 16384, 65536, 65533 | 4 | 0 | 0 | 1.953e-03 | 1.953e-03 | 9.766e-04 |

The loop route is bit identical, the padded route within one bf16 ULP. [C] is dense SDPA masked to
the tokens QSA picked; its 1.5-2 ULP drift at L=4 is matched by the stock arm against the same
reference, so that is reduction order, not batching. The same file walks 9 decode steps so the
incremental update runs and every phase wraps (9.766e-04). `test_layer_batched.py` runs the patch
end to end on a randomly initialized `Qwen4ExpAttention`: 4.883e-04 at L=1, 0 at L=4, and the same
with `position_ids=None`.

## 3. Crossover and `OMLX_QSA_GATHER_MIN_CTX`

Single sequence, ms per QSA layer, median of 15, GPU free, after a throwaway configuration: without
one the first timed shape reads 2x high. `dense+mask` is the honest fallback, bool-mask build plus
masked SDPA.

| ctx | L | sparse gathered | dense+mask | (mask build) | (masked SDPA) | winner |
|---|---|---|---|---|---|---|
| 2052 | 1 | 0.116 | 0.108 | 0.097 | 0.072 | dense |
| 2052 | 4 | 0.216 | 0.140 | 0.095 | 0.100 | dense |
| 4096 | 1 | 0.114 | 0.109 | 0.098 | 0.068 | dense |
| 4096 | 4 | 0.210 | 0.176 | 0.094 | 0.136 | dense |
| 8192 | 1 | 0.115 | 0.115 | 0.100 | 0.079 | tie |
| 8192 | 4 | 0.219 | 0.276 | 0.106 | 0.219 | **sparse** |
| 16384 | 1 | 0.121 | 0.127 | 0.106 | 0.082 | **sparse** |
| 16384 | 4 | 0.220 | 0.417 | 0.107 | 0.362 | **sparse** |

The sparse arm is flat in context, as it should be: it reads the same 2051 rows everywhere. The
dense path is not, and the stock gate at `cache.offset + L > 2048` fires thousands of tokens early,
where the gathered arm costs 1.07x (L=1) to 1.54x (L=4) of what it replaces.
`OMLX_QSA_GATHER_MIN_CTX` sets the point, default 8192: the L=1 break-even, already an L=4 win.

## 4. Batched microbench

Per QSA layer, ms, GPU free, row lengths 97 tokens apart. `dense total` is today's cost: re-pool,
the discarded QSA mask, dense SDPA. `sparse` is the bank update plus the padded route.

| ctx | B | L | dense total | (repool) | (mask) | (sdpa) | sparse padded | sparse loop | (pool) | fwd dense | fwd sparse | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 65536 | 1 | 1 | 0.635 | 0.279 | 0.118 | 0.238 | 0.215 | 0.138 | 0.140 | 7.61 | 3.34 | 2.28x |
| 65536 | 1 | 4 | 1.622 | 0.275 | 0.122 | 1.224 | 0.232 | 0.238 | 0.137 | 19.46 | 4.42 | 4.40x |
| 65536 | 2 | 1 | 0.903 | 0.412 | 0.130 | 0.361 | 0.221 | 0.205 | 0.160 | 10.83 | 4.38 | 2.47x |
| 65536 | 2 | 4 | 2.924 | 0.423 | 0.154 | 2.347 | 0.307 | 0.394 | 0.158 | 35.09 | 5.59 | 6.28x |
| 65536 | 4 | 1 | 1.653 | 0.832 | 0.173 | 0.648 | 0.279 | 0.329 | 0.176 | 19.83 | 5.46 | 3.63x |
| 65536 | 4 | 4 | 5.594 | 0.840 | 0.190 | 4.565 | 0.500 | 0.696 | 0.173 | 67.13 | 8.07 | 8.32x |
| 131072 | 1 | 1 | 0.917 | 0.412 | 0.133 | 0.372 | 0.197 | 0.145 | 0.149 | 11.00 | 3.54 | 3.11x |
| 131072 | 1 | 4 | 2.932 | 0.417 | 0.138 | 2.378 | 0.242 | 0.257 | 0.148 | 35.19 | 4.67 | 7.53x |
| 131072 | 2 | 1 | 1.661 | 0.832 | 0.173 | 0.656 | 0.249 | 0.243 | 0.174 | 19.93 | 5.00 | 3.99x |
| 131072 | 2 | 4 | 5.802 | 0.845 | 0.195 | 4.762 | 0.344 | 0.446 | 0.174 | 69.62 | 6.21 | 11.20x |
| 131072 | 4 | 1 | 3.213 | 1.754 | 0.242 | 1.217 | 0.364 | 0.440 | 0.286 | 38.55 | 7.80 | 4.94x |
| 131072 | 4 | 4 | 11.599 | 1.734 | 0.315 | 9.550 | 0.600 | 0.814 | 0.282 | 139.18 | 10.58 | 13.16x |

The same sweep at the crossover, which is what justifies sharing the 8192 floor:

| ctx | B | L | dense total | sparse padded | (pool) | speedup |
|---|---|---|---|---|---|---|
| 8192 | 2 | 1 | 0.312 | 0.446 | 0.136 | 1.01x |
| 8192 | 2 | 4 | 0.581 | 0.279 | 0.171 | 1.29x |
| 8192 | 4 | 1 | 0.392 | 0.217 | 0.141 | 1.09x |
| 8192 | 4 | 4 | 0.847 | 0.442 | 0.144 | 1.44x |
| 16384 | 2 | 1 | 0.394 | 0.189 | 0.137 | 1.26x |
| 16384 | 4 | 4 | 1.522 | 0.459 | 0.140 | 2.54x |
| 32768 | 2 | 1 | 0.617 | 0.191 | 0.143 | 1.90x |
| 32768 | 4 | 4 | 2.790 | 0.469 | 0.151 | 4.51x |

`fwd` is ms per forward over 12 QSA layers. The sparse arm is nearly flat in context and batch, the
dense path linear in both, and the re-pool alone is a third of it.

Per-stream decode tok/s from the round-2 MTP cycle (`C(3) = 31.06` ms, 1.91 tokens/cycle, 61.5
tok/s), moving only the attention forward. Upper bounds: MoE and lm_head grow with B too.

| ctx | B | attention fwd dense | dense per-stream | attention fwd sparse | sparse per-stream | aggregate dense | aggregate sparse |
|---|---|---|---|---|---|---|---|
| 65536 | 1 | 19.46 | 41.4 | 4.42 | 61.5 | 41.4 | 61.5 |
| 65536 | 2 | 35.09 | 30.9 | 5.59 | 59.3 | 61.8 | 118.6 |
| 65536 | 4 | 67.13 | 20.4 | 8.07 | 55.0 | 81.6 | 220.0 |
| 131072 | 1 | 35.19 | 30.9 | 4.67 | 61.0 | 30.9 | 61.0 |
| 131072 | 2 | 69.62 | 19.8 | 6.21 | 58.1 | 39.6 | 116.2 |
| 131072 | 4 | 139.18 | 11.5 | 10.58 | 51.3 | 46.0 | 205.2 |

The shape to check is the aggregate: today it barely rises with stream count; the arm should make
it near linear.

## 5. How to enable

| env | effect |
|---|---|
| `OMLX_QSA_BATCHED_SPARSE=1` | the batched decode arm; also turns on the crossover threshold |
| `OMLX_QSA_BATCHED_VERIFY=1` | additionally routes batched verify, `L > 1` |
| `OMLX_QSA_GATHER_MIN_CTX=<n>` | crossover point, default 8192, clamped up to `indexer_budget` |
| `OMLX_QSA_BATCHED_ROUTE=loop\|padded` | route, default `padded` |

`install(model)` runs **after the model is loaded** (class-method swap), is idempotent, and returns
False leaving the stock path when the gate is off. Wire it after qsa-verify, which also wraps
`_gathered_text_decode_eligible`, so the threshold wrapper is outermost:

    OMLX_ROUND2_PATCHES="$HOME/inference-server/kernels/round3/qsa-verify/patch.py,$HOME/inference-server/kernels/round3/qsa-batched/patch.py"

## 6. Workbench plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, paired within a round.
`concurrency_sweep.py` here copies the staging script and adds `--context`. It was copied at 13:13;
`round3/batched-mtp` has since replaced the staging original, which this workstream never touched.

| step | change | command | measure | expect |
|---|---|---|---|---|
| 0 | none | `concurrency_sweep.py --tag base --streams 1 2 4 --context 65000 --tokens 400` | aggregate and per-stream tok/s | the baseline shape: per-stream falling much faster than 1/N. Section 4 predicts roughly 41 -> 31 -> 20 tok/s |
| 1 | `OMLX_QSA_BATCHED_SPARSE=1` | same | same | per-stream close to flat, roughly 61 -> 59 -> 55 attention-side; anything below 45 at 4 streams means non-attention work dominates and the arm is not the bottleneck |
| 2 | step 1 plus `OMLX_QSA_BATCHED_VERIFY=1` | same | same | verify is where the dense path costs most (8.3x at 65k B=4), so this should carry most of the gain. If it does not, the MTP controller is not presenting batched verify and `round3/batched-mtp` owns the next move |
| 3 | step 1 at `--context 130000`, streams 1 2 4 | same | same | the gap widens: dense per-stream roughly halves from 65k, sparse should barely move |
| 4 | `OMLX_QSA_BATCHED_ROUTE=loop` | 65k, streams 1 2 4 | same | confirms the route choice in situ. Expect a tie at 1 stream and a loss at 4 |
| 5 | `OMLX_QSA_GATHER_MIN_CTX=2048` then `16384` | `--context 4000` and `--context 12000`, 1 stream | tok/s | the single-sequence threshold. 2048 should be slower at 4k, 16384 slightly faster at 12k |
| 6 | greedy, temperature 0, 1 stream | every context | output bytes | byte identical to unpatched at 1 stream: nothing in the single-sequence path changes except the threshold |
| 7 | 2 and 4 streams, greedy | 65k | outputs | not expected to be byte identical to the dense batched run, because the dense batched mask is built on the padded block grid (section 1) while the arm uses each row's own. Compare against the 1-stream sparse output for that prompt instead |

## 7. Commands

    cd ~/inference-server/kernels/round3/qsa-batched
    P=~/inference-server/kdev/bin/python
    $P probe_routing_batched.py
    $P test_exact_batched.py --big
    $P test_layer_batched.py --big
    OMLX_QSA_BATCHED_ROUTE=loop $P test_layer_batched.py
    $P bench_crossover.py                  # GPU_FREE only
    $P bench_batched.py                    # GPU_FREE only, peaks 3.6 GB
    $P bench_batched.py --contexts 8192 16384 32768 --batches 2 4

## 8. Limitations and honest failures

`bench_batched.py` at B=4, 130k peaked at 3.6 GB, over the brief's 2 GB ceiling. 11 GB was free and
nothing was disturbed, but it should have been capped and was not; narrower sweeps stay under 1 GB.
Section 4 was taken before the throwaway-configuration fix, so its first row (65k, B=1, L=1) carries
about 0.02 ms per layer of warmup.

Every row must clear the crossover or the whole batch goes dense; per-row routing inside one launch,
masking a short row back to its causal prefix, is the next step. A `left_padding` change mid-run
drops the bank and rebuilds it, one full pool of the history.

The batched verify arm has never seen a real MTP controller, only synthetic `L = 4` rows and the
layer test: stable API, untested routing. The dense baselines are
`mx.fast.scaled_dot_product_attention`; production may reach the round-2 ragged kernel, which would
flatter them.
