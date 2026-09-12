# Fine store boundaries that actually store: root cause and fix

Workstream `round4/cache-boundary`. Read-only against production, no model load, no GPU work
beyond a few hundred KB of synthetic tensors. Env var: `OMLX_CACHE_FINE_TAIL=512`
(`OMLX_CACHE_COARSE_CHUNK` defaults to 2048). Import-time install.

## 1. Why round 3 failed

Not the restore path, and not the snapshots. The store side rejected everything.

`config.paged_cache_block_size` does three jobs. It is the prefix match and flooring
granularity (`cache/paged_cache.py:1035` `get_computed_blocks`), the grid on which prefill
chunks are cut and GDN snapshots emitted (`scheduler.py:3686, 3830, 3973, 5498, 5602, 5735`
through `prefill_boundaries.py:4` and `:19`), and the grid on which `store_cache` demands a
committed recurrent sidecar for **every** block. That third job is the one round 3 missed.

The chain: `cache/prefix_cache.py:1285` calls `_commit_split_gdn_checkpoint`
(`prefix_cache.py:410`) for each newly allocated block, which calls
`_BoundarySnapshotProvider.commit_gdn_checkpoint` (`scheduler.py:1743`), which calls
`take_staged_file(request_id, token_count)` (`cache/boundary_snapshot_store.py:482`) and
returns False when no snapshot was staged at exactly that token count. On False,
`prefix_cache.py:1294-1329` logs "Rejecting split-GDN placeholder block N", deletes the SSD
block, pops it from the table and **breaks**. With block size 512 and snapshots still only at
2048 multiples, the very first block (end 512) had no staged file, so the store truncated to
zero blocks. The `storing 26112/26482 ... 12 intermediate snapshots` lines in
`server-plain.log.133730` are the scheduler's optimistic pre-store log; the truncation happens
after. Hence `cached=0` on every warm turn. The dedup branch (`prefix_cache.py:1024-1046`) has
the same gate, so a second attempt could not repair it either.

Restore was never the problem: the split-GDN endpoint search at `prefix_cache.py:3186-3330`
already walks back from the last block to the newest one whose sidecar loads and truncates the
chain there, counting `_gdn_checkpoint_walkbacks`.

Where the 2048 assumption lives:

| site | role |
|---|---|
| `scheduler.py:2848` `_ARRAYS_CACHE_BLOCK_SIZE = 2048` | pins block size in `_enlarge_block_size_for_arrays_cache` (`:2850-2901`) |
| `scheduler.py:2749` `_POOLING_ROTATING_BLOCK_SIZE = 2048` | same for pooling/rotating models, not this one |
| `scheduler.py:3686, 5498` | `clamp_prefill_chunk_to_boundary`, chunk width |
| `scheduler.py:3830, 3973, 5602, 5735` | `should_emit_prefill_boundary`, snapshot emission |
| `scheduler.py:7146-7160` | `_maybe_capture_boundary_snapshot`, decode-time captures |
| `scheduler.py:7038` | `_enable_mtp_boundary_alignment` |
| `scheduler.py:7253-7285` | `_get_boundary_store_override`, the `tc % block_size == 0` gate that produces `boundary_snapshot_unavailable available_boundaries=0` |
| `scheduler.py:7698` | store fallback floor `(len // block_size) * block_size` |
| `cache/paged_cache.py:1035` | lookup floors to whole blocks |
| `cache/prefix_cache.py:851-870` | store drops the trailing partial block |
| `cache/prefix_cache.py:1285-1330` | per-block sidecar commit, the failure |
| `cache/paged_ssd_cache.py:2812` | `gdn_cache_signature_for` embeds `block_size` in the sidecar signature, so changing it invalidates existing sidecars and blocks |

## 2. The change

Four small pieces, all in `patch.py`:

1. chunking: 2048 forwards for the prompt body, and the last chunk is cut once more so it ends
   at the highest 512 multiple at or below the prompt end. One extra snapshot per prompt.
2. `paged_cache_block_size` drops to 512 after `_enlarge_block_size_for_arrays_cache` runs, so
   matching, storing and restoring can address that boundary.
3. **the fix**: a block whose end is off the coarse grid may be stored without a committed
   sidecar instead of truncating the chain. Its QSA KV is complete and sliceable; it simply
   cannot be a restore endpoint, and the walk-back already handles that. A missing sidecar on a
   2048 multiple is still an anomaly and still truncates, as in stock.
4. decode-time captures and MTP commit alignment stay on the coarse grid, so lowering the block
   size does not quadruple 110 MiB snapshots during generation. Cost: an output-inclusive store
   still floors its decode tail to 2048.

Cost per prompt: one extra forward launch, one extra sidecar, 4x the paged block count (same KV
bytes, 12 MiB per file instead of 48 MiB).

| quantity | stock | patched |
|---|---|---|
| GDN sidecars, 30k prompt | 14 x 110.2 MiB = 1.51 GiB | 15 x 110.2 MiB = 1.61 GiB |
| snapshot pending buffer peak | 110.2 MiB in flight | 110.2 MiB in flight (budget 512 MB) |
| SSD KV, 30k prompt | 703 MiB in 14 blocks | 738 MiB in 60 blocks |
| uniform 512 blocks over 64k (rejected) | n/a | 128 snapshots, 13.8 GiB |

## 3. Test

`test_fine_tail.py` drives the real `PagedCacheManager`, `PagedSSDCacheManager`,
`BlockAwarePrefixCache`, `BoundarySnapshotSSDStore` and `_BoundarySnapshotProvider` with a
fabricated two-layer cache (one ArraysCache layer for the 36 GDN layers, one KVCache layer for
the 12 QSA layers) whose every value is a pure function of the token count. Prefill 10,167,
store, then a 13,084-token request sharing the first 10,167, lookup, reconstruct.

| arm | stored | cached on the warm turn |
|---|---|---|
| stock, block 2048 | 8192 | 8192 |
| round-3 repro (block 512, stock commit gate) | 0 | 0 |
| patched, `OMLX_CACHE_FINE_TAIL=512` | 9728 | 9728 |

The repro arm reproduces the production log line verbatim, which pins the root cause. The
patched arm restores GDN state and KV bit-identical to a fresh prefill to 9728 (byte compare on
the raw buffers), five aligned boundaries survive so the trailing-partial path never reports
`available_boundaries=0`, and prompts of 1500, 2048 and 4096 emit exactly the same boundaries
and floor to the same length as stock.

```
cd ~/inference-server/kernels/round4/cache-boundary
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
  ~/inference-server/kdev/bin/python test_fine_tail.py
```

Verified separately with the bundled interpreter: `install()` returns False without the env
var, True and idempotent with it, `clamp(4096, cache_tokens=0)` stays 2048,
`clamp(1975, cache_tokens=8192)` returns 1536 so the chunk ends at 9728, a 300-token tail passes
through unsplit, and `uninstall()` restores the stock 512 clamp.

## 4. Expected saving

Recomputed tokens on turn k are the new text plus whatever turn k-1 discarded at store time.
The measured discard is a median of 1155 and p90 1734 tokens; on a 512 grid it becomes uniform
on [0, 512), mean 256.

| | stock | patched | saved |
|---|---|---|---|
| median recomputed | 2000 | 1101 | 899 tok, 0.56 s at 1600 tok/s |
| p90 recomputed | 3343 | 1865 | 1478 tok, 0.92 s |

On the six-turn probe (prompts 25043 to 32233, +1438 per turn), warm turns 2 to 6 recompute
11,617 tokens today against a predicted 8,545, down 26.4%, 1.9 s over five turns. The larger
effect is on the 603 stores currently dropped with `available_boundaries=0`: a suffix that
crosses no 2048 boundary crosses a 512 one, so turn 2 of the probe starts storing instead of
forcing turn 3 to recompute 3343.

## 5. Workbench plan

Needs `~/inference-server/staging/GPU_FREE`, which does not exist right now. Never touch 8083.
Because the sidecar signature embeds the block size, give the patched arm a cold cache dir or
accept that the first turn of each arm is cold.

```
cd ~/inference-server/staging
BASE=(OMLX_PLE_PACKED=1 OMLX_PLE_PACKED_MODE=rows OMLX_QWEN4_BF16_NORM=1 \
      OMLX_QWEN4_GDN_NORM_GATE=1 OMLX_WSUM_TOPK10=1 OMLX_WSUM_TOPK10_MIN_TOKENS=64 \
      OMLX_MTP_SHORTLIST_DRAFT=1 OMLX_MOE_INT8_PREFILL=1 OMLX_MOE_INT8_SKIP_DOWN=1 \
      OMLX_GUARD_GB=110)
POST=~/inference-server/kernels/round2/gdn-norm/patch.py,~/inference-server/kernels/round2/wsum10/patch.py
IMP=~/inference-server/kernels/round2/mtp/patch.py:install_shortlist_draft

# arm 1
env "${BASE[@]}" OMLX_ROUND2_PATCHES="$POST" OMLX_ROUND2_IMPORT_PATCHES="$IMP" \
  ./staging.sh start plain && sleep 20
~/inference-server/kdev/bin/python prefill_ab.py --tag cache-base4 --tokens 32000 --turns 6

# arm 2
env "${BASE[@]}" OMLX_ROUND2_PATCHES="$POST" \
  OMLX_ROUND2_IMPORT_PATCHES="$IMP,~/inference-server/kernels/round4/cache-boundary/patch.py" \
  OMLX_CACHE_FINE_TAIL=512 ./staging.sh start plain && sleep 20
~/inference-server/kdev/bin/python prefill_ab.py --tag cache-fine-tail --tokens 32000 --turns 6
./staging.sh stop
```

Expected cached counts per turn:

| turn | prompt | base (measured) | fine tail (predicted) |
|---|---|---|---|
| 1 | 25043 | 0 | 0 |
| 2 | 26481 | 24576 | 24576 |
| 3 | 27919 | 24576 | 26112 |
| 4 | 29357 | 26624 | 27648 |
| 5 | 30795 | 28672 | 29184 |
| 6 | 32233 | 30720 | 30720 |

Acceptance: every warm cached count is a 512 multiple and matches the table, turns 3 to 5 drop
by roughly 0.5 s each against the base run (base 3.06, 2.66, 2.21 s), the median warm latency
does not regress, the server log shows `Using boundary cache snapshot` on every turn with no
`boundary_snapshot_unavailable`, and the six answers are identical to the base arm.
