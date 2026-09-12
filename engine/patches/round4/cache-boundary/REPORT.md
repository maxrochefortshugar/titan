# Fine store boundaries, second pass: why 27648 failed and what the extra snapshot costs

Workstream `round4/cache-boundary`. Read-only against production, no model load, no server, no
GPU work beyond a few hundred KB of synthetic tensors. Env var `OMLX_CACHE_FINE_TAIL=512`, plus
three new knobs listed below. Import-time install.

## 1. The workbench run

| turn | prompt | base cached | base s | 4a cached | 4a s |
|---|---|---|---|---|---|
| 1 | 25043 | 0 | 16.35 | 0 | 17.62 |
| 2 | 26481 | 24576 | 1.72 | 24576 | 2.69 |
| 3 | 27919 | 24576 | 2.47 | 26112 | 1.66 |
| 4 | 29357 | 26624 | 2.17 | 26112 | 3.11 |
| 5 | 30795 | 28672 | 1.82 | 29184 | 1.64 |
| 6 | 32233 | 30720 | 1.28 | 30720 | 1.57 |

Warm turns recomputed 11617 tokens under stock and 10081 under the patch, so the store side
mostly worked. One store did not, and the median got worse.

## 2. Why 27648 had no committed checkpoint

The suffix crossed a coarse boundary. Turn 3 restored 26112 and had 1807 tokens to prefill. The
round-4a tail clamp jumped straight to the fine target, so it ran one chunk of 1536 tokens from
26112 to 27648 and then 271 more. That chunk stepped over 26624, which is 13 x 2048, without
ending there, so `should_emit_prefill_boundary` never fired at 26624 and no snapshot was staged.
At store time `store_cache` walks every new block and demands a committed sidecar
(`cache/prefix_cache.py:1285-1330`). The block ending at 26624 is coarse-aligned, so the
carve-out in the patch deliberately did not cover it, `commit_gdn_checkpoint` found nothing
staged (`scheduler.py:1753`, `cache/boundary_snapshot_store.py:482`), and the chain truncated at
26112. Block 150 in the log is that block, the first of the three the turn allocated.

The other four boundaries survived because no coarse multiple sat inside their suffixes: 26112,
29184, 30720 and 31744 are all reachable from their restore points without crossing one, or the
one they crossed was the chunk end itself. Suffix length, remainder size and block numbering
have nothing to do with it.

## 3. The corrected patch

Same file, same env flag. Four changes on top of round 4a.

1. The tail clamp stops at the next coarse boundary first when the suffix crosses one, and takes
   the fine cut on the following call. Every coarse-aligned block again ends a chunk, so it
   always has a snapshot.
2. Emission is now restricted to the coarse grid plus the fine cuts the patch itself chose.
   Without that, a block size of 512 emits a 110 MiB snapshot at every chunk end whenever the
   scheduler drops the prefill step to 512 tokens under decode contention
   (`scheduler.py:1542-1550, 5177-5197`), four times the stock snapshot rate on the path that is
   already contended.
3. Three gates decide whether the extra fine cut is worth it: `OMLX_CACHE_FINE_TAIL_MIN_GAIN`
   (default 384 tokens, the break-even from section 4), `OMLX_CACHE_FINE_TAIL_MIN_REMAINDER`
   (default 0, see below) and `OMLX_CACHE_FINE_TAIL_MAX_PENDING_MB` (default 192, skip the cut
   while the snapshot writer still holds that much unwritten). Each gate falls back to the stock
   cut, never to a cut the store cannot commit.
4. A rejected coarse boundary now logs the token count and the two grids, which the stock line
   (block id only) does not, and each snapshot logs its wall time and the writer backlog it saw.

A small remainder is not the expensive case, so `MIN_REMAINDER` ships off. Splitting a chunk of
T into T1 and T2 costs exactly one extra fixed chunk launch whatever the ratio, and the
attention work is identical, so a 271-token remainder is no worse than a 900-token one.

## 4. Cost model

`cost_model.py` predicts a turn as `K + r * cached + sum over chunks of (F + T * (a + b * ctx))
+ S per snapshot`. F = 73 ms and a = 0.4488 ms/token solve the audit's 992 ms per cold 2048
chunk together with its 22 percent penalty on 512-token chunks; b = 7.63e-6 is the audit's
992 to 1472 ms slope over 32k; r = 2.77e-3 ms/token matches all five measured reconstruct times
(reconstruct is bytes-bound, so 48 blocks of 512 cost what 12 of 2048 cost, which rules the
block count out). K = 53 ms is fitted on the base arm.

S is the one term the run pins directly. On turn 6 both arms restore 30720 and prefill 1513
tokens, and the only difference is one extra cut and one extra snapshot: 1.57 against 1.28 s, so
F + S is 290 ms and S is about 210 ms. Turn 2 has the same structural difference and cost 970
ms, because it lands while the writer is still draining turn 1's store of 48 blocks and 12
sidecars, and `save()` runs on the inference thread and can wait up to 2 s for the 512 MB
pending budget (`cache/boundary_snapshot_store.py:296-312`). That is the turn-2 slowdown: the
extra tail snapshot, made expensive by the preceding store, not the block count and not the
split. Turn 1 is structurally identical in both arms and still ran 7.8 percent slower, so treat
about that much of every fine-arm number as arm drift.

Model against the run, with drift applied to the fine arm:

| turn | stock model / measured | 4a model / measured |
|---|---|---|
| 2 | 1.41 / 1.72 | 1.83 / 2.69 |
| 3 | 2.62 / 2.47 | 1.78 / 1.66 |
| 4 | 2.28 / 2.17 | 3.42 / 3.11 |
| 5 | 1.91 / 1.82 | 1.70 / 1.64 |
| 6 | 1.25 / 1.28 | 1.65 / 1.57 |

Every turn lands within 0.31 s except turn 2, the backpressure turn.

## 5. Predicted next run

| turn | stock | fine 512 | fine 1024 |
|---|---|---|---|
| 2 | 1.41 | 1.69 | 1.70 |
| 3 | 2.62 | 1.94 | 2.27 |
| 4 | 2.28 | 1.90 | 1.62 |
| 5 | 1.91 | 1.57 | 1.91 |
| 6 | 1.25 | 1.53 | 1.53 |
| warm total | 9.46 | 8.64 | 9.02 |
| median | 1.91 | 1.69 | 1.70 |

Fine 512 wins while S stays under about 400 ms; at S = 600 ms both fine grids lose to stock, and
the new snapshot timing line will say which regime the machine is in. Cached counts should be 0,
24576, 26112, 27648, 29184, 30720, and warm turns should recompute 8545 tokens against 11617
under stock.

## 6. Test

```
cd ~/inference-server/kernels/round4/cache-boundary
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
  ~/inference-server/kdev/bin/python test_fine_tail.py
~/inference-server/kdev/bin/python cost_model.py
```

Section (e) replays all six probe turns through the real `PagedCacheManager`,
`PagedSSDCacheManager`, `BlockAwarePrefixCache`, `BoundarySnapshotSSDStore` and
`_BoundarySnapshotProvider` in three arms. The stock arm reproduces the measured base cached
sequence, the round-4a arm reproduces the measured fine sequence including the failed 27648
store, and the fixed arm stores every fine boundary. Sections (a) to (d) are unchanged: bit
identical GDN and KV restore, and no change below 2048 or on a 2048 multiple. Section (f) covers
the three gates and the emission guard. All 26 checks pass.

## 7. Next step

`store_exact_prefix` already knows how to persist an unaligned terminal block and refuses
split-GDN layouts only because nothing commits a terminal sidecar
(`cache/prefix_cache.py:1452-1460`). The last prefill chunk already ends at the prompt end, so a
snapshot there needs no extra chunk and no extra forward, and it would cache the whole prompt
instead of a 512-token floor. One snapshot per turn against a mean gain of 1024 tokens rather
than 768, with the F term gone. Worth a round 5.
