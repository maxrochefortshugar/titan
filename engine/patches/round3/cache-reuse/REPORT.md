# Cache reuse for a 36/12 hybrid: measurement, a shipped fix, and a negative result on CacheBlend

Workstream `round3/cache-reuse`. Read-only against production. All numbers come from the
oMLX logs on this box, the model config, and synthetic tensors under 200 MB with `kdev`.

## 1. What the traffic actually costs

Parsed from `/var/log/omlx/stderr.log`, `staging/omlx-home/logs/server.log*` and the
`staging/server-plain.log.*` rotations: 92 warm restores, 276 successful boundary stores,
603 store skips.

| quantity | median | p90 | max |
|---|---|---|---|
| cached prefix tokens on a warm restore | 24576 | 28672 | 49152 |
| recomputed tokens (`suffix=`) | 2000 | 3343 | 33863 |
| tokens discarded at store time (`storing X/Y`) | 1155 | 1734 | 6562 |
| reconstruct time for the cached part | 46.9 ms | 85.5 ms | 483 ms |

Cached prefixes are always an exact multiple of 2048, because
`config.paged_cache_block_size` is both the flooring granularity and the prefill chunk
width. The tokens discarded at store time in turn `k` are the tokens turn `k+1`
re-prefills, so the median warm turn recomputes about 1155 of its 2000 tokens as text the
server already processed one turn earlier: 58%. At 1438 tok/s that is 0.80 s of a 2.73 s
median warm turn (`staging/clean-seq.log`, baseline).

Worse is the skip path. 603 stores were dropped with `reason=boundary_snapshot_unavailable
available_boundaries=0`; 74 of those carried 2048 tokens or more, median 26482, 2.28 M
tokens in total. These are warm turns whose new suffix fit inside one sub-2048 tail chunk,
so prefill never crossed a block boundary, no GDN snapshot existed, and the whole turn was
discarded. Turn 2 of the five-turn replay does this: prompt 26481, cached 24576, stores
nothing, so turn 3 still restores at 24576 and recomputes 3343.

## 2. What is reusable on this architecture

**(a) CacheBlend for the 12 QSA layers: do not build it.** CacheBlend saves by skipping the
forward entirely for the ~85% of post-edit tokens whose attention does not deviate. On a
36/12 hybrid you cannot skip those forwards: the GDN state at position `t` is a function of
every token before `t`, so every post-edit token must be pushed through all 48 layers to
advance the recurrence, and once it is, its K and V come out for free. What is left to
reuse is the QSA k/v projection, and the audit's chunk budget bounds it: MoE 460 ms, GDN
366 ms, hyper-connections 133 ms, QSA 84 ms out of a ~1000 ms body. The whole QSA operator
is 8.4% of a chunk, the projections a fraction of that, and reusing stale KV is lossy. The
attention-over-growing-cache term (~400 ms/chunk at 65k) is not recoverable either, since
post-edit tokens still attend over the unchanged prefix. The 2.2-3.3x does not port.

**(b) GDN-only recompute of the post-edit region.** Not available. The layer order is three
`linear_attention` layers then one `full_attention`, twelve times over, and each GDN layer
consumes the hidden state from the MoE block below it. No schedule runs the 36 GDN layers
without the 48 MoE blocks and the hyper-connections, 59% of the chunk.

**(c) Session-level flooring: this is the whole win, and it needs no kernel.**
`preserve_mid_system_cache` is already `true` in `omlx-home/settings.json`, so a
mid-conversation system message renders as an inline user note (`server.py:1888-1896`)
instead of being hoisted and invalidating the prefix. Leave it on. The remaining lever is
the 2048-token flooring, below.

## 3. Prototype: fine boundaries in the tail only

`patch.py`, gated by `OMLX_CACHE_FINE_BOUNDARY=512` (`OMLX_CACHE_COARSE_CHUNK` defaults to
2048). Import-time install, before the engine starts. It rebinds
`omlx.scheduler.clamp_prefill_chunk_to_boundary` and wraps
`Scheduler._enlarge_block_size_for_arrays_cache` (`scheduler.py:2850-2901`, which pins the
block size to 2048).

Prompt bodies still run 2048-token forwards. Only the trailing remainder is cut once more,
at the largest 512 multiple it contains, emitting one extra snapshot there. The paged block
size drops to 512 so the prefix cache can keep that boundary. Blocks between snapshots carry
no GDN checkpoint, which restore already handles by walking back to the newest block that
has one (`cache/prefix_cache.py:3311-3330`).

Memory (`snapshot_cost.py`, config only, no weights read): the GDN recurrent state is
`[48,128,128]` fp32 = 3.00 MiB per layer over 36 layers, plus a `[3,10240]` bf16 conv
state, so one snapshot is **110.2 MiB**. The patch adds exactly one per prompt. Uniform
512-token blocks would need 128 snapshots over a 64k prompt, 13.8 GiB, against a 512 MB
pending-write budget. That is why the tail-only shape matters.

Simulated over the five-turn replay in `staging/clean-seq.log` (`test_clamp.py`):

| turn | prompt | stock cached | recomputed | fine cached | recomputed |
|---|---|---|---|---|---|
| 1 | 25043 | 0 | 25043 | 0 | 25043 |
| 2 | 26481 | 24576 | 1905 | 24576 | 1905 |
| 3 | 27919 | 24576 | 3343 | 26112 | 1807 |
| 4 | 29357 | 26624 | 2733 | 27648 | 1709 |
| 5 | 30795 | 28672 | 2123 | 29184 | 1611 |

Warm turns 2-5: 10104 recomputed tokens down to 7032, **30.4% fewer**. At 1438 tok/s that
is 0.47 s off a 2.73 s median warm turn, about 17% of measured TTFT, and much more on the
74 large requests that currently store nothing. Cost is one extra forward launch per
prompt and 4x the paged block count.

Correctness (`test_boundary_equivalence.py`, real `gated_delta_update` kernel at Flash-Next
shapes, one GDN layer with conv state plus one QSA layer with KV cache, logits head):

| comparison | max abs | mean abs | rel RMS |
|---|---|---|---|
| resume from the 512 snapshot vs live carry across the same split | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| split at 1536 (the fine boundary) vs one unsplit 2048 forward | 3.125e-02 | 8.83e-04 | 2.01e-03 |
| split at 1024 (control) vs one unsplit 2048 forward | 3.125e-02 | 2.28e-04 | 1.00e-03 |

The snapshot round trip is bit-exact: restoring at a 512 boundary reproduces the logits of
a run that never stopped there. The residual is bf16 re-association from cutting the scan,
one bf16 quantum at that magnitude (0.585% of the logit scale), and cutting at the fine
boundary measures the same as cutting anywhere else, so the patch adds no error class that
2048-token chunking does not already have.

Commands:

```
cd ~/inference-server/kernels/round3/cache-reuse
~/inference-server/kdev/bin/python test_clamp.py
PYTHONPATH=/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
  ~/inference-server/kdev/bin/python test_boundary_equivalence.py
python3 snapshot_cost.py
```

Verified against the real module with the bundled interpreter: `install()` returns True and
is idempotent, `clamp(4096, cache_tokens=0)` stays 2048, `clamp(1155, cache_tokens=24576)`
returns 1024 (the last 512 multiple), a 300-token tail passes through unsplit, and
`uninstall()` restores the stock 512 clamp.

Not yet measured end to end. `test_clamp.py` simulates the scheduler loop; the A/B on 8084
needs `GPU_FREE` and the same five-turn script that produced `clean-seq.log`.

## 4. If someone still wants CacheBlend

Functions to change: `Qwen4ExpAttention.__call__` (`LANG:1490-1540`) to accept a reusable
KV block and a deviation mask, `Scheduler._get_boundary_store_override`
(`scheduler.py:7230-7320`) to key snapshots by content chunk rather than absolute offset,
and `BlockAwarePrefixCache.store_cache` (`cache/prefix_cache.py:828-1300`) to admit
non-prefix blocks. Expected gain on the measured traffic: bounded by the 8.4% QSA share of
a chunk, so under 200 ms on a 2000-token post-edit region, against 470 ms already available
from section 3 at no accuracy cost. Risk: the approximation is lossy by construction. The
deviation threshold bounds it only in the sense that recomputing the top-`r` fraction of
tokens by attention deviation leaves the remaining tokens' KV stale by whatever the
threshold admits, and the paper reports no bound for recurrent layers because it never had
any. Recommendation: skip it, and spend the effort on the 74 dropped stores instead.
