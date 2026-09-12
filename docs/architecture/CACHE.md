# The prefix cache

2026-09-12. M2. This document covers `titan/adapters/cache/`: what a block is,
where a snapshot is allowed to sit, how the two tiers behave, and the exact
sequence the engine calls. It assumes ARCHITECTURE.md sections 4 and 5.

## 1. Why there are two grids

Qwen3.8-Flash-Next is a hybrid. Twelve of its 48 layers are sparse attention,
whose KV is positional: token 700's keys and values do not depend on how you
arrived at token 700, so a KV cache can be cut anywhere and reassembled from
pieces. The other 36 are Gated DeltaNet, whose state is a fold over every token
processed so far. The fold has no inverse. You cannot slice it, and you cannot
run it backwards.

So the cache stores two different things with two different resolutions.

* Attention KV is stored per **block** of 512 tokens. Blocks are cheap: restore
  is bytes bound at 2.77e-3 ms per token, so 48 blocks of 512 cost what 12 of
  2048 cost. Block count is free. Resolution is not.
* Recurrent state is stored as a **snapshot**, and a snapshot only exists where
  the engine staged one during prefill. Each is about 110 MiB and takes roughly
  210 ms of writer time, so they are rationed.

A prefix is restorable at position P only if P is a block end **and** a
snapshot sits at P. The overlay welded the two together, which forced the block
size up to 2048 to keep the snapshot rate sane, and a six-turn conversation
then recomputed 11617 tokens it had already processed. Separating them and
adding one snapshot at each prompt end brings that to 8545 and the warm median
from 2.59 s to 2.10 s. `tests/cache/test_probe.py` replays those six turns and
asserts both numbers.

## 2. Snapshot policy

Snapshots are staged at:

1. every multiple of the snapshot grid (2048) inside the prefill suffix, and
2. the end of the prompt, rounded down to the block grid.

Nothing else. In particular not at every chunk end: under decode contention the
scheduler shortens prefill chunks to 512 tokens, and emitting a snapshot at
each one quadruples the snapshot rate on the path that is already struggling.

Rounding the prompt end down is what makes rule 2 legal. A prompt of 25043
tokens gets its snapshot at 24576, because a restore point has to be a block
end: the KV either covers whole blocks or the chain hash of every block after
it changes. The 467 tokens past 24576 are recomputed next turn, which is a far
better trade than falling back to the previous 2048 multiple.

Because the backend can only stage a snapshot at the end of a chunk, every
point in that list has to end a chunk. `plan_chunks` guarantees it. Round 4a of
the overlay work did not: it clamped straight to its fine target, ran a chunk
from 26112 to 27648 that stepped over the grid multiple at 26624 without
landing on it, nothing was staged there, and the store found no snapshot for a
grid-aligned block and truncated the whole chain back to 26112. That one is
`test_every_coarse_multiple_inside_a_suffix_still_ends_a_chunk`.

The prompt-end cut has three gates, from `_fine_cut_allowed`. It is skipped
when it would gain less than 384 tokens over the coarse point it is competing
with, because the extra chunk launch plus the snapshot costs about 283 ms and
the gain has to beat that. It is skipped while the writer holds more than 192
MB of unwritten bytes, because that is the regime where a snapshot stops
costing 210 ms and starts costing the turn-2 stall from the report. And it is
skipped when it coincides with a grid multiple, since there is nothing to buy.

## 3. Naming: the chain hash

A block is named by

```
sha256(parent_hash or "titan-prefix-root" || signature || len(ids) || ids)
```

so a digest identifies the whole prefix ending at that block, not the 512
tokens it covers. Two prompts that share their first 24576 tokens produce the
same 48 digests and diverge at the 49th. Matching is a forward walk over that
chain: no trie, no ranges, one dict lookup per block.

The signature is `sha256` over the model name, the per-layer cache layout, the
block size and the snapshot dtype. It goes into every digest and into every
file header. Change the layer layout and the whole cache becomes unreachable
rather than becoming wrong, which is the only acceptable behaviour for bytes
that get loaded straight into a running model. Note what is **not** in it: the
tokenizer version. Ids are hashed, not text, so an adapter or template change
that alters tokenisation has to be threaded into the model name.

## 4. What a lookup returns

Two walks. Forward over full blocks while the chain hash is in the index and
the bytes still exist in some tier, which gives the KV frontier. Then backward
from that frontier to the newest block end that also has a snapshot, which
gives the answer. `PrefixMatch.matched_tokens` is the second number. The
distance between them is counted as `prefix.kv_only_blocks`, and a number that
climbs there means the snapshot policy is too coarse for the traffic.

A match never covers the whole prompt. Prefill needs at least one token to
produce logits from, so the deepest candidate block end is `len(tokens) - 1`
rounded down to the block grid.

The recompute tail is `len(tokens) - matched_tokens`, exactly, before a single
byte is read. The engine gets it from `recompute_tail(match, tokens)` and the
counters accumulate it as `prefix.recompute_tokens`.

## 5. Call sequence

```python
match  = cache.match(tokens)                     # alias of lookup
lease  = cache.reserve(match)                    # pin against eviction
n      = cache.restore(match, state)             # n <= match.matched_tokens
ends   = cache.plan_chunks(n, len(tokens), contended)
snaps  = cache.snapshot_boundaries(n, len(tokens), contended)

writer   = cache.begin_store()                   # incremental, one per sequence

position = n
for end in ends:
    backend.prefill(state, tokens[position:end],
                    want_logits=(end == len(tokens)),
                    snapshot=(end in snaps))
    position = end
    if end in snaps:
        writer.note_boundary(end)
    writer.pump(tokens, state, force_one=True)   # once per turn, budgeted

# ... decode, pumping each cycle ...

cache.commit(tokens, state, snaps)               # alias of store
cache.release(lease)
```

The engine drives the session and falls back to `commit` alone when the cache
does not offer one, which is what the engine's own fakes and the benches do.
`commit` is then the whole prompt in one call, which is correct and slow.

Four things about that sequence are load bearing.

`plan_chunks` takes `n`, the number `restore` returned, never
`match.matched_tokens`. A block can go missing between the lookup and the read,
and when it does `restore` shortens itself to the newest snapshot the surviving
blocks reach and reports the smaller number. Planning against the promise
rather than the delivery leaves a hole in the middle of the sequence.

`reserve` before `restore`. The pin is what stops the request that arrives one
millisecond later from evicting the blocks this one is halfway through reading.

`commit` is called with the same boundary list the prefill used. Passing the
raw prompt length also works, since store rounds every boundary down to the
block grid, but passing the plan keeps the two in step.

`release` is idempotent and must happen on every exit path, including aborts.
A lease that is never released pins its prefix for the life of the process.

## 6. Storing

Storing is incremental. The engine opens a **write session** when a sequence is
admitted, tells it about each boundary as prefill reaches it, and pumps it once
per turn; `store(tokens, state, boundaries)` is the same thing with the budget
removed, which is what a test or a bench that can afford to block gets.

The one-shot form is what the integration report caught. At the retirement of a
65k-token request it wrote 32 snapshots, 5.49 GB, in 15.4 s, in one call on the
loop thread, and the next request decoded at 25 tok/s instead of 55 for the
duration. Retirement is now O(the tail): everything prefill reached is already
bytes, and what is normally left is the prompt-end boundary the first decode
cycle staged.

Per boundary, `store` writes snapshots first and blocks second, and it writes
blocks only up to the deepest snapshot that committed. A block chain reaching past the last
snapshot cannot be resumed from, so recording it would let a later lookup
report a length it cannot restore. A boundary whose `export_snapshot` raises,
because nothing was staged there, is dropped and the chain truncates. That is
the third invariant, and it is counted as `prefix.chain_truncations`.

Everything is deduplicated by digest. Two requests that arrive on the same
prefix each get a lease, and the second one's store is a sequence of index
touches and `duplicate_puts`. Neither serialises a byte the other already did.

### The timeline, and which thread owns each part

For one boundary at length L, in order:

| Step | Thread | Cost |
|---|---|---|
| stage the recurrent snapshot | loop, inside the prefill | already paid by the forward |
| note the boundary | loop, end of the same turn | a list append |
| `export_snapshot(state, L)` | loop | one `mx.eval` of the staged slice, then the copy into `bytes` |
| `export_blocks` per new block | loop | one `mx.eval` of a 512-token slice per layer, then the copy |
| chain hash per new block | loop | sha256 over 512 ids, about 20 us |
| hot-tier insert | loop | a dict insert of bytes that already exist, no copy |
| record framing and crc32 | writer | one more copy of the payload, tens of ms on a 171 MB snapshot |
| temp file, write, `os.replace` | writer | the disk |

Only the first four rows are worth milliseconds, and all four are codec calls.
The chain hash stays on the loop thread deliberately: moving it would leave the
index that `lookup` reads behind the bytes the store already holds, and it is
four orders of magnitude cheaper than the copy it sits next to.

### The budget, and what happens when it runs out

`cache.max_stall_ms` (50 ms) is spent as a per-sequence per-cycle budget. A
session enters a boundary only when the measured cost of the last one still
fits what is left, because nothing can interrupt `export_snapshot` once it
starts: not entering is the only lever the loop has. The estimate is an
exponential mean over that session's own boundaries, so a 4k prompt and a 64k
one are each judged against themselves.

Two deliberate exceptions:

* the boundary a prefill chunk just staged is always serialised on that turn,
  whatever the estimate says. A snapshot cannot be cut in half, the turn that
  staged it is the cheapest moment it will ever have, and the alternative to
  paying one here is paying for all thirty-two at retirement.
* a **drain** always makes progress. A retirement that cannot finish inside the
  budget hands the state handle to a drain, which serialises one boundary per
  turn, yields to the loop between them, and closes the handle when it is
  done. The drain is bounded twice: by the budget per call, and by
  `store_drain_max_s` (2 s), past which the rest is abandoned, counted as
  `prefix.boundaries_abandoned`, and the chain truncates to the deepest
  snapshot that did commit. A draining state is still resident, so its
  estimated bytes stay in `resident_gb` and admission sees them.

### The loop-thread cost model, per snapshot

To size the caps, one boundary costs the loop thread

```
ms = C + S/B_snap + N * (c + K/B_kv)
```

with `S` the recurrent snapshot bytes (about 171 MB at 65k on the measured
run), `N` the blocks the boundary newly covers, `K` the KV bytes per block, and
`B` the copy bandwidth of `pack_arrays` over already-evaluated arrays. From the
report's retirement, 15.4 s over 32 snapshots and 5.49 GB, the old path came to
**about 480 ms per boundary at roughly 360 MB/s effective**.

That 360 MB/s was never the copy. Measured directly on this machine,
`pack_arrays` runs at **11 to 15 GB/s** on 16.8 MB of float32, so 171 MB of
payload is 12 to 15 ms of copying. The other 465 ms was the rest of the old
path: a second full copy in `pack_arrays` itself (the per-array `tobytes()`
that `join` then copied again, now one copy), the record framing and its crc32
(now on the writer thread), and the queue wait (now gone). What is left on the
loop thread per boundary is the `mx.eval` of the staged slice, which is a
device sync and does not appear in a numpy measurement, plus that one copy.

The fixed part is tiny. Measured on the numpy fakes, which have the same call
structure and no payload to speak of: **0.017 ms per boundary** covering four
new 8-token blocks, and **0.0036 ms per block** once the per-boundary overhead
is amortised (0.129 ms for 8 blocks, 0.296 for 64, 0.915 for 256). So `C` and
`c` are microseconds and every millisecond at production sizes is payload.

Scale it as `bytes / 360 MB/s`. A 50 ms budget buys about 18 MB, which is why a
171 MB snapshot is one boundary per turn rather than several, and why the
budget is spent as "always one, then as many more as fit" rather than as a hard
refusal.

What is not yet measured is the `mx.eval` term at production sizes, because it
needs the real checkpoint on the GPU. Until it is, size the budget by assuming
a boundary is one device sync plus `S / 10 GB/s`, and read
`prefix.max_store_stall_s` off `/metrics` on a real run to correct it. All of
it wants re-measuring whenever the snapshot dtype or the layer layout changes,
because both move `S` directly.

## 7. The two tiers

`TwoTierStateStore` implements `KVStateStore` and knows nothing about prefixes.

**Hot tier.** An ordered dict of byte strings under a byte budget, 4 GB by
default, evicted least recently used first, skipping pinned entries.
Accounting is in bytes rather than entries because a 512-token KV block and a
110 MiB snapshot differ by three orders of magnitude.

**SSD tier.** One file per record at
`<dir>/<model-slug>/<b|s>/<xx>/<key>.tkv`, a 256-way fan-out so no directory
holds a million entries, blocks and snapshots in separate namespaces. Writes
are write behind on one daemon thread: the caller hands over bytes and returns,
the thread writes a temporary file in the target directory and `os.replace`s it
into place, so a reader never sees a partial record. Stray temporaries from a
crash are swept at startup. The tier keeps a size-capped LRU index and deletes
oldest first past its capacity.

**Backpressure.** The pending queue is bounded in bytes and a put never waits
on it. Over budget the queue sheds: the oldest entry that no lease has pinned
is dropped to make room, and if that is not enough the incoming write drops
itself. Either way the payload stays in the hot tier and only its durability is
lost, counted as `store.writes_dropped`, `store.queue_evictions` and
`store.dropped_bytes`.

The wait is gone rather than shortened. The overlay waited up to two seconds
for the same budget, which is the report's 2.69 s turn where the model needed
1.7; Titan's first answer was a 50 ms cap on that wait, and the integration
report shows what a per-put cap is worth when a retirement makes thirty-two
puts back to back: `store.max_stall_s` read 1.50 s. A pinned record is never
the victim, because a pin means a live lease is matching against it, and that
is the one drop that can cost a warm turn.

The budget bounds real RAM. A queued write's payload is also in the hot tier,
so a backlog is counted twice until the disk catches up, and 5.49 GB of pending
snapshots is 5.49 GB that the model cannot have.

**Failure.** Nothing in the store raises into the engine. A missing file, a
truncated one, a bad crc, a record from another build, a full disk: each
returns `None`, counts itself, and lets the caller recompute.

## 8. The record format

```
magic     b"TITANKV"                       7 bytes
version   1                                1 byte
hdr_len   little-endian uint32             4 bytes
header    JSON, sorted keys, UTF-8
payload   opaque bytes from the codec
```

The header carries `kind`, `key`, `signature`, `tokens`, `payload_len` and
`payload_crc`. Loading checks all of them and treats any disagreement as a
miss. The checksum is crc32 rather than sha256 on purpose: loads happen on the
scheduler thread, and sha256 over a 110 MiB snapshot would cost about 200 ms
against crc32's 20.

Payloads are opaque. The store does not know a KV block from a recurrent
snapshot, and neither does the record format.

## 9. The codec seam

`StateCodec` is the only place the cache meets device memory: `export_blocks`,
`import_blocks`, `export_snapshot`, `import_snapshot`, plus the signature. The
mlx implementation wraps `ModelState`; the tests substitute numpy, which is why
the whole suite runs in a quarter of a second with no model and no GPU.

Two rules bind every implementation.

Serialise on the caller's thread. All four methods run on the scheduler thread,
and the cache hands the writer nothing but bytes. Serialising on a worker is a
Metal command buffer race, not a performance choice.

Evaluate on import. An array built from a buffer that came off disk stays
attached to that buffer until `mx.eval` runs, and a state holding file-backed
arrays across an eviction is a use after free. The store reads whole files into
`bytes` and never mmaps, which closes the other half of the same hole.

## 10. Counters

`stats()` returns a flat mapping of floats, `prefix.*` from the policy and
`store.*` from the tiers, so `/metrics` can dump it without knowing anything
about either.

The ones worth an alert: `prefix.hit_rate` and `prefix.recompute_tokens` for
whether the cache is doing its job; `prefix.kv_only_blocks` for whether the
snapshot policy is too coarse; `prefix.snapshot_seconds` divided by
`prefix.snapshots_written` against the 400 ms line from the report, past which
fine boundaries stop paying; `store.pending_bytes` and `store.queue_depth` for
the writer backlog the planner gates on, with `store.pending_peak_bytes` for
sizing the budget against it; `store.writes_dropped` for how often it lost that
bet; and `store.corrupt_skipped` with `store.signature_rejected`, which should
be zero outside a version change.

The store path has its own set. `prefix.max_store_stall_s` is the real number
the 50 ms cap is about: the longest a single pump held the loop thread, and the
one that read 1.50 s in the report. `prefix.store_seconds` over
`prefix.store_pumps` is the average, `prefix.snapshots_deferred` counts pumps
that stopped short on purpose, `prefix.drain_snapshots` counts boundaries
finished after their sequence retired, and `prefix.boundaries_abandoned` counts
the ones a drain gave up on, which should be zero.

## 11. Configuration

From `CacheConfig`: `block_tokens` 512, `snapshot_grid` 2048,
`snapshot_at_prompt_end` true, `fine_tail` true, `fine_min_gain_tokens` 384,
`ram_tier_mb` 4096, `ssd_dir`, `ssd_capacity_gb` 200, `max_stall_ms` 50,
`pending_write_budget_mb` 512.

`fine_min_gain_tokens` is the prompt-end gate from section 2, the 384 tokens a
fine cut has to buy back before it pays for its own chunk launch and snapshot.
It replaced `fine_tail_blocks`, which was validated at startup and read by
nothing: the threshold came from a constant in wiring instead, and deriving it
from a block count put it at 2048 and refused every fine cut on a prompt
shorter than the snapshot grid.

`max_stall_ms` is now a loop-thread budget rather than a queue wait. It bounds
what one sequence's store may spend per cycle, and the store checks its own put
path against it and counts `store.stall_cap_exceeded` if it ever breaches.
Startup validation refuses a grid that is not a multiple of the block size and
a prefill chunk that is not; the constructor refuses the same things again,
because a test that builds a cache directly deserves the same guardrail.

An empty `ssd_dir` makes the store RAM only. That is a legitimate production
choice on a machine whose disk is busy with the n-gram table, and it is what
the engine's own tests use.
