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

position = n
for end in ends:
    backend.prefill(state, tokens[position:end],
                    want_logits=(end == len(tokens)),
                    snapshot=(end in snaps))
    position = end

# ... decode ...

cache.commit(tokens, state, snaps)               # alias of store
cache.release(lease)
```

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

`store` writes snapshots first and blocks second, and it writes blocks only up
to the deepest snapshot that committed. A block chain reaching past the last
snapshot cannot be resumed from, so recording it would let a later lookup
report a length it cannot restore. A boundary whose `export_snapshot` raises,
because nothing was staged there, is dropped and the chain truncates. That is
the third invariant, and it is counted as `prefix.chain_truncations`.

Everything is deduplicated by digest. Two requests that arrive on the same
prefix each get a lease, and the second one's store is a sequence of index
touches and `duplicate_puts`. Neither serialises a byte the other already did.

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

**Backpressure.** The pending queue is bounded in bytes. A put that cannot fit
waits, but never for longer than `cache.max_stall_ms` (50 ms), and then drops
the write and counts it. The record stays in RAM; only its durability is lost.
The overlay's version waited up to two seconds for the same budget, and that is
directly visible in the report as a 2.69 s turn where the model needed 1.7.

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
fine boundaries stop paying; `store.pending_bytes` for the writer backlog the
planner gates on; `store.writes_dropped` for how often it lost that bet; and
`store.corrupt_skipped` with `store.signature_rejected`, which should be zero
outside a version change.

## 11. Configuration

From `CacheConfig`: `block_tokens` 512, `snapshot_grid` 2048,
`snapshot_at_prompt_end` true, `ram_tier_gb` 4.0, `ssd_dir`,
`ssd_capacity_gb` 200, `max_stall_ms` 50, `pending_write_budget_mb` 512.
Startup validation refuses a grid that is not a multiple of the block size and
a prefill chunk that is not; the constructor refuses the same things again,
because a test that builds a cache directly deserves the same guardrail.

An empty `ssd_dir` makes the store RAM only. That is a legitimate production
choice on a machine whose disk is busy with the n-gram table, and it is what
the engine's own tests use.
