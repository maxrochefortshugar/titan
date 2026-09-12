# Prefill fixes for Qwen3.8-Flash-Next: packed PLE table, bf16 grouped norm, int8 MoE gather

M5 Max 40-core / 128 GB, macOS 26.5, oMLX 0.7.0.dev2 (bundled CPython 3.11,
mlx 0.32.2, mlx_vlm 0.6.3), model
`~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`.

Everything here is measured with the model loaded exactly as oMLX loads it
(`profile-flashnext/load_omlx.py`: vendored `mlx_vlm.models.qwen4_exp`, the M5
`gather_qmm` reroute, `apply_qwen35_moe_gate_up_fusion`, the sdpa256 patch, PLE
in `mmap` mode), and follows the protocol from `profile-flashnext/REPORT.md`: a
12 s cooldown, a minimum of 3, and a **fresh baseline measured immediately
before every variant**. The live daemon on 8083 was left alone, nothing under
`/Applications` was modified, and the packed table was never loaded resident.

## Headline

| | 2048-token chunk (body) | tok/s | 32k prefill, warm cache | tok/s |
|---|---|---|---|---|
| stock oMLX | 1107.5 ms | 1849 | 24.34 s | 1346 |
| + packed PLE + bf16 norm + int8 MoE | **955.7 ms** | **2143** | **20.90 s** | **1568** |
| | **1.159x** | | **1.165x** | |

The two agree to within 0.6%, which is the useful result: the per-chunk win
survives a realistic 16-chunk prefill with a growing KV context.

| patch | per-chunk saving | what it fixes |
|---|---|---|
| bf16 grouped RMSNorm (`norm_patch.py`) | **82.0 ms** | the fp32 round trip in 97 hyper-connection norms |
| int8 MoE gather (`moe-int8/patch.py`) | **55.3 ms** | routed-expert matmuls at 25 TFLOP/s |
| packed PLE table (`patch.py`, rows mode) | **12.3 ms warm, 366 ms cold** | 3 pages per table row instead of 1 |
| sum of the three, measured separately | 149.6 ms | |
| measured together | **151.8 ms** | |

---

## 1. Per-chunk prefill, 2048 tokens

Warm page cache, body only (no `lm_head`), fresh KV cache per run, minimum of 3
after a 12 s cooldown, each variant against its own baseline measured moments
earlier. "PLE lookup" is the wall time inside `Qwen4ExpNGramEmbedding.__call__`
(n-gram hash evaluation plus the table gather), instrumented identically in
every configuration.

| configuration | body ms | tok/s | paired base ms | speedup | saving | PLE lookup ms (vs base) |
|---|---|---|---|---|---|---|
| `+ple` | 1076.3 | 1903 | 1088.6 | 1.011x | 12.3 ms | **6.4** vs 18.1 |
| `+norm` | 1007.4 | 2033 | 1089.5 | **1.081x** | 82.0 ms | 17.7 vs 18.2 |
| `+ple+norm` | 1003.3 | 2041 | 1090.7 | 1.087x | 87.4 ms | 6.4 vs 18.1 |
| `+int8` | 1048.2 | 1954 | 1103.4 | **1.053x** | 55.3 ms | 18.4 vs 18.5 |
| **all** | **955.7** | **2143** | 1107.5 | **1.159x** | 151.8 ms | 5.9 vs 18.4 |

* **The norm fix lands exactly where the profile predicted.** `profile.py ablate
  bf16_grouped_norm` measured the fp32 round trip at 85 ms of the 1085 ms body;
  removing it saves 82.0 ms. One line of code, 7.5% of the body, no new memory,
  and it is the cheapest item in the list.
* **int8 is worth 55 ms, not the 121 ms the profile hoped for.** That estimate
  assumed 1.42x on 413 ms of routed-expert matmul; 55.3 ms implies about 1.15x
  in situ. The kernel's own benchmark measures 1.48x on the isolated
  `gather_qmm` calls, so about two thirds of the isolated win does not survive in
  the full graph, most plausibly because the MoE block's sort, unsort and
  weighted sum overlap with the matmuls. It is comfortably above the noise: the
  three repetitions were 1048/1064/1088 ms against baselines of 1103/1115/1153.
* **The packed PLE table is only worth 12 ms on a warm chunk**, because a warm
  chunk is not where the 963 ms lived. Its value is in the cold case below and
  in the 32k loop, where a growing fraction of rows is unseen.
* Every baseline drifts upward inside its own triple (1107.5 -> 1123.6 ->
  1155.1 ms, for example). That is the +33% power drift from the profile, and it
  is why each variant is paired and reported as a minimum.

## 2. The PLE table gather

### Cold ids (the case the repack exists for)

Three independent fresh random 2048-token chunks per configuration, each
genuinely cold, minimum reported.

| | stock (3 tensors, `mmap`) | packed `rows` | ratio |
|---|---|---|---|
| body ms, cold ids | 1916.2 | **1549.0** | 1.24x |
| PLE lookup ms | 821.9 | **455.9** | **1.80x** |
| all three draws (lookup ms) | 1009 / 889 / 822 | 572 / 459 / 456 | |
| 16 KB pages `pread` per chunk | 93,046 (profile) | **31,855** | **2.92x** |
| SSD bytes read per chunk | 1,524 MB | **522 MB** | 2.92x |
| useful payload | 3.3 MB | 3.3 MB | |
| read amplification | 460x | **158x** | |

The page count is exactly the predicted 3x: one row is now one contiguous 100 B
read instead of a weight page, a scales page and a biases page in three
different tensors. (31,855 is `pages_read` 127,420 divided by the four distinct
2048-token chunks that went through rows mode in that process: 32,768 rows per
chunk, minus duplicate n-gram ids, plus rows that straddle a page boundary.)

The time ratio (1.80x) is smaller than the byte ratio (2.92x) because the
remaining 522 MB is still 32k scattered single-page reads, so it is now bound by
IOPS and per-`pread` overhead in the 48-thread `_PLE_IO_POOL` rather than by
bandwidth. Getting the rest requires the resident table, which does not fit
here (section 5).

### Warm ids

| | stock | packed `rows` |
|---|---|---|
| PLE lookup, 2048-token chunk | 18.1 ms | **6.4 ms** (2.83x) |
| PLE lookup over a 32k prefill (16 chunks) | 268 ms | **32 ms** (8.4x) |

## 3. 32k-token multi-chunk prefill, end to end

oMLX's chunk loop as `mlx_vlm_qwen4_exp_compat` patches it into
`PromptProcessingBatch.prompt`: 16 chunks of 2048, one KV cache, `prefetch_ple`
lookahead on, page cache warmed by an untimed pass per configuration. Three
timed repetitions per configuration, **30 s cooldown**, minimum reported.

| | base | all three patches | |
|---|---|---|---|
| total | 24.34 s | **20.90 s** | **1.165x** |
| tok/s | 1346.1 | **1567.6** | |
| three repetitions | 24.64 / 24.44 / 24.34 s | 21.02 / 20.90 / 20.96 s | |
| chunk 1 | 1289 ms | 992 ms | 1.30x |
| steady chunk (median of 2..16) | 1548 ms | **1273 ms** | 1.22x |
| last chunk | 1626 ms | 1472 ms | |
| PLE lookup, all 16 chunks | 268 ms | 32 ms | |

Per-chunk times rise with context in both configurations (1289 -> 1626 ms base,
992 -> 1472 ms patched): that is the QSA layers attending over a growing cache,
which none of these patches touch. The patched run is flatter, so the advantage
is largest early and narrows to about 1.10x by chunk 16.

**A 12 s cooldown is not enough for this stage.** The first attempt used 12 s and
produced base runs of 33.4 and 40.2 s (982 and 815 tok/s) against 36.9 and
32.6 s patched, a meaningless 1.02x. A 33 s sustained prefill does not recover
its clocks in 12 s. At 30 s the spread inside each configuration collapses to
1.2%. Both runs are kept in `combined_bench.json` (`long` and
`long_cool12_contaminated`). This is the same failure mode the profile flagged
for 4096-token chunks, and it is worth adding to the protocol: **cool for about
the duration of the run, not a fixed 12 s.**

---

## 4. Exactness

### 4.1 Packed PLE lookup: bit-exact

Not re-verified here. `verify_full.log` reports all 320,001,536 rows bit-exact
against the source shards, and `test_exact.json` reports the runtime lookup
bitwise identical to stock in both `rows` and `resident` mode, before and after
removal. The patch feeds the same nibbles, scales and biases into the same
`mx.dequantize(..., mode="affine")` and the same `* weight_scale`. Nothing in
this report changes that.

### 4.2 bf16 grouped RMSNorm: <= 1 ULP at the norm

`norm_exact.py` instruments a real 2048-token chunk of real text and, at **all
97 hyper-connection call sites**, compares three answers on the real
activations: the shipped norm, the kernel, and an fp32 reference (fp32
throughout, rounded to bf16 exactly once).

| comparison | elements | bitwise identical | within 1 ULP | max ULP | mean ULP |
|---|---|---|---|---|---|
| stock vs kernel | 2,034,237,440 | 99.9996% | **100.000%** | **1** | 4.41e-06 |
| fp32 reference vs stock | 2,034,237,440 | 99.9998% | 100.000% | 1 | 1.59e-06 |
| fp32 reference vs kernel | 2,034,237,440 | 99.9995% | 100.000% | 1 | 5.31e-06 |

Activation RMS across the call sites spanned 0.0043 to 0.638, so this is the
real dynamic range. **No element anywhere differs by more than one bf16 ULP**,
and about 4 elements per million differ at all. The kernel is 3.3x further from
the fp32 reference than the shipped path (5.3e-6 vs 1.6e-6 mean ULP), a real but
negligible difference: both are one ULP at worst, and the shipped path already
does its arithmetic in fp32, so it should win a rounding contest.

Through the whole `prefill_forward` (norm, both projections, the gated mean and
the injection) on a real call site:

| output | bitwise identical | max ULP | RMS error |
|---|---|---|---|
| `mixed` | 99.9935% | 2 | **0.0051%** of output RMS |
| `hyper_input` | 100% | 0 | 0 |
| `injection` | 100% | 0 | 0 |

The kernel was taken on every call (`kernel_calls` 97, `fallback_calls` 0).

### 4.3 The end-to-end divergence is the model, not the patch

The same script then runs the full 48-layer body plus `lm_head` on real text.
The result looks alarming at first:

| real text | body hidden RMS error | greedy argmax agreement | median top-1 margin where they disagreed |
|---|---|---|---|
| prose (2048 tok) | 12.22% | 93.46% (134 of 2048) | 0.125 (vs 1.875 where they agreed) |
| code (2048 tok) | 12.46% | 97.12% (59 of 2048) | 0.125 (vs 4.000 where they agreed) |

A one-ULP change in 4 elements per million should not move a hidden state by
12%, so `norm_chaos.py` runs two controls alongside the kernel on identical
inputs:

* **exact**: the norm done in fp32 throughout, strictly *more* accurate than the
  shipped path.
* **dither**: the shipped norm plus a random +-1 ULP on a random 4.4-per-million
  of its output elements. The same perturbation density as the kernel, carrying
  no information at all.

| vs stock, prose | body hidden RMS error | argmax agreement |
|---|---|---|
| kernel (the patch) | 12.22% | 93.46% |
| exact fp32 norm | 12.28% | 93.51% |
| random +-1 ULP dither | 12.46% | 93.26% |

| vs stock, code | body hidden RMS error | argmax agreement |
|---|---|---|
| kernel (the patch) | 12.46% | 97.12% |
| exact fp32 norm | 12.37% | 96.09% |
| random +-1 ULP dither | 12.72% | 96.53% |

All three land in the same place. Replacing the norm with a **more accurate** one
moves the output as much as the kernel does, and so does meaningless noise of the
same size. Measured against the fp32-exact norm as the reference, the kernel is
marginally *closer* than the shipped path (93.95% vs 93.51% argmax on prose,
97.27% vs 96.09% on code), though that gap is inside the noise.

Divergence by hyper-connection call site (relative RMS %, every 8th of the 97,
prose, kernel vs stock):

```
0  0.93  1.8  2.3  3.6  5.2  5.5  7.4  13  16  18  17  13
```

It compounds smoothly with depth: 36 gated-DeltaNet layers running a recurrent
scan over 2048 steps, on a 4-wide hyper-connected residual stream, amplify any
perturbation. A control repeat of the stock path was bitwise identical both
times, so the model is deterministic and this is genuinely the patch's 1 ULP
being amplified.

**Verdict: PASS.** The norm itself is correct to <= 1 bf16 ULP everywhere, the
whole fused block to 0.005% of output RMS, and the end-to-end movement is the
model's own sensitivity rather than a defect. Two consequences:

1. **Bitwise agreement with the current server cannot be the acceptance test**
   for any numerical change to this model. Fixing a rounding bug in the *right*
   direction would fail such a test.
2. The argmax disagreements sit at a median top-1 margin of 0.125 against 1.9 to
   4.0 where they agree, so they are near-ties. That is the expected signature,
   but it is an argument, not a measurement of quality. **Before production, run
   a real eval** (perplexity on a held-out set, or a short multiple-choice
   benchmark) with the norm patch on, the way `moe-int8/REPORT.md` section 10
   step 5 prescribes for the int8 kernel. I did not run one; it needs a dataset
   and a scoring harness that is not in this directory.

### 4.4 int8 MoE gather: not bit-exact, by design

Unchanged from `moe-int8/REPORT.md` section 5: 0.64% to 1.37% of output RMS
depending on activation distribution, against the 9.10% that 4-bit weights
already cost. Routed on prefill shapes only (`sorted_indices=True`, affine 4-bit
gs64, `tokens >= 512`); every decode and MTP-verify shape falls through
bit-identically. Routing was confirmed in this run: **96 routed, 0 fallback** per
2048-token chunk, the expected count with oMLX's gate/up fusion on (48 layers x
fused gate_up + down).

---

## 5. Memory

**`ps` RSS is useless for this process.** It reports 14 to 22 GB while
`mx.get_active_memory()` reports 72.8 GB, because the weights are file-backed
mmap pages that macOS does not charge to the task. Every figure below is MLX's
own accounting, which matches the 72.8 GB the profile reported.

| stage | live MLX arrays | peak during a 2048-token chunk |
|---|---|---|
| model loaded, PLE `mmap` | **72.82 GB** | 75.90 GB |
| + packed PLE, `rows` mode | 72.82 GB (**+0**) | 75.90 GB |
| + bf16 norm patch | 72.82 GB (**+0**) | 75.90 GB, minus an 84 MB fp32 temporary per call |
| + int8 MoE tables | **84.15 GB (+11.32)** | **87.23 GB** |
| + packed PLE, `resident` mode (not run) | 104.8 GB (+32.0) | - |

* **Packed PLE, `rows` mode: zero resident cost.** It mmaps `layer1.rows.bin`
  read-only and preads bounded ranges; the only allocation is a 1.95 MB
  `_seen_pages` bitmap (one byte per 16 KB page of a 32 GB file). It adds about
  522 MB of *page cache* per cold 2048-token chunk, reclaimable, and a third of
  what the stock path adds.
* **Packed PLE, `resident` mode: +32.0 GB, and it does not fit here.**
  72.8 + 32.0 = 104.8 GB, over the 85 GB ceiling this work was held to. It was
  never loaded in this session. `choose_mode`'s `auto` would pick it on a large
  enough machine, so **pin `OMLX_PLE_PACKED_MODE=rows` in any bootstrap on this
  box** rather than relying on `auto` plus a headroom guess.
* **bf16 norm: zero.** It allocates nothing and removes an fp32 intermediate.
* **int8 MoE: +11.32 GB measured** (84.146 - 72.822), exactly the 11.3 GB the
  moe-int8 report predicted, built in 1.9 s by `warmup(model)`.

### The int8 kernel is over the 85 GB line, and I measured it anyway

With the int8 tables the process holds **84.15 GB of live arrays and peaked at
87.8 GB** during a chunk. The brief's ceiling was ~85 GB, so on the live-array
figure it just fits and on the peak it does not.

The benchmark's headroom check passed for the wrong reason: it read `ps` RSS
(16.4 GB), projected 27.7 GB, and installed. The run was nonetheless safe. A
guard sampling `vm_stat` before every measurement never saw system
free+inactive below 28.7 GB, swap did not grow, and the watchdog did not fire.
I have since changed the check to use `max(RSS, mx.get_peak_memory())`, which
projects 87.2 GB and **skips** int8 by default; `--force-int8` reproduces what
was measured here.

So: the int8 numbers are real and were taken safely, but **int8 plus this model
plus a 2048-token chunk is a ~88 GB peak on a 128 GB machine, and it should not
be enabled in a server that also has to hold a KV cache for concurrent users.**
If it is wanted, `moe-int8/REPORT.md` suggests gating to `gate_up` only, about
+7.5 GB instead of +11.3 GB, which should keep most of the 55 ms (gate_up is
5.76 of the 8.24 ms of routed GEMM per layer). I did not measure that variant.

---

## 6. How these compose with oMLX's own patches

All three are additive with the oMLX stack and with each other. Nothing under
`/Applications` is modified at runtime; every patch is a monkeypatch with a
removal path.

| oMLX patch | interaction |
|---|---|
| `m5_gather_qmm` reroute | The int8 patch wraps `mx.gather_qmm` **after** the reroute and captures it as its own fallback, so the reroute still handles everything int8 declines (`K % 64 != 0`, the >32768-row segmentation, 5/6/8-bit tensors, every decode shape). Order matters: install int8 last. The segmentation does not trigger at a 2048-token chunk (top-10 gathers 20,480 rows) but does above 3277 tokens, so the two must coexist on long chunks. They did: 96 routed / 0 fallback with the reroute active. |
| `qwen35_moe_gate_up` fusion | The int8 patch hooks `SwitchGLU.__call__` only to record the token count, then delegates, so the fused `gate_up_proj` is preserved. With fusion on there are 2 routed matmuls per layer (96 per chunk), without it 3 (144). The 96/0 routing count confirms fusion was active throughout. |
| PLE SSD offload (`qwen4_ple_ssd_offload`, `mmap` mode) | The PLE patch **requires** it. It rebinds the live `DiskBackedShardedEmbedding` instances, so it is a no-op in `resident` PLE mode and logs and skips if the runtime is not `mmap`. It keeps the stock mmap readers open so a paired A/B can flip back. |
| PLE gather-ahead (`prefetch_ple` / `PromptProcessingBatch.prompt`) | Preserved. The patch installs a matching `prefetch` that queues `assemble_host` on the same `_prefetch_executor`, so the lookahead keeps hiding the gather, now with a third of the bytes to hide. Used in the 32k measurement above. |
| `hc_fused.fused_forward` (decode) | Untouched. `fused_forward` covers 1 to 16 rows and already uses `_kernel_norm`; the patch replaces only `prefill_forward`, the >16-row path. **Decode and MTP verify (M=2..8) are unaffected**, which is also why the int8 kernel's `min_tokens=512` floor and the norm patch cannot move decode throughput. |
| `sdpa256_attention` | No interaction. |
| `_force_qwen4_exp_sanitize_on_load` | No interaction; all three patches apply after load. |

`DiskBackedShardedEmbedding._host_indices`'s `mx.eval` (the one host sync inside
the body forward) is still there in `rows` mode. Only `resident` mode removes it.

---

## 7. Enabling this in a server bootstrap

`~/inference-server/staging/bootstrap.py` already applies patches in-process
before `omlx.cli serve` starts, and already handles the int8 kernel. The other
two need a **post-load** hook, because the PLE patch needs the live embedding
instances and the norm patch needs `mlx_vlm.models.qwen4_exp` to exist, which it
does not until `maybe_apply_pre_load_patches` has run.

Add this to `bootstrap.py`, before the `runpy.run_module("omlx.cli", ...)` line:

```python
# Qwen4-Exp prefill patches (workstream ple-fix), opt-in, applied post-load.
if os.environ.get("OMLX_PLE_PACKED") == "1" or os.environ.get("OMLX_QWEN4_BF16_NORM") == "1":
    sys.path.insert(0, os.path.expanduser("~/inference-server/kernels/ple-fix"))
    from omlx.engine.vlm import VLMBatchedEngine

    _vlm_start = VLMBatchedEngine.start

    async def _patched_start(self):
        await _vlm_start(self)
        model = getattr(self, "_vlm_model", None)
        if model is None:
            return
        try:
            import patch as _ple_packed       # ple-fix/patch.py
            n = _ple_packed.apply_ple_packed_patch(model, self._model_name)
            sys.stderr.write(f"[staging] PLE packed table: {n} layer(s) patched\n")
        except Exception as exc:
            sys.stderr.write(f"[staging] PLE packed patch failed (stock path kept): {exc!r}\n")
        try:
            import norm_patch
            ok = norm_patch.apply_bf16_norm_patch()
            sys.stderr.write(f"[staging] bf16 grouped norm: applied={ok}\n")
        except Exception as exc:
            sys.stderr.write(f"[staging] bf16 norm patch failed: {exc!r}\n")
        sys.stderr.flush()

    VLMBatchedEngine.start = _patched_start
```

Both calls are gated on their own env var internally and return 0 / False when
it is unset, so the block is safe to leave in place permanently. Note that
`ple-fix/patch.py` and `moe-int8/patch.py` are both named `patch`: import the
MoE one first (as the existing block does) or load one of them by path, or the
second `import patch` will silently return the first.

### Environment

| variable | value | effect |
|---|---|---|
| `OMLX_PLE_PACKED` | `1` | enable the packed PLE lookup (default off) |
| `OMLX_PLE_PACKED_MODE` | **`rows`** | pin the SSD layout. Do **not** leave this at `auto` on a 128 GB box: `auto` will choose `resident` and add 32 GB |
| `OMLX_PLE_PACKED_DIR` | (optional) | override the pack location; defaults to `<model>/ple-packed` |
| `OMLX_QWEN4_BF16_NORM` | `1` | enable the bf16 grouped norm in `prefill_forward` |
| `OMLX_QWEN4_BF16_NORM_ALL` | leave **unset** | would also route the three PLE norms through bf16. A strictly larger change, not measured here |
| `OMLX_MOE_INT8_PREFILL` | `1` (see the memory caveat) | enable the int8 routed-expert prefill kernel |
| `OMLX_MOE_INT8_MIN_TOKENS` | `512` (default) | chunk token floor |
| `OMLX_QWEN4_PLE_MODE` | leave unset (`mmap` for this model) | the PLE patch is a no-op in `resident` mode |

Recommended first production step: **`OMLX_QWEN4_BF16_NORM=1` and
`OMLX_PLE_PACKED=1` with `OMLX_PLE_PACKED_MODE=rows`.** Together that is 87 ms
of the 152 ms, zero extra memory, and one of the two is bit-exact. Hold the int8
kernel back until the memory question is settled and an eval has run.

### Verifying it took, on a running server

1. The bootstrap prints `PLE packed table: 1 layer(s) patched` and
   `bf16 grouped norm: applied=True`.
2. `norm_patch.STATS` shows `kernel_calls` climbing by 97 per prefill chunk and
   `fallback_calls` at 0. A nonzero `fallback_calls` means the kernel declined a
   shape and the canonical norm ran: correct, but slow.
3. `patch.packed_tables()[1].lookups` climbs by 1 per chunk, and
   `.last_lookup_s` is about 6 ms warm rather than 18 ms.
4. `moe_int8_patch.stats()` reads `{'routed': 96, 'fallback': 0}` per chunk. A
   zero there means the shape gate rejected everything, the failure mode that
   made an earlier trial measure the unpatched server twice.
5. Decode tok/s must be unchanged. All three patches gate themselves off below
   17 rows (norm), 512 tokens (int8), or affect only the prefill gather (PLE).

---

## 8. Can the 64 GB pack be made smaller? Yes, by half, today

`<model>/ple-packed/` currently holds **64.0 GB in two complete copies of the
same data**:

| file | size | used by |
|---|---|---|
| `layer1.rows.bin` | **32.00 GB** | `rows` mode only |
| `layer1.weight.u32` | 25.60 GB | `resident` mode only |
| `layer1.scales.bf16` | 3.20 GB | `resident` mode only |
| `layer1.biases.bf16` | 3.20 GB | `resident` mode only |

`rows.bin` is the row-interleaved layout (80 B of packed nibbles + 10 B of
scales + 10 B of biases = 100 B per row, 320,001,536 rows). The three `.u32` /
`.bf16` files are the plane-separated layout: the identical bytes, regrouped.
They are alternatives, not complements. `patch.py` reads exactly one set,
depending on `OMLX_PLE_PACKED_MODE`.

**The patch as deployed uses `layer1.rows.bin`.** `resident` mode needs 32 GB of
resident arrays on top of a 72.8 GB model, which does not fit under the 85 GB
ceiling, so the planar trio is dead weight on this machine and **the three
planar files can be deleted, taking the pack from 64.0 GB to 32.0 GB.**

Two caveats before deleting:

* **Resident mode is the bigger prize.** `fused_gather_est.py` measured the same
  32,768-row gather from one packed device buffer at **0.21 ms**, against 6.4 ms
  warm and 456 ms cold for `rows`. That removes the PLE gather from the budget
  entirely, and the mid-body host sync with it. It needs about 105 GB of process
  footprint, so it is a "when the machine or the model changes" option, not a
  today option. Keep the planar files if a 192 GB box or a smaller model is on
  the roadmap.
* **Deleting is cheap to undo.** `repack.py` rebuilds either layout from the
  checkpoint in about 66 s (measured: 65.5 s for both), so there is no reason to
  keep 32 GB against that.

The 34 GB of original PLE shards inside the checkpoint must stay: they are part
of the safetensors index the model loads from, and they are the fallback the
patch reverts to.

---

## 9. Limitations and honest failures

* **No eval.** Section 4.3 argues from ULPs, controls and top-1 margins that the
  norm patch is safe, and 4.4 quotes the int8 kernel's RMS errors. Nobody has run
  perplexity or a benchmark with either patch on. That is the missing piece
  before production, and it is a dataset problem, not a kernel problem.
* **int8's in-situ win is 55 ms against an isolated 1.48x that predicted ~121 ms.**
  I did not chase the discrepancy. An ablation that stubs the MoE block with and
  without the kernel would settle whether the gather matmuls are genuinely off
  the critical path.
* **The 12 s cooldown was too short for the 32k stage** and produced a
  meaningless first result (section 3). Both runs are in the JSON. Any future
  long-prefill A/B needs a cooldown comparable to the run length.
* **Single batch row, no MTP, no concurrent load.** A server with a real KV cache
  and several sessions has less headroom than this benchmark did, which matters
  most for the int8 tables.
* **The PLE lookup timer forces `mx.eval` on the embedding output**, adding a
  small barrier production does not have. It is applied identically to every
  configuration, so comparisons are fair, but the absolute "PLE lookup ms" is a
  slight overstatement.
* **`rows` mode keeps the mid-body host sync** and still issues ~32k single-page
  preads. Only a resident table removes both.
* **`auto` mode is dangerous on this machine.** `choose_mode` compares against
  `max_recommended_working_set_size`, which on a 128 GB M5 Max is large enough
  that it will pick `resident` and add 32 GB. Pin `rows`.

## 10. Files

| file | what it is |
|---|---|
| `patch.py` | the PLE packed-table lookup patch (pre-existing; `rows` and `resident` modes) |
| `norm_patch.py` | the bf16 grouped-RMSNorm patch for `prefill_forward` (pre-existing) |
| `repack.py`, `repack.log`, `verify_full.log` | the conversion and its bit-exact verification (pre-existing) |
| `test_exact.py`, `test_exact.json` | bit-exactness of the packed lookup (pre-existing) |
| **`norm_exact.py`, `norm_exact.json`, `norm_exact.log`** | sections 4.2 and 4.3: per-call-site ULPs on real activations, whole-block error, argmax through the model |
| **`norm_chaos.py`, `norm_chaos.json`, `norm_chaos.log`** | section 4.3's controls: exact-fp32 norm and random-dither runs |
| **`combined_bench.py`, `combined_bench.json`, `combined_bench.log`, `combined_bench_long.log`** | sections 1, 2, 3 and 5 |
| `combined_bench_chunk.json` | the chunk and cold-id stages before the 32k re-run was merged into `combined_bench.json` |

```
. ./_env.sh
$PY norm_exact.py                                  # numerics
$PY norm_chaos.py                                  # the controls
$PY combined_bench.py                              # everything (int8 now skipped by budget)
BENCH_LONG_REPS=3 BENCH_COOL=30 $PY combined_bench.py --stages long
$PY combined_bench.py --force-int8                 # reproduce the int8 rows, ~88 GB peak
```
