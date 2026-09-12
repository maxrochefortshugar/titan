# titan/kernels

The kernel library. Ten ops, each with a reference implementation in plain MLX
ops and an accelerated implementation (a `mx.fast.metal_kernel`, or a
specialised MLX call sequence), selected by the registry.

Nothing here monkeypatches anything, and nothing imports oMLX, mlx_vlm or
mlx_lm. Every op is a pure function: inputs in, outputs out. State that has to
persist between calls (weight tables, tile tables, reader pools, memory maps)
lives in objects the caller constructs and owns, and the only cache the library
keeps for itself is the compiled-kernel handle, which is a property of the
process rather than of the model.

## The ops

| op | what it does | exactness class | test status | source report |
|---|---|---|---|---|
| `gdn_norm_gate` | fused grouped RMSNorm + sigmoid/silu gate on the Gated DeltaNet value heads, T >= 1 | bit-identical | pass | `engine/patches/round2/gdn-norm/REPORT.md` |
| `moe_weighted_sum` | top-k unsort and routed weighted sum in one launch; sorted and unsorted layouts; clone, ops and fast modes | bit-identical in `clone` | pass | `engine/patches/round2/wsum10/REPORT.md`, `round3/small-items` item 3 |
| `hc_prefill` | fused hyper-connection block: kernel norm, one epilogue, one tail | 99.998% bit-identical downstream of the norm, rrmse 5.5e-6; norm itself <= 1 ULP | pass | `engine/patches/round3/hc-fuse/REPORT.md` |
| `gdn_chunk_scan` | PR #4020 C = 8 chunked delta-rule scan; NAX C = 16 behind a flag | state rrmse ~6e-7 vs the fp32 per-token recurrence (bar 1e-5). NAX variant 5.4e-4, **not exact** | pass | `engine/patches/round3/gdn-scan/REPORT.md` |
| `moe_gather_ws` | weight-stationary bf16 sorted-gather MoE matmul with a ragged tile table | bit-identical | pass | `engine/patches/round3/gather-ws/REPORT.md` |
| `moe_gather_int8` | int8 x int4 sorted gather on the tensor units, with its table builder | **not exact**: ~0.65% of output RMS from activation quantisation | pass | `engine/patches/moe-int8/REPORT.md` |
| `grouped_rmsnorm_bf16` | grouped RMSNorm over a wide residual stream, bf16 in and out | <= 1 bf16 ULP | pass | `engine/patches/ple-fix/REPORT.md` section 4.2 |
| `topk_radix` | three-launch radix top-K over one very wide logits row | exact top-K set; ties broken arbitrarily as `mx.topk` does | pass | `engine/patches/round3/small-items/REPORT.md` item 1 |
| `ple_packed_lookup` | packed n-gram table reader: manifest, rows layout, pooled host page reads, device dequantise | bit-identical | pass | `engine/patches/ple-fix/REPORT.md` section 4.1, `round3/small-items` item 2 |
| `qsa_gathered_attention` | sparse gathered QSA, single and batched, with the phase-correct padded index logic | loop arm bit identical per row; padded arm <= 1 bf16 ULP | pass | `engine/patches/round3/qsa-batched/REPORT.md` section 2 |

Each op also registers under the name the engine looks it up by:
`ngram_gather`, `rms_norm_grouped`, `gdn_scan_chunked`, `moe_gather_gate_up`,
`hyper_connection_block`, `qsa_sparse_decode`, `mtp_shortlist`.

Tests live in `tests/kernels/test_<op>.py`, one file per op, plus
`tests/kernels/test_registry.py` for selection and forcing. Run them with
`pytest tests/kernels`; `-m "not slow"` skips the real-shape cases, which are
the only ones that allocate more than a few hundred MB.

## Two ops that are not bit-identical

`moe_gather_int8` quantises the activations to uint8, which costs about 0.65%
of output RMS. That is the whole point of it: uint8 x uint4 peaks at 113 TOP/s
against bf16's 65.7 TFLOP/s, and the packed nibbles go to the tensor unit
without a dequantise round trip. Because it changes the numbers it is opt-in
twice over: the registry will not select it unless the caller has built and
passed the `Int8Tables`, and the exactness test asserts a relative-RMS bound
rather than a ULP bound.

`gdn_chunk_scan`'s NAX variant misses the state bar by 50x and is kept only for
measurement. `supports()` never selects it, and a test asserts that it misses
the bar, so it cannot become the default by accident. The cause is
`matmul2d_descriptor`'s `relaxed_precision`, which the source PR and MLX's own
steel NAX GEMM leave true; setting it false yields garbage, so the fragment
copies are tied to the relaxed layout.

## What changed on the way over from the overlay

The overlay kernels were monkeypatches against oMLX and mlx_vlm internals. Four
things had to change to make them a library.

**Weight tables became explicit.** The int8 gather memoised its `qsum` table in
a module dict keyed on `id(weight_tensor)`. That table is roughly the size of
the scales array, 3.7 GB across the 48 layers of Flash-Next, and it is not
something to hide from whoever is accounting for memory. It is now
`Int8Tables`, built by `build_tables` and held by the caller. The
weight-stationary gather's tile table went the same way, and the caller reuses
one table across both projections of a SwitchGLU, which is what the memo was
for in the first place.

**The hyper-connection block stopped reading a module.** It took an mlx_vlm
module object and pulled `input_mix_weight_down`, `hc_count` and the rest off
it. It now takes `HCWeights`, a value object of `QuantLinear` projections. The
merged down|inject bank is a `ConcatBank` the caller builds rather than an
attribute stashed on the module.

**The n-gram reader lost its resident mode.** It faulted three 32 GB planes
into device arrays when the machine looked like it had room. On 128 GB that
panicked the kernel. Only the rows mode is ported, and a test asserts the
resident mode is absent.

**The env vars are gone.** Twenty or so `OMLX_*` flags became the registry's
`KernelConfig` and per-call keyword arguments (`mode`, `variant`, `fp32_mean`,
`use_bank`), so the running configuration is in one validated file rather than
a launchd plist.

## Exactness numbers that moved

`hc_prefill` is the one op whose headline number is different from the overlay
report, and it is worth saying why rather than quietly restating it.

The report measured the fused block **bit-identical**, 0 ULP, chain included.
Its reference was oMLX's own fused tail, which is itself a JIT Metal kernel.
Titan's reference is a plain chain of MLX ops, and against that reference 0 ULP
is not available: a literal transcription of MLX's own `Sigmoid` expression
into `mx.fast.metal_kernel` still lands 1 ULP away on about 0.08% of inputs, so
`mx.sigmoid` is not bit-reproducible from a JIT kernel on this build. What is
reproducible, and what the tests assert, is 99.998% of elements bit-identical
at the real shape with rrmse 5.5e-6, the residue concentrated where the stream
mean cancels. The parts that can be exact are checked separately: the tail
reproduces MLX's bf16 stream mean (a sequential bf16 sum and one divide)
exactly.

The two options that move results further are asserted to move them, so neither
can become the default without a test failing: `fp32_mean`, which is more
accurate but up to 11 ULP from MLX's bf16 reduce where the four stream terms
cancel, and the merged input projection, whose `hc` injection rows come out 2
to 4 ULP away at the real K = 10240.

## Selection

`build_registry(config)` registers every op and applies the enable/disable
policy; `reference_only()` returns a registry with every fast path off, which
is the parity baseline and the control arm of every kernel A/B.
`registry.resolve(name)` returns a dispatcher, and the dispatcher decides
between fast and reference **once per op per shape class**, memoised. A shape
class is operand shapes, operand dtypes, the device, and whatever non-array
discriminator changes which implementation is legal.

A name in `kernels.disabled` that is not a registered op raises `ConfigError` at
startup, so a stale bisect flag stops the process rather than quietly doing
nothing. A fast path that raises falls back to the reference and increments a
counter, which is safe precisely because the exactness tests make the output
independent of which one ran.

## Licence

Our own code, MIT, same as the rest of Titan. Two provenance notes:

- `gdn_chunk_scan`'s Metal body is ported from ml-explore/mlx pull/4020, commit
  c7e1a2a (`gated_delta_update.h`, `gated_delta_update_nax.h`). MLX is MIT.
- `grouped_rmsnorm_bf16`, `gdn_norm_gate` and `hc_prefill` reproduce MLX's own
  rounding behaviour deliberately, and `gdn_norm_gate`'s reduction follows the
  decomposition in MLX's `rms_single_row`. That is a numerical requirement, not
  a copy: the exactness bar is stated against MLX's results, so the arithmetic
  has to match it.
