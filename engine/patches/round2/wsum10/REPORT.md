# wsum10: fused MoE unsort + weighted sum for top_k=10 (Flash-Next)

## What was replaced

Flash-Next (`qwen4_exp`) builds its MoE block from `Qwen3_5MoeSparseMoeBlock`
(vendored `.../qwen4_exp/language.py:32, 2784, 2841`), so the routed-expert
combine is the stock chain at `mlx_vlm/models/qwen3_5_moe/language.py:64-65`:

```
y = _target_verify_switch_glu(self.switch_mlp, x, inds, target_verify)
y = (y * scores[..., None]).sum(axis=-2)
```

with the scatter one level down at `mlx_lm/models/switch_layers.py:195-196`
(`_scatter_unsort`, from `SwitchGLU.__call__:175-199`, or oMLX's gate_up-fused
override at `omlx/patches/qwen35_moe_gate_up.py:138-140`). Three full passes over
a [T, k, D] bf16 tensor, 105 MB at T=2048, k=10, D=2560. oMLX has a native kernel
for this spot but whitelists `top_k in (6, 8)`
(`omlx/patches/qwen35_moe_weighted_sum.py:58`), so top-10 never reaches it.

`kernel.py` replaces the tail with one `mx.fast.metal_kernel` launch reading the
*sorted* expert output, applying `inv_order` as a gather while it accumulates,
writing [B, T, D]: one 105 MB read plus a 10 MB write instead of ~520 MB. bf16
in, bf16 out, fp32 arithmetic. The router chain (`:56-62`) is reproduced
verbatim, so expert sets and scores are unchanged.

Layouts confirmed from the code: expert output arrives **sorted** as `[T*k, 1, D]`
whenever `inds.size >= 64` (always at prefill, and from T=7 up), paired with
`inv_order` from `_gather_sort`; `scores` is `[B, T, k]` bf16, already softmaxed
(`precise=True`) and already renormalised. The unsorted small-T layout
`[B, T, k, D]` goes through the same kernel with an identity permutation.

## Exactness

`test_exact.py` compares the kernel against the stock ops path and an fp32
golden at T = 1, 8, 512, 2048 (5.24 M elements at T=2048), plus end to end
through a real 4-bit `SwitchGLU`.

Finding worth recording: **MLX's stock bf16 axis reduce accumulates in bf16**,
eight partial accumulators combined sequentially. I recovered that model exactly
(0 mismatches over 256 probes); it is why the ops path sits up to 2 bf16 ULP from
a correctly rounded sum. Two defensible targets, both shipped:

| mode | arithmetic | vs stock ops path | vs fp32 golden |
|---|---|---|---|
| `clone` (default) | bf16 products, 8 bf16 partials, bf16 combine | **0 of 5,242,880 elements differ, 0 ULP** | 0.064 max abs / rms |
| `fast` | fp32 products, 8 fp32 partials | 0.064 max abs / rms (this is the ops path's own error) | **0 of 5,242,880 elements differ, 0 ULP** |
| `ops` | bf16 products, fp32 partials | intermediate | intermediate |

The default is `clone`: bit-identical to production output today, so deploying it
carries no behavioural risk. `fast` is correctly rounded at the same speed, the
better choice if a small output change is acceptable.

End to end (E=32, D=2560, inter=640, 4-bit, top_k=10) `clone` is bit-identical at
T=1, 13 and 512. Raw max-ULP is not a headline number here because the operator
cancels to near zero on some channels, where answers 1e-7 apart are thousands of
ULPs apart; the test prints max-abs-over-rms alongside.

## Speed

M5 Max 40-core, mlx 0.32.2, k=10, D=2560, bf16, warm, median of 15 (8 at
T>=512), min of 3 rounds, each variant in a fresh process.

| T | stock ops (ms) | kernel (ms) | speedup | kernel GB/s |
|---|---|---|---|---|
| 1 | 0.0243 | 0.0231 | 1.05 |  2 |
| 8 | 0.0273 | 0.0210 | 1.30 | 21 |
| 13 | 0.0283 | 0.0209 | 1.35 | 35 |
| 512 | 0.2676 | 0.0520 | 5.15 | 554 |
| 2048 | 1.1098 | 0.2268 | 4.89 | 509 |

509 GB/s is 71% of this machine's 718 GB/s read ceiling, and faster than oMLX's
native k=8 kernel in the audit (0.360 ms at T=2048). The kernel wins at every T
measured, so nothing is routed to stock on speed grounds: `min_tokens` defaults
to 1 and decode goes through it too.

**Per 2048-token chunk, 48 layers:** 0.883 ms saved per layer = **42 ms**, about
+4% on the 1,085 ms GPU body; against the audit's own 1.977 ms ops figure, 84 ms
(+8%). The audit predicted +4.5% for a native k=10 kernel, so this lands on it.

**Per decode step:** 0.06 ms at T=1 and 0.36 ms at T=13 over 48 layers against a
~16 ms step. Under 2%; prefill is the reason to ship this.

Two measurement notes. The ops path is very sensitive to memory pressure: the
same T=2048 number ranged 1.10 to 7.61 ms across the session with free pages,
since it holds two 105 MB transients per call. The table was taken with ~7 GB
free. And partial accumulators matter: a single fp32 accumulator serialises the
k-long FMA chain and costs 1.45x.

## Integration

`patch.py` exposes `install() -> bool`, idempotent, returning False and leaving
the stock path on any failure. It monkeypatches class `__call__` (no instance
state) after a compile-and-verify self check at two shapes, so a kernel that
fails to build never reaches the model. Targets `Qwen3_5MoeSparseMoeBlock` in
`mlx_vlm.models.qwen3_5_moe.language` plus the three mlx-lm equivalents.

**Call it after the model is loaded**, from the post-load hook
`prod/bootstrap.py` already uses. It touches no instances, so order against
oMLX's own patches does not matter: `top_k` 6 and 8 are not claimed, so oMLX's
native kernel keeps them whichever wrapper is outermost.

Env vars (all optional except the master switch):

| var | default | meaning |
|---|---|---|
| `OMLX_WSUM_TOPK10` | `1` | `0` disables, install() returns False |
| `OMLX_WSUM_TOPK10_MODE` | `clone` | `clone` / `fast` / `ops` |
| `OMLX_WSUM_TOPK10_MIN_TOKENS` | `1` | raise to 64 for prefill only |
| `OMLX_WSUM_TOPK10_K` | `10` | which `top_k` values to claim |
| `OMLX_WSUM_TOPK10_VEC` | `4` | outputs per thread |
| `OMLX_WSUM_TOPK10_CONTIG` | `1` | adjacent columns per thread |

## Commands

```
cd ~/inference-server/kernels/round2/wsum10
~/inference-server/kdev/bin/python test_exact.py
~/inference-server/kdev/bin/python bench.py
~/inference-server/kdev/bin/python bench.py --ts 1,2,4,8,13,32,64,128,512,2048 --modes clone
```

Saved output: `test-20260912.txt`, `bench-20260912.txt`.

## Limitations and honest gaps

- Never run against the real model. mlx_vlm is absent from the kdev venv, so the
  end-to-end test uses a stand-in class carrying the verbatim body of
  `Qwen3_5MoeSparseMoeBlock.__call__` over a real quantized `SwitchGLU`.
- Target-verify (MTP) falls through to stock, matching oMLX. That path uses
  unsorted `[B*T, k]` indices and needs its own gate; worth a follow-up, since
  verify runs every decode step.
- 509 GB/s leaves ~30% of the read ceiling. The gather is a random permutation
  over 105 MB, so it is TLB and miss bound; ordering the output rows, or having
  `down_proj` write unsorted, would recover some.
- `clone` bit-identity depends on MLX keeping its eight-partial-accumulator bf16
  reduce. The install-time self check verifies it at the real hidden size and
  refuses to patch if it changes, so a future MLX degrades to stock rather than
  to wrong numbers.
