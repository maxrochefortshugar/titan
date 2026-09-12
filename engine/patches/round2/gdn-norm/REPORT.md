# Fused GDN grouped norm + output gate, T >= 1 (round 2, gdn-norm)

Audit item D1. The Gated DeltaNet output path runs a per-head RMSNorm and a gate
multiply as separate MLX ops with fp32 intermediates. oMLX already fuses them for
one-row decode; this delivers the same fusion for arbitrary T and hooks it so that
decode is untouched.

## What was changed

Stock code replaced, one method:

`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:1080-1088`,
`Qwen4ExpRMSNormGated.__call__`:

```python
y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
gate = gate.astype(mx.float32)
gate = mx.sigmoid(gate)          # output_gate_type == "sigmoid"
return (y * gate).astype(dtype)
```

Called once per GDN layer at `qwen3_5/language.py:1720` (`out = self.norm(out, z)`),
36 layers per forward. Shapes from the oQ4e config and the decode kernel's own
eligibility check: `x` and `gate` are `[B, S, 48, 128]` bf16 (48 value heads,
head_v_dim 128, the grouping axis), `norm.weight` is `[128]` bf16, eps 1e-6.
At T=2048 that is 75 MB of payload moved as ~352 MB by the op chain, 36 times.

`kernel.py` does it in one `mx.fast.metal_kernel` dispatch: one 32-lane simdgroup
per (token, head), four contiguous bf16 elements per lane, fp32 sum of squares via
`simd_sum`, `metal::precise::rsqrt(acc/128 + eps)`, then the gate in fp32 and a
single bf16 store. Eight heads per threadgroup (256 threads). No fp32 tensor exists
at any point. Both gate activations are implemented (`sigmoid`, which is what this
checkpoint uses, and `silu`/`swish`); the brief said SiLU, but
`text_config.output_gate_type == "sigmoid"` in the oQ4e config, and
`qwen35_gdn_prework.py:466` refuses anything else, so sigmoid is the live arm.

`patch.py` rebinds `Qwen4ExpRMSNormGated.__call__`. The fused arm engages only for
4-D input with `B * S > 1`; T=1 and any other shape or dtype falls through to the
original method. oMLX's fused decode arm (`qwen35_gdn_prework.py:669`) never calls
`self.norm` at all, so decode keeps `qwen4_decode_norm_gate_fused` whether or not
this patch is installed, and the `B*S > 1` gate is the second line of defence for
the case where the decode arm declines a token.

**Relationship to `~/inference-server/kernels/ple-fix/norm_patch.py`: no overlap,
they compose.** That patch rebinds `hc_fused.prefill_forward` and, under
`OMLX_QWEN4_BF16_NORM_ALL=1`, `Qwen4ExpRMSNorm.__call__`, the *ungated* grouped norm
on the hyper-connection and PLE streams. This patch rebinds
`Qwen4ExpRMSNormGated.__call__`, a different class on a different tensor. Verified
installed together, in both orders, both hooks live and the gated output still
bit-identical.

## Exactness

Bit-identical, not merely within 1 ULP. The kernel reproduces mlx's own
`rms_single_row` arithmetic exactly (same 32-lane/4-element decomposition, same
`simd_sum` order at axis 128, same `w * static_cast<T>(x * normalizer)` rounding,
so the bf16 intermediate that the stock Python then casts back to fp32 is
preserved), and the gate product is fp32 as in the stock code.

| case (x, gate `[1,T,48,128]` bf16) | elements | max abs err | max ULP | > 1 ULP |
|---|---|---|---|---|
| sigmoid, unit normal, T = 1 / 2 / 7 / 512 / 2048 | 6.1k … 12.6M | 0 | 0 | 0 |
| sigmoid, x scaled 1e-3 (eps regime), T=2048 | 12.6M | 0 | 0 | 0 |
| sigmoid, x scaled 64, T=2048 | 12.6M | 0 | 0 | 0 |
| sigmoid, saturating gate (z scaled 40), T=2048 | 12.6M | 0 | 0 | 0 |
| silu, unit normal and saturating gate, T=2048 | 25.2M | 0 | 0 | 0 |
| all-zero rows (normalizer = rsqrt(eps)) | 393k | 0 | 0 | 0 |

Same result under kdev python and the bundled interpreter.

## Microbench

Median of 15, warm, CHAIN=10, `mx.synchronize` around each timing.

| T | stock ms | fused ms | speedup | fused GB/s (payload) | x36 layers, ms saved |
|---|---|---|---|---|---|
| 1 | 0.037 | 0.026 | 1.43x | launch-bound | 0.40 |
| 2 | 0.036 | 0.026 | 1.37x | launch-bound | 0.35 |
| 512 | 0.252 | 0.041 | 6.14x | 460 (64% of 718) | 7.59 |
| 2048 | 0.993 | 0.133 | 7.49x | 569 (79% of 718) | 30.98 |

At T=2048 the kernel moves its 75 MB of payload at 569 GB/s, above the machine's
557 GB/s read+write copy figure, so there is nothing left to win here.

## Expected end-to-end gain

**Prefill: 31.0 ms per 2048-token chunk**, against a ~1000 ms body, so **+3.1%**.
That lands on the audit's estimate of ~30 ms and on its profiled 42 ms for this site
(the bench measures 35.7 ms for 36 stock calls in isolation; the profile figure
includes graph overhead the fused path also removes). **Decode: 0** by construction.

## How to enable

Env var `OMLX_QWEN4_GDN_NORM_GATE=1`. `install()` in
`~/inference-server/kernels/round2/gdn-norm/patch.py` is idempotent, returns False
with the flag unset or on any precondition failure, and self-checks the kernel
against the stock path on a small synthetic tensor before rebinding.

**Install after the model is loaded.** It monkeypatches a class, not instances, so
it does not need the model, but the target class lives in the vendored `mlx_vlm`
tree that oMLX's compat patch puts on the import path, and that is only reliably
importable post-load. Add to `prod/bootstrap.py` inside the existing
`VLMBatchedEngine.start` hook, beside the two calls already there:

```python
if os.environ.get("OMLX_QWEN4_GDN_NORM_GATE") == "1":
    try:
        ok = _load("gdn_norm_gate", "~/inference-server/kernels/round2/gdn-norm/patch.py").install()
        _log(f"fused GDN norm+gate: applied={ok}")
    except Exception as exc:
        _log(f"GDN norm+gate patch failed: {exc!r}")
```

and add `OMLX_QWEN4_GDN_NORM_GATE` to the `if` on line 8 of that file so the hook is
installed when only this flag is set.

## Commands

```sh
~/inference-server/kdev/bin/python ~/inference-server/kernels/round2/gdn-norm/test_exact.py
~/inference-server/kdev/bin/python ~/inference-server/kernels/round2/gdn-norm/bench.py
# integration check needs the bundled interpreter, which is where mlx_vlm lives.
# No model is loaded; peak use is under 200 MB.
cd /tmp && source ~/inference-server/kernels/ple-fix/_env.sh && \
  OMLX_QWEN4_GDN_NORM_GATE=1 $PY ~/inference-server/kernels/round2/gdn-norm/test_install.py
```

## Limitations

- Only 4-D `[B, S, HV, DV]` bf16/fp16 with `DV % 32 == 0` and `DV/32 <= 8`. Anything
  else takes the stock path, silently and correctly.
- Not measured in situ. The 31 ms is 36 x the isolated per-call delta; verifying it
  on a real chunk needs a model load, which the hard rules forbid here.
- The three ungated PLE grouped norms are a separate site, covered by
  `OMLX_QWEN4_BF16_NORM_ALL=1` (audit D4), not by this patch.
