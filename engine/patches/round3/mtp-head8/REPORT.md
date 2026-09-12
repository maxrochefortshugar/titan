# 8-bit MTP draft block

Workstream `kernels/round3/mtp-head8/`, 2026-09-12. No model loaded, no shard opened whole (headers
and single tensors by byte range only), peak under 1.5 GB. Ports 8083 and 8084 untouched.

## 1. Inventory of `mtp.*`

76 tensors, 1.483 GB, all in shards 1 and 21. `config.json`'s `quantization` block defaults to
4-bit group 64 and carries 15 per-tensor overrides under the `mtp.` prefix. Seven modules take the
default and are therefore 4-bit; the rest are 5, 6 or 8 bits, or bf16.

| module | weight | scales | bits | group | bytes |
|---|---|---|---|---|---|
| **fc_embedding** | U32 [2560,320] | [2560,40] | **4** | 64 | 3.69 M |
| **fc_hidden** | U32 [2560,320] | [2560,40] | **4** | 64 | 3.69 M |
| **hyper_connection_mixer.input_mix_weight_down** | U32 [320,1280] | [320,160] | **4** | 64 | 1.84 M |
| **hyper_connection_mixer.input_mix_weight_up** | U32 [10240,40] | [10240,5] | **4** | 64 | 1.84 M |
| **layers.0.mlp.switch_mlp.gate_proj** | U32 [512,640,320] | [512,640,40] | **4** | 64 | 471.9 M |
| **layers.0.mlp.switch_mlp.up_proj** | U32 [512,640,320] | [512,640,40] | **4** | 64 | 471.9 M |
| **layers.0.mlp.switch_mlp.down_proj** | U32 [512,2560,80] | [512,2560,10] | **4** | 64 | 471.9 M |
| layers.0.self_attn.q_proj | U32 [12288,480] | [12288,40] | 6 | 64 | 25.6 M |
| layers.0.self_attn.o_proj | U32 [2560,960] | [2560,96] | 5 | 64 | 10.8 M |
| layers.0.self_attn.k_proj, v_proj, indexer.index_qk_proj | U32 | | 6 | 64/128/64 | 3.4 M |
| layers.0.{attn,mlp}_hyper_connection.{down,up,block_inject} | U32 | | 5 | 64 | 9.1 M |
| layers.0.mlp.shared_expert.{gate,up,down}_proj | U32 | | 8 | 128 | 5.07 M |
| layers.0.mlp.shared_expert_gate | U32 [1,640] | [1,40] | 8 | 64 | 0.003 M |
| layers.0.mlp.gate, 6 RMSNorm weights | BF16 | | 16 | | 2.69 M |

The brief named `fc_embedding` and `fc_hidden`. Both are 4-bit, but they are 0.5% of the problem:
the 512-expert `switch_mlp` inside the draft block is also 4-bit and carries 96% of the requantised
bytes. The shared expert is already 8-bit, so it is left alone.

## 2. Getting the bf16 originals

`Qwen/Qwen3.8-Flash-Next` is public and ungated. Its index puts the seven tensors in six shards
totalling 14.44 GB (1.94, 1.84, 3.47, 3.51, 1.95, 1.73), over the 12 GB line, so nothing was
downloaded whole. One Range request read each header, a second pulled the tensor's byte span: 5.07
instead of 14.44 GB, in four minutes. Upstream keeps the experts fused as
`mtp.layers.0.mlp.experts.gate_up_proj` BF16 [512,1280,2560]; the split is contiguous, gate first,
confirmed against the deployed tensors (relative error 0.099 contiguous, 1.39 interleaved).

## 3. Quantisation and the quality argument

`mx.quantize(group_size=64, bits=8, mode="affine")`, streamed 32 experts at a time, into
`mtp-8bit.safetensors` (2.695 GB) with `manifest.json` (per-tensor bits, group size, shapes, sha256,
errors, and a file sha256). Errors are against bf16; the 4-bit column is the deployed checkpoint,
sampled on 32 experts for the expert tensors, exact for the rest.

| tensor | 8-bit max abs | 8-bit rel Frobenius | 4-bit max abs | 4-bit rel Frobenius | ratio |
|---|---|---|---|---|---|
| fc_embedding | 9.77e-4 | 0.0074 | 2.89e-2 | 0.1004 | 13.5x |
| fc_hidden | 3.51e-3 | 0.0152 | 4.03e-2 | 0.0732 | 4.8x |
| mixer.input_mix_weight_down | 1.07e-2 | 0.0090 | 3.91e-1 | 0.1228 | 13.7x |
| mixer.input_mix_weight_up | 5.25e-2 | 0.0109 | **8.27** | **0.4351** | 39.8x |
| switch_mlp.gate_proj | 5.19e-3 | 0.0075 | 1.05e-1 | 0.1003 | 13.4x |
| switch_mlp.up_proj | 5.07e-3 | 0.0074 | 1.43e-1 | 0.0995 | 13.4x |
| switch_mlp.down_proj | 4.39e-3 | 0.0073 | 3.63e-1 | 0.1062 | 14.5x |

`input_mix_weight_up` is the outlier. At 4-bit its relative error is 0.435 and its worst weight is
off by 8.27, so affine 4-bit group 64 is not representing it at all. It is 1.8 MB, so fixing that
one alone costs nothing.

## 4. The patch

`patch.py`, `install(model, sidecar=None) -> bool`, gated on `OMLX_MTP_HEAD8=1`, sidecar override
`OMLX_MTP_HEAD8_SIDECAR`. Runs **after model load**, since it touches instances. Import it by path
from `prod/bootstrap.py` alongside `install_verify_gate_up`, once `VLMBatchedEngine.start` holds the
model.

It finds the head through `model.mtp`, falling back to `language_model.get_mtp_module()`
(`language.py:3063`), then mutates the seven modules in place: new `weight`, `scales`, `biases`,
`bits = 8`. In place rather than by constructing replacements, because
`qwen35_moe_gate_up._fuse_one` (`patches/qwen35_moe_gate_up.py:95-115`) reuses the `gate_proj`
object as the fused `gate_up_proj` container and deletes `gate_proj` and `up_proj`, so a cached
reference must keep pointing at the object that is now 8-bit. When the fusion has already run,
`install` detects `gate_up_proj` and writes the concatenation of sidecar gate and up along axis 1,
matching `_fuse_one`'s layout. Order does not matter: `_can_fuse` only requires gate and up to agree
on bits and group size, so fusing after the swap also works.

Both quantized classes read `bits`, `group_size`, `mode` and the arrays at call time, as do the
patched `SwitchGLU.__call__` and the vlm verify patch, so nothing needs rebinding. `hc_fused.py`
accepts 8-bit at group 64 (`_SUPPORTED_BITS = (4,5,6,8)`, `_GROUP_SIZE = 64`) and templates its
Metal kernels on `.bits` per call, so the fused hyper-connection path stays engaged; group 128 would
have silently disabled it. `install` also drops `_compiled_forward` and
`_omlx_exact_hybrid_projection`, which capture arrays in a compiled graph, as insurance: both are
already skipped while the MTP runtime is enabled.

`test_swap.py` builds a real `Qwen4ExpMTPModule` at a shrunken geometry with random weights,
quantises it to 4-bit group 64, and swaps it. All 16 checks pass.

| check | result |
|---|---|
| [4] all seven swapped modules bit-identical to a direct 8-bit module | max abs 0.000e+00 |
| [3] module identity preserved | pass |
| [5] forward output shapes unchanged, [6] `fuse_inputs` unchanged | pass |
| [8..11] real `_fuse_one` applied first, fused 8-bit output equals concat of separate gate and up | max abs 0.000e+00 |
| [12] `SwitchGLU` forward runs post-swap | pass |
| [13..15] returns False with the env off, on a missing sidecar, with no mtp block, weights untouched | pass |

## 5. Cost and expected gain

Memory: the seven tensors go from 1.427 to 2.695 GB, **+1.268 GB**, the whole `mtp.*` block from
1.483 to 2.751 GB, against 70 GB resident on 128 GB.

Time: `bench.py`, real shapes, M=1, chain 10, median of 15, top-10 routing.

| part | 4-bit ms | 8-bit ms | 4-bit read KB | 8-bit read KB |
|---|---|---|---|---|
| fc_embedding | 0.0272 | 0.0278 | 3600 | 6800 |
| fc_hidden | 0.0248 | 0.0266 | 3600 | 6800 |
| mixer down / up | 0.0278 / 0.0253 | 0.0243 / 0.0243 | 1800 / 1800 | 3400 / 3400 |
| switch gate / up / down | 0.0202 / 0.0237 / 0.0213 | 0.0199 / 0.0219 / 0.0216 | 9000 each | 17000 each |
| **total** | **0.170** | **0.166** | 37800 | 71400 |

Bytes double, time does not move. Every projection sits at 20 to 28 microseconds, near the dispatch
floor, and the run-to-run spread of the total (0.170 to 0.181 ms) exceeds the 4-bit to 8-bit
difference. Treat the decode cost as zero and the whole case as acceptance.

Round 2 measured `C(3) = 31.06` ms at 1.91 tokens per cycle, 61.5 tok/s. Solving
`a + a^2 + a^3 = 0.91` puts conditional acceptance at about **0.51** per depth step today. Cycle
time unchanged, tok/s moves linearly with `1 + S_3`:

| conditional acceptance | tokens/cycle | tok/s at C(3)=31.06 | vs baseline |
|---|---|---|---|
| 0.51 (today) | 1.91 | 61.5 | |
| 0.55 | 2.02 | 65.0 | +5.7% |
| 0.60 | 2.18 | 70.1 | +14% |
| 0.63 | 2.28 | 73.5 | +19% |
| 0.65 | 2.35 | 75.6 | +23% |
| 0.70 | 2.53 | 81.6 | +33% |

One point of acceptance is worth about 1.2 tok/s. MTPLX's 2.3x to 3.0x is a 1.30x ratio, which here
needs acceptance at 0.63. Hold the change to that number, but it is a vendor claim on Qwen3.8-27B
under a different runtime, so it is the hypothesis rather than the forecast. What is measured here:
the draft weights carry 7 to 43% relative error, and removing it is free in time for 1.27 GB.

## 6. Workbench test plan

Isolated instance on 8084, guard 100 GB, quiet GPU, 45 s cooldown, patched and unpatched paired
within a round, two rounds.

| step | change | measure | confirm or refute |
|---|---|---|---|
| 0 | `OMLX_MTP_HEAD8=1`, no traffic | log `bits` of the seven modules and RSS | all 8, RSS up 1.2 to 1.3 GB. Anything else, stop |
| 1 | `OMLX_MTP_DEPTH_TRACE=1`, `..._EVERY=64`, patched vs not | `p` per depth from the `MTP depth:` lines | baseline `p` near [0.51, 0.26, 0.13]. Confirm: `p1` up 5 points or more. Refute: `p1` within 1 point, in which case 4-bit was not the limiter and the 1.27 GB buys nothing |
| 2 | same, agentic suite | decode tok/s, `accept=A/D`, `tok/cycle` from the `MTP[...]` line | `tok/cycle` and tok/s must move together by the same factor. If tok/s lags `tok/cycle`, the extra bytes are costing cycle time after all, which the microbench says they should not |
| 3 | as step 2 on four prompts: 200-token code completion, 2k prose, 25k tool-calling turn, 50k uncached prefill then 300 decoded | tok/s per prompt | code and tool-calling should gain most, prose least. A uniform gain across all four is suspicious and points at cycle time rather than acceptance |
| 4 | patched, `set_mtp_depth(4)` | `_DepthController` scores | at `a = 0.63` depth 4 clears its threshold. Only worth a sweep if step 2 confirms |
| 5 | greedy, fixed prompt, patched vs unpatched | output text | must **differ**. Identical greedy output means the swap never reached the live modules |
| 6 | `input_mix_weight_up` alone at 8-bit, everything else 4-bit | as step 2 | isolates the one tensor whose 4-bit error is 0.435, for 1.8 MB. If most of the gain is here, ship this and skip the 1.27 GB |

Step 6 is the cheap experiment and should run first.

## 7. Commands

    cd ~/inference-server/kernels/round3/mtp-head8
    ~/inference-server/kdev/bin/python test_swap.py
    ~/inference-server/kdev/bin/python bench.py
    ~/inference-server/kdev/bin/python quantize.py --chunk 32     # rebuilds the sidecar

## 8. Limitations

Acceptance has never been measured on the real head. Section 5 is arithmetic over round 2's cycle
timings, and 0.51 is inferred from tokens per cycle, not read from a depth trace. The 4-bit error
for the expert tensors is sampled on 32 of 512. The bench ran with the production daemon holding the
GPU. The swap has only been exercised on a synthetic head. The `orig/` slabs (5.07 GB) are
deletable once the sidecar is accepted.
