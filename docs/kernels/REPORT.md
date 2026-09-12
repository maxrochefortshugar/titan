# Kernel workstreams: combined report (2026-09-11, M5 Max 40c, 128 GB, macOS 26.5, mlx 0.32.2)

Three Opus agents each delivered a JIT Metal kernel in oMLX patch style (mx.fast.metal_kernel, env-gated, stock fallback), an exactness test, a benchmark and a REPORT.md in their directory. All three exactness tests were re-run by the orchestrator and pass. Nothing was installed into the running daemon.

## Scoreboard

| Workstream | Kernel-level result | Whole-model effect (est.) | Correctness | Status |
|---|---|---|---|---|
| 1. Small-M matmul (smallm/) | lm_head at M=8: 4-bit 3.16 -> 1.31 ms (2.4x), 8-bit 3.94 -> 1.25 ms (3.2x); q_proj M=8/M=1 = 1.24x (stock ~2x) | Batched decode and MTP verify: removes the 7x lm_head penalty at 2..8 rows | argmax 100% vs stock at M=2..16; routed-band error at bf16 ULP; fallback bit-identical | Ready for in-situ trial |
| 2. MoE decode fusion (decode-fusion/) | Routed-MLP layer T=1: 88 -> 71 us (1.23x), launches 14 -> 4; T=2 1.23x; beats stock out to T=64 | Only +3% to +6% single-stream decode | within 1 bf16 ULP of fp32 golden, closer than stock; T>=8 falls back bit-exact; router fusion opt-in (1/40 top-k tie flips) | Correct, low payoff |
| 3. Native int4 prefill (prefill-int4/) | q_proj/o_proj at M=2048: 1.19x / 1.12x via int8 activations x int4 weights on the tensor units; measured NAX peaks | 2-3% now (dense projections only); 12-15% once the MoE gather is converted; ~30% with macOS 27 cooperative tensors | adds 0.2-1.2% relative error on top of the 9.1% the 4-bit weights already carry; argmax 98% vs stock (stock itself 96.7% vs fp32) | Prototype; MoE gather not done |

## What we learned (this changes the earlier estimates)
1. M5 Max NAX ceilings, measured for the first time: bf16 65.7 TFLOPS, int8 x int8 129.8, int8 x int4 120, bf16 x int4 65.5, fp32 15.7. A 4-bit weight operand buys no tensor-core time; the wider operand sets the rate. Stock quantized_matmul at M=2048 already runs at 87% of the bf16 ceiling, so "skip the bf16 expansion" was worth at most 15%, not 2x. Real prefill upside needs int8 activations (done, 1.1-1.2x) and macOS 27 cooperative tensors (est. +1.5x on the K loop).
2. The "20-35 us per small kernel" premise was an artefact of isolated timing. Inside a layer's command buffer a small kernel costs ~4-5 us, so launch fusion buys ~17 us per layer, not ~200. Decode is genuinely bandwidth-bound: 1.36 GB of routed-expert weights per token is 2.5 ms at the 549 GB/s ceiling, and the fused kernels already hit 92% of it.
3. The earlier "~100 TFLOPS sorted gather" figure over-counted FLOPs (it exceeds the measured 65.7 bf16 ceiling); the sorted path is still ~45x faster than the unsorted fallback, which is why the 0.7.0.dev2 upgrade (mlx 0.32.2) matters.
4. Small-M 4-bit above M~4 is scalar-ALU bound (BN x M fp32 FMAs per weight), not bandwidth bound; 8-bit stays on the memory side and goes flat in M. The remaining lever is simdgroup_matrix.
5. JIT kernels on macOS 26.5 CAN use Metal tensor ops with int8 and int4 operands (`__HAVE_INT4B_FORMAT_TYPE__` = 1), with no Xcode and no MLX rebuild. Constraints: 4-bit operands only from memory tensors, K % 32 == 0, no mixed signedness.

## Revised whole-model estimate for Flash-Next on this machine
- Single-stream decode: +5% from fusion; the rest of the gap to the 133 tok/s roofline is weight bandwidth, not kernels. Further gains need fewer bytes per token (lower-bit experts, MTP acceptance) rather than faster kernels.
- 2-8 concurrent sessions / MTP verify: 1.5-2.4x on the vocabulary head, modest elsewhere; whole-model gain depends on the share of lm_head time (largest at short contexts).
- Prefill: 1.1-1.2x now on dense projections; 1.15x whole-model with the MoE gather converted; ~1.3x with macOS 27.

## Recommended next steps (in order)
1. In-situ trial of workstream 1 on a staging server: second `omlx serve` on port 8084 with a sitecustomize that imports smallm/kernel.py and calls apply() (OMLX_SMALLM_QMM=1), run bench.py against it with 4 concurrent sessions and MTP on; adopt if tok/s improves with identical outputs.
2. Convert the sorted MoE gather to the int8 x int4 tensor path (workstream 3's missing piece; needs a 40-rows/expert tiling). This is the largest remaining prefill win on macOS 26.
3. Enable decode fusion only if the staging trial shows the 3-6%; it is correct but small.
4. On macOS 27 (after MLX/oMLX declare support): cooperative tensors direct to matmul for the prefill kernel; re-measure.
5. Upstream candidates to ml-explore/mlx: the small-M kernel (issue #4265 territory) and the int8-activation prefill path.

Per-workstream details, tables and failure notes: smallm/REPORT.md, decode-fusion/REPORT.md, prefill-int4/REPORT.md. Shared conventions: COMMON.md. Baseline microbench: ../kbench.py. Analysis notes: ../kernel-notes.md.

## Staging trial of workstream 1 (2026-09-11 evening, port 8084, Qwen3.6-35B-A3B-4bit, isolated oMLX 0.7.0.dev2 instance)
Method: same server binary, plain vs patched (kernel applied in-process via bootstrap.py), 1/4/8 concurrent greedy streams, 2-3 rounds each, GPU shared with the live M4 session (noise about +/-5%: plain 4-stream measured 229 to 246 across repeats).
Two false starts are recorded for honesty: (1) oMLX ships its own sitecustomize.py that shadowed the loader, and a stale server held the port, so the first "patched" run measured the plain server; (2) the patch only routed 2-D inputs, but oMLX batched decode presents [B, 1, K], so nothing routed. Fixed by flattening leading dims (argmax-identical on a 4-row lm_head check).
Result with routing confirmed (1,580 kernel calls, all lm_head at M=2..4, K=2048):
| streams | plain tok/s (aggregate) | patched tok/s | delta |
|---|---|---|---|
| 1 | 131.9 | 127.6 | -3% |
| 4 | 246.5 (229 repeat) | 234.4 / 213.7 | -5% to -13% (within/below noise band) |
| 8 | 183.3 | 178.7 | -3% |
Greedy outputs identical 6/6. Verdict: no gain, slight loss. oMLX never presents M>4 at the lm_head even with 8 streams (its scheduler splits decode batches), and at M=2..4 on 4-bit the kernel's edge over stock is small while the Python-side metal_kernel dispatch and reshape add per-call overhead inside an already overlapped graph. Do not deploy. The kernel's 2.4-3.2x microbenchmark win at M=8 is real but unreachable through oMLX's batching as configured; it would need either oMLX to batch the head at M>=8 or the kernel to be wired at the C++ dispatch level (upstream MLX), where the small-M case is issue #4265 territory.

## Workbench experiments on Flash-Next (2026-09-11 late evening, no other sessions, isolated instance on 8084)
Cold 50k prefill, tok/s (quiet GPU unless noted): baseline 1,387 and 1,016 (repeat) | int8 gather kernel 1,371 and 1,060 (routed 4,214 calls, confirmed active) | prefill chunk 4,096: 1,254 | MTP off: 994 (and decode 41 vs 58-61) | cache block 1,024: 1,045 | block 512: 1,028 | n-gram table page-cached: 1,134 | SSD prefix cache disabled: 1,220 / 1,135 | table in RAM: swapped, invalid.
Conclusions: chunk size, cache block size, SSD cache writes (~62 MB/s during prefill, harmless) and table reads (page-cached, 6 MB disk reads per 30 s) are all non-factors. MTP must stay on. The int8 gather kernel is active but moves whole-model prefill ~1%, so expert matmuls are not the Flash-Next bottleneck; the 3x gap to the qwen3_5_moe proxy (4,500 tok/s) is inside Flash-Next's own layers (ple n-gram modules, QSA indexer, hyper-connections). Direct profile in progress (profile-flashnext/).
Run-to-run spread is 30% even on a quiet GPU when memory is near full (page cache vs wired weights); keep >10 GB free for stable numbers.
Correction: the earlier "sorted gather ~100 TFLOPS" microbenchmark figure over-counted FLOPs 10x; stock sorted gather is 23 TFLOPS (38% of ceiling).

## Direct Flash-Next profile (profile-flashnext/, 2026-09-11 night): the prefill bottleneck found
- GPU part of a 2048-token chunk = 1,085 ms (1,888 tok/s): MoE 42.5% (gate+up 22.8 TFLOPS, down 24.6; 35-37% of ceiling), GDN 33.7% (in_proj 76% of ceiling; recurrent scan 123 ms at 5%), hyper-connections 19.7% (215 ms, 85 ms of it an fp32 round-trip), QSA 7.8%, PLE GPU part 3.4%.
- THE bottleneck: the PLE n-gram table lookup on the CPU. Per chunk it needs 3.3 MB of rows but preads 1,524 MB over 93,046 pages (963 ms, GPU idle) because each row's weight/scales/biases sit in three tensors = three 16 KB pages per row (460x read amplification). Reproduced oMLX loop: 1,465 tok/s with lookahead, 1,087 without = the band measured all day. Resident-table mode falls into a Python loop because oMLX gates the fused path on 192 GB RAM; a fused device gather from a resident 34 GB table is 0.21 ms.
- GPU power-manages under sustained prefill: +33% drift across 12 chunks, recovers after 45 s idle. Measurement protocol: 12 s cooldown, min of 3, paired baseline. Earlier unpaired A/Bs (incl. the int8 gather "1%") are unreliable; int8 gather should be ~8% of a chunk.
- 4096-token chunks are 20% worse per token (QSA indexer engages above 2048; worse power state); 512 is 22% worse. 2048 stays.
- Ranked fixes: PLE row repack (+25-30%), int8 gather (+11% of body), bf16 grouped RMSNorm in prefill_forward (+8%, one-line), fuse hyper-connection block (+6-8%), chunked delta scan (+7-11%), fuse MoE permutation (+5%). Combined estimate 1,465 -> ~2,500 tok/s. PLE repack + norm fix + combined bench in progress (kernels/ple-fix/).

## Decision (2026-09-11 late): fast model dropped; Flash-Next gets the memory
Resident packed PLE table (32 GB, bit-exact, 42 s load) + 73 GB model = ~105 GB, fits under the 118 GB wired limit without a second model. Killed-run numbers: streamed packed table cuts the per-chunk lookup from 93,046 pages/963 ms to 32,727 pages/416 ms; resident mode is the target. Haiku slot moves to a no-think Flash-Next profile alias. Details: kernels/ple-fix/FINDINGS-killed-run.md.

## 2026-09-12 00:23 kernel panic: resident PLE table ruled out on 128 GB
Panic report: "watchdog timeout: no checkins from watchdogd in 91 seconds" while the workbench loaded the resident packed PLE table (32 GB) on top of the 73 GB model. Third memory incident on that path (two UI freezes earlier). Verdict: the n-gram table stays on SSD on this machine; deploy the streamed packed layout (rows mode, bit-exact, zero resident cost, 93k -> 32k page reads per chunk) plus the bf16 norm fix. Resident mode only on the 256 GB Ultra. Keep wired memory under ~100 GB with headroom for the OS.

## 2026-09-12 00:50: safe A/B result and production deployment

Workbench (port 8084, guard 100 GB), fresh random 65k-token prompt per rep, 45 s cooldown, two rounds each, patches confirmed in the server log ("PLE packed table: 1 layer(s) patched (mode=rows)", "bf16 grouped norm: applied=True"):

| run | cold prefill tok/s | decode tok/s |
|---|---|---|
| baseline | 1347 (1357, 1338) | 47.9 |
| baseline-repeat | 1267 (1189, 1345) | 52.0 |
| rows + norm | 1453 (1454, 1453) | 61.7 |
| rows + norm-repeat | 1453 (1449, 1458) | 61.4 |

Cold prefill +8% to +15%, decode +18% to +28%, and the patched runs are much less noisy (the n-gram lookup runs per decode token too, so cutting 3 page reads to 1 per row helps decode as much as prefill). Resident mode is banned on this machine; the streamed rows layout gives this result at zero resident cost.

Deployment: `prod/run-omlx.sh` (wrapper with the patch toggles) + `prod/bootstrap.py` (hooks VLMBatchedEngine.start, applies the two patches post-load, stock path kept on any failure). `prod/deploy-optimisations.sh` rewrites the LaunchDaemon plist to call the wrapper, lowers the memory guard 110 -> 100 GB, reloads the daemon, re-pins Flash-Next and verifies the patch lines. Haiku/subagent slot: profile alias `Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think` (created 2026-09-12; the Qwen3.6 fast model is dropped). Not deployed: int8 MoE gather (+~5%/chunk for +11.3 GB resident; revisit if memory allows), small-M qmm (no gain in-graph).

## 2026-09-12 01:20: production verified

Daemon launched via prod/run-omlx.sh; log shows "PLE packed table: 1 layer(s) patched (mode=rows)" and "bf16 grouped norm: applied=True". Production numbers: cold 65k prefill 1438 tok/s median (1466, 1409), decode 61.5 tok/s. Agentic suite (no-think, 32k): short decode 86.6 tok/s, 50k uncached prefill 1408 tok/s, 49k cached follow-up 2.2 s, coding 5/5, tool calls ok at short and 25k context with no leaked markup. Pre-patch reference (2026-09-11, thinking on): 47.7 tok/s, 1035 tok/s, 4.1 s, 4/5. The no-think alias run hit the prefix cache left by the main-model run (50k prompt answered in 2.0 s), confirming the alias shares engine and cache. Results: prod/bench-prod-20260912.json, prod/bench-prod-alias-20260912.json.

## 2026-09-12 02:00: optimisation audit (two Opus agents)

Reports: kernels/AUDIT-2026-09-12.md (code, profiles, disassembly) and kernels/RESEARCH-2026-09-12.md (upstream and community). Key corrections: read-only bandwidth is 718 GB/s (549 was copy), so the decode roofline is 171 tok/s and production's 61.5 is 25-35% of it; the MTP verify pass eats most of the speculative gain (MTPLX measures 61 plain on M5 Max); prefill 1438 tok/s over 65k is ahead of every published number, and the gap to the 2048 tok/s chunk-1 body is attention over the growing cache, never profiled. Native int4 tensor ops: no int4 datatype exists in mlx 0.32.2's Metal side; the expert gather already runs on NAX with bf16 operands after per-tile dequant, BM=32 vs 40 rows/expert streams every expert twice. Next in order: profile attention vs cache depth, top_k=10 native weighted-sum (oMLX upstream), fused gated norm for GDN prefill, MTP verify/drafter work (mlx-vlm 0.7.0 items), LRU n-gram row cache, mlx PR #4481 when merged.

## 2026-09-12 02:50: round 2 (four Opus agents, workbench A/B, deployed)

Deliverables in kernels/round2/: gdn-norm (fused GDN grouped norm + sigmoid gate for T>1, bit-identical, 7.5x at T=2048), wsum10 (fused MoE weighted sum for top_k=10, bit-identical in default mode, 4.9x at T=2048), ple-lru (set-associative LRU row cache in front of the packed n-gram reader, bit-exact), mtp (shortlist drafter: MTP draft steps 2..3 on a top-K head, verified output identical; gate_up leak refuted; 0.7.0 diff: nothing portable). Workbench A/B, 65k cold prompt, two reps each:

| run | prefill tok/s | decode tok/s |
|---|---|---|
| base | 1441 | 59.8 |
| base repeat (55 min later) | 1373 | 58.6 |
| gdn-norm | 1473 | 61.1 |
| wsum10 (all T) | 1492 | 55.6 |
| wsum10 prefill-only (min tokens 64) | 1422 | 58.6 |
| ple-lru 2 GB | 1455 | 60.8 |
| mtp shortlist | 1453 | 61.0 |
| mtp depth 4 + shortlist | 1399 | 56.3 |
| int8 gather gate_up-only | 1497 | 60.2 |
| combined (norm+wsum+lru+shortlist) | 1532 | 57.3 |
| final (norm + wsum prefill-only + shortlist + int8 gate_up) | 1581 | 58.4 |

Prompt-length sweep (base config, clean points): 8k 1641, 16k 1677, 32k 1519, 65k 1441, 130k 1321 tok/s; combined at 130k 1354. Attention over the cache costs ~20% from 16k to 130k, not the 30% at 65k the audit estimated. Decisions: deploy the final set; MTP depth stays 3; LRU cache not deployed (neutral when the page cache is healthy, which it was even at 78 GB resident); wsum10 must not run at decode widths (-7% decode). Swap stayed at 182 MB through every run including int8 tables. Deployment: prod/run-omlx.sh env block "Round 2", prod/bootstrap.py import-time and post-load hooks.

Production after round 2 (2026-09-12 03:05, all six patches confirmed in /var/log/omlx/stderr.log): cold 65k prefill 1544 tok/s median (1572, 1515), decode 62.3 tok/s; agentic suite no-think: short decode 86.1, coding 5/5, tool calls ok at short and 25k context, no leaked markup; no-think alias 5/5, 81.9 tok/s. Swap unchanged at 182 MB with the int8 tables resident. Running record: 2026-09-11 baseline 1035 prefill / 47.7 decode -> 1438 / 61.5 (round 1) -> 1544 / 62.3 (round 2).

## 2026-09-12 12:40: concurrency finding (production)

Sweep at 1/2/4 streams (short prompts, 300-400 tokens, thinking off). With the 100 GB guard and the int8 tables resident (85.7 GB used vs 84.6 GB soft limit) aggregate was flat: 78.6 / 75.3 / 75.9 tok/s. Cause (scheduler.py `_schedule_waiting`, "Generation memory guard"): once one request is admitted, new ones are deferred while usage exceeds the soft limit, so requests ran strictly one after another. Fix: memory guard ceiling 100 -> 110 GB (soft 93.5, hard 104.5), applied live via POST /admin/api/global-settings and persisted in prod/run-omlx.sh (rewrites the plist's 100). After: 79.4 / 81.8 / 116.3 tok/s aggregate (per-stream 79 / 41 / 29). Two streams gain nothing because oMLX routes MTP only for a lone request and falls back to plain batched decode (41 tok/s per stream) when a peer exists; four streams reach 1.46x. The enforcer also dropped from "soft" to "ok" pressure, which removes the adaptive prefill chunk throttle.

## 2026-09-12 14:00: round 3 results (workbench, one server at a time)

Decode probe = four 600-token greedy generations (edit, code, prose, json), tokens/s median; cold prefill = one 65k rep. GPU power drift over the two hours: the first base run measured 89.1 / 1501, the base repeat 81.1 / 1372, so every candidate is judged against the repeat.

| config | decode probe | MTP tok/cycle, accept | cold prefill | verdict |
|---|---|---|---|---|
| base (first, cool) | 89.1 | 2.79, 80.9% | 1501 | |
| base repeat | 81.1 | | 1372 | reference |
| 8-bit MTP draft block (+1.27 GB) | 81.4 | 2.93, 80.7% | 1364 | neutral: acceptance identical to 4-bit; out |
| 8-bit draft, small projections only | 82.5 | 2.80, 78.6% | 1408 | neutral; out |
| confidence-gated depth (ceiling 5) | 75.8 | 3.00, 71.9% | 1349 | -6%: deeper drafts cost more verify rows than they return; out |
| GDN chunk kernel (PR #4020 C=8) | 82.1 | 2.80, 79.5% | 1424 | neutral (+0.7% predicted, inside noise); optional |
| n-gram copy lane | 82.6 | | 1415 | neutral on this probe; optional |
| all decode patches | 80.3 | | 1418 | neutral |
| int8 gather incl. down projections | 81.7 | 2.79, 80.0% | 1409 | no gain over gate_up-only; keep down off |
| weight-stationary gather + int8 | 80.7 | | 1365 | neutral (int8 owns the big shapes) |
| weight-stationary gather, int8 off | 84.2 | | 1342 | recovers about half of int8's gain at zero memory; fallback option |
| park policy | 69.3 | | 1176 | worse at short and long context; out |
| fine cache boundary 512 | | | | broke the store path: zero cache hits on every warm turn; out |
| batched MTP verify | 1 stream 71, 2 streams 55, 4 streams one request failed, 8 streams 47.5 aggregate | | | plain batching gives 60 / 73 / 97 / 130; fused path never engaged; out |

Confirmed already right: MTP verify uses the sparse QSA arm; prefill is serialised; gate_up is fused; depth already adapts per cycle; the prose prompt never parks (the earlier 35 tok/s was a contended run).
Net: round 3 deploys nothing new. Production config stays at the round-2 set plus the 110 GB guard. Optional zero-cost extras validated as exact and neutral: GDN chunk kernel, copy lane, weight-stationary gather.

## 2026-09-12 14:45: rounds 4, 4b, 5 and the thermal finding

Round 4 (paired base/extras/base/extras; extras = hc-fuse2 + gdn-scan C=8 + verify weighted sum, all bit-identical or 1 ULP, zero memory): pair a 1527/91.5 vs 1304/74.0 (confounded: the first run of a round is on a rested GPU), pair b 1376/79.9 vs 1421/79.7 (+3.3% prefill, decode equal). Bisect (4b, hot GPU): hc-fuse2 1403 (+2%), verify weighted sum 1410 and decode 82.3 (neutral), fast top-K did not install under the loader (0.1-0.4% at best anyway), base-c 1378 / 82.7. Decision: hc-fuse2, gdn-scan and the verify weighted sum join the production set.

Thermal (powermetrics 14:28-14:33, Max ran it): thermal pressure "Heavy" throughout, GPU 1060-1260 MHz against a 1620 MHz top bin at only 21-31 W. The all-day "drift" was thermal throttling. Round 5 with Macs Fan Control at maximum: cold 65k prefill 1604 tok/s (hot auto-fan base-c 1378, +16%; best number ever recorded on this machine), decode probe 91.2 (82.7, +10%), long-context tail 59.7 (55.8). Decision: keep a fan floor while serving (Macs Fan Control, high or max under load; the app can be left as a login item). Every earlier hot-GPU comparison in rounds 3 and 4 stands as a paired comparison but understates absolute throughput by ~10-16%.

Long-context concurrency (4c, 68k prompts): dense batched attention (today) vs the batched sparse QSA arm (round3/qsa-batched, 1 ULP). With warm prefixes the sparse arm gives 20 tok/s per stream at 2 streams and 12-16 at 4; the dense figures under identical warm conditions are being measured (4c2). MTP is inactive for any batch above one row (oMLX policy).
4c2 (dense, warm prefixes, 68k): 2 streams 16.1 per stream / 31.5 aggregate, 4 streams 10.4 / 35.2. Sparse (4c): 19.7 / 38.4 and 14.2 / 44.9. Decision: OMLX_QSA_BATCHED_SPARSE=1 with min ctx 8192 joins production. Final production set (2026-09-12 15:00): round-2 patches + hc-fuse2 + gdn-scan C=8 + verify weighted sum + batched sparse QSA, guard 110 GB, fan floor.

## 2026-09-12 15:55: cache boundary A/B (6-turn 32k probe, cold SSD cache per arm, paired twice)

| arm | per-turn latency (s) | cached tokens per turn | median |
|---|---|---|---|
| base a | 16.35 / 1.72 / 2.47 / 2.17 / 1.82 / 1.28 | 0 / 24576 / 24576 / 26624 / 28672 / 30720 | 2.17 |
| fine 512 a | 17.62 / 2.69 / 1.66 / 3.11 / 1.64 / 1.57 | 0 / 24576 / 26112 / 26112 / 29184 / 30720 | 2.69 |
| base b | 16.66 / 1.68 / 2.52 / 2.18 / 1.86 / 1.30 | same as a | 2.18 |
| fine 512 b | 17.76 / 2.61 / 1.66 / 3.02 / 1.63 / 1.54 | same as a | 2.61 |

Reproducible: the fine tail stores on the 512 grid for most turns (26112, 29184, 30720, 31744) and turn 3 gains 0.8 s, but the store after turn 3 is rejected every time ("Rejecting split-GDN placeholder block 150 because its recurrent checkpoint was not committed" at 27648), so turn 4 loses 0.9 s, and turn 2 is 0.9 s slower than base at an identical cached count. Net worse. Not deployed; second pass dispatched with this log.
