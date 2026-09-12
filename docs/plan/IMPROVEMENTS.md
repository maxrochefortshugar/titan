# Every improvement to the M5 Max inference server, 2026-09-11 to 2026-09-12

Machine: MacBook Pro 16" M5 Max, 40-core GPU, 128 GB, macOS 26.5. Model: Qwen3.8-Flash-Next oQ4e-mtp (125B-A6B MoE). Server: oMLX. Numbers are measured on this machine unless marked as estimates. Sections are in the order the work happened.

## 1. Turning the laptop into a server (2026-09-11)

| change | before | after |
|---|---|---|
| Inference server | mlx-serve ("MLX Core") 0.x, LM Studio, EXO, litellm all installed | oMLX only, first 0.6.4 then 0.7.0.dev2 |
| Process model | server started by hand from a menu-bar app | LaunchDaemon io.titan.omlx, starts at boot before login, KeepAlive, log rotation, wired GPU limit raised 96 -> 118 GB |
| Network | localhost only | Tailscale Serve with automatic TLS, tailnet only, API key required |
| Concurrency | 1 request | 4 |
| Context window | 32k | 262k |
| Prefix cache | SSD tier only | SSD tier + hot RAM tier (16 GB, later tuned to 4 GB after measurement showed the same hit rate at lower memory pressure) |
| Machine hygiene | 13 login items, Spotlight, Time Machine, screensaver, EXO daemon | Docker only; Spotlight and Time Machine off; server power profile, restart after power loss; display sleep instead of system sleep |
| Disk | 410 GB free, three overlapping model stores | 695 GB free, one store with one model |
| Apps removed | Draw Things (+53 GB sandbox), Comfy, EXO, pgAdmin, colima, lima, postgresql, GarageBand, iMovie, Xcode, mlx-serve, LM Studio, litellm | Docker, ChatGPT, Signal kept |
| Models removed | Qwen3.6-35B, Qwen3-Coder-Next 8-bit, Devstral 24B, Nemotron 4B, Qwen3-4B, Qwen3.8-27B AWQ, DFlash2 draft, mtp-head experiments, oQ5e partial | Flash-Next oQ4e only |

## 2. Model and client decisions

- Flash-Next oQ4e-mtp chosen after two research passes: the strongest open model that fits 128 GB with usable context (SWE-bench Pro 62.5). Alternatives ruled out by size or tool-calling quality: GLM-5.3-Flash (178 GB), DeepSeek-V4/V4.1-Flash (250-430 GB), MiniMax M2.x, gpt-oss-120b, Gemma 4 26B MoE, Nemotron 3 Super, Kimi K2/K3, Qwen3.8-27B dense.
- Fast model dropped. Claude Code's Haiku slot and subagents use a no-think profile alias of Flash-Next itself (same engine, shared prefix cache, zero extra memory), created as an oMLX profile.
- Server-side default reasoning effort set to medium (the template's own quiet setting); low, xhigh and off remain per-request overrides.
- Client wiring: claude-local wrapper on the M4 with per-project pinning (local or cloud per repo), opencode provider with thinking variants (low, medium, xhigh, off, cycled with ctrl+t), CLAUDE_CODE_ATTRIBUTION_HEADER=0 so the prefix cache hits every turn, non-essential traffic off, 10-minute API timeout.
- oMLX 0.6.4 -> 0.7.0.dev2 (mlx 0.32.2, which also fixed a long-prefill corruption bug for quantised MoE on M5).
- November plan: Mac Studio M5 Ultra 256 GB as an independent second node (no clustering: tensor parallel over mismatched nodes was measured by others to gain nothing). Ultra model: GLM-5.3-Flash 4-bit or Flash-Next 6-bit.

## 3. Tuning found from the logs (2026-09-11)

| finding | fix | effect |
|---|---|---|
| hot cache 16 GB competing with weights, memory pressure warnings | 4 GB hot cache | throttling gone |
| daemon relaunch reused stale CLI args (kickstart) | bootout + bootstrap after plist edits | settings actually apply |
| decode 48 tok/s short context, agent turns effectively 9-15 tok/s due to prefix misses | attribution header off, chunked prefill on, cache tiers tuned | agent-turn effective 68 tok/s |

## 4. Kernel round 1 (2026-09-11 to 12): finding the real bottleneck

Measured hardware ceilings first (roofline harness): bf16 tensor 65.7 TFLOPS, int8 129.8, int8 x int4 120, copy 549 GB/s, read 718 GB/s. Established that GPU power drifts +33% under sustained prefill, so all A/Bs use paired baselines with cooldowns.

Ruled out as non-factors by measurement: prefill chunk size (2048 is right; 512 is 22% worse per token, 4096 20% worse), cache block granularity, MTP depth (3 is right; off costs 1.5x), SSD cache writes, small-M quantised matmul kernel (lost 3-13% in situ), oMLX's INT8 prefill flag (excludes this model type).

| change | mechanism | gain |
|---|---|---|
| Packed n-gram table (rows layout, streamed from SSD) | the 51B-parameter n-gram table was stored as three tensors per row, so every lookup faulted three pages: 93,046 pages per 2048-token chunk, 460x read amplification, ~963 ms. Repacked contiguously: 31,855 pages, 456 ms cold. Bit-exact. | prefill +8-15%, decode +18-28% (the lookup runs per decode token too) |
| bf16 grouped RMSNorm | fp32 norm replaced by bf16 with fp32 accumulate, <=1 ULP | +8% per chunk |
| Resident n-gram table | tried; 32 GB wired on top of 73 GB caused two freezes and a kernel panic | banned on this machine; planar files deleted |

Production after round 1: cold 65k prefill 1035 -> 1438 tok/s, decode 47.7 -> 61.5, coding probes 4/5 -> 5/5.

## 5. Kernel round 2 (2026-09-12): the remaining prefill items

Audit (disassembly-level) corrected the record: read bandwidth is 718 GB/s not 549, so the decode roofline is 171 tok/s; the expert gather runs on NAX with bf16 operands after per-tile dequant and its 32-row tile streams every expert twice against ~40 rows per expert; no int4 tensor datatype exists in mlx's Metal backend, so "native int4" is not a lever.

| change | exactness | in situ |
|---|---|---|
| Fused GDN grouped norm + sigmoid gate at prefill | bit-identical, 7.5x on the kernel | +2.2% prefill |
| Fused MoE weighted sum for top_k=10 (oMLX's native kernel only covered 6 and 8) | bit-identical in default mode, 4.9x on the kernel | +3.5% prefill; restricted to prefill after it cost 7% at decode |
| int8 x int4 expert gather, gate_up projections only | 1.42x on the gathers | +3.9% prefill for ~7.5 GB of tables |
| MTP shortlist drafter (draft steps 2-3 on a top-K head instead of the full 248k vocabulary) | verified output identical | +2% decode |
| Tested and rejected | MTP depth 4 (-6% decode), LRU n-gram row cache (neutral), mlx-vlm 0.7.0 swap (breaks two patches) | |

Production after round 2: cold 65k prefill 1544 tok/s, decode 62.3, all suites pass. Prompt-length profile: 8k 1641, 16k 1677, 32k 1519, 65k 1441, 130k 1321 tok/s (attention over the cache costs ~20% from 16k to 130k, not the 30% estimated).

## 6. Operations improvements

- Production launches through prod/run-omlx.sh + prod/bootstrap.py: patch toggles are env vars in the wrapper; every patch keeps the stock path on failure; a daemon restart needs no sudo (kill the process, launchd relaunches through the wrapper).
- Isolated workbench on port 8084 with its own cache dir, memory guard, log rotation, watchdog (swap growth and kernel pressure), and reusable probes: cold 65k prefill, decode-heavy four-prompt probe, concurrency sweep, agentic suite, MTP acceptance summariser.
- Memory guard 100 -> 110 GB after finding that the soft limit serialised concurrent requests (aggregate flat at 76 tok/s for 1, 2 and 4 streams; after: 79 / 82 / 116).

## 7. Research conclusions that shaped the work

- Prefill at 1544 tok/s is ahead of every published Flash-Next figure (700-1113 elsewhere). The packed table is ahead of upstream (llama.cpp has the same idea as an open PR).
- Decode at 62 is at MTPLX's plain-decode level for this model on M5 Max, i.e. the verify pass eats most of the speculative gain; mlx-serve reaches 83-93 on comparable bandwidth via cheaper speculation. Decode is 36% of the read roofline; the gap is occupancy (one row per kernel) and speculation efficiency.
- Neural Engine, CPU SME and media engine ruled out with numbers. SSD reads are 13.6 GB/s (double the earlier assumption).
- No better model for 128 GB exists as of 2026-09-12; oQ5e (5-bit experts, ~90-96 GB resident) is the only quality step up that fits.

## 8. Round 3 (in progress, 2026-09-12 afternoon)

Built and under measurement: 8-bit MTP draft block (the shipped quant left its seven draft modules at 4-bit with relative error up to 0.435; sidecar +1.27 GB), confidence-gated draft depth, n-gram copy lane (up to 14-token blocks from the prompt, verified exactly), GDN chunk kernel from mlx PR #4020 (exact variant +0.7%; NAX variant 3.15x blocked on precision), weight-stationary bf16 expert gather (addresses the double weight stream), batched MTP for concurrent requests, sparse attention for batched decode, remaining hyper-connection fusion, fused top-K for the shortlist, packed-reader worker tuning, finer cache boundaries for mid-context edits.

Confirmed already right (no change needed): MTP verify uses the sparse attention arm; prefill is serialised; the fused gate_up projection is in place; depth already adapts per cycle.

Results (all against a drifted base repeat of 81.1 tok/s decode probe and 1372 tok/s cold prefill): every round-3 candidate was neutral or negative. The 8-bit draft block did not move acceptance (80.7% vs 80.9%), the confidence gate drafted deeper but lost 6%, the copy lane and GDN kernel were inside noise, the gather kernel recovers about half of int8's gain at zero memory, the park policy and the fine cache boundary regressed, and batched MTP lost to plain batching at every stream count (plain batching: 60 / 73 / 97 / 130 tok/s aggregate at 1 / 2 / 4 / 8 streams once the memory guard stopped serialising requests). Nothing from round 3 is deployed. Full table in kernels/REPORT.md.

## 9. Where this leaves the machine (2026-09-12 14:00)

Production config: round-2 patch set (packed n-gram table, bf16 norms, fused GDN gated norm, fused weighted sum at prefill, int8 gate_up gather, MTP shortlist drafter), memory guard 110 GB, medium reasoning default, one model only. Measured: cold 65k prefill 1544 tok/s on a cool GPU (1370 to 1420 after an hour of sustained load), decode 62 tok/s at long context and 80 to 90 at short context, 130 tok/s aggregate at eight concurrent streams. Prefill is at the practical ceiling of this GPU for this architecture; decode is bounded by one-row occupancy and speculation acceptance, and every cheap lever on it has now been measured.

## 10. Round 4 and the thermal finding (2026-09-12, afternoon)

- Three more exact, zero-memory kernels measured together in a paired A/B and deployed: fused prefill hyper-connection block (bit-identical, +2%), chunked GDN prefill scan from mlx PR #4020 (state error 6e-7, neutral to +0.7%), fused weighted sum on the MTP verify layout (bit-identical, neutral). Together +3.3% prefill on the paired run, decode unchanged.
- The machine had been thermally throttled all day. powermetrics showed thermal pressure "Heavy" with the GPU at 1060 to 1260 MHz against a 1620 MHz ceiling, at only 21 to 31 W. With Macs Fan Control at maximum: cold 65k prefill 1378 -> 1604 tok/s (+16%), decode probe 82.7 -> 91.2 (+10%), long-context decode 55.8 -> 59.7. This is the largest single gain since the packed n-gram table, and it means every hot-GPU number earlier in the day understated the machine by 10 to 16%. Decision: a fan floor while serving is part of the production setup.
- Long-context concurrency: the batched sparse attention arm (1 ULP) replaces dense attention over the whole cache when several requests decode together; measured 20 tok/s per stream at two 68k streams with warm prefixes, against 12 on the dense path.
- Fixed in passing: the memory guard serialising concurrent requests (100 -> 110 GB); a packing-width slip in the int8 gather warmup; a replaced benchmark script that sent no API key.

## 11. Closing numbers (2026-09-12 15:15, production, final patch set, fan floor)

| measure | 2026-09-11 start | now |
|---|---|---|
| cold 65k-token prefill | 1035 tok/s | 1642 tok/s |
| decode after a long prefill | 47.7 tok/s | 71.0 tok/s |
| short-context decode, thinking off | ~48 tok/s | 84 to 89 tok/s |
| 49k cached follow-up | 4.1 s | 1.4 s |
| coding probes | 4/5 | 5/5 |
| tool calls at short and 25k context | ok | ok, no leaked markup |
| concurrent streams | serialised | 130 tok/s aggregate at 8 short streams; +22 to 28% per stream at 68k with 2 to 4 streams |
| GPU clock under load | throttled to 1060 to 1260 MHz | full clock with the fan floor |

Everything deployed is bit-identical or within one bf16 ULP of the stock computation; nothing changes model output. All ten patches verified live in the daemon log after the final restart, model pinned, swap flat.

What did not survive measurement today, kept on record so it is not retried blindly: 8-bit MTP draft block (acceptance unchanged), confidence-gated draft depth (-6%), MTP park policy (worse), n-gram copy lane (neutral), fine cache boundary (broke the store path), batched MTP verify (lost to plain batching), NAX GDN kernel (exact after a hi/lo split but register-bound and slower), int8 down projections (no gain), LRU n-gram row cache (neutral), mlx-vlm 0.7.0 swap (breaks two patches), Neural Engine offload (no eligible work).

Open items for another day: a finer cache boundary that also emits GDN snapshots at the finer grid (58% of warm-turn recompute is text already processed), a fused batched MTP verify that actually engages, the weight-stationary gather as a zero-memory replacement for the int8 tables if memory gets tight, and the oQ5e 5-bit build as the only quality step up that fits 128 GB.

## 12. Round 6: decode budget measured, cache boundary fixed (2026-09-12, late afternoon)

- Decode profiler (kernels/round4/decode-profile) with under 1% overhead. Budget per decode cycle: verify forward 75%, acceptance host sync 13%, draft chain 11%. Verify runs at 45% of measured memory bandwidth. At 64k context the median accepted tokens per cycle drops to one, so long-context decode is an acceptance problem. This is the baseline the Titan engine is designed against.
- Prefix-cache boundary fix, second version (kernels/round4/cache-boundary, OMLX_CACHE_FINE_TAIL=512): the tail of every prompt is now stored on a 512-token grid instead of 2048, with the GDN recurrent snapshot committed at each fine boundary. Multi-turn median latency on the 32k probe 2.59 s to 2.10 s, warm-turn total 11.6 s to 9.9 s, zero rejected blocks. Deployed in prod/run-omlx.sh; live at the next restart.
- Production is stopped at Max's request while Titan work proceeds. Restart: `sudo launchctl kickstart system/io.titan.omlx`, then re-pin the model and run prod/verify-and-bench.sh.
