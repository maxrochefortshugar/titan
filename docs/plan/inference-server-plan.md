# M5 Max inference server: decisions and plan

Date: 2026-09-11. Machine: MacBook Pro 16" M5 Max, 40-core GPU, 128 GB, 1.8 TB, macOS 26.5.
Tailscale host `m5-128gb` (<server-tailnet-ip>, `<server>.<tailnet>.ts.net`). Second node: Mac Studio M5 Ultra 256 GB, November 2026.

## Decisions

### Server: oMLX 0.6.4 (already installed)
Why: it is the only Apple Silicon server built around the thing agentic coding actually needs, a paged
KV cache with a hot RAM tier and a cold SSD tier that survives prompt edits and restarts. It has
concurrent requests, Lightning MTP, a real Anthropic Messages endpoint with Claude Code specific
handling (context scaling for auto-compact, SSE keep-alive during long prefill), per-family tool-call
parsers, and an experimental cluster mode that plans unequal shards, which is what the 128 + 256 pairing
needs in November. It is also where your existing tuning lives (491 requests logged).

Rejected: mlx-serve (good Anthropic endpoint, but two releases behind, concurrency defaulted to 1,
smaller project), LM Studio (no continuous batching in the MLX engine), mlx_lm.server (open deadlock and
OOM issues under agentic load), llama.cpp and Ollama (no access to the M5 neural accelerators; MLX is
about 3 to 4x faster at prefill on this chip), vLLM Metal (no Anthropic API yet).

### Primary model: Qwen3.8-Flash-Next oQ4e-mtp (already configured as default)
125B total, 6B active MoE with a 51B n-gram table streamed from SSD. Highest SWE-bench Pro of any open
model that fits in 128 GB (62.5). Resident 70 GB. KV cache about 2.5 GB per 100k tokens, so the
full 262k window is affordable. Measured today: 37 tok/s decode; your logs show 35 to 45 tok/s at 40k
context in real tool loops. License is qwen-community-1.0, fine for personal use.

Fast model: dropped on 2026-09-12 (Qwen3.6-35B-A3B-4bit weights still on disk, 19 GB, unused). Subagents and
Claude Code's Haiku slot use the alias `Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think`: same engine,
thinking off, shared prefix cache, no extra memory. Server default reasoning effort is medium (set 2026-09-12 via model chat_template_kwargs); clients can override with reasoning_effort low/medium/xhigh or enable_thinking false. The earlier Qwen3.6 numbers below are kept for the record.

Fallback that runs on any stock runtime: Qwen3-Coder-Next 8-bit (moved into the oMLX dir by
cleanup-models.sh). Best prefill speed measured on M5 Max, Apache 2.0, no thinking mode.

Ruled out at 128 GB, with the reason:
- GLM-5.3-Flash: 178 GB at 4-bit. Nothing in the current GLM line fits; there is no "Air" any more.
- DeepSeek-V4-Flash: 151 GB at 4-bit; the 2.4-bit build that fits loses real quality (MMLU-Pro 0.65 to 0.57).
- MiniMax M2.x: full attention on all 62 layers, 25 GB of KV per 100k tokens; the 3-bit builds have reported loops and path corruption.
- gpt-oss-120b: 13 months old, weak on Terminal-Bench, Harmony tool parsing broken in MLX.
- Gemma 4 26B MoE: tool calling broken in mlx-lm. Muse Glimmer 30B and Qwen3.8-27B: dense, 15 to 30 tok/s.
- Nemotron 3 Super: poor tool-calling reports in every coding harness.
- Kimi K2/K3: 247 GB at 1-bit. Not reachable on either machine.

### Clustering: no, not yet
- A 384 GB pool unlocks no model the 256 GB Ultra cannot run alone. DeepSeek 671B class needs 380 to 404 GB before KV.
- Tensor parallel over Thunderbolt 5 RDMA beat pipeline parallel by 2.3% on matched nodes. Your nodes are mismatched in both memory and bandwidth (614 GB/s vs 1.2 TB/s), the worst case.
- Two independent servers give two concurrent sessions and simple routing: Claude Code's Opus/Sonnet slots point at the Ultra, the Haiku slot at the Max.
- The one experiment worth running: oMLX cluster mode for prefill. Prefill parallelises and is what you wait on in agentic loops. Nobody has published an asymmetric two-node measurement. Benchmark TTFT at 8k/32k/64k prompts against single node; keep it only if it wins.

## Measured on this machine (2026-09-11)
| Probe | Flash-Next oQ4e | Qwen3.6-35B think | Qwen3.6-35B no-think |
|---|---|---|---|
| Short decode | 48 tok/s | 135 tok/s | 133 tok/s |
| 50k prefill, cold | ~50 s | 63 s | (cached) |
| 50k prefix, cached | 4 s | 6 s | 2 s |
| 50k needle lookups | 2/2 | 2/2 | 1/2 |
| Coding tasks (tests run) | 4/5 | 2/5 (3 hit 16k cap) | 5/5 |
| Tool call, short / at 25k | ok / n.a. | ok / ok | ok / ok |

## oMLX 0.7.0.dev2 installed 2026-09-11 17:33 (bundles mlx 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.3)
- Installed by replacing /Applications/oMLX.app (0.6.4 kept at ~/Downloads/omlx/oMLX-0.6.4.app.bak) and SIGTERM-ing omlx-server; launchd relaunched it. Settings and the no-think alias survived. Short decode went 48 -> 60 tok/s on the same prompt.
- Rollback: quit app, `mv ~/Downloads/omlx/oMLX-0.6.4.app.bak /Applications/oMLX.app`, restore ~/.omlx/settings.json.backup-*-pre-0.7.0.dev2, `pkill -x omlx-server`.
- Kernel dev env: ~/inference-server/kdev (mlx 0.32.2) and kbench.py (roofline microbench). oMLX integration point: omlx/patches/*.py modules using mx.fast.metal_kernel JIT kernels (see m5_gather_qmm.py, qwen35_verify_qmm.py, qwen35_moe_gate_up.py as templates).

## Live-session tuning (2026-09-11, from the daemon log)
- Chunked prefill ON. Without it the recurrent-state snapshots this model family needs are never taken, so
  cache stores are skipped and each turn re-prefills 6k to 13k tokens; it also cut peak prefill memory (was 22 to 26 GB).
- Hot cache 16 GB was too much: prefill throttling at 91 to 95 GB and the fast model being evicted on every turn. Now 4 GB (SSD tier unchanged).
- The oMLX daemon rewrites settings.json from its own arguments at startup. Change settings in the plist AND the file,
  then reload with bootout + bootstrap. `launchctl kickstart` restarts the process with the OLD arguments.
  Correct reload:  sudo launchctl bootout system/io.titan.omlx; sudo launchctl bootstrap system /Library/LaunchDaemons/io.titan.omlx.plist
- The menu-bar app shows "Start server" because the daemon owns the server now. Do not start it from the app.

## Paused (network): resume with `bash ~/inference-server/resume-downloads.sh`
- oQ5e build of Flash-Next (128 GB, 1.9 GB fetched). switch-to-oq5e.sh then verifies, benchmarks, and promotes it only if it beats oQ4e.
- The three fast-model candidates, benchmarked thinking on and off.
- Trade-off to decide after the numbers: oQ5e at ~92 GB resident leaves no room to keep the fast model warm under the 110 GB guard; oQ4e at 70 GB does.

## Done today
- Removed: Draw Things (+53 GB sandbox), Comfy, EXO app, pgAdmin, colima, lima, postgresql, mlx-serve ("MLX Core"), LM Studio, litellm. Free disk 410 GB to 495 GB.
- oMLX settings: concurrency 1 to 4, hot RAM cache 0 to 16 GB (SSD tier stays on), context window 32k to 262k, real API key generated (`~/.omlx/api_key.txt`), tailnet hostname added to allowed hosts, backup in `~/.omlx/settings.json.backup-20260911-before-server-hardening`.
- Published over Tailscale Serve with automatic TLS: `https://<server>.<tailnet>.ts.net` (tailnet only, never Funnel). Verified a completion end to end through it.
- Verified oMLX runs headless via `omlx serve` (needed for boot-time start).
- Screensaver disabled. Login items down to Docker only.
- Started download of Qwen3.6-35B-A3B-4bit into the oMLX model dir.

## To run (need your password)
1. `bash remove-root-owned.sh`: GarageBand, iMovie, Xcode, Creator Studio apps, EXO daemon, repoint dev tools to Command Line Tools.
2. `bash harden-server.sh`: pmset server profile, restart after power loss, SSH on, auto-install of macOS updates off, Spotlight off, Time Machine off, EXO network services removed, GPU wired limit 118 GB (default cap is 96 GB, which blocks any model over that), oMLX as a boot-time LaunchDaemon with log rotation.
3. `bash cleanup-models.sh` after reviewing it: removes about 330 GB of duplicates across the three old model stores.
4. In System Settings: charge limit 80%, power mode Automatic (not High Power), restrict Remote Login and Screen Sharing to your user, Do Not Disturb, sign out of iCloud.

## One decision only you can make: FileVault
FileVault is on. After a power cut the machine stops at the pre-boot unlock screen. Tailscale is not
running there (its state is on the encrypted volume), so you cannot reach it from the tailnet.
- Option A, keep FileVault: macOS 26 supports unlocking over SSH at the pre-boot screen, but reliably only over wired Ethernet, and the GUI Tailscale app cannot start before login. You would need a Thunderbolt Ethernet adapter, the Homebrew `tailscale` daemon instead of the app, and a device on your home LAN to recover from. Encryption at rest stays, which matters because `~/Engineering` (230 GB of your code) lives here.
- Option B, disable FileVault and enable auto-login: the machine self-heals after any outage with nobody home, the GUI Tailscale app is fine, and the setup is much simpler. Weights are public files. Protect anything sensitive in an encrypted disk image instead.
Recipe for A, if chosen: buy a Thunderbolt 5 dock or adapter with 2.5 GbE (needed for the Studio link anyway); `brew install tailscale && sudo tailscaled install-system-daemon && sudo tailscale up --ssh`; then remove the GUI Tailscale app; keep Remote Login on; pull the power once and confirm `ssh maxshugar@<lan-ip>` unlocks the disk from the pre-boot screen. Everything else in the plan stays the same.
Recommendation: B for a home machine that never moves, unless the code on it must stay encrypted at rest. Until you decide, `autorestart 1` is set and the laptop's own battery covers roughly an hour of outage, so this only bites on a long power cut.

## Thermals and battery (research summary)
- The 16" chassis holds full GPU clocks indefinitely in Automatic mode (Notebookcheck: no measurable drop; the 14" loses 25%). No fan-curve hacks needed; Macs Fan Control can go.
- High Power Mode changes only the fan curve, not GPU clocks: 13 dB louder for nothing. Low Power Mode caps GPU frequency by about a third. Leave it on Automatic.
- Lid open, flat, 10 cm clearance behind the hinge, display off via `displaysleep 5`. Clamshell has no measured benefit on this generation and forces the external-display sleep hack. Vertical stands and bottom cooling pads do nothing; intake is at the hinge.
- Decode is memory-bound and draws far less than a stress test. Expect 60 to 90 W sustained, keyboard deck low 40s C.
- Battery: macOS 26.4+ has a native 80% charge limit. Set it and forget AlDente. At 39 cycles and 100% health this is cheap insurance.
- Wired Ethernet matters more than expected: lower jitter on token streams, and it is the only reliable pre-boot unlock path. A Thunderbolt 5 dock with 2.5 GbE covers this and the future link to the Studio.
- Put a small UPS on the router and dock, not the laptop. The laptop is its own UPS.

## Findings 2026-09-11 (colibri, DeepSeek V4, macOS 27)
- colibri (JustVugg/colibri): expert streaming across VRAM/RAM/NVMe. Runs on M5 Max 128 GB at 1.8 tok/s for GLM-5.2. Every SSD expert-streaming runtime measured lands at 0.03 to 4 tok/s. It is a "fits at all" technique, not a speed one. Skip.
- antirez/ds4: C + Metal runtime, beta. DeepSeek V4 Flash on M5 Max 128 GB: ~500 tok/s prefill, 35 to 40 tok/s decode. GLM-5.3-Flash Q2 is ~90 GB resident on 128 GB. Worth a quality A/B against Flash-Next; Q2 may lose more than it gains.
- oMLX 0.7.0.dev2 (2026-09-11): DeepSeek-V4.1-Flash support with an oQ3e build designed for 256 GB with the Engram table on SSD (39.7 tok/s on M3 Ultra at 64k with MTP); +34% prefill for Qwen on M5 via INT8 activations (827 tok/s at 32k on M5 Max); optional MoE expert SSD offload (slower, avoid). Upgrade the server to 0.7.0.dev2 when a quiet window allows; it is a dev build, keep the 0.6.4 settings backups.
- macOS 27 "Golden Gate" ships 2026-09-14. Neural accelerators and Thunderbolt RDMA already work on 26.2+. 27 adds Metal tensor features MLX has not adopted yet. Do not upgrade the server on day one; wait for MLX/oMLX to publish macOS 27 numbers.
- Mac Studio M5: 256 GB ships 2026-09-22; a 512 GB configuration arrives late October. If the November date is flexible, 512 GB runs GLM-5.3-Flash 4-bit and DeepSeek-V4.1-Flash fully resident with no offload tricks.

## November: second node
- Ultra (256 GB) first choice now: **DeepSeek-V4.1-Flash oQ3e** on oMLX 0.7.x with Engram on SSD and DSpark MTP (built for 256 GB; Terminal-Bench 2.1 90.6 in DeepSeek's harness). Second: **GLM-5.3-Flash oQ4e-mtp** (`dfp-official/GLM-5.3-Flash-oQ4e-mtp`, 181.6 GB, ~183 GB resident at 100k context because only 11 of 45 layers hold a context-sized cache). Zhipu reports Terminal-Bench 2.1 84.3 in the Claude Code harness and DeepSWE 63.4 vs Flash-Next's 58.7. Evidence-backed alternative: `orcarouter/GLM-5.3-Flash-MLX` 4-bit (204 GB, +3% perplexity, 96% top-1 agreement vs FP8). The only measured near-lossless build is orcarouter 6-bit at 296 GB, which does not fit.
- DeepSeek-V4.1-Flash (released 2026-09-10) is out of reach: about 251 GiB resident even with its memory tables on SSD, more than the Ultra's usable memory. oMLX support merged the day after release and is untested. Revisit only if a smaller variant appears.
- Before deploying GLM-5.3-Flash on oMLX for long sessions, check that issue #3421 (hot cache growth to hundreds of GB, ends in a reboot) is fixed; the workaround is a watchdog that clears the hot cache.
- Max (128 GB): Flash-Next and the fast model, serving subagents and a second session.
- Routing: two Tailscale Serve hostnames, Claude Code model slots split across them. No LiteLLM unless you want per-key accounting.
- Try oMLX cluster mode over a Thunderbolt 5 cable for prefill only, with RDMA enabled from Recovery (`rdma_ctl enable`). Measure TTFT; tear down if it does not win.
- Keep the Max on mlx 0.32.2 or newer: older builds silently corrupt long one-shot prefill on quantised MoE models on M5 (fixed in 0.32.2). Check what oMLX bundles when it updates.

## Files
- `remove-root-owned.sh`, `harden-server.sh`, `cleanup-models.sh`: run in that order.
- `client-configs.md`: Claude Code, opencode and qwen-code configuration for your other machines.

## 2026-09-12 update: kernel patches in production
- Measured on the workbench (65k-token cold prompt): prefill 1347 -> 1453 tok/s, decode 48-52 -> 61.5 tok/s with the streamed packed n-gram table (rows mode) plus the bf16 grouped norm. Details in kernels/REPORT.md.
- Production runs through `prod/run-omlx.sh` (toggles live there; edit and `sudo launchctl kickstart -k system/io.titan.omlx`). Deploy or redeploy with `bash prod/deploy-optimisations.sh`.
- The n-gram table stays on SSD. Resident mode caused a kernel panic on this 128 GB machine; it is only for the 256 GB Ultra.
- Fast model dropped. Haiku and subagent slots use the alias `Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think` (same engine, thinking off). Re-run `m4/setup-m4.sh` on the M4 to pick this up.
- Memory guard lowered to 100 GB. The 32 GB planar half of `ple-packed/` (weight.u32, scales, biases) is only used by resident mode and can be deleted on this machine.

- 2026-09-12: model store trimmed to Flash-Next only (oQ4e in production; the oQ5e partial download deleted as well on 2026-09-12; resume-downloads.sh and the oQ5e/GLM queue scripts moved to archive/). Qwen3.6, Qwen3-Coder-Next 8-bit fallback, Devstral, Nemotron, Qwen3-4B, Qwen3.8-27B AWQ and the mtp-head experiments deleted.

## 2026-09-12 research pass (research/MODELS-2026-09-12.md, research/APPLE-SILICON-2026-09-12.md)
- Models: nothing new fits 128 GB better than Flash-Next oQ4e. November Ultra pick unchanged: GLM-5.3-Flash 4-bit (177.5 GB, licence unresolved) with Flash-Next 6-bit (148 GB, +0.0013 ppl vs bf16) as the second option. DeepSeek-V4.1-Flash moved further out of reach (427 GB, or 275 GB pruned at a quality cost); it is the only argument for the 512 GB config.
- Hardware: ANE unreachable for qwen4_exp and disabled on M5 by mlx-serve anyway; CPU/SME under 10% of GPU; media engine dead end; SSD reads measured at 13.6 GB/s on M5 Max (we assumed 7). Biggest idle resource: memory headroom and concurrency (rows per kernel).
- Next decode levers, in order: requantise the MTP draft head (mtp.fc_embedding / fc_hidden are 4-bit gs64 in the checkpoint; MTPLX measured 2.3x -> 3.0x going to 8-bit, ~1 GB), confidence-gated adaptive depth instead of fixed 3, a prompt-sliced n-gram copy lane for edit-heavy turns, confirm MTP verify uses the sparse QSA arm. Prefill: mlx PR #4020 GDN Metal kernels (1.86-2.16x on the scan at our head shape, unmerged), PR #4481/#4483 when merged. macOS 27 fp4/fp8 tensor formats: no runtime uses them yet, hold.
