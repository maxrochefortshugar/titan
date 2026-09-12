# Round 2 brief (2026-09-12)

Target: Qwen3.8-Flash-Next-oQ4e-mtp (mlx_vlm model_type qwen4_exp) on oMLX 0.7.0.dev2, MacBook Pro M5 Max 40-core, 128 GB, macOS 26.5.
Read first: ~/inference-server/kernels/AUDIT-2026-09-12.md (file:line evidence for every hot path), ~/inference-server/kernels/COMMON.md (conventions), ~/inference-server/kernels/REPORT.md (what already shipped).
Bundled runtime: /Applications/oMLX.app/Contents/Resources (omlx package at ./omlx, python at ./Python/cpython-3.11/bin/python3, mlx + mlx_vlm + mlx_lm under ./Python/framework-mlx-base/lib/python3.11/site-packages). Dev venv with the same mlx 0.32.2: ~/inference-server/kdev/bin/python.

HARD RULES
- Production oMLX (port 8083) holds 70 GB of the model. NEVER load the model, never open the safetensors, never start a server, never touch ports 8083/8084. A kernel panic already happened from memory pressure.
- Synthetic-tensor tests and microbenchmarks only, under 2 GB of GPU memory, with kdev python. Check `vm_stat` free pages before anything above 500 MB.
- Do not edit anything under /Applications/oMLX.app. Do not edit files outside your own workstream directory ~/inference-server/kernels/round2/<name>/ except your REPORT.md.
- Every kernel must ship with an exactness test against the stock MLX-ops path (report max abs and ULP error over random inputs at the real shapes, bf16) and a microbench (median of 15, warm, mx.synchronize, CHAIN=10 as in ~/inference-server/kbench.py).
- Integration contract: expose `install() -> bool` in `patch.py` (idempotent, returns False and leaves the stock path when preconditions fail), gated by an env var named in your report, so ~/inference-server/prod/bootstrap.py can import it by path (importlib spec_from_file_location; do not rely on the module name "patch"). Install must run AFTER the model is loaded if it touches instances, or at import time if it monkeypatches module functions. Say which in the report.
- Report format (REPORT.md in your dir, under 800 words plus tables): what was changed (file:line of the stock code you replace), exactness result, microbench before/after at the real shapes, expected end-to-end gain per 2048-token chunk and per decode step, env var, and the exact command to run the tests. Plain prose, no em dashes, no "it's not X, it's Y".
- Model policy: you are an Opus subagent. Do not spawn further agents.

## Round-3 addendum (2026-09-12 13:00)
- The workbench (port 8084) runs A/B measurements during the day. Exactness tests at tiny shapes (< 200 MB GPU) may run at any time. Performance microbenchmarks may run ONLY while the file ~/inference-server/staging/GPU_FREE exists; if it does not exist, finish everything else first, then poll for it (`until [ -f ~/inference-server/staging/GPU_FREE ]; do sleep 30; done`) before benchmarking. Never touch port 8084.
- The oMLX source you may read and copy from is under /Applications/oMLX.app/Contents/Resources/omlx (never edit it). Patches live in ~/inference-server/kernels/round3/<name>/ and hook in via the bootstrap (import-time: env OMLX_ROUND2_IMPORT_PATCHES="path:func"; post-load: OMLX_ROUND2_PATCHES="path[:func]"; a post-load function whose first parameter is positional receives the loaded model).
- Existing deployed patches you must compose with: kernels/ple-fix (packed table, bf16 norms), round2/gdn-norm, round2/wsum10 (prefill only), round2/mtp (shortlist drafter), moe-int8 (gate_up only), and round-3 candidates mtp-head8, mtp-depth (confidence gate wraps _chain_next_drafts), copy-lane (wraps _chain_next_drafts, must be outermost), gdn-scan, gather-ws. Read their REPORT.md files before touching the same functions.
