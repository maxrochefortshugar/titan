# Titan

An inference stack for agentic coding on Apple Silicon, built to reach and hold state-of-the-art throughput on one machine. Today it runs Qwen3.8-Flash-Next (125B-A6B mixture of experts) on a MacBook Pro M5 Max with 128 GB, serving opencode over the tailnet.

Titan uses MLX as its substrate (arrays, quantised matmuls, Metal command scheduling, the M5 tensor-op kernels) and oMLX as the serving layer, and replaces the parts that are slow with its own Metal kernels, loaded as a runtime overlay. Nothing in the upstream projects is modified; every replacement is env-gated, keeps the stock path on failure, and ships with an exactness test (bit-identical or one bf16 ULP against the stock computation) and a benchmark.

## Where it stands (2026-09-12)

| measure | stock oMLX, day one | Titan |
|---|---|---|
| cold 65k-token prefill | 1035 tok/s | 1642 tok/s |
| decode after a long prefill | 47.7 tok/s | 71 tok/s |
| short-context decode | ~48 tok/s | 84 to 91 tok/s |
| 49k cached follow-up turn | 4.1 s | 1.4 s |
| 8 concurrent short streams | serialised | 130 tok/s aggregate |

Prefill is ahead of every published number for this model on any Mac. Decode sits at about 40% of the memory-bandwidth roofline and is the current focus; see `docs/plan/IMPROVEMENTS.md` for the full record, including what was tried and did not work.

## Layout

- `engine/bootstrap/`: launches oMLX's CLI in-process and installs the overlay before and after model load.
- `engine/patches/`: the kernels and patches, one directory per workstream, each with `patch.py` exposing `install()`, an exactness test, a microbenchmark and a `REPORT.md`.
- `bench/`: the measurement harness (cold prefill, decode probe, multi-turn cache, concurrency sweep, agentic suite, MTP acceptance summary, the isolated workbench launcher and its memory watchdog).
- `prod/`: the production wrapper with the patch toggles, deploy and verify scripts, machine hardening.
- `clients/`: opencode setup for a client machine (provider, thinking variants, search via SearXNG).
- `docs/`: plan, kernel audit, research notes, and the improvements record.

## Deployed overlay

Packed contiguous n-gram table streamed from SSD (removes 3x page-read amplification), bf16 grouped norms, fused Gated DeltaNet norm and gate, fused MoE weighted sum for top_k=10 at prefill and on the MTP verify layout, int8 x int4 expert gather for the gate_up projections, MTP shortlist drafter, fused prefill hyper-connection block, chunked GDN prefill scan (from mlx PR #4020), sparse attention for batched decode. Plus operational findings that mattered as much as the kernels: the memory guard that serialised concurrent requests, the sampling defaults, and thermal throttling worth 16% (keep a fan floor).

## Measured and rejected

8-bit MTP draft block, confidence-gated draft depth, n-gram copy lane, MTP park policy, batched MTP verify, NAX GDN kernel (exact after a hi/lo split, but register-bound), int8 down projections, LRU n-gram row cache, mlx-vlm 0.7.0 swap, Neural Engine offload, fine cache boundary (first two attempts). Details and numbers in `docs/kernels/REPORT.md`.

## Requirements

macOS 26.5 or later on Apple Silicon with at least 128 GB, oMLX 0.7.0.dev2 (bundles mlx 0.32.2), the Flash-Next oQ4e-mtp checkpoint, and a packed n-gram table built with `engine/patches/ple-fix/repack.py`. Secrets and host names are not in this repository: the server API key is read from `~/.omlx/api_key.txt`, the tailnet host name is a placeholder in the client scripts.

## Licence

MIT.
