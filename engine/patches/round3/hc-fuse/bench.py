#!/usr/bin/env python3
"""Per-layer microbenchmark of the hyper-connection block.

Four paths at T = 512 and T = 2048:

    canonical   Qwen4ExpGatedResidual._forward (fp32 grouped norm)
    shipped     hc_fused.prefill_forward
    deployed    shipped + kernels/ple-fix/norm_patch.py (production today)
    hc-fuse2    this workstream (optionally --concat)

Warm, median of 15, CHAIN=10 per eval as in ~/inference-server/kbench.py.
Only run this while ~/inference-server/staging/GPU_FREE exists.

    ~/inference-server/kdev/bin/python bench.py [--iters 15] [--concat]
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics as st
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel as hc2  # noqa: E402
from synth import GatedResidual, load_hc_fused, make_input  # noqa: E402

CHAIN = 10
READ_CEILING = 718.0   # GB/s, audit A
COPY_CEILING = 549.0   # GB/s, COMMON.md


def timeit(fn, iters, warm=2):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        flat = []
        for o in outs:
            flat.extend(o if isinstance(o, tuple) else (o,))
        mx.eval(*flat)

    for _ in range(warm):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)


def deployed_prefill(hc_fused):
    path = Path.home() / "inference-server/kernels/ple-fix/norm_patch.py"
    spec = importlib.util.spec_from_file_location("ple_norm_patch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_prefill_forward(hc_fused)


def stream_bytes(rows, width=10240, hidden=2560, lowrank=320, hc=4,
                 weights_mb=3.4):
    """Activation traffic per call: six passes over the residual stream.

    Both paths move the same six: norm reads x and writes normed, the two
    input projections read normed, the up projection writes its result, and
    the tail reads that result and normed again.  Weights add ~3.4 MB per
    call at BM = 64 row tiles.  The low-rank tensors are noise beside them.
    """
    s = rows * width * 2
    return 6 * s + rows * hidden * 2 + rows * (lowrank + hc) * 4 + weights_mb * 1e6


def stage_bench(a, hc_fused):
    """Where the per-call time actually goes, stage by stage."""
    import mlx.nn as nn

    for rows in a.rows:
        mod = GatedResidual(bits=a.bits, seed=0)
        x = make_input(rows, seed=7)
        hc, hidden, lowrank = mod.hc_count, mod.hidden_size, mod.hc_lowrank
        width = hc * hidden
        flat = x.reshape(rows, width)
        normed = hc_fused._kernel_norm(mod, flat, rows, hc, hidden, mx.bfloat16)
        raw_d = mod.input_mix_weight_down(normed)
        raw_j = mod.block_inject_weight(normed)
        act, inj = hc2._epilogue(raw_d, raw_j, rows, lowrank, hc, mx.bfloat16)
        up_out = mod.input_mix_weight_up(act)
        mx.eval(normed, raw_d, raw_j, act, inj, up_out)
        stages = [
            ("canonical norm (fp32)", lambda: mod.hc_norm(x)),
            ("bf16 norm kernel", lambda: hc_fused._kernel_norm(
                mod, flat, rows, hc, hidden, mx.bfloat16)),
            ("down qmm", lambda: mod.input_mix_weight_down(normed)),
            ("inject qmm", lambda: mod.block_inject_weight(normed)),
            ("stock silu chain", lambda: nn.silu(raw_d / hc)),
            ("stock inject chain", lambda: 2 * mx.sigmoid(raw_j / hc)),
            ("hc2 epilogue", lambda: hc2._epilogue(
                raw_d, raw_j, rows, lowrank, hc, mx.bfloat16)),
            ("up qmm", lambda: mod.input_mix_weight_up(act)),
            ("stock compiled tail", lambda: hc_fused._tail(hc, hidden)(
                up_out, normed)),
            ("hc2 tail kernel", lambda: hc2._tail(
                up_out, normed, rows, hc, hidden, mx.bfloat16)),
        ]
        print(f"\nstages at T={rows}")
        for name, fn in stages:
            print(f"  {name:24s} {timeit(fn, a.iters)*1e3:7.4f} ms")
        del mod, x, flat, normed, raw_d, raw_j, act, inj, up_out
        gc.collect()
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--rows", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--bits", type=int, default=5)
    ap.add_argument("--concat", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--stages", action="store_true",
                    help="also time each stage of the block on its own")
    a = ap.parse_args()

    flag = Path.home() / "inference-server/staging/GPU_FREE"
    if not flag.exists():
        print("GPU_FREE is absent: refusing to benchmark.")
        return 2

    hc_fused = load_hc_fused()
    dep = deployed_prefill(hc_fused)
    results = []
    for rows in a.rows:
        mod = GatedResidual(bits=a.bits, seed=0)
        x = make_input(rows, seed=7)
        mx.eval(x)
        paths = [
            ("canonical", lambda: mod._forward(x)),
            ("shipped", lambda: hc_fused.prefill_forward(mod, x)),
            ("deployed", lambda: dep(mod, x)),
            ("hc-fuse2", lambda: hc2._fused_prefill_forward(
                hc_fused, mod, x, use_concat=False)),
        ]
        if a.concat:
            paths.append(("hc-fuse2+concat", lambda: hc2._fused_prefill_forward(
                hc_fused, mod, x, use_concat=True)))
        for name, fn in paths:
            t = timeit(fn, a.iters)
            by = stream_bytes(rows)
            results.append({
                "rows": rows, "path": name, "ms": t * 1e3,
                "gbps": by / t / 1e9,
                "per_chunk_ms_96": t * 1e3 * 96,
            })
            print(f"T={rows:5d} {name:16s} {t*1e3:7.3f} ms  "
                  f"{by/t/1e9:6.0f} GB/s  x96 = {t*1e3*96:7.1f} ms/chunk")
        del mod, x
        gc.collect()
        mx.clear_cache()

    if a.stages:
        stage_bench(a, hc_fused)

    print(f"\nceilings: read {READ_CEILING} GB/s, copy {COPY_CEILING} GB/s")
    for rows in a.rows:
        base = next(r for r in results if r["rows"] == rows and r["path"] == "deployed")
        for r in results:
            if r["rows"] != rows or r["path"] == "deployed":
                continue
            d = (base["ms"] - r["ms"]) * 96
            print(f"T={rows:5d} {r['path']:16s} {base['ms']/r['ms']:5.3f}x vs deployed, "
                  f"{d:+7.1f} ms per 2048-token chunk over 96 blocks")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
