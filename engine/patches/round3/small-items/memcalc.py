#!/usr/bin/env python3
"""How much resident memory the int8 down-projection tables actually cost.

Item 4 of the round-3 small-items brief.  No benchmark and no model load: the
table sizes come from ``kernels/moe-int8/kernel.py:prepare_weights`` itself,
run at a cut-down expert count and scaled, so the number is the kernel's own
accounting rather than a paper estimate.

    ~/inference-server/kdev/bin/python memcalc.py
"""
from __future__ import annotations

import importlib.util
import sys

import mlx.core as mx

MOE_INT8 = "~/inference-server/kernels/moe-int8/kernel.py"

E, LAYERS, GROUP, BITS = 512, 48, 64, 4
# Flash-Next routed experts: up/gate [512, 640, 2560] fused to [512, 1280, 2560],
# down [512, 2560, 640]  (COMMON.md).
TENSORS = {"gate_up (fused)": (1280, 2560), "down": (2560, 640)}

# Measured state, 2026-09-12 (kernels/REPORT.md "concurrency finding").
IN_USE_GB = 85.7          # model + gate_up tables + 4 GB hot cache, short prompts
GUARD_GB = 110.0
SOFT_GB = 93.5
HARD_GB = 104.5
HOT_CACHE_GB = 4.0        # already inside IN_USE_GB (inference-server-plan.md:67)
KV_PER_100K_GB = 2.5      # inference-server-plan.md:23
KV_65K_GB = KV_PER_100K_GB * 0.65


def load_kernel():
    spec = importlib.util.spec_from_file_location("moe_int8_kernel", MOE_INT8)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_int8_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod


def measured_table_bytes(k, n, kdim, e_small=4):
    """prepare_weights() output size at e_small experts, scaled to E."""
    kw = kdim * BITS // 32
    g = kdim // GROUP
    wq = mx.random.randint(0, 2 ** 31 - 1, (e_small, n, kw), dtype=mx.uint32)
    sc = (mx.random.normal((e_small, n, g)) * 0.01).astype(mx.bfloat16)
    bi = (mx.random.normal((e_small, n, g)) * 0.01).astype(mx.bfloat16)
    mx.eval(wq, sc, bi)
    tabs = k.prepare_weights(wq, sc, bi)
    mx.eval(*tabs)
    per_expert = sum(t.nbytes for t in tabs) / e_small
    shapes = [tuple(t.shape) for t in tabs]
    dtypes = [str(t.dtype).split(".")[-1] for t in tabs]
    del wq, sc, bi, tabs
    mx.clear_cache()
    return per_expert * E, shapes, dtypes


def main():
    k = load_kernel()
    print("Per-tensor int8 tables, measured through "
          "moe-int8/kernel.py:prepare_weights\n")
    print(f"{'tensor':<18} {'N':>6} {'K':>6} {'G':>4} {'tables':>28} "
          f"{'MB/tensor':>10} {'GB x48':>8}")
    totals = {}
    for name, (n, kdim) in TENSORS.items():
        by, shapes, dtypes = measured_table_bytes(k, n, kdim)
        totals[name] = by * LAYERS
        desc = ", ".join(f"{d}{list(s[1:])}" for s, d in zip(shapes, dtypes))
        print(f"{name:<18} {n:>6} {kdim:>6} {kdim//GROUP:>4} {desc:>28} "
              f"{by/1e6:>10.1f} {by*LAYERS/1e9:>8.3f}")

    down = totals["down"] / 1e9
    gate_up = totals["gate_up (fused)"] / 1e9
    print(f"\ngate_up only (deployed today): {gate_up:.2f} GB")
    print(f"down projections add:          {down:.2f} GB "
          f"({down * 1e9 / 2**30:.2f} GiB)")
    print(f"both:                          {gate_up + down:.2f} GB")

    print("\nBudget, memory guard 110 GB (soft 93.5, hard 104.5)")
    rows = [
        ("in use today (model + gate_up tables + 4 GB hot cache)", IN_USE_GB),
        ("+ down tables", down),
        ("+ 65k KV cache", KV_65K_GB),
    ]
    total = 0.0
    for label, v in rows:
        total += v
        print(f"  {label:<54} {v:>7.2f} -> {total:>7.2f} GB")
    print(f"  {'soft limit':<54} {SOFT_GB:>7.2f} GB   "
          f"headroom {SOFT_GB - total:+.2f} GB")
    print(f"  {'hard limit':<54} {HARD_GB:>7.2f} GB   "
          f"headroom {HARD_GB - total:+.2f} GB")
    for streams in (2, 4):
        extra = KV_65K_GB * (streams - 1)
        print(f"  {streams} concurrent 65k streams: {total + extra:.2f} GB, "
              f"soft headroom {SOFT_GB - total - extra:+.2f} GB")

    print("\nwarmup() skip condition, moe-int8/patch.py:191")
    for name, (n, kdim) in TENSORS.items():
        kw = kdim * BITS // 32
        as_coded = kw * 8 // BITS
        correct = kw * 32 // BITS
        print(f"  {name:<18} weight.shape[-1]={kw:>4}  "
              f"coded 'shape[-1]*8//bits'={as_coded:>5} (<=1024: "
              f"{as_coded <= 1024})  actual K={correct}")


if __name__ == "__main__":
    main()
