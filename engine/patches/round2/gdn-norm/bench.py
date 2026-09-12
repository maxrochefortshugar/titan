#!/usr/bin/env python3
"""Microbench: fused GDN norm+gate vs the stock MLX-ops path, [1,T,48,128] bf16.

Median of 15, warm, CHAIN=10 ops per eval, mx.synchronize around each timing,
as in ~/inference-server/kbench.py.

Run:  ~/inference-server/kdev/bin/python ~/inference-server/kernels/round2/gdn-norm/bench.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import GATE_SIGMOID, norm_gate_fused, norm_gate_stock  # noqa: E402

HV, DV, EPS = 48, 128, 1e-6
GDN_LAYERS = 36           # linear-attention layers per forward (48 total, 1 in 4 is QSA)
CHAIN = 10
PEAK_READ = 718e9         # measured read ceiling on this machine (AUDIT correction 1)


def timeit(fn, iters=15, warm=3):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)

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


def row(T):
    mx.random.seed(T)
    x = mx.random.normal((1, T, HV, DV)).astype(mx.bfloat16)
    z = mx.random.normal((1, T, HV, DV)).astype(mx.bfloat16)
    w = mx.random.normal((DV,)).astype(mx.bfloat16)
    mx.eval(x, z, w)

    t_stock = timeit(lambda: norm_gate_stock(x, z, w, eps=EPS, activation=GATE_SIGMOID))
    t_fused = timeit(lambda: norm_gate_fused(x, z, w, eps=EPS, activation=GATE_SIGMOID))

    payload = 3 * T * HV * DV * 2                 # x + z read, out written
    # stock traffic: x, bf16 norm out, fp32 cast, z, fp32 cast, fp32 product, bf16 out
    stock_traffic = T * HV * DV * (2 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 2)
    r = {
        "T": T,
        "stock_ms": t_stock * 1e3,
        "fused_ms": t_fused * 1e3,
        "speedup": t_stock / t_fused,
        "fused_GBs": payload / t_fused / 1e9,
        "fused_pct_ceiling": 100 * payload / t_fused / PEAK_READ,
        "stock_GBs_payload": payload / t_stock / 1e9,
        "payload_MB": payload / 1e6,
        "stock_traffic_MB": stock_traffic / 1e6,
        "saved_ms_per_chunk": (t_stock - t_fused) * 1e3 * GDN_LAYERS,
    }
    print(
        f"T={T:<6} stock {r['stock_ms']:>7.3f} ms   fused {r['fused_ms']:>7.3f} ms"
        f"   {r['speedup']:>5.2f}x   fused {r['fused_GBs']:>4.0f} GB/s"
        f" ({r['fused_pct_ceiling']:.0f}% of 718)"
        f"   x{GDN_LAYERS} layers: {r['saved_ms_per_chunk']:>7.2f} ms saved"
    )
    return r


def main():
    print(f"mlx {mx.__version__}  {mx.device_info().get('device_name','?')}")
    print("Qwen4-Exp GDN gated norm, x/gate [1,T,48,128] bf16, eps=1e-6, sigmoid gate")
    print("payload = 75 MB at T=2048; the stock op chain moves ~352 MB of it\n")
    rows = [row(T) for T in (1, 2, 512, 2048)]
    print()
    for r in rows:
        print(
            f"  T={r['T']:<6} payload {r['payload_MB']:>7.2f} MB   "
            f"stock traffic {r['stock_traffic_MB']:>7.2f} MB"
        )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.json")
    with open(out, "w") as f:
        json.dump({"mlx": mx.__version__, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
