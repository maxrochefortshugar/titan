#!/usr/bin/env python3
"""Microbenchmark: MoE unsort + routed weighted sum, top_k=10, hidden 2560.

Stock tail (mlx_lm/models/switch_layers.py:195-198 then
mlx_vlm/models/qwen3_5_moe/language.py:65):
    y = _scatter_unsort(ys, inv_order, indices.shape).squeeze(-2)
    out = (y * scores[..., None]).sum(axis=-2)
against the fused kernel, which reads the sorted expert output once.

Run: ~/inference-server/kdev/bin/python bench.py
"""
import argparse
import importlib.util
import os
import statistics as st
import time

import mlx.core as mx
from mlx_lm.models.switch_layers import _scatter_unsort

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("wsum_kernel", os.path.join(_HERE, "kernel.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

def timeit(fn, iters=15, warm=3, chain=10):
    def run():
        outs = [fn() for _ in range(chain)]
        mx.eval(*outs)

    for _ in range(warm):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / chain)
    return st.median(ts)


def make(T, k, D, sort):
    n = T * k
    ys = mx.random.normal((n, 1, D)).astype(mx.bfloat16)
    if sort:
        order = mx.argsort(mx.random.uniform(shape=(n,)))
        inv = mx.argsort(order).astype(mx.uint32)
    else:
        inv = None
    sc = mx.random.uniform(shape=(1, T, k)).astype(mx.bfloat16)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    mx.eval(ys, sc, inv if inv is not None else mx.array(0))
    return ys, inv, sc


def ops_tail(ys, inv, sc, T, k, D):
    if inv is None:
        y = ys.reshape(1, T, k, D)
    else:
        y = _scatter_unsort(ys, inv, (1, T, k)).squeeze(-2)
    return (y * sc[..., None]).sum(axis=-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--modes", default="clone,fast")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ts", default="1,8,512,2048")
    ap.add_argument(
        "--variant",
        default="all",
        help="all (re-runs each variant in a fresh process), or ops/clone/fast",
    )
    a = ap.parse_args()
    k, D = a.k, a.dim
    if a.variant == "all":
        return _drive(a)
    modes = [] if a.variant == "ops" else [a.variant]

    mx.set_cache_limit(int(256e6))
    print(f"mlx {mx.__version__}  {mx.device_info().get('device_name','?')}  "
          f"k={k} D={D} VEC={K._env_int(K.ENV_VEC, 4)}")
    hdr = f"{'T':>6}{'sorted':>8}{'ops ms':>10}"
    for m in modes:
        hdr += f"{m+' ms':>11}{'x':>7}"
    hdr += f"{'GB/s':>9}{'chain':>6}"
    print(hdr)

    saved = {}
    for T in [int(t) for t in a.ts.split(",")]:
        sort = (T * k) >= 64
        ys, inv, sc = make(T, k, D, sort)
        iters = 15 if T < 512 else 8
        # Keep live transients under ~500 MB: the ops path holds two [T,k,D]
        # bf16 temporaries per call, so long chains at T=2048 thrash the
        # allocator (and the box already has 78 GB of model resident).
        chain = max(1, min(10, int(500e6 / max(1.0, 2.2 * T * k * D * 2))))
        mx.clear_cache()
        # Paired rounds: the GPU power-manages under load, so interleave the
        # variants and keep the minimum of each (see AUDIT section on drift).
        t_ops = None
        t_mode = {m: None for m in modes}
        for _ in range(a.rounds):
            if a.variant in ("ops", "all"):
                v = timeit(lambda: ops_tail(ys, inv, sc, T, k, D), iters, chain=chain)
                t_ops = v if t_ops is None else min(t_ops, v)
            for m in modes:
                v = timeit(lambda m=m: K.weighted_sum(ys, inv, sc, (1, T, D), mode=m), iters, chain=chain)
                t_mode[m] = v if t_mode[m] is None else min(t_mode[m], v)
        row = f"{T:>6}{str(sort):>8}" + (
            f"{t_ops*1e3:>10.4f}" if t_ops else f"{'-':>10}"
        )
        best = None
        for m in modes:
            t_k = t_mode[m]
            row += f"{t_k*1e3:>11.4f}" + (
                f"{t_ops/t_k:>7.2f}" if t_ops else f"{'-':>7}"
            )
            if best is None or t_k < best:
                best = t_k
        by = T * k * D * 2 + T * D * 2
        row += f"{by/(best or t_ops)/1e9:>9.0f}"
        row += f"{chain:>6}"
        print(row)
        saved[T] = (t_ops, best)
        del ys, inv, sc
        mx.clear_cache()

    if a.variant == "ops":
        return
    print()
    for T in sorted(saved):
        t_ops, t_k = saved[T]
        if not t_ops:
            continue
        print(f"T={T}: per layer {(t_ops-t_k)*1e3:.3f} ms saved, "
              f"x{a.layers} layers = {(t_ops-t_k)*a.layers*1e3:.1f} ms per chunk")


def _drive(a):
    """Run each variant in its own process.

    The ops path churns two [T, k, D] bf16 temporaries per call; measuring it
    in the same process as the kernel leaves the allocator in a state that
    inflates whatever runs next by 2-3x.  Fresh processes make both sides
    reproducible.
    """
    import subprocess
    import sys as _sys

    base = [_sys.executable, os.path.abspath(__file__),
            "--k", str(a.k), "--dim", str(a.dim), "--layers", str(a.layers),
            "--rounds", str(a.rounds), "--ts", a.ts]
    for variant in ["ops"] + a.modes.split(","):
        print(f"### variant: {variant}", flush=True)
        subprocess.run(base + ["--variant", variant, "--modes", a.modes], check=True)
        print(flush=True)


if __name__ == "__main__":
    main()
