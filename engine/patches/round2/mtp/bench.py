#!/usr/bin/env python3
"""Small-M verify microbench for Qwen3.8-Flash-Next (M = 1 + MTP depth).

Baselines
  stock      mx.quantized_matmul (what mlx dispatches today)
  vk_qmm     oMLX omlx/patches/qwen35_verify_qmm.py (live in production for
             N >= 16384, i.e. lm_head only)
Candidate
  v070       mlx-vlm 0.7.0 mlx_vlm/models/qwen3_5/speculative_verifier.py
             _target_verify_quantized_linear / _target_verify_quantized_argmax

Median of 15, warm, CHAIN=10 ops per eval, mx.synchronize around each sample.
"""
import argparse, gc, importlib.util, statistics as st, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vloader

CHAIN = 10
FORCE_VK = False
OMLX = Path("/Applications/oMLX.app/Contents/Resources/omlx/patches/qwen35_verify_qmm.py")

# (name, K, N) for the dense projections that a verify forward runs
SHAPES = [
    ("q_proj      2560->6144", 2560, 6144),
    ("o_proj      6144->2560", 6144, 2560),
    ("shared_up   2560->640 ", 2560, 640),
    ("lm_head     2560->248320", 2560, 248320),
]


def load_vk():
    spec = importlib.util.spec_from_file_location("_omlx_vk", OMLX)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_omlx_vk"] = m
    spec.loader.exec_module(m)
    return m


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


def qbytes(n, bits, gs):
    return n * bits / 8 + (n / gs) * 2 * 2


def make_qlinear(K, N, bits, gs, exact=False):
    """Quantized linear at the real shape.

    For the 248320-row head a bf16 staging tensor is 1.3 GB, so unless an
    exact reference is needed the packed buffers are built directly with
    random bits: timing depends on shape and dtype, not on the values.
    """
    ql = nn.QuantizedLinear(K, N, bias=False, group_size=gs, bits=bits)
    if exact or N * K <= 64 * 1024 * 1024:
        lin = nn.Linear(K, N, bias=False)
        lin.weight = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
        ql = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)
        del lin
    else:
        ql.weight = mx.random.randint(
            0, 2**31 - 1, (N, K * bits // 32), dtype=mx.uint32)
        ql.scales = (mx.random.normal((N, K // gs)) * 0.01).astype(mx.bfloat16)
        ql.biases = (mx.random.normal((N, K // gs)) * 0.01).astype(mx.bfloat16)
    mx.eval(ql.parameters())
    return ql


def bench_shape(name, K, N, bits, gs, ms, vk, v070, iters):
    ql = make_qlinear(K, N, bits, gs)
    ok_head = v070._can_target_verify_quantized_head(ql)
    rows = []
    for M in ms:
        x3 = (mx.random.normal((1, M, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(x3)
        x2 = x3[0]

        t_stock = timeit(lambda: mx.quantized_matmul(
            x2, ql.weight, ql.scales, ql.biases,
            transpose=True, group_size=gs, bits=bits), iters)

        t_vk = None
        if vk is not None and (FORCE_VK or vk.vk_eligible(M, K, N, bits, gs, mx.bfloat16)) and 3 <= M <= 6:
            try:
                t_vk = timeit(lambda: vk.vk_qmm(
                    x2, ql.weight, ql.scales, ql.biases, bits=bits,
                    group_size=gs), iters)
            except Exception as exc:
                t_vk = None
                print(f"    vk_qmm failed at M={M}: {exc}")

        t_v = t_va = None
        if ok_head and v070._can_target_verify_quantized(ql, x3):
            t_v = timeit(lambda: v070._target_verify_quantized_linear(ql, x3), iters)
            t_va = timeit(lambda: v070._target_verify_quantized_argmax(ql, x3), iters)

        # stock greedy needs matmul + argmax; that is the real comparand for argmax
        t_stock_am = timeit(lambda: mx.argmax(mx.quantized_matmul(
            x2, ql.weight, ql.scales, ql.biases,
            transpose=True, group_size=gs, bits=bits), axis=-1), iters)

        by = qbytes(N * K, bits, gs) + (M * K + M * N) * 2
        rows.append((M, t_stock, t_vk, t_v, t_stock_am, t_va, by))
    del ql
    gc.collect()
    mx.clear_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--ms", type=str, default="1,2,4,6,8")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--force-vk", action="store_true")
    a = ap.parse_args()
    global FORCE_VK
    FORCE_VK = a.force_vk
    ms = [int(v) for v in a.ms.split(",")]

    v070 = _vloader.load()
    try:
        vk = load_vk()
    except Exception as exc:
        print("vk_qmm unavailable:", exc)
        vk = None

    print(f"mlx {mx.__version__}  bits={a.bits} gs={a.gs}  "
          f"{mx.device_info().get('device_name','?')}")
    hdr = (f"{'M':>3} {'stock ms':>9} {'vk ms':>9} {'v070 ms':>9} "
           f"{'v070/stock':>10} | {'stock+am':>9} {'v070 amx':>9} {'amx gain':>9} "
           f"{'GB/s stock':>10} {'GB/s v070':>10}")
    for name, K, N in SHAPES:
        if a.only and a.only not in name:
            continue
        print(f"\n### {name}   K={K} N={N} {a.bits}-bit gs{a.gs}")
        print(hdr)
        for (M, ts, tv, t7, tsa, t7a, by) in bench_shape(
                name, K, N, a.bits, a.gs, ms, vk, v070, a.iters):
            f = lambda t: f"{t*1e3:9.3f}" if t else "        -"
            r1 = f"{t7/ts:10.2f}" if t7 else "         -"
            r2 = f"{tsa/t7a:9.2f}" if t7a else "        -"
            g1 = f"{by/ts/1e9:10.0f}"
            g2 = f"{by/t7/1e9:10.0f}" if t7 else "         -"
            print(f"{M:>3} {f(ts)} {f(tv)} {f(t7)} {r1} | {f(tsa)} {f(t7a)} {r2} {g1} {g2}")


if __name__ == "__main__":
    main()
