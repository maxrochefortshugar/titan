#!/usr/bin/env python3
"""Exactness of the fused verify tail against the stock ops tail.

Stock (mlx_vlm/models/qwen3_5_moe/language.py:65, unsorted verify layout):

    y = _target_verify_switch_glu(...)      # [B, T, k, D], token order
    y = (y * scores[..., None]).sum(axis=-2)

Bar in the default ``clone`` mode: bit-identical, 0 differing elements.
Peak allocation under 20 MB, so this may run at any time.
"""
from __future__ import annotations

import mlx.core as mx

import wsum_verify as W

K, D = 10, 2560


def check(b, t, k, d, mode, dtype=mx.bfloat16):
    y = (mx.random.normal((b, t, k, d)) * 0.7).astype(dtype)
    sc = mx.random.uniform(shape=(b, t, k)).astype(dtype)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    got = W.verify_tail(y, sc, mode)
    ref = (y * sc[..., None]).sum(axis=-2)
    mx.eval(got, ref)
    diff = int(mx.sum(got != ref).item())
    g, r = got.astype(mx.float32), ref.astype(mx.float32)
    maxabs = float(mx.max(mx.abs(g - r)).item())
    gold = (y.astype(mx.float32) * sc.astype(mx.float32)[..., None]).sum(axis=-2)
    mx.eval(gold)
    e_kern = float(mx.max(mx.abs(g - gold)).item())
    e_ops = float(mx.max(mx.abs(r - gold)).item())
    # clone: bit-identity with the ops path. fp32 modes: no worse than the ops
    # path against a correctly rounded fp32 golden.
    ok = (diff == 0) if mode == "clone" else (e_kern <= e_ops)
    print(f"mode={mode:<6} B={b} T={t:<3} k={k} D={d:<5} {dtype} "
          f"elems={got.size:<8} differ={diff:<8} vs_ops={maxabs:.3e} "
          f"err_kernel={e_kern:.3e} err_ops={e_ops:.3e} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mx.random.seed(1)
    ok = True
    print("clone mode: bit-identity with the stock ops tail is the bar\n")
    for t in (2, 3, 4, 5, 6, 8, 10, 13, 15):
        ok &= check(1, t, K, D, "clone")
    ok &= check(1, 4, K, D, "clone", mx.float16)
    ok &= check(2, 4, K, D, "clone")
    ok &= check(1, 4, 8, D, "clone")

    print("\nfp32 modes (the ops path is the looser side here)")
    for mode in ("ops", "fast"):
        ok &= check(1, 4, K, D, mode)
        ok &= check(1, 13, K, D, mode)

    print("\nagainst an fp32 golden")
    y = (mx.random.normal((1, 4, K, D)) * 0.7).astype(mx.bfloat16)
    sc = mx.random.uniform(shape=(1, 4, K)).astype(mx.bfloat16)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    gold = (y.astype(mx.float32) * sc.astype(mx.float32)[..., None]).sum(axis=-2)
    ops = (y * sc[..., None]).sum(axis=-2).astype(mx.float32)
    fast = W.verify_tail(y, sc, "fast").astype(mx.float32)
    mx.eval(gold, ops, fast)
    print(f"stock ops vs fp32 golden : max_abs {float(mx.max(mx.abs(ops-gold)).item()):.3e}")
    print(f"kernel fast vs fp32 golden: max_abs {float(mx.max(mx.abs(fast-gold)).item()):.3e}")

    print("\ninstall gate")
    import os
    os.environ["OMLX_WSUM_TOPK10_VERIFY"] = "1"
    print("self_check:", W._self_check())

    print("\nALL PASS" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
