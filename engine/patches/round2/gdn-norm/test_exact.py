#!/usr/bin/env python3
"""Exactness of the fused GDN norm+gate against the stock MLX-ops path.

Real shapes: [1, T, 48, 128] bf16, eps 1e-6, sigmoid gate (output_gate_type in
the oQ4e config).  Reports max abs error and the bf16 ULP distribution.

Run:  ~/inference-server/kdev/bin/python ~/inference-server/kernels/round2/gdn-norm/test_exact.py
"""
from __future__ import annotations

import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import GATE_SIGMOID, GATE_SILU, norm_gate_fused, norm_gate_stock  # noqa: E402

HV, DV, EPS = 48, 128, 1e-6


def bf16_ulp(a: mx.array, b: mx.array) -> np.ndarray:
    """Distance in bf16 representable steps between two bf16 arrays."""
    av = np.asarray(a.astype(mx.float32)).ravel()
    bv = np.asarray(b.astype(mx.float32)).ravel()
    ai = np.asarray(a.view(mx.uint16)).ravel().astype(np.int32)
    bi = np.asarray(b.view(mx.uint16)).ravel().astype(np.int32)
    # map sign-magnitude to a monotone ordering so adjacent codes differ by 1
    def mono(i):
        return np.where(i & 0x8000, 0x8000 - (i & 0x7FFF), i + 0x8000)
    d = np.abs(mono(ai) - mono(bi)).astype(np.int64)
    # NaN/Inf would make the ordering meaningless
    assert np.isfinite(av).all() and np.isfinite(bv).all()
    return d


def case(name, T, activation, scale=1.0, seed=0, gate_scale=1.0):
    mx.random.seed(seed)
    x = (scale * mx.random.normal((1, T, HV, DV))).astype(mx.bfloat16)
    z = (gate_scale * mx.random.normal((1, T, HV, DV))).astype(mx.bfloat16)
    w = mx.random.normal((DV,)).astype(mx.bfloat16)
    ref = norm_gate_stock(x, z, w, eps=EPS, activation=activation)
    got = norm_gate_fused(x, z, w, eps=EPS, activation=activation)
    mx.eval(ref, got)
    assert got.shape == ref.shape and got.dtype == ref.dtype, (got.shape, got.dtype)
    ulp = bf16_ulp(got, ref)
    rv = np.asarray(ref.astype(mx.float32))
    gv = np.asarray(got.astype(mx.float32))
    absr = np.abs(gv - rv)
    denom = np.maximum(np.abs(rv), 1e-30)
    row = {
        "case": name,
        "T": T,
        "activation": "sigmoid" if activation == GATE_SIGMOID else "silu",
        "elems": int(ulp.size),
        "max_abs": float(absr.max()),
        "max_rel": float((absr / denom).max()),
        "ulp_max": int(ulp.max()),
        "ulp_mean": float(ulp.mean()),
        "exact_frac": float((ulp == 0).mean()),
        "gt1ulp": int((ulp > 1).sum()),
    }
    print(
        f"{name:<34} T={T:<5} elems={row['elems']:>9}  max_abs={row['max_abs']:.3e}"
        f"  ulp_max={row['ulp_max']}  exact={100*row['exact_frac']:.4f}%"
        f"  >1ulp={row['gt1ulp']}"
    )
    return row


def main():
    print(f"mlx {mx.__version__}  {mx.device_info().get('device_name','?')}")
    print("stock = mx.fast.rms_norm -> fp32 -> * fp32 gate -> bf16  (LANG:1080-1088)\n")
    rows = []
    for T in (1, 2, 7, 512, 2048):
        rows.append(case("sigmoid gate, unit normal", T, GATE_SIGMOID, seed=T))
    # a T=1 4-D call is refused by eligible() in production but the kernel must
    # still be correct there, which the T=1 row above checks.
    rows.append(case("sigmoid, small x (eps regime)", 2048, GATE_SIGMOID, scale=1e-3, seed=11))
    rows.append(case("sigmoid, large x", 2048, GATE_SIGMOID, scale=64.0, seed=12))
    rows.append(case("sigmoid, saturating gate", 2048, GATE_SIGMOID, seed=13, gate_scale=40.0))
    rows.append(case("silu gate, unit normal", 2048, GATE_SILU, seed=14))
    rows.append(case("silu, saturating gate", 2048, GATE_SILU, seed=15, gate_scale=40.0))

    # near-zero rows: every element of a head is 0 -> normalizer is rsqrt(eps)
    mx.random.seed(99)
    x = mx.zeros((1, 64, HV, DV), dtype=mx.bfloat16)
    z = mx.random.normal((1, 64, HV, DV)).astype(mx.bfloat16)
    w = mx.random.normal((DV,)).astype(mx.bfloat16)
    ref = norm_gate_stock(x, z, w, eps=EPS, activation=GATE_SIGMOID)
    got = norm_gate_fused(x, z, w, eps=EPS, activation=GATE_SIGMOID)
    mx.eval(ref, got)
    zeros_ok = bool(mx.array_equal(ref, got).item())
    print(f"{'all-zero rows identical':<34} {zeros_ok}")

    worst = max(r["ulp_max"] for r in rows)
    total_gt1 = sum(r["gt1ulp"] for r in rows)
    print(f"\nworst ULP over all cases: {worst}   elements above 1 ULP: {total_gt1}")
    ok = worst <= 1 and total_gt1 == 0 and zeros_ok
    print("RESULT:", "PASS (<= 1 ULP bf16)" if ok else "FAIL")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_exact.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "zeros_identical": zeros_ok, "pass": ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
