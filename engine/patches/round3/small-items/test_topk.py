#!/usr/bin/env python3
"""Exactness of the fused top-K against mx.topk / mx.argpartition.

Bar: the multiset of returned values is bit-identical to mx.topk's, every
returned index really carries its returned value, indices are distinct, and
the index SET matches mx.argpartition's whenever the K-th largest value is
unique (when it is not, both implementations break the tie arbitrarily and
only the value multiset is defined).

Tiny shapes: peak allocation is a few MB, so this may run at any time.
"""
from __future__ import annotations

import mlx.core as mx

import topk as T

V_REAL = 248320


def _ref_idx(a, k):
    v = int(a.size)
    return mx.argpartition(a, kth=v - k, axis=-1)[..., -k:]


def check(a, k, label):
    a = a.reshape(-1)
    val, idx = T.fast_topk(a, k)
    mx.eval(val, idx)
    ref_v = mx.sort(mx.topk(a, k))
    got_v = mx.sort(val)
    ref_i = _ref_idx(a, k).reshape(-1)
    mx.eval(ref_v, got_v, ref_i)

    same_vals = bool(mx.all(got_v == ref_v).item())
    gathered = a[idx]
    mx.eval(gathered)
    idx_ok = bool(mx.all(gathered == val).item())
    distinct = len(set(idx.tolist())) == k
    kth = float(ref_v[0].item())
    ties = int(mx.sum(a == ref_v[0]).item())
    set_eq = set(idx.tolist()) == set(ref_i.tolist())
    ok = same_vals and idx_ok and distinct and (set_eq or ties > 1)
    print(f"{label:<34} k={k:<5} values={'ok' if same_vals else 'FAIL'} "
          f"idx->val={'ok' if idx_ok else 'FAIL'} distinct={'ok' if distinct else 'FAIL'} "
          f"set_eq={set_eq} kth={kth:+.6g} ties_at_kth={ties} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mx.random.seed(0)
    ok = True

    # 1. the real shape, float32 and bfloat16, over the whole K range
    for dt in (mx.float32, mx.bfloat16):
        a = (mx.random.normal((V_REAL,)) * 6.0).astype(dt)
        mx.eval(a)
        for k in (512, 1024, 2048, 4096):
            ok &= check(a, k, f"real V=248320 {dt}")

    # 2. logit-like: a heavy tail plus a flat bulk (many exact ties in bf16)
    a = (mx.random.normal((V_REAL,)) * 0.5 - 8.0).astype(mx.bfloat16)
    spike = mx.random.randint(0, V_REAL, (5000,))
    a = mx.put_along_axis(a, spike, mx.full((5000,), 4.0, dtype=mx.bfloat16), axis=-1)
    mx.eval(a)
    for k in (512, 2048, 4096):
        ok &= check(a, k, "logit-like bf16 (tie heavy)")

    # 3. degenerate inputs
    ok &= check(mx.zeros((V_REAL,), dtype=mx.float32), 2048, "all zeros")
    ok &= check(mx.full((V_REAL,), -3.0, dtype=mx.bfloat16), 1024, "all equal bf16")
    ok &= check(mx.arange(V_REAL, dtype=mx.float32), 2048, "monotone ascending")
    ok &= check(-mx.arange(V_REAL, dtype=mx.float32), 2048, "monotone descending")
    a = (mx.random.normal((V_REAL,)) * 6.0).astype(mx.float32)
    a = mx.where(mx.random.uniform(shape=(V_REAL,)) < 0.5, -a, a)
    ok &= check(a, 3000, "mixed sign, k not a power of two")

    # 4. small and boundary shapes
    ok &= check(mx.random.normal((4096,)), 512, "V=4096")
    ok &= check(mx.random.normal((5000,)), 5000, "k == V")
    ok &= check(mx.random.normal((1, V_REAL)), 2048, "shape [1, V]")

    # 5. repeated calls on the same input are deterministic in the value set
    a = (mx.random.normal((V_REAL,)) * 6.0).astype(mx.float32)
    mx.eval(a)
    v1, _ = T.fast_topk(a, 2048)
    v2, _ = T.fast_topk(a, 2048)
    same = bool(mx.all(mx.sort(v1) == mx.sort(v2)).item())
    print(f"{'repeat determinism':<34} {'PASS' if same else 'FAIL'}")
    ok &= same

    print("\nALL PASS" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
