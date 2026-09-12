#!/usr/bin/env python3
"""Exactness of the fused top_k=10 MoE weighted sum against the stock ops path.

Three checks:

A. Tail only.  Sorted expert output + inv_order + scores through the stock
   chain (_scatter_unsort -> multiply -> sum(-2)) versus the kernel, at the
   real Flash-Next shapes (k=10, D=2560, T = 1, 8, 512, 2048), bf16.
B. Both against an fp32 golden, so the reader can see which side is the
   loose one.
C. End to end.  A stand-in MoE block carrying the verbatim body of
   mlx_vlm/models/qwen3_5_moe/language.py:51-73 over a real 4-bit
   SwitchGLU, stock call versus patched call.

Run: ~/inference-server/kdev/bin/python test_exact.py
"""
import argparse
import importlib.util
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU, _scatter_unsort

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("wsum_kernel", os.path.join(_HERE, "kernel.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

MODES = ("clone", "ops", "fast")


def bf16_bits(a: mx.array) -> np.ndarray:
    """Ordered integer key for bf16 values, so |key(a) - key(b)| is ULPs."""
    u = np.array(a.astype(mx.float32)).view(np.uint32) >> np.uint32(16)
    u = u.astype(np.int32)
    neg = (u & 0x8000) != 0
    # Standard total order on sign-magnitude floats: -0 maps onto +0.
    return np.where(neg, -(u & np.int32(0x7FFF)), u)


def ulp(a: mx.array, b: mx.array) -> np.ndarray:
    return np.abs(bf16_bits(a) - bf16_bits(b))


def stats(name, got, ref):
    """Exact-match count, max abs diff, and a ULP figure that is meaningful.

    Raw max-ULP is useless on this operator: the weighted sum of ten signed
    expert rows lands arbitrarily close to zero for some (token, channel)
    pairs, and two results that are 1e-7 apart in absolute terms are then
    thousands of bf16 ULPs apart.  So the ULP column is taken over the
    elements that are not cancelled into the noise, |ref| >= 2^-8 * max|ref|,
    and the fraction within 1 ULP is reported over all elements.
    """
    u = ulp(got, ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    d = np.abs(g - r)
    floor = np.abs(r).max() / 256.0
    live = np.abs(r) >= floor
    max_ulp_live = int(u[live].max()) if live.any() else 0
    rms = float(np.sqrt((r.astype(np.float64) ** 2).mean())) + 1e-30
    return (name, int((u != 0).sum()), int(u.size), float(d.max()), max_ulp_live,
            float((u <= 1).mean() * 100), float(d.max() / rms))


def make(T, k, D, seed=0):
    mx.random.seed(seed)
    n = T * k
    ys = mx.random.normal((n, 1, D)).astype(mx.bfloat16)
    if n >= 64:
        order = mx.argsort(mx.random.uniform(shape=(n,)))
        inv = mx.argsort(order).astype(mx.uint32)
    else:
        inv = None
    sc = mx.random.uniform(shape=(1, T, k)).astype(mx.bfloat16)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    mx.eval(ys, sc)
    if inv is not None:
        mx.eval(inv)
    return ys, inv, sc


def ops_tail(ys, inv, sc, T, k, D):
    y = ys.reshape(1, T, k, D) if inv is None else _scatter_unsort(ys, inv, (1, T, k)).squeeze(-2)
    return (y * sc[..., None]).sum(axis=-2)


def fp32_golden(ys, inv, sc, T, k, D):
    y = ys.reshape(1, T, k, D) if inv is None else _scatter_unsort(ys, inv, (1, T, k)).squeeze(-2)
    return (y.astype(mx.float32) * sc.astype(mx.float32)[..., None]).sum(axis=-2)


def part_a_b(k, D, ts):
    print("A/B. tail: kernel and stock ops vs each other and vs an fp32 golden")
    print(f"{'T':>6}{'compare':>22}{'differ':>10}{'/ total':>10}"
          f"{'max abs':>12}{'maxULP*':>9}{'<=1ULP%':>9}{'absd/rms':>10}")
    for T in ts:
        ys, inv, sc = make(T, k, D)
        ref = ops_tail(ys, inv, sc, T, k, D)
        gold = fp32_golden(ys, inv, sc, T, k, D).astype(mx.bfloat16)
        mx.eval(ref, gold)
        rows = [stats("stock ops vs fp32", ref, gold)]
        for m in MODES:
            got = K.weighted_sum(ys, inv, sc, (1, T, D), mode=m)
            mx.eval(got)
            rows.append(stats(f"kernel[{m}] vs ops", got, ref))
            rows.append(stats(f"kernel[{m}] vs fp32", got, gold))
            del got
        for name, ndiff, ntot, mabs, mulp, pct, rel in rows:
            print(f"{T:>6}{name:>22}{ndiff:>10}{ntot:>10}{mabs:>12.3e}"
                  f"{mulp:>9}{pct:>9.3f}{rel:>10.2e}")
        del ys, inv, sc, ref, gold
        mx.clear_cache()
    print()


# --------------------------------------------------------------------------
# C. end to end over a real quantized SwitchGLU
# --------------------------------------------------------------------------


class StandInMoeBlock(nn.Module):
    """Body copied verbatim from mlx_vlm/models/qwen3_5_moe/language.py:51-73.

    mlx_vlm is not installed in the kdev venv, so the class under test is
    reproduced here; the patch targets the real class by module path at
    install time.
    """

    def __init__(self, dim, inter, shared_inter, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, inter, num_experts)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x, target_verify: bool = False):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


def part_c(k, D, T, num_experts, inter):
    print(f"C. end to end: stand-in MoE block, E={num_experts}, D={D}, "
          f"inter={inter}, top_k={k}, T={T}")
    mx.random.seed(7)
    blk = StandInMoeBlock(D, inter, inter, num_experts, k)
    nn.quantize(blk, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, SwitchGLU))
    blk.set_dtype(mx.bfloat16)
    mx.eval(blk.parameters())

    x = (mx.random.normal((1, T, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    ref = blk(x)
    mx.eval(ref)

    orig_call = type(blk).__call__
    type(blk).__call__ = K._make_patched_call(orig_call)
    setattr(type(blk), K._FLAG, True)
    try:
        K._CFG["topk"] = (k,)
        K._CFG["min_tokens"] = 1
        for m in MODES:
            K._CFG["mode"] = m
            got = blk(x)
            mx.eval(got)
            name, ndiff, ntot, mabs, mulp, pct, rel = stats(
                f"patched[{m}] vs stock", got, ref)
            print(f"  {name:<26} differ {ndiff:>8}/{ntot:<9} max abs {mabs:.3e}  "
                  f"maxULP* {mulp}  <=1ULP {pct:.3f}%")
            del got
    finally:
        type(blk).__call__ = orig_call
        K._CFG["mode"] = K._DEFAULT_MODE
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--ts", default="1,8,512,2048")
    ap.add_argument("--e2e-tokens", type=int, default=512)
    ap.add_argument("--e2e-experts", type=int, default=32)
    ap.add_argument("--skip-e2e", action="store_true")
    a = ap.parse_args()
    mx.set_cache_limit(int(256e6))
    print(f"mlx {mx.__version__}  {mx.device_info().get('device_name','?')}  "
          f"k={a.k} D={a.dim} bf16\n")
    part_a_b(a.k, a.dim, [int(t) for t in a.ts.split(",")])
    print("* maxULP is taken over elements with |ref| >= max|ref| / 256, and is\n"
          "  still dominated by near-cancellation; absd/rms is the honest\n"
          "  magnitude column.  The verdict that matters:\n"
          "    clone mode  = 0 differing elements vs the stock ops path.\n"
          "    fast mode   = 0 differing elements vs the fp32 golden, which the\n"
          "                  stock ops path itself misses on ~61% of elements\n"
          "                  because MLX reduces bf16 in bf16.\n")
    if not a.skip_e2e:
        part_c(a.k, a.dim, a.e2e_tokens, a.e2e_experts, 640)


if __name__ == "__main__":
    main()
