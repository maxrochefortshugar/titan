#!/usr/bin/env python
"""Before/after timing and launch count for the fused MoE decode path.

Shapes are the Qwen3.8-Flash-Next routed MLP: hidden 2560, 512 experts, top-10,
gate_up [512, 1280, 2560] and down [512, 2560, 640], affine 4-bit gs64.
Timing is min-of-N (the GPU is shared with a live daemon; the median is
contaminated by contention, the minimum is the clean kernel time).
"""
import argparse
import os
import time

import mlx.core as mx

import fused_kernels as fk
import graphcount
import kernel as K
from baseline import timeit, timeit_pair
from moe_ref import HIDDEN, SparseMoeBlock


def gpu_is_quiet(target=530.0, tries=20):
    """Wait until the shared GPU looks idle (copy bandwidth near the ceiling)."""
    big = mx.zeros((256, 1024, 1024), dtype=mx.bfloat16)
    mx.eval(big)
    bw = 0.0
    for i in range(tries):
        t = timeit(lambda: big + 1, 10)
        bw = 2 * big.nbytes / t / 1e9
        if bw >= target:
            break
        time.sleep(3.0)
    del big
    mx.clear_cache()
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--tokens", type=int, nargs="*", default=[1, 2, 8])
    a = ap.parse_args()

    bw = gpu_is_quiet()
    print(f"copy ceiling now: {bw:.0f} GB/s "
          f"({'quiet' if bw > 530 else 'CONTENDED - numbers unreliable'})")

    K.apply()
    blk = SparseMoeBlock(E=a.experts)
    mx.eval(blk.parameters())
    gu, dn = blk.switch_mlp.gate_up_proj, blk.switch_mlp.down_proj
    print(f"E={a.experts} gate_up {tuple(gu['weight'].shape)} "
          f"down {tuple(dn['weight'].shape)}  weights "
          f"{(gu['weight'].nbytes + dn['weight'].nbytes) / 1e9:.2f} GB\n")

    hdr = (f"{'T':>3s} {'stock us':>9s} {'fused us':>9s} {'speedup':>8s} "
           f"{'stock prims':>12s} {'fused prims':>12s}   breakdown (us)")
    print(hdr)
    print("-" * len(hdr))
    for T in a.tokens:
        x = mx.random.normal((1, T, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)

        def stock_call():
            os.environ[K._ENV] = "0"
            return blk(x)

        def fused_call():
            os.environ[K._ENV] = "1"
            return blk(x)

        os.environ[K._ENV] = "0"
        mx.eval(blk(x))
        n_stock, c_stock = graphcount.count(blk(x))
        os.environ[K._ENV] = "1"
        mx.eval(blk(x))
        n_fused, c_fused = graphcount.count(blk(x))
        os.environ[K._ENV_ROUTER] = "1"
        n_fr, _ = graphcount.count(blk(x))
        os.environ[K._ENV_ROUTER] = "0"
        t_stock, t_fused = timeit_pair([stock_call, fused_call])

        # sub-step breakdown of the fused path
        inds, scores = blk.route(x)
        mx.eval(inds, scores)
        idx = inds.reshape(T, 10).astype(mx.uint32)
        sc = scores.reshape(T, 10).astype(mx.float32)
        mx.eval(idx, sc)
        xf = x.reshape(T, HIDDEN)
        f1 = lambda: fk.gate_up_silu(xf, gu["weight"], gu["scales"],
                                     gu["biases"], idx, 64)
        h = f1()
        mx.eval(h)
        f2 = lambda: fk.down_wsum(h, dn["weight"], dn["scales"], dn["biases"],
                                  idx, sc, 64, out_dtype=mx.bfloat16)
        mx.eval(f2())
        gq = blk.gate
        f3 = lambda: fk.router_topk(fk.router_logits(
            xf, gq["weight"], gq["scales"], gq["biases"], 64), 10, True)
        mx.eval(*f3())
        y = blk.switch_mlp(x, inds)
        mx.eval(y)
        t_route, t_gu, t_dn, t_frouter, t_smlp, t_ws = timeit_pair([
            lambda: blk.route(x), f1, f2, lambda: f3()[0],
            lambda: blk.switch_mlp(x, inds),
            lambda: (y * scores[..., None].astype(y.dtype)).sum(axis=-2)])

        # whole block with the fused router as well
        def fused_router_call():
            os.environ[K._ENV] = "1"; os.environ[K._ENV_ROUTER] = "1"
            r = blk(x)
            os.environ[K._ENV_ROUTER] = "0"
            return r
        os.environ[K._ENV] = "1"; os.environ[K._ENV_ROUTER] = "1"
        mx.eval(blk(x)); os.environ[K._ENV_ROUTER] = "0"
        (t_fr,) = timeit_pair([fused_router_call])

        tag = "" if T <= K._max_tokens() else "  (fallback)"
        print(f"{T:3d} {t_stock*1e6:9.1f} {t_fused*1e6:9.1f} "
              f"{t_stock/t_fused:7.2f}x {n_stock:12d} {n_fused:12d}{tag}")
        print(f"    +fused router: {t_fr*1e6:.1f} us "
              f"({t_stock/t_fr:.2f}x, {n_fr} prims)")
        print(f"    sub-steps us: stock[ route {t_route*1e6:.1f} | switch_mlp "
              f"{t_smlp*1e6:.1f} | wsum {t_ws*1e6:.1f} ]  fused[ router "
              f"{t_frouter*1e6:.1f} | gate_up_silu {t_gu*1e6:.1f} | down_wsum "
              f"{t_dn*1e6:.1f} ]")

        if T == a.tokens[0]:
            print(f"\n    stock primitives: {dict(c_stock)}")
            print(f"    fused primitives: {dict(c_fused)}\n")

    # Where the fused kernels stop paying: they re-read the expert weights for
    # every routed (token, expert) pair, so past the decode regime the sorted
    # gather_qmm wins.  This is the data behind the T<=2 default threshold.
    print("\ncrossover: expert kernels only (switch_mlp + weighted sum)")
    print(f"{'T':>4s} {'stock us':>10s} {'fused us':>10s} {'ratio':>7s}")
    for T in (1, 2, 4, 8, 16, 32, 64):
        x = mx.random.normal((1, T, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        inds, scores = blk.route(x)
        mx.eval(inds, scores)
        idx = inds.reshape(T, 10).astype(mx.uint32)
        sc = scores.reshape(T, 10).astype(mx.float32)
        xf = x.reshape(T, HIDDEN)
        mx.eval(idx, sc, xf)

        def stock():
            y = blk.switch_mlp(x, inds)
            return (y * scores[..., None].astype(y.dtype)).sum(axis=-2)

        def fused():
            h = fk.gate_up_silu(xf, gu["weight"], gu["scales"], gu["biases"],
                                idx, 64)
            return fk.down_wsum(h, dn["weight"], dn["scales"], dn["biases"],
                                idx, sc, 64, out_dtype=mx.bfloat16)

        ts, tf = timeit_pair([stock, fused], iters=8)
        print(f"{T:4d} {ts*1e6:10.1f} {tf*1e6:10.1f} {ts/tf:6.2f}x")

    os.environ[K._ENV] = "0"
    os.environ[K._ENV_ROUTER] = "0"


if __name__ == "__main__":
    main()
