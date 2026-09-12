#!/usr/bin/env python3
"""Exactness of the fused hyper-connection block against the stock path.

Two references, because production does not run the canonical arithmetic:

  canonical  Qwen4ExpGatedResidual._forward, fp32 grouped norm (language.py
             :1684-1757).  hc_fused.prefill_forward is bit identical to it.
  deployed   the same block with kernels/ple-fix/norm_patch.py applied, which
             is what runs on 8083 today under OMLX_QWEN4_BF16_NORM=1.

The 1 ULP bar is against the deployed path, which is the path this patch
replaces; the canonical column shows what the bf16 norm already costs.
Errors are bf16 bit-pattern distances, so they stay meaningful near zero.

    ~/inference-server/kdev/bin/python test_exact.py [--rows 2048]
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel as hc2  # noqa: E402
from synth import GatedResidual, load_hc_fused, make_input, ulp_stats  # noqa: E402


def deployed_prefill(hc_fused):
    """The bf16-norm prefill_forward exactly as ple-fix deploys it."""
    path = Path.home() / "inference-server/kernels/ple-fix/norm_patch.py"
    spec = importlib.util.spec_from_file_location("ple_norm_patch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_prefill_forward(hc_fused)


def _unpack(v):
    return v if isinstance(v, tuple) else (v, None, None)


def run_case(name, rows, bits, inject_bits, scale, use_combine=True,
             concat=False, seed=0):
    hc_fused = load_hc_fused()
    dep = deployed_prefill(hc_fused)
    mod = GatedResidual(bits=bits, inject_bits=inject_bits,
                        use_combine=use_combine, seed=seed)
    x = make_input(rows, scale=scale, seed=seed + 7)

    canon = _unpack(mod._forward(x))
    mx.eval(canon)
    dep_out = _unpack(dep(mod, x))
    mx.eval(dep_out)
    fused = hc2._fused_prefill_forward(hc_fused, mod, x, use_concat=concat)
    assert fused is not None, "fused path failed closed"
    fused = _unpack(fused)
    mx.eval(fused)

    out = [(name, "mixed", *ulp_stats(fused[0], dep_out[0]),
            *ulp_stats(fused[0], canon[0])[2:])]
    if canon[2] is not None:
        out.append((name, "inject", *ulp_stats(fused[2], dep_out[2]),
                    *ulp_stats(fused[2], canon[2])[2:]))
    del canon, dep_out, fused, x, mod
    gc.collect()
    mx.clear_cache()
    return out


def chained(rows, layers=4, bits=5, scale=1.0, concat=False):
    """Four hyper-connection blocks chained as the decoder chains them.

    The branch between blocks is a fixed random projection, identical in both
    runs, so any divergence in the final stream is the block's own.
    """
    hc_fused = load_hc_fused()
    dep = deployed_prefill(hc_fused)
    hidden = 2560
    mx.random.seed(99)
    mods = [GatedResidual(bits=bits, seed=100 + i) for i in range(layers)]
    branch_w = [
        (mx.random.normal((hidden, hidden)) * (1.0 / hidden**0.5)).astype(mx.bfloat16)
        for _ in range(layers)
    ]
    x0 = make_input(rows, scale=scale, seed=5)

    def sweep(fn):
        x = x0
        for mod, bw in zip(mods, branch_w):
            mixed, hyper_input, inj = fn(mod, x)
            branch = mixed @ bw
            injection = branch[..., None, :] * inj[..., None]
            x = hyper_input + injection.reshape(hyper_input.shape)
            mx.eval(x)
        return x

    canon = sweep(lambda m, v: m._forward(v))
    dep_x = sweep(lambda m, v: dep(m, v))
    got = sweep(lambda m, v: hc2._fused_prefill_forward(hc_fused, m, v,
                                                        use_concat=concat))
    out = [("chain x%d T=%d" % (layers, rows), "stream",
            *ulp_stats(got, dep_x), *ulp_stats(got, canon)[2:])]
    del canon, dep_x, got, mods, branch_w, x0
    gc.collect()
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--chain-rows", type=int, default=2048)
    ap.add_argument("--concat", action="store_true",
                    help="also exercise the merged down|inject bank")
    a = ap.parse_args()
    R = a.rows
    c = a.concat

    table = []
    table += run_case(f"attn 5-bit T={R}", R, 5, None, 1.0, seed=0, concat=c)
    table += run_case(f"mlp 5-bit T={R}", R, 5, None, 1.0, seed=3, concat=c)
    table += run_case(f"attn 4-bit T={R}", R, 4, None, 1.0, seed=11, concat=c)
    table += run_case(f"mlp 6-bit T={R}", R, 6, None, 1.0, seed=13, concat=c)
    table += run_case(f"attn 8-bit T={R}", R, 8, None, 1.0, seed=17, concat=c)
    # the five layers whose injection bank is 5-bit over a 4-bit down bank
    table += run_case(f"mixed 4/5-bit T={R}", R, 4, 5, 1.0, seed=19, concat=c)
    # the final mixer has no block_inject_weight
    table += run_case(f"mixer no-inject T={R}", R, 4, None, 1.0,
                      use_combine=False, seed=29, concat=c)
    # magnitude sweep: the residual stream grows with depth
    table += run_case(f"5-bit T={R} x8", R, 5, None, 8.0, seed=31, concat=c)
    table += run_case(f"5-bit T={R} /8", R, 5, None, 0.125, seed=33, concat=c)
    table += run_case("5-bit T=512", 512, 5, None, 1.0, seed=37, concat=c)
    table += chained(a.chain_rows, concat=c)

    w = max(len(r[0]) for r in table)
    print(f"\n{'case':<{w}}  {'tensor':<7} {'max abs':>10} {'max rel':>10} "
          f"{'ULP max':>8} {'ULP mean':>9} | {'vs canon max':>12} {'mean':>8}")
    bad = 0
    for name, tensor, mabs, mrel, mulp, aulp, culp, caulp in table:
        flag = ""
        if mulp > 1.0:
            flag = "  <-- OVER 1 ULP"
            bad += 1
        print(f"{name:<{w}}  {tensor:<7} {mabs:10.3e} {mrel:10.3e} {mulp:8.2f} "
              f"{aulp:9.5f} | {culp:12.1f} {caulp:8.4f}{flag}")
    print("\nSTATS", hc2.STATS)
    print("PASS" if bad == 0 else f"FAIL: {bad} case(s) over 1 bf16 ULP")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
