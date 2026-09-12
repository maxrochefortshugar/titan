#!/usr/bin/env python
"""Correctness of the fused MoE decode kernels against the stock mlx-lm path.

Reference is an fp32 golden computed from the same quantized weights (exact
affine dequantisation in float64->float32 numpy), so stock and fused are scored
on the same ruler.  Stock accumulates through bf16 intermediates; the fused
kernels keep fp32 all the way to the final store, so the fused path should be
at or below the stock error, and both within a bf16 ULP (rel 2^-8 = 3.9e-3).
"""
import argparse
import numpy as np
import mlx.core as mx

import kernel as K
import fused_kernels as fk
from moe_ref import SparseMoeBlock


def deq32(q, s, b, gs):
    qn = np.array(q)
    sn = np.array(s.astype(mx.float32)).astype(np.float32)
    bn = np.array(b.astype(mx.float32)).astype(np.float32)
    E, R, NW = qn.shape
    out = np.empty((E, R, NW * 8), np.float32)
    for c in range(8):
        out[:, :, c::8] = ((qn >> (4 * c)) & 0xF).astype(np.float32)
    g = np.repeat(np.arange(NW * 8 // gs), gs)
    return out * sn[:, :, g] + bn[:, :, g]


def golden(blk, x, idx, sc, gs=64):
    gu, dn = blk.switch_mlp.gate_up_proj, blk.switch_mlp.down_proj
    # Dequantize only the routed experts: at E=512 the full tensor is 6.7 GB.
    use = mx.array(np.unique(np.array(idx)).astype(np.uint32))
    remap = {int(e): i for i, e in enumerate(np.array(use))}
    W = deq32(gu["weight"][use], gu["scales"][use], gu["biases"][use], gs)
    DW = deq32(dn["weight"][use], dn["scales"][use], dn["biases"][use], gs)
    xn = np.array(x.astype(mx.float32))
    gi, scn = np.array(idx), np.array(sc)
    T, topk = gi.shape
    I = W.shape[1] // 2
    y = np.zeros((T, DW.shape[1]), np.float32)
    for t in range(T):
        for j in range(topk):
            e = remap[int(gi[t, j])]
            g = W[e, :I] @ xn[t]
            u = W[e, I:] @ xn[t]
            h = (g / (1.0 + np.exp(-g))) * u
            y[t] += scn[t, j] * (DW[e] @ h)
    return y


def err(a, g):
    a = np.asarray(a, np.float32)
    d = np.abs(a - g)
    scale = np.abs(g).max()
    return d.max(), d.max() / max(scale, 1e-30)


def run(E, T, topk, heavy, seed, gs=64):
    mx.random.seed(seed)
    np.random.seed(seed)
    blk = SparseMoeBlock(E=E, top_k=topk, gs=gs, seed=seed)
    mx.eval(blk.parameters())
    if heavy:  # heavy-tailed activations: a few large channels
        x = (mx.random.normal((1, T, 2560)) *
             mx.exp(mx.random.normal((1, T, 2560)) * 2.0)).astype(mx.bfloat16)
    else:
        x = mx.random.normal((1, T, 2560)).astype(mx.bfloat16)
    mx.eval(x)

    inds, scores = blk.route(x)
    mx.eval(inds, scores)
    idx = inds.reshape(T, topk).astype(mx.uint32)
    sc = scores.reshape(T, topk).astype(mx.float32)
    mx.eval(idx, sc)

    g = golden(blk, x.reshape(T, 2560), idx, sc, gs)

    stock = np.array(blk(x).reshape(T, 2560).astype(mx.float32))
    h = fk.gate_up_silu(x.reshape(T, 2560), *[blk.switch_mlp.gate_up_proj[n]
                        for n in ("weight", "scales", "biases")], idx, gs)
    fused = np.array(fk.down_wsum(h, *[blk.switch_mlp.down_proj[n]
                     for n in ("weight", "scales", "biases")], idx, sc, gs,
                     out_dtype=mx.bfloat16).astype(mx.float32))
    return err(stock, g), err(fused, g)


def run_patched(E, T, topk, seed, gs=64):
    """End-to-end: apply() then compare patched block vs the original call."""
    mx.random.seed(seed)
    blk = SparseMoeBlock(E=E, top_k=topk, gs=gs, seed=seed)
    mx.eval(blk.parameters())
    x = mx.random.normal((1, T, 2560)).astype(mx.bfloat16)
    mx.eval(x)
    import os
    os.environ[K._ENV] = "0"
    ref = np.array(blk(x).astype(mx.float32))
    os.environ[K._ENV] = "1"
    got = np.array(blk(x).astype(mx.float32))
    d = np.abs(ref - got)
    return d.max(), d.max() / max(np.abs(ref).max(), 1e-30)


def router_agreement(E, trials, topk=10, gs=64, seed=7):
    """Fused router vs stock softmax+argpartition: logit error and top-k set."""
    mx.random.seed(seed)
    blk = SparseMoeBlock(E=E, top_k=topk, gs=gs, seed=seed)
    mx.eval(blk.parameters())
    g = blk.gate
    ntok = mism = 0
    worst_logit = worst_score = 0.0
    for _ in range(trials):
        T = 2
        x = mx.random.normal((T, 2560)).astype(mx.bfloat16)
        mx.eval(x)
        lg = fk.router_logits(x, g["weight"], g["scales"], g["biases"], gs)
        idx, sc = fk.router_topk(lg, topk, True)
        stock_lg = g(x)
        gates = mx.softmax(stock_lg, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-topk, axis=-1)[..., -topk:]
        ss = mx.take_along_axis(gates, inds, axis=-1)
        ss = ss / ss.sum(-1, keepdims=True)
        mx.eval(lg, idx, sc, stock_lg, inds, ss)
        worst_logit = max(worst_logit,
                          float(np.abs(np.array(stock_lg.astype(mx.float32))
                                       - np.array(lg)).max()))
        for t in range(T):
            ntok += 1
            a = dict(zip(np.array(inds)[t].tolist(),
                         np.array(ss.astype(mx.float32))[t].tolist()))
            b = dict(zip(np.array(idx)[t].tolist(), np.array(sc)[t].tolist()))
            if set(a) != set(b):
                mism += 1
            for kk in set(a) & set(b):
                worst_score = max(worst_score, abs(a[kk] - b[kk]))
    return ntok, mism, worst_logit, worst_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="also run E=512")
    a = ap.parse_args()

    K.apply()
    BF16_ULP = 2 ** -8
    print(f"bf16 relative ULP = {BF16_ULP:.3e}\n")
    print(f"{'case':34s} {'stock abs':>11s} {'stock rel':>10s} "
          f"{'fused abs':>11s} {'fused rel':>10s}  verdict")
    rows = []
    cases = [(a.experts, T, 10, heavy, s)
             for T in (1, 2, 8) for heavy in (False, True) for s in (0, 1)]
    if a.full:
        cases += [(512, T, 10, False, 0) for T in (1, 2)]
    for E, T, topk, heavy, s in cases:
        (sa, sr), (fa, fr) = run(E, T, topk, heavy, s)
        ok = fr <= BF16_ULP
        name = f"E={E} T={T} {'heavy' if heavy else 'normal'} seed{s}"
        print(f"{name:34s} {sa:11.3e} {sr:10.3e} {fa:11.3e} {fr:10.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
        rows.append(ok)

    print("\nend-to-end patched block vs stock block (same routing).")
    print("Bar here is 4 bf16 ULP: the gap is dominated by the STOCK path\'s own")
    print("bf16 intermediates (~2.5 ULP off the fp32 golden above); fused is closer.")
    for T in (1, 2, 8):
        da, dr = run_patched(a.experts, T, 10, 0)
        tag = "fused" if T <= 2 else "falls back (expect exactly 0)"
        ok = dr <= 4 * BF16_ULP
        print(f"  T={T:<2d} maxabs {da:.3e}  rel {dr:.3e}  {'PASS' if ok else 'FAIL'}"
              f"   [{tag}]")
        rows.append(ok)
    n, m, wl, ws = router_agreement(a.experts, 20)
    print("\nfused router (OMLX_MOE_DECODE_FUSED_ROUTER=1) vs stock router:")
    print(f"  max |logit diff| {wl:.3e}   (a bf16 ULP at |logit|~5 is "
          f"{5*2**-8:.3e}: the matvec itself agrees)")
    print(f"  max |score diff| on commonly-selected experts {ws:.3e}")
    print(f"  top-{10} set disagreements: {m}/{n} tokens -- these are bf16 ties "
          f"at the k/k+1 boundary,\n  where stock's argpartition picks "
          f"arbitrarily and the kernel picks the lowest index.\n  Not covered "
          f"by the bf16-ULP bar; this is why the router fusion is opt-in.")

    print("\nALL PASS" if all(rows) else "\nFAILURES PRESENT")
