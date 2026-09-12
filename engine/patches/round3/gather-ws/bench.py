# SPDX-License-Identifier: Apache-2.0
"""Stock mx.gather_qmm(sorted) vs the weight-stationary bf16 kernel.

Rounds alternate stock and kernel so a busy GPU perturbs both equally; min over
rounds and median are both reported.  Routing is Zipf-like (skewed expert
popularity), matching real top-10 routing rather than uniform.  Segmentation
(build_tiles) is timed separately and also included in an end-to-end column.
"""
import sys, time, argparse
import mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/round3/gather-ws')
import kernel as K

E, TOPK = 512, 10
PEAK = 65.7e12
CHAIN, ROUNDS = 5, 11


def zipf_indices(R, E, a=0.7, seed=0):
    """Skewed routing: expert popularity ~ 1/rank^a, then sorted."""
    mx.random.seed(seed)
    p = 1.0 / mx.power(mx.arange(1, E + 1).astype(mx.float32), a)
    p = p / p.sum()
    perm = mx.random.permutation(E)
    cdf = mx.cumsum(p)
    u = mx.random.uniform(shape=(R,))
    pos = (u[:, None] > cdf[None, :]).sum(axis=1)
    idx = perm[mx.minimum(pos, E - 1)].astype(mx.uint32)
    return mx.sort(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None, help="TM,TN,SGN")
    ap.add_argument("--ts", default="512,2048")
    ap.add_argument("--shapes", default="gate_up,down")
    a = ap.parse_args()
    bases = [tuple(int(v) for v in c.split(",")) for c in a.cfg.split(";")] if a.cfg else [K.DEFAULT_CFG]
    TS = [int(v) for v in a.ts.split(",")]
    allsh = {"gate_up": ("gate_up (fused)", 1280, 2560), "down": ("down", 2560, 640)}
    print(f"{'shape':16s} {'T':>5s} {'rows':>6s} {'GFLOP':>6s} | {'stock ms':>8s} {'med':>7s} "
          f"{'TF/s':>5s} {'%pk':>4s} | {'ws ms':>7s} {'med':>7s} {'TF/s':>5s} {'%pk':>4s} "
          f"| {'seg ms':>6s} {'ws+seg':>7s} {'x':>5s}")
    for key in a.shapes.split(","):
        name, N, Kd = allsh[key]
        w = (mx.random.normal((E, N, Kd)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        del w
        mx.eval(wq, s, b)
        mx.clear_cache()
        for T in TS:
            R = T * TOPK
            idx = zipf_indices(R, E)
            x = (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16)
            mx.eval(idx, x)
            maxt = K.max_tiles(R, E, 48)
            xf = mx.contiguous(x.reshape(R, Kd))
            mx.eval(xf)
            fns = {
                "stock": lambda: mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                                               group_size=64, bits=4, sorted_indices=True),
                "seg": lambda: K.build_tiles(idx, E, 48, maxt)[0],
            }
            for bi, base in enumerate(bases):
                cfgb = K.pick_cfg(N, base)
                tb = K.build_tiles(idx, E, cfgb[0], K.max_tiles(R, E, cfgb[0]))
                mx.eval(tb)
                fns[f"ws{bi}"] = (lambda c=cfgb, tb=tb, mt=K.max_tiles(R, E, cfgb[0]):
                                  K.gather_ws(xf, wq, s, b, tb[0], tb[1], tb[2], tb[3], mt, c))
            res = {k: [1e9, []] for k in fns}
            for _ in range(ROUNDS):
                for k, f in fns.items():
                    o = f(); mx.eval(o)
                    mx.synchronize(); t0 = time.perf_counter()
                    outs = [f() for _ in range(CHAIN)]
                    mx.eval(outs); mx.synchronize()
                    dt = (time.perf_counter() - t0) / CHAIN
                    res[k][0] = min(res[k][0], dt); res[k][1].append(dt)
            med = lambda k: sorted(res[k][1])[ROUNDS // 2]
            fl = 2 * R * N * Kd
            sm, gm = res["stock"][0], res["seg"][0]
            print(f"{name:16s} {T:5d} {R:6d} {fl/1e9:6.1f} | "
                  f"{sm*1e3:8.3f} {med('stock')*1e3:7.3f} {fl/sm/1e12:5.1f} {100*fl/sm/PEAK:4.0f} | "
                  f"seg {gm*1e3:.3f}")
            for bi, base in enumerate(bases):
                wm = res[f"ws{bi}"][0]; tot = wm + gm
                print(f"    cfg {str(base):20s} {wm*1e3:8.3f} {med(f'ws{bi}')*1e3:7.3f} "
                      f"{fl/wm/1e12:5.1f} {100*fl/wm/PEAK:4.0f} | +seg {tot*1e3:7.3f} {sm/tot:5.2f}x")
            del idx, x, xf
            mx.clear_cache()
        del wq, s, b
        mx.clear_cache()


main()
