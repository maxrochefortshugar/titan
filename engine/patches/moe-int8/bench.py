# SPDX-License-Identifier: Apache-2.0
"""Before/after for the routed-expert prefill GEMMs of Qwen3.8-Flash-Next.

Stock is ``mx.gather_qmm(..., sorted_indices=True)`` (mlx 0.32.2's M5 NAX sorted
gather).  Rounds alternate stock and kernel so a busy GPU perturbs both equally;
min over rounds is the cleanest number on a shared GPU, median is also reported.

FLOPs are counted over the *routed rows actually gathered* (T * top_k), not the
full expert tensor.
"""
import time, sys
import mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/moe-int8')
import kernel as K

E, TOPK = 512, 10
SHAPES = (("gate_up (fused)", 1280, 2560), ("down", 2560, 640))
TS = (512, 1024, 2048, 4096)
CHAIN, ROUNDS = 5, 7


def main():
    print(f"{'shape':16s} {'T':>5s} {'rows':>7s} {'GFLOP':>7s} | "
          f"{'stock min':>9s} {'med':>7s} {'TF/s':>6s} | {'int8 min':>8s} {'med':>7s} {'TOP/s':>6s} "
          f"{'quant':>6s} | {'speedup':>7s}")
    for name, N, Kd in SHAPES:
        w = (mx.random.normal((E, N, Kd)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        del w
        mx.eval(wq, s, b)
        st, bt, qt = K.prepare_weights(wq, s, b)
        mx.eval(st, bt, qt)
        cfg = K.pick_cfg(N)
        TMT = cfg[0] * cfg[2]
        for T in TS:
            R = T * TOPK
            idx = mx.sort(mx.random.randint(0, E, (R,)).astype(mx.uint32))
            x = (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16)
            mx.eval(idx, x)
            maxt = K.max_tiles(R, E, TMT)
            offs, trow, texp, nt = K.build_tiles(idx, E, TMT, maxt)
            mx.eval(offs, trow, texp, nt)
            xf = x.reshape(R, Kd)

            fns = {
                "stock": lambda: mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                                               group_size=64, bits=4, sorted_indices=True),
                "int8": lambda: K.gather_qmm_int8(xf, wq, st, bt, qt, offs, trow, texp,
                                                  nt, maxt, cfg),
                "quant": lambda: K.quantize_activations(xf, TMT),
            }
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
            sm, im, qm = res["stock"][0], res["int8"][0], res["quant"][0]
            print(f"{name:16s} {T:5d} {R:7d} {fl/1e9:7.1f} | "
                  f"{sm*1e3:9.3f} {med('stock')*1e3:7.3f} {fl/sm/1e12:6.1f} | "
                  f"{im*1e3:8.3f} {med('int8')*1e3:7.3f} {fl/im/1e12:6.1f} {qm*1e3:6.3f} | "
                  f"{sm/im:6.2f}x")
            del idx, x, xf, offs, trow, texp, nt
            mx.clear_cache()
        del wq, s, b, st, bt, qt
        mx.clear_cache()


main()
