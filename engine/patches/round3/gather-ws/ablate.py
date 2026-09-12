# SPDX-License-Identifier: Apache-2.0
"""Where the time goes: dequantise-and-stage only, matmul only, and both.

Both halves are compiled from the shipped source with one macro neutered, so
the tiling, the loop structure and the register pressure are unchanged.
"""
import sys, time
import mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/round3/gather-ws')
import kernel as K

E, TOPK, CHAIN, ROUNDS = 512, 10, 5, 7


def build(cfg, Kd, N, G, mode):
    consts, src = K._source(cfg, Kd, N, G)
    if mode == "stage":       # keep the dequantise + threadgroup store, drop the matmul
        src = src.replace("      qop.run(tA, tB, acc); }",
                          "      if (Wsg[lane] > bfloat(1e30)) qop.run(tA, tB, acc); }")
    elif mode == "mm":        # keep the matmul, drop the dequantise + store
        i = src.index("#define WS_STORE(buf)")
        j = src.index("#define WS_MM(")
        # zero the staging buffer once: uninitialised threadgroup contents can be
        # denormal bf16 and make the tensor op look slower than it is
        zero = ("#define WS_STORE(buf) {}\n\n")
        src = src[:i] + zero + src[j:]
        src = src.replace("    WS_FETCH(0u)\n    for (uint g",
                          "    for (uint z = lane; z < TN_ * 64u; z += 32u) Wsg[z] = bfloat(0.001f);\n"
                          "    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
                          "    WS_FETCH(0u)\n    for (uint g")
        src = src.replace("TN_", str(cfg[1]))
    return mx.fast.metal_kernel(
        name=f"abl_{mode}_{'_'.join(map(str,cfg))}_{Kd}_{N}",
        input_names=["x","wq","scales","biases","offsets","tile_row","tile_exp","ntiles"],
        output_names=["y"], header=K._HEADER + consts, source=src)


for name, N, Kd in (("gate_up", 1280, 2560), ("down", 2560, 640)):
    w = (mx.random.normal((E, N, Kd)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4); del w
    mx.eval(wq, s, b); mx.clear_cache()
    T = 2048; R = T * TOPK
    idx = mx.sort(mx.random.randint(0, E, (R,)).astype(mx.uint32))
    x = (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16); mx.eval(idx, x)
    xf = mx.contiguous(x.reshape(R, Kd)); mx.eval(xf)
    cfg = K.pick_cfg(N)
    TM, TN, SGN = cfg[0], cfg[1], cfg[2]
    maxt = K.max_tiles(R, E, TM)
    tb = K.build_tiles(idx, E, TM, maxt); mx.eval(tb)
    for mode in ("full", "stage", "mm"):
        k = build(cfg, Kd, N, Kd // 64, mode)
        f = lambda: k(inputs=[xf, wq, s, b, *tb], output_shapes=[(R, N)],
                      output_dtypes=[mx.bfloat16],
                      grid=(32 * SGN * (N // (TN * SGN)), maxt, 1),
                      threadgroup=(32 * SGN, 1, 1), template=[("ROWS", R)])[0]
        best = 1e9
        for _ in range(ROUNDS):
            o = f(); mx.eval(o); mx.synchronize(); t0 = time.perf_counter()
            outs = [f() for _ in range(CHAIN)]; mx.eval(outs); mx.synchronize()
            best = min(best, (time.perf_counter() - t0) / CHAIN)
        print(f"{name:8s} cfg={cfg} {mode:6s} {best*1e3:7.3f} ms")
    del wq, s, b, idx, x, xf, tb
    mx.clear_cache()
