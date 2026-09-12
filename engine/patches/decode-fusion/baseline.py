import time, statistics as st, sys
import mlx.core as mx
from moe_ref import SparseMoeBlock, HIDDEN
import graphcount

CHAIN = 10

def timeit(fn, iters=12, warm=3):
    """Min-of-iters: the GPU is shared with a live daemon, so the median is
    contaminated by contention while the minimum is the clean kernel time."""
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)
    for _ in range(warm): run()
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return min(ts)

if __name__ == "__main__":
    print("building E=512 block ...", flush=True)
    t0 = time.time()
    blk = SparseMoeBlock()
    mx.eval(blk.parameters())
    print(f"built in {time.time()-t0:.1f}s  active={mx.get_active_memory()/1e9:.2f} GB", flush=True)

    for T in (1, 2, 8):
        x = mx.random.normal((1, T, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        y = blk(x); mx.eval(y)
        n, c = graphcount.count(blk(x))
        t = timeit(lambda: blk(x))
        # sub-step breakdown
        xn = blk.input_layernorm(x); mx.eval(xn)
        inds, sc = blk.route(xn); mx.eval(inds, sc)
        t_norm = timeit(lambda: blk.input_layernorm(x))
        t_route = timeit(lambda: blk.route(xn))
        t_mlp = timeit(lambda: blk.switch_mlp(xn, inds))
        ymlp = blk.switch_mlp(xn, inds); mx.eval(ymlp)
        t_ws = timeit(lambda: (ymlp * sc[..., None].astype(ymlp.dtype)).sum(axis=-2))
        print(f"\nT={T}: total {t*1e6:8.1f} us   prims={n}")
        print(f"   norm {t_norm*1e6:7.1f}  route {t_route*1e6:7.1f}  switch_mlp {t_mlp*1e6:7.1f}  wsum {t_ws*1e6:7.1f}")
        print("   ", dict(c))


def timeit_pair(fns, iters=15, warm=3):
    """Interleaved min-of-N for several closures.

    The GPU is shared with a live daemon; measuring A for N rounds and then B
    for N rounds lets a burst of contention land on only one of them.  Round
    robin instead and take each one's own minimum.
    """
    def run(fn):
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)
    for _ in range(warm):
        for fn in fns:
            run(fn)
    best = [float("inf")] * len(fns)
    for _ in range(iters):
        for i, fn in enumerate(fns):
            mx.synchronize(); t0 = time.perf_counter(); run(fn); mx.synchronize()
            best[i] = min(best[i], (time.perf_counter() - t0) / CHAIN)
    return best
