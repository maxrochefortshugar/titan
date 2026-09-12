"""Price the QSA arms at the real Flash-Next shapes. Synthetic tensors only."""
import gc, importlib, statistics, sys, time

RES = "/Applications/oMLX.app/Contents/Resources"
sys.path.insert(0, RES + "/Python/framework-mlx-base/lib/python3.11/site-packages")
sys.path.insert(0, RES)
import mlx.core as mx
import mlx_vlm, mlx_vlm.models
VENDOR = RES + "/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm"
mlx_vlm.__path__.append(VENDOR)
mlx_vlm.models.__path__.append(VENDOR + "/models")
qsa = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

H_Q, H_KV, D = 24, 2, 256
ID = 128            # indexer head dim
IN_H = 4            # indexer query heads
RATIO, BUDGET = 4, 2048
LAYERS = 12
BW = 718e9
REPS = 15


def med_ms(fn, reps=REPS):
    fn(); mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(3):
            out = fn()
        mx.eval(out); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000 / 3)
    return statistics.median(ts)


class Norm:
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def rope(x, pos):
    return x


def run(N, M):
    q = mx.random.normal((1, H_Q, M, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, H_KV, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, H_KV, N, D)).astype(mx.bfloat16)
    iq = mx.random.normal((1, M, IN_H, ID)).astype(mx.bfloat16)
    ik = mx.random.normal((1, N, ID)).astype(mx.bfloat16)
    ipos = mx.arange(N, dtype=mx.int32).reshape(1, N)
    pooled = mx.random.normal((1, N // RATIO, ID)).astype(mx.bfloat16)
    mx.eval(q, k, v, iq, ik, ipos, pooled)

    def sparse():
        if M == 1:
            return qsa.contiguous_causal_gathered_qsa_decode(
                q, k, v, iq.reshape(1, 1, IN_H, ID), pooled,
                num_query_heads=H_Q, num_key_value_heads=H_KV, head_dim=D,
                indexer_head_dim=ID, compress_ratio=RATIO, token_budget=BUDGET,
            )
        return qsa.contiguous_causal_gathered_qsa(
            q, k, v, iq, ik, ipos,
            num_query_heads=H_Q, num_key_value_heads=H_KV, head_dim=D,
            indexer_head_dim=ID, compress_ratio=RATIO, token_budget=BUDGET,
            index_key_norm=Norm(), apply_index_rope=rope,
            pooled_index_keys=pooled,
        )

    def dense_causal():
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=D ** -0.5, mask="causal")

    # the real dense fallback: indexer bool mask [1,1,M,N] over the whole cache
    bmask = mx.zeros((1, 1, M, N), dtype=mx.bool_)
    ones = mx.ones((1, 1, M, 2048), dtype=mx.bool_)
    bmask = mx.concatenate([bmask[..., :-2048], ones], axis=-1)
    mx.eval(bmask)

    def dense_masked():
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=D ** -0.5, mask=bmask)

    res = {}
    res["sparse"] = med_ms(sparse)
    res["dense_causal"] = med_ms(dense_causal)
    res["dense_masked"] = med_ms(dense_masked)
    del q, k, v, iq, ik, ipos, pooled, bmask, ones
    gc.collect(); mx.clear_cache()
    return res


print(f"{'ctx':>8} {'M':>3} | {'sparse ms/l':>11} {'dense ms/l':>10} "
      f"{'dense+mask':>10} | {'sparse fwd':>10} {'dense fwd':>9} | "
      f"{'KV dense MB':>11} {'roofline ms':>11}")
for N in (8192, 65536, 131072):
    for M in (1, 4, 6):
        r = run(N, M)
        kvmb = 2 * H_KV * N * D * 2 / 1e6
        roof = (kvmb * 1e6) / BW * 1e3
        print(f"{N:>8} {M:>3} | {r['sparse']:>11.3f} {r['dense_causal']:>10.3f} "
              f"{r['dense_masked']:>10.3f} | {r['sparse']*LAYERS:>10.2f} "
              f"{r['dense_causal']*LAYERS:>9.2f} | {kvmb:>11.1f} {roof:>11.3f}")
    print()
print(f"peak GPU MB: {mx.get_peak_memory()/1e6:.0f}")
