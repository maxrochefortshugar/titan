"""Exactness of the gathered QSA verify arm at the real Flash-Next shapes.

[A] verify(M rows) == dense SDPA over the whole cache masked to exactly the
    tokens QSA selected. Reduces to plain dense attention when the mask is all
    ones, which is the "identical to dense when all blocks are selected" case.
[B] verify row j == the gathered sparse DECODE arm run on a cache truncated to
    that row's visible prefix.
No model is loaded; every tensor is synthetic.
"""
import importlib, math, sys

RES = "/Applications/oMLX.app/Contents/Resources"
sys.path.insert(0, RES + "/Python/framework-mlx-base/lib/python3.11/site-packages")
sys.path.insert(0, RES)
import mlx.core as mx
import mlx_vlm, mlx_vlm.models

V = RES + "/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm"
mlx_vlm.__path__.append(V)
mlx_vlm.models.__path__.append(V + "/models")
qsa = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

H_Q, H_KV, D, ID, IN_H = 24, 2, 256, 128, 4
RATIO, BUDGET = 4, 2048
BLOCK_BUDGET = BUDGET // RATIO


class Norm:
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def rope(x, pos):
    return x


def selection_for_row(pooled, iq_row, q_abs):
    """Reproduce the arm's block choice for one query at absolute index q_abs."""
    max_blocks = pooled.shape[1]
    scores = qsa._portable_indexer_scores(iq_row, pooled, ID)  # (1,1,max_blocks)
    complete = (q_abs + 1) // RATIO
    valid = mx.arange(max_blocks)[None, None, :] < complete
    scores = mx.where(valid, scores, mx.finfo(scores.dtype).min)
    if max_blocks > BLOCK_BUDGET:
        sel = mx.argpartition(scores, kth=-BLOCK_BUDGET, axis=-1)[..., -BLOCK_BUDGET:]
    else:
        sel = mx.broadcast_to(mx.arange(max_blocks, dtype=mx.int32)[None, None], (1, 1, max_blocks))
    sel = mx.sort(sel.astype(mx.int32), axis=-1)
    keep = min(int(complete), BLOCK_BUDGET)
    toks = set()
    for b in sel.reshape(-1).tolist()[:keep] if keep else []:
        toks.update(range(b * RATIO, b * RATIO + RATIO))
    toks.update(range(int(complete) * RATIO, q_abs + 1))   # the visible tail
    return toks


def run(N, M, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, H_Q, M, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, H_KV, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, H_KV, N, D)).astype(mx.bfloat16)
    iq = mx.random.normal((1, M, IN_H, ID)).astype(mx.bfloat16)
    ik = mx.random.normal((1, N, ID)).astype(mx.bfloat16)
    ipos = mx.arange(N, dtype=mx.int32).reshape(1, N)
    pooled = Norm()(mx.mean(ik[:, : (N // RATIO) * RATIO].reshape(1, N // RATIO, RATIO, ID), axis=2)).astype(mx.bfloat16)
    mx.eval(q, k, v, iq, ik, ipos, pooled)

    out = qsa.contiguous_causal_gathered_qsa(
        q, k, v, iq, ik, ipos,
        num_query_heads=H_Q, num_key_value_heads=H_KV, head_dim=D,
        indexer_head_dim=ID, compress_ratio=RATIO, token_budget=BUDGET,
        index_key_norm=Norm(), apply_index_rope=rope, pooled_index_keys=pooled,
    )                                   # (1, M, H_Q, D)
    mx.eval(out)

    # --- [A] dense SDPA masked to the selected tokens -----------------------
    query_start = N - M
    rows = []
    for j in range(M):
        toks = selection_for_row(pooled, iq[:, j : j + 1], query_start + j)
        m = mx.zeros((N,), dtype=mx.bool_)
        idx = mx.array(sorted(toks), dtype=mx.int32)
        m = mx.zeros((N,)).at[idx].add(1.0) > 0.5
        rows.append(m)
    bmask = mx.stack(rows)[None, None]                       # (1,1,M,N)
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=D ** -0.5, mask=bmask).transpose(0, 2, 1, 3)
    mx.eval(ref)
    errA = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
    denom = float(mx.max(mx.abs(ref.astype(mx.float32)))) or 1.0
    frac_dense = float(mx.mean(bmask.astype(mx.float32)))

    # --- [B] each verify row against the gathered sparse DECODE arm ---------
    errB = 0.0
    for j in range(M):
        n = query_start + j + 1
        mb = n // RATIO
        if mb <= BLOCK_BUDGET:
            continue
        dec = qsa.contiguous_causal_gathered_qsa_decode(
            mx.contiguous(q[:, :, j : j + 1]), mx.contiguous(k[:, :, :n]), mx.contiguous(v[:, :, :n]),
            mx.contiguous(iq[:, j : j + 1].reshape(1, 1, IN_H, ID)), mx.contiguous(pooled[:, :mb]),
            num_query_heads=H_Q, num_key_value_heads=H_KV, head_dim=D,
            indexer_head_dim=ID, compress_ratio=RATIO, token_budget=BUDGET,
        )
        mx.eval(dec)
        errB = max(errB, float(mx.max(mx.abs(
            dec.astype(mx.float32) - out[:, j : j + 1].astype(mx.float32)))))
    return errA, errA / denom, errB, frac_dense


print(f"{'N':>7} {'M':>3} | {'[A] max abs':>12} {'rel':>9} | {'[B] max abs':>12} | "
      f"{'frac of cache read':>18}")
for N, M in ((4096, 4), (8192, 2), (8192, 4), (8192, 6), (16384, 4), (65536, 4)):
    a, r, b, f = run(N, M)
    print(f"{N:>7} {M:>3} | {a:>12.3e} {r:>9.2e} | {b:>12.3e} | {f:>17.3%}")

# All blocks selected: budget >= complete blocks makes the mask all-ones, so [A]
# degenerates to plain dense causal attention.
mx.random.seed(1)
N, M = 2048 + 4, 4
a, r, b, f = run(N, M)
print(f"\nbudget-covering case N={N} M={M}: [A] max abs {a:.3e}, "
      f"cache fraction read {f:.1%} (dense equivalent)")
print(f"peak GPU MB: {mx.get_peak_memory()/1e6:.0f}")
