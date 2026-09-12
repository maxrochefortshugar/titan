"""Prove which QSA arm each regime takes, using the real predicate code.

No model is loaded. The predicates are called unbound on a stub whose only
attributes are the ones they read.
"""
import importlib.util, os, sys, types

RES = "/Applications/oMLX.app/Contents/Resources"
sys.path.insert(0, RES + "/Python/framework-mlx-base/lib/python3.11/site-packages")
sys.path.insert(0, RES)

import mlx.core as mx
VENDOR = RES + "/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm"
import mlx_vlm, mlx_vlm.models
mlx_vlm.__path__.append(VENDOR)
mlx_vlm.models.__path__.append(VENDOR + "/models")
lang = importlib.import_module("mlx_vlm.models.qwen4_exp.language")

Att = lang.Qwen4ExpAttention
QSAKVCache = lang.QSAKVCache


class Idx:
    token_budget = 2048
    compress_ratio = 4
    block_topk = 512


class Stub:
    indexer = Idx()
    _batch_one_text_position_ids = staticmethod(
        Att._batch_one_text_position_ids.__func__
        if hasattr(Att._batch_one_text_position_ids, "__func__")
        else Att._batch_one_text_position_ids
    )


def warm_cache(offset, cls=QSAKVCache):
    c = cls()
    # Fake an aligned cache without allocating the real KV (predicates only
    # read offset and the two indexer shapes).
    c.offset = offset
    c.index_keys = mx.zeros((1, offset, 1), dtype=mx.bfloat16) if offset else None
    c.index_position_ids = mx.zeros((1, offset), dtype=mx.int32) if offset else None
    return c


def route(L, offset, target_verify, batch=1, pos="rank2"):
    x = mx.zeros((batch, L, 10240), dtype=mx.bfloat16)
    cache = warm_cache(offset)
    mask = None if L == 1 else "causal"
    if pos == "rank2":
        position_ids = mx.arange(offset, offset + L, dtype=mx.int32).reshape(1, L)
    elif pos == "none":
        position_ids = None
    else:  # 3-plane mRoPE
        position_ids = mx.zeros((3, batch, L), dtype=mx.int32)
    s = Stub()
    args = (x, mask, cache, position_ids, None, target_verify)
    if Att._gathered_text_decode_eligible(s, *args):
        return "gathered decode (sparse, 2048+3 KV)"
    if Att._gathered_text_prefill_eligible(s, *args):
        return "gathered prefill (sparse, 2048+3 KV per query)"
    if Att._gathered_text_verify_eligible(s, *args):
        return "gathered verify (sparse, 2048+3 KV per query)"
    return "DENSE masked SDPA (whole cache)"


print(f"OMLX_QWEN4_QSA_GATHERED_VERIFY disabled = {lang._GATHERED_VERIFY_DISABLED}")
print(f"gathered min query tokens = {lang._gathered_min_query_tokens()}")
print()
rows = []
for ctx in (8192, 65536, 131072):
    rows.append((f"decode L=1, MTP off (tv=False), ctx={ctx}", route(1, ctx, False)))
    rows.append((f"decode L=1, MTP depth-0 (tv=True),  ctx={ctx}", route(1, ctx, True)))
    for L in (2, 3, 4, 5, 6):
        rows.append((f"verify L={L} (tv=True),            ctx={ctx}", route(L, ctx, True)))
    rows.append((f"prefill L=2048 (tv=False),        ctx={ctx}", route(2048, ctx, False)))
    rows.append((f"prefill L=2048 (tv=True),         ctx={ctx}", route(2048, ctx, True)))
    rows.append((f"prefill L=512 contended,          ctx={ctx}", route(512, ctx, False)))
    rows.append((f"prefill L=2048 batch=2,           ctx={ctx}", route(2048, ctx, False, batch=2)))
    rows.append((f"verify L=4, 3-plane mRoPE ids,    ctx={ctx}", route(4, ctx, True, pos="mrope")))
    rows.append((f"verify L=4, position_ids=None,    ctx={ctx}", route(4, ctx, True, pos="none")))
for k, v in rows:
    print(f"{k:46s} -> {v}")
