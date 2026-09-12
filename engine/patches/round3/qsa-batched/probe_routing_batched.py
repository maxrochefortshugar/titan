# SPDX-License-Identifier: Apache-2.0
"""Which arm each regime takes, before and after the patch.

Calls the real predicates on real cache objects (warm KV state is faked at a
tiny head dim so nothing large is allocated). No model, no server.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from common import QSAKVCache, lang, mx  # noqa: E402

BatchQSAKVCache = lang.BatchQSAKVCache
Att = lang.Qwen4ExpAttention


class Indexer:
    token_budget = 2048
    compress_ratio = 4
    block_topk = 512


class Stub:
    """Only the attributes the predicates read."""

    indexer = Indexer()
    _batch_one_text_position_ids = staticmethod(
        Att._batch_one_text_position_ids.__func__
        if hasattr(Att._batch_one_text_position_ids, "__func__")
        else Att._batch_one_text_position_ids
    )


def single(offset):
    cache = QSAKVCache()
    cache.offset = offset
    cache.index_keys = mx.zeros((1, offset, 1), dtype=mx.bfloat16)
    cache.index_position_ids = mx.zeros((1, offset), dtype=mx.int32)
    return cache


def batched(lengths):
    width = max(lengths)
    pads = [width - n for n in lengths]
    cache = BatchQSAKVCache(pads)
    cache.kv_cache.keys = mx.zeros((len(lengths), 1, width, 1), dtype=mx.bfloat16)
    cache.kv_cache.values = cache.kv_cache.keys
    cache.kv_cache._idx = width
    cache.index_keys = mx.zeros((len(lengths), width, 1), dtype=mx.bfloat16)
    cache.index_position_ids = mx.zeros((len(lengths), width), dtype=mx.int32)
    cache.index_offset = width
    return cache


def arm(stub, x, mask, cache, position_ids, target_verify):
    args = (x, mask, cache, position_ids, None, target_verify)
    eligible = getattr(Att, "_omlx_batched_eligible", None)
    if eligible is not None and eligible(stub, *args) is not None:
        return "BATCHED gathered (sparse)"
    if Att._gathered_text_decode_eligible(stub, *args):
        return "gathered decode (sparse)"
    if Att._gathered_text_prefill_eligible(stub, *args):
        return "gathered prefill (sparse)"
    if Att._gathered_text_verify_eligible(stub, *args):
        return "gathered verify (sparse)"
    return "DENSE masked SDPA (whole cache)"


def sweep(label):
    stub = Stub()
    print(f"\n=== {label}")
    rows = []
    for ctx in (4096, 8192, 65536):
        rows.append(
            (
                f"single decode L=1, ctx={ctx}",
                arm(stub, mx.zeros((1, 1, 8), dtype=mx.bfloat16), None, single(ctx),
                    mx.zeros((1, 1), dtype=mx.int32), False),
            )
        )
        rows.append(
            (
                f"single verify L=4, ctx={ctx}",
                arm(stub, mx.zeros((1, 4, 8), dtype=mx.bfloat16), "causal", single(ctx),
                    mx.zeros((1, 4), dtype=mx.int32), True),
            )
        )
    for lengths in ([4096, 6000], [9000, 65536], [9000, 16384, 40000, 65536]):
        cache = batched(lengths)
        batch = len(lengths)
        rows.append(
            (
                f"batch decode B={batch} L=1, ctx={lengths}",
                arm(stub, mx.zeros((batch, 1, 8), dtype=mx.bfloat16),
                    "left_padded_decode", cache, None, False),
            )
        )
        rows.append(
            (
                f"batch verify B={batch} L=4, ctx={lengths}",
                arm(stub, mx.zeros((batch, 4, 8), dtype=mx.bfloat16), None, cache,
                    None, True),
            )
        )
    for key, value in rows:
        print(f"{key:>50s} -> {value}")


def main():
    sweep("stock (no patch)")
    os.environ["OMLX_QSA_BATCHED_SPARSE"] = "1"
    os.environ["OMLX_QSA_BATCHED_VERIFY"] = "1"
    os.environ.setdefault("OMLX_QSA_GATHER_MIN_CTX", "8192")
    spec = importlib.util.spec_from_file_location(
        "_qsab_patch", Path(__file__).resolve().parent / "patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"\ninstall() -> {module.install(None)}")
    sweep(f"patched (OMLX_QSA_GATHER_MIN_CTX={os.environ['OMLX_QSA_GATHER_MIN_CTX']})")


if __name__ == "__main__":
    main()
