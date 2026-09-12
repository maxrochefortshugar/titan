# SPDX-License-Identifier: Apache-2.0
"""Round-3 batched sparse QSA arm for qwen4_exp (Flash-Next).

Two changes, both env gated.

1. ``OMLX_QSA_BATCHED_SPARSE=1`` adds a gathered sparse arm for
   ``BatchQSAKVCache``. Every stock gathered predicate tests
   ``type(cache) is QSAKVCache`` (language.py:1292, 1327, 1376), so joining the
   decode batch (scheduler.py:1015 -> ``QSAKVCache.to_batch`` ->
   ``BatchQSAKVCache``, language.py:562) drops all three sparse arms and decode
   reads the whole cache densely. This installs a batched decode arm and, under
   ``OMLX_QSA_BATCHED_VERIFY=1``, a batched verify arm (B sequences x L rows).

2. ``OMLX_QSA_GATHER_MIN_CTX`` moves the gathered decode/verify engagement
   point off the stock ``cache.offset + L > token_budget`` (2048) to the
   measured crossover, default 8192. Below the crossover the gathered arm is
   slower than dense masked SDPA even though it reads less.

Install AFTER the model is loaded: it swaps methods on ``Qwen4ExpAttention``,
so it is idempotent and instance independent. Compose it AFTER
round3/qsa-verify, whose patch wraps ``_gathered_text_decode_eligible``; the
threshold wrapper here has to sit outermost to constrain it.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_MARK = "_omlx_round3_qsa_batched"
_MARK_CTX = "_omlx_round3_qsa_min_ctx"


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_sibling(name: str):
    """Import a module next to this file without relying on sys.path."""
    existing = sys.modules.get(f"_omlx_qsa_batched_{name}")
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_omlx_qsa_batched_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _language_module():
    module = sys.modules.get("mlx_vlm.models.qwen4_exp.language")
    if module is not None:
        return module
    try:
        import importlib as _importlib

        return _importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# position ids and mask validation
# --------------------------------------------------------------------------


def _batch_positions(cache, position_ids, geo, length):
    """Return ``(text [B, L], rotary)`` or ``None`` if the ids are not text ids.

    Both the derived-position array and the three-plane equality answer are
    memoized, because a host round trip per QSA layer per step would cost more
    than the arm saves.
    """
    batch = geo.batch
    if position_ids is None:
        pads = getattr(cache, "_qsab_pads_array", None)
        if pads is None or pads.shape[0] != batch:
            pads = mx.array(geo.pads, dtype=mx.int32)[:, None]
            mx.eval(pads)
            cache._qsab_pads_array = pads
        text = (geo.width - pads) + mx.arange(length, dtype=mx.int32)[None, :]
        return text, mx.broadcast_to(text[None], (3, batch, length))
    if not isinstance(position_ids, mx.array):
        return None
    if position_ids.ndim == 2:
        if tuple(position_ids.shape) != (batch, length):
            return None
        return position_ids, position_ids
    if position_ids.ndim != 3 or tuple(position_ids.shape) != (3, batch, length):
        return None
    memo = getattr(cache, "_qsab_position_memo", None)
    if memo is not None and memo[0] is position_ids:
        same = memo[1]
    else:
        same = bool(
            mx.array_equal(position_ids[0], position_ids[1]).item()
            and mx.array_equal(position_ids[1], position_ids[2]).item()
        )
        cache._qsab_position_memo = (position_ids, same)
    if not same:
        return None
    return position_ids[0], position_ids


def _mask_is_canonical(cache, mask, length) -> bool:
    """True when ``mask`` is exactly the batch's own causal/left-padded mask.

    A batched verify arrives with a materialized ``[B, 1, L, T]`` bool mask.
    The gathered arm encodes causality and left padding structurally, so it may
    only run when the incoming mask says the same thing and nothing else. The
    result is memoized by mask identity: the model builds one mask per forward
    and hands the same object to all twelve QSA layers.
    """
    if mask is None:
        return True
    if isinstance(mask, str):
        return mask in {"causal", "left_padded_decode"}
    if not isinstance(mask, mx.array):
        return False
    memo = getattr(cache, "_qsab_mask_memo", None)
    if memo is not None and memo[0] is mask and memo[1] == length:
        return memo[2]
    try:
        canonical = cache.make_mask(length, return_array=True)
    except Exception:  # noqa: BLE001
        canonical = None
    ok = False
    if isinstance(canonical, mx.array) and canonical.dtype == mask.dtype:
        width = min(canonical.shape[-1], mask.shape[-1])
        ok = bool(
            canonical.shape[-2] == mask.shape[-2]
            and mx.array_equal(
                canonical[..., :width], mask[..., :width]
            ).item()
        )
    cache._qsab_mask_memo = (mask, length, ok)
    return ok


# --------------------------------------------------------------------------
# the batched arm
# --------------------------------------------------------------------------


def _make_batched_arm(lang, bq):
    _target_verify_linear = lang._target_verify_linear
    _target_verify_linears = lang._target_verify_linears
    BatchQSAKVCache = lang.BatchQSAKVCache

    def batched_eligible(self, x, mask, cache, position_ids, position_embeddings,
                         target_verify):
        """Fail closed outside batched text decode/verify on a warm QSA batch."""
        if type(cache) is not BatchQSAKVCache:
            return None
        if position_embeddings is not None or x.ndim != 3:
            return None
        batch, length, _ = x.shape
        if length > 1 and not _truthy("OMLX_QSA_BATCHED_VERIFY"):
            return None
        geo = bq.batch_geometry(cache)
        if geo is None or geo.batch != batch:
            return None
        if cache.index_keys is None or cache.index_position_ids is None:
            return None
        if (
            cache.index_offset != geo.width
            or cache.index_keys.shape[1] < geo.width
            or cache.index_position_ids.shape[-1] < geo.width
        ):
            return None
        floor = bq.gather_min_context(self.indexer.token_budget)
        # Every row must clear the crossover; a mixed batch stays dense rather
        # than paying the gathered arm on its short rows.
        if min(geo.lengths) + length <= floor:
            return None
        if not _mask_is_canonical(cache, mask, length):
            return None
        positions = _batch_positions(cache, position_ids, geo, length)
        if positions is None:
            return None
        return geo, positions

    def batched_gathered_attention(self, x, cache, positions, target_verify):
        """Batched projections and RoPE, sparse gathered attention, batched out."""
        batch, length, _ = x.shape
        text_position_ids, rotary_position_ids = positions
        q_proj_output, keys, values = _target_verify_linears(
            (self.q_proj, self.k_proj, self.v_proj), x, target_verify
        )
        queries, gate = mx.split(
            q_proj_output.reshape(batch, length, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch, length, self.num_key_value_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        queries, keys = self.rotary_emb.apply_rotary(
            queries, keys, rotary_position_ids, unsqueeze_dim=1
        )
        keys, values = cache.update_and_fetch(keys, values)

        indexer = self.indexer
        projected = _target_verify_linear(
            indexer.index_qk_proj, x, target_verify
        ).reshape(
            batch, length, indexer.n_heads + indexer.kv_heads, indexer.head_dim
        )
        index_queries = indexer.q_layernorm(
            projected[:, :, : indexer.n_heads]
        ).transpose(0, 2, 1, 3)
        raw_index_keys = projected[:, :, indexer.n_heads :].squeeze(2)
        cache.update_indexer(raw_index_keys, text_position_ids)

        geo = bq.batch_geometry(cache)
        pooled, bank = bq.batched_pooled_index_keys(
            cache,
            geo,
            compress_ratio=indexer.compress_ratio,
            index_key_norm=indexer.k_layernorm,
            apply_index_rope=indexer._apply_rope,
            cache_tag=indexer,
        )
        index_queries = indexer._apply_rope(
            index_queries, text_position_ids
        ).transpose(0, 2, 1, 3)

        output = bq.gathered_qsa_batched(
            queries,
            keys,
            values,
            index_queries,
            pooled,
            bank,
            geo,
            num_query_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            indexer_head_dim=indexer.head_dim,
            compress_ratio=indexer.compress_ratio,
            token_budget=indexer.token_budget,
            index_keys=cache.index_keys,
            index_position_ids=cache.index_position_ids,
        )
        output = output.reshape(batch, length, -1)
        return _target_verify_linear(
            self.o_proj, output * mx.sigmoid(gate), target_verify
        )

    return batched_eligible, batched_gathered_attention


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


def _install_min_ctx(lang, bq) -> bool:
    """AND the configurable crossover onto the stock decode/verify predicates."""
    att = lang.Qwen4ExpAttention
    if getattr(att, _MARK_CTX, False):
        return False
    for name in ("_gathered_text_decode_eligible", "_gathered_text_verify_eligible"):
        original = getattr(att, name)

        def wrapper(self, x, mask, cache, position_ids, position_embeddings,
                    target_verify, _original=original):
            if not _original(
                self, x, mask, cache, position_ids, position_embeddings,
                target_verify,
            ):
                return False
            offset = getattr(cache, "offset", 0)
            if not isinstance(offset, int):
                return True
            return offset + x.shape[1] > bq.gather_min_context(
                self.indexer.token_budget
            )

        setattr(att, name, wrapper)
    setattr(att, _MARK_CTX, True)
    return True


def _warn_quantized_qsa(lang) -> None:
    """Name the latent hazard rather than letting it fail silently."""
    quantized = getattr(lang, "QSAQuantizedKVCache", None)
    if quantized is None or getattr(quantized, "_qsab_warned", False):
        return
    quantized._qsab_warned = True
    logger.warning(
        "qsa-batched: QSAQuantizedKVCache still fails the strict "
        "'type(cache) is QSAKVCache' tests at language.py:1292/1327/1376, so "
        "enabling KV quantization silently drops every QSA sparse arm. The "
        "gathered kernels read raw K/V rows and cannot consume the quantized "
        "cache, so widening those tests would be wrong; the fix is a "
        "dequantizing gather, out of scope for this workstream."
    )


def install(model=None) -> bool:
    """Post-load install. Returns False and leaves the stock path untouched."""
    lang = _language_module()
    if lang is None:
        logger.warning("qsa-batched: qwen4_exp language module not importable")
        return False
    att = getattr(lang, "Qwen4ExpAttention", None)
    if att is None or getattr(lang, "BatchQSAKVCache", None) is None:
        return False

    bq = _load_sibling("batched_qsa")
    installed = False

    if os.environ.get("OMLX_QSA_GATHER_MIN_CTX", "").strip():
        installed |= _install_min_ctx(lang, bq)

    if not _truthy("OMLX_QSA_BATCHED_SPARSE"):
        return installed

    _warn_quantized_qsa(lang)
    if getattr(att, _MARK, False):
        return installed

    # Default the crossover on whenever the batched arm is on: the batched arm
    # inherits the same threshold, and leaving decode at 2048 would keep the
    # single-sequence regression the batch is meant to fix.
    installed |= _install_min_ctx(lang, bq)

    eligible, forward = _make_batched_arm(lang, bq)
    original_call = att.__call__

    def __call__(self, x, mask=None, cache=None, position_ids=None,
                 position_embeddings=None, target_verify=False):
        decision = eligible(
            self, x, mask, cache, position_ids, position_embeddings, target_verify
        )
        if decision is not None:
            _geo, positions = decision
            return forward(self, x, cache, positions, target_verify)
        return original_call(
            self, x, mask, cache, position_ids, position_embeddings, target_verify
        )

    att.__call__ = __call__
    att._omlx_batched_eligible = eligible
    att._omlx_batched_gathered_attention = forward
    setattr(att, _MARK, True)
    logger.info(
        "qsa-batched: sparse gathered arm installed for BatchQSAKVCache "
        "(route=%s, min_ctx=%d, verify=%s)",
        bq.batched_route(),
        bq.gather_min_context(2048),
        _truthy("OMLX_QSA_BATCHED_VERIFY"),
    )
    return True
