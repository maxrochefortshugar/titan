"""Trimmed vendor copy of ``mlx_vlm/models/base.py`` (mlx-vlm 0.6.3, MIT).

Local modification (Titan): only the four symbols the Qwen3.8-Flash-Next
language path uses are kept -- ``BaseModelConfig``, ``LanguageModelOutput``,
``InputEmbeddingsFeatures``, ``create_attention_mask``, ``create_ssm_mask`` and
``scaled_dot_product_attention``.  The image-processor classes, the PIL and
transformers imports and the TurboQuant KV-cache branches of
``scaled_dot_product_attention`` are removed: Titan does not run the vision
tower and does not use TurboQuant caches, so the remaining body is the
``mlx_lm`` delegation the stock code falls through to for every KV cache the
text path constructs.
"""

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx

from ...mlx_lm.models.base import create_attention_mask  # noqa: F401
from ...mlx_lm.models.base import create_ssm_mask  # noqa: F401
from ...mlx_lm.models.base import (
    scaled_dot_product_attention as mlx_scaled_dot_product_attention,
)


@dataclass
class LanguageModelOutput:
    logits: mx.array
    hidden_states: Optional[List[mx.array]] = None
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None
    gdn_states: Optional[List] = None
    shared_kv_states: Optional[Dict[str, tuple]] = None


@dataclass
class InputEmbeddingsFeatures:
    inputs_embeds: mx.array
    attention_mask_4d: Optional[mx.array] = None
    per_layer_inputs: Optional[mx.array] = None
    position_ids: Optional[mx.array] = None
    rope_deltas: Optional[mx.array] = None


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    return mlx_scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
