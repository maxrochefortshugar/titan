"""Trimmed vendor copy of ``mlx_vlm/models/qwen3_vl/config.py`` (mlx-vlm 0.6.3, MIT).

Local modification (Titan): only the config helpers and the base
``VisionConfig`` / ``TextConfig`` dataclasses that the Qwen3.5 and Qwen4-Exp
configs subclass are kept; the Qwen3-VL ``ModelConfig`` is dropped.
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .base import BaseModelConfig


def _config_kwargs(config_cls, params):
    return {
        k: v for k, v in params.items() if k in inspect.signature(config_cls).parameters
    }


def _has_required_config_fields(config_cls, params):
    return all(
        name in params
        for name, param in inspect.signature(config_cls).parameters.items()
        if param.default is inspect.Parameter.empty
    )


def _maybe_deserialize_config(config_cls, params, *, require_all_fields=False):
    if not isinstance(params, dict):
        return params
    if require_all_fields and not _has_required_config_fields(config_cls, params):
        return params
    return config_cls(**_config_kwargs(config_cls, params))


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_vl"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    window_size: int = 112
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])
    deepstack_visual_indexes: list[int] = field(default_factory=list)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    norm_topk_prob: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default", "mrope_section": [24, 20, 20]}
    )
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            # Normalize rope_scaling keys (accept both 'rope_type' and 'type')
            if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")

            required_keys = {"mrope_section", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if not self.rope_scaling["type"] in ["mrope", "default"]:
                raise ValueError(f"rope_scaling type must be 'mrope' or 'default'")
