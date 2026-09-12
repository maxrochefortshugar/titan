"""Trimmed vendor copy of ``mlx_vlm/models/qwen3_5/qwen3_5.py`` (mlx-vlm 0.6.3, MIT).

Local modification (Titan): only ``sanitize_key`` is kept.  The ``Model``
wrapper it lives beside builds a Qwen3-VL vision tower and is not part of the
language path.
"""


def sanitize_key(key):
    if key.startswith("model.language_model.visual"):
        key = key.replace("model.language_model.visual", "vision_tower", 1)
    elif key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    elif key.startswith("model.visual"):
        key = key.replace("model.visual", "vision_tower", 1)
    elif key.startswith("lm_head"):
        key = key.replace("lm_head", "language_model.lm_head", 1)
    return key
