# SPDX-License-Identifier: Apache-2.0
"""Round-3 QSA verify/decode routing patch for qwen4_exp (Flash-Next).

Finding: MTP verify (M = 1 + draft depth, 2..6 rows) ALREADY takes the gathered
sparse QSA arm; see REPORT.md for the file:line proof. Two corners still fall to
dense attention over the whole cache:

  (1) the depth-0 verify cycle, where the MTP controller presents M = 1 rows with
      ``target_verify=True``. ``_gathered_text_decode_eligible`` requires
      ``not target_verify`` (language.py:1330) and ``_gathered_text_verify_eligible``
      requires ``x.shape[1] > 1`` (language.py:1375), so M=1 verify matches neither.

  (2) any request whose position ids arrive as a 3-plane mRoPE block, because
      ``_rank_two_text_position_ids`` (language.py:81-92) accepts only ``(1, L)``.
      Text requests are unaffected: the prefill predicate uses the wider
      ``_batch_one_text_position_ids``. Enable (2) only with a VLM regression run.

Install AFTER the model is loaded (the class must be imported); it is a class
method swap, so it is idempotent and instance-independent.

Env: OMLX_QSA_VERIFY_SPARSE=1        enables (1)
     OMLX_QSA_VERIFY_SPARSE_MROPE=1  additionally enables (2)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MARK = "_omlx_round3_qsa_verify_sparse"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _language_module():
    import sys

    mod = sys.modules.get("mlx_vlm.models.qwen4_exp.language")
    if mod is not None:
        return mod
    try:
        import importlib

        return importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    except Exception:  # noqa: BLE001
        return None


def install_verify_sparse(model=None) -> bool:
    """Route M=1 target_verify rows through the gathered sparse decode arm."""
    if not _truthy("OMLX_QSA_VERIFY_SPARSE"):
        return False
    lang = _language_module()
    if lang is None:
        logger.warning("qsa-verify: qwen4_exp language module not importable")
        return False
    att = getattr(lang, "Qwen4ExpAttention", None)
    if att is None:
        return False
    if getattr(att, _MARK, False):
        return False

    original = att._gathered_text_decode_eligible
    allow_mrope = _truthy("OMLX_QSA_VERIFY_SPARSE_MROPE")
    rank_two = lang._rank_two_text_position_ids
    broadcast_text = lang._broadcast_text_mrope_position_ids

    def _gathered_text_decode_eligible(
        self, x, mask, cache, position_ids, position_embeddings, target_verify
    ):
        if original(
            self, x, mask, cache, position_ids, position_embeddings, target_verify
        ):
            return True
        if not target_verify:
            return False
        # Same predicate, minus the target_verify veto. The gathered decode arm
        # reads no verify-only state: it appends one K/V row and attends the
        # selected blocks, exactly as the M=1 non-verify step does.
        if not (hasattr(x, "ndim") and x.ndim == 3 and x.shape[:2] == (1, 1)):
            return False
        ok_positions = rank_two(position_ids, 1)
        if not ok_positions and allow_mrope:
            ok_positions = broadcast_text(position_ids, 1)
        if not ok_positions:
            return False
        return original(self, x, mask, cache, None, position_embeddings, False)

    att._gathered_text_decode_eligible = _gathered_text_decode_eligible
    setattr(att, _MARK, True)
    logger.info(
        "qsa-verify: M=1 target_verify rows routed to the gathered sparse arm "
        "(mrope=%s)",
        allow_mrope,
    )
    return True


def install(model=None) -> bool:
    return install_verify_sparse(model)
