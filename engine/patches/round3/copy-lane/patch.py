# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup ("n-gram copy") draft lane for the qwen4_exp MTP path.

One env-gated install, ``install_copy_lane()`` (alias ``install()``), gated by
``OMLX_MTP_COPY_LANE=1``. It wraps the module function
``omlx.patches.mlx_lm_mtp.batch_generator._chain_next_drafts`` (stock code at
batch_generator.py:2303-2434) at IMPORT TIME and delegates to whatever was
bound there before, so it composes with the round-2 shortlist drafter and with
the round-3 draft-head and depth workstreams. It must be installed LAST so it
sits outermost and can hand the cycle back to the MTP chain on a miss.

Mechanism. At each decode cycle the drafter is handed ``committed`` (the
tokens the verify pass just confirmed). The lane hashes the last N ids of the
stream tail, looks them up in a dict built once per request from THIS
request's prompt ids, and, on a hit, installs the following up-to-MAX prompt
ids as ``state.drafts`` in place of the MTP chain's drafts. The stock verify
pass (batch_generator.py:2908-3010) already reads ``k = state.drafts.shape[0]``
per cycle, so a copy block is verified, accepted, rolled back and rewound by
exactly the same code as an MTP chain of the same length. Nothing about the
acceptance rule changes: greedy compares argmax ids, and the stochastic path
gets a genuine one-hot proposal density q, which is the standard
Leviathan/Chen setup for a deterministic drafter and stays exact.

Env knobs (defaults in brackets):
  OMLX_MTP_COPY_LANE          [0]  master switch
  OMLX_MTP_COPY_NGRAM         [8]  match length, highest order tried
  OMLX_MTP_COPY_NGRAM_MIN     [=NGRAM]  lowest order tried (descending)
  OMLX_MTP_COPY_MAX           [14] block length cap (see M_LIMIT)
  OMLX_MTP_COPY_M_LIMIT       [15] hard cap on the verify rows M = block + 1
  OMLX_MTP_COPY_MIN_BLOCK     [2]  shorter blocks fall back to the MTP chain
  OMLX_MTP_COPY_SOURCE        [prompt] "prompt" only; "context" is refused
  OMLX_MTP_COPY_ADAPTIVE      [1]  size the block from the accepted-length EMA
  OMLX_MTP_COPY_MIN_ACCEPT    [0.75] disable the lane below this EMA
  OMLX_MTP_COPY_MAX_SAMPLED   [8]  block cap when the target is not greedy
  OMLX_MTP_COPY_STOP_IDS      []   extra comma-separated stop ids
  OMLX_MTP_COPY_MAX_PROMPT    [65536] index at most this many trailing prompt
                                   ids (the table costs ~185 bytes per token)
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BG = "omlx.patches.mlx_lm_mtp.batch_generator"
_NEG_INF = -3.0e38
_INSTALLED = False


def _envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envfloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class _Cfg:
    __slots__ = (
        "ngram", "ngram_min", "max_block", "m_limit", "min_block",
        "adaptive", "min_accept", "max_sampled", "extra_stop", "max_prompt",
    )

    def __init__(self):
        self.ngram = max(1, _envint("OMLX_MTP_COPY_NGRAM", 8))
        self.ngram_min = max(1, min(self.ngram, _envint("OMLX_MTP_COPY_NGRAM_MIN", self.ngram)))
        # M = block + 1 must stay inside the fused verify envelope. The
        # tightest documented one is TurboQuant's decode-shaped multi-row
        # attention, _DECODE_MULTIROW_MAX_Q_LEN = 15 in
        # omlx/patches/turboquant_attention.py:30; above it MTP verify drops
        # into the prefill fallbacks that re-dequantize the whole KV cache.
        self.m_limit = max(2, _envint("OMLX_MTP_COPY_M_LIMIT", 15))
        self.max_block = max(1, min(_envint("OMLX_MTP_COPY_MAX", 14), self.m_limit - 1))
        self.min_block = max(1, _envint("OMLX_MTP_COPY_MIN_BLOCK", 2))
        self.adaptive = _enabled("OMLX_MTP_COPY_ADAPTIVE", "1")
        self.min_accept = _envfloat("OMLX_MTP_COPY_MIN_ACCEPT", 0.75)
        self.max_sampled = max(1, min(_envint("OMLX_MTP_COPY_MAX_SAMPLED", 8), self.max_block))
        self.max_prompt = max(0, _envint("OMLX_MTP_COPY_MAX_PROMPT", 65536))
        self.extra_stop = tuple(
            int(x) for x in os.environ.get("OMLX_MTP_COPY_STOP_IDS", "").replace(" ", "").split(",") if x
        )


# ---------------------------------------------------------------------------
# Per-request prompt index
# ---------------------------------------------------------------------------


class PromptIndex:
    """N-gram -> next position, built once per request from prompt ids only.

    One pass over the prompt at build time, O(1) per decode-cycle lookup: a
    dict from the tuple of the N ids at positions i-N..i-1 to i, most recent
    occurrence winning (later writes overwrite earlier ones), so a rewrite
    that walks the prompt forward keeps matching the copy it is producing.
    Separate maps per order so a descending search is still O(1) per order.
    """

    __slots__ = ("orders", "maps", "prompt", "length")

    def __init__(self, prompt_ids: List[int], orders: Tuple[int, ...]):
        self.prompt = list(prompt_ids)
        self.length = len(self.prompt)
        self.orders = tuple(sorted({int(o) for o in orders if o >= 1}, reverse=True))
        self.maps: Dict[int, Dict[Tuple[int, ...], int]] = {}
        p = self.prompt
        n = self.length
        for order in self.orders:
            table: Dict[Tuple[int, ...], int] = {}
            if n > order:
                # zip over order shifted slices: no per-position Python
                # indexing arithmetic, one C-level pass per order.
                grams = zip(*(p[j : n - order + j] for j in range(order)))
                for i, gram in enumerate(grams):
                    table[gram] = i + order
            self.maps[order] = table

    def lookup(self, tail: Tuple[int, ...], want: int) -> Optional[List[int]]:
        """Longest-order match on the stream tail -> up to ``want`` prompt ids."""
        for order in self.orders:
            if order > len(tail):
                continue
            pos = self.maps[order].get(tail[-order:])
            if pos is None:
                continue
            block = self.prompt[pos : pos + want]
            if block:
                return block
        return None


# ---------------------------------------------------------------------------
# Per-sequence lane state
# ---------------------------------------------------------------------------


class _LaneStats:
    __slots__ = ("cycles", "hits", "installed", "accepted", "blocks", "block_toks",
                 "stop_truncated", "stop_saved", "budget_truncated", "disabled_at")

    def __init__(self):
        self.cycles = 0          # drafter calls seen
        self.hits = 0            # cycles with an n-gram match
        self.installed = 0       # cycles where copy replaced the MTP chain
        self.accepted = 0        # copy draft tokens accepted by verify
        self.blocks = 0          # verified copy blocks
        self.block_toks = 0      # copy draft tokens proposed
        self.stop_truncated = 0
        self.stop_saved = 0   # draft rows dropped at a stop token
        self.budget_truncated = 0
        self.disabled_at = -1


class _Lane:
    __slots__ = ("index", "tail", "prompt_len", "stats", "ema", "pending",
                 "accepts_at_install", "vocab", "neg", "disabled")

    def __init__(self):
        self.index: Optional[PromptIndex] = None
        self.tail: List[int] = []
        self.prompt_len = -1
        self.stats = _LaneStats()
        self.ema = None            # EMA of accepted tokens per copy block
        self.pending = 0           # length of the copy block awaiting verify
        self.accepts_at_install = 0
        self.vocab = 0
        self.neg = None
        self.disabled = False


_EMA_ALPHA = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _language_model(model: Any):
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "mtp_forward"):
            return inner
    return model if hasattr(model, "mtp_forward") else None


def _vocab_size(lm: Any) -> int:
    args = getattr(lm, "args", None)
    v = int(getattr(args, "vocab_size", 0) or 0)
    if v:
        return v
    head = getattr(lm, "lm_head", None)
    w = getattr(head, "weight", None)
    if w is not None:
        return int(w.shape[0])
    emb = getattr(getattr(lm, "model", None), "embed_tokens", None)
    w = getattr(emb, "weight", None)
    return int(w.shape[0]) if w is not None else 0


def _stop_ids(gen_batch: Any, cfg: _Cfg) -> frozenset:
    """Single-token stop ids for this sequence, cached on the row's matcher."""
    ids = set(cfg.extra_stop)
    sm = None
    try:
        sm = gen_batch.state_machines[0]
    except Exception:  # noqa: BLE001
        sm = None
    for attr in ("eos_token_ids", "eos_ids", "stop_ids", "eos_token_id"):
        val = getattr(sm, attr, None)
        if val is None:
            continue
        if isinstance(val, int):
            ids.add(int(val))
        else:
            try:
                ids.update(int(x) for x in val)
            except Exception:  # noqa: BLE001
                pass
    return frozenset(ids)


def _truncate_at_stop(gen_batch: Any, block: List[int], stops: frozenset) -> Tuple[List[int], bool]:
    """Cut the block so its LAST token is the first stop token it contains.

    Hazard (a). A copy block that runs past a stop token would have the
    backbone (and the recurrent GDN session cache behind it) advanced over
    tokens that are never emitted, and the finished request hands that cache
    out for prefix reuse (batch_generator.py:2668 ``prompt_cache=...``). GDN
    state is not trimmable, so the next turn inherits the poison. Ending the
    block AT the stop bounds the over-run to the single bonus row, which is
    exactly stock MTP's worst case.

    Multi-token stop sequences are handled by dry-running the sequence state
    machine, which is a pure ``match(state, id) -> (state, seq, current)``.
    """
    sm = None
    st = None
    try:
        sm = gen_batch.state_machines[0]
        st = gen_batch._matcher_states[0]
    except Exception:  # noqa: BLE001
        sm = None
    for j, tid in enumerate(block):
        if tid in stops:
            return block[: j + 1], True
        if sm is not None and st is not None:
            try:
                st, seq, cur = sm.match(st, int(tid))
            except Exception:  # noqa: BLE001
                sm = None
                continue
            if seq is not None and cur is None:
                return block[: j + 1], True
    return block, False


def _budget_left(gen_batch: Any, state: Any) -> int:
    """Emitted-token budget left, counting the not-yet-drained queue."""
    try:
        queued = len(getattr(state, "queue", ()) or ())
        return int(gen_batch.max_tokens[0]) - int(gen_batch._num_tokens[0]) - queued
    except Exception:  # noqa: BLE001
        return 1 << 30


# ---------------------------------------------------------------------------
# The patched drafter
# ---------------------------------------------------------------------------


def _make_chain_next_drafts(bg, orig, cfg: _Cfg):
    import mlx.core as mx

    orders = tuple(range(cfg.ngram, cfg.ngram_min - 1, -1))

    def _lane(state) -> _Lane:
        lane = getattr(state, "_omlx_copy_lane", None)
        if lane is None:
            lane = _Lane()
            state._omlx_copy_lane = lane
            # _log_mtp_stats only receives ``stats``; give it a way back.
            try:
                state.stats._omlx_copy_lane_ref = lane
            except Exception:  # noqa: BLE001
                pass
        return lane

    def _settle_previous(state, lane: _Lane) -> None:
        """Score the block verified in the cycle that just ran.

        ``state.stats.accepts`` is bumped by the verify pass, so the accepted
        length of the copy block is the delta since it was installed. The
        same loop un-does the copy block's contribution to the per-depth MTP
        histogram, keeping ``depth[d1..dk]`` in the MTP log line a statement
        about the MTP head only (what the depth policy is read from).
        """
        k = lane.pending
        lane.pending = 0
        if k <= 0:
            return
        m = max(0, min(k, int(state.stats.accepts) - lane.accepts_at_install))
        lane.stats.blocks += 1
        lane.stats.accepted += m
        lane.ema = float(m) if lane.ema is None else (1 - _EMA_ALPHA) * lane.ema + _EMA_ALPHA * m
        # Mirror the verify pass's own loop exactly (batch_generator.py:3086-
        # 3092): it breaks at the first rejected position, so it touched
        # depth_drafted[0..min(m, k-1)] and depth_accepted[0..m-1].
        dd = state.stats.depth_drafted
        da = state.stats.depth_accepted
        for j in range(min(m + 1, k, len(dd))):
            dd[j] -= 1
        for j in range(min(m, len(da))):
            da[j] -= 1
        if cfg.min_accept > 0 and lane.stats.blocks >= 8 and lane.ema < cfg.min_accept:
            lane.disabled = True
            lane.stats.disabled_at = lane.stats.cycles

    def _ensure_index(gen_batch, state, lane: _Lane) -> bool:
        if lane.index is not None:
            return lane.index.length > cfg.ngram_min
        try:
            toks = gen_batch.tokens[0]
            generated = int(gen_batch._num_tokens[0])
        except Exception:  # noqa: BLE001
            return False
        prompt_len = len(toks) - generated
        if prompt_len <= cfg.ngram_min:
            lane.index = PromptIndex([], orders)
            return False
        # Prompt-sliced, never generated output: everything at or after
        # prompt_len was produced by this request and is excluded.
        lane.prompt_len = prompt_len
        # Index the trailing window: the table costs roughly 185 bytes per
        # indexed token, and an edit target lives at the end of the prompt.
        lo = max(0, prompt_len - cfg.max_prompt) if cfg.max_prompt else 0
        lane.index = PromptIndex(list(toks[lo:prompt_len]), orders)
        lane.tail = list(toks[max(0, prompt_len - cfg.ngram) : prompt_len])
        return True

    def _tail_after(lane: _Lane, gen_batch, committed_ids: List[int]) -> Tuple[int, ...]:
        """Stream tail = emitted tokens + this cycle's committed tokens.

        ``gen_batch.tokens`` only grows when the queue is drained, so the
        cycle's own committed ids are appended here.
        """
        toks = gen_batch.tokens[0]
        want = cfg.ngram
        base = list(toks[-want:]) if want else []
        tail = base + list(committed_ids)
        return tuple(tail[-want:]) if want else tuple()

    def patched(gen_batch, state, hidden_rows, committed, prev_buf):
        lane = _lane(state)
        ctrl = getattr(state, "controller", None)
        was_copy = lane.pending > 0
        if ctrl is not None:
            # Read by the observe() guard installed below: the cycle that is
            # about to be scored verified a copy block, so its wall time and
            # acceptance say nothing about MTP depth.
            ctrl._omlx_copy_active = was_copy
        _settle_previous(state, lane)
        lane.stats.cycles += 1

        if not getattr(state, "chain", False) or lane.disabled:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)
        # Never speculate on a copy while the depth controller is still
        # measuring its per-depth costs: the skipped observe() would stall
        # the warmup sweep and the depth policy would never calibrate.
        if ctrl is not None and getattr(ctrl, "_warmup", None):
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)
        if not _ensure_index(gen_batch, state, lane):
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        try:
            committed_ids = [int(x) for x in committed.tolist()]
        except Exception:  # noqa: BLE001
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        greedy = bg._is_greedy(gen_batch)
        want = cfg.max_block if greedy else cfg.max_sampled
        if cfg.adaptive and lane.ema is not None:
            want = min(want, max(cfg.min_block, int(lane.ema * 1.5) + 2))
        budget = _budget_left(gen_batch, state)
        want = min(want, max(1, budget - 1))

        tail = _tail_after(lane, gen_batch, committed_ids)
        block = lane.index.lookup(tail, want) if len(tail) >= cfg.ngram_min else None
        if not block:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)
        lane.stats.hits += 1

        n_before = len(block)
        block, cut = _truncate_at_stop(gen_batch, block, _stop_ids(gen_batch, cfg))
        if cut:
            lane.stats.stop_truncated += 1
            lane.stats.stop_saved += n_before - len(block)
        if len(block) > budget:
            block = block[:budget]
            lane.stats.budget_truncated += 1
        if len(block) < cfg.min_block:
            return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        # --- install the copy block in place of the MTP chain ---
        if not lane.vocab:
            lm = _language_model(gen_batch.model)
            lane.vocab = _vocab_size(lm) if lm is not None else 0
            if not lane.vocab:
                lane.disabled = True
                logger.warning(
                    "MTP copy lane disabled: vocabulary size not discoverable "
                    "on %s", type(gen_batch.model).__name__)
                return orig(gen_batch, state, hidden_rows, committed, prev_buf)

        k = len(block)
        ids = mx.array(block, dtype=mx.uint32)
        state.drafts = ids
        mx.async_eval(state.drafts)
        # Proposal density of a deterministic drafter: a one-hot row per
        # position. Built as one lazy (k, V) array so the stochastic path's
        # mx.stack costs one concatenate, and so a greedy request (which
        # never touches these) evaluates none of it.
        q = mx.put_along_axis(
            mx.full((k, lane.vocab), _NEG_INF, dtype=mx.float32),
            ids.astype(mx.int32)[:, None],
            mx.zeros((k, 1), dtype=mx.float32),
            axis=-1,
        )
        rows = [q[j] for j in range(k)]
        state.draft_lps = rows
        state.draft_accept_lps = rows

        # The verify pass indexes stats.depth_drafted[j] for j < k after
        # padding only to state.depth, so a block longer than the MTP depth
        # needs the histogram extended here (undone in _settle_previous).
        dd, da = state.stats.depth_drafted, state.stats.depth_accepted
        if len(dd) < k:
            dd.extend([0] * (k - len(dd)))
            da.extend([0] * (k - len(da)))

        # Hazard (c): keep the MTP head's history exactly as the stock chain
        # would leave it, so the next MTP cycle folds from the same timeline.
        # Only the head layer is evaluated; logits_keep=1's lm_head projection
        # is dead in the lazy graph because nothing consumes it.
        lm = _language_model(gen_batch.model)
        rows_in = hidden_rows
        head_prenorm = getattr(gen_batch.model, "_omlx_mtp_head_prenorm", False) or getattr(
            getattr(gen_batch.model, "_language_model", None), "_omlx_mtp_head_prenorm", False
        )
        if bg._HEAD_HIDDEN_POST_NORM and not head_prenorm and rows_in.ndim == 3:
            rows_in = bg._trunk_norm_module(gen_batch.model)(rows_in)
        begin = getattr(gen_batch.model, "mtp_begin_cycle", None) or getattr(
            getattr(gen_batch.model, "_language_model", None), "mtp_begin_cycle", None
        )
        if begin is not None:
            begin(state.mtp_cache, 0)
        n = int(committed.shape[0])
        lm.mtp_forward(
            rows_in,
            committed.reshape(1, n),
            state.mtp_cache,
            return_hidden=True,
            logits_keep=1,
        )
        state.hist_offset += n

        lane.pending = k
        lane.accepts_at_install = int(state.stats.accepts)
        lane.stats.installed += 1
        lane.stats.block_toks += k
        return None

    patched._omlx_mtp_copy_lane = True
    return patched


# ---------------------------------------------------------------------------
# Depth-controller guard and end-of-request logging
# ---------------------------------------------------------------------------


def _install_controller_guard(bg) -> bool:
    """Hide copy cycles from the MTP depth controller.

    ``observe(used, accepted, cycle_ms)`` clamps ``used`` to ``max_depth``, so
    a 14-token copy block would be recorded as a full-accept depth-3 cycle at
    roughly twice the wall time, corrupting both ``p`` and ``t`` and, through
    them, the depth policy. Copy cycles are skipped entirely.
    """
    ctrl = None
    for name in dir(bg):
        obj = getattr(bg, name)
        if isinstance(obj, type) and hasattr(obj, "_score") and hasattr(obj, "observe"):
            ctrl = obj
            break
    if ctrl is None:
        return False
    if getattr(ctrl, "_omlx_copy_guard", False):
        return True
    orig_observe = ctrl.observe

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        if getattr(self, "_omlx_copy_active", False):
            self._omlx_copy_active = False
            return
        return orig_observe(self, used, accepted, cycle_ms, time_sample=time_sample)

    ctrl.observe = observe
    ctrl._omlx_copy_guard = True
    return True


def _lane_stats(lane: Any) -> Optional[dict]:
    if lane is None:
        return None
    s = lane.stats
    return {
        "cycles": s.cycles,
        "hits": s.hits,
        "installed": s.installed,
        "blocks": s.blocks,
        "block_toks": s.block_toks,
        "accepted": s.accepted,
        "hit_rate": (s.hits / s.cycles) if s.cycles else 0.0,
        "mean_block": (s.block_toks / s.blocks) if s.blocks else 0.0,
        "mean_accept": (s.accepted / s.blocks) if s.blocks else 0.0,
        "stop_truncated": s.stop_truncated,
        "stop_saved": s.stop_saved,
        "budget_truncated": s.budget_truncated,
        "prompt_len": lane.prompt_len,
        "disabled_at": s.disabled_at,
    }


def copy_lane_stats(state: Any) -> Optional[dict]:
    """Copy-lane counters for one sequence (introspection / tests)."""
    return _lane_stats(getattr(state, "_omlx_copy_lane", None))


def _install_stats_log(bg) -> bool:
    """Append a COPY[...] line next to the existing MTP[...] request summary."""
    orig = getattr(bg, "_log_mtp_stats", None)
    if orig is None or getattr(orig, "_omlx_copy_log", False):
        return orig is not None
    orig_drop = getattr(bg, "_drop_mtp_state", None)

    def log_stats(uid, stats, finish_reason):
        orig(uid, stats, finish_reason)
        lane = getattr(stats, "_omlx_copy_lane_ref", None)
        st = _lane_stats(lane) if lane is not None else None
        if st and st["cycles"]:
            logger.info(
                "COPY[%s] cycles=%d hit=%d (%.1f%%) installed=%d blocks=%d "
                "proposed=%d accepted=%d mean_block=%.2f mean_accept=%.2f "
                "stop_cut=%d(-%d rows) budget_cut=%d prompt=%d%s",
                uid, st["cycles"], st["hits"], 100.0 * st["hit_rate"],
                st["installed"], st["blocks"], st["block_toks"], st["accepted"],
                st["mean_block"], st["mean_accept"], st["stop_truncated"],
                st["stop_saved"], st["budget_truncated"], st["prompt_len"],
                "" if st["disabled_at"] < 0 else f" disabled@{st['disabled_at']}",
            )

    log_stats._omlx_copy_log = True
    bg._log_mtp_stats = log_stats
    _ = orig_drop
    return True


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install_copy_lane(model: Any = None) -> bool:
    """Wrap the MTP drafter with the prompt-lookup lane. Install at IMPORT
    TIME, and AFTER every other ``_chain_next_drafts`` patch, so this wrapper
    is outermost and can delegate to the MTP chain on a miss."""
    global _INSTALLED
    if not _enabled("OMLX_MTP_COPY_LANE"):
        return False
    if _INSTALLED:
        return True
    source = os.environ.get("OMLX_MTP_COPY_SOURCE", "prompt").strip().lower()
    if source not in ("", "prompt"):
        logger.warning(
            "OMLX_MTP_COPY_SOURCE=%s refused: only prompt-sliced lookup is "
            "supported (generated output must never be a copy source)", source
        )
        return False
    try:
        bg = importlib.import_module(_BG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP copy lane: %s unavailable (%s)", _BG, exc)
        return False
    orig = getattr(bg, "_chain_next_drafts", None)
    if orig is None:
        return False
    if getattr(orig, "_omlx_mtp_copy_lane", False):
        _INSTALLED = True
        return True
    cfg = _Cfg()
    try:
        bg._chain_next_drafts = _make_chain_next_drafts(bg, orig, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP copy lane not installed: %s", exc)
        return False
    bg._omlx_copy_lane_orig = orig
    _install_controller_guard(bg)
    _install_stats_log(bg)
    _INSTALLED = True
    logger.info(
        "MTP copy lane installed (ngram=%d..%d max_block=%d M<=%d min_block=%d "
        "adaptive=%s min_accept=%.2f, inner drafter=%s)",
        cfg.ngram, cfg.ngram_min, cfg.max_block, cfg.max_block + 1,
        cfg.min_block, cfg.adaptive, cfg.min_accept,
        getattr(orig, "__name__", type(orig).__name__),
    )
    return True


install = install_copy_lane

__all__ = ["install", "install_copy_lane", "copy_lane_stats", "PromptIndex"]
