#!/usr/bin/env python3
"""Synthetic proof for the fused batched MTP verify (OMLX_MTP_BATCHED=1).

A miniature stand-in for oMLX's batch_generator is registered under the real
module name, so patch.py swaps the SAME symbols it swaps in production. Every
piece of oMLX logic that the fused cycle has to stay compatible with is lifted
verbatim out of the shipped file and exec'd here:

    _MtpStats, _MtpState, _MtpBatchState, _MtpStepFallback,
    _DepthController, _make_row_batch, _chain_next_drafts,
    _chain_rollback, _run_verify_cycle_chain, _emit_batch_responses

so the reference "each request alone" stream is produced by oMLX's real
singleton verify cycle, and the fused stream by patch.py.

The fake model is a 256-token toy whose next token depends on a recurrent
accumulator carried in the cache (the GDN analogue) and on a rolling history
window (the PLE analogue). Both are restored per row by a batch-aware
``rollback_speculative_cache`` that mirrors the real one's contract
(per-row accepted list, uniform block_size). A wrong rollback on any row
therefore changes that row's tokens, which is exactly what check [2] looks
for. Each row gets its own drafter skill, so acceptance differs across rows
within a cycle.

Checks

  [0] plumbing    install swaps the three module functions; the late-arrival
                  pin (_generation_batch_has_active_mtp) is lifted
  [1] identity    B=2 and B=4 fused greedy streams are token-identical to the
                  same rows run alone through oMLX's real singleton cycle
  [2] rollback    per-row recurrent state, history window and cache offset
                  after every cycle match the alone-run, with acceptance
                  differing across rows in the same cycle
  [3] isolation   no cross-row leakage: head caches, hist_offset, PLE history,
                  GDN accumulator and per-row depth-controller stats all match
                  the alone-run row by row
  [4] fallback    non-greedy rows, a row mid-queue, and B above the cap all
                  route back to the stock row-wise loop, and the stock loop
                  still produces the identical stream

Run: ~/inference-server/kdev/bin/python test_batched.py
"""

import importlib.util
import math
import os
import re
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple  # noqa: F401

import mlx.core as mx

V, H = 256, 32
BG_NAME = "omlx.patches.mlx_lm_mtp.batch_generator"
HERE = os.path.dirname(os.path.abspath(__file__))
OMLX_BG = (
    "/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_lm_mtp/"
    "batch_generator.py"
)

FAIL = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# lift real oMLX blocks
# ---------------------------------------------------------------------------
_SRC = open(OMLX_BG).read()


def lift(kind, name):
    pat = rf"^(?:@[\w.]+\n)*{kind} {re.escape(name)}\b.*?(?=^(?:@|def |class |# ---))"
    m = re.search(pat, _SRC, re.S | re.M)
    if not m:
        raise RuntimeError(f"could not lift {kind} {name}")
    return m.group(0)


LIFTED = [
    ("class", "_MtpStepFallback"),
    ("class", "_MtpStats"),
    ("class", "_MtpState"),
    ("class", "_MtpBatchState"),
    ("class", "_DepthController"),
    ("def", "_row_value"),
    ("def", "_make_row_batch"),
    ("def", "_chain_next_drafts"),
    ("def", "_chain_rollback"),
    ("def", "_run_verify_cycle_chain"),
    ("def", "_emit_batch_responses"),
    ("def", "_mtp_batch_next"),
]


# ---------------------------------------------------------------------------
# the toy world
# ---------------------------------------------------------------------------
HIST = 4  # PLE-style rolling window length


def _mix(*vals):
    h = 1469598103
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 1099511628) & 0xFFFFFFFF
    return h


def true_step(tok, acc, hist):
    """One committed token: advance the accumulator and the history window."""
    acc2 = (acc * 3 + int(tok) + 1) % 97
    hist2 = (hist + [int(tok)])[-HIST:]
    return acc2, hist2


def true_next(tok, acc, hist):
    """The target's argmax after consuming ``tok`` from state (acc, hist)."""
    acc2, hist2 = true_step(tok, acc, hist)
    return (int(tok) * 7 + 13 + acc2 * 5 + sum(hist2)) % V, acc2, hist2


class ToyCache:
    """Batched target cache: per-row offset, GDN accumulator, PLE window."""

    def __init__(self, B):
        self.B = B
        self.offsets = [0] * B
        self.acc = [0] * B
        self.hist = [[] for _ in range(B)]
        self.rollback_state = None

    @property
    def offset(self):
        return max(self.offsets)

    def clone_row(self, b):
        c = ToyCache(1)
        c.offsets = [self.offsets[b]]
        c.acc = [self.acc[b]]
        c.hist = [list(self.hist[b])]
        return c

    def snapshot_row(self, b):
        return (self.offsets[b], self.acc[b], tuple(self.hist[b]))


class ToyHeadCache:
    """MTP head cache: committed length plus the head's own state replica."""

    def __init__(self, seed=0, skill=100):
        self.seed = seed
        self.skill = skill
        self.n = 0
        self.acc = 0
        self.hist = []
        self.trail = []  # (acc, hist) after each appended entry

    def append(self, tok):
        self.acc, self.hist = true_step(tok, self.acc, self.hist)
        self.trail.append((self.acc, list(self.hist)))
        self.n += 1

    def trim_to(self, n):
        while self.n > n:
            self.trail.pop()
            self.n -= 1
        if self.trail:
            self.acc, self.hist = self.trail[-1][0], list(self.trail[-1][1])
        else:
            self.acc, self.hist = 0, []

    def snapshot(self):
        return (self.n, self.acc, tuple(self.hist))


class ToyModel:
    """Batched backbone + a per-row MTP head with a controllable hit rate."""

    _uses_mrope = False
    _omlx_mtp_commit_align = 0

    def __init__(self):
        self.skill = {}  # id(head_cache) -> hit percentage
        self.calls = []  # (kind, rows, cols)

    # -- backbone ------------------------------------------------------
    def __call__(self, inputs, cache=None, return_hidden=True, n_confirmed=0):
        c = cache[0]
        B, M = inputs.shape
        assert B == c.B, f"cache batch {c.B} != input batch {B}"
        self.calls.append(("backbone", B, M))
        toks = inputs.tolist()
        preds = []
        gdn = []
        for b in range(B):
            acc, hist = c.acc[b], list(c.hist[b])
            trail = [(acc, list(hist))]
            row = []
            for j in range(M):
                p, acc, hist = true_next(toks[b][j], acc, hist)
                row.append(p)
                trail.append((acc, list(hist)))
            preds.append(row)
            gdn.append({"base_offset": c.offsets[b], "trail": trail})
            c.acc[b], c.hist[b] = acc, hist
            c.offsets[b] += M
        logits = mx.zeros((B, M, V))
        idx = mx.array(preds, dtype=mx.int32)
        logits = mx.put_along_axis(
            logits, idx[..., None], mx.full((B, M, 1), 10.0), axis=-1
        )
        hidden = mx.zeros((B, M, H)) + idx.astype(mx.float32)[..., None]
        return logits, hidden, gdn

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        acc_list = (
            [int(accepted)] if isinstance(accepted, int) else [int(a) for a in accepted]
        )
        c = caches[0]
        if len(acc_list) == 1:
            acc_list = acc_list * c.B
        assert len(acc_list) == c.B, "accepted list must cover the batch"
        assert len(gdn_states) == c.B
        for b, a in enumerate(acc_list):
            assert 0 <= a < block_size
            st = gdn_states[b]
            keep = a + 1
            c.acc[b], c.hist[b] = st["trail"][keep][0], list(st["trail"][keep][1])
            c.offsets[b] = st["base_offset"] + keep
        return len(acc_list)

    # -- MTP head ------------------------------------------------------
    def mtp_forward(
        self, hidden_rows, committed, cache, return_hidden=True, logits_keep=None
    ):
        hc = cache[0]
        n = committed.shape[1]
        self.calls.append(("head", 1, n))
        toks = committed.reshape(-1).tolist()
        for t in toks[:-1]:
            hc.append(int(t))
        acc_prev, hist_prev = hc.acc, list(hc.hist)
        last = int(toks[-1])
        hc.append(last)
        pred, _, _ = true_next(last, acc_prev, hist_prev)
        # deterministic corruption keyed by the row seed and the head's own
        # committed position, so the alone-run and the fused run corrupt at
        # exactly the same places and the drafts are comparable
        if _mix(hc.seed, hc.n) % 100 >= hc.skill:
            pred = (pred + 37) % V
        keep = 1 if logits_keep else n
        logits = mx.zeros((1, keep, V))
        logits = mx.put_along_axis(
            logits,
            mx.full((1, keep, 1), pred, dtype=mx.int32),
            mx.full((1, keep, 1), 10.0),
            axis=-1,
        )
        head_hidden = mx.zeros((1, n, H)) + float(pred)
        return logits, head_hidden


# ---------------------------------------------------------------------------
# fake batch_generator
# ---------------------------------------------------------------------------
def install_fake_bg():
    for pkg in ("omlx", "omlx.patches", "omlx.patches.mlx_lm_mtp"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod
    bg = types.ModuleType(BG_NAME)
    ns = bg.__dict__
    ns.update(
        {
            "mx": mx,
            "math": math,
            "os": os,
            "time": __import__("time"),
            "logger": SimpleNamespace(
                debug=lambda *a, **k: None,
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
            ),
            "dataclass": dataclass,
            "field": field,
            "deque": deque,
            "SimpleNamespace": SimpleNamespace,
            "Any": Any,
            "Deque": Deque,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "_STD_TAX_MAX": 1.5,
            "_HEAD_HIDDEN_POST_NORM": False,
        }
    )
    for kind, name in LIFTED:
        exec(compile(lift(kind, name), OMLX_BG, "exec"), ns)

    # stubs for everything the lifted blocks reach for
    ns["_dspark_host"] = lambda model: None
    ns["_proc_list"] = lambda gb: getattr(gb, "_procs", None)
    ns["_is_greedy"] = lambda gb: getattr(gb, "_greedy", True)
    ns["_resolve_sampler"] = lambda gb: (lambda lp: mx.argmax(lp, axis=-1))
    ns["_resolve_draft_sampler"] = lambda gb, st: (lambda lp: mx.argmax(lp, axis=-1))
    ns["_trunk_norm_module"] = lambda model: (lambda x: x)
    ns["_clone_mtp_head_cache"] = lambda c: c
    ns["_snap_snapshotable"] = lambda procs: None
    ns["_restore_snapshotable"] = lambda procs, snap: None
    ns["_apply_processors"] = lambda procs, prev, lg: lg
    ns["_logprobs"] = lambda lg: lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    ns["_ensure_uint32"] = lambda a: a.astype(mx.uint32)
    ns["_accept_lp_for"] = lambda sampler, lp: lp
    ns["_trim_token_buffer"] = lambda gb, n: None
    ns["_bump_emit_stat"] = _bump_emit_stat
    ns["_log_mtp_stats"] = lambda uid, stats, reason: None
    ns["_materialize_mtp_boundary_emit"] = lambda gb, st: None
    ns["_mtp_head_trim_to"] = lambda cache, off: cache[0].trim_to(off)
    ns["_clear_rollback"] = lambda cache: None
    ns["_set_singleton_mrope_delta"] = lambda gb: None
    ns["_replace_cache_rows"] = lambda gb, repl: None
    ns["_run_verify_cycle"] = lambda gb, st: ns["_run_verify_cycle_chain"](gb, st)
    ns["_rowwise_batch_mtp_enabled"] = lambda: False
    ns["_generation_batch_has_active_mtp"] = lambda gb: gb is not None
    ns["_prefill_activity_recent"] = lambda: False
    ns["_effective_loop_tax"] = lambda model: None

    def _call_backbone(model, inputs, cache, n_confirmed=0):
        return model(inputs, cache=cache, return_hidden=True,
                     n_confirmed=n_confirmed)

    ns["_call_backbone"] = _call_backbone

    sys.modules[BG_NAME] = bg
    return bg


def _bump_emit_stat(state, source):
    key = {"init": "emit_init", "draft": "emit_draft", "bonus": "emit_bonus",
           "verify": "emit_verify"}.get(source)
    if key and hasattr(state.stats, key):
        setattr(state.stats, key, getattr(state.stats, key) + 1)


# ---------------------------------------------------------------------------
# driving harness
# ---------------------------------------------------------------------------
class ToyMatcher:
    def match(self, s, tok):
        return (s, None, None)


class GenBatch(SimpleNamespace):
    Response = None


def make_gen_batch(bg, model, uids, cache, greedy=True):
    gb = GenBatch(
        model=model,
        prefill_step_size=512,
        uids=list(uids),
        prompt_cache=[cache],
        tokens=[[1] for _ in uids],
        samplers=[None] * len(uids),
        fallback_sampler=None,
        logits_processors=[[] for _ in uids],
        state_machines=[ToyMatcher() for _ in uids],
        max_tokens=[10**6] * len(uids),
        _next_tokens=None,
        _next_logprobs=[],
        _token_context=[None] * len(uids),
        _num_tokens=[0] * len(uids),
        _matcher_states=[None] * len(uids),
        _greedy=greedy,
        _procs=None,
    )
    gb.extract_cache = lambda idx: [cache.clone_row(idx)]
    gb.filter = lambda keep: None
    GenBatch.Response = _Response
    return gb


class _Response:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def new_state(bg, model, uid, depth, skill, first_tok, cache, row):
    """Seed one row's _MtpState the way _post_init_mtp would."""
    st = bg._MtpState()
    st.uid = uid
    st.chain = True
    st.depth = depth
    st.head_clone = False
    st.mtp_cache = [ToyHeadCache(seed=uid, skill=skill)]
    st.hist_offset = 0
    st.next_main = mx.array([first_tok], dtype=mx.uint32)
    # first chain: fold the seed token, then draft
    bg._chain_next_drafts(
        row, st, mx.zeros((1, 1, H)), mx.array([first_tok], dtype=mx.uint32), None
    )
    return st


def _snap(cache, b, st, tgt, head):
    """Record state keyed by position, so runs with different cycle
    boundaries can still be compared where they describe the same prefix."""
    off, acc, hist = cache.snapshot_row(b)
    tgt[off] = (acc, hist)
    n, hacc, hhist = st.mtp_cache[0].snapshot()
    head[st.hist_offset] = (n, hacc, hhist)


def run_alone(bg, uid, depth, skill, first_tok, n_tokens):
    """oMLX's real singleton verify cycle, one row, unpatched.

    One token per call, exactly like _mtp_next: run a cycle only when the
    queue is dry. Snapshots are taken per CYCLE so they line up with the
    fused run cycle for cycle.
    """
    model = ToyModel()
    cache = ToyCache(1)
    gb = make_gen_batch(bg, model, [uid], cache)
    st = new_state(bg, model, uid, depth, skill, first_tok, cache, gb)
    gb._omlx_mtp_state = st
    stream, tgt, head, ncyc = [], {}, {}, 0
    while len(stream) < n_tokens:
        if not st.queue:
            bg._run_verify_cycle_chain(gb, st)
            ncyc += 1
            _snap(cache, 0, st, tgt, head)
        tok, _lp, src = st.queue.popleft()
        stream.append((int(tok), src))
    return stream, (tgt, head, ncyc), st, model


def run_fused(bg, rows, n_tokens, greedy=True):
    """The patched multi-row path, one token per row per call."""
    B = len(rows)
    model = ToyModel()
    cache = ToyCache(B)
    uids = [r["uid"] for r in rows]
    gb = make_gen_batch(bg, model, uids, cache, greedy=greedy)
    states = {}
    for b, r in enumerate(rows):
        row_view = bg._make_row_batch(gb, b, prompt_cache=[cache], state=None)
        states[r["uid"]] = new_state(
            bg, model, r["uid"], r["depth"], r["skill"], r["first"], cache, row_view
        )
    bs = bg._MtpBatchState(states=states)
    gb._omlx_mtp_batch_state = bs

    streams = {u: [] for u in uids}
    tgt = {u: {} for u in uids}
    head = {u: {} for u in uids}
    seen = {u: 0 for u in uids}
    calls = 0
    while min(len(streams[u]) for u in uids) < n_tokens and calls < 40 * n_tokens:
        calls += 1
        responses = bg._mtp_batch_next(gb, bs)
        for b, u in enumerate(uids):
            st = states[u]
            if st.stats.cycles > seen[u]:
                seen[u] = st.stats.cycles
                _snap(cache, b, st, tgt[u], head[u])
        for r in responses:
            streams[r.uid].append((int(r.token), None))
    snaps = {u: (tgt[u], head[u], seen[u]) for u in uids}
    return streams, snaps, states, model, calls


def fused_backbone_calls(model):
    return [c for c in model.calls if c[0] == "backbone"]


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def main():
    os.environ["OMLX_MTP_BATCHED"] = "1"
    os.environ.setdefault("OMLX_MTP_BATCHED_MAX_CTX", "0")  # no context guard here
    bg = install_fake_bg()

    spec = importlib.util.spec_from_file_location(
        "batched_mtp_patch", os.path.join(HERE, "patch.py")
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)

    print("\n[0] plumbing")
    stock_batch_next = bg._mtp_batch_next
    ok = patch.install_batched_mtp()
    check("install returns True", ok)
    check("_mtp_batch_next swapped", bg._mtp_batch_next is not stock_batch_next)
    check("_rowwise_batch_mtp_enabled lifted", bg._rowwise_batch_mtp_enabled() is True)
    check(
        "late-arrival pin lifted (_generation_batch_has_active_mtp)",
        bg._generation_batch_has_active_mtp(object()) is False,
    )
    check("install idempotent", patch.install_batched_mtp() is True)

    ROWS4 = [
        {"uid": 101, "depth": 3, "skill": 95, "first": 5},
        {"uid": 102, "depth": 3, "skill": 60, "first": 9},
        {"uid": 103, "depth": 3, "skill": 30, "first": 17},
        {"uid": 104, "depth": 3, "skill": 80, "first": 33},
    ]
    N = 40

    for B in (2, 4):
        rows = ROWS4[:B]
        print(f"\n[1] identity, B={B}")
        alone = {}
        for r in rows:
            s, sn, st, _m = run_alone(
                bg, r["uid"], r["depth"], r["skill"], r["first"], N
            )
            alone[r["uid"]] = (s, sn, st)
        streams, snaps, states, model, ncalls = run_fused(bg, rows, N)

        for r in rows:
            u = r["uid"]
            a_toks = [t for t, _ in alone[u][0]]
            f_toks = [t for t, _ in streams[u]]
            n = min(len(a_toks), len(f_toks))
            check(
                f"uid {u} (skill {r['skill']}%) token stream identical",
                n > 0 and a_toks[:n] == f_toks[:n],
                f"{n} tokens",
            )


        calls = fused_backbone_calls(model)
        alone_fwd = sum(alone[r["uid"]][1][2] for r in rows)
        check(
            f"B={B}: every backbone forward carries all {B} rows",
            calls and all(c[1] == B for c in calls),
            f"{len(calls)} fused forwards vs {alone_fwd} row-wise, "
            f"cols={sorted({c[2] for c in calls})}",
        )

        print(f"\n[2] rollback per row, B={B}")
        ncyc = min(snaps[r["uid"]][2] for r in rows)
        ragged = sum(
            len({max(snaps[r["uid"]][0]) for r in rows}) > 1 for _ in [0]
        )
        offsets_seen = [sorted(snaps[r["uid"]][0]) for r in rows]
        divergent = len({tuple(o) for o in offsets_seen}) > 1
        check(
            "acceptance differs across rows (per-row cache offsets diverge)",
            divergent,
            f"row commit points {[o[:6] for o in offsets_seen]}",
        )
        del ragged, ncyc
        for r in rows:
            u = r["uid"]
            a_tgt = alone[u][1][0]
            f_tgt = snaps[u][0]
            shared = sorted(set(a_tgt) & set(f_tgt))
            bad = [o for o in shared if a_tgt[o] != f_tgt[o]]
            check(
                f"uid {u} target state (GDN acc + PLE window) exact at every "
                "shared commit point",
                shared and not bad,
                f"{len(shared)} commit points"
                if not bad
                else f"first mismatch at offset {bad[0]}",
            )

        print(f"\n[3] isolation, B={B}")
        for r in rows:
            u = r["uid"]
            a_head = alone[u][1][1]
            f_head = snaps[u][1]
            shared = sorted(set(a_head) & set(f_head))
            bad = [o for o in shared if a_head[o] != f_head[o]]
            check(
                f"uid {u} MTP head cache exact at every shared hist_offset",
                shared and not bad,
                f"{len(shared)} points"
                if not bad
                else f"first mismatch at hist_offset {bad[0]}",
            )
            st = states[u]
            emitted_by_cycles = st.stats.accepts + st.stats.cycles
            check(
                f"uid {u} stats self-consistent (accepts + cycles == generated)",
                emitted_by_cycles >= len(streams[u]),
                f"cycles={st.stats.cycles} accepts={st.stats.accepts} "
                f"rejects={st.stats.rejects} emitted={len(streams[u])}",
            )
            check(
                f"uid {u} depth histogram monotone and non-negative",
                all(v >= 0 for v in st.stats.depth_drafted)
                and all(
                    a <= d
                    for a, d in zip(st.stats.depth_accepted, st.stats.depth_drafted)
                ),
                f"drafted={list(st.stats.depth_drafted)} "
                f"accepted={list(st.stats.depth_accepted)}",
            )
        head_ids = {id(states[r["uid"]].mtp_cache[0]) for r in rows}
        check("head caches are distinct objects", len(head_ids) == B)
        cyc_alone = sum(alone[r["uid"]][1][2] for r in rows)
        cyc_fused = sum(snaps[r["uid"]][2] for r in rows) // B
        check(
            f"B={B}: fused forwards below row-wise forwards",
            cyc_fused < cyc_alone,
            f"{cyc_fused} fused vs {cyc_alone} row-wise "
            f"({cyc_alone / max(1, cyc_fused):.2f}x fewer)",
        )

    print("\n[4] fallback routing")
    # (a) non-greedy: must go to the stock loop, which the fake wires to the
    #     real singleton cycle per row
    rows = ROWS4[:2]
    model = ToyModel()
    cache = ToyCache(2)
    gb = make_gen_batch(bg, model, [r["uid"] for r in rows], cache, greedy=True)
    states = {}
    for b, r in enumerate(rows):
        rv = bg._make_row_batch(gb, b, prompt_cache=[cache], state=None)
        states[r["uid"]] = new_state(
            bg, model, r["uid"], r["depth"], r["skill"], r["first"], cache, rv
        )
    bs = bg._MtpBatchState(states=states)
    before = patch._TRACE["fallbacks"]
    gb._greedy = False
    try:
        bg._mtp_batch_next(gb, bs)
    except Exception:
        pass
    check("non-greedy batch falls back to the stock loop",
          patch._TRACE["fallbacks"] == before + 1)

    # (b) B above the cap
    os.environ["OMLX_MTP_BATCHED_MAX_B"] = "1"
    gb._greedy = True
    before = patch._TRACE["fallbacks"]
    try:
        bg._mtp_batch_next(gb, bs)
    except Exception:
        pass
    check("B above OMLX_MTP_BATCHED_MAX_B falls back",
          patch._TRACE["fallbacks"] == before + 1)
    os.environ["OMLX_MTP_BATCHED_MAX_B"] = "4"

    # (c) context guard
    os.environ["OMLX_MTP_BATCHED_MAX_CTX"] = "1"
    before = patch._TRACE["fallbacks"]
    try:
        bg._mtp_batch_next(gb, bs)
    except Exception:
        pass
    check("context guard falls back past OMLX_MTP_BATCHED_MAX_CTX",
          patch._TRACE["fallbacks"] == before + 1)
    os.environ["OMLX_MTP_BATCHED_MAX_CTX"] = "0"

    # (d) mixed depths across rows: k collapses to the minimum and the fused
    #     stream still matches the alone-run at that depth
    print("\n[5] ragged draft depth")
    rows = [
        {"uid": 201, "depth": 3, "skill": 90, "first": 7},
        {"uid": 202, "depth": 2, "skill": 55, "first": 21},
    ]
    alone = {}
    for r in rows:
        # the alone-run at the depth the fused cycle will actually verify
        s, sn, st, _m = run_alone(bg, r["uid"], 2, r["skill"], r["first"], 25)
        alone[r["uid"]] = (s, sn, st)
    streams, snaps, states, model, _nc = run_fused(bg, rows, 25)
    for r in rows:
        u = r["uid"]
        a = [t for t, _ in alone[u][0]]
        f = [t for t, _ in streams[u]]
        n = min(len(a), len(f))
        check(
            f"uid {u} matches depth-min alone-run",
            a[:n] == f[:n],
            f"{n} tokens at k=min(3,2)=2",
        )
    calls = fused_backbone_calls(model)
    check("ragged depths verify at a common window",
          {c[2] for c in calls} == {3}, f"cols={sorted({c[2] for c in calls})}")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: " + ", ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
