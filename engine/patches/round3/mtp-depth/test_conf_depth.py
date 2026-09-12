#!/usr/bin/env python3
"""Synthetic proof for the per-cycle confidence gate (OMLX_MTP_CONF_DEPTH=1).

A miniature stand-in for oMLX's batch_generator is registered under the real
module name, so patch.py monkeypatches and then exercises the SAME
``_chain_next_drafts`` and the SAME ``_DepthController`` (its class body is
lifted verbatim out of
/Applications/oMLX.app/.../omlx/patches/mlx_lm_mtp/batch_generator.py
lines 1823-2179 and exec'd here) against a fake 512-token model.

The fake model has a controllable draft-confidence distribution: a per-cycle
logit gain makes some cycles sharp (the head is sure) and others flat (the
head is guessing), so the gate has something to separate.

Checks:

  [0] plumbing   the gate installs, the controller's _best collapses to
                 {0, max_depth}, and the ceiling install raises the marker
  [1] identity   greedy emitted stream is bit-identical to the fixed-depth
                 chain over 600 tokens, with and without the round-2
                 shortlist drafter
  [2] shape      the gate drafts deeper on sharp cycles than on flat ones,
                 and the depth histogram spreads instead of pinning
  [3] state      after the same token stream, the recurrent target state,
                 the MTP head cache and hist_offset match the fixed-depth
                 run exactly, with M varying cycle to cycle
  [4] floors     the break-even floor computed from a controller's live
                 t[] and p[] matches the closed form

Run: ~/inference-server/kdev/bin/python test_conf_depth.py
"""

import importlib.util
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401  (used by exec'd class)

import mlx.core as mx
import mlx.nn as nn

V, H, GS, BITS = 512, 64, 64, 4
BG_NAME = "omlx.patches.mlx_lm_mtp.batch_generator"
HERE = Path(__file__).resolve().parent
OMLX_BG = Path(
    "/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_lm_mtp/"
    "batch_generator.py"
)


# ---------------------------------------------------------------------------
# the real _DepthController, lifted out of the shipped file
# ---------------------------------------------------------------------------
def _load_real_controller():
    src = OMLX_BG.read_text()
    m = re.search(r"^class _DepthController:.*?(?=^# Draft sampler)", src,
                  re.S | re.M)
    if not m:
        raise RuntimeError("could not locate _DepthController in " + str(OMLX_BG))
    ns = {"math": math, "Dict": Dict, "List": List, "Optional": Optional,
          "_STD_TAX_MAX": 1.5}
    exec(compile(m.group(0), str(OMLX_BG), "exec"), ns)
    return ns["_DepthController"]


# ---------------------------------------------------------------------------
# fake batch_generator
# ---------------------------------------------------------------------------
def _install_fake_bg(ctrl_cls):
    for pkg in ("omlx", "omlx.patches", "omlx.patches.mlx_lm_mtp"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod
    bg = types.ModuleType(BG_NAME)
    bg._HEAD_HIDDEN_POST_NORM = False
    bg._DepthController = ctrl_cls
    bg._dspark_host = lambda model: None
    bg._proc_list = lambda gen_batch: None
    bg._trunk_norm_module = lambda model: (lambda x: x)
    bg._clone_mtp_head_cache = lambda c: c
    bg._snap_snapshotable = lambda procs: None
    bg._restore_snapshotable = lambda procs, snap: None
    bg._apply_processors = lambda procs, prev, lg: lg
    bg._logprobs = lambda lg: lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    bg._ensure_uint32 = lambda a: a.astype(mx.uint32)
    bg._accept_lp_for = lambda sampler, lp: lp
    bg._resolve_draft_sampler = lambda gb, st: (
        lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32)
    )
    bg._chain_next_drafts = _stock_chain_next_drafts
    sys.modules[BG_NAME] = bg
    sys.modules["omlx.patches.mlx_lm_mtp"].batch_generator = bg
    return bg


def _stock_chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf):
    """Structural copy of batch_generator.py:2303-2434, greedy path."""
    bg = sys.modules[BG_NAME]
    model = gen_batch.model
    sampler = bg._resolve_draft_sampler(gen_batch, state)
    depth = state.controller.cur if state.controller is not None else state.depth
    n = committed.shape[0]
    logits, head_hidden = model.mtp_forward(
        hidden_rows, committed.reshape(1, n), state.mtp_cache,
        return_hidden=True, logits_keep=1,
    )
    state.hist_offset += int(n)
    draft_toks, draft_lps, draft_accept_lps = [], [], []
    h = head_hidden[:, -1:]
    for j in range(depth):
        lp_2d = bg._logprobs(logits[:, -1, :])
        tok = bg._ensure_uint32(sampler(lp_2d))
        draft_toks.append(tok)
        draft_lps.append(lp_2d.squeeze(0))
        draft_accept_lps.append(lp_2d.squeeze(0))
        if j + 1 == depth:
            break
        logits, head_hidden = model.mtp_forward(
            h, tok.reshape(1, 1), state.mtp_cache, return_hidden=True
        )
        h = head_hidden[:, -1:]
    state.drafts = (
        mx.concatenate(draft_toks) if draft_toks
        else mx.zeros((0,), dtype=mx.uint32)
    )
    state.draft_lps = draft_lps
    state.draft_accept_lps = draft_accept_lps


# ---------------------------------------------------------------------------
# fake model: a stateful MTP head with a controllable confidence schedule
# ---------------------------------------------------------------------------
class HeadCache:
    """Stands in for the MTP head's KV cache: an append/trim history ring."""

    def __init__(self):
        self.ids: List[int] = []

    def append(self, ids):
        self.ids.extend(int(v) for v in ids)

    def trim_to(self, n: int):
        del self.ids[n:]

    def key(self):
        return tuple(self.ids)


class FakeMTPModule:
    """(hidden, ids) -> (out_hidden, out_hidden), history-dependent.

    ``gain`` is looked up per cycle so a caller can make the head sharp or
    flat on demand; the head's output also depends on the committed history
    length, so a cache that is trimmed wrongly changes the result.
    """

    def __init__(self, embed, owner):
        self.embed = embed
        self.owner = owner
        self.wa = mx.random.normal((H, H)) * 0.9
        self.wb = mx.random.normal((H, H)) * 0.3
        mx.eval(self.wa, self.wb)

    def __call__(self, hidden, ids, embed_tokens, cache):
        if cache is not None and hasattr(cache, "append"):
            cache.append(ids.reshape(-1).tolist())
            n = len(cache.ids)
        else:
            n = 1
        e = embed_tokens(ids)
        drift = 0.01 * float(n % 7)
        out = mx.tanh(hidden @ self.wa + e @ self.wb + drift)
        return out, out


class FakeInner:
    def __init__(self, embed):
        self.embed_tokens = embed


class FakeLM:
    def __init__(self, seed=11):
        mx.random.seed(seed)
        self.args = types.SimpleNamespace(tie_word_embeddings=False)
        embed = nn.Embedding(V, H)
        self.model = FakeInner(embed)
        self._mtp = FakeMTPModule(embed, self)
        lin = nn.Linear(H, V, bias=False)
        lin.weight = mx.random.normal((V, H)) * 0.35
        self.lm_head = nn.QuantizedLinear.from_linear(lin, group_size=GS, bits=BITS)
        self.gain = 1.0  # per-cycle confidence knob
        mx.eval(self.lm_head.parameters(), embed.parameters())

    def get_mtp_module(self):
        return self._mtp

    def mtp_forward(self, hidden, ids, cache, return_hidden=False, logits_keep=0):
        out, hc = self._mtp(hidden, ids, self.model.embed_tokens, cache)
        src = out
        if logits_keep and src.shape[1] > logits_keep:
            src = src[:, -logits_keep:, :]
        logits = self.lm_head(src) * self.gain
        return (logits, hc) if return_hidden else logits


class GatedQL(nn.QuantizedLinear):
    pass


class State:
    """The fields _chain_next_drafts touches on _MtpState."""

    def __init__(self, depth, controller=None):
        self.depth = depth
        self.controller = controller
        self.mtp_cache = HeadCache()
        self.head_clone = False
        self.hist_offset = 0
        self.drafts = None
        self.draft_lps = []
        self.draft_accept_lps = []


# ---------------------------------------------------------------------------
# target with a recurrent state that must be rolled back exactly
# ---------------------------------------------------------------------------
class Target:
    """Greedy oracle with a GDN/PLE-style recurrent state.

    ``verify(window)`` runs the whole verify window in one go, captures the
    per-position intermediate states the way _call_backbone(n_confirmed=1)
    does, and ``rollback(accepted)`` restores the state to the accepted
    prefix from those captures. A bug that assumes a fixed window size shows
    up as a state mismatch against the fixed-depth run.
    """

    def __init__(self, lm: FakeLM, eps: float, seed=29):
        mx.random.seed(seed)
        self.wa = lm._mtp.wa + eps * mx.random.normal(lm._mtp.wa.shape)
        self.wb = lm._mtp.wb + eps * mx.random.normal(lm._mtp.wb.shape)
        self.emb = lm.model.embed_tokens.weight + eps * mx.random.normal(
            lm.model.embed_tokens.weight.shape
        )
        lin = nn.Linear(H, V, bias=False)
        lin.weight = mx.dequantize(
            lm.lm_head.weight, lm.lm_head.scales, lm.lm_head.biases,
            group_size=GS, bits=BITS,
        ) + eps * mx.random.normal((V, H))
        self.head = lin
        self.h = mx.zeros((1, 1, H))
        self.ring: List[int] = []      # PLE-style n-gram ring
        self.pos = 0
        self._snap = None
        mx.eval(self.wa, self.wb, self.emb, self.head.weight)

    def _advance(self, h, tok_id):
        e = self.emb[int(tok_id)].reshape(1, 1, H)
        return mx.tanh(h @ self.wa + e @ self.wb)

    def verify(self, window):
        """Run the (variable length) verify window in one pass and capture
        the per-position intermediates, the way _call_backbone(n_confirmed=1)
        hands gdn_states to _chain_rollback. Row j consumes window[j] and
        predicts the token after it."""
        h = self.h
        ring = list(self.ring)
        inter_h, inter_ring = [], []
        preds, hiddens = [], []
        for w in window:
            h = self._advance(h, int(w))
            ring = (ring + [int(w)])[-8:]
            inter_h.append(h)
            inter_ring.append(list(ring))
            hiddens.append(h)
            preds.append(int(mx.argmax(self.head(h)[0, -1]).item()))
        self._snap = (inter_h, inter_ring, len(window))
        self.h = h
        self.ring = ring
        return preds, mx.concatenate(hiddens, axis=1)

    def rollback(self, accepted):
        """Keep accepted+1 of the window's positions (mirrors
        rollback_speculative_cache(caches, gdn, accepted, block_size))."""
        inter_h, inter_ring, win = self._snap
        keep = accepted + 1
        assert 0 < keep <= win, (keep, win)
        self.h = inter_h[keep - 1]
        self.ring = list(inter_ring[keep - 1])
        self._snap = None

    def key(self):
        return (
            tuple(round(float(v), 4) for v in self.h.reshape(-1).tolist()),
            tuple(self.ring),
        )


# ---------------------------------------------------------------------------
# the decode loop: mirrors _run_verify_cycle_chain's structure
# ---------------------------------------------------------------------------
def run_loop(bg, chain_fn, n_tokens, depth, controller=None, eps=0.02,
             gain_schedule=None, seed=7, cycle_ms=30.0):
    mx.random.seed(seed)
    lm = FakeLM()
    target = Target(lm, eps)
    gen_batch = types.SimpleNamespace(model=lm)
    state = State(depth, controller)

    # init (batch_generator.py:2480-2540): one confirmed forward at main_tok
    # gives next_main; the chain is seeded with committed=[next_main] and the
    # hidden AT main_tok. next_main is emitted but not yet folded into the
    # recurrent state, which is the one-token skew the verify window closes.
    main_tok = 3
    preds, hid = target.verify([main_tok])
    next_main = preds[0]
    emitted: List[int] = [main_tok, next_main]
    hidden = hid[:, :1]
    committed = mx.array([next_main], dtype=mx.uint32)

    ks: List[int] = []
    gains: List[float] = []
    snaps: Dict[int, Any] = {}
    head_ok = True
    cycles = 0
    accepted_tot = drafted_tot = 0

    while len(emitted) < n_tokens:
        if gain_schedule is not None:
            lm.gain = gain_schedule(cycles)
        gains.append(lm.gain)
        # batch_generator.py:3126 — drop the previous cycle's speculative
        # head entries before folding this cycle's committed tokens.
        state.mtp_cache.trim_to(state.hist_offset)
        chain_fn(gen_batch, state, hidden, committed, None)
        # The head's committed history must be exactly the emitted stream
        # from next_main onward, whatever depth the previous cycles drafted.
        head_ok &= (
            state.mtp_cache.ids[: state.hist_offset]
            == emitted[1 : 1 + state.hist_offset]
        )
        drafts = [int(t) for t in state.drafts.tolist()]
        k = len(drafts)
        ks.append(k)

        # verify [next_main, d1..dk]: the window is sized from
        # state.drafts.shape[0], so M = k + 1 varies cycle to cycle
        preds, hid = target.verify([next_main] + drafts)
        m = 0
        while m < k and drafts[m] == preds[m]:
            m += 1
        target.rollback(m)
        emit_last = preds[m]
        new_toks = drafts[:m] + [emit_last]
        emitted.extend(new_toks)
        drafted_tot += k
        accepted_tot += m

        hidden = hid[:, : m + 1]
        committed = mx.array(new_toks, dtype=mx.uint32)
        next_main = emit_last
        cycles += 1
        snaps[len(emitted)] = target.key()
        if controller is not None:
            controller.observe(k, m, cycle_ms + 1.9 * k)

    return {
        "tokens": emitted[:n_tokens],
        "ks": ks,
        "gains": gains,
        "cycles": cycles,
        "accept": (accepted_tot, drafted_tot),
        "snaps": snaps,
        "head_ok": head_ok,
        "hist_offset": state.hist_offset,
    }


# ---------------------------------------------------------------------------


def _load_patch():
    spec = importlib.util.spec_from_file_location("conf_patch", HERE / "patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_gate(bg, patch, stock, **env):
    """Reinstall the gate from scratch with a given env."""
    for k, v in env.items():
        os.environ[k] = str(v)
    bg._chain_next_drafts = stock
    bg._omlx_mtp_shortlist_orig = None
    patch._CHAIN_INSTALLED = False
    patch._CEILING_INSTALLED = False
    assert patch.install_conf_depth(), "gate failed to install"
    return bg._chain_next_drafts


def main():
    os.environ["OMLX_MTP_CONF_DEPTH"] = "1"
    os.environ.setdefault("OMLX_MTP_CONF_MAX_DEPTH", "5")
    ctrl_cls = _load_real_controller()
    bg = _install_fake_bg(ctrl_cls)
    stock = bg._chain_next_drafts
    patch = _load_patch()
    fails = 0

    # ---- [0] plumbing ---------------------------------------------------
    print("\n[0] install and controller hand-off")
    gated = _fresh_gate(bg, patch, stock, OMLX_MTP_SHORTLIST_DRAFT=0,
                        OMLX_MTP_CONF_PMIN=0.25, OMLX_MTP_CONF_MAX_DEPTH=5)
    print(f"  _chain_next_drafts is gated: "
          f"{getattr(gated, '_omlx_mtp_conf_depth', False)}")
    c = ctrl_cls(5)
    for _ in range(5 + 3):
        c.observe(c.cur, max(0, c.cur - 1), 30.0)
    picks = set()
    for _ in range(400):
        c.observe(c.cur, max(0, c.cur - 1), 30.0 + 1.9 * c.cur)
        if c.probe_left == 0 and not c._warmup:
            picks.add(c.cur)
    print(f"  post-warmup non-probe depths chosen by the wrapped _best: "
          f"{sorted(picks)}  (want a subset of {{0, 5}})")
    if not picks <= {0, 5} or not getattr(gated, "_omlx_mtp_conf_depth", False):
        print("  FAIL"); fails += 1
    else:
        print("  PASS")

    class M:
        _omlx_mtp_chain = True
        _omlx_mtp_depth = 3
    m = M()
    patch._CEILING_INSTALLED = False
    ok_ceiling = patch.install_conf_depth_ceiling(m) and m._omlx_mtp_depth == 5
    print(f"  ceiling install raised _omlx_mtp_depth 3 -> {m._omlx_mtp_depth}: "
          f"{ok_ceiling}")
    if not ok_ceiling:
        print("  FAIL"); fails += 1

    # ---- [1] greedy identity --------------------------------------------
    print("\n[1] greedy stream identity, 600 tokens")
    ref = run_loop(bg, stock, 600, 3)
    print(f"  fixed depth 3 (stock)      tokens={len(ref['tokens'])} "
          f"cycles={ref['cycles']} accept={ref['accept'][0]}/{ref['accept'][1]} "
          f"tok/cycle={len(ref['tokens'])/ref['cycles']:.2f}")
    ref5 = run_loop(bg, stock, 600, 5)
    print(f"  fixed depth 5 (stock)      tokens={len(ref5['tokens'])} "
          f"cycles={ref5['cycles']} accept={ref5['accept'][0]}/{ref5['accept'][1]} "
          f"tok/cycle={len(ref5['tokens'])/ref5['cycles']:.2f}")
    if ref5["tokens"] != ref["tokens"]:
        print("  FAIL: the fixed-depth reference is not depth-invariant")
        fails += 1

    for sl in (0, 1):
        for pmin in ("0.15", "0.25", "0.40"):
            g = _fresh_gate(bg, patch, stock, OMLX_MTP_SHORTLIST_DRAFT=sl,
                            OMLX_MTP_SHORTLIST_K=64, OMLX_MTP_CONF_PMIN=pmin,
                            OMLX_MTP_CONF_ADAPT=0, OMLX_MTP_CONF_MAX_DEPTH=5)
            got = run_loop(bg, g, 600, 5)
            same = got["tokens"] == ref["tokens"]
            hist = [got["ks"].count(i) for i in range(6)]
            print(f"  gate pmin={pmin} shortlist={sl}  identical={same}  "
                  f"cycles={got['cycles']}  "
                  f"accept={got['accept'][0]}/{got['accept'][1]}  "
                  f"depth hist={hist}")
            if not same:
                fails += 1

    # ---- [2] the gate follows confidence --------------------------------
    print("\n[2] depth follows the head's confidence")

    def schedule(i):
        return 3.0 if (i // 4) % 2 == 0 else 0.25

    g = _fresh_gate(bg, patch, stock, OMLX_MTP_SHORTLIST_DRAFT=0,
                    OMLX_MTP_CONF_PMIN=0.25, OMLX_MTP_CONF_ADAPT=0,
                    OMLX_MTP_CONF_MAX_DEPTH=5)
    got = run_loop(bg, g, 600, 5, gain_schedule=schedule)
    sharp = [k for k, gn in zip(got["ks"], got["gains"]) if gn > 1.0]
    flat = [k for k, gn in zip(got["ks"], got["gains"]) if gn < 1.0]
    ms = sum(sharp) / max(1, len(sharp))
    mf = sum(flat) / max(1, len(flat))
    hist = [got["ks"].count(i) for i in range(6)]
    print(f"  sharp cycles: n={len(sharp)} mean depth={ms:.2f}")
    print(f"  flat  cycles: n={len(flat)} mean depth={mf:.2f}")
    print(f"  depth histogram k=0..5: {hist}")
    print(f"  identical to fixed depth 3: {got['tokens'] == ref['tokens']}")
    if not (ms > mf + 0.5) or len([h for h in hist if h]) < 2:
        print("  FAIL: the gate is not separating the two regimes")
        fails += 1
    elif got["tokens"] != ref["tokens"]:
        print("  FAIL: output changed"); fails += 1
    else:
        print("  PASS")

    # ---- [3] no state corruption with varying M -------------------------
    print("\n[3] recurrent state, head cache and hist_offset vs fixed M")
    print("  compared at every emitted-token count the two runs share")

    def cmp_snaps(a, b):
        common = sorted(set(a["snaps"]) & set(b["snaps"]))
        bad = [n for n in common if a["snaps"][n] != b["snaps"][n]]
        return len(common), bad

    ok3 = True
    nc, bad = cmp_snaps(ref, ref5)
    print(f"  fixed 3 vs fixed 5: {nc} shared checkpoints, "
          f"{len(bad)} mismatched, head cache clean "
          f"{ref['head_ok'] and ref5['head_ok']}")
    ok3 &= not bad and nc > 20 and ref["head_ok"] and ref5["head_ok"]
    for pmin in ("0.15", "0.40"):
        for adapt in (0, 1):
            g = _fresh_gate(bg, patch, stock, OMLX_MTP_SHORTLIST_DRAFT=0,
                            OMLX_MTP_CONF_PMIN=pmin, OMLX_MTP_CONF_ADAPT=adapt,
                            OMLX_MTP_CONF_MAX_DEPTH=5)
            ctl = ctrl_cls(5)
            got = run_loop(bg, g, 600, 5, controller=ctl,
                           gain_schedule=schedule)
            hist = [got["ks"].count(i) for i in range(6)]
            nc, bad = cmp_snaps(ref, got)
            varied = len(set(got["ks"])) > 1
            print(f"  pmin={pmin} adapt={adapt} hist={hist} M varied={varied}  "
                  f"tokens={got['tokens'] == ref['tokens']}  "
                  f"head cache clean={got['head_ok']}  "
                  f"{nc} shared checkpoints, {len(bad)} mismatched")
            ok3 &= (not bad) and nc > 20 and varied and got["head_ok"] \
                and got["tokens"] == ref["tokens"]
    if not ok3:
        print("  FAIL"); fails += 1
    else:
        print("  PASS: identical recurrent state, head cache and offset")

    # ---- [4] the break-even floor ---------------------------------------
    print("\n[4] adaptive floor against the closed form")
    cfg = patch._GateCfg()
    cfg.adapt = True
    ctl = ctrl_cls(5)
    ctl.t = {0: 24.4, 1: 26.0, 2: 27.92, 3: 29.84, 4: 31.76, 5: 33.68}
    ctl.p = [0.61, 0.43, 0.303, 0.214, 0.151]
    exp_, run = 1.0, 1.0
    ok4 = True
    for j in range(5):
        got = patch._break_even_floor(cfg, ctl, j, exp_)
        c_j, c_j1 = ctl.t[j], ctl.t[j + 1]
        want = min(0.90, max(0.02,
                             (c_j1 - c_j) * exp_ / (ctl.p[j] * c_j)))
        print(f"  step {j+1}: C({j})={c_j:5.2f} D={c_j1-c_j:.2f} "
              f"p={ctl.p[j]:.3f} E={exp_:.3f}  floor={got:.4f} "
              f"(closed form {want:.4f})")
        ok4 &= abs(got - want) < 1e-9
        run *= ctl.p[j]
        exp_ += run
    print("  PASS" if ok4 else "  FAIL")
    if not ok4:
        fails += 1

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{fails} CHECK(S) FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
