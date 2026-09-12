#!/usr/bin/env python3
"""Synthetic proof that the shortlist drafter cannot change the output.

A miniature stand-in for oMLX's batch_generator is registered under the real
module name, so patch.py monkeypatches and then exercises the SAME
``_chain_next_drafts`` code that runs in production, against a fake 512-token
model that fits in a few MB.

Three checks:

  1. lemma   whenever the full-vocabulary argmax of a draft step lands inside
             the shortlist, the shortlisted drafter picks the identical token
  2. stream  the accepted+corrected token stream over 200 greedy cycles is
             bit-identical between the stock drafter and the shortlisted one
  3. sampled the shortlisted proposal row is a proper log-density: -inf off
             the shortlist, log-sum-exp 0 on it, so the Leviathan/Chen ratio
             and residual stay exact

Run: ~/inference-server/kdev/bin/python test_shortlist.py
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMLX_MTP_SHORTLIST_DRAFT", "1")

import mlx.core as mx
import mlx.nn as nn

V, H, GS, BITS = 512, 64, 64, 4
DEPTH = 3
BG_NAME = "omlx.patches.mlx_lm_mtp.batch_generator"


# --------------------------------------------------------------------------
# fake batch_generator: only the helpers patch.py reaches for
# --------------------------------------------------------------------------
def _install_fake_bg():
    for pkg in ("omlx", "omlx.patches", "omlx.patches.mlx_lm_mtp"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    bg = types.ModuleType(BG_NAME)
    bg._HEAD_HIDDEN_POST_NORM = False
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
    """Structural copy of batch_generator.py:2300-2422, greedy path."""
    bg = sys.modules[BG_NAME]
    model = gen_batch.model
    sampler = bg._resolve_draft_sampler(gen_batch, state)
    depth = state.depth
    n = committed.shape[0]
    logits, head_hidden = model.mtp_forward(
        hidden_rows, committed.reshape(1, n), state.mtp_cache,
        return_hidden=True, logits_keep=1,
    )
    draft_toks, draft_lps, draft_accept_lps = [], [], []
    h = head_hidden[:, -1:]
    for j in range(depth):
        logits_2d = logits[:, -1, :]
        lp_2d = bg._logprobs(logits_2d)
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
    state.drafts = mx.concatenate(draft_toks)
    state.draft_lps = draft_lps
    state.draft_accept_lps = draft_accept_lps


# --------------------------------------------------------------------------
# fake model
# --------------------------------------------------------------------------
class FakeMTP:
    def __init__(self, embed, sticky=0.0):
        self.embed = embed
        # ``sticky`` mixes in an identity term so successive hidden states
        # stay correlated, the way a real LM's next-token distributions do.
        self.wa = (sticky * mx.eye(H) + (1.0 - sticky) * mx.random.normal((H, H))) * 0.9
        self.wb = mx.random.normal((H, H)) * 0.3 * (1.0 - sticky) + 0.05 * sticky

    def __call__(self, hidden, next_token_ids, embed_tokens, cache):
        e = embed_tokens(next_token_ids)
        out = mx.tanh(hidden @ self.wa + e @ self.wb)
        return out, out


class FakeInner:
    def __init__(self, embed):
        self.embed_tokens = embed


class FakeLM:
    def __init__(self, sticky=0.0):
        mx.random.seed(11)
        self.args = types.SimpleNamespace(tie_word_embeddings=False)
        embed = nn.Embedding(V, H)
        self.model = FakeInner(embed)
        self._mtp = FakeMTP(embed, sticky)
        lin = nn.Linear(H, V, bias=False)
        lin.weight = mx.random.normal((V, H)) * 0.35
        self.lm_head = nn.QuantizedLinear.from_linear(lin, group_size=GS, bits=BITS)
        mx.eval(self.lm_head.parameters(), embed.parameters())

    def get_mtp_module(self):
        return self._mtp

    def mtp_forward(self, hidden, ids, cache, return_hidden=False, logits_keep=0):
        out, hc = self._mtp(hidden, ids, self.model.embed_tokens, cache)
        src = out
        if logits_keep and src.shape[1] > logits_keep:
            src = src[:, -logits_keep:, :]
        logits = self.lm_head(src)
        return (logits, hc) if return_hidden else logits


class State:
    def __init__(self, depth):
        self.depth = depth
        self.controller = None
        self.mtp_cache = []
        self.head_clone = False
        self.hist_offset = 0
        self.drafts = None
        self.draft_lps = []
        self.draft_accept_lps = []


# --------------------------------------------------------------------------
# a target the drafter approximates closely (so acceptance is realistic)
# --------------------------------------------------------------------------
class Oracle:
    """Exact greedy target: the same recurrence as the draft head, with the
    weights perturbed by ``eps``. Small eps -> high acceptance, which is what
    makes the stream-identity check bite (accepted drafts really are emitted).
    """

    def __init__(self, lm: "FakeLM", eps: float):
        mx.random.seed(29)
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
        mx.eval(self.wa, self.wb, self.emb, self.head.weight)

    def _advance(self, h, tok_id):
        e = self.emb[tok_id].reshape(1, 1, H)
        return mx.tanh(h @ self.wa + e @ self.wb)

    def window(self, prev_id, k):
        """Greedy continuation of length k+1 from the committed state."""
        h = self.h
        out = []
        tok = prev_id
        for _ in range(k + 1):
            h = self._advance(h, tok)
            tok = int(mx.argmax(self.head(h)[0, -1]).item())
            out.append(tok)
        return out

    def commit(self, prev_id, toks):
        """Advance the persistent state over the tokens actually emitted."""
        h = self.h
        tok = prev_id
        hidden = []
        for t in toks:
            h = self._advance(h, tok)
            hidden.append(h)
            tok = int(t)
        self.h = h
        return mx.concatenate(hidden, axis=1)


def run_loop(chain_fn, n_tokens, depth, tag, eps=0.02, sticky=0.0):
    mx.random.seed(7)
    lm = FakeLM(sticky)
    oracle = Oracle(lm, eps)
    gen_batch = types.SimpleNamespace(model=lm)
    state = State(depth)

    emitted = []
    prev_id = 3
    committed = mx.array([prev_id], dtype=mx.uint32)
    hidden = oracle.commit(prev_id, [prev_id])
    drafted = accepted = cycles = 0
    while len(emitted) < n_tokens:
        cycles += 1
        chain_fn(gen_batch, state, hidden, committed, None)
        drafts = [int(t) for t in state.drafts.tolist()]
        k = len(drafts)
        targets = oracle.window(prev_id, k)
        m = 0
        while m < k and drafts[m] == targets[m]:
            m += 1
        new_toks = drafts[:m] + [targets[m]]
        emitted.extend(new_toks)
        drafted += k
        accepted += m
        hidden = oracle.commit(prev_id, new_toks)
        prev_id = new_toks[-1]
        committed = mx.array(new_toks, dtype=mx.uint32)
    rate = accepted / drafted if drafted else 0.0
    print(f"  {tag:<22} tokens={len(emitted):4d} cycles={cycles:4d} "
          f"accept={accepted}/{drafted} ({rate*100:.1f}%) "
          f"tok/cycle={len(emitted)/cycles:.2f}")
    return emitted[:n_tokens], rate


def main():
    bg = _install_fake_bg()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import importlib.util

    spec = importlib.util.spec_from_file_location("mtp_patch", here / "patch.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)

    fails = 0

    # ---- 0. vocabulary passes per cycle ---------------------------------
    print("\n[0] full-vocabulary lm_head passes per draft chain (depth 3)")
    counts = {}

    def count_run(fn, tag):
        lm = FakeLM()
        n = [0]
        cls = type(lm.lm_head)
        real = cls.__call__

        def counting(self, x, _r=real, _n=n, _h=lm.lm_head):
            if self is _h and x.shape[-2] * x.shape[0] <= 8:
                _n[0] += 1
            return _r(self, x)

        cls.__call__ = counting
        gen_batch = types.SimpleNamespace(model=lm)
        st_ = State(DEPTH)
        committed = mx.array([3], dtype=mx.uint32)
        hidden = mx.random.normal((1, 1, H)) * 0.2
        try:
            fn(gen_batch, st_, hidden, committed, None)
            mx.eval(st_.drafts)
        finally:
            cls.__call__ = real
        counts[tag] = n[0]
        print(f"  {tag:<22} draft lm_head passes = {n[0]}  "
              f"(+1 for the verify forward = {n[0] + 1} per cycle)")

    count_run(bg._chain_next_drafts, "stock drafter")
    os.environ["OMLX_MTP_SHORTLIST_K"] = "64"
    os.environ["OMLX_MTP_SHORTLIST_FROM_STEP"] = "1"
    _orig0 = bg._chain_next_drafts
    patch._SHORTLIST_INSTALLED = False
    assert patch.install_shortlist_draft()
    count_run(bg._chain_next_drafts, "shortlist from step 1")
    bg._chain_next_drafts = _orig0
    if counts["stock drafter"] != 3 or counts["shortlist from step 1"] != 1:
        print("  FAIL: unexpected pass count")
        fails += 1
    else:
        print("  PASS: 3+1 -> 1+1 full-vocabulary passes per cycle")

    # ---- 1. lemma -------------------------------------------------------
    print("\n[1] shortlist lemma: argmax inside the shortlist -> same token")
    lm = FakeLM()
    sl = patch._Shortlist()
    mx.random.seed(5)
    hits = same = total = 0
    for _ in range(200):
        h = mx.random.normal((1, H)) * 0.6
        full = lm.lm_head(h[None])[0]
        full_lp = full - mx.logsumexp(full, axis=-1, keepdims=True)
        patch._refresh_shortlist(mx, sl, lm.lm_head, full_lp, 64)
        h2 = h + 0.25 * mx.random.normal((1, H))
        exact = lm.lm_head(h2[None])[0]
        exact_arg = int(mx.argmax(exact).item())
        short_lp = patch._shortlist_logprobs(mx, sl, lm.lm_head, h2)
        short_arg = int(mx.argmax(short_lp).item())
        in_list = exact_arg in set(int(v) for v in sl.ids.tolist())
        total += 1
        if in_list:
            hits += 1
            same += int(short_arg == exact_arg)
    print(f"  argmax covered by a 64-wide shortlist: {hits}/{total}")
    print(f"  same token whenever covered:           {same}/{hits}")
    if same != hits:
        print("  FAIL: covered argmax not reproduced")
        fails += 1
    else:
        print("  PASS")

    # ---- 2. stream identity --------------------------------------------
    print("\n[2] greedy stream identity: first 400 emitted tokens")
    stock, r_stock = run_loop(bg._chain_next_drafts, 400, DEPTH, "stock drafter")
    orig = bg._chain_next_drafts
    for K in (32, 64, 128):
        os.environ["OMLX_MTP_SHORTLIST_K"] = str(K)
        os.environ["OMLX_MTP_SHORTLIST_FROM_STEP"] = "1"
        patch._SHORTLIST_INSTALLED = False
        bg._chain_next_drafts = orig
        assert patch.install_shortlist_draft()
        got, r_sl = run_loop(bg._chain_next_drafts, 400, DEPTH, f"shortlist K={K}")
        ok = got == stock
        print(f"    identical to stock: {ok}   accept {r_stock*100:.1f}% -> "
              f"{r_sl*100:.1f}%")
        if not ok:
            fails += 1
    bg._chain_next_drafts = orig

    # ---- 2c. FROM_STEP=0 (shortlist every step, refresh every R cycles) --
    print("\n[2c] FROM_STEP=0, all draft steps shortlisted (K=64)")
    os.environ["OMLX_MTP_SHORTLIST_K"] = "64"
    os.environ["OMLX_MTP_SHORTLIST_FROM_STEP"] = "0"
    for R in ("1", "2"):
        os.environ["OMLX_MTP_SHORTLIST_REFRESH"] = R
        patch._SHORTLIST_INSTALLED = False
        bg._chain_next_drafts = orig
        assert patch.install_shortlist_draft()
        got, r_sl = run_loop(bg._chain_next_drafts, 400, DEPTH,
                             f"from_step=0 refresh={R}")
        ok = got == stock
        print(f"    identical to stock: {ok}   accept {r_stock*100:.1f}% -> "
              f"{r_sl*100:.1f}%")
        if not ok:
            fails += 1
    os.environ["OMLX_MTP_SHORTLIST_FROM_STEP"] = "1"
    os.environ["OMLX_MTP_SHORTLIST_REFRESH"] = "1"
    bg._chain_next_drafts = orig

    # ---- 2b. acceptance loss vs how correlated successive steps are -----
    print("\n[2b] acceptance loss vs step-to-step correlation (K=64 of 512)")
    os.environ["OMLX_MTP_SHORTLIST_K"] = "64"
    for sticky in (0.0, 0.5, 0.8, 0.95):
        bg._chain_next_drafts = orig
        a, ra = run_loop(orig, 400, DEPTH, f"  stock  sticky={sticky}",
                         sticky=sticky)
        patch._SHORTLIST_INSTALLED = False
        bg._chain_next_drafts = orig
        assert patch.install_shortlist_draft()
        b, rb = run_loop(bg._chain_next_drafts, 400, DEPTH,
                         f"  short  sticky={sticky}", sticky=sticky)
        same = a == b
        print(f"    identical={same}  accept {ra*100:.1f}% -> {rb*100:.1f}% "
              f"(loss {100*(ra-rb):.1f} pts)")
        if not same:
            fails += 1
    bg._chain_next_drafts = orig

    # ---- 3. proposal density -------------------------------------------
    print("\n[3] shortlisted proposal row is a proper log-density")
    sl = patch._Shortlist()
    h = mx.random.normal((1, H)) * 0.6
    full = lm.lm_head(h[None])[0]
    patch._refresh_shortlist(mx, sl, lm.lm_head, full, 64)
    row = patch._shortlist_logprobs(mx, sl, lm.lm_head, h)
    ids = set(int(v) for v in sl.ids.tolist())
    off = [i for i in range(V) if i not in ids]
    off_max = float(mx.max(row[0, mx.array(off)]).item())
    on_lse = float(mx.logsumexp(row[0, mx.array(sorted(ids))]).item())
    tot = float(mx.exp(mx.logsumexp(row)).item())
    print(f"  max log-prob off the shortlist: {off_max:.3e}  (want -inf-like)")
    print(f"  log-sum-exp over the shortlist: {on_lse:.3e}  (want 0)")
    print(f"  total probability mass:         {tot:.6f}  (want 1)")
    if off_max > -1e30 or abs(on_lse) > 1e-3 or abs(tot - 1.0) > 1e-3:
        print("  FAIL")
        fails += 1
    else:
        print("  PASS")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{fails} CHECK(S) FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
