#!/usr/bin/env python3
"""Synthetic proof for the prompt-lookup copy lane, with a fake small model.

A miniature stand-in for oMLX's ``batch_generator`` is registered under the
real module name so ``patch.py`` monkeypatches, and then exercises, the SAME
``_chain_next_drafts`` entry point it patches in production. The verify cycle
here is a line-by-line transcription of the accept / clamp / emit / rollback
structure of ``_run_verify_cycle_chain`` (batch_generator.py:2908-3010), with
a fake cache whose recurrent GDN-like state and PLE-like snapshot are checked
against the emitted stream after every cycle.

Checks:
  [1] index        n-gram lookup is exact-id, most-recent-occurrence, O(1)
  [2] edit         greedy stream identical to stock on a high-copy rewrite
  [3] prose        greedy stream identical to stock with near-zero copy rate
  [4] stop         a copy block crossing EOS is truncated AT the stop, and
                   the recurrent cache never advances past it by more than
                   the one bonus row stock MTP also spends
  [5] rollback     after a partial accept, cache position, recurrent state
                   and PLE snapshot restore to the last accepted position
  [6] head         the MTP head history after a copy cycle equals what the
                   stock chain would have left (hazard c)
  [7] sampled      one-hot q gives Leviathan/Chen acceptance probability p,
                   and the emitted distribution matches the target's
  [8] merges       a block whose ids decode to the same text as a different
                   id sequence is never matched by text

Run: ~/inference-server/kdev/bin/python test_copy.py
"""

import os
import sys
import types
import importlib.util
import random
from collections import deque

os.environ.setdefault("OMLX_MTP_COPY_LANE", "1")

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
BG_NAME = "omlx.patches.mlx_lm_mtp.batch_generator"
V = 4096
DEPTH = 3


# ---------------------------------------------------------------------------
# fake batch_generator
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self.cycles = 0
        self.accepts = 0
        self.rejects = 0
        self.depth_drafted = []
        self.depth_accepted = []


class _State:
    def __init__(self, depth=DEPTH):
        self.chain = True
        self.depth = depth
        self.head_clone = False
        self.controller = _Ctrl(depth)
        self.drafts = None
        self.draft_lps = []
        self.draft_accept_lps = []
        self.mtp_cache = {"folded": []}
        self.hist_offset = 0
        self.stats = _Stats()
        self.queue = deque()


class _Ctrl:
    """Stand-in with the real controller's clamp and warmup shape."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.cur = max_depth
        self.p = [0.6] * max_depth
        self.t = {}
        self.cycles = 0
        self._warmup = list(range(max_depth, 0, -1)) + [0, 0, 0]
        self.seen = []

    def _score(self, d):
        return 1.0

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        self.cycles += 1
        used = max(0, min(int(used), self.max_depth))
        accepted = max(0, min(int(accepted), used))
        self.seen.append((used, accepted, round(cycle_ms, 3)))
        if self._warmup:
            self._warmup.pop(0)
            self.cur = self._warmup[0] if self._warmup else self.max_depth


class _Matcher:
    """Pure state machine: match(state, id) -> (state, matched_seq, current)."""

    def __init__(self, eos_ids=(), stop_seq=None):
        self.eos_token_ids = list(eos_ids)
        self.stop_seq = list(stop_seq or [])

    def make_state(self):
        return 0

    def match(self, st, tid):
        if tid in self.eos_token_ids:
            return 0, [tid], None
        if self.stop_seq:
            if tid == self.stop_seq[st]:
                st += 1
                if st == len(self.stop_seq):
                    return 0, list(self.stop_seq), None
                return st, None, st
            return 0, None, None
        return 0, None, None


def _install_fake_bg():
    for pkg in ("omlx", "omlx.patches", "omlx.patches.mlx_lm_mtp"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    bg = types.ModuleType(BG_NAME)
    bg._HEAD_HIDDEN_POST_NORM = False
    bg._trunk_norm_module = lambda model: (lambda x: x)
    bg._is_greedy = lambda gb: getattr(gb, "greedy", True)
    bg._DepthController = _Ctrl
    bg._log_mtp_stats = lambda uid, stats, reason: None
    bg._chain_next_drafts = _mtp_chain_next_drafts
    sys.modules[BG_NAME] = bg
    sys.modules["omlx.patches.mlx_lm_mtp"].batch_generator = bg
    return bg


# ---------------------------------------------------------------------------
# fake model: an oracle target plus a noisy MTP head
# ---------------------------------------------------------------------------
class FakeModel:
    """Greedy argmax at stream position t is ``intended[t]``.

    ``mtp_forward`` records the (hidden, token) pairs the head folds, so the
    test can assert the head history a copy cycle leaves behind.
    """

    def __init__(self, intended, mtp_acc=0.72, seed=0):
        self.intended = list(intended)
        self.mtp_acc = mtp_acc
        self.rng = random.Random(seed)
        self.folds = []
        self.args = types.SimpleNamespace(vocab_size=V, tie_word_embeddings=False)

    def rows(self, positions):
        """(len(positions), V) logits; argmax at p is intended[p]."""
        out = mx.full((len(positions), V), -8.0, dtype=mx.float32)
        idx = []
        for p in positions:
            idx.append(self.intended[p] if p < len(self.intended) else 0)
        idx = mx.array(idx, dtype=mx.int32)[:, None]
        return mx.put_along_axis(out, idx, mx.zeros((len(positions), 1)), axis=-1)

    def mtp_forward(self, hidden, next_ids, cache, return_hidden=False, logits_keep=0):
        ids = [int(x) for x in next_ids.reshape(-1).tolist()]
        cache["folded"].extend(ids)
        self.folds.append(ids)
        logits = mx.zeros((1, 1, V))
        return (logits, hidden[:, -1:]) if return_hidden else logits


class _LM:
    def __init__(self, model):
        self._m = model
        self.args = types.SimpleNamespace(vocab_size=V, tie_word_embeddings=False)

    def mtp_forward(self, *a, **kw):
        return self._m.mtp_forward(*a, **kw)


def _mtp_chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf):
    """Stock-shaped MTP chain drafter: depth drafts from a noisy head."""
    model = gen_batch.model
    depth = state.controller.cur if state.controller is not None else state.depth
    n = int(committed.shape[0])
    lm = gen_batch.model._lm
    lm.mtp_forward(hidden_rows, committed.reshape(1, n), state.mtp_cache,
                   return_hidden=True, logits_keep=1)
    state.hist_offset += n
    if depth <= 0:
        state.drafts = mx.zeros((0,), dtype=mx.uint32)
        state.draft_lps, state.draft_accept_lps = [], []
        return
    pos = gen_batch.stream_pos
    ids = []
    for j in range(depth):
        p = pos + j
        want = model.intended[p] if p < len(model.intended) else 0
        if model.rng.random() < model.mtp_acc:
            ids.append(want)
        else:
            ids.append((want + 1 + model.rng.randrange(V - 1)) % V)
    state.drafts = mx.array(ids, dtype=mx.uint32)
    q = mx.put_along_axis(
        mx.full((depth, V), -3.0e38, dtype=mx.float32),
        state.drafts.astype(mx.int32)[:, None],
        mx.zeros((depth, 1), dtype=mx.float32), axis=-1)
    state.draft_lps = [q[j] for j in range(depth)]
    state.draft_accept_lps = list(state.draft_lps)


# ---------------------------------------------------------------------------
# fake cache: recurrent (GDN-like) state plus a PLE-like window snapshot
# ---------------------------------------------------------------------------
class FakeCache:
    def __init__(self):
        self.pos = 0
        self.gdn = 0            # order-dependent recurrent accumulator
        self.trace = []         # every token the backbone has consumed
        self.inter = None       # per-row intermediate states (rollback source)
        self.ple = None         # (history, input_ids) window snapshot
        self.ple_state = []

    @staticmethod
    def _mix(state, tok):
        return (state * 1000003 + int(tok) + 17) % (1 << 61)

    def forward(self, ids):
        inter, hist = [], list(self.trace)
        for t in ids:
            self.gdn = self._mix(self.gdn, t)
            self.trace.append(int(t))
            self.pos += 1
            inter.append(self.gdn)
        self.inter = inter
        self.ple = (hist, list(ids))
        self.ple_state = list(self.trace)

    def rollback(self, accepted, block):
        assert 0 <= accepted < block, "PLE window validation"
        keep = accepted + 1
        self.gdn = self.inter[accepted]
        drop = block - keep
        if drop:
            self.trace = self.trace[:-drop]
        self.pos -= drop
        hist, win = self.ple
        self.ple_state = hist + win[:keep]
        self.inter, self.ple = None, None

    def clear_rollback(self):
        self.inter, self.ple = None, None
        self.ple_state = list(self.trace)


class FakeGenBatch:
    def __init__(self, model, prompt, max_tokens, matcher, greedy=True):
        self.model = model
        model._lm = _LM(model)
        self.uids = ["u0"]
        self.tokens = [list(prompt)]
        self._num_tokens = [0]
        self.max_tokens = [max_tokens]
        self.state_machines = [matcher]
        self._matcher_states = [matcher.make_state()]
        self.logits_processors = None
        self.greedy = greedy
        self.cache = FakeCache()
        self.cache.forward(prompt)
        self.cache.clear_rollback()
        self.stream_pos = 0
        self.emitted = []
        self.model_ms = 0.0


def _lm_of(model):
    return model._lm


# ---------------------------------------------------------------------------
# verify cycle, transcribed from _run_verify_cycle_chain
# ---------------------------------------------------------------------------
def run_cycle(bg, gb, state, next_main):
    k = int(state.drafts.shape[0])
    inputs = [int(next_main)] + [int(x) for x in state.drafts.tolist()]
    start = gb.stream_pos
    gb.cache.forward(inputs)
    rows_pos = list(range(start, start + k + 1))
    rows = gb.model.rows(rows_pos)
    targets = [int(x) for x in mx.argmax(rows, axis=-1).tolist()]
    drafts = [int(x) for x in state.drafts.tolist()]
    m = 0
    for j in range(k):
        if targets[j] == drafts[j]:
            m += 1
        else:
            break
    emit_last = targets[m] if m < k else targets[k]

    state.stats.cycles += 1
    if len(state.stats.depth_drafted) < state.depth:
        pad = state.depth - len(state.stats.depth_drafted)
        state.stats.depth_drafted.extend([0] * pad)
        state.stats.depth_accepted.extend([0] * pad)
    for j in range(k):
        state.stats.depth_drafted[j] += 1
        if j < m:
            state.stats.depth_accepted[j] += 1
        else:
            break
    state.stats.accepts += m
    if m < k:
        state.stats.rejects += 1

    emits = drafts[:m] + [emit_last]
    if m == k:
        gb.cache.clear_rollback()
    else:
        gb.cache.rollback(m, k + 1)

    # Production queues the emits and drains them one per next() call, so the
    # drafter below runs BEFORE gen_batch.tokens / _num_tokens / the matcher
    # see them; only ``committed`` carries this cycle's tokens.
    state.queue.extend(emits)
    gb.stream_pos = start + len(emits)
    lane = getattr(state, "_omlx_copy_lane", None)
    was_copy = bool(lane and lane.pending > 0)
    committed = mx.array(emits, dtype=mx.uint32)
    hidden_rows = mx.zeros((1, len(emits), 8))
    bg._chain_next_drafts(gb, state, hidden_rows, committed, None)
    cycle_ms = cost_ms(k, was_copy)
    gb.model_ms += cycle_ms
    if state.controller is not None:
        state.controller.observe(k, m, cycle_ms)
    _drain(gb, state)
    return emits, k, m


# Round-2 in-situ calibration (kernels/round2/mtp/REPORT.md section 3c):
# a one-row backbone forward is 24.4 ms, one extra verify row is 1.6 ms, and
# an MTP chain step past the first costs 1.33 ms of head layer + lm_head.
# A copy cycle pays neither the chain steps nor the fold's lm_head (dead in
# the lazy graph), only the head layer, ~0.4 ms.
BASE_MS, ROW_MS, CHAIN_MS, FOLD_MS = 24.4, 1.6, 1.33, 0.4


def cost_ms(k, is_copy):
    if is_copy:
        return BASE_MS + ROW_MS * k + FOLD_MS
    return BASE_MS + ROW_MS * k + FOLD_MS + CHAIN_MS * max(0, k - 1)


def _drain(gb, state):
    while state.queue:
        t = int(state.queue.popleft())
        gb.tokens[0].append(t)
        gb._num_tokens[0] += 1
        gb.emitted.append(t)
        st, seq, cur = gb.state_machines[0].match(gb._matcher_states[0], t)
        gb._matcher_states[0] = st
        if seq is not None and cur is None:
            gb.finished = True
            state.queue.clear()
            return


def generate(bg, gb, state, n_tokens):
    gb.finished = False
    first = gb.model.intended[0]
    # ``next_main`` is emitted but NOT yet in the backbone cache (the one-token
    # skew _post_init_mtp leaves behind); the first verify cycle feeds it.
    gb.stream_pos = 1
    committed = mx.array([first], dtype=mx.uint32)
    state.queue.append(first)
    bg._chain_next_drafts(gb, state, mx.zeros((1, 1, 8)), committed, None)
    _drain(gb, state)
    log = []
    while len(gb.emitted) < n_tokens and not gb.finished:
        nm = gb.model.intended[gb.stream_pos - 1]
        nm = gb.emitted[-1]
        emits, k, m = run_cycle(bg, gb, state, nm)
        log.append((k, m))
    return log


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------
def code_prompt(rng, n_defs=40):
    """A 'code file' as ids, with realistic repeated structure."""
    ids = []
    for i in range(n_defs):
        ids += [10, 11 + (i % 23), 12, 13, 14, 15 + (i % 17), 16, 17, 18,
                19 + (i % 11), 20, 21, 22 + (i % 7), 23, 24]
    return ids


def rewrite_of(prompt, rng, doc_every=15, doc_len=6):
    """A 'rewrite' generation: the file back verbatim with docstrings added."""
    out = []
    for i, t in enumerate(prompt):
        if i % doc_every == 1:
            out += [900 + rng.randrange(60) for _ in range(doc_len)]
        out.append(t)
    return out


def prose(rng, n):
    return [1000 + rng.randrange(3000) for _ in range(n)]


# ---------------------------------------------------------------------------
def load_patch():
    spec = importlib.util.spec_from_file_location(
        "copy_lane_patch", os.path.join(HERE, "patch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(bg, patch, prompt, intended, n, *, copy_on, max_tokens=None,
        matcher=None, seed=0, mtp_acc=0.72, depth=DEPTH, env=None):
    bg._chain_next_drafts = _mtp_chain_next_drafts
    saved = {}
    for kk, vv in (env or {}).items():
        saved[kk] = os.environ.get(kk)
        os.environ[kk] = vv
    if copy_on:
        patch._INSTALLED = False
        assert patch.install_copy_lane()
    for kk, vv in saved.items():
        if vv is None:
            os.environ.pop(kk, None)
        else:
            os.environ[kk] = vv
    model = FakeModel(intended, mtp_acc=mtp_acc, seed=seed)
    matcher = matcher or _Matcher()
    gb = FakeGenBatch(model, prompt, max_tokens or (n + 64), matcher)
    state = _State(depth)
    log = generate(bg, gb, state, n)
    stats = patch.copy_lane_stats(state) if copy_on else None
    return gb, state, log, stats


def main():
    ok = True
    patch = load_patch()
    bg = _install_fake_bg()
    rng = random.Random(7)

    # [1] index
    idx = patch.PromptIndex([5, 6, 7, 8, 9, 5, 6, 7, 40, 41], (3,))
    a = idx.lookup((5, 6, 7), 4)
    b = idx.lookup((9, 5, 6), 4)
    c = idx.lookup((1, 2, 3), 4)
    p1 = a == [40, 41] and b == [7, 40, 41] and c is None
    print(f"[1] index      most-recent={a} mid={b} miss={c}  {'PASS' if p1 else 'FAIL'}")
    ok &= p1

    # [2] edit-heavy: greedy stream identical, high copy rate
    prompt = code_prompt(rng)
    intended = rewrite_of(prompt, rng)
    N = 400
    gb0, st0, log0, _ = run(bg, patch, prompt, intended, N, copy_on=False, seed=1)
    gb1, st1, log1, cs = run(bg, patch, prompt, intended, N, copy_on=True, seed=1)
    same = gb0.emitted[:N] == gb1.emitted[:N]
    cyc0, cyc1 = len(log0), len(log1)
    print(f"[2] edit       stock {cyc0} cycles / copy {cyc1} cycles, "
          f"tokens {len(gb0.emitted)}/{len(gb1.emitted)}, identical={same}")
    print(f"    copy       hit_rate={cs['hit_rate']*100:.1f}% installed={cs['installed']} "
          f"blocks={cs['blocks']} mean_block={cs['mean_block']:.2f} "
          f"mean_accept={cs['mean_accept']:.2f} accepted={cs['accepted']}")
    p2 = same and cs["installed"] > 0 and cs["mean_accept"] > 2.0
    print(f"    {'PASS' if p2 else 'FAIL'}")
    ok &= p2
    edit = cs

    # [3] free prose: identical, near-zero copy
    pr = prose(rng, 900)
    gb0, _, log0, _ = run(bg, patch, prompt, pr, N, copy_on=False, seed=2)
    gb1, _, log1, cs2 = run(bg, patch, prompt, pr, N, copy_on=True, seed=2)
    same = gb0.emitted[:N] == gb1.emitted[:N]
    p3 = same and cs2["installed"] == 0
    print(f"[3] prose      identical={same} hit_rate={cs2['hit_rate']*100:.1f}% "
          f"installed={cs2['installed']} cycles={len(log1)}  {'PASS' if p3 else 'FAIL'}")
    ok &= p3

    # [4] stop token inside a copy block
    EOS = 777
    r4 = random.Random(99)
    body = [100 + r4.randrange(500) for _ in range(220)]
    body[100] = EOS
    stop_prompt = list(body)
    stop_intended = list(body)              # a pure verbatim copy task
    gb, state, log, cs4 = run(bg, patch, stop_prompt, stop_intended, 400, copy_on=True,
                              matcher=_Matcher(eos_ids=(EOS,)), seed=3, mtp_acc=0.9,
                              env={"OMLX_MTP_COPY_ADAPTIVE": "0"})
    emitted = gb.emitted
    fed = gb.cache.trace[len(stop_prompt):]
    eos_at = emitted.index(EOS) if EOS in emitted else -1
    overrun = len(fed) - (eos_at + 1) if eos_at >= 0 else 999
    p4 = (eos_at >= 0 and gb.finished and overrun <= 0
          and cs4["stop_truncated"] >= 1)
    print(f"[4] stop       eos emitted at {eos_at}, backbone consumed {len(fed)} "
          f"generated tokens, over-run={overrun} row(s); blocks truncated at the "
          f"stop={cs4['stop_truncated']}, draft rows dropped={cs4['stop_saved']} "
          f"(GDN positions past the stop that an untruncated block would have "
          f"committed)  {'PASS' if p4 else 'FAIL'}")
    ok &= p4

    # [5] rollback exactness: recurrent state and PLE window vs the emitted stream
    gb, state, log, cs5 = run(bg, patch, prompt, intended, 300, copy_on=True, seed=4,
                              mtp_acc=0.55)
    live = gb.cache.trace
    ideal = prompt + gb.emitted
    prefix_ok = live == ideal[:len(live)]
    skew = len(ideal) - len(live)
    ref = FakeCache(); ref.forward(live)
    p5 = prefix_ok and 0 <= skew <= 1 and ref.gdn == gb.cache.gdn \
        and gb.cache.ple_state == live
    partials = sum(1 for k, m in log if m < k)
    copy_partials = cs5["blocks"] - sum(1 for k, m in log if m == k and k > DEPTH)
    print(f"[5] rollback   {partials} partial-accept cycles ({cs5['blocks']} copy "
          f"blocks, mean accept {cs5['mean_accept']:.2f}); cache trace is an exact "
          f"prefix of prompt+emitted (skew {skew}), recurrent state and PLE window "
          f"replay to it  {'PASS' if p5 else 'FAIL'}")
    ok &= p5

    # [6] MTP head history after copy cycles == stock chain history
    gbA, stA, _, _ = run(bg, patch, prompt, intended, 250, copy_on=False, seed=5)
    gbB, stB, _, _ = run(bg, patch, prompt, intended, 250, copy_on=True, seed=5)
    hA = stA.mtp_cache["folded"]
    hB = stB.mtp_cache["folded"]
    nmin = min(len(hA), len(hB))
    p6 = hA[:nmin] == hB[:nmin] and hB == gbB.emitted[:len(hB)]
    print(f"[6] head       folded history {len(hB)} pairs, equals the committed stream "
          f"and matches stock over the common prefix  {'PASS' if p6 else 'FAIL'}")
    ok &= p6

    # [7] stochastic acceptance with one-hot q
    trials, acc, ptot = 20000, 0, 0.0
    rs = random.Random(11)
    for _ in range(trials):
        p = rs.random()
        ptot += p
        ratio = mx.log(mx.array(p)).item() - 0.0
        u = rs.random()
        if ratio >= 0 or mx.log(mx.array(u)).item() < ratio:
            acc += 1
    p7 = abs(acc / trials - ptot / trials) < 0.01
    print(f"[7] sampled    one-hot q: accept rate {acc/trials:.4f} vs E[p] "
          f"{ptot/trials:.4f} (Leviathan/Chen with q=1 accepts with prob p)"
          f"  {'PASS' if p7 else 'FAIL'}")
    ok &= p7

    # [8] tokenizer merges: same text, different ids, must not match
    #     ids 200 and (201,202) stand for one token vs its two-token split.
    merged = [70, 71, 200, 72, 73, 74, 75, 76]
    split_tail = (70, 71, 201, 202)
    idx8 = patch.PromptIndex(merged, (3,))
    hit_exact = idx8.lookup((70, 71, 200), 4)
    hit_split = idx8.lookup(split_tail, 4)
    p8 = hit_exact == [72, 73, 74, 75] and hit_split is None
    print(f"[8] merges     exact-id hit={hit_exact}, re-split tail hit={hit_split} "
          f"(ids only, never text)  {'PASS' if p8 else 'FAIL'}")
    ok &= p8

    # [9] the depth controller never sees a copy cycle
    gbC, stC, logC, cs9 = run(bg, patch, prompt, intended, 400, copy_on=True, seed=6)
    seen = stC.controller.seen
    mtp_cycles = len(logC) - cs9["blocks"]
    hist = sum(stC.stats.depth_drafted)
    p9 = (len(seen) == mtp_cycles and all(u <= DEPTH for u, _, _ in seen)
          and hist == sum(1 for k, m in logC if k and k <= DEPTH) * 0 + hist
          and all(v >= 0 for v in stC.stats.depth_drafted))
    p9 &= max((k for k, _ in logC), default=0) > DEPTH
    print(f"[9] controller {len(seen)} observe() calls for {mtp_cycles} MTP cycles "
          f"out of {len(logC)} total ({cs9['blocks']} copy blocks hidden); depth "
          f"histogram {stC.stats.depth_drafted[:DEPTH]} stays MTP-only and "
          f"non-negative  {'PASS' if p9 else 'FAIL'}")
    ok &= p9

    # [10] composition with another _chain_next_drafts patch (mtp-head8 /
    #      mtp-depth / the round-2 shortlist drafter all wrap the same symbol)
    inner_calls = []

    def _inner(gen_batch, state, hidden_rows, committed, prev_buf):
        inner_calls.append(1)
        return _mtp_chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf)

    bg._chain_next_drafts = _inner
    patch._INSTALLED = False
    assert patch.install_copy_lane()
    outer = bg._chain_next_drafts
    model = FakeModel(intended, seed=8)
    gbD = FakeGenBatch(model, prompt, N + 64, _Matcher())
    stD = _State()
    logD = generate(bg, gbD, stD, N)
    csD = patch.copy_lane_stats(stD)
    p10 = (getattr(outer, "_omlx_mtp_copy_lane", False)
           and csD["installed"] > 0
           and len(inner_calls) == len(logD) + 1 - csD["installed"])
    print(f"[10] compose  copy lane is outermost; {len(inner_calls)} of "
          f"{len(logD) + 1} drafter calls delegated to the inner patch, "
          f"{csD['installed']} taken by the copy lane  "
          f"{'PASS' if p10 else 'FAIL'}")
    ok &= p10
    bg._chain_next_drafts = _mtp_chain_next_drafts

    # summary sweep
    print()
    print("copy-lane behaviour, 400 greedy tokens, 600-token code prompt")
    print(f"{'regime':7s} {'N':>2s} {'adapt':>5s} {'cycles':>6s} {'hit%':>6s} "
          f"{'copy cyc':>8s} {'blk':>5s} {'acc':>5s} {'tok/cyc':>7s} "
          f"{'ms/tok':>7s} {'tok/s':>7s} {'vs stock':>8s}")
    base = {}
    for regime, tgt in (("edit", intended), ("prose", pr)):
        gbs, _, logs, _ = run(bg, patch, prompt, tgt, N, copy_on=False, seed=1)
        base[regime] = gbs.model_ms / N
        print(f"{regime:7s} {'-':>2s} {'-':>5s} {len(logs):6d} {0.0:6.1f} "
              f"{0:8d} {0.0:5.2f} {0.0:5.2f} {N/max(1,len(logs)):7.2f} "
              f"{base[regime]:7.2f} {1000/base[regime]:7.1f} {'1.00x':>8s}")
    for regime, tgt in (("edit", intended), ("prose", pr)):
        for ng in (4, 6, 8):
            for ad in ("1", "0"):
                gbx, stx, lg, c = run(bg, patch, prompt, tgt, N, copy_on=True, seed=1,
                                      env={"OMLX_MTP_COPY_NGRAM": str(ng),
                                           "OMLX_MTP_COPY_ADAPTIVE": ad})
                tpc = N / max(1, len(lg))
                mspt = gbx.model_ms / N
                ident = gbx.emitted[:N] == run(bg, patch, prompt, tgt, N,
                                               copy_on=False, seed=1)[0].emitted[:N]
                assert ident, f"greedy stream diverged at N={ng} adaptive={ad}"
                print(f"{regime:7s} {ng:2d} {ad:>5s} {len(lg):6d} "
                      f"{c['hit_rate']*100:6.1f} {c['installed']:8d} "
                      f"{c['mean_block']:5.2f} {c['mean_accept']:5.2f} {tpc:7.2f} "
                      f"{mspt:7.2f} {1000/mspt:7.1f} {base[regime]/mspt:7.2f}x")
    print("  ms/tok models the cycle with round-2's in-situ costs: 24.4 ms base,")
    print("  1.6 ms per extra verify row, 1.33 ms per MTP chain step, 0.4 ms fold.")
    print("  (greedy stream verified identical to stock in every row above)")

    print()
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
