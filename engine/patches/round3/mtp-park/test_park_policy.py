#!/usr/bin/env python3
"""Synthetic proof for the MTP park policy (OMLX_MTP_PARK_POLICY=1).

A miniature stand-in for oMLX's batch_generator is registered under the real
module name, so patch.py monkeypatches and then drives the SAME
``_DepthController``, the SAME ``_MtpParkState`` and the SAME
``_maybe_finish_mtp_reentry_probe`` (all three class/function bodies are
lifted verbatim out of
/Applications/oMLX.app/.../omlx/patches/mlx_lm_mtp/batch_generator.py).

Pure host-side simulation: no model, no mlx, no GPU. The fake model has an
acceptance profile that switches mid-stream between prose-like (d1 ~60%) and
code-like (d1 ~90%), and a cost model t(k) = t0 + k*D for an MTP cycle
against t0/tax for a standard step, calibrated from the round-3 workbench
logs (see analyze_logs.py).

Checks:

  [0] plumbing   the patch installs, wraps _best / should_exit / observe and
                 the three module functions, and composes on top of a
                 mtp-depth-style _best wrapper without losing its answer
  [1] identity   the greedy emitted stream is identical to stock, to a
                 never-park run and to a pure standard-decode run
  [2] shape      chosen depth over time, park and probe counts, modelled
                 tok/s, stock vs policy, on prose / code / mixed streams
  [3] backoff    the stock premature-probe-success path restarts the cooldown
                 at 128 tokens; the patch keeps the doubling
  [4] floors     breakeven_p1 matches the closed form tax*t1/t0 - 1

Run: ~/inference-server/kdev/bin/python test_park_policy.py
"""

import importlib.util
import math
import os
import random
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional  # noqa: F401 (exec'd code)

HERE = Path(__file__).resolve().parent
BG_NAME = "omlx.patches.mlx_lm_mtp.batch_generator"
OMLX_BG = Path(
    "/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_lm_mtp/"
    "batch_generator.py"
)

# Cost model, milliseconds. Calibrated from the round-3 workbench prose
# request at 64k context (analyze_logs.py): standard step 17.7 ms/token,
# MTP cycle 34.1 ms at mean k = 1.48, so t0 = tax*17.7 and D = 9.3.
TAX = 1.15
T_STD = 17.7
T0 = TAX * T_STD          # the in-loop depth-0 step, 20.4 ms
D_ROW = 9.3               # one extra verify row
T_REPRIME = 25.0          # _post_init_mtp's 1-token forward + fresh head cache
T_SHAPE = 40.0            # one-off Metal shape warmup on re-entry

PROSE = [0.60, 0.47, 0.30, 0.20, 0.14]   # conditional acceptance per position
CODE = [0.90, 0.82, 0.72, 0.62, 0.52]


# ---------------------------------------------------------------------------
# real code, lifted out of the shipped file
# ---------------------------------------------------------------------------
def _lift(src: str, pattern: str, ns: dict) -> None:
    m = re.search(pattern, src, re.S | re.M)
    if not m:
        raise RuntimeError("could not locate %r in %s" % (pattern, OMLX_BG))
    exec(compile(m.group(0), str(OMLX_BG), "exec"), ns)


def load_real():
    src = OMLX_BG.read_text()
    ns: dict = {
        "math": math, "Dict": Dict, "List": List, "Optional": Optional,
        "Any": Any, "dataclass": dataclass, "field": field,
        "_STD_TAX_MAX": 1.5,
        "logger": _Logger(),
        "_prefill_activity_recent": lambda: False,
        "_MTP_REENTRY_INITIAL_COOLDOWN_TOKENS": 128,
        "_MTP_REENTRY_MAX_COOLDOWN_TOKENS": 4096,
        "_MtpState": object,
    }
    _lift(src, r"^class _DepthController:.*?(?=^# Draft sampler)", ns)
    _lift(src, r"^@dataclass\nclass _MtpParkState:.*?(?=^def _mtp_park_state_for_batch)", ns)
    _lift(src, r"^def _mtp_park_state_for_batch.*?(?=^def _record_parked_standard_step)", ns)
    _lift(src, r"^def _record_parked_standard_step.*?(?=^def _maybe_finish_mtp_reentry_probe)", ns)
    _lift(src, r"^def _maybe_finish_mtp_reentry_probe.*?(?=^# ------)", ns)
    return ns


class _Logger:
    def __init__(self):
        self.lines: List[str] = []

    def _rec(self, fmt, *a):
        self.lines.append(fmt % a if a else fmt)

    info = debug = warning = _rec


# ---------------------------------------------------------------------------
# fake batch_generator
# ---------------------------------------------------------------------------
class GenBatch:
    def __init__(self, uid):
        self.uids = [uid]


class MtpState:
    def __init__(self, uid, depth, controller):
        self.uid = uid
        self.depth = depth
        self.controller = controller
        self.reentry_probe = False


def install_fake_bg(ns):
    for pkg in ("omlx", "omlx.patches", "omlx.patches.mlx_lm_mtp"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod
    bg = types.ModuleType(BG_NAME)
    for name in ("_DepthController", "_MtpParkState", "_mtp_park_state_for_batch",
                 "_record_parked_standard_step", "_maybe_finish_mtp_reentry_probe",
                 "_MTP_REENTRY_INITIAL_COOLDOWN_TOKENS",
                 "_MTP_REENTRY_MAX_COOLDOWN_TOKENS", "logger"):
        setattr(bg, name, ns[name])

    def _park_mtp_to_standard(gen_batch, state):
        """Structural copy of BG:2738-2765 with the cache surgery removed."""
        ps = bg._mtp_park_state_for_batch(gen_batch)
        if state.reentry_probe and ps is not None:
            ps.restart_after_failed_probe()
        else:
            ps = bg._MtpParkState(uid=state.uid)
            gen_batch._omlx_mtp_park_state = ps
        bg.logger.info("MTP[%s] parked for %d standard tokens before re-entry probe",
                       state.uid, ps.cooldown_tokens)
        gen_batch._omlx_mtp_state = None
        return True

    def _prepare_mtp_state_for_next(gen_batch):
        """Structural copy of BG:1199-1231: a fresh controller every re-entry."""
        state = getattr(gen_batch, "_omlx_mtp_state", None)
        if state is not None:
            return state
        ps = bg._mtp_park_state_for_batch(gen_batch)
        state = MtpState(gen_batch.uids[0], bg._MAX_DEPTH,
                         bg._DepthController(bg._MAX_DEPTH, exit_margin=TAX))
        gen_batch._omlx_mtp_state = state
        if ps is not None:
            state.reentry_probe = True
            bg.logger.info("MTP[%s] re-entry probe started after %d standard tokens",
                           state.uid, ps.cooldown_tokens)
        return state

    bg._park_mtp_to_standard = _park_mtp_to_standard
    bg._prepare_mtp_state_for_next = _prepare_mtp_state_for_next
    bg._MAX_DEPTH = 3
    sys.modules[BG_NAME] = bg
    sys.modules["omlx.patches.mlx_lm_mtp"].batch_generator = bg
    return bg


# ---------------------------------------------------------------------------
# the simulator: mirrors patched_next / _mtp_next
# ---------------------------------------------------------------------------
def regime(i, kind):
    if kind == "prose":
        return PROSE
    if kind == "code":
        return CODE
    # mixed: prose 0-399, code 400-799, prose 800+
    return CODE if 400 <= i < 800 else PROSE


def run(bg, n_tokens, kind, seed=7, max_depth=3):
    bg._MAX_DEPTH = max_depth
    rng = random.Random(seed)
    gb = GenBatch("u0")
    gb._omlx_mtp_state = None
    stream, ms = [], 0.0
    depth_trace = []          # (token index, depth or -1 for standard)
    parks = probes = 0
    std_tokens = mtp_tokens = 0
    while len(stream) < n_tokens:
        ps = bg._mtp_park_state_for_batch(gb)
        if ps is not None and not ps.probe_ready():
            stream.append(len(stream))
            depth_trace.append((len(stream) - 1, -1))
            ms += T_STD
            std_tokens += 1
            bg._record_parked_standard_step(gb)
            continue
        fresh = gb._omlx_mtp_state is None
        state = bg._prepare_mtp_state_for_next(gb)
        if fresh:
            ms += T_REPRIME + (T_SHAPE if ps is not None else 0.0)
            probes += 1 if ps is not None else 0
        ctl = state.controller
        k = int(ctl.cur)
        was_warmup = bool(ctl._warmup)
        a = regime(len(stream), kind)
        accepted = 0
        for j in range(k):
            if rng.random() < a[min(j, len(a) - 1)]:
                accepted += 1
            else:
                break
        emits = accepted + 1
        cycle_ms = T0 + k * D_ROW
        for _ in range(min(emits, n_tokens - len(stream))):
            stream.append(len(stream))
            depth_trace.append((len(stream) - 1, k))
        ms += cycle_ms
        mtp_tokens += emits
        ctl.observe(k, accepted, cycle_ms)
        bg._maybe_finish_mtp_reentry_probe(gb, state, was_warmup=was_warmup)
        if ctl.should_exit():
            bg._park_mtp_to_standard(gb, state)
            parks += 1
    return {
        "stream": stream, "ms": ms, "tok_s": 1000.0 * len(stream) / ms,
        "parks": parks, "probes": probes, "std": std_tokens, "mtp": mtp_tokens,
        "trace": depth_trace,
    }


def run_standard_only(n_tokens):
    return {"stream": list(range(n_tokens)), "ms": n_tokens * T_STD,
            "tok_s": 1000.0 / T_STD, "parks": 0, "probes": 0,
            "std": n_tokens, "mtp": 0, "trace": []}


# ---------------------------------------------------------------------------
def load_patch():
    spec = importlib.util.spec_from_file_location("mtp_park_patch", HERE / "patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_env(**kw):
    for k in list(os.environ):
        if k.startswith("OMLX_MTP_PARK_"):
            del os.environ[k]
    for k, v in kw.items():
        os.environ["OMLX_MTP_PARK_" + k.upper()] = str(v)


def depth_profile(trace, bucket=100):
    out = []
    for start in range(0, len(trace), bucket):
        chunk = [d for _, d in trace[start:start + bucket]]
        spec = [d for d in chunk if d >= 0]
        out.append((start, sum(spec) / len(spec) if spec else -1.0,
                    100.0 * (len(chunk) - len(spec)) / len(chunk)))
    return out


def main():
    fails = []
    ns = load_real()
    bg = install_fake_bg(ns)

    # a mtp-depth-style _best wrapper goes on FIRST, as in production
    ctrl = bg._DepthController
    orig_best = ctrl._best

    def conf_best(self):
        d = orig_best(self)
        return 0 if d == 0 else self.max_depth
    ctrl._best = conf_best
    ctrl._omlx_conf_depth = True

    patch = load_patch()

    # [0] plumbing --------------------------------------------------------
    set_env(policy=0)
    assert patch.install_park_policy() is False, "install without the flag"
    set_env(policy=1)
    ok = patch.install_park_policy()
    checks = [
        ("install returns True", ok),
        ("_best wrapped", getattr(ctrl, "_omlx_park_policy", False)),
        ("_park_mtp_to_standard wrapped",
         getattr(bg._park_mtp_to_standard, "_omlx_park_policy", False)),
        ("idempotent", patch.install_park_policy()),
        ("cooldown default raised", bg._MtpParkState("x").cooldown_tokens == 512),
    ]
    # the mtp-depth answer survives: a non-zero collapse stays at max_depth
    c = bg._DepthController(3, exit_margin=TAX)
    c._warmup = []
    c.t = {0: T0, 1: T0 + D_ROW, 2: T0 + 2 * D_ROW, 3: T0 + 3 * D_ROW}
    c.p = [0.9, 0.8, 0.7]
    checks.append(("composes with mtp-depth", c._best() == 3))
    print("[0] plumbing")
    for name, val in checks:
        print(f"    {'pass' if val else 'FAIL'}  {name}")
        if not val:
            fails.append(name)

    # [4] break-even ------------------------------------------------------
    print("[4] break-even floor")
    c = bg._DepthController(3, exit_margin=TAX)
    c.t = {0: T0, 1: T0 + D_ROW}
    closed = TAX * (T0 + D_ROW) / T0 - 1.0
    got = patch.breakeven_p1(c)
    good = abs(got - closed) < 1e-9
    print(f"    {'pass' if good else 'FAIL'}  p1_min = {got:.4f} "
          f"(closed form tax*t1/t0-1 = {closed:.4f})")
    if not good:
        fails.append("break-even")
    print(f"    prose d1 = {PROSE[0]:.2f} -> speculation "
          f"{'loses' if PROSE[0] < got else 'wins'};  "
          f"code d1 = {CODE[0]:.2f} -> "
          f"{'loses' if CODE[0] < got else 'wins'}")

    # [1] identity --------------------------------------------------------
    print("[1] greedy identity, 1200 tokens")
    ref = run_standard_only(1200)["stream"]
    rows = []
    set_env(policy=0)
    rows.append(("stock", run(bg, 1200, "mixed")))
    set_env(policy=1, min_depth=1, probe_cycles=32, tokens=512, max_probes=2)
    rows.append(("policy", run(bg, 1200, "mixed")))
    set_env(policy=1, min_depth=1, accept_floor=0.0)   # (a) never park
    rows.append(("never-park", run(bg, 1200, "mixed")))
    for name, r in rows:
        good = r["stream"] == ref
        print(f"    {'pass' if good else 'FAIL'}  {name}: {len(r['stream'])} tokens")
        if not good:
            fails.append("identity " + name)

    # [2] shape and modelled tok/s ---------------------------------------
    print("[2] chosen depth over time and modelled throughput")
    print(f"    {'stream':7s} {'policy':12s} {'tok/s':>7s} {'parks':>6s} "
          f"{'probes':>7s} {'std tok':>8s}")
    table = {}
    for kind in ("prose", "code", "mixed"):
        set_env(policy=0)
        s = run(bg, 1200, kind)
        set_env(policy=1, min_depth=1, probe_cycles=32, tokens=512, max_probes=2)
        p = run(bg, 1200, kind)
        set_env(policy=1, min_depth=1, accept_floor=0.0)
        a = run(bg, 1200, kind)
        std = run_standard_only(1200)
        table[kind] = (s, p, a, std)
        for label, r in (("stock", s), ("park policy", p), ("never park", a),
                         ("MTP off", std)):
            print(f"    {kind:7s} {label:12s} {r['tok_s']:7.1f} {r['parks']:6d} "
                  f"{r['probes']:7d} {r['std']:8d}")
    print()
    print("    mean chosen depth per 100 tokens, mixed stream "
          "(-1 = standard decoder)")
    s, p = table["mixed"][0], table["mixed"][1]
    sp, pp = depth_profile(s["trace"]), depth_profile(p["trace"])
    print(f"    {'tokens':>8s} {'stock d':>9s} {'stock std%':>11s} "
          f"{'policy d':>9s} {'policy std%':>12s}")
    for (i, sd, ss), (_, pd, pstd) in zip(sp, pp):
        print(f"    {i:8d} {sd:9.2f} {ss:11.0f} {pd:9.2f} {pstd:12.0f}")

    # [3] backoff ---------------------------------------------------------
    print("[3] cooldown backoff across a premature probe success")
    for flag, want in ((0, 128), (1, 512)):
        set_env(policy=flag, min_depth=1, probe_cycles=32, tokens=512,
                max_probes=9, sticky=1)
        r = run(bg, 1200, "prose", seed=11)
        cds = [int(x) for x in re.findall(r"parked for (\d+) standard",
                                          "\n".join(bg.logger.lines))]
        bg.logger.lines.clear()
        rising = all(b >= a for a, b in zip(cds, cds[1:])) if len(cds) > 1 else True
        first_ok = (cds[0] == want) if cds else False
        label = "stock" if flag == 0 else "policy"
        print(f"    {label:7s} cooldowns {cds}  first={want} "
              f"{'pass' if first_ok else 'FAIL'}  monotone "
              f"{'pass' if rising else 'FAIL'}")
        if not first_ok:
            fails.append(f"cooldown {label}")
        if flag == 1 and not rising:
            fails.append("backoff monotone")

    # [5] knob sweep ------------------------------------------------------
    print("[5] knob sweep, modelled tok/s over 1200 tokens")
    print(f"    {'tokens':>7s} {'probe':>6s} {'probes':>7s}  "
          f"{'prose':>6s} {'mixed':>6s} {'code':>6s}")
    for tok in (128, 256, 512, 1024):
        for pc in (8, 32):
            for mp in (2, 0):
                set_env(policy=1, min_depth=1, probe_cycles=pc, tokens=tok,
                        max_probes=mp)
                r = [sum(run(bg, 1200, k, seed=sd)["tok_s"]
                         for sd in (3, 7, 11, 17, 23)) / 5.0
                     for k in ("prose", "mixed", "code")]
                print(f"    {tok:7d} {pc:6d} {mp if mp else 999:7d}  "
                      f"{r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f}")

    print()
    if fails:
        print("FAILURES:", ", ".join(fails))
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
