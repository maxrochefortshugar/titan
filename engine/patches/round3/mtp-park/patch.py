# SPDX-License-Identifier: Apache-2.0
"""Park-and-probe policy for the oMLX MTP depth controller (qwen4_exp).

OMLX_MTP_PARK_POLICY=1 changes *when* a sequence leaves the MTP loop for the
standard decoder and *how* it comes back. It changes nothing about drafting,
verification, acceptance or emission, so greedy output is untouched.

Stock behaviour, all in
/Applications/oMLX.app/.../omlx/patches/mlx_lm_mtp/batch_generator.py (BG):

  BG:2093-2100  _DepthController._score / _best pick a depth every cycle
  BG:2101-2110  _speculation_losing: best speculative score < score(0)*1.15
  BG:1911       EXIT_STREAK = 16 losing decisions in a row
  BG:2820-2829  _mtp_next calls _park_mtp_to_standard on should_exit()
  BG:2738-2765  _park_mtp_to_standard: feed next_main, drop MTP state,
                create _MtpParkState(cooldown 128 tokens)
  BG:777-779    _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS = 128, MAX = 4096
  BG:398-412    _mtp_common_eligible refuses MTP while the cooldown runs
  BG:1208-1224  the first eligible call builds a FRESH controller and marks
                state.reentry_probe
  BG:832-854    _maybe_finish_mtp_reentry_probe deletes the park state as
                soon as one post-warmup cycle is not losing

Four measured problems (see REPORT.md section 2):

  1. A fresh controller restarts p=[0.6]*d and t={}, so a probe spends its
     first max_depth+3 cycles on a warmup sweep and then estimates acceptance
     from a handful of samples. Field probes lasted 3 to 22 cycles.
  2. Because BG:832 clears the park state on the FIRST non-losing post-warmup
     decision, a probe that is 4 cycles old can "succeed", and the next park
     therefore starts a fresh 128-token cooldown instead of 256. The
     exponential backoff at BG:800-806 never engages. Observed live.
  3. Re-entry pays a 1-token backbone forward (_post_init_mtp, BG:2484), a
     fresh MTP head cache and Metal shape re-specialisation for M=2..d+1.
  4. Depth 0 is reachable whenever the score says so, including on the noisy
     prior, so a sequence can park before acceptance has been measured.

What this patch does instead:

  * a depth floor: _best never returns 0 unless the depth-1 acceptance EMA is
    genuinely below the break-even that makes speculation lose;
  * a break-even that is stated, not implied: p1_min = tax * t[1]/t[0] - 1,
    with tax the controller's own measured loop tax (EXIT_MARGIN);
  * a probe window long enough to estimate acceptance (default 32 post-warmup
    cycles) before the probe may either succeed or park again, with a hard
    abort when acceptance is far below the floor;
  * a longer, genuinely sticky cooldown (default 512 tokens, doubling across
    parks even when a probe declared success in between);
  * MTP off for the remainder of the request after N failed probes.

Env vars

  OMLX_MTP_PARK_POLICY       0     master gate
  OMLX_MTP_PARK_MIN_DEPTH    1     lowest depth _best may return (0 = stock)
  OMLX_MTP_PARK_PROBE_CYCLES 32    post-warmup cycles a probe must run
  OMLX_MTP_PARK_TOKENS       512   initial cooldown, standard tokens
  OMLX_MTP_PARK_MAX_TOKENS   8192  cooldown ceiling
  OMLX_MTP_PARK_ACCEPT_FLOOR -1    static p1 floor; <0 = live break-even
  OMLX_MTP_PARK_MAX_PROBES   2     failed probes before MTP is off for good
  OMLX_MTP_PARK_STICKY       1     carry the backoff across a "successful" probe
  OMLX_MTP_PARK_TRACE        0     log every park/probe decision

Install: import time, one call, idempotent.

    install_park_policy() -> bool

Composition. This module wraps _DepthController._best, .should_exit and
.observe, plus the module functions _park_mtp_to_standard,
_maybe_finish_mtp_reentry_probe and _prepare_mtp_state_for_next. It does NOT
touch _chain_next_drafts, so round-2's shortlist drafter, round-3 mtp-depth
and round-3 copy-lane keep that function to themselves.

  * mtp-depth also wraps _best (collapsing it to {0, max_depth}). Install
    THIS patch AFTER install_conf_depth() so the floor is applied to
    mtp-depth's answer: a collapsed 0 becomes the best speculative depth
    instead of a park.
  * copy-lane wraps observe to hide copy cycles. The observe wrapper here
    checks the same _omlx_copy_active marker, so a copy cycle is not counted
    toward the probe window whichever order the two installs run in.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BG = "omlx.patches.mlx_lm_mtp.batch_generator"

_INSTALLED = False


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def park_enabled() -> bool:
    return os.environ.get("OMLX_MTP_PARK_POLICY", "0") not in ("0", "", "false")


class _Cfg:
    """Read once per call so the workbench can flip knobs between runs."""

    @property
    def min_depth(self) -> int:
        return max(0, _env_int("OMLX_MTP_PARK_MIN_DEPTH", 1))

    @property
    def probe_cycles(self) -> int:
        return max(0, _env_int("OMLX_MTP_PARK_PROBE_CYCLES", 32))

    @property
    def park_tokens(self) -> int:
        return max(1, _env_int("OMLX_MTP_PARK_TOKENS", 512))

    @property
    def max_tokens(self) -> int:
        return max(1, _env_int("OMLX_MTP_PARK_MAX_TOKENS", 8192))

    @property
    def accept_floor(self) -> float:
        return _env_float("OMLX_MTP_PARK_ACCEPT_FLOOR", -1.0)

    @property
    def max_probes(self) -> int:
        return max(0, _env_int("OMLX_MTP_PARK_MAX_PROBES", 2))

    @property
    def sticky(self) -> bool:
        return _env_int("OMLX_MTP_PARK_STICKY", 1) != 0

    @property
    def trace(self) -> bool:
        return _env_int("OMLX_MTP_PARK_TRACE", 0) != 0


CFG = _Cfg()

# Effectively "no more MTP on this request". Kept finite so the standard
# decoder's own bookkeeping never sees a negative or absurd counter.
_OFF_TOKENS = 1 << 30

# Park history survives the deletion of _MtpParkState (BG:849), which is what
# makes the backoff stick across a probe that declared success and then lost
# again. It lives on the GenerationBatch beside oMLX's own markers and is
# keyed by uid, so a reused GenerationBatch cannot inherit another request's
# escalation and nothing leaks when the request ends.
def _park_hist(gen_batch: Any, uid: Any) -> "tuple[int, int]":
    rec = getattr(gen_batch, "_omlx_park_hist", None)
    if not rec or rec[0] != uid:
        return (0, 0)
    return (int(rec[1]), int(rec[2]))


def _remember(gen_batch: Any, uid: Any, cooldown: int, probes: int) -> None:
    try:
        gen_batch._omlx_park_hist = (uid, int(cooldown), int(probes))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# the break-even
# ---------------------------------------------------------------------------


def breakeven_p1(ctl: Any) -> float:
    """Depth-1 acceptance below which speculation loses to the standard step.

    One MTP cycle at depth 1 costs ``t[1]`` and yields ``1 + p1`` tokens. The
    standard decoder emits one token per ``t[0] / tax``: ``t[0]`` is measured
    INSIDE the MTP loop, so it carries the synchronous host round-trip the
    standard step pipelines away, and ``tax`` (the controller's EXIT_MARGIN,
    measured per machine by the std-tax probe at BG:1728) removes it.
    Speculation pays while

        (1 + p1) / t[1]  >  tax / t[0]      i.e.   p1 > tax * t[1]/t[0] - 1
    """
    static = CFG.accept_floor
    if static >= 0.0:
        return max(0.0, min(0.95, static))
    try:
        t0 = float(ctl._t_est(0))
        t1 = float(ctl._t_est(1))
    except Exception:  # noqa: BLE001
        return 0.0
    if t0 <= 0.0 or t1 <= 0.0:
        return 0.0
    tax = float(getattr(ctl, "EXIT_MARGIN", 1.15) or 1.15)
    return max(0.0, min(0.95, tax * t1 / t0 - 1.0))


def _p1(ctl: Any) -> float:
    p = getattr(ctl, "p", None)
    if not p:
        return 1.0
    return float(p[0])


def _zero_allowed(ctl: Any) -> bool:
    """True when depth 0 (and therefore a park) is defensible."""
    if CFG.min_depth <= 0:
        return True  # stock escape hatch
    if getattr(ctl, "_warmup", None):
        return False  # never park on the prior
    return _p1(ctl) < breakeven_p1(ctl)


def _best_speculative(ctl: Any) -> int:
    lo = max(1, CFG.min_depth)
    hi = int(getattr(ctl, "max_depth", 1))
    best_d, best_s = lo, -1.0
    for d in range(lo, max(lo, hi) + 1):
        try:
            s = float(ctl._score(d))
        except Exception:  # noqa: BLE001
            continue
        if s > best_s:
            best_d, best_s = d, s
    return best_d


# ---------------------------------------------------------------------------
# controller wrappers
# ---------------------------------------------------------------------------


def _find_controller(bg) -> Optional[type]:
    ctrl = getattr(bg, "_DepthController", None)
    if ctrl is not None:
        return ctrl
    for name in dir(bg):
        obj = getattr(bg, name)
        if isinstance(obj, type) and hasattr(obj, "_score") and hasattr(obj, "observe"):
            return obj
    return None


def _wrap_controller(ctrl: type) -> bool:
    if getattr(ctrl, "_omlx_park_policy", False):
        return True

    orig_best = ctrl._best
    orig_exit = ctrl.should_exit
    orig_observe = ctrl.observe

    def _best(self):
        d = orig_best(self)
        if not park_enabled():
            return d
        floor = CFG.min_depth
        if floor <= 0:
            return d
        if d == 0 and not _zero_allowed(self):
            d = _best_speculative(self)
            if CFG.trace:
                logger.info(
                    "MTP park: depth 0 refused, p1=%.3f floor=%.3f -> depth %d",
                    _p1(self), breakeven_p1(self), d,
                )
            return d
        if d == 0:
            return 0
        return max(d, floor)

    def should_exit(self):
        if not park_enabled():
            return orig_exit(self)
        if not _zero_allowed(self):
            return False
        want = CFG.probe_cycles
        if getattr(self, "_omlx_park_probe", False) and want > 0:
            seen = int(getattr(self, "_omlx_park_cycles", 0))
            if seen < want:
                # Hopeless probes still die early: half the break-even is far
                # enough below the line that no sample size will rescue it.
                if not (seen >= 8 and _p1(self) < 0.5 * breakeven_p1(self)):
                    return False
        return orig_exit(self)

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        # copy-lane marks a copy cycle by setting _omlx_copy_active; its own
        # observe wrapper consumes the flag, so read it without clearing.
        copy_cycle = bool(getattr(self, "_omlx_copy_active", False))
        r = orig_observe(self, used, accepted, cycle_ms, time_sample=time_sample)
        if park_enabled() and not copy_cycle and not getattr(self, "_warmup", None):
            self._omlx_park_cycles = int(getattr(self, "_omlx_park_cycles", 0)) + 1
        return r

    ctrl._best = _best
    ctrl.should_exit = should_exit
    ctrl.observe = observe
    ctrl._omlx_park_policy = True
    return True


# ---------------------------------------------------------------------------
# park / probe bookkeeping
# ---------------------------------------------------------------------------


def _wrap_module_funcs(bg) -> bool:
    orig_park = getattr(bg, "_park_mtp_to_standard", None)
    orig_finish = getattr(bg, "_maybe_finish_mtp_reentry_probe", None)
    orig_prepare = getattr(bg, "_prepare_mtp_state_for_next", None)
    if orig_park is None or orig_finish is None or orig_prepare is None:
        return False
    if getattr(orig_park, "_omlx_park_policy", False):
        return True

    def _park_mtp_to_standard(gen_batch, state):
        if not park_enabled():
            return orig_park(gen_batch, state)
        was_probe = bool(getattr(state, "reentry_probe", False))
        uid = getattr(state, "uid", None)
        ok = orig_park(gen_batch, state)
        if not ok:
            return ok
        ps = getattr(gen_batch, "_omlx_mtp_park_state", None)
        if ps is None:
            return ok
        prev_cd, probes = _park_hist(gen_batch, uid)
        if was_probe:
            probes += 1
        base = max(CFG.park_tokens, int(getattr(ps, "cooldown_tokens", 0)))
        if CFG.sticky and prev_cd:
            base = max(base, min(CFG.max_tokens, prev_cd * 2))
        base = min(CFG.max_tokens, base)
        if CFG.max_probes and probes >= CFG.max_probes:
            base = _OFF_TOKENS
        ps.cooldown_tokens = base
        ps.tokens_remaining = base
        _remember(gen_batch, uid, min(base, CFG.max_tokens), probes)
        if CFG.trace or base == _OFF_TOKENS:
            logger.info(
                "MTP park policy[%s]: cooldown %d tokens (failed probes %d%s)",
                uid, base, probes,
                ", MTP off for this request" if base == _OFF_TOKENS else "",
            )
        return ok

    def _maybe_finish_mtp_reentry_probe(gen_batch, state, *, was_warmup):
        if not park_enabled():
            return orig_finish(gen_batch, state, was_warmup=was_warmup)
        ctl = getattr(state, "controller", None)
        want = CFG.probe_cycles
        if (
            ctl is not None
            and want > 0
            and getattr(state, "reentry_probe", False)
            and int(getattr(ctl, "_omlx_park_cycles", 0)) < want
        ):
            # Not enough evidence yet. Keeping the park state alive is the
            # point: it is what makes the next park double the cooldown.
            return False
        return orig_finish(gen_batch, state, was_warmup=was_warmup)

    def _prepare_mtp_state_for_next(gen_batch):
        state = orig_prepare(gen_batch)
        if park_enabled() and state is not None:
            ctl = getattr(state, "controller", None)
            if ctl is not None and getattr(state, "reentry_probe", False):
                ctl._omlx_park_probe = True
        return state

    _park_mtp_to_standard._omlx_park_policy = True
    bg._park_mtp_to_standard = _park_mtp_to_standard
    bg._maybe_finish_mtp_reentry_probe = _maybe_finish_mtp_reentry_probe
    bg._prepare_mtp_state_for_next = _prepare_mtp_state_for_next
    return True


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def install_park_policy() -> bool:
    """Import-time install. Returns False and leaves the stock path alone."""
    global _INSTALLED
    if not park_enabled():
        return False
    if _INSTALLED:
        return True
    try:
        bg = importlib.import_module(_BG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP park policy: %s unavailable (%s)", _BG, exc)
        return False
    ctrl = _find_controller(bg)
    if ctrl is None:
        logger.warning("MTP park policy: _DepthController not found")
        return False
    if not _wrap_controller(ctrl):
        return False
    if not _wrap_module_funcs(bg):
        logger.warning("MTP park policy: park helpers not found; floor only")
    # _MTP_REENTRY_MAX_COOLDOWN_TOKENS is read at call time inside
    # _MtpParkState.restart_after_failed_probe, so rebinding the module global
    # is enough. The INITIAL constant is only a dataclass default, baked into
    # the generated __init__ at class creation, so rewrite that too: otherwise
    # BG:2760's log line reports 128 while the effective cooldown is 512.
    try:
        bg._MTP_REENTRY_INITIAL_COOLDOWN_TOKENS = CFG.park_tokens
        bg._MTP_REENTRY_MAX_COOLDOWN_TOKENS = CFG.max_tokens
        ps_cls = getattr(bg, "_MtpParkState", None)
        init = getattr(ps_cls, "__init__", None)
        names = [f for f in getattr(ps_cls, "__dataclass_fields__", {})]
        if (
            init is not None
            and not getattr(ps_cls, "_omlx_park_policy", False)
            and getattr(init, "__defaults__", None) is not None
            and names[-2:] == ["cooldown_tokens", "tokens_remaining"]
            and len(init.__defaults__) == 2
        ):
            def _park_state_init(self, uid, *a, **kw):
                if not a and not kw and park_enabled():
                    return init(self, uid, CFG.park_tokens, CFG.park_tokens)
                return init(self, uid, *a, **kw)

            ps_cls.__init__ = _park_state_init
            ps_cls._omlx_park_policy = True
    except Exception:  # noqa: BLE001
        pass
    _INSTALLED = True
    logger.info(
        "MTP park policy: min_depth=%d probe_cycles=%d park_tokens=%d "
        "max_probes=%d floor=%s",
        CFG.min_depth, CFG.probe_cycles, CFG.park_tokens, CFG.max_probes,
        "live break-even" if CFG.accept_floor < 0 else f"{CFG.accept_floor:.2f}",
    )
    return True


def install() -> bool:
    return install_park_policy()
