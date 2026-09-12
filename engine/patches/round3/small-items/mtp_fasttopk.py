# SPDX-License-Identifier: Apache-2.0
"""Route the MTP shortlist drafter's top-K through the fused Metal kernel.

``kernels/round2/mtp/patch.py:105`` ``_refresh_shortlist`` builds the shortlist
with

    ids = mx.argpartition(logits_1d, kth=v - k, axis=-1)[..., -k:]

over the 248,320-wide lm_head row, once per decode cycle.  That partition is
the single most expensive op in the shortlist machinery (0.27 to 0.38 ms
against 0.027 ms for ``mx.argmax`` over the same array).

Rather than copy the eight lines of gather code that follow it, this hook
exploits the fact that ``_refresh_shortlist`` takes its ``mx`` module as its
first argument: it wraps the function with one that passes a proxy whose
``argpartition`` is the fused kernel and whose every other attribute is the
real ``mlx.core``.  The gather, the -inf row and the eval are untouched, and
the shortlist is the same set of candidate ids.

round2/mtp/patch.py is not modified.  The wrapper attaches to the module
object the bootstrap already loaded, so this install has to run AFTER
``OMLX_ROUND2_IMPORT_PATCHES`` has loaded round2/mtp/patch.py; if that module
is not loaded yet the install returns False and the stock path stays.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys

logger = logging.getLogger(__name__)

ENV_ENABLE = "OMLX_MTP_SHORTLIST_FASTTOPK"
MTP_PATCH = os.path.realpath(
    os.path.expanduser("~/inference-server/kernels/round2/mtp/patch.py")
)

_STATE = {"installed": False, "calls": 0, "fallbacks": 0}


def _here():
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


class _Tail:
    """Stands in for the argpartition result until its ``[..., -k:]`` slice."""

    __slots__ = ("_ids", "_src", "_kth", "_axis", "_k")

    def __init__(self, ids, src, kth, axis, k):
        self._ids, self._src, self._kth, self._axis, self._k = ids, src, kth, axis, k

    def _expected(self, key) -> bool:
        return (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] is Ellipsis
            and isinstance(key[1], slice)
            and key[1].start == -self._k
            and key[1].stop is None
            and key[1].step is None
        )

    def __getitem__(self, key):
        if self._expected(key):
            return self._ids
        import mlx.core as mx

        _STATE["fallbacks"] += 1
        return mx.argpartition(self._src, kth=self._kth, axis=self._axis)[key]


class _MXProxy:
    """``mlx.core`` with one op replaced."""

    def __init__(self, mx):
        object.__setattr__(self, "_mx", mx)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_mx"), name)

    def argpartition(self, a, kth=None, axis=-1):
        mx = object.__getattribute__(self, "_mx")
        v = int(a.shape[axis])
        k = v - int(kth)
        last = axis in (-1, a.ndim - 1)
        if not last or k <= 0 or k > v or a.size != v:
            return mx.argpartition(a, kth=kth, axis=axis)
        _here()
        from topk import fast_topk

        _STATE["calls"] += 1
        _, ids = fast_topk(a, k)
        return _Tail(ids, a, kth, axis, k)


def _mtp_modules():
    out = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            if os.path.realpath(f) != MTP_PATCH:
                continue
        except Exception:                                    # noqa: BLE001
            continue
        if hasattr(mod, "_refresh_shortlist"):
            out.append(mod)
    return out


def stats() -> dict:
    return dict(_STATE)


def install() -> bool:
    """Idempotent; returns False and leaves the stock path when preconditions fail.

    Import time, ordered after round2/mtp/patch.py.
    """
    if _STATE["installed"]:
        return True
    if os.environ.get(ENV_ENABLE, "0") != "1":
        return False
    _here()
    try:
        from topk import self_check
    except Exception:                                        # noqa: BLE001
        logger.warning("fused top-K: kernel module missing", exc_info=True)
        return False
    try:
        if not self_check():
            logger.warning("fused top-K self-check failed; shortlist unchanged")
            return False
    except Exception:
        logger.warning("fused top-K self-check raised", exc_info=True)
        return False

    mods = _mtp_modules()
    if not mods:
        logger.warning("fused top-K: round2/mtp/patch.py is not loaded; install "
                       "this after OMLX_ROUND2_IMPORT_PATCHES")
        return False

    done = False
    for mod in mods:
        orig = mod._refresh_shortlist
        if getattr(orig, "_omlx_fast_topk", False):
            done = True
            continue
        try:
            src = inspect.getsource(orig)
        except Exception:                                    # noqa: BLE001
            src = ""
        if "mx.argpartition" not in src or "[..., -k:]" not in src:
            logger.warning("fused top-K: _refresh_shortlist no longer selects "
                           "with argpartition; leaving it alone")
            continue

        def wrapped(mx, state_sl, head, logits_1d, k, _orig=orig):
            return _orig(_MXProxy(mx), state_sl, head, logits_1d, k)

        wrapped._omlx_fast_topk = True
        wrapped._omlx_original = orig
        mod._refresh_shortlist = wrapped
        done = True

    _STATE["installed"] = done
    if done:
        logger.info("MTP shortlist top-K routed through the fused Metal kernel")
    return done


__all__ = ["install", "stats"]
