#!/usr/bin/env python3
"""Env-gated per-cycle decode profiler for the qwen4_exp MTP path on oMLX.

Off by default.  ``install()`` returns False and touches nothing unless
``OMLX_DECODE_PROFILE=1``.  With the flag on it wraps a small set of module
functions and model classes with ``time.perf_counter`` brackets, counts every
host sync (``mx.eval`` / ``mx.async_eval`` / ``mx.synchronize`` /
``mx.array.tolist`` / ``mx.array.item``), estimates bytes read per stage, and
writes one JSON file per request under ``OMLX_DECODE_PROFILE_DIR``
(default ``~/inference-server/staging/decode-profile``) plus a compact
``DPROF[...]`` summary line next to the stock ``MTP[...]`` line.

Wall time on a lazy runtime is dispatch time plus whatever queue backpressure
the caller happens to absorb.  ``OMLX_DECODE_PROFILE_SYNC=1`` adds an
``mx.synchronize()`` at the end of each instrumented sub-stage so the numbers
become device time.  Sync mode serialises the pipeline and makes the whole
request slower: it is for attribution only, never for a throughput number.

Enable at IMPORT time (it swaps module-level functions):

    OMLX_ROUND2_IMPORT_PATCHES="...,round4/decode-profile/patch.py:install,\
    $R3/copy-lane/patch.py:install_copy_lane"

Install it after the other ``_chain_next_drafts`` wrappers and before
copy-lane, which must stay outermost.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_INSTALLED = False
_SYNC = False
_MAX_CYCLES = 4096
_OUTDIR = Path("~/inference-server/staging/decode-profile").expanduser()

# ------------------------------------------------------------------ env
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("OMLX_DECODE_PROFILE")


# ------------------------------------------------- byte model (audit A)
# Bytes a single backbone forward reads that do not scale with rows.  Sources:
# kernels/AUDIT-2026-09-12.md section A ("a decode forward reads 4.20 GB of
# text-path weights").  Override any of them from the environment if the
# checkpoint changes.
def _fenv(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


BYTES = {
    "gdn": _fenv("OMLX_DPROF_B_GDN", 1.25e9),          # 36 GDN layers
    "qsa": _fenv("OMLX_DPROF_B_QSA", 0.40e9),          # 12 QSA layers, weights only
    "hyper": _fenv("OMLX_DPROF_B_HYPER", 0.34e9),      # 96 hyper-connection blocks
    "shared": _fenv("OMLX_DPROF_B_SHARED", 0.23e9),    # shared expert
    "router": _fenv("OMLX_DPROF_B_ROUTER", 0.12e9),
    "lm_head": _fenv("OMLX_DPROF_B_HEAD", 0.63e9),     # 248320 x 2560, 8-bit
    "expert": _fenv("OMLX_DPROF_B_EXPERT", 2.765e6),   # 3 x 640 x 2560 at 4-bit gs64
    "ngram_row": _fenv("OMLX_DPROF_B_NGRAM_ROW", 100.0),  # packed row stride
    "read_ceiling": _fenv("OMLX_DPROF_READ_GBS", 718.0),
}
N_MOE_LAYERS = int(_fenv("OMLX_DPROF_MOE_LAYERS", 48))
N_EXPERTS = int(_fenv("OMLX_DPROF_EXPERTS", 512))
TOP_K = int(_fenv("OMLX_DPROF_TOPK", 10))


def expected_distinct_experts(rows: int) -> float:
    """Experts touched by ``rows`` independent top-k draws out of N_EXPERTS."""
    if rows <= 0:
        return 0.0
    p_miss = (1.0 - TOP_K / N_EXPERTS) ** rows
    return N_EXPERTS * (1.0 - p_miss)


# ------------------------------------------------------------ recording
class _Run:
    __slots__ = ("uid", "t0", "cycles", "dropped")

    def __init__(self, uid):
        self.uid = uid
        self.t0 = time.time()
        self.cycles = []
        self.dropped = 0


_runs: dict = {}
_cur = None          # the cycle record being filled
_phase = "idle"      # coarse phase, used to attribute host syncs
_mx = None           # mlx.core
_orig = {}           # saved originals


def _new_cycle():
    return {
        "phase_ms": {},
        "sub_ms": {},
        "sub_n": {},
        "sync_ms": {},
        "sync_n": {},
        "rows_ngram": 0,
        "lm_head_rows": [],
        "expert_rows": [],
    }


def _add(bucket: str, key: str, value: float) -> None:
    if _cur is None:
        return
    d = _cur[bucket]
    d[key] = d.get(key, 0.0) + value


def _bump(bucket: str, key: str) -> None:
    if _cur is None:
        return
    d = _cur[bucket]
    d[key] = d.get(key, 0) + 1


class _Stage:
    """Time one nested sub-stage; optionally force device completion.

    The key is prefixed with the phase unless the phase is the verify forward,
    so that the same module called from the draft chain (the MTP head reuses a
    decoder layer, the MoE block, the embedding and the vocabulary head) lands
    in its own bucket instead of inflating the backbone's.
    """

    __slots__ = ("name", "t", "ph")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.ph = _phase
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _SYNC and "synchronize" in _orig:
            _orig["synchronize"]()
        dt = (time.perf_counter() - self.t) * 1000.0
        key = self.name if self.ph in ("verify", "pre") else f"{self.ph}.{self.name}"
        _add("sub_ms", key, dt)
        _bump("sub_n", key)
        return False


# ------------------------------------------------------- sync wrappers
def _install_sync_counters(mx):
    def timed(name, fn, method=False):
        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = (time.perf_counter() - t) * 1000.0
                if _cur is not None:
                    key = f"{_phase}.{name}"
                    _add("sync_ms", key, dt)
                    _bump("sync_n", key)
        return wrapper

    _orig["eval"] = mx.eval
    _orig["async_eval"] = mx.async_eval
    _orig["synchronize"] = mx.synchronize
    _orig["tolist"] = mx.array.tolist
    _orig["item"] = mx.array.item
    mx.eval = timed("eval", _orig["eval"])
    mx.async_eval = timed("async_eval", _orig["async_eval"])
    mx.synchronize = timed("synchronize", _orig["synchronize"])
    mx.array.tolist = timed("tolist", _orig["tolist"])
    mx.array.item = timed("item", _orig["item"])


def _remove_sync_counters(mx):
    if "eval" not in _orig:
        return
    mx.eval = _orig["eval"]
    mx.async_eval = _orig["async_eval"]
    mx.synchronize = _orig["synchronize"]
    mx.array.tolist = _orig["tolist"]
    mx.array.item = _orig["item"]


# -------------------------------------------------- model-side wrapping
_model_wrapped = False


def _wrap_call(cls, name: str, tag_fn):
    """Wrap ``cls.__call__`` once, timing it under the name ``tag_fn(self)``."""
    if getattr(cls, "_dprof_wrapped", None) == name:
        return False
    stock = cls.__call__

    def wrapper(self, *a, **k):
        tag = tag_fn(self)
        if _cur is None or tag is None:
            return stock(self, *a, **k)
        with _Stage(tag):
            return stock(self, *a, **k)

    cls.__call__ = wrapper
    cls._dprof_wrapped = name
    cls._dprof_stock_call = stock
    return True


def _kv_bytes(prompt_cache) -> float:
    total = 0.0
    for c in prompt_cache or ():
        nb = getattr(c, "nbytes", None)
        try:
            total += float(nb() if callable(nb) else (nb or 0))
        except Exception:  # noqa: BLE001
            pass
    return total


def _wrap_model(model) -> None:
    """Wrap the layer classes of a live qwen4_exp model.  Idempotent."""
    global _model_wrapped
    if _model_wrapped:
        return
    _model_wrapped = True
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", None)
    layers = getattr(inner, "layers", None) or []
    if not layers:
        logger.warning("decode-profile: no decoder layers found; backbone split disabled")
        return

    # decoder layer: GDN vs attention
    _wrap_call(
        type(layers[0]),
        "layer",
        lambda self: "bb_gdn" if getattr(self, "is_linear", False) else "bb_attn",
    )
    # MoE block (nested inside the layer bucket)
    mlp = getattr(layers[0], "mlp", None)
    if mlp is not None:
        _wrap_call(type(mlp), "moe", lambda self: "bb_moe")
    # PLE layer and its n-gram table (nested inside the layer bucket)
    for layer in layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        _wrap_call(type(ple), "ple", lambda self: "bb_ple")
        emb = getattr(ple, "ple_embedding", None)
        if emb is not None:
            _wrap_ngram(type(emb))
            _wrap_table(getattr(emb, "ngram_embedding", None))
        break
    # token embedding
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None:
        _wrap_call(type(embed), "embed", lambda self: "embed")
    # vocabulary head
    head = getattr(lm, "lm_head", None)
    if head is not None:
        _wrap_head(head)
    # MTP head forward + rollback live on the LanguageModel
    _wrap_mtp_forward(type(lm))
    _wrap_rollback(type(lm))


def _wrap_ngram(cls) -> None:
    """Time the whole n-gram lookup and record how many rows it read."""
    if getattr(cls, "_dprof_ngram", False):
        return
    stock = cls.__call__

    def wrapper(self, *a, **k):
        if _cur is None:
            return stock(self, *a, **k)
        with _Stage("ngram_lookup"):
            out = stock(self, *a, **k)
        table = getattr(self, "ngram_embedding", None)
        _cur["rows_ngram"] = _cur.get("rows_ngram", 0) + int(
            getattr(table, "rows_read", 0) or 0
        )
        return out

    cls.__call__ = wrapper
    cls._dprof_ngram = True


def _wrap_head(head) -> None:
    """Time the vocabulary head.  Instance-keyed, so other Linear modules and
    every isinstance / identity check keep the stock path."""
    cls = type(head)

    def wrapper(self, x, *a, **k):
        stock = cls._dprof_stock_head
        if _cur is None:
            return stock(self, x, *a, **k)
        try:
            rows = int(x.shape[-2]) if x.ndim >= 2 else 1
        except Exception:  # noqa: BLE001
            rows = -1
        _cur["lm_head_rows"].append(rows)
        with _Stage("lm_head"):
            return stock(self, x, *a, **k)

    if not getattr(cls, "_dprof_head", False):
        stock_call = cls.__call__

        def dispatch(self, *a, **k):
            if self.__dict__.get("_dprof_head_call") is not None:
                return self.__dict__["_dprof_head_call"](*a, **k)
            return stock_call(self, *a, **k)

        cls._dprof_stock_head = stock_call
        cls.__call__ = dispatch
        cls._dprof_head = True
    head.__dict__["_dprof_head_call"] = wrapper.__get__(head)


def _wrap_table(table) -> None:
    """Split the n-gram lookup into host page reads and the device upload."""
    if table is None:
        return
    packed = getattr(table, "_packed", None)
    if packed is not None:
        for name, tag in (("assemble_host", "ngram_pread"),
                          ("dequantize_host", "ngram_upload")):
            fn = getattr(type(packed), name, None)
            if fn is None or getattr(fn, "_dprof", False):
                continue

            def make(fn=fn, tag=tag):
                def wrapper(self, *a, **k):
                    if _cur is None:
                        return fn(self, *a, **k)
                    with _Stage(tag):
                        return fn(self, *a, **k)
                wrapper._dprof = True
                return wrapper

            setattr(type(packed), name, make())
        return
    # stock SSD path
    cls = type(table)
    for name, tag in (("_assemble", "ngram_pread"), ("_host_indices", "ngram_ids")):
        fn = getattr(cls, name, None)
        if fn is None or getattr(fn, "_dprof", False):
            continue

        def make(fn=fn, tag=tag):
            def wrapper(self, *a, **k):
                if _cur is None:
                    return fn(self, *a, **k)
                with _Stage(tag):
                    return fn(self, *a, **k)
            wrapper._dprof = True
            return wrapper

        setattr(cls, name, make())


def _wrap_mtp_forward(cls) -> None:
    fn = getattr(cls, "mtp_forward", None)
    if fn is None or getattr(fn, "_dprof", False):
        return

    def wrapper(self, *a, **k):
        if _cur is None:
            return fn(self, *a, **k)
        n = _cur["sub_n"].get("draft.mtp_head1", 0)
        tag = "mtp_head1" if n == 0 else "mtp_headk"
        with _Stage(tag):
            return fn(self, *a, **k)

    wrapper._dprof = True
    cls.mtp_forward = wrapper


def _wrap_rollback(cls) -> None:
    fn = getattr(cls, "rollback_speculative_cache", None)
    if fn is not None and not getattr(fn, "_dprof", False):
        def wrapper(self, *a, **k):
            if _cur is None:
                return fn(self, *a, **k)
            with _Stage("rollback_gdn_kv"):
                return fn(self, *a, **k)
        wrapper._dprof = True
        cls.rollback_speculative_cache = wrapper
    ple = getattr(cls, "_restore_ple_state", None)
    if ple is not None and not getattr(ple, "_dprof", False):
        raw = ple.__func__ if hasattr(ple, "__func__") else ple

        def ple_wrapper(*a, **k):
            if _cur is None:
                return raw(*a, **k)
            with _Stage("rollback_ple"):
                return raw(*a, **k)
        ple_wrapper._dprof = True
        cls._restore_ple_state = staticmethod(ple_wrapper)


# ------------------------------------------------------- cycle wrapping
def _install_bg(bg) -> None:
    stock_cycle = bg._run_verify_cycle_chain
    stock_backbone = bg._call_backbone
    stock_drafts = bg._chain_next_drafts
    stock_rollback = bg._chain_rollback
    stock_clear = bg._clear_rollback
    stock_log = bg._log_mtp_stats

    marks: dict = {}

    def call_backbone(model, inputs, cache, n_confirmed=0):
        global _phase
        if _cur is None:
            return stock_backbone(model, inputs, cache, n_confirmed=n_confirmed)
        _wrap_model(model)
        try:
            _cur["M"] = int(inputs.shape[-1])
        except Exception:  # noqa: BLE001
            pass
        _cur["kv_bytes"] = _kv_bytes(cache)
        _cur["ctx"] = int(getattr(cache[0], "offset", 0)) if cache else 0
        _phase = "verify"
        marks["bb0"] = time.perf_counter()
        try:
            return stock_backbone(model, inputs, cache, n_confirmed=n_confirmed)
        finally:
            if _SYNC:
                _orig["synchronize"]()
            marks["bb1"] = time.perf_counter()
            _phase = "accept"

    def chain_rollback(model, prompt_cache, accepted, num_drafts, gdn_states=None):
        global _phase
        if _cur is None:
            return stock_rollback(model, prompt_cache, accepted, num_drafts, gdn_states)
        _phase = "commit"
        marks["_rb"] = True
        marks.setdefault("cm0", time.perf_counter())
        try:
            return stock_rollback(model, prompt_cache, accepted, num_drafts, gdn_states)
        finally:
            if _SYNC:
                _orig["synchronize"]()
            marks["cm1"] = time.perf_counter()

    def clear_rollback(prompt_cache):
        global _phase
        if _cur is None:
            return stock_clear(prompt_cache)
        _phase = "commit"
        marks.setdefault("cm0", time.perf_counter())
        try:
            return stock_clear(prompt_cache)
        finally:
            marks["cm1"] = time.perf_counter()

    def chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf):
        global _phase
        if _cur is None:
            return stock_drafts(gen_batch, state, hidden_rows, committed, prev_buf)
        _phase = "draft"
        marks["dr0"] = time.perf_counter()
        try:
            return stock_drafts(gen_batch, state, hidden_rows, committed, prev_buf)
        finally:
            if _SYNC:
                _orig["synchronize"]()
            marks["dr1"] = time.perf_counter()
            _phase = "post"

    def run_cycle(gen_batch, state):
        global _cur, _phase
        uid = getattr(state, "uid", None)
        run = _runs.get(uid)
        if run is None:
            while len(_runs) >= 8:
                _runs.pop(next(iter(_runs)))
            run = _runs[uid] = _Run(uid)
        if len(run.cycles) >= _MAX_CYCLES:
            run.dropped += 1
            return stock_cycle(gen_batch, state)
        rec = _new_cycle()
        rec["k"] = int(state.drafts.shape[0]) if state.drafts is not None else 0
        marks.clear()
        _cur = rec
        _phase = "pre"
        t0 = time.perf_counter()
        try:
            return stock_cycle(gen_batch, state)
        finally:
            t1 = time.perf_counter()
            _phase = "idle"
            _finish_cycle(rec, marks, t0, t1, state)
            _cur = None
            run.cycles.append(rec)

    def log_stats(uid, stats, finish_reason):
        try:
            _emit(uid, stats, finish_reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("decode-profile: emit failed: %r", exc)
        return stock_log(uid, stats, finish_reason)

    bg._call_backbone = call_backbone
    bg._chain_rollback = chain_rollback
    bg._clear_rollback = clear_rollback
    bg._chain_next_drafts = chain_next_drafts
    bg._run_verify_cycle_chain = run_cycle
    bg._log_mtp_stats = log_stats


def _finish_cycle(rec, marks, t0, t1, state) -> None:
    """Turn the timestamps into an exact partition of the cycle wall."""
    ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
    bb0 = marks.get("bb0", t0)
    bb1 = marks.get("bb1", bb0)
    cm0 = marks.get("cm0", bb1)
    cm1 = marks.get("cm1", cm0)
    dr0 = marks.get("dr0", cm1)
    dr1 = marks.get("dr1", dr0)
    p = rec["phase_ms"]
    p["pre"] = ms(t0, bb0)
    p["verify_dispatch"] = ms(bb0, bb1)
    p["accept"] = ms(bb1, cm0)
    p["commit"] = ms(cm0, cm1)
    p["head_gap"] = ms(cm1, dr0)
    p["draft"] = ms(dr0, dr1)
    p["post"] = ms(dr1, t1)
    rec["cycle_ms"] = ms(t0, t1)
    rec["m"] = int(getattr(state.stats, "accepts", 0))  # cumulative; differenced later
    rec["rolled_back"] = bool(marks.get("_rb", False))
    # bytes
    M = rec.get("M", rec["k"] + 1)
    experts = rec.get("experts_exact")
    if experts is None:
        experts = expected_distinct_experts(M) * N_MOE_LAYERS
    heads = sum(n for k, n in rec["sub_n"].items()
                if k.endswith("lm_head")) or 1
    rec["bytes"] = {
        "dense": BYTES["gdn"] + BYTES["qsa"] + BYTES["hyper"]
        + BYTES["shared"] + BYTES["router"],
        "experts": experts * BYTES["expert"],
        "lm_head": heads * BYTES["lm_head"],
        "kv": rec.get("kv_bytes", 0.0),
        "ngram": rec.get("rows_ngram", 0) * BYTES["ngram_row"],
    }


# ------------------------------------------------------------- reporting
def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _emit(uid, stats, finish_reason) -> None:
    run = _runs.pop(uid, None)
    if run is None or not run.cycles:
        return
    cycles = run.cycles
    # de-cumulate accepted counts
    prev = 0
    for rec in cycles:
        cum = rec.get("m", 0)
        rec["m"] = max(0, cum - prev)
        prev = cum
    med = {k: _median([c["phase_ms"].get(k, 0.0) for c in cycles])
           for k in cycles[0]["phase_ms"]}
    sub_keys = sorted({k for c in cycles for k in c["sub_ms"]})
    sub = {k: _median([c["sub_ms"].get(k, 0.0) for c in cycles]) for k in sub_keys}
    sync_n = sum(sum(c["sync_n"].values()) for c in cycles) / len(cycles)
    sync_ms = {k: _median([c["sync_ms"].get(k, 0.0) for c in cycles])
               for k in sorted({k for c in cycles for k in c["sync_ms"]})}
    cyc = _median([c["cycle_ms"] for c in cycles])
    accept_sync = sum(v for k, v in sync_ms.items() if k.startswith("accept."))
    payload = {
        "uid": str(uid),
        "finish": finish_reason,
        "started": run.t0,
        "sync_mode": _SYNC,
        "cycles": len(cycles),
        "dropped": run.dropped,
        "bytes_model": BYTES,
        "median": {"cycle_ms": cyc, "phase_ms": med, "sub_ms": sub,
                   "sync_ms": sync_ms, "syncs_per_cycle": sync_n},
        "records": cycles,
    }
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    path = _OUTDIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{str(uid)[:12]}.json"
    path.write_text(json.dumps(payload))
    logger.info(
        "DPROF[%s] cycles=%d cycle=%.1fms verify=%.1f accept=%.1f(sync %.1f) "
        "commit=%.1f draft=%.1f | gdn=%.1f attn=%.1f moe=%.1f ple=%.1f "
        "ngram=%.1f head=%.1f draft1=%.1f draftk=%.1f rb=%.1f "
        "syncs/cycle=%.1f ctx=%d sync_mode=%d -> %s",
        uid, len(cycles), cyc,
        med.get("verify_dispatch", 0.0), med.get("accept", 0.0), accept_sync,
        med.get("commit", 0.0), med.get("draft", 0.0),
        sub.get("bb_gdn", 0.0), sub.get("bb_attn", 0.0), sub.get("bb_moe", 0.0),
        sub.get("bb_ple", 0.0), sub.get("ngram_lookup", 0.0),
        sub.get("lm_head", 0.0), sub.get("draft.mtp_head1", 0.0),
        sub.get("draft.mtp_headk", 0.0),
        sub.get("rollback_gdn_kv", 0.0) + sub.get("rollback_ple", 0.0),
        sync_n, cycles[-1].get("ctx", 0), int(_SYNC), path.name,
    )


# ---------------------------------------------------------------- install
def install() -> bool:
    """Install the profiler.  Import time.  Returns False when the flag is off."""
    global _INSTALLED, _SYNC, _OUTDIR, _MAX_CYCLES, _mx
    if _INSTALLED:
        return True
    if not enabled():
        return False
    try:
        import mlx.core as mx
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
    except Exception as exc:  # noqa: BLE001
        logger.warning("decode-profile: preconditions not met (%r)", exc)
        return False
    if not hasattr(bg, "_run_verify_cycle_chain"):
        logger.warning("decode-profile: no _run_verify_cycle_chain; not installing")
        return False
    _SYNC = _flag("OMLX_DECODE_PROFILE_SYNC")
    _MAX_CYCLES = int(_fenv("OMLX_DECODE_PROFILE_MAX_CYCLES", 4096))
    _OUTDIR = Path(
        os.environ.get("OMLX_DECODE_PROFILE_DIR")
        or "~/inference-server/staging/decode-profile"
    ).expanduser()
    _mx = mx
    _install_sync_counters(mx)
    _install_bg(bg)
    _INSTALLED = True
    logger.info(
        "decode-profile installed (sync_mode=%s, dir=%s)", _SYNC, _OUTDIR
    )
    return True
