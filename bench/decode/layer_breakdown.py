#!/usr/bin/env python3
"""Where one decode step's time goes, by layer type.

``host_overhead.py profile`` answers "how much of a step is the host". This
answers "which part of the model is it", which is the question ROUND3 opens
with: a plain decode cycle is 18.3 ms at a short context and 21.9 ms at 64k,
and something in the forward is 3.6 ms more expensive when the cache is a
hundred times longer even though the sparse arms are supposed to read a fixed
2,051 rows either way.

What this script may and may not do
-----------------------------------

An earlier version of this file panicked the machine (``docs/ops/INCIDENTS.md``,
2026-09-12 23:04). It forced evaluation of every scope's own output and called
``mx.synchronize()`` per scope, on the real checkpoint at 64k, in a loop that
rewound the cache after every step: thousands of tiny command buffers with
forced evaluation of intermediates, which the GPU firmware did not survive.

The rule that followed is INCIDENTS rule 1, and this file now enforces it
rather than documenting it:

* **Per-scope evaluation barriers are a synthetic-configuration feature.**
  ``--sync-level top`` and ``--sync-level all`` are refused with ``--model``.
* **The real-checkpoint mode times the Python call only** and synchronises
  exactly once per step, at the end. Nothing inside the step is evaluated out
  of order, so the graph the GPU sees is the graph an ordinary step builds.

Configurations
--------------

``--config small`` (default)
    The four-layer, hidden-128 Qwen4-Exp from ``host_overhead.py``. Cheap, and
    its shapes are not the checkpoint's, so its *timings* mean little.

``--config real-shapes``
    A Qwen4-Exp carrying the checkpoint's per-layer shapes -- hidden 2560, 24
    query heads over 2 K/V heads at head dimension 256, the 4-head indexer at
    dimension 128 with budget 2048 and compress ratio 4, the 48/16-head Gated
    DeltaNet, the 4-wide hyper-connection at low rank 320 -- with the layer
    count and the expert count cut so it fits beside a busy GPU. This is where
    per-scope attribution is measured, because a QSA layer here costs what a
    QSA layer on the checkpoint costs.

    What it does not reproduce: 48 layers (it runs 8), 512 experts (16), the
    248,320-row head (4096), and the PLE n-gram layer, which needs the
    checkpoint's table on SSD. None of those depend on the cache length, which
    is what this measurement is about.

``--model <dir>``
    The real checkpoint, whole-step timings plus per-scope *host* time. One
    sync per step, no per-scope evaluation.

Arms
----

``--arm`` picks what the step is. The engine runs three different forwards and
they take three different attention paths, which is the whole point of the
attribution:

``decode``
    width 1, ``return_hidden=False``. What ``speculation.enabled=false`` runs.
    Takes the gathered sparse decode arm.
``decode-hidden``
    width 1, ``return_hidden=True``. What the engine runs when the drafter
    wants a hidden state. ``return_hidden`` sets ``capture_layer_ids``, which
    sets ``target_verify`` on every layer, which disqualifies the gathered
    decode arm -- so this arm is dense.
``verify``
    width > 1, ``return_hidden=True``. Takes the gathered block arm.

Usage
-----

    python bench/decode/layer_breakdown.py --config real-shapes \\
        --contexts 600,64000 --widths 1,4 --repeats 12

    python bench/decode/layer_breakdown.py \\
        --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \\
        --context 600 --widths 1,4 --repeats 12      # host time only

Results and logs belong under ``bench/decode/`` (INCIDENTS rule 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

import host_overhead as ho  # noqa: E402


#: The checkpoint's per-layer shapes, with layer/expert/vocab counts cut to fit
#: beside a busy GPU. Every field that a cache-length-dependent cost could
#: depend on is the checkpoint's own value.
REAL_SHAPES = dict(
    hidden_size=2560,
    num_hidden_layers=8,
    num_attention_heads=24,
    num_key_value_heads=2,
    head_dim=256,
    hc_count=4,
    hc_lowrank=320,
    num_experts=16,
    num_experts_per_tok=10,
    moe_intermediate_size=640,
    shared_expert_intermediate_size=640,
    vocab_size=4096,
    linear_num_value_heads=48,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    indexer_n_heads=4,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    max_position_embeddings=262144,
)


# ---------------------------------------------------------------------------
# the scope recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Nested wall-clock scopes with optional per-scope evaluation barriers.

    The barriers are opt-in and, on the real checkpoint, unavailable: see the
    module docstring and INCIDENTS rule 1.
    """

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.children: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.sync: set[str] = set()
        self.enabled = False
        self._stack: list[list[Any]] = []

    def reset(self) -> None:
        self.total.clear()
        self.children.clear()
        self.calls.clear()
        self._stack.clear()

    def run(self, label: str, fn: Callable[..., Any], args, kwargs) -> Any:
        if not self.enabled:
            return fn(*args, **kwargs)
        frame = [0.0]                       # child time accumulated in here
        self._stack.append(frame)
        start = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            if label in self.sync:
                arrays = _arrays(out)
                if arrays:
                    mx.eval(arrays)
                mx.synchronize()
        finally:
            elapsed = time.perf_counter() - start
            self._stack.pop()
            if self._stack:
                self._stack[-1][0] += elapsed
            self.total[label] += elapsed
            self.children[label] += frame[0]
            self.calls[label] += 1
        return out

    def rows(self, repeats: int) -> list[dict[str, Any]]:
        out = []
        for label in sorted(self.total):
            inclusive = self.total[label] * 1000.0 / repeats
            self_time = (self.total[label] - self.children[label]) * 1000.0 / repeats
            out.append({
                "group": label,
                "calls": self.calls[label] / repeats,
                "inclusive_ms": inclusive,
                "self_ms": self_time,
            })
        return out


def _arrays(value: Any, depth: int = 0) -> list[mx.array]:
    if depth > 4:
        return []
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        found: list[mx.array] = []
        for item in value:
            found.extend(_arrays(item, depth + 1))
        return found
    return []


RECORDER = Recorder()


# ---------------------------------------------------------------------------
# installing the scopes
# ---------------------------------------------------------------------------


class Instrumentation:
    """Wrap functions and methods, and put every one of them back."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []

    def wrap_attr(self, owner: Any, name: str, label: str,
                  gate: Optional[Callable[..., bool]] = None) -> None:
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            if gate is not None and not gate(*args, **kwargs):
                return original(*args, **kwargs)
            return RECORDER.run(label, original, args, kwargs)

        wrapper.__name__ = getattr(original, "__name__", name)
        setattr(owner, name, wrapper)
        self._undo.append(lambda: setattr(owner, name, original))

    def wrap_method(self, cls: type, name: str, label: str,
                    gate: Optional[Callable[..., bool]] = None) -> None:
        self.wrap_attr(cls, name, label, gate)

    def restore(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()


#: outermost groups. These get the eval barrier at ``--sync-level top``.
TOP_SCOPES = (
    "gdn", "qsa", "moe", "hyper_connection", "hyper_inject", "ple", "lm_head",
)

#: every label ``install`` can produce, for ``--sync-level all``.
ALL_SCOPES = TOP_SCOPES + (
    "qsa.arm.gathered_decode", "qsa.arm.gathered_block", "qsa.arm.dense_indexer",
    "qsa.arm.dense_sdpa", "qsa.pool", "qsa.update_indexer", "qsa.gather",
    "qsa.scores", "qsa.sdpa", "moe.experts", "moe.shared", "ple.ngram",
    "ple.lookup",
)


def install(model, *, sync_level: str) -> Instrumentation:
    """Wrap every layer type of the loaded Qwen4-Exp. Returns the undo handle."""
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen3_5 import (
        language as q35,
    )
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen3_5_moe import (
        language as moe_mod,
    )
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp import (
        language as q4,
    )
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp import qsa_fast

    inst = Instrumentation()

    # --- the four layer-type blocks ---------------------------------------
    inst.wrap_method(q4.Qwen4ExpGatedDeltaNet, "__call__", "gdn")
    inst.wrap_method(q4.Qwen4ExpAttention, "__call__", "qsa")
    inst.wrap_method(moe_mod.Qwen3_5MoeSparseMoeBlock, "__call__", "moe")
    inst.wrap_method(q4.Qwen4ExpGatedResidual, "__call__", "hyper_connection")
    inst.wrap_method(q4.Qwen4ExpPLELayer, "__call__", "ple")
    inst.wrap_attr(q4, "_hyper_inject", "hyper_inject")
    inst.wrap_attr(q4, "_hyper_inject_ops", "hyper_inject")

    # --- QSA, split ------------------------------------------------------
    # ``qsa.arm.*`` says which arm the step actually took, which matters more
    # than any single timing here: the dense arm at 64k is the failure mode.
    # ``language.py`` imported these by name, so the call site reads its own
    # module global; wrapping only ``qsa_fast`` would miss every real call.
    for module in (qsa_fast, q4):
        inst.wrap_attr(module, "contiguous_causal_gathered_qsa_decode",
                       "qsa.arm.gathered_decode")
        inst.wrap_attr(module, "contiguous_causal_gathered_qsa",
                       "qsa.arm.gathered_block")
    inst.wrap_method(q4.Qwen4ExpQSAIndexer, "__call__", "qsa.arm.dense_indexer")
    # the dense masked arm itself: ``Qwen4ExpAttention`` reaches it through
    # ``super().__call__``, so wrapping the parent catches exactly that route.
    inst.wrap_method(q35.Qwen3_5Attention, "__call__", "qsa.arm.dense_sdpa")
    inst.wrap_method(q4._QSAIndexerCache, "pooled_indexer_keys", "qsa.pool")
    inst.wrap_method(q4._QSAIndexerCache, "update_indexer", "qsa.update_indexer")
    inst.wrap_attr(qsa_fast, "_gather_kv_rows", "qsa.gather")
    inst.wrap_attr(qsa_fast, "_portable_indexer_scores", "qsa.scores")
    inst.wrap_attr(qsa_fast, "_decode_qsa_sdpa", "qsa.sdpa")

    # --- MoE, split -------------------------------------------------------
    inst.wrap_attr(moe_mod, "_target_verify_switch_glu", "moe.experts")
    inst.wrap_method(moe_mod.Qwen3_5MoeMLP, "__call__", "moe.shared")

    # --- PLE, split -------------------------------------------------------
    inst.wrap_method(q4.Qwen4ExpNGramEmbedding, "__call__", "ple.ngram")
    inst.wrap_method(q4.DiskBackedShardedEmbedding, "__call__", "ple.lookup")

    # --- the head ---------------------------------------------------------
    language_model = getattr(model, "language_model", model)
    head = getattr(language_model, "lm_head", None)
    if head is not None:
        head._bench_label = "lm_head"

    def is_head(linear, *_args, **_kwargs):
        return getattr(linear, "_bench_label", None) == "lm_head"

    inst.wrap_attr(q35, "_target_verify_linear", "lm_head", gate=is_head)

    if sync_level == "top":
        RECORDER.sync = set(TOP_SCOPES)
    elif sync_level == "all":
        RECORDER.sync = set(ALL_SCOPES)
    else:
        RECORDER.sync = set()
    return inst


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def _forward(harness, width: int, want_hidden: bool) -> Any:
    ids = harness.next_tokens(width)
    return harness.language_model(
        ids, cache=harness.cache, return_hidden=want_hidden
    )


def _step(harness, width: int, want_hidden: bool) -> None:
    """One step, one evaluation, one rewind. No per-scope barrier."""
    output = _forward(harness, width, want_hidden)
    mx.eval(ho._step_outputs(output, harness.cache))
    harness.rewind(width)


def measure(harness, width: int, *, want_hidden: bool, repeats: int,
            warmup: int, sync_level: str) -> dict[str, Any]:
    """Two or three passes over the same step.

    1. the plain step, recorder off, one sync at each end: the honest whole-step
       number, uninstrumented.
    2. the scopes timing the Python call only: the per-group host cost. One
       ``mx.eval`` at the end of the step, exactly as pass 1.
    3. only at ``--sync-level top|all``, and only on a synthetic configuration:
       the scopes again with an evaluation barrier inside each, so a group's
       wall time contains the GPU work it forced. ``gpu = sync - build``.
    """
    for _ in range(warmup):
        _step(harness, width, want_hidden)

    RECORDER.enabled = False
    plain: list[float] = []
    plain_build: list[float] = []
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        output = _forward(harness, width, want_hidden)
        built = time.perf_counter()
        mx.eval(ho._step_outputs(output, harness.cache))
        # The one synchronize of the step.
        mx.synchronize()
        end = time.perf_counter()
        harness.rewind(width)
        plain.append((end - start) * 1000.0)
        plain_build.append((built - start) * 1000.0)

    saved_sync = RECORDER.sync
    RECORDER.sync = set()
    RECORDER.reset()
    RECORDER.enabled = True
    for _ in range(repeats):
        _step(harness, width, want_hidden)
    build_rows = {row["group"]: row for row in RECORDER.rows(repeats)}

    sync_rows: dict[str, dict[str, Any]] = {}
    sync_total: list[float] = []
    if saved_sync:
        RECORDER.sync = saved_sync
        RECORDER.reset()
        for _ in range(repeats):
            mx.synchronize()
            start = time.perf_counter()
            _step(harness, width, want_hidden)
            mx.synchronize()
            sync_total.append((time.perf_counter() - start) * 1000.0)
        sync_rows = {row["group"]: row for row in RECORDER.rows(repeats)}
    RECORDER.enabled = False
    RECORDER.sync = saved_sync

    groups = []
    for name in sorted(set(build_rows) | set(sync_rows)):
        build = build_rows.get(name, {})
        synced = sync_rows.get(name, {})
        build_self = float(build.get("self_ms", 0.0))
        sync_self = float(synced.get("self_ms", 0.0))
        groups.append({
            "group": name,
            "calls": float(build.get("calls", synced.get("calls", 0.0))),
            "build_ms": build_self,
            "gpu_ms": max(0.0, sync_self - build_self) if sync_rows else None,
            "build_inclusive_ms": float(build.get("inclusive_ms", 0.0)),
            "sync_inclusive_ms": float(synced.get("inclusive_ms", 0.0)),
        })

    return {
        "width": width,
        "arm": "verify" if width > 1 else (
            "decode-hidden" if want_hidden else "decode"),
        "want_hidden": want_hidden,
        "step_ms": ho._median(plain),
        "build_ms": ho._median(plain_build),
        "gpu_tail_ms": max(0.0, ho._median(plain) - ho._median(plain_build)),
        "sync_step_ms": ho._median(sync_total) if sync_total else None,
        "has_gpu_column": bool(sync_rows),
        "groups": groups,
    }


TOP_ORDER = TOP_SCOPES


def _print(result: dict[str, Any], context: int) -> None:
    step = result["step_ms"]
    print(f"\ncontext {context}, width {result['width']}, arm {result['arm']}")
    tail = ""
    if result["sync_step_ms"] is not None:
        tail = f"; barriered attribution pass {result['sync_step_ms']:.2f} ms"
    print(f"  step {step:.2f} ms = host build {result['build_ms']:.2f} + "
          f"gpu tail {result['gpu_tail_ms']:.2f}{tail}")
    by_name = {row["group"]: row for row in result["groups"]}
    rows = []
    for name in TOP_ORDER:
        if name in by_name:
            rows.append((name, by_name[name], False))
            for child in sorted(by_name):
                if child.startswith(name + ".") and by_name[child]["calls"]:
                    rows.append((child, by_name[child], True))
    build_sum = sum(row["build_ms"] for _n, row, _i in rows)
    gpu_sum = sum((row["gpu_ms"] or 0.0) for _n, row, _i in rows)
    print(f"  {'group':<26} {'calls':>6} {'build ms':>9} {'%host':>7} "
          f"{'gpu ms':>8} {'%gpu':>7}")
    for name, row, indent in rows:
        _print_row(name, row, indent, result["build_ms"], gpu_sum)
    print(f"  {'-' * 68}")
    _print_row("attributed", {"calls": 0, "build_ms": build_sum,
                              "gpu_ms": gpu_sum}, False,
               result["build_ms"], gpu_sum)
    rest = max(0.0, result["build_ms"] - build_sum)
    _print_row("host, unattributed", {"calls": 0, "build_ms": rest,
                                      "gpu_ms": 0.0}, False,
               result["build_ms"], gpu_sum)
    print("  (unattributed host = embedding, mask build, the model and "
          "decoder-layer bodies,")
    print("   the residual norms, and the eval/dispatch tail)")
    if not result["has_gpu_column"]:
        print("  (no gpu column: per-scope barriers are off, which is "
              "mandatory on the real checkpoint)")


def _print_row(name: str, row: dict[str, Any], indent: bool,
               host_total: float, gpu_total: float) -> None:
    label = ("  " + name.split(".", 1)[1]) if indent else name
    host_pct = 100.0 * row["build_ms"] / host_total if host_total else 0.0
    gpu = row.get("gpu_ms") or 0.0
    gpu_pct = 100.0 * gpu / gpu_total if gpu_total else 0.0
    calls = f"{row['calls']:.0f}" if row["calls"] else ""
    print(f"  {label:<26} {calls:>6} {row['build_ms']:>9.2f} {host_pct:>6.1f}% "
          f"{gpu:>8.2f} {gpu_pct:>6.1f}%")


# ---------------------------------------------------------------------------
# elimination: the same step under different arms, paired
# ---------------------------------------------------------------------------


#: name -> the adapter routes that define it. Elimination is by *arm*, not by
#: scope timer: a scope timer at 64k needs barriers, and barriers are what
#: panicked the machine. Turning one arm off and re-measuring the whole step
#: costs nothing but time and answers the same question with no caveat.
ELIMINATION = {
    "default": {},
    # Every QSA layer back on the dense masked path, whatever the context.
    # The difference against ``default`` is what the gathered arms are worth.
    "dense-qsa": {
        "qsa_gather_min_context": 1 << 40,
        "qsa_gather_min_context_verify": 1 << 40,
    },
    # The one-row target-verify routing on its own.
    "no-singleton-verify": {"qsa_sparse_singleton_verify": False},
    # The gathered arm taken as early as it is legal, which is what the
    # vendored gate did before the crossover route existed. Identical to
    # ``default`` past the crossover; the difference below it is what the
    # crossover is worth, in whichever direction.
    "gathered-at-budget": {
        "qsa_gather_min_context": 0,
        "qsa_gather_min_context_verify": 0,
    },
}


def _paired(harness, arms, cells, *, repeats, warmup):
    """Measure every cell at every arm, in A B B A order within each arm.

    The machine drifts by 10 to 16% over a run (ROUND2 step 1c), so the order
    matters more than the repeat count: measuring all of A then all of B hands
    the drift to whichever ran second.
    """
    from titan.adapters.mlx import kernels as adapter_kernels

    names = list(cells)
    order = names + names[::-1]
    results: dict[tuple[str, int, bool], list[float]] = {}
    for name in order:
        with adapter_kernels.overridden(**cells[name]):
            for width, want_hidden in arms:
                for _ in range(warmup):
                    _step(harness, width, want_hidden)
                times = []
                for _ in range(repeats):
                    mx.synchronize()
                    start = time.perf_counter()
                    output = _forward(harness, width, want_hidden)
                    mx.eval(ho._step_outputs(output, harness.cache))
                    mx.synchronize()
                    times.append((time.perf_counter() - start) * 1000.0)
                    harness.rewind(width)
                results.setdefault((name, width, want_hidden), []).extend(times)
    return results


def _print_elimination(results, context: int) -> None:
    print(f"\ncontext {context}: whole-step ms by arm, paired A B B A")
    arms = sorted({(w, h) for _n, w, h in results})
    names = []
    for name, _w, _h in results:
        if name not in names:
            names.append(name)
    header = "  " + f"{'arm':<22}" + "".join(
        f"{('w%d %s' % (w, 'hidden' if h else 'plain')):>16}" for w, h in arms
    )
    print(header)
    baseline = {}
    for name in names:
        cells = []
        for width, want_hidden in arms:
            value = ho._median(results[(name, width, want_hidden)])
            key = (width, want_hidden)
            baseline.setdefault(key, value)
            delta = value - baseline[key]
            cells.append(f"{value:>10.2f}{delta:>+6.2f}")
        print("  " + f"{name:<22}" + "".join(f"{c:>16}" for c in cells))


def _arms(text: str, widths: Sequence[int]) -> list[tuple[int, bool]]:
    """(width, want_hidden) pairs to measure."""
    out: list[tuple[int, bool]] = []
    for name in (part.strip() for part in text.split(",") if part.strip()):
        if name == "decode":
            out.append((1, False))
        elif name == "decode-hidden":
            out.append((1, True))
        elif name == "verify":
            out.extend((w, True) for w in widths if w > 1)
        else:
            raise SystemExit(f"unknown arm {name!r}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None,
                        help="real checkpoint directory; forces "
                             "--sync-level none (INCIDENTS rule 1)")
    parser.add_argument("--config", default="small",
                        choices=("small", "real-shapes"),
                        help="which synthetic model, when --model is absent")
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--contexts", default=None,
                        help="comma-separated context points, prefilled "
                             "forward in one process")
    parser.add_argument("--widths", default="1,4")
    parser.add_argument("--arms", default="decode,decode-hidden,verify")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--sync-level", default=None,
                        choices=("none", "top", "all"),
                        help="per-scope evaluation barriers. Default: 'all' on "
                             "a synthetic configuration, 'none' on --model, "
                             "where the other two are refused")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--eliminate", action="store_true",
                        help="also measure each arm with one route turned off, "
                             "paired A B B A, and print the difference")
    args = parser.parse_args(argv)

    sync_level = args.sync_level
    if args.model:
        if sync_level in ("top", "all"):
            raise SystemExit(
                "--sync-level top/all forces per-scope evaluation and "
                "per-scope mx.synchronize() on the real checkpoint, which is "
                "what panicked this machine on 2026-09-12 (INCIDENTS rule 1). "
                "Drop --sync-level, or drop --model and use "
                "--config real-shapes."
            )
        sync_level = "none"
    elif sync_level is None:
        sync_level = "all"

    contexts = sorted(
        int(c) for c in (args.contexts or str(args.context)).split(",") if c.strip()
    )
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    arms = _arms(args.arms, widths)
    harness = _make_harness(args, contexts[0])
    inst = install(harness.model, sync_level=sync_level)
    results = []
    try:
        reached = contexts[0]
        for context in contexts:
            # One model, one cache, prefilled forward. Reloading 73 GB per
            # context point would cost more than the measurement.
            if context > reached:
                harness.prefill(context - reached, chunk=args.prefill_chunk)
                reached = context
            for width, want_hidden in arms:
                result = measure(
                    harness, width, want_hidden=want_hidden,
                    repeats=args.repeats, warmup=args.warmup,
                    sync_level=sync_level,
                )
                result["context"] = context
                result["tag"] = args.tag
                result["config"] = "real" if args.model else args.config
                _print(result, context)
                results.append(result)
            if args.eliminate:
                paired = _paired(
                    harness, arms, ELIMINATION,
                    repeats=args.repeats, warmup=args.warmup,
                )
                _print_elimination(paired, context)
                results.append({
                    "context": context,
                    "elimination": {
                        f"{name}|w{width}|{'hidden' if hidden else 'plain'}":
                            ho._median(times)
                        for (name, width, hidden), times in paired.items()
                    },
                })
    finally:
        inst.restore()
        RECORDER.enabled = False

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def _make_harness(args, context: int):
    if args.model:
        model = ho.load_real(args.model)
        vocab = model.language_model.args.vocab_size
    else:
        fields: dict[str, Any] = {"quantize": not args.no_quantize}
        if args.config == "real-shapes":
            fields.update(REAL_SHAPES)
        if args.layers is not None:
            fields["num_hidden_layers"] = args.layers
        if args.hidden is not None:
            fields["hidden_size"] = args.hidden
        spec = ho.SyntheticSpec(**fields)
        model = ho.build_synthetic(spec)
        vocab = spec.vocab_size
    harness = ho.Harness(model=model, context=context, vocab=vocab)
    if context:
        harness.prefill(context, chunk=args.prefill_chunk)
    return harness


if __name__ == "__main__":
    raise SystemExit(main())
