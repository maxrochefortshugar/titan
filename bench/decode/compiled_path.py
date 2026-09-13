#!/usr/bin/env python3
"""The compiled decode path, measured against the eager one.

``titan/adapters/mlx/compiled.py`` rebuilds the decode forward as one traced
graph per layer type and one for the whole model. This script says what that
buys, and it is the only place the claim is allowed to come from.

Three subcommands.

``synthetic``
    Eager against compiled on the synthetic Qwen4-Exp, paired A B B A inside
    one process, at every requested width. Op counts, host build time, step
    time and the implied GPU time. This is the before/after table in
    ``docs/architecture/COMPILED.md``.

``cache``
    What the compile cache costs and what it keeps: first-call compile time per
    width over 1..8, the second-call time at the same width, and the same
    across the KV buffer growth points ``capacity_for`` chooses. A trace is
    keyed on input shapes, so this is the whole story about when the host pays.

``real``
    The checkpoint arm. Eager against compiled at two contexts and two widths
    with exactly one device sync per step, writing JSON under ``bench/decode/``.
    Nothing in this workstream has run it; see "the real-model command" below
    and rule 1 in ``docs/ops/INCIDENTS.md``.

Measurement rules this file follows
-----------------------------------

One sync per step, never per scope. ``docs/ops/INCIDENTS.md`` records a kernel
panic from a harness that evaluated every layer's own output and called
``mx.synchronize()`` per scope on the real checkpoint at 64k. Every loop here
builds the whole step, evaluates once at the end, and synchronises once. The op
counter walks the graph with ``mx.export_to_dot``, which reads the graph
without evaluating it, and on the ``real`` subcommand it is off by default
anyway.

Results are written under ``bench/decode/results/``, inside the repo, because
rule 5 says a measurement that has to survive a crash does not live in /tmp.

The real-model command, which nothing here has run
--------------------------------------------------

    python bench/decode/compiled_path.py real \\
        --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \\
        --contexts 600,64000 --widths 1,4 --repeats 20

It loads the checkpoint through ``titan.adapters.mlx.loader.load_model``,
prefills to each context through the ordinary eager path, snapshots the
recurrent state, then runs the two arms alternately at each width and writes
``bench/decode/results/compiled_real_<context>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

from host_overhead import (  # noqa: E402
    SyntheticSpec,
    Trace,
    _flatten,
    _median,
    build_synthetic,
)

from titan.adapters.mlx import compiled as C  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------


@dataclass
class EagerArm:
    """The stock forward on the stock caches, rewound after every step."""

    language_model: Any
    cache: list
    length: int
    name: str = "eager"

    def step(self, tokens: mx.array):
        return self.language_model(tokens, cache=self.cache, return_hidden=True)

    def outputs(self, result) -> list:
        arrays = [result.logits]
        for cache in self.cache:
            held = getattr(cache, "state", None)
            for value in held if isinstance(held, (list, tuple)) else (held,):
                if isinstance(value, mx.array):
                    arrays.append(value)
        for group in (result.hidden_states or [], result.gdn_states or []):
            arrays.extend(group)
        return [a for a in arrays if a is not None]

    def rewind(self, width: int) -> None:
        for cache in self.cache:
            trim = getattr(cache, "trim", None)
            if trim is not None and getattr(cache, "is_trimmable", lambda: False)():
                trim(width)


@dataclass
class CompiledArm:
    """The compiled model step on a functional state, rolled back after each step.

    The rollback is the state contract doing its job rather than a measurement
    trick: ``truncate_state`` moves the attention offset and puts the recurrent
    pair back from the snapshot staged at the prefill length, which is exactly
    what a rejected verify block does in the engine.
    """

    model: C.CompiledModel
    state: C.DecodeState
    snapshot: dict
    name: str = "compiled"
    #: layer index -> the eager layer's cache arrays as they stood at the
    #: prefill length. A layer that stays eager holds vendored state the
    #: functional rollback cannot reach, so it is restored rather than rewound.
    eager_baseline: dict = dc_field(default_factory=dict)

    def step(self, tokens: mx.array):
        logits, hidden, state = self.model(tokens, self.state)
        self._pending = state
        return logits, hidden, state

    def outputs(self, result) -> list:
        logits, hidden, state = result
        arrays = [logits, hidden]
        for entry in state.layers:
            arrays.extend(entry.arrays)
        return arrays

    def rewind(self, width: int) -> None:
        self.state = C.truncate_state(
            self._pending, self._pending.length - width, self.snapshot
        )
        # A layer that stays eager holds a vendored cache, which the functional
        # rollback cannot reach: its recurrent slots are not in the snapshot
        # and an ArraysCache is not trimmable. So it is put back to the state
        # it had at the prefill length, which is what the rest of the state
        # comes back to, and the arm steps from the same length every repeat.
        for index, saved in self.eager_baseline.items():
            _restore_cache(self.model.eager_caches[index], saved)


def make_arms(model, length: int, *, use_kernel: bool = True):
    """Prefill once, then hand back an eager arm and a compiled arm at *length*.

    Both arms start from the same prefilled cache, so a difference between them
    is the decode step and not the prefix.
    """
    language_model = getattr(model, "language_model", model)
    cache = language_model.make_cache()
    vocab = int(language_model.args.vocab_size)
    done = 0
    while done < length:
        piece = min(512, length - done)
        ids = mx.array([[(done + i) % max(2, vocab - 1) + 1 for i in range(piece)]])
        out = language_model(ids.astype(mx.int64), cache=cache, skip_logits=True)
        mx.eval([a for a in (out.logits,) if a is not None])
        mx.eval(_cache_arrays(cache))
        done += piece

    compiled_model = C.build_model(model, use_kernel=use_kernel)
    C.eval_weights(compiled_model.weights)
    # The layers that cannot be traced keep their own vendored caches, and the
    # two arms must not share one: the compiled arm's eager layer would advance
    # the cache the eager arm is stepping from. So it gets a copy, prefilled to
    # the same length.
    eager_ids = C.untraceable_layers(model)
    if eager_ids:
        copies = _clone_caches(language_model, cache)
        compiled_model.eager_caches = {index: copies[index] for index in eager_ids}
    state = C.read_layer_state(cache, eager_indices=eager_ids)
    mx.eval([a for entry in state.layers for a in entry.arrays])
    snapshot = state.recurrent_arrays()

    baseline = {
        index: _snapshot_cache(compiled_model.eager_caches[index])
        for index in eager_ids
    }
    return (
        EagerArm(language_model=language_model, cache=cache, length=length),
        CompiledArm(
            model=compiled_model,
            state=state,
            snapshot=snapshot,
            eager_baseline=baseline,
        ),
    )


def _snapshot_cache(cache) -> dict:
    """Everything a vendored cache holds, copied. Small: one layer's worth."""
    saved: dict = {}
    if getattr(cache, "keys", None) is not None:
        saved["keys"] = mx.array(cache.keys)
        saved["values"] = mx.array(cache.values)
        saved["offset"] = cache.offset
        if getattr(cache, "index_keys", None) is not None:
            saved["index_keys"] = mx.array(cache.index_keys)
            saved["index_position_ids"] = mx.array(cache.index_position_ids)
    held = getattr(cache, "state", None)
    if held is not None:
        saved["state"] = [None if v is None else mx.array(v) for v in held]
    return saved


def _restore_cache(cache, saved: dict) -> None:
    for name in ("keys", "values", "offset", "index_keys", "index_position_ids"):
        if name in saved:
            value = saved[name]
            setattr(cache, name, mx.array(value) if isinstance(value, mx.array) else value)
    if "state" in saved:
        cache.state = [None if v is None else mx.array(v) for v in saved["state"]]


def _clone_caches(language_model, caches) -> list:
    """A deep copy of a cache list, so two arms cannot see each other."""
    copy = language_model.make_cache()
    for fresh, held in zip(copy, caches):
        if getattr(held, "keys", None) is not None:
            fresh.keys = mx.array(held.keys)
            fresh.values = mx.array(held.values)
            fresh.offset = held.offset
            if getattr(held, "index_keys", None) is not None:
                fresh.index_keys = mx.array(held.index_keys)
                fresh.index_position_ids = mx.array(held.index_position_ids)
        elif getattr(held, "state", None) is not None:
            fresh.state = [None if v is None else mx.array(v) for v in held.state]
    return copy


def _cache_arrays(caches) -> list:
    arrays = []
    for cache in caches:
        held = getattr(cache, "state", None)
        if held is None:
            continue
        for value in held if isinstance(held, (list, tuple)) else (held,):
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def _tokens(width: int, vocab: int) -> mx.array:
    return mx.array([[(7 + i) % max(2, vocab - 1) + 1 for i in range(width)]], mx.int64)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure(arm, tokens: mx.array, *, repeats: int, warmup: int, count_ops: bool):
    """Ops, host build ms, step ms and implied GPU ms for one arm at one width.

    ``build_ms`` is the Python time to construct the step's graph with its
    evaluation pushed to the end; ``step_ms`` adds the wait for the device.
    Exactly one ``mx.synchronize()`` bounds each timed step, per INCIDENTS
    rule 1.
    """
    width = tokens.shape[1]
    for _ in range(warmup):
        result = arm.step(tokens)
        mx.eval(_flatten(arm.outputs(result)))
        arm.rewind(width)

    ops = 0
    primitives: Counter = Counter()
    build: list[float] = []
    total: list[float] = []
    for index in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        with Trace(count_ops=count_ops and index == 0) as trace:
            result = arm.step(tokens)
            built = time.perf_counter()
            trace.finish(arm.outputs(result))
        mx.synchronize()
        end = time.perf_counter()
        if index == 0 and count_ops:
            ops = trace.result.ops
            primitives = trace.result.by_primitive
        else:
            build.append((built - start) * 1000.0)
            total.append((end - start) * 1000.0)
        arm.rewind(width)

    return {
        "arm": arm.name,
        "width": width,
        "ops": ops,
        "top_primitives": primitives.most_common(8),
        "build_ms": _median(build),
        "step_ms": _median(total),
        "gpu_ms": max(0.0, _median(total) - _median(build)),
    }


def paired(eager, compiled_arm, tokens, *, repeats: int, warmup: int, count_ops: bool):
    """One A B B A pass, so a machine warming up cannot favour either arm."""
    a1 = measure(eager, tokens, repeats=repeats, warmup=warmup, count_ops=count_ops)
    b1 = measure(
        compiled_arm, tokens, repeats=repeats, warmup=warmup, count_ops=count_ops
    )
    b2 = measure(compiled_arm, tokens, repeats=repeats, warmup=0, count_ops=False)
    a2 = measure(eager, tokens, repeats=repeats, warmup=0, count_ops=False)
    return {
        "eager": _merge(a1, a2),
        "compiled": _merge(b1, b2),
    }


def _merge(first: dict, second: dict) -> dict:
    out = dict(first)
    for key in ("build_ms", "step_ms"):
        out[key] = 0.5 * (first[key] + second[key])
    out["gpu_ms"] = max(0.0, out["step_ms"] - out["build_ms"])
    return out


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_synthetic(args) -> None:
    spec = SyntheticSpec(num_hidden_layers=args.layers)
    model = build_synthetic(spec)
    eager, compiled_arm = make_arms(model, args.context)
    vocab = spec.vocab_size

    rows = []
    for width in args.widths:
        tokens = _tokens(width, vocab)
        result = paired(
            eager,
            compiled_arm,
            tokens,
            repeats=args.repeats,
            warmup=args.warmup,
            count_ops=True,
        )
        result["width"] = width
        rows.append(result)

    payload = {
        "kind": "compiled_synthetic",
        "layers": args.layers,
        "context": args.context,
        "repeats": args.repeats,
        "rows": rows,
    }
    _report(payload)
    _write(payload, f"compiled_synthetic_{args.layers}L_{args.context}.json")


def cmd_cache(args) -> None:
    """First-call compile cost and cache behaviour across widths and capacities."""
    spec = SyntheticSpec(num_hidden_layers=args.layers)
    model = build_synthetic(spec)
    language_model = getattr(model, "language_model", model)
    compiled_model = C.build_model(model)
    C.eval_weights(compiled_model.weights)

    widths = []
    state = C.new_state(model, capacity=C.capacity_for(args.context))
    state = C.DecodeState(layers=state.layers, length=args.context)
    for entry in state.layers:
        if entry.kind == C.ATTENTION:
            entry.arrays = (*entry.arrays[:3], mx.array(args.context, mx.int32))
    mx.eval([a for entry in state.layers for a in entry.arrays])
    snapshot = state.recurrent_arrays()

    for width in range(1, args.max_width + 1):
        tokens = _tokens(width, spec.vocab_size)
        first = _timed_call(compiled_model, tokens, state)
        second = _timed_call(compiled_model, tokens, state)
        widths.append({"width": width, **_prefixed(first, "first"), **_prefixed(second, "second")})

    capacities = []
    for tokens_held in args.growth:
        capacity = C.capacity_for(tokens_held)
        fresh = C.new_state(model, capacity=capacity)
        fresh = C.DecodeState(layers=fresh.layers, length=tokens_held)
        for entry in fresh.layers:
            if entry.kind == C.ATTENTION:
                entry.arrays = (*entry.arrays[:3], mx.array(tokens_held, mx.int32))
        mx.eval([a for entry in fresh.layers for a in entry.arrays])
        tok = _tokens(1, spec.vocab_size)
        first = _timed_call(compiled_model, tok, fresh)
        second = _timed_call(compiled_model, tok, fresh)
        capacities.append(
            {
                "tokens": tokens_held,
                "capacity": capacity,
                **_prefixed(first, "first"),
                **_prefixed(second, "second"),
            }
        )

    payload = {
        "kind": "compiled_cache",
        "layers": args.layers,
        "context": args.context,
        "widths": widths,
        "capacities": capacities,
    }
    print("\nfirst call against second, by width. build ms is host time to")
    print("return from the step, step ms adds the wait for the device.")
    print(
        f"{'width':>6} {'1st build':>10} {'2nd build':>10} "
        f"{'1st step':>9} {'2nd step':>9}"
    )
    for row in widths:
        print(
            f"{row['width']:>6} {row['first_build_ms']:>10.2f} "
            f"{row['second_build_ms']:>10.2f} {row['first_step_ms']:>9.2f} "
            f"{row['second_step_ms']:>9.2f}"
        )
    print("\nthe same across the KV buffer growth points")
    print(
        f"{'tokens':>8} {'capacity':>9} {'1st build':>10} {'2nd build':>10} "
        f"{'1st step':>9} {'2nd step':>9}"
    )
    for row in capacities:
        print(
            f"{row['tokens']:>8} {row['capacity']:>9} "
            f"{row['first_build_ms']:>10.2f} {row['second_build_ms']:>10.2f} "
            f"{row['first_step_ms']:>9.2f} {row['second_step_ms']:>9.2f}"
        )
    _write(payload, f"compiled_cache_{args.layers}L.json")



def _timed_call(compiled_model, tokens, state) -> tuple[float, float]:
    """Host build time and end-to-end time for one call of the model step.

    Build time is what a first call pays for the trace and a later call does
    not, which is the number the compile cache is judged on. One synchronize
    on each side, never one per layer.
    """
    mx.synchronize()
    start = time.perf_counter()
    logits, hidden, _next = compiled_model(tokens, state)
    built = time.perf_counter()
    mx.eval(logits, hidden)
    mx.synchronize()
    end = time.perf_counter()
    return (built - start) * 1000.0, (end - start) * 1000.0


def _prefixed(pair: tuple[float, float], prefix: str) -> dict:
    return {f"{prefix}_build_ms": pair[0], f"{prefix}_step_ms": pair[1]}


def cmd_real(args) -> None:
    """The checkpoint arm. One sync per step, no per-scope evaluation."""
    from titan.adapters.mlx import loader

    model, _plan = loader.load_model(Path(args.model).expanduser())
    for context in args.contexts:
        eager, compiled_arm = make_arms(model, context)
        vocab = int(getattr(model, "language_model", model).args.vocab_size)
        rows = []
        for width in args.widths:
            tokens = _tokens(width, vocab)
            result = paired(
                eager,
                compiled_arm,
                tokens,
                repeats=args.repeats,
                warmup=args.warmup,
                count_ops=args.ops,
            )
            result["width"] = width
            rows.append(result)
        payload = {
            "kind": "compiled_real",
            "model": str(args.model),
            "context": context,
            "repeats": args.repeats,
            "rows": rows,
        }
        _report(payload)
        _write(payload, f"compiled_real_{context}.json")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _report(payload: dict) -> None:
    print(f"\ncontext {payload['context']}, {payload.get('layers', 'real')} layers")
    header = (
        f"{'width':>6} {'arm':>10} {'ops':>7} {'build ms':>9} "
        f"{'step ms':>9} {'gpu ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["rows"]:
        for name in ("eager", "compiled"):
            entry = row[name]
            print(
                f"{row['width']:>6} {name:>10} {entry['ops']:>7} "
                f"{entry['build_ms']:>9.2f} {entry['step_ms']:>9.2f} "
                f"{entry['gpu_ms']:>8.2f}"
            )
        e, c = row["eager"], row["compiled"]
        if e["ops"]:
            print(
                f"{'':>6} {'delta':>10} "
                f"{100 * (c['ops'] - e['ops']) / e['ops']:>6.1f}% "
                f"{100 * (c['build_ms'] - e['build_ms']) / e['build_ms']:>8.1f}% "
                f"{100 * (c['step_ms'] - e['step_ms']) / e['step_ms']:>8.1f}%"
            )


def _write(payload: dict, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")


def _int_list(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("synthetic", help="eager against compiled, synthetic")
    synthetic.add_argument("--layers", type=int, default=24)
    synthetic.add_argument("--context", type=int, default=600)
    synthetic.add_argument("--widths", type=_int_list, default=[1, 4])
    synthetic.add_argument("--repeats", type=int, default=12)
    synthetic.add_argument("--warmup", type=int, default=3)
    synthetic.set_defaults(func=cmd_synthetic)

    cache = sub.add_parser("cache", help="compile cost and cache behaviour")
    cache.add_argument("--layers", type=int, default=24)
    cache.add_argument("--context", type=int, default=600)
    cache.add_argument("--max-width", type=int, default=8)
    cache.add_argument(
        "--growth", type=_int_list, default=[200, 500, 1000, 2000, 3000, 5000]
    )
    cache.set_defaults(func=cmd_cache)

    real = sub.add_parser("real", help="the checkpoint arm; one sync per step")
    real.add_argument("--model", required=True)
    real.add_argument("--contexts", type=_int_list, default=[600, 64000])
    real.add_argument("--widths", type=_int_list, default=[1, 4])
    real.add_argument("--repeats", type=int, default=20)
    real.add_argument("--warmup", type=int, default=3)
    real.add_argument("--ops", action="store_true", help="count graph primitives too")
    real.set_defaults(func=cmd_real)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
