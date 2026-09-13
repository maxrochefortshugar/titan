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


def bf16_synthetic(spec: SyntheticSpec, seed: int = 0):
    """The synthetic model in bfloat16, which is what the checkpoint is.

    ``host_overhead.build_synthetic`` leaves the model in float32, and float32
    is a configuration in which none of the fused decode kernels run:
    ``hc_fused`` declines a non-bfloat16 norm weight by layout, and
    ``gdn_norm_gate`` declines a float32 input by dtype. Measuring generation 2
    against an eager arm that has no kernels in it would be measuring nothing,
    and the numerics claim -- same kernels, same bits -- would have nothing to
    check. So the model is cast, and the RMS norm scales are refolded because
    the cast invalidates the ones ``build_synthetic`` prepared.

    ``mamba_ssm_dtype`` is float32 in the config and stays float32: the cast
    walks the parameters, and the recurrent state is allocated per step.
    """
    from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.language import (
        prepare_rmsnorm_scales,
    )

    model = build_synthetic(spec, seed=seed)
    model.set_dtype(mx.bfloat16)
    prepare_rmsnorm_scales(model)
    mx.eval(model.parameters())
    return model


def make_arms(model, length: int, *, use_kernel: bool = True, gen2: bool = False):
    """Prefill once, then hand back an eager arm and a compiled arm at *length*.

    Both arms start from the same prefilled cache, so a difference between them
    is the decode step and not the prefix.

    ``gen2`` builds the generation 2 compiled arm: the eager path's fused Metal
    kernels inside the traces, and the layers that cannot hold one (PLE, and
    sparse attention past the indexer budget) as eager islands between them.
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

    if gen2:
        # The build is for the length the decode starts at, plus the widest
        # verify block, because that is what decides whether the sparse layers
        # are inside the traces or beside them.
        compiled_model = C.build_gen2(model, length=length + 8, use_kernel=use_kernel)
    else:
        compiled_model = C.build_model(model, use_kernel=use_kernel)
    C.eval_weights(compiled_model.weights)
    # The layers that cannot be traced keep their own vendored caches, and the
    # two arms must not share one: the compiled arm's eager layer would advance
    # the cache the eager arm is stepping from. So it gets a copy, prefilled to
    # the same length.
    eager_ids = (
        list(compiled_model.island_indices) if gen2 else C.untraceable_layers(model)
    )
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
            name="gen2" if gen2 else "compiled",
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
    gen2 = getattr(args, "gen2", False)
    # ``--bf16`` without ``--gen2`` is the middle arm of the three-way
    # comparison: generation 1's compiled step on a model whose *eager* arm has
    # the fused kernels in it. That is the arm that shows what generation 1
    # gave away, which is the reason generation 2 exists.
    bf16 = gen2 or getattr(args, "bf16", False)
    model = bf16_synthetic(spec) if bf16 else build_synthetic(spec)
    eager, compiled_arm = make_arms(model, args.context, gen2=gen2)
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
        "generation": 2 if gen2 else 1,
        "dtype": "bfloat16" if bf16 else "float32",
        "islands": list(compiled_arm.model.island_indices),
        "fused_kernels": compiled_arm.model.fused_kernels,
        "layers": args.layers,
        "context": args.context,
        "repeats": args.repeats,
        "rows": rows,
    }
    _report(payload)
    suffix = "_gen2" if gen2 else ("_gen1_bf16" if bf16 else "")
    _write(payload, f"compiled_synthetic{suffix}_{args.layers}L_{args.context}.json")


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



def rotated(arms, tokens_, *, repeats: int, warmup: int, count_ops: bool):
    """Measure every arm forwards and then backwards, and average the pair.

    ``paired`` does A B B A for two arms; this is the same idea for three, so
    the eager, generation 1 and generation 2 arms are measured inside one
    process against one prefill. That matters more than it sounds on this
    machine: the workbench beside it moves absolute step times by a factor of
    three between runs, so a generation 1 number from one process and a
    generation 2 number from another are not comparable, and the whole claim of
    generation 2 is a comparison with generation 1.
    """
    forward = [
        measure(arm, tokens_, repeats=repeats, warmup=warmup, count_ops=count_ops)
        for arm in arms
    ]
    backward = {
        arm.name: measure(arm, tokens_, repeats=repeats, warmup=0, count_ops=False)
        for arm in reversed(arms)
    }
    return {row["arm"]: _merge(row, backward[row["arm"]]) for row in forward}


def cmd_generations(args) -> None:
    """Eager, generation 1 and generation 2, in one process on one prefill.

    The table the generation 2 section of ``docs/architecture/COMPILED.md``
    reports. The model is bfloat16 for all three arms, which is what makes the
    eager arm the honest baseline: in float32 the eager arm has no fused
    kernels in it and generation 1 looks fine.
    """
    spec = SyntheticSpec(num_hidden_layers=args.layers)
    model = bf16_synthetic(spec)
    eager, gen1 = make_arms(model, args.context, gen2=False)
    _eager2, gen2 = make_arms(model, args.context, gen2=True)
    arms = [eager, gen1, gen2]
    if gen1.model.sparse_budget and args.context >= gen1.model.sparse_budget:
        # Generation 1 refuses a length past the indexer budget, by design and
        # for a good reason -- a dense trace there attends to keys the eager
        # path drops. So past the budget there are two arms, not three, and the
        # missing one is the point rather than a gap in the measurement.
        print(
            f"\ncontext {args.context} is past the indexer budget "
            f"{gen1.model.sparse_budget}: the generation 1 arm refuses it, so "
            "this run is eager against generation 2 only"
        )
        arms = [eager, gen2]

    rows = []
    for width in args.widths:
        ids = _tokens(width, spec.vocab_size)
        result = rotated(
            arms,
            ids,
            repeats=args.repeats,
            warmup=args.warmup,
            count_ops=True,
        )
        result["width"] = width
        rows.append(result)

    payload = {
        "kind": "compiled_generations",
        "layers": args.layers,
        "context": args.context,
        "repeats": args.repeats,
        "dtype": "bfloat16",
        "gen2_islands": list(gen2.model.island_indices),
        "rows": rows,
    }
    header = (
        f"{'width':>6} {'arm':>10} {'ops':>7} {'build ms':>9} "
        f"{'step ms':>9} {'gpu ms':>8} {'vs eager':>9}"
    )
    print(f"\ncontext {args.context}, {args.layers} layers, bfloat16")
    print(header)
    print("-" * len(header))
    for row in rows:
        base = row["eager"]["step_ms"]
        for name in ("eager", "compiled", "gen2"):
            entry = row.get(name)
            if entry is None:
                continue
            delta = 100 * (entry["step_ms"] - base) / base
            print(
                f"{row['width']:>6} {name:>10} {entry['ops']:>7} "
                f"{entry['build_ms']:>9.2f} {entry['step_ms']:>9.2f} "
                f"{entry['gpu_ms']:>8.2f} {delta:>8.1f}%"
            )
    _write(payload, f"compiled_generations_{args.layers}L_{args.context}.json")


def cmd_layout(args) -> None:
    """Traces and islands for a checkpoint's layer layout, from its config.

    No model is loaded and no GPU is touched: the answer is a function of
    ``layer_types``, ``ple_layer_ids`` and ``indexer_budget``, all of which are
    in ``config.json``. That matters because the checkpoint is 70 GB and one
    process at a time.
    """
    config = json.loads(Path(args.config).expanduser().read_text())
    text = config.get("text_config", config)
    layer_types = text["layer_types"]
    ple = text.get("ple_layer_ids", []) or []
    budget = int(text.get("indexer_budget", 0) or 0)
    widths = args.widths
    print(f"{len(layer_types)} layers, indexer budget {budget}, PLE at {list(ple)}")
    print(
        f"{'context':>9} {'islands':>8} {'segments':>9} {'cap in key':>11} "
        f"{'traces':>7}"
    )
    rows = []
    for context in args.contexts:
        plan = C.plan_layout(
            layer_types=layer_types,
            ple_layer_ids=ple,
            length=context,
            indexer_budget=budget,
        )
        buckets = C.capacity_buckets(up_to=C.capacity_for(context + max(widths)))
        traces = plan.traces(widths, buckets)
        rows.append(
            {
                "context": context,
                "islands": list(plan.islands),
                "segments": [list(s) for s in plan.segments],
                "capacity_in_trace_key": plan.capacity_in_trace_key,
                "capacity_buckets": buckets,
                "traces": traces,
            }
        )
        print(
            f"{context:>9} {plan.island_count:>8} {plan.segment_count:>9} "
            f"{str(plan.capacity_in_trace_key):>11} {traces:>7}"
        )
    payload = {
        "kind": "compiled_layout",
        "config": str(args.config),
        "layers": len(layer_types),
        "indexer_budget": budget,
        "ple_layer_ids": list(ple),
        "widths": list(widths),
        "rows": rows,
    }
    _write(payload, "compiled_layout.json")


def cmd_kernels(args) -> None:
    """Which custom kernels trace as opaque nodes, and what the others say.

    One row per kernel the decode path can reach, with the shapeless error
    verbatim where there is one. This is the table in the generation 2 section
    of COMPILED.md, generated rather than remembered, because an MLX error
    message is a fact about a version.
    """
    from titan.adapters.mlx import compiled_kernels as CK
    from titan.adapters.mlx.vendor.mlx_lm.models.gated_delta import gated_delta_kernel
    from titan.adapters.mlx.vendor.mlx_vlm.models.rope_utils import (
        _fast_mrope_apply,
        _mrope_apply_kernel,
    )
    from titan.kernels import gdn_norm_gate as _gng
    from titan.kernels import moe_weighted_sum as _mws

    spec = SyntheticSpec(num_hidden_layers=4)
    model = bf16_synthetic(spec)
    inner = getattr(model, "language_model", model).model
    rows = []

    def record(name, fn, args_tuple, note=""):
        try:
            mx.eval(mx.compile(fn)(*args_tuple))
            fixed, fixed_error = True, ""
        except Exception as exc:  # noqa: BLE001 - the message is the result
            fixed, fixed_error = False, f"{type(exc).__name__}: {exc}"
        shapeless_error = (
            CK.opaque_under_shapeless(fn, *args_tuple) if fixed else "not attempted"
        )
        rows.append(
            {
                "kernel": name,
                "traced_at_fixed_shapes": fixed,
                "fixed_shape_error": fixed_error,
                "shapeless_error": shapeless_error or "",
                "note": note,
            }
        )

    # hc_fused: the three kernels of the hyper-connection block
    hc = inner.layers[0].attn_hyper_connection
    plan = CK.hc_fused_plan(hc)
    arrays = CK.hc_fused_arrays(hc)
    mx.eval([a for v in arrays.values() for a in (v if isinstance(v, tuple) else (v,))])
    x = mx.zeros((1, 4, hc.hc_count * hc.hidden_size), mx.bfloat16)
    record(
        "hc_fused (norm + down/inject + up)",
        lambda xx, aa: CK.hc_fused_body(xx, aa, plan),
        (x, arrays),
        "three mx.fast.metal_kernel launches, vendored qwen4_exp/hc_fused.py",
    )

    # gdn_norm_gate
    gdn = inner.layers[0].linear_attn
    heads, dim = gdn.num_v_heads, gdn.head_v_dim
    xg = mx.zeros((1, 4, heads, dim), mx.bfloat16)
    gg = mx.zeros((1, 4, heads, dim), mx.bfloat16)
    wg = mx.ones((dim,), mx.bfloat16)
    record(
        "gdn_norm_gate",
        lambda a, b, c: _gng.metal(a, b, c, eps=1e-6, activation=0),
        (xg, gg, wg),
        "titan.kernels, fused grouped RMS norm and output gate",
    )

    # the gated delta recurrence
    q = mx.zeros((1, 4, gdn.num_k_heads, gdn.head_k_dim), mx.bfloat16)
    k = mx.zeros((1, 4, gdn.num_k_heads, gdn.head_k_dim), mx.bfloat16)
    v = mx.zeros((1, 4, heads, dim), mx.bfloat16)
    g = mx.zeros((1, 4, heads), mx.float32)
    beta = mx.zeros((1, 4, heads), mx.bfloat16)
    ssm = mx.zeros((1, heads, dim, gdn.head_k_dim), mx.float32)
    record(
        "gated_delta_kernel",
        lambda *a: gated_delta_kernel(*a, None),
        (q, k, v, g, beta, ssm),
        "vendored qwen3_5/gated_delta.py, the decode recurrence",
    )

    # the mrope application
    attn = None
    for layer in inner.layers:
        if not layer.is_linear:
            attn = layer.self_attn
            break
    if attn is not None:
        rope = attn.rotary_emb
        rk = _mrope_apply_kernel(rope.dim, 2, rope.pairing)
        heads_q = attn.num_attention_heads
        kvh = attn.num_key_value_heads
        hd = attn.head_dim
        qq = mx.zeros((1, heads_q, 4, hd), mx.bfloat16)
        kk = mx.zeros((1, kvh, 4, hd), mx.bfloat16)
        pos = mx.zeros((1, 4), mx.int32)
        selector = rope.position_selector
        selector = selector if selector is not None else mx.zeros((1,), mx.int32)
        if rk is not None:
            record(
                "_mrope_apply_kernel",
                lambda a, b, c, d, e: _fast_mrope_apply(rk, a, b, c, d, e),
                (qq, kk, pos, rope.inv_freq, selector),
                "vendored rope_utils, the fused rotary application",
            )

    # the MoE weighted sum
    y = mx.zeros((1, 4, spec.num_experts_per_tok, spec.hidden_size), mx.bfloat16)
    scores = mx.zeros((1, 4, spec.num_experts_per_tok), mx.float32)
    record(
        "moe_weighted_sum",
        lambda a, b: _mws.metal(a, None, b, (1, 4, spec.hidden_size)),
        (y, scores),
        "titan.kernels, the top-k unsort and routed weighted sum",
    )

    # The verify block's two ops. They are outside the layer traces, but the
    # goal is a compiled decode *and verify* step, so whether they trace is
    # part of the answer.
    from titan.kernels import topk_radix as _tk
    from titan.kernels import verify_accept as _va

    logits_row = mx.zeros((1, spec.vocab_size), mx.float32)
    record(
        "topk_radix",
        lambda a: _tk.metal(a, 10),
        (logits_row,),
        "titan.kernels, three-launch radix top-K over one wide logits row",
    )
    vlogits = mx.zeros((1, 4, spec.vocab_size), mx.float32)
    drafted = mx.zeros((1, 3), mx.int32)
    record(
        "verify_accept",
        lambda a, b: _va.fast(a, b),
        (vlogits, drafted),
        "titan.kernels; plain MLX ops today, the fused launch is a later change",
    )

    # Two that are not traceable by construction rather than by an MLX error,
    # which is why they are islands rather than nodes.
    rows.append(
        {
            "kernel": "ple_packed_lookup",
            "traced_at_fixed_shapes": False,
            "fixed_shape_error": "not an MLX graph: reads rows from a packed "
            "table on SSD through an mmap and numpy, mid-forward",
            "shapeless_error": "not attempted",
            "note": "eager island (PLE layer)",
        }
    )
    rows.append(
        {
            "kernel": "qsa_gathered_attention",
            "traced_at_fixed_shapes": False,
            "fixed_shape_error": "not a pure array function: QSAGeometry is "
            "built from int(cache.offset) and the pooled index bank's host-side "
            "lengths, offsets and phases, so the selection is host work by "
            "construction",
            "shapeless_error": "not attempted",
            "note": "eager island (sparse attention past the indexer budget)",
        }
    )

    width = max(len(row["kernel"]) for row in rows)
    print(f"\n{'kernel':<{width}}  {'fixed':>5}  shapeless")
    for row in rows:
        mark = "yes" if row["traced_at_fixed_shapes"] else "NO"
        print(f"{row['kernel']:<{width}}  {mark:>5}  {row['shapeless_error']}")
        if row["fixed_shape_error"]:
            print(f"{'':<{width}}         fixed-shape error: {row['fixed_shape_error']}")
    payload = {"kind": "compiled_kernels_probe", "rows": rows}
    _write(payload, "compiled_kernels_probe.json")


def cmd_warm(args) -> None:
    """What warming the (width, capacity) grid costs, and what it holds."""
    spec = SyntheticSpec(num_hidden_layers=args.layers)
    model = bf16_synthetic(spec)
    compiled_model = C.build_gen2(model, length=args.length)
    C.eval_weights(compiled_model.weights)
    capacities = args.capacities or C.capacity_buckets(up_to=args.up_to)
    report = C.warm_traces(
        compiled_model,
        widths=args.widths,
        capacities=capacities,
        budget_seconds=args.budget_seconds,
        budget_mb=args.budget_mb,
        vocab=spec.vocab_size,
    )
    print(
        f"\nwarmed {report.warmed} grid points in {report.seconds:.2f} s, "
        f"{len(report.skipped)} skipped"
    )
    print(f"{'width':>6} {'capacity':>9} {'first ms':>9} {'cached ms':>10}")
    for row in report.rows:
        print(
            f"{row['width']:>6} {row['capacity']:>9} {row['first_ms']:>9.2f} "
            f"{row['second_ms']:>10.2f}"
        )
    summary = report.summary()
    print(
        f"\nfirst-call total {summary['first_call_total_ms']:.1f} ms, "
        f"cached-call median {summary['cached_call_median_ms']:.2f} ms, "
        f"active memory grew {summary['active_growth_mb']:.1f} MB, "
        f"peak {summary['peak_mb']:.1f} MB"
    )
    payload = {
        "kind": "compiled_warm",
        "layers": args.layers,
        "length": args.length,
        "islands": list(compiled_model.island_indices),
        "rows": report.rows,
        "skipped": report.skipped,
        "summary": summary,
    }
    _write(payload, f"compiled_warm_{args.layers}L.json")


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


def _commit(arm) -> None:
    """Keep the state a compiled step returned instead of rewinding it."""
    pending = getattr(arm, "_pending", None)
    if pending is not None:
        arm.state = pending


def _greedy_ids(arm, *, count: int, vocab: int) -> list[int]:
    """*count* greedy tokens from where the arm stands, one eval per token.

    This is the ordinary decode shape and nothing else: the whole step is
    built, the step's outputs and the argmax are evaluated together once, and
    the token is read. No per-scope evaluation, no per-scope synchronize.
    """
    token = _tokens(1, vocab)
    ids: list[int] = []
    for _ in range(count):
        result = arm.step(token)
        arrays = arm.outputs(result)
        nxt = mx.argmax(arrays[0][:, -1, :], axis=-1)
        mx.eval(arrays + [nxt])
        _commit(arm)
        ids.append(int(nxt.item()))
        token = nxt.reshape(1, 1).astype(mx.int64)
    return ids


def _trace_shape(compiled_arm) -> dict:
    """Segments, islands and the trace count a width grid implies."""
    plan = list(getattr(compiled_arm.model, "plan", ()) or ())
    segments = sum(1 for kind, _ in plan if kind == "traced") or 1
    islands = list(compiled_arm.model.island_indices)
    return {"segments": segments, "islands": len(islands), "island_indices": islands}


def cmd_real(args) -> None:
    """The checkpoint arm. One sync per step, no per-scope evaluation."""
    if getattr(args, "reference_only", False):
        # Before the checkpoint is loaded, so the adapter's first lookup and
        # every build-time plan in the compiled path find this registry.
        from titan.adapters.mlx import kernels as adapter_kernels
        from titan.kernels.registry import reference_only

        reference_only()
        adapter_kernels.reset()

    from titan.adapters.mlx import loader

    gen2 = getattr(args, "gen2", False)
    model, _plan = loader.load_model(Path(args.model).expanduser())
    lossless_contexts = args.lossless_contexts or args.contexts[:1]
    for context in args.contexts:
        eager, compiled_arm = make_arms(model, context, gen2=gen2)
        vocab = int(getattr(model, "language_model", model).args.vocab_size)

        warm = None
        if args.warm_widths:
            before = int(mx.get_active_memory())
            report = C.warm_traces(
                compiled_arm.model,
                widths=args.warm_widths,
                capacities=[compiled_arm.state.capacity],
                budget_seconds=600.0,
                budget_mb=32768.0,
                vocab=vocab,
            )
            warm = report.summary()
            shape = _trace_shape(compiled_arm)
            warm.update(shape)
            warm["traces"] = shape["segments"] * report.warmed
            warm["active_before_mb"] = before / 2**20
            warm["rows"] = report.rows

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

        lossless = None
        if args.lossless and context in lossless_contexts:
            # A fresh pair of arms, because the timed rows do not leave one.
            # ``EagerArm.rewind`` trims the KV caches and nothing else, which is
            # right for timing -- the shapes come back and the step is the step
            # -- and wrong for a numerics comparison: the Gated DeltaNet's
            # recurrent state is not trimmable, so after the timed rows the
            # eager arm is dozens of tokens of recurrence ahead of its own KV.
            # Re-prefilling is the only honest way to start both arms level.
            eager, compiled_arm = make_arms(model, context, gen2=gen2)
            eager_ids = _greedy_ids(eager, count=args.lossless, vocab=vocab)
            compiled_ids = _greedy_ids(compiled_arm, count=args.lossless, vocab=vocab)
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(eager_ids, compiled_ids))
                    if a != b
                ),
                None,
            )
            lossless = {
                "tokens": args.lossless,
                "identical": eager_ids == compiled_ids,
                "first_divergence": first_diff,
                "eager_ids": eager_ids,
                "compiled_ids": compiled_ids,
            }

        payload = {
            "kind": "compiled_real",
            "generation": 2 if gen2 else 1,
            "model": str(args.model),
            "context": context,
            "repeats": args.repeats,
            "reference_only": bool(getattr(args, "reference_only", False)),
            "islands": list(compiled_arm.model.island_indices),
            "fused_kernels": compiled_arm.model.fused_kernels,
            "warm": warm,
            "lossless": lossless,
            "peak_memory_gb": mx.get_peak_memory() / 2**30,
            "active_memory_gb": mx.get_active_memory() / 2**30,
            "rows": rows,
        }
        _report(payload)
        if warm:
            print(
                f"  warm: {warm['traces']} traces "
                f"({warm['segments']} segments x {warm['warmed']} widths), "
                f"{warm['seconds']:.2f} s, "
                f"+{warm['active_growth_mb']:.1f} MB active, "
                f"{warm['islands']} islands"
            )
        if lossless:
            print(
                f"  lossless: {lossless['tokens']} greedy tokens, "
                f"identical={lossless['identical']} "
                f"first_divergence={lossless['first_divergence']}"
            )
        print(
            f"  memory: active {payload['active_memory_gb']:.1f} GB, "
            f"peak {payload['peak_memory_gb']:.1f} GB"
        )
        suffix = "_gen2" if gen2 else ""
        if getattr(args, "reference_only", False):
            suffix += "_refonly"
        _write(payload, f"compiled_real{suffix}_{context}.json")


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
                f"{row['width']:>6} {entry.get('arm', name):>10} {entry['ops']:>7} "
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
    synthetic.add_argument(
        "--gen2",
        action="store_true",
        help="generation 2: fused kernels inside the traces, eager islands "
        "around them, and a bfloat16 synthetic model so the eager arm has the "
        "kernels in it too",
    )
    synthetic.add_argument(
        "--bf16",
        action="store_true",
        help="cast the synthetic model to bfloat16 without turning generation "
        "2 on, so a generation 1 arm can be measured against an eager arm that "
        "has the fused kernels in it",
    )
    synthetic.set_defaults(func=cmd_synthetic)

    generations = sub.add_parser(
        "generations",
        help="eager, generation 1 and generation 2 in one process, bfloat16",
    )
    generations.add_argument("--layers", type=int, default=24)
    generations.add_argument("--context", type=int, default=600)
    generations.add_argument("--widths", type=_int_list, default=[1, 4])
    generations.add_argument("--repeats", type=int, default=16)
    generations.add_argument("--warmup", type=int, default=3)
    generations.set_defaults(func=cmd_generations)

    layout = sub.add_parser(
        "layout", help="traces and islands from a config.json; loads nothing"
    )
    layout.add_argument("--config", required=True)
    layout.add_argument("--contexts", type=_int_list, default=[600, 2048, 64000])
    layout.add_argument("--widths", type=_int_list, default=[1, 2, 3, 4, 5, 6, 7, 8])
    layout.set_defaults(func=cmd_layout)

    kernels = sub.add_parser(
        "kernels", help="which custom kernels trace as opaque nodes, and the errors"
    )
    kernels.set_defaults(func=cmd_kernels)

    warm = sub.add_parser("warm", help="the trace cache: grid warm cost and memory")
    warm.add_argument("--layers", type=int, default=24)
    warm.add_argument("--length", type=int, default=600)
    warm.add_argument("--widths", type=_int_list, default=[1, 2, 3, 4, 5, 6, 7, 8])
    warm.add_argument("--capacities", type=_int_list, default=None)
    warm.add_argument("--up-to", type=int, default=8192)
    warm.add_argument("--budget-seconds", type=float, default=120.0)
    warm.add_argument("--budget-mb", type=float, default=2048.0)
    warm.set_defaults(func=cmd_warm)

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
    real.add_argument(
        "--gen2",
        action="store_true",
        help="generation 2: fused kernels inside the traces, eager islands "
        "for the PLE layer and for the sparse-attention layers past the "
        "indexer budget",
    )
    real.add_argument(
        "--reference-only",
        action="store_true",
        help="INCIDENTS rule 2's control arm: every fast path off, on both "
        "arms, so the compiled step and the eager step resolve the same ops "
        "through the same registry door",
    )
    real.add_argument(
        "--warm-widths",
        type=_int_list,
        default=[],
        help="warm the trace grid at these verify widths before measuring, "
        "and report what the grid cost",
    )
    real.add_argument(
        "--lossless",
        type=int,
        default=0,
        help="after the timed rows, take this many greedy tokens on each arm "
        "and compare the argmax sequences",
    )
    real.add_argument(
        "--lossless-contexts",
        type=_int_list,
        default=[],
        help="contexts the greedy comparison runs at; default is the first",
    )
    real.set_defaults(func=cmd_real)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
