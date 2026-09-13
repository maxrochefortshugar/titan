#!/usr/bin/env python3
"""Where a decode cycle's host time goes, and what the dispatch knobs do to it.

One decode cycle on the real model at a 600-token context measures 18.1 ms, of
which 15.8 ms is the host building the graph in Python and 2.0 ms is the GPU
running it. This script is the reproducible measurement of that split and of
the two dispatch experiments the survey asks for, and it runs against either a
small synthetic model (default, a few MB, safe to run beside a busy GPU) or the
real checkpoint.

Four subcommands.

``profile``
    Op counts, eval counts and Python time by module for a decode step at
    width 1 and a verify step at width 4. This is the table in
    ``docs/architecture/FORWARD.md``.

``paths``
    The same two steps with each forward path on and off, paired inside one
    process. This is the before/after table for a change to the forward.

``dispatch``
    Experiment 1 and 2 from ``docs/research/OPTIMISATION-SOURCES.md`` section
    5: eager dispatch in {on, off} crossed with MLX_MAX_OPS_PER_BUFFER in a
    sweep. MLX reads its buffer budget once at process start, so each cell runs
    in its own subprocess and reports back as JSON.

``buffers``
    The check experiment 1 calls mandatory: generate past 12k tokens on the
    winning configuration and watch memory grow. mlx-lm issue #1332 records a
    model dying at 11,300 tokens on a limit that counts buffers rather than
    bytes, so a configuration that is faster over 600 tokens is not yet a win.

Real-model measurement, the one command
---------------------------------------

    python bench/decode/host_overhead.py profile \
        --model ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
        --context 600 --widths 1,4 --repeats 20

It loads the checkpoint through ``titan.adapters.mlx.loader.load_model``, runs
a real prefill to the requested context, and prints the same table the
synthetic run prints. Nothing else in this file touches the checkpoint, so the
sweeps can be pointed at it one at a time:

    python bench/decode/host_overhead.py paths --model <dir> --context 600
    python bench/decode/host_overhead.py dispatch --model <dir> --context 600
    python bench/decode/host_overhead.py buffers --model <dir> --tokens 12288

Counting method
---------------

MLX is lazy, so "how many ops did this step build" is a question about the
graph, not about the wall clock. :class:`Trace` wraps ``mx.eval`` and
``mx.async_eval`` for the duration of a step, walks the graph reachable from
each call's arguments with ``mx.export_to_dot`` before letting the call
through, and counts the primitive nodes it finds. Nodes evaluated by an earlier
call are leaves by the time a later call sees them, so nothing is counted
twice, and the closing ``finish`` sweeps up whatever the step left unevaluated.
The op count is therefore the number of primitives the host built, which is
what dispatch cost is proportional to. It is not the number of Metal
dispatches: MLX fuses some primitives and a ``mx.fast.metal_kernel`` is one
node whatever it does inside.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from titan.adapters.mlx.vendor.mlx_vlm.models import forward_paths  # noqa: E402
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.config import (  # noqa: E402
    ModelConfig,
    TextConfig,
    VisionConfig,
)
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    prepare_rmsnorm_scales,
)
from titan.adapters.mlx.vendor.mlx_vlm.models.qwen4_exp.qwen4_exp import (  # noqa: E402
    Model,
)

# ---------------------------------------------------------------------------
# the synthetic model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticSpec:
    """A Qwen4-Exp small enough to profile beside a busy GPU.

    Every structural property the host cost depends on is kept: the linear and
    sparse attention layer mix, the 4-wide hyper-connection residual, the MoE
    router with a fused gate/up SwitchGLU, the QSA indexer, and the 4-bit
    affine quantisation the checkpoint carries. What is dropped is size, and
    the PLE n-gram layers, which read their table from the checkpoint on SSD
    and cannot be synthesised (see FORWARD.md, "what the synthetic model does
    not cover").
    """

    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    hc_count: int = 4
    hc_lowrank: int = 64
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 64
    shared_expert_intermediate_size: int = 64
    vocab_size: int = 512
    linear_num_value_heads: int = 8
    linear_num_key_heads: int = 4
    linear_key_head_dim: int = 32
    linear_value_head_dim: int = 32
    linear_conv_kernel_dim: int = 4
    indexer_n_heads: int = 2
    indexer_head_dim: int = 32
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    max_position_embeddings: int = 8192
    quantize: bool = True
    quant_bits: int = 4
    quant_group_size: int = 64


def synthetic_config(spec: SyntheticSpec) -> ModelConfig:
    text = TextConfig(
        model_type="qwen4_exp",
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        linear_num_value_heads=spec.linear_num_value_heads,
        linear_num_key_heads=spec.linear_num_key_heads,
        linear_key_head_dim=spec.linear_key_head_dim,
        linear_value_head_dim=spec.linear_value_head_dim,
        linear_conv_kernel_dim=spec.linear_conv_kernel_dim,
        num_experts=spec.num_experts,
        num_experts_per_tok=spec.num_experts_per_tok,
        shared_expert_intermediate_size=spec.shared_expert_intermediate_size,
        moe_intermediate_size=spec.moe_intermediate_size,
        rms_norm_eps=1e-6,
        vocab_size=spec.vocab_size,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=spec.max_position_embeddings,
        hc_count=spec.hc_count,
        hc_lowrank=spec.hc_lowrank,
        head_dim=spec.head_dim,
        full_attention_interval=4,
        ple_layer_ids=[],
        indexer_n_heads=spec.indexer_n_heads,
        indexer_kv_heads=1,
        indexer_head_dim=spec.indexer_head_dim,
        indexer_budget=spec.indexer_budget,
        indexer_compress_ratio=spec.indexer_compress_ratio,
        eos_token_id=0,
        tie_word_embeddings=False,
    )
    return ModelConfig(
        text_config=text,
        vision_config=VisionConfig(model_type="qwen4_exp"),
        model_type="qwen4_exp",
        vocab_size=spec.vocab_size,
    )


def build_synthetic(spec: SyntheticSpec = SyntheticSpec(), seed: int = 0):
    """Build and initialise the synthetic model. Deterministic given *seed*."""
    mx.random.seed(seed)
    model = Model(synthetic_config(spec))
    if spec.quantize:
        def predicate(_path, module):
            # Mirror the checkpoint: every projection whose input is a whole
            # number of quantisation groups is quantised, the rest stay bf16.
            weight = getattr(module, "weight", None)
            if weight is None or not hasattr(module, "to_quantized"):
                return False
            return weight.shape[-1] % spec.quant_group_size == 0

        nn.quantize(
            model,
            group_size=spec.quant_group_size,
            bits=spec.quant_bits,
            class_predicate=predicate,
        )
    model.eval()
    mx.eval(model.parameters())
    # The real loader does this in ``Model.load_weights``; the synthetic model
    # never goes through it, and without it the norms rebuild ``1 + weight``
    # every step and the op counts here would not match the real path.
    prepare_rmsnorm_scales(model)
    return model


def load_real(model_dir: str):
    """The real checkpoint, through the loader the server uses."""
    from titan.adapters.mlx import loader

    model, _plan = loader.load_model(Path(model_dir).expanduser())
    return model


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------


def _graph_ops(arrays: Sequence[Any]) -> tuple[int, Counter]:
    """Primitive nodes reachable from *arrays*, by name.

    An array that has already been evaluated is a leaf and contributes
    nothing, which is what makes summing over the eval boundaries of one step
    add up rather than double count.
    """
    live = [a for a in arrays if isinstance(a, mx.array)]
    if not live:
        return 0, Counter()
    buf = io.StringIO()
    try:
        mx.export_to_dot(buf, *live)
    except Exception:  # noqa: BLE001 - counting must never break a measurement
        return 0, Counter()
    names = Counter()
    for line in buf.getvalue().splitlines():
        marker = 'label ="'
        start = line.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = line.find('"', start)
        if end > start:
            names[line[start:end]] += 1
    return sum(names.values()), names


def _flatten(args: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    stack = list(args)
    while stack:
        item = stack.pop()
        if isinstance(item, mx.array):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    return out


@dataclass
class TraceResult:
    ops: int = 0
    evals: int = 0
    async_evals: int = 0
    by_primitive: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ops": self.ops,
            "evals": self.evals,
            "async_evals": self.async_evals,
            "top_primitives": self.by_primitive.most_common(12),
        }


class Trace:
    """Count graph primitives and eval calls for the duration of a step.

    Used as a context manager. ``finish`` is called on the way out with
    whatever the step returned, so the tail of the graph is counted and
    evaluated in one place.
    """

    def __init__(self, count_ops: bool = True) -> None:
        self.count_ops = count_ops
        self.result = TraceResult()
        self._saved: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> "Trace":
        self._saved = {"eval": mx.eval, "async_eval": mx.async_eval}
        real_eval = self._saved["eval"]
        real_async = self._saved["async_eval"]

        def traced_eval(*args, **kwargs):
            self.result.evals += 1
            self._count(args)
            return real_eval(*args, **kwargs)

        def traced_async(*args, **kwargs):
            self.result.async_evals += 1
            self._count(args)
            return real_async(*args, **kwargs)

        mx.eval = traced_eval
        mx.async_eval = traced_async
        return self

    def __exit__(self, *exc) -> None:
        mx.eval = self._saved["eval"]
        mx.async_eval = self._saved["async_eval"]
        return None

    def _count(self, args: Sequence[Any]) -> None:
        if not self.count_ops:
            return
        total, names = _graph_ops(_flatten(list(args)))
        self.result.ops += total
        self.result.by_primitive.update(names)

    def finish(self, *outputs: Any) -> TraceResult:
        """Count and evaluate the step's tail. Call inside the ``with``."""
        arrays = _flatten(list(outputs))
        self._count(arrays)
        self._saved["eval"](arrays)
        return self.result


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    """A model plus one live cache, at a known context length."""

    model: Any
    context: int
    vocab: int

    def __post_init__(self) -> None:
        self.language_model = self.model.language_model
        self.cache = self.language_model.make_cache()
        self._token = 1

    def next_tokens(self, width: int) -> mx.array:
        ids = [(self._token + i) % max(2, self.vocab - 1) + 1 for i in range(width)]
        self._token = (self._token + width) % max(2, self.vocab - 1)
        return mx.array([ids], dtype=mx.int64)

    def prefill(self, length: int, chunk: int = 512) -> None:
        done = 0
        while done < length:
            piece = min(chunk, length - done)
            ids = self.next_tokens(piece)
            out = self.language_model(ids, cache=self.cache, skip_logits=True)
            mx.eval(_cache_arrays(self.cache), out.logits)
            done += piece

    def decode_step(self, width: int) -> Any:
        """One forward at *width* rows. Width 1 is a decode, width > 1 a verify.

        Both ask for the hidden state, which is what the engine does: decode
        feeds the MTP head and verify needs the recurrent intermediates for a
        replay-free rollback.
        """
        ids = self.next_tokens(width)
        return self.language_model(ids, cache=self.cache, return_hidden=True)

    def rewind(self, width: int) -> None:
        """Undo a step so the next one runs at the same context length."""
        for cache in self.cache:
            trim = getattr(cache, "trim", None)
            if trim is not None and getattr(cache, "is_trimmable", lambda: False)():
                trim(width)


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


def _step_outputs(output, caches) -> list:
    arrays = [output.logits]
    arrays.extend(_cache_arrays(caches))
    for group in (output.hidden_states or [], output.gdn_states or []):
        arrays.extend(group)
    return [a for a in arrays if a is not None]


def measure_step(
    harness: Harness,
    width: int,
    *,
    repeats: int = 10,
    warmup: int = 3,
    count_ops: bool = True,
    profile_python: bool = False,
) -> dict[str, Any]:
    """Op count, host build time and end-to-end step time at one width.

    ``build_ms`` is the Python time to construct the graph, measured with the
    step's own evaluation pushed to the end. ``step_ms`` adds the wait for the
    GPU. ``gpu_ms`` is the difference, which is the idle-time figure the
    integration report quotes, and it is a lower bound on GPU time rather than
    a measurement of it: whatever the GPU finished while the host was still
    building does not appear.
    """
    for _ in range(warmup):
        output = harness.decode_step(width)
        mx.eval(_step_outputs(output, harness.cache))
        harness.rewind(width)

    ops = evals = async_evals = 0
    primitives: Counter = Counter()
    build = []
    total = []
    for index in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        with Trace(count_ops=count_ops and index == 0) as trace:
            output = harness.decode_step(width)
            built = time.perf_counter()
            trace.finish(_step_outputs(output, harness.cache))
        mx.synchronize()
        end = time.perf_counter()
        if index == 0 and count_ops:
            ops = trace.result.ops
            evals = trace.result.evals
            async_evals = trace.result.async_evals
            primitives = trace.result.by_primitive
        else:
            build.append((built - start) * 1000.0)
            total.append((end - start) * 1000.0)
        harness.rewind(width)

    python_time = None
    if profile_python:
        python_time = profile_python_time(harness, width)

    return {
        "width": width,
        "ops": ops,
        "evals": evals,
        "async_evals": async_evals,
        "top_primitives": primitives.most_common(10),
        "build_ms": _median(build),
        "step_ms": _median(total),
        "gpu_ms": max(0.0, _median(total) - _median(build)),
        "python_by_module": python_time,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def profile_python_time(harness: Harness, width: int, repeats: int = 5) -> list:
    """Cumulative Python time by module over *repeats* steps, largest first.

    cProfile inflates every call, so read the shares rather than the
    milliseconds: the question this answers is which module the host is in,
    not how long the step takes.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeats):
        output = harness.decode_step(width)
        mx.eval(_step_outputs(output, harness.cache))
        harness.rewind(width)
    profiler.disable()

    stats = pstats.Stats(profiler, stream=io.StringIO())
    by_module: Counter = Counter()
    for (filename, _line, _name), entry in stats.stats.items():  # type: ignore[attr-defined]
        total_time = entry[2]  # tottime, exclusive of subcalls
        by_module[_module_label(filename)] += total_time
    grand = sum(by_module.values()) or 1.0
    return [
        {
            "module": name,
            "ms_per_step": value * 1000.0 / repeats,
            "share": value / grand,
        }
        for name, value in by_module.most_common(15)
    ]


def _module_label(filename: str) -> str:
    if filename in ("~", "<built-in>") or filename.startswith("<"):
        return "builtins/mlx C extension"
    path = Path(filename)
    marker = "mlx_vlm/models/"
    text = str(path)
    if marker in text:
        return "vendor/" + text.split(marker, 1)[1]
    if "/titan/" in text:
        return "titan/" + text.split("/titan/", 1)[1]
    if "/mlx/" in text:
        return "mlx/" + path.name
    return path.name


# ---------------------------------------------------------------------------
# harness construction
# ---------------------------------------------------------------------------


def make_harness(args) -> Harness:
    if args.model:
        model = load_real(args.model)
        vocab = model.config.text_config.vocab_size
    else:
        spec = SyntheticSpec(
            quantize=not args.no_quantize,
            num_hidden_layers=args.layers,
            hidden_size=args.hidden,
        )
        model = build_synthetic(spec)
        vocab = spec.vocab_size
    harness = Harness(model=model, context=args.context, vocab=vocab)
    if args.context:
        harness.prefill(args.context)
    return harness


def _widths(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# subcommand: profile
# ---------------------------------------------------------------------------


def cmd_profile(args) -> int:
    harness = make_harness(args)
    rows = []
    for width in _widths(args.widths):
        rows.append(
            measure_step(
                harness,
                width,
                repeats=args.repeats,
                count_ops=True,
                profile_python=True,
            )
        )
    _print_header(args)
    _print_step_table(rows)
    for row in rows:
        print()
        print(f"  width {row['width']}: top primitives")
        for name, count in row["top_primitives"]:
            print(f"    {count:6d}  {name}")
        print(f"  width {row['width']}: Python time by module (cProfile, tottime)")
        for entry in row["python_by_module"] or []:
            print(
                f"    {entry['ms_per_step']:8.2f} ms  {entry['share'] * 100:5.1f}%  "
                f"{entry['module']}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


def _print_header(args) -> None:
    target = args.model or "synthetic"
    print(f"target        {target}")
    print(f"context       {args.context}")
    print(f"forward paths {forward_paths.snapshot()}")
    print(
        "mlx buffers   MLX_MAX_OPS_PER_BUFFER="
        f"{os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')} "
        f"MLX_MAX_MB_PER_BUFFER={os.environ.get('MLX_MAX_MB_PER_BUFFER', 'default')}"
    )
    print()


def _print_step_table(rows: Sequence[dict]) -> None:
    print(
        f"{'width':>5}  {'ops':>7}  {'eval':>5}  {'async':>5}  "
        f"{'build ms':>9}  {'step ms':>8}  {'gpu ms':>7}"
    )
    for row in rows:
        print(
            f"{row['width']:>5}  {row['ops']:>7}  {row['evals']:>5}  "
            f"{row['async_evals']:>5}  {row['build_ms']:>9.3f}  "
            f"{row['step_ms']:>8.3f}  {row['gpu_ms']:>7.3f}"
        )


# ---------------------------------------------------------------------------
# subcommand: paths
# ---------------------------------------------------------------------------


def cmd_paths(args) -> int:
    """Every path on and off, paired in one process.

    Pairing matters more than it looks. The GPU is shared, the machine warms
    up, and two runs of this script minutes apart disagree by more than any of
    these paths is worth. Two settings inside one process, interleaved by
    width, do not.

    The last block is the one the write-up quotes: every path this workstream
    added, off together against on together, which is the forward before
    against the forward after.
    """
    names = (
        [n.strip() for n in args.paths.split(",") if n.strip()]
        if args.paths
        else sorted(forward_paths.DEFAULTS)
    )
    harness = make_harness(args)
    widths = _widths(args.widths)
    _print_header(args)
    header = (
        f"{'path':<26} {'setting':<4} {'width':>5} {'ops':>7} {'async':>5} "
        f"{'build ms':>9} {'step ms':>8}"
    )
    print(header)
    results = []

    def block(label: str, paths: dict) -> None:
        with forward_paths.overridden(**paths):
            for width in widths:
                row = measure_step(
                    harness, width, repeats=args.repeats, count_ops=True
                )
                row["path"] = label
                row["paths"] = forward_paths.snapshot()
                results.append(row)
                print(
                    f"{label:<26} {'':<4} "
                    f"{width:>5} {row['ops']:>7} {row['async_evals']:>5} "
                    f"{row['build_ms']:>9.3f} {row['step_ms']:>8.3f}"
                )

    for name in names:
        for setting in (True, False):
            block(f"{name} {'on' if setting else 'off'}", {name: setting})

    print()
    print("combined: every added path off against on, in A B B A order so the")
    print("machine warming up over the run cannot favour either side")
    print(header)
    off = {name: False for name in forward_paths.ADDED}
    on = {name: True for name in forward_paths.ADDED}
    for label, paths in (
        ("added off (A)", off),
        ("added on (B)", on),
        ("added on (B)", on),
        ("added off (A)", off),
    ):
        block(label, paths)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
# subcommand: dispatch (experiments 1 and 2)
# ---------------------------------------------------------------------------

#: (ops per buffer, MB per buffer). ``None`` keeps MLX's per-architecture
#: default, which PR #1864 sets to 50 and 50 on Max and Ultra parts.
DEFAULT_BUFFER_SWEEP: tuple[Optional[tuple[int, int]], ...] = (
    None,
    (200, 200),
    (500, 500),
)


def cmd_dispatch(args) -> int:
    cells = []
    for eager in (True, False):
        for budget in DEFAULT_BUFFER_SWEEP:
            cells.append((eager, budget))

    print(f"target        {args.model or 'synthetic'}")
    print(f"context       {args.context}   widths {args.widths}")
    print("Each cell is its own process: MLX reads its command-buffer budget once,")
    print("at start, so it cannot be swept in place.")
    print()
    print(
        f"{'eager':<6} {'ops/buf':>8} {'MB/buf':>7} {'width':>5} {'ops':>7} "
        f"{'async':>5} {'build ms':>9} {'step ms':>8}"
    )
    rows = []
    for eager, budget in cells:
        env = dict(os.environ)
        if budget is None:
            env.pop("MLX_MAX_OPS_PER_BUFFER", None)
            env.pop("MLX_MAX_MB_PER_BUFFER", None)
        else:
            env["MLX_MAX_OPS_PER_BUFFER"] = str(budget[0])
            env["MLX_MAX_MB_PER_BUFFER"] = str(budget[1])
        cell = _run_cell(args, env, eager)
        for row in cell:
            rows.append(
                {
                    "eager_dispatch": eager,
                    "ops_per_buffer": budget[0] if budget else None,
                    "mb_per_buffer": budget[1] if budget else None,
                    **row,
                }
            )
            print(
                f"{'on' if eager else 'off':<6} "
                f"{(budget[0] if budget else 'default'):>8} "
                f"{(budget[1] if budget else 'default'):>7} "
                f"{row['width']:>5} {row['ops']:>7} {row['async_evals']:>5} "
                f"{row['build_ms']:>9.3f} {row['step_ms']:>8.3f}"
            )
    print()
    print("Experiment 1 acceptance: a cell whose step time moves by more than 3%")
    print("against the eager-on default pins the flag with a Titan measurement.")
    print("Before acting on any move, run:  host_overhead.py buffers --tokens 12288")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


def _run_cell(args, env: dict, eager: bool) -> list[dict]:
    """One sweep cell, in a fresh process, returning its rows."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_cell",
        "--context",
        str(args.context),
        "--widths",
        args.widths,
        "--repeats",
        str(args.repeats),
        "--eager",
        "1" if eager else "0",
    ]
    if args.model:
        command += ["--model", args.model]
    if getattr(args, "no_quantize", False):
        command += ["--no-quantize"]
    command += ["--layers", str(args.layers), "--hidden", str(args.hidden)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr)
        raise SystemExit(f"sweep cell failed: eager={eager}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def cmd_cell(args) -> int:
    """Internal: one sweep cell. Prints one line of JSON on stdout."""
    forward_paths.set_paths(eager_dispatch=bool(int(args.eager)))
    harness = make_harness(args)
    rows = [
        measure_step(harness, width, repeats=args.repeats, count_ops=True)
        for width in _widths(args.widths)
    ]
    for row in rows:
        row.pop("python_by_module", None)
        row["top_primitives"] = []
    print(json.dumps(rows))
    return 0


# ---------------------------------------------------------------------------
# subcommand: buffers
# ---------------------------------------------------------------------------


def _eager_stepper(harness):
    """One ordinary decode step, evaluated once."""

    def once() -> None:
        output = harness.decode_step(1)
        mx.eval(_step_outputs(output, harness.cache))

    return once


def _compiled_stepper(harness):
    """One compiled decode step, evaluated once, state in and state out.

    COMPILED.md section 7 asks for this arm by name: the compiled path holds
    *fewer* live buffers than the eager one at steady state, because its KV is
    fixed capacity rather than reallocating, but ``grow_state`` holds both the
    old and the new buffer at every growth point and at 64k both are large.
    The question this answers is whether that shows up as unbounded growth.
    """
    from titan.adapters.mlx import compiled as C

    model = C.build_model(harness.model)
    C.eval_weights(model.weights)
    # ``buffers`` defaults to no prefill, and an empty vendored cache has no
    # arrays to bridge from. Starting from a fresh state is also the harder
    # question here: it crosses every capacity growth point on the way to
    # 12,288 tokens rather than starting past most of them.
    state = (
        C.read_layer_state(harness.cache)
        if harness.context
        else C.new_state(harness.model)
    )
    mx.eval([array for entry in state.layers for array in entry.arrays])
    held = {"state": state}

    def once() -> None:
        tokens = harness.next_tokens(1)
        logits, _hidden, new_state = model(tokens, held["state"])
        held["state"] = new_state
        mx.eval(logits, *[a for e in new_state.layers for a in e.arrays])

    return once


def cmd_buffers(args) -> int:
    """Generate past 12k tokens and watch memory, per mlx-lm issue #1332.

    MLX 0.32.2 exposes bytes, not the Metal buffer count the 499,000 limit
    actually counts, so this reports active, peak and cache memory and the
    per-step growth in active memory. A configuration that holds every
    per-step intermediate live shows up as active memory rising without bound;
    one that does not shows a flat line after the cache reaches its steady
    state.
    """
    harness = make_harness(args)
    stepper = _compiled_stepper(harness) if args.compiled else _eager_stepper(harness)
    mx.reset_peak_memory()
    samples = []
    start = time.perf_counter()
    previous = mx.get_active_memory()
    for step in range(1, args.tokens + 1):
        stepper()
        if step % args.every == 0 or step == args.tokens:
            active = mx.get_active_memory()
            samples.append(
                {
                    "step": step,
                    "active_mb": active / 1e6,
                    "peak_mb": mx.get_peak_memory() / 1e6,
                    "cache_mb": mx.get_cache_memory() / 1e6,
                    "growth_kb_per_step": (active - previous) / 1e3 / args.every,
                    "elapsed_s": time.perf_counter() - start,
                }
            )
            previous = active

    print(f"target        {args.model or 'synthetic'}")
    print(f"arm           {'compiled' if args.compiled else 'eager'}")
    print(f"forward paths {forward_paths.snapshot()}")
    print(
        "mlx buffers   MLX_MAX_OPS_PER_BUFFER="
        f"{os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')}"
    )
    print()
    print(
        f"{'step':>7} {'active MB':>10} {'peak MB':>9} {'cache MB':>9} "
        f"{'growth KB/step':>15} {'elapsed s':>10}"
    )
    for row in samples:
        print(
            f"{row['step']:>7} {row['active_mb']:>10.1f} {row['peak_mb']:>9.1f} "
            f"{row['cache_mb']:>9.1f} {row['growth_kb_per_step']:>15.2f} "
            f"{row['elapsed_s']:>10.1f}"
        )
    tail = samples[-3:]
    drift = sum(row["growth_kb_per_step"] for row in tail) / max(1, len(tail))
    print()
    print(f"steady-state growth over the last samples: {drift:.2f} KB/step")
    print(
        "mlx-lm #1332 measured 205 KB/step before the fix and 7 KB/step after; "
        "a run that stays near zero here is not accumulating per-step graphs."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(samples, indent=2))
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, context_default=600):
        p.add_argument("--model", default=None, help="checkpoint directory; "
                       "omit for the synthetic model")
        p.add_argument("--context", type=int, default=context_default)
        p.add_argument("--widths", default="1,4")
        p.add_argument("--repeats", type=int, default=20)
        p.add_argument("--no-quantize", action="store_true",
                       help="synthetic model in bf16 rather than 4-bit affine")
        p.add_argument("--layers", type=int, default=4,
                       help="synthetic decoder layers; deeper reads per-layer "
                            "effects out of the noise")
        p.add_argument("--hidden", type=int, default=128)
        p.add_argument("--json", default=None, help="also write the rows here")

    profile = sub.add_parser("profile", help="op counts and Python time by module")
    common(profile)
    profile.set_defaults(func=cmd_profile)

    paths = sub.add_parser("paths", help="each forward path on and off, paired")
    common(paths)
    paths.add_argument("--paths", default=None, help="comma-separated subset")
    paths.set_defaults(func=cmd_paths)

    dispatch = sub.add_parser(
        "dispatch", help="experiments 1 and 2: eager dispatch x buffer budget"
    )
    common(dispatch)
    dispatch.set_defaults(func=cmd_dispatch)

    buffers = sub.add_parser(
        "buffers", help="the mandatory 12k-token buffer growth check"
    )
    common(buffers, context_default=0)
    buffers.add_argument("--tokens", type=int, default=12288)
    buffers.add_argument("--every", type=int, default=512)
    buffers.add_argument(
        "--compiled",
        action="store_true",
        help="drive the compiled decode step rather than the eager forward",
    )
    buffers.set_defaults(func=cmd_buffers)

    cell = sub.add_parser("_cell", help=argparse.SUPPRESS)
    common(cell)
    cell.add_argument("--eager", default="1")
    cell.set_defaults(func=cmd_cell)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
