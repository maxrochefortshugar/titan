# SPDX-License-Identifier: Apache-2.0
"""End-to-end exactness of the installed patch on a real Qwen4ExpAttention layer.

Builds one randomly initialized attention layer at the real Flash-Next geometry
(no checkpoint is read, no model is loaded), installs patch.py, and compares:

  batched arm on a BatchQSAKVCache of B mixed-length rows
      vs
  the stock single-sequence gathered arm on each of those rows separately.

Same weights, same inputs, same RoPE, so the outputs must agree row by row.

Usage: test_layer_batched.py [--big]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from common import BatchQSAKVCache, lang, make_row, mx, ulp_bf16  # noqa: E402

CONFIG_PATH = Path(
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/config.json"
)


def load_patch():
    spec = importlib.util.spec_from_file_location(
        "_qsab_patch", Path(__file__).resolve().parent / "patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_layer():
    raw = json.loads(CONFIG_PATH.read_text())
    text = raw.get("text_config", raw)
    config = lang.TextConfig.from_dict(text)
    layer = lang.Qwen4ExpAttention(config)
    mx.random.seed(3)

    def randomize(module):
        for name, value in list(module.parameters().items()):
            if isinstance(value, mx.array):
                module.update(
                    {name: (mx.random.normal(value.shape) * 0.02).astype(mx.bfloat16)}
                )
            elif isinstance(value, dict):
                randomize(getattr(module, name))

    for name, child in layer.children().items():
        if isinstance(child, object) and hasattr(child, "parameters"):
            randomize(child)
    randomize(layer)
    mx.eval(layer.parameters())
    return layer, config


def run(layer, config, lengths, length, explicit_positions=True):
    hidden = config.hidden_size
    batch = len(lengths)
    mx.random.seed(11)
    x = (mx.random.normal((batch, length, hidden)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    rows = [make_row(n, 1000 + i) for i, n in enumerate(lengths)]
    batched_cache = BatchQSAKVCache.merge(rows)
    position_ids = mx.stack(
        [mx.arange(n, n + length, dtype=mx.int32) for n in lengths]
    )
    batched = layer(
        x,
        None,
        batched_cache,
        position_ids if explicit_positions else None,
        None,
        length > 1,
    )
    mx.eval(batched)

    singles = []
    for i, n in enumerate(lengths):
        cache = make_row(n, 1000 + i)
        row_x = mx.contiguous(x[i : i + 1])
        row_ids = position_ids[i : i + 1]
        if length == 1:
            out = layer._gathered_text_decode(row_x, cache, row_ids)
        else:
            out = layer._gathered_text_prefill(
                row_x, cache, row_ids, target_verify=True
            )
        mx.eval(out)
        singles.append(out)
    reference = mx.concatenate(singles, axis=0)
    err = float(
        mx.max(mx.abs(batched.astype(mx.float32) - reference.astype(mx.float32)))
    )
    return err, ulp_bf16(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", action="store_true")
    args = parser.parse_args()

    os.environ["OMLX_QSA_BATCHED_SPARSE"] = "1"
    os.environ["OMLX_QSA_BATCHED_VERIFY"] = "1"
    os.environ.setdefault("OMLX_QSA_GATHER_MIN_CTX", "8192")
    patch = load_patch()
    print(f"install() -> {patch.install(None)}")
    print(f"install() again -> {patch.install(None)}   (idempotent)")

    layer, config = build_layer()
    configs = [
        ([9000, 16384], 1),
        ([9000, 16384], 4),
        ([9001, 16382, 12288, 20477], 1),
        ([9001, 16382, 12288, 20477], 4),
    ]
    if args.big:
        configs += [([16384, 65536], 1), ([16384, 65536], 4)]

    print(f"\nroute = {os.environ.get('OMLX_QSA_BATCHED_ROUTE', 'padded')}")
    print(f"{'lengths':>34} {'L':>2} | {'max abs':>10} {'1 bf16 ULP':>11}  verdict")
    failures = 0
    for lengths, length in configs:
        err, ulp = run(layer, config, lengths, length)
        ok = err <= ulp
        failures += not ok
        label = ",".join(str(n) for n in lengths)
        print(
            f"{label:>34} {length:>2} | {err:>10.3e} {ulp:>11.3e}  "
            f"{'ok' if ok else 'FAIL'}"
        )
        mx.clear_cache()

    # position_ids=None is the common production shape: the arm has to derive
    # per-row positions from the cache's own padding.
    err, ulp = run(layer, config, [9001, 16382, 12288, 20477], 1,
                   explicit_positions=False)
    failures += err > ulp
    print(
        f"\nderived positions (position_ids=None), B=4 L=1: max abs {err:.3e}, "
        f"1 bf16 ULP {ulp:.3e}  {'ok' if err <= ulp else 'FAIL'}"
    )
    print(f"peak GPU MB: {mx.get_peak_memory() / 1e6:.0f}")
    print("routing is covered by probe_routing_batched.py")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
