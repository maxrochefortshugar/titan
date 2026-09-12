#!/usr/bin/env python3
"""Check ngram.py against the vendored mlx implementation, op for op.

The vendored module cannot be imported without dragging in mlx_vlm and the
runtime PLE globals, so the four module functions and the two methods that
compute row ids are lifted out of the file by AST and exec'd verbatim.  Any
divergence between that source and ngram.py shows up as a mismatch here.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram import NGramIndexer  # noqa: E402

LANG = Path(
    "/Applications/oMLX.app/Contents/Resources/omlx/patches/"
    "mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py"
)
WANT_FUNCS = {
    "_splitmix64",
    "_build_layer_multipliers",
    "_is_prime",
    "_find_nth_prime_after",
}
WANT_METHODS = {"_shift_right_ignore_eos", "_ngram_indices"}


def lift_reference():
    tree = ast.parse(LANG.read_text())
    ns = {"mx": mx, "np": np, "math": math}
    src = LANG.read_text().splitlines()
    picked = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
            picked[node.name] = node
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpNGramEmbedding":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in WANT_METHODS:
                    picked[sub.name] = sub
    missing = (WANT_FUNCS | WANT_METHODS) - set(picked)
    if missing:
        raise SystemExit(f"could not lift {missing} from {LANG}")
    # module-level constants the lifted code reads
    for name in ("_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1"):
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
                exec(compile(ast.Module([node], []), str(LANG), "exec"), ns)
    for name, node in picked.items():
        exec(compile(ast.Module([node], []), str(LANG), "exec"), ns)
    lifted_lines = {name: node.lineno for name, node in picked.items()}
    return ns, lifted_lines, src


class Ref:
    """The vendored methods, bound to an object carrying only the attributes they read."""

    def __init__(self, ns, idx: NGramIndexer):
        self.ngram_size = idx.ngram_size
        self.heads_per_ngram = idx.heads_per_ngram
        self.eos_token_id = idx.eos_token_id
        self.ngram_heads_vocab_sizes = mx.array(idx.sizes.tolist(), dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(idx.offsets.tolist(), dtype=mx.int64)
        self.layer_multipliers = ns["_build_layer_multipliers"](
            idx.vocab_size, idx.ngram_size, 0, 1234
        )
        self._shift_right_ignore_eos = types.MethodType(ns["_shift_right_ignore_eos"], self)
        self._ngram_indices = types.MethodType(ns["_ngram_indices"], self)


def main():
    ns, lifted, _ = lift_reference()
    idx = NGramIndexer()
    ref = Ref(ns, idx)

    out = {
        "lifted_from": {k: f"{LANG.name}:{v}" for k, v in lifted.items()},
        "multipliers_match": ref.layer_multipliers.tolist() == idx.multipliers.tolist(),
        "multipliers": [str(v) for v in idx.multipliers.tolist()],
        "head_vocab_sizes": idx.sizes.tolist(),
        "total_vocab_size": idx.total_vocab_size,
        "padded_vocab_size": idx.padded_vocab_size,
        "cases": [],
    }
    manifest = json.loads(
        Path(
            "~/Engineering/MLX/_models/Jundot/"
            "Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed/manifest.json"
        ).read_text()
    )
    out["manifest_rows"] = manifest["layers"]["1"]["rows"]
    out["padded_matches_manifest"] = idx.padded_vocab_size == manifest["layers"]["1"]["rows"]

    rng = np.random.default_rng(0)
    ok = True
    for name, tokens in [
        ("random_2048", rng.integers(0, 248320, size=2048, dtype=np.int64)),
        ("random_with_eos", np.where(rng.random(2048) < 0.01, 248044,
                                     rng.integers(0, 248320, size=2048)).astype(np.int64)),
        ("repetitive", np.tile(rng.integers(0, 5000, size=37, dtype=np.int64), 40)),
        ("low_ids", rng.integers(0, 64, size=512, dtype=np.int64)),
        ("all_eos", np.full(64, 248044, dtype=np.int64)),
    ]:
        hist = np.concatenate([np.full(2, 248044, dtype=np.int64), tokens])
        mine = idx.ngram_indices(hist, tokens.size)
        theirs = np.asarray(ref._ngram_indices(mx.array(hist.tolist(), dtype=mx.int64)[None],
                                               tokens.size))[0]
        same = bool(np.array_equal(mine, theirs))
        ok &= same
        out["cases"].append({
            "case": name, "shape": list(mine.shape), "identical": same,
            "min_row": int(mine.min()), "max_row": int(mine.max()),
            "max_abs_diff": int(np.abs(mine - theirs).max()),
        })

    # chunked streaming must equal one-shot over the same token stream
    tokens = rng.integers(0, 248320, size=5000, dtype=np.int64)
    chunked = np.concatenate([ids for _, ids in idx.stream(tokens, chunk=2048)])
    whole = idx.ngram_indices(
        np.concatenate([np.full(2, 248044, dtype=np.int64), tokens]), tokens.size)
    out["chunked_equals_whole"] = bool(np.array_equal(chunked, whole))
    ok &= out["chunked_equals_whole"]

    # and chunked streaming must equal the vendored code run chunk by chunk
    ctx = mx.full((1, 2), 248044, dtype=mx.int64)
    ref_chunks = []
    for start in range(0, tokens.size, 2048):
        piece = mx.array(tokens[start:start + 2048].tolist(), dtype=mx.int64)[None]
        hist = mx.concatenate([ctx, piece], axis=-1)
        ref_chunks.append(np.asarray(ref._ngram_indices(hist, piece.shape[1]))[0])
        ctx = hist[:, -2:]
    out["chunked_matches_vendor"] = bool(np.array_equal(chunked, np.concatenate(ref_chunks)))
    ok &= out["chunked_matches_vendor"]
    out["all_ok"] = bool(ok)

    print(json.dumps(out, indent=1))
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
