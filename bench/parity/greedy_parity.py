#!/usr/bin/env python3
"""Compare Titan's greedy continuations against captured oMLX references.

    # once, against a running oMLX:
    python bench/parity/capture_reference.py --model <id> --out reference.json
    # then, with oMLX stopped and the GPU free:
    python bench/parity/greedy_parity.py --model-dir <checkpoint> \
        --reference bench/parity/reference.json

This loads the model through Titan's own loader (``titan.adapters.mlx.loader``),
generates greedy continuations for the same twenty prompts, and reports, per
prompt, the length of the agreeing prefix and the first position that differs.

Read the result the way the overlay reports ask you to.  Divergence is expected
and is not by itself a failure: ``engine/patches/ple-fix/REPORT.md`` showed with
two controls that on this model a *more accurate* norm moves the 2048-token
output by roughly the same amount as an approximate one, because 36 recurrent
GDN layers amplify anything. What this harness measures is how far the two
implementations track each other, and where they part. A short-prompt divergence
inside the first few tokens is a bug; a long-prompt divergence after a hundred
is the model's sensitivity, and the number to watch is whether it moves when you
change one op.

Reference-only mode (``--reference-only``) replays the reference file and prints
its shape without loading anything, so the file can be checked on any machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import prompts  # noqa: E402

DEFAULT_MODEL_DIR = (
    Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
)


def load_reference(path: Path) -> dict:
    data = json.loads(path.read_text())
    corpus = {item["id"]: item for item in prompts()}
    for record in data["records"]:
        item = corpus.get(record["id"])
        if item is None:
            raise SystemExit(f"reference has an unknown prompt id: {record['id']}")
        digest = hashlib.sha256(item["prompt"].encode()).hexdigest()
        if digest != record["prompt_sha256"]:
            raise SystemExit(
                f"{record['id']}: prompts.py has changed since the reference was "
                "captured; recapture rather than comparing different prompts"
            )
    return data


def greedy(model, tokenizer, prompt: str, max_tokens: int) -> tuple[list[int], float]:
    """Greedy continuation through Titan's forward API."""
    import mlx.core as mx

    from titan.adapters.mlx.state import ModelState

    ids = tokenizer.encode(prompt)
    state = ModelState.new(model.model)
    started = time.perf_counter()
    result = model.prefill(ids, state, want_logits=True)
    token = int(mx.argmax(result.logits[0, -1]).item())
    out = [token]
    eos = tokenizer.eos_token_ids
    for _ in range(max_tokens - 1):
        if token in eos:
            break
        logits = model.decode([token], state)
        token = int(mx.argmax(logits[0]).item())
        out.append(token)
    return out, time.perf_counter() - started


def _tokenizer(model_dir: Path):
    """Titan's tokenizer adapter when it exists, transformers otherwise.

    The tokenizer is another workstream's module; this harness only needs
    ``encode`` and the eos id set, so it does not wait for it.
    """
    try:
        from titan.adapters.mlx.tokenizer import load_tokenizer  # type: ignore

        return load_tokenizer(model_dir)
    except Exception:
        pass

    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(model_dir))
    config = json.loads((model_dir / "config.json").read_text())
    eos = config.get("eos_token_id") or hf.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    class _Adapter:
        eos_token_ids = eos_ids

        @staticmethod
        def encode(text: str) -> list[int]:
            return hf.encode(text, add_special_tokens=False)

    return _Adapter()


def compare_text(reference: str, produced: str) -> dict:
    n = 0
    for a, b in zip(reference, produced):
        if a != b:
            break
        n += 1
    return {
        "text_agree_chars": n,
        "text_reference_chars": len(reference),
        "text_identical": reference == produced,
    }


def compare(reference: list[int], produced: list[int]) -> dict:
    n = 0
    for a, b in zip(reference, produced):
        if a != b:
            break
        n += 1
    return {
        "agree_prefix": n,
        "reference_len": len(reference),
        "produced_len": len(produced),
        "identical": n == len(reference) == len(produced),
        "first_divergence": None if n == min(len(reference), len(produced)) else n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parent / "reference.json",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--only", default=None, help="substring filter on prompt id")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--no-mtp", action="store_true",
                        help="load without the MTP head (target model only)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.reference.exists():
        raise SystemExit(
            f"no reference at {args.reference}. Capture one first:\n"
            "  python bench/parity/capture_reference.py --model <id>"
        )
    reference = load_reference(args.reference)
    max_tokens = args.max_tokens or reference["max_tokens"]

    if args.reference_only:
        for record in reference["records"]:
            ids = record["token_ids"] or []
            print(f"{record['id']}: {len(ids)} tokens, {record['finish_reason']}")
        return

    from titan.adapters.mlx.loader import load_model
    from titan.adapters.mlx.model import TitanQwenFlashNext

    print(f"loading {args.model_dir} ...", flush=True)
    loaded, plan = load_model(args.model_dir, mtp_enabled=not args.no_mtp)
    model = TitanQwenFlashNext(loaded)
    tokenizer = _tokenizer(args.model_dir)

    rows = []
    for record in reference["records"]:
        if args.only and args.only not in record["id"]:
            continue
        prompt = next(p["prompt"] for p in prompts() if p["id"] == record["id"])
        produced, seconds = greedy(model, tokenizer, prompt, max_tokens)
        row = {"id": record["id"], "kind": record["kind"], "seconds": round(seconds, 2)}
        row.update(compare(record["token_ids"] or [], produced))
        row.update(compare_text(record.get("text") or "", tokenizer.decode(produced)))
        row["produced"] = produced
        rows.append(row)
        print(
            f"{row['id']:<10} agree {row['agree_prefix']:>4}/"
            f"{row['reference_len']:<4} text {row['text_agree_chars']:>5}/"
            f"{row['text_reference_chars']:<5} {'exact' if row['text_identical'] else ''}",
            flush=True,
        )

    identical = sum(1 for r in rows if r["identical"])
    print(
        f"\n{identical}/{len(rows)} identical; "
        f"median agreeing prefix "
        f"{sorted(r['agree_prefix'] for r in rows)[len(rows) // 2] if rows else 0}"
    )
    if args.out:
        args.out.write_text(json.dumps({"reference": str(args.reference),
                                        "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
