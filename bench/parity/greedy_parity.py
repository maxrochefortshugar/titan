#!/usr/bin/env python3
"""Compare Titan's greedy continuations against captured oMLX references.

    # once, against a running oMLX:
    python bench/parity/capture_reference.py --model <id> --out reference-prod.json
    # then, with oMLX stopped and the GPU free:
    python bench/parity/greedy_parity.py --reference prod stock

This loads the model through Titan's own loader (``titan.adapters.mlx.loader``),
generates one greedy continuation per prompt, and reports, per prompt and per
reference, the length of the agreeing prefix and the first position that
differs. The generation runs once even when two references are given, because
the model is the expensive part and the comparison is not.

Read the result the way the overlay reports ask you to. Divergence is expected
and is not by itself a failure: ``engine/patches/ple-fix/REPORT.md`` showed with
two controls that on this model a *more accurate* norm moves the 2048-token
output by roughly the same amount as an approximate one, because 36 recurrent
GDN layers amplify anything. What this harness measures is how far the two
implementations track each other, and where they part. A short-prompt divergence
inside the first few tokens is a bug; a long-prompt divergence after a hundred
is the model's sensitivity, and the number to watch is whether it moves when you
change one op.

Two references ship, and they answer different questions. ``reference-prod``
came from the production oMLX with its own kernels; ``reference-stock`` came
from the stock stack. Agreement with prod is the deployment question, agreement
with stock is the correctness question, and a run that tracks one and not the
other tells you which layer moved.

``--kernels reference`` runs Titan with every fast path off, which is the
control arm of any kernel A/B. It publishes a reference-only registry before
the model is loaded, which is the same mechanism ``kernels.reference_only`` in
the config file uses, so the arm this bench calls the control is the arm the
server would run under that setting. It was an environment variable until
ROUND4; the trouble with that was that the adapter then had two policies, the
registry's and the environment's, and a bisect with two policies cannot say
which one produced a number.

The token ids in a reference may be re-encoded from its text rather than
reported by the server (the record's ``ids_source`` says which). When they are,
the harness re-encodes the same text with Titan's tokenizer and says whether it
agrees, so an id-level difference that is really a tokenizer difference cannot
be mistaken for a model difference.

Reference-only mode (``--reference-only``) replays the reference files and
prints their shape without loading anything, so they can be checked on any
machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import prompts  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = (
    Path.home() / "Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
)

REFERENCE_FILES = {
    "prod": HERE / "reference-prod.json",
    "stock": HERE / "reference-stock.json",
}


def resolve_reference(name: str) -> Path:
    """``prod``, ``stock`` or a path. A shorthand that is not there is fatal."""
    if name in REFERENCE_FILES:
        path = REFERENCE_FILES[name]
        if not path.exists():
            raise SystemExit(
                f"no {name} reference at {path}. Capture one first:\n"
                "  python bench/parity/capture_reference.py --model <id> "
                f"--out {path.name}"
            )
        return path
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"no reference at {path}")
    return path


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
    data["path"] = str(path)
    data["name"] = path.stem.replace("reference-", "")
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
    """Titan's tokenizer adapter, with transformers as a stated fallback.

    The fallback exists so the harness runs on a checkout where the adapter is
    still someone's open branch, and it says so out loud rather than silently
    measuring parity with a different tokenizer than the engine uses.
    """
    try:
        from titan.adapters.mlx.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(model_dir)
        print(
            f"tokenizer: titan.adapters.mlx, {tokenizer.vocab_size} ids, "
            f"eos {sorted(tokenizer.eos_token_ids)}"
        )
        return tokenizer
    except Exception as exc:  # noqa: BLE001 - the fallback is the point
        print(f"tokenizer: titan adapter unavailable ({exc}); using transformers")

    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(model_dir))
    config = json.loads((model_dir / "config.json").read_text())
    eos = config.get("eos_token_id") or hf.eos_token_id
    eos_ids = frozenset(eos if isinstance(eos, list) else [eos])

    class _Adapter:
        eos_token_ids = eos_ids
        vocab_size = len(hf)

        @staticmethod
        def encode(text: str) -> list[int]:
            return hf.encode(text, add_special_tokens=False)

        @staticmethod
        def decode(ids) -> str:
            return hf.decode(list(ids), skip_special_tokens=False)

    return _Adapter()


def strip_eos(ids: list[int], eos) -> list[int]:
    """Drop the terminator before decoding.

    The generation loop keeps the eos token because the id comparison wants it.
    The reference text never contains it, so the text comparison must not see
    it either.
    """
    return ids[:-1] if ids and ids[-1] in eos else ids


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


def tokenizer_agreement(tokenizer, record: dict) -> bool | None:
    """Do we re-encode the reference text to the ids it carries?

    ``None`` when the question does not arise, either because the server
    reported the ids itself or because there is no text to re-encode.
    """
    if record.get("ids_source") != "reencoded_text":
        return None
    text, ids = record.get("text"), record.get("token_ids")
    if not text or not ids:
        return None
    return tokenizer.encode(text) == list(ids)


def summarise(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: nothing compared")
        return
    identical = sum(1 for r in rows if r["identical"])
    median = sorted(r["agree_prefix"] for r in rows)[len(rows) // 2]
    mismatched = sum(1 for r in rows if r.get("tokenizer_agrees") is False)
    print(f"\n{name}: {identical}/{len(rows)} identical; median agreeing prefix {median}")
    if mismatched:
        print(
            f"{name}: {mismatched} record(s) re-encode to different ids under "
            "Titan's tokenizer; their id comparison is not a model comparison"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--reference",
        nargs="+",
        default=["prod"],
        metavar="NAME|PATH",
        help="prod, stock, or a path. More than one compares against each.",
    )
    parser.add_argument(
        "--kernels",
        choices=("reference", "fast"),
        default="fast",
        help="reference turns every fast path off (kernels.reference_only)",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--only", default=None, help="substring filter on prompt id")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--no-mtp", action="store_true",
                        help="load without the MTP head (target model only)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Before the checkpoint is loaded, so the adapter's first lookup finds it.
    if args.kernels == "reference":
        from titan.adapters.mlx import kernels as adapter_kernels
        from titan.kernels.registry import reference_only

        reference_only()
        adapter_kernels.reset()

    references = [load_reference(resolve_reference(n)) for n in args.reference]
    max_tokens = args.max_tokens or min(r["max_tokens"] for r in references)

    if args.reference_only:
        for reference in references:
            print(f"\n{reference['name']}  {reference['path']}")
            for record in reference["records"]:
                if args.only and args.only not in record["id"]:
                    continue
                ids = record["token_ids"] or []
                print(
                    f"  {record['id']:<10} {len(ids):>4} tokens  "
                    f"{record['finish_reason']:<8} ids={record.get('ids_source')}"
                )
        return

    from titan.adapters.mlx.loader import load_model
    from titan.adapters.mlx.model import TitanQwenFlashNext

    print(f"loading {args.model_dir} (kernels={args.kernels}) ...", flush=True)
    loaded, plan = load_model(args.model_dir, mtp_enabled=not args.no_mtp)
    model = TitanQwenFlashNext(loaded)
    tokenizer = _tokenizer(args.model_dir)

    corpus = {item["id"]: item["prompt"] for item in prompts()}
    wanted = [
        pid
        for pid in dict.fromkeys(
            record["id"] for reference in references for record in reference["records"]
        )
        if not args.only or args.only in pid
    ]

    produced: dict[str, dict] = {}
    for pid in wanted:
        ids, seconds = greedy(model, tokenizer, corpus[pid], max_tokens)
        text = tokenizer.decode(strip_eos(ids, tokenizer.eos_token_ids))
        produced[pid] = {"ids": ids, "text": text, "seconds": round(seconds, 2)}
        print(f"{pid:<10} {len(ids):>4} tokens in {seconds:6.2f}s", flush=True)

    results: dict[str, list[dict]] = {}
    for reference in references:
        name = reference["name"]
        print(f"\n-- {name} ({reference['path']})")
        rows = []
        for record in reference["records"]:
            got = produced.get(record["id"])
            if got is None:
                continue
            row = {
                "id": record["id"],
                "kind": record["kind"],
                "seconds": got["seconds"],
                "ids_source": record.get("ids_source"),
                "tokenizer_agrees": tokenizer_agreement(tokenizer, record),
            }
            row.update(compare(record["token_ids"] or [], got["ids"]))
            row.update(compare_text(record.get("text") or "", got["text"]))
            row["produced"] = got["ids"]
            rows.append(row)
            flag = "" if row["tokenizer_agrees"] is not False else " tok!"
            print(
                f"{row['id']:<10} agree {row['agree_prefix']:>4}/"
                f"{row['reference_len']:<4} text {row['text_agree_chars']:>5}/"
                f"{row['text_reference_chars']:<5} "
                f"{'exact' if row['text_identical'] else ''}{flag}",
                flush=True,
            )
        results[name] = rows
        summarise(name, rows)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "model_dir": str(args.model_dir),
                    "kernels": args.kernels,
                    "max_tokens": max_tokens,
                    "references": {r["name"]: r["path"] for r in references},
                    "rows": results,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
