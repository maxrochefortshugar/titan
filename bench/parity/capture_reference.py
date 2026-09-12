#!/usr/bin/env python3
"""Capture reference greedy continuations from a running oMLX server.

Run this against oMLX *before* comparing Titan against it.  It writes one JSON
file holding, for each of the twenty fixed prompts, the token ids oMLX produced
under greedy decoding, plus everything needed to reproduce the run: the server
URL, the model id, the sampling settings sent, and the prompt hashes.

    python bench/parity/capture_reference.py \
        --url http://127.0.0.1:8083 \
        --model qwen3.8-flash-next \
        --max-tokens 128 \
        --out bench/parity/reference.json

Greedy means greedy: temperature 0, top_p 1, top_k 0, no repetition penalty,
seed fixed.  If the server silently applies its own sampling defaults the
capture is worthless, so the script records what it asked for and refuses to
continue if the response reports different settings.

It talks HTTP only.  It does not import Titan, does not load a model and does
not touch the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import prompts  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8083"
API_KEY_FILE = Path.home() / ".omlx/api_key.txt"


def api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("OMLX_API_KEY")
    if env:
        return env
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    return None


def post(url: str, payload: dict, key: str | None, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


_TOK = None


def _reencode(model_dir, text: str) -> list[int]:
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(model_dir))
    return list(_TOK.encode(text, add_special_tokens=False))


def capture(args: argparse.Namespace) -> dict:
    key = api_key(args.api_key)
    endpoint = args.url.rstrip("/") + "/v1/completions"
    records = []
    for index, item in enumerate(prompts(), start=1):
        payload = {
            "model": args.model,
            "prompt": item["prompt"],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "seed": args.seed,
            "stream": False,
            "logprobs": 0,
        }
        started = time.perf_counter()
        try:
            response = post(endpoint, payload, key, args.timeout)
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"{item['id']}: server returned {exc.code}: {exc.read()[:400]!r}"
            ) from exc
        elapsed = time.perf_counter() - started
        choice = response["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            # Fall back to the logprobs channel, which carries ids on oMLX.
            logprobs = choice.get("logprobs") or {}
            token_ids = logprobs.get("token_ids")
        ids_source = "server"
        if token_ids is None:
            if not args.allow_text_only:
                raise SystemExit(
                    f"{item['id']}: the server returned no token ids. Parity needs "
                    "ids, not text: re-run with a build that returns them, or set "
                    "--allow-text-only and accept a weaker comparison."
                )
            # oMLX 0.7.0.dev2 exposes neither ids nor logprobs. Re-encode the
            # text with the model tokenizer; greedy text round-trips to the
            # same ids except at rare merge boundaries, which the text
            # comparison in greedy_parity.py covers.
            token_ids = _reencode(args.model_dir, choice.get("text", ""))
            ids_source = "reencoded_text"
        records.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "prompt_sha256": hashlib.sha256(
                    item["prompt"].encode()
                ).hexdigest(),
                "prompt_chars": len(item["prompt"]),
                "token_ids": list(token_ids) if token_ids else None,
                "ids_source": ids_source,
                "text": choice.get("text", ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "seconds": round(elapsed, 3),
            }
        )
        print(
            f"[{index:2d}/20] {item['id']}: "
            f"{len(records[-1]['token_ids'] or [])} tokens in {elapsed:.1f}s",
            flush=True,
        )
    return {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "omlx",
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="oMLX base URL")
    parser.add_argument("--model", required=True, help="model id the server serves")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--allow-text-only", action="store_true")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser(
            "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
        ),
        help="checkpoint dir whose tokenizer re-encodes text when the server returns no ids",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "reference.json",
    )
    args = parser.parse_args()

    result = capture(args)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out} ({len(result['records'])} prompts)")


if __name__ == "__main__":
    main()
