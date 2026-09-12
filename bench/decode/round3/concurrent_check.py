#!/usr/bin/env python3
"""Two streams at once, past the QSA crossover: does batched decode work?

Every arm this round wired for a ``BatchQSAKVCache`` is unreachable from a
single-stream benchmark, because one sequence never joins a batch. This is the
smallest thing that reaches it: two requests long enough to be past the
gathered arm's crossover, submitted together, decoded in lockstep.

It is also the regression test for what the "arm switched off" test found. The
dense batched path raised a ``TypeError`` out of
``Qwen4ExpQSAIndexer.from_projected`` -- a batched cache's offset is one logical
length per row and ``mx.arange`` does not take an array -- so before this round
two concurrent sequences through a QSA layer did not run slowly, they failed.
Run this against a server from the "before" tree and it reports the failure;
against one from the "after" tree both streams complete.

    python bench/decode/round3/concurrent_check.py --context 12000 --tokens 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://127.0.0.1:8085/v1/chat/completions"
KEY = (Path.home() / ".omlx" / "api_key.txt").read_text().strip()
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp:no-think"


def prompt(words: int, seed: int) -> str:
    """A wall of seeded words, then one instruction. Two different seeds so the
    prefix cache cannot serve the second request from the first."""
    rng = random.Random(seed)
    alphabet = [f"w{n}" for n in range(4096)]
    body = " ".join(rng.choice(alphabet) for _ in range(words))
    return (
        f"{body}\n\nIgnore the list above. In two short paragraphs, explain how "
        "TCP congestion control reacts to a single dropped packet."
    )


def ask(words: int, tokens: int, seed: int) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(words, seed)}],
        "max_tokens": tokens,
        "temperature": 0.0,
    }).encode()
    request = urllib.request.Request(
        URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        return {"seed": seed, "ok": False, "error": error.read().decode()[:400]}
    except Exception as error:  # noqa: BLE001 - a hung stream is a result
        return {"seed": seed, "ok": False, "error": repr(error)[:400]}
    usage = body.get("usage", {})
    return {
        "seed": seed,
        "ok": True,
        "seconds": round(time.perf_counter() - start, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text": body["choices"][0]["message"]["content"][:120],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=12000,
                        help="approximate prompt words per stream")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--streams", type=int, default=2)
    args = parser.parse_args()

    with ThreadPoolExecutor(max_workers=args.streams) as pool:
        futures = [
            pool.submit(ask, args.context, args.tokens, 1000 + n)
            for n in range(args.streams)
        ]
        results = [future.result() for future in futures]

    for result in results:
        print(json.dumps(result))
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} streams completed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
