#!/usr/bin/env python3
"""One arm of the ROUND4 kernel bisect, against a running Titan on 8085.

The difference from ``mtp_ab.py`` is only the packaging: this runs every mode
the bisect needs inside a single server lifetime, so a kernel arm costs one
model load instead of four. The prompts, the client and the metrics view are
``mtp_ab``'s, imported rather than copied, so a ROUND4 number and a ROUND3
number are the same measurement.

    probe.py --arm refonly --modes short,long,lossless

Cold prefill is derived from the ``long`` run rather than measured separately.
The 64k request is a cold prompt with the prefix store off, so its wall time is
prefill plus decode and decode is the profiler's own cycle count times its own
cycle time. Subtracting gives the prefill seconds for the same 64,779 tokens
every arm sees. It carries the request's tokenisation with it, so read it as a
number to compare across arms rather than as the kernel's own prefill rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtp_ab import (  # noqa: E402
    LOSSLESS_PROMPT,
    PROMPTS,
    Client,
    cold_prompt,
    decode_view,
)

HERE = Path(__file__).resolve().parent


def run_short(client: Client, tokens: int) -> dict:
    """Four decode-heavy prompts at 600 tokens, the production-like shape."""
    before = client.cycles()
    rates = []
    per_prompt = {}
    completions = 0
    decode_wall = 0.0
    for name, prompt in PROMPTS.items():
        usage, elapsed, _ = client.complete(prompt, tokens)
        decode_only = max(0.05, elapsed - usage["prompt_tokens"] / 1400.0)
        rate = usage["completion_tokens"] / decode_only
        per_prompt[name] = round(rate, 1)
        rates.append(rate)
        completions += usage["completion_tokens"]
        decode_wall += decode_only
    window = max(1, client.cycles() - before)
    out = {
        "per_prompt_tok_s": per_prompt,
        "median_tok_s": round(statistics.median(rates), 1),
        "aggregate_tok_s": round(completions / decode_wall, 1),
        "completion_tokens": completions,
    }
    out.update(decode_view(client.metrics(window=window)))
    return out


def run_long(client: Client, words: int, tokens: int) -> dict:
    """One cold 64,779-token prompt, then 300 tokens of decode from it."""
    prompt = cold_prompt(words)
    before = client.cycles()
    usage, elapsed, _ = client.complete(prompt, tokens)
    window = max(1, client.cycles() - before)
    view = decode_view(client.metrics(window=window))
    decode_s = view["cycles"] * view["cycle_ms"] / 1000.0
    prefill_s = max(0.001, elapsed - decode_s)
    out = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "wall_s": round(elapsed, 1),
        "decode_s": round(decode_s, 2),
        "prefill_s": round(prefill_s, 2),
        "prefill_tok_s": round(usage["prompt_tokens"] / prefill_s, 1),
        "wall_tok_s": round(usage["completion_tokens"] / max(0.001, decode_s), 1),
    }
    out.update(view)
    return out


def run_lossless(client: Client, tokens: int, arm: str) -> dict:
    """The same prompt every arm answers. Bytes and their digest."""
    _usage, elapsed, text = client.complete(LOSSLESS_PROMPT, tokens)
    path = HERE / f"lossless-{arm}.txt"
    path.write_text(text)
    return {
        "lossless_md5": hashlib.md5(text.encode()).hexdigest(),
        "lossless_chars": len(text),
        "lossless_path": str(path),
        "wall_s": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--modes", default="short,long,lossless")
    parser.add_argument("--url", default=os.environ.get("TITAN_URL", "http://127.0.0.1:8085"))
    parser.add_argument("--key-file", default="~/.omlx/api_key.txt")
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--words", type=int, default=11000)
    parser.add_argument("--long-tokens", type=int, default=300)
    parser.add_argument("--lossless-tokens", type=int, default=150)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--note", default="")
    parser.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = parser.parse_args()

    client = Client(args.url, args.key_file)
    config = client.metrics().get("config", {})
    kernels = dict(config.get("kernels", {})) if isinstance(config, dict) else {}

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode == "short":
            result = run_short(client, args.tokens)
        elif mode == "long":
            result = run_long(client, args.words, args.long_tokens)
        elif mode == "lossless":
            result = run_lossless(client, args.lossless_tokens, args.arm)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
        result.update(
            arm=args.arm,
            mode=mode,
            repeat=args.repeat,
            note=args.note,
            kernels=kernels,
            at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        line = json.dumps(result)
        print(line, flush=True)
        with open(args.out, "a") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
