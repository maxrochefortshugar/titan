#!/usr/bin/env python3
"""One arm of the MTP drafter A/B, against a running Titan on 8085.

Three modes, and the third is the one that makes the other two mean anything:

    short    four decode-heavy prompts, 600 tokens each, greedy
    long     one cold 64k prompt, then 300 tokens of decode from it
    lossless one fixed prompt, printing the completion so arms can be diffed

Rates come from two places on purpose. The wall rate is what a client sees,
prefill included, and it is the number that pays the bills. The profiler rate
is committed tokens over the summed wall time of the decode cycles themselves,
which is the number a drafter can actually move; a change that shows up in one
and not the other is a change in prefill or in scheduling, not in decode.

Usage:
    mtp_ab.py --arm b_chain_d3 --mode short
    mtp_ab.py --arm b_chain_d3 --mode long --words 11000
    mtp_ab.py --arm b_chain_d3 --mode lossless > lossless.b.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8085"
DEFAULT_KEY = "~/.omlx/api_key.txt"
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp:no-think"

CODE = "\n".join(
    f"def step_{i}(x, cfg):\n"
    f"    # stage {i}: scale then offset\n"
    f"    y = x * cfg.get('scale_{i}', {i % 7 + 1})\n"
    f"    return y + cfg.get('offset_{i}', {i % 13})"
    for i in range(40)
)

PROMPTS = {
    "edit": (
        f"Here is a Python file:\n```python\n{CODE}\n```\n"
        "Rewrite the whole file so every function has a docstring and type "
        "hints. Output the complete file, nothing else."
    ),
    "code": (
        "Write a complete Python module implementing an LRU cache with TTL "
        "expiry, a background sweeper thread, thread safety, and a small test "
        "suite using unittest. Output code only."
    ),
    "prose": (
        "Explain in detail how TCP congestion control evolved from Tahoe to "
        "BBR, covering the motivation for each algorithm and the failure modes "
        "it addressed."
    ),
    "json": (
        "Produce a JSON array of 60 objects describing fictional cities, each "
        "with fields name, country, population, founded, mayor, and three "
        "landmarks. Output JSON only."
    ),
}

LOSSLESS_PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted "
    "list, then explain the time complexity in one paragraph."
)

WORDS = (
    "alpha beta gamma delta kernel tensor buffer stride lambda socket thread "
    "mutex cache page index router expert gather scatter norm residual"
).split()


def cold_prompt(n_words: int, seed: int = 20260912) -> str:
    """A prompt no prefix cache can have seen. Seeded, so arms share one."""
    rng = random.Random(seed)
    body = " ".join(
        f"{rng.choice(WORDS)}{rng.randint(0, 99999)}" for _ in range(n_words)
    )
    return (
        body
        + "\n\nIgnore the token soup above. Explain in detail how TCP "
        "congestion control evolved from Tahoe to BBR, covering the motivation "
        "for each algorithm and the failure modes it addressed."
    )


class Client:
    def __init__(self, url: str, key_file: str) -> None:
        self.url = url.rstrip("/")
        self.key = open(os.path.expanduser(key_file)).read().strip()

    def _request(self, path: str, body: dict | None = None, timeout: float = 2400.0):
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def complete(self, text: str, max_tokens: int) -> tuple[dict, float, str]:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.time()
        payload = self._request("/v1/chat/completions", body)
        elapsed = time.time() - started
        content = payload["choices"][0]["message"]["content"] or ""
        return payload["usage"], elapsed, content

    def metrics(self, window: int = 0) -> dict:
        path = "/metrics" if not window else f"/metrics?window={int(window)}"
        return self._request(path, timeout=60.0)

    def cycles(self) -> int:
        """Cumulative decode cycles, so a run can name its own window."""
        return int(self.metrics().get("counters", {}).get("cycles", 0))


def decode_view(metrics: dict) -> dict:
    decode = dict(metrics.get("decode", {}))
    stages = dict(metrics.get("stages", {}))
    return {
        "cycles": decode.get("cycles", 0),
        "profiler_tok_s": round(decode.get("tokens_per_second", 0.0), 1),
        "accepted_per_cycle": round(decode.get("mean_accepted_per_cycle", 0.0), 3),
        "acceptance_rate": round(decode.get("acceptance_rate", 0.0), 3),
        "mean_rows": round(decode.get("mean_rows", 0.0), 2),
        "cycle_ms": round(stages.get("wall_ms", 0.0), 2),
        "verify_ms": round(stages.get("verify_ms", 0.0), 2),
        "draft_ms": round(stages.get("draft_ms", 0.0), 2),
        "other_ms": round(stages.get("other_ms", 0.0), 2),
        "host_syncs_per_cycle": round(decode.get("host_syncs_per_cycle", 0.0), 3),
        "position_acceptance": [
            round(v, 3) for v in metrics.get("position_acceptance", [])
        ],
        "accepted_histogram": metrics.get("accepted_histogram", {}),
    }


def run_short(client: Client, tokens: int) -> dict:
    rates = []
    completions = 0
    decode_wall = 0.0
    per_prompt = {}
    for name, prompt in PROMPTS.items():
        usage, elapsed, _ = client.complete(prompt, tokens)
        # The prefill share of a short prompt, at the measured cold rate. Small
        # and roughly constant across arms, which is why it can be subtracted
        # rather than measured per call.
        decode_only = max(0.05, elapsed - usage["prompt_tokens"] / 1400.0)
        rate = usage["completion_tokens"] / decode_only
        per_prompt[name] = round(rate, 1)
        rates.append(rate)
        completions += usage["completion_tokens"]
        decode_wall += decode_only
    return {
        "per_prompt_tok_s": per_prompt,
        "median_tok_s": round(statistics.median(rates), 1),
        "aggregate_tok_s": round(completions / decode_wall, 1),
        "completion_tokens": completions,
    }


def run_long(client: Client, words: int, tokens: int) -> dict:
    prompt = cold_prompt(words)
    usage, elapsed, _ = client.complete(prompt, tokens)
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "wall_s": round(elapsed, 1),
    }


# Measured on this checkpoint: 11,000 words of seeded soup tokenises to 64,779
# tokens. The sweep asks for contexts, not for words, so it converts here and
# reports the prompt length the server actually saw.
TOKENS_PER_WORD = 64779 / 11000


def run_sweep(client: Client, contexts: list[int], tokens: int) -> list[dict]:
    """Accepted tokens a cycle against context length, in one process.

    One process on purpose: a model load per point would put a cold cache and a
    cold set of kernels between the points, and the question here is whether
    acceptance is a slope or a step, which is a question about the differences
    rather than about any one number. Each point reads its own cycles back
    through the metrics window, so the points do not average into each other.
    """
    rows = []
    for index, context in enumerate(contexts):
        # A different seed per point, so no point is a prefix of another, the
        # prefix cache cannot serve the second half of the sweep, and a context
        # repeated in the list is a genuine second measurement rather than a
        # cache hit.
        prompt = cold_prompt(
            max(1, round(context / TOKENS_PER_WORD)),
            seed=20260912 + context + 1000 * index,
        )
        before = client.cycles()
        usage, elapsed, _ = client.complete(prompt, tokens)
        after = client.cycles()
        row = {
            "context_target": context,
            "point": index,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "wall_s": round(elapsed, 1),
            "cycles_measured": after - before,
        }
        row.update(decode_view(client.metrics(window=max(1, after - before))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--mode", choices=("short", "long", "lossless", "sweep"), default="short"
    )
    parser.add_argument("--url", default=os.environ.get("TITAN_URL", DEFAULT_URL))
    parser.add_argument("--key-file", default=DEFAULT_KEY)
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--words", type=int, default=11000)
    parser.add_argument("--lossless-tokens", type=int, default=400)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--contexts",
        default="8192,16384,32768,49152,63488,64779,67584",
        help="comma-separated target context lengths for --mode sweep",
    )
    parser.add_argument("--sweep-tokens", type=int, default=200)
    args = parser.parse_args()

    client = Client(args.url, args.key_file)
    if args.mode == "lossless":
        _usage, _elapsed, text = client.complete(LOSSLESS_PROMPT, args.lossless_tokens)
        print(text)
        return 0

    if args.mode == "sweep":
        contexts = [int(v) for v in args.contexts.split(",") if v.strip()]
        rows = run_sweep(client, contexts, args.sweep_tokens)
        if args.out:
            with open(args.out, "a") as handle:
                for row in rows:
                    row["arm"] = args.arm
                    row["mode"] = "sweep"
                    handle.write(json.dumps(row) + "\n")
        return 0

    if args.mode == "short":
        before = client.cycles()
        result = run_short(client, args.tokens)
        window = max(1, client.cycles() - before)
    else:
        before = client.cycles()
        result = run_long(client, args.words, 300)
        window = max(1, client.cycles() - before)
    result["arm"] = args.arm
    result["mode"] = args.mode
    result.update(decode_view(client.metrics(window=window)))
    line = json.dumps(result)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
