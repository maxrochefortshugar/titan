#!/usr/bin/env python3
"""Concurrency sweep at a chosen context length.

Copy of ~/inference-server/staging/concurrency_sweep.py with --context: each
stream's prompt is padded with distinct filler to roughly that many tokens, so
the batch decodes against a long KV cache instead of a few hundred tokens. The
original is left untouched.

Env: OMLX_URL (default 8084), OMLX_KEY_FILE, BENCH_MODEL.
Usage: concurrency_sweep.py --tag X --streams 1 2 4 --context 65000 [--tokens 400]
"""
import argparse
import json
import os
import random
import statistics as st
import threading
import time
import urllib.request

URL = os.environ.get("OMLX_URL", "http://127.0.0.1:8084") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-oQ4e-mtp")
KEY = open(
    os.path.expanduser(
        os.environ.get("OMLX_KEY_FILE", "~/inference-server/staging/omlx-home/api_key.txt")
    )
).read().strip()
TOPICS = [
    "B-trees vs LSM trees",
    "the Raft consensus protocol",
    "how garbage collectors handle cycles",
    "TCP vs QUIC",
    "Rust ownership rules",
    "how a JIT compiler works",
    "Paxos in practice",
    "memory-mapped IO pitfalls",
]
WORDS = (
    "buffer page latch commit replica shard vector cache tenant quorum lease "
    "epoch segment compaction manifest checkpoint journal snapshot tombstone"
).split()


def filler(tokens, seed):
    """Roughly ``tokens`` tokens of distinct prose, so no two streams share a prefix."""
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(int(tokens * 0.9)))


def call(i, max_tokens, out, context):
    topic = TOPICS[i % len(TOPICS)]
    prompt = f"Explain {topic} in detail with examples (variant {i})."
    if context:
        prompt = (
            "Reference notes, stream "
            f"{i}:\n{filler(context, 1000 + i)}\n\nIgnore the notes above. {prompt}"
        )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    start = time.time()
    payload = json.load(urllib.request.urlopen(request, timeout=3600))
    usage = payload["usage"]
    out[i] = (usage["completion_tokens"], time.time() - start, usage.get("prompt_tokens", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--context", type=int, default=0,
                        help="approximate prompt tokens per stream (0 = short prompts)")
    args = parser.parse_args()

    warm = {}
    call(0, 4, warm, 0)
    for streams in args.streams:
        out = {}
        threads = [
            threading.Thread(target=call, args=(i, args.tokens, out, args.context))
            for i in range(streams)
        ]
        start = time.time()
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.time() - start
        total = sum(v[0] for v in out.values())
        per = [v[0] / v[1] for v in out.values()]
        prompt_tokens = st.median([v[2] for v in out.values()])
        print(
            f"[{args.tag}] streams={streams} ctx~{prompt_tokens:.0f}: {total} tok in "
            f"{wall:.1f}s -> aggregate {total / wall:.1f} tok/s, per-stream median "
            f"{st.median(per):.1f} tok/s",
            flush=True,
        )
        time.sleep(20)
    print(f"[{args.tag}] SWEEPDONE")


if __name__ == "__main__":
    main()
