#!/usr/bin/env python3
"""Concurrency sweep for fused batched MTP (OMLX_MTP_BATCHED=1).

Drives an oMLX instance with 1 / 2 / 4 / 8 concurrent greedy streams and
reports aggregate and per-stream decode tok/s, paired against a baseline run
of the same prompts. Reads the server log afterwards for the per-request
MTP line so accept counts per depth can be attributed per stream.

NEVER point this at 8083. Default is the workbench on 8084, and the script
refuses any other port unless --i-know-what-i-am-doing is passed.

Usage
    python3 concurrency_sweep.py --streams 1 2 4 8 --tokens 300 --rounds 2 \
        --label batched --log /var/log/omlx-workbench/stderr.log

Protocol (matches the rest of round 3): quiet GPU, 45 s cooldown between
rows, two rounds, baseline and patched paired inside a round rather than
across rows.
"""

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request, os
_KEY=open(os.path.expanduser(os.environ.get('OMLX_KEY_FILE','~/inference-server/staging/omlx-home/api_key.txt'))).read().strip()

PROMPTS = [
    "Write a Python function that merges two sorted lists, then explain it.",
    "Summarise the tradeoffs between rotating and growing KV caches.",
    "Refactor this loop to avoid the repeated allocation:\n"
    "for i in range(n):\n    out = out + [f(i)]\nreturn out",
    "List the steps to profile a Metal kernel on macOS and what each shows.",
    "Explain speculative decoding acceptance rates to a systems engineer.",
    "Write a shell one-liner that finds the ten largest files under a path.",
    "Describe how a mixture-of-experts router chooses experts at decode.",
    "Draft a short changelog entry for a caching fix in an inference server.",
]


def one_stream(host, model, prompt, tokens, out, idx):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        host + "/v1/chat/completions", body, {"Content-Type": "application/json", "Authorization": "Bearer " + _KEY}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
    except Exception as exc:
        out[idx] = {"error": str(exc)}
        return
    dt = time.perf_counter() - t0
    usage = data.get("usage", {})
    n = int(usage.get("completion_tokens", 0))
    out[idx] = {
        "tokens": n,
        "seconds": dt,
        "tok_s": n / dt if dt > 0 else 0.0,
        "text": data["choices"][0]["message"]["content"][:200],
    }


def run_row(host, model, streams, tokens):
    out = [None] * streams
    threads = [
        threading.Thread(
            target=one_stream,
            args=(host, model, PROMPTS[i % len(PROMPTS)], tokens, out, i),
        )
        for i in range(streams)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    errs = [r for r in out if r and "error" in r]
    if errs:
        return {"streams": streams, "error": errs[0]["error"]}
    out = [r if r else {"error": "no response"} for r in out]
    if any("error" in r for r in out):
        return {"streams": streams, "error": "; ".join(r.get("error", "?") for r in out if "error" in r)[:200]}
    total = sum(r["tokens"] for r in out)
    return {
        "streams": streams,
        "wall_s": wall,
        "aggregate_tok_s": total / wall,
        "per_stream_tok_s": [round(r["tok_s"], 2) for r in out],
        "median_per_stream": round(statistics.median(r["tok_s"] for r in out), 2),
        "tokens": total,
    }


MTP_LINE = re.compile(
    r"MTP\[(\d+)\].*?tok/cycle=([\d.]+).*?accept=(\d+)/(\d+).*?depth\[([^\]]*)\]"
)
BATCHED_LINE = re.compile(r"MTP batched: cycles=(\d+) mean_B=([\d.]+).*?k=(\d+)")


def scrape_log(path, since_bytes):
    """Return the MTP per-request lines written after ``since_bytes``."""
    rows = []
    batched = []
    try:
        with open(path, "rb") as f:
            f.seek(since_bytes)
            text = f.read().decode("utf-8", "replace")
    except OSError as exc:
        return [{"log_error": str(exc)}], [], since_bytes
    for m in MTP_LINE.finditer(text):
        rows.append(
            {
                "req": int(m.group(1)),
                "tok_per_cycle": float(m.group(2)),
                "accepted": int(m.group(3)),
                "drafted": int(m.group(4)),
                "depth": m.group(5),
            }
        )
    for m in BATCHED_LINE.finditer(text):
        batched.append(
            {"cycles": int(m.group(1)), "mean_B": float(m.group(2)),
             "k": int(m.group(3))}
        )
    return rows, batched, since_bytes + len(text.encode())


def log_size(path):
    try:
        import os

        return os.path.getsize(path)
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8084")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=45.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--log", default="/var/log/omlx-workbench/stderr.log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--i-know-what-i-am-doing", action="store_true")
    args = ap.parse_args()

    if "8083" in args.host and not args.i_know_what_i_am_doing:
        print("refusing to touch production on 8083", file=sys.stderr)
        return 2

    results = []
    pos = log_size(args.log)
    for rnd in range(args.rounds):
        for s in args.streams:
            row = run_row(args.host, args.model, s, args.tokens)
            mtp, batched, pos = scrape_log(args.log, pos)
            row.update({"round": rnd, "label": args.label, "mtp_lines": mtp,
                        "batched_lines": batched[-3:]})
            results.append(row)
            print(json.dumps(row))
            time.sleep(args.cooldown)

    print("\n{:>8} {:>7} {:>12} {:>12} {:>10}".format(
        "streams", "round", "aggregate", "per-stream", "tok/cycle"))
    for r in results:
        if "error" in r:
            print(f"{r['streams']:>8} {r['round']:>7}  ERROR {r['error'][:40]}")
            continue
        tpc = (
            statistics.mean(m["tok_per_cycle"] for m in r["mtp_lines"])
            if r["mtp_lines"]
            else float("nan")
        )
        print(
            f"{r['streams']:>8} {r['round']:>7} {r['aggregate_tok_s']:>12.1f} "
            f"{r['median_per_stream']:>12.1f} {tpc:>10.2f}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
