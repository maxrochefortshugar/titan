#!/usr/bin/env python3
"""Fold ``results.jsonl`` into the ROUND4 bisect table.

Every arm is measured twice, once in each direction of the arm list, so the
statistic that goes in the table is the median of the pair rather than either
one of them. The columns are the two rates that mean different things: the wall
rate is what a client sees and the profiler rate is committed tokens over the
summed wall time of the decode cycles themselves, which is the part a kernel
can actually move.

    summarise.py [--results results.jsonl] [--baseline refonly]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    rows = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["arm"], row["mode"])].append(row)
    return rows


def med(rows, key, default=0.0):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(HERE / "results.jsonl"))
    parser.add_argument("--baseline", default="refonly")
    parser.add_argument(
        "--prefix",
        default="",
        help="only arms whose name starts with this (fd_ for the fixed-depth pass)",
    )
    parser.add_argument(
        "--exclude-prefix",
        default="",
        help="drop arms whose name starts with this",
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rows = load(Path(args.results))
    arms = []
    for arm, mode in rows:
        if args.prefix and not arm.startswith(args.prefix):
            continue
        if args.exclude_prefix and arm.startswith(args.exclude_prefix):
            continue
        if arm not in arms:
            arms.append(arm)

    table = {}
    for arm in arms:
        short = rows.get((arm, "short"), [])
        long_ = rows.get((arm, "long"), [])
        lossless = rows.get((arm, "lossless"), [])
        table[arm] = {
            "n_short": len(short),
            "n_long": len(long_),
            "short_wall": med(short, "median_tok_s"),
            "short_prof": med(short, "profiler_tok_s"),
            "short_cycle_ms": med(short, "cycle_ms"),
            "short_accept": med(short, "accepted_per_cycle"),
            "long_wall": med(long_, "wall_tok_s"),
            "long_prof": med(long_, "profiler_tok_s"),
            "long_cycle_ms": med(long_, "cycle_ms"),
            "long_accept": med(long_, "accepted_per_cycle"),
            "prefill_tok_s": med(long_, "prefill_tok_s"),
            "md5": lossless[0]["lossless_md5"] if lossless else "",
        }

    base = table.get(args.baseline, {})
    header = (
        f"{'arm':<30} {'short wall':>10} {'short prof':>10} {'cyc ms':>7} "
        f"{'acc/cyc':>8} {'64k wall':>9} {'64k prof':>9} {'cyc ms':>7} "
        f"{'prefill':>8} {'d short%':>9} {'d 64k%':>8} {'lossless':>10}"
    )
    print(header)
    print("-" * len(header))
    for arm in arms:
        entry = table[arm]
        ds = dl = 0.0
        if base.get("short_prof"):
            ds = 100 * (entry["short_prof"] - base["short_prof"]) / base["short_prof"]
        if base.get("long_prof"):
            dl = 100 * (entry["long_prof"] - base["long_prof"]) / base["long_prof"]
        print(
            f"{arm:<30} {entry['short_wall']:>10.1f} {entry['short_prof']:>10.1f} "
            f"{entry['short_cycle_ms']:>7.2f} {entry['short_accept']:>8.3f} "
            f"{entry['long_wall']:>9.1f} {entry['long_prof']:>9.1f} "
            f"{entry['long_cycle_ms']:>7.2f} {entry['prefill_tok_s']:>8.1f} "
            f"{ds:>+9.1f} {dl:>+8.1f} {entry['md5'][:8]:>10}"
        )

    digests = {e["md5"] for e in table.values() if e["md5"]}
    print()
    print(f"distinct 150-token digests across {len(table)} arms: {len(digests)}")
    for digest in sorted(digests):
        names = [a for a, e in table.items() if e["md5"] == digest]
        print(f"  {digest}  {', '.join(sorted(names))}")

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
