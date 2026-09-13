#!/usr/bin/env python3
"""Turn ``results.jsonl`` into the tables ROUND5.md carries.

One reader for every step, because a step's question decides which columns
matter: step 1 is about the *spread* across repeats of one arm, which is the
one number ROUND4 could not report; step 2 is about the memory line either side
of the prefill next to the decode rate; step 3 is about per-chunk rate against
chunk size.

    summarise.py step1 | step2 | step3
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(mode: str | None = None) -> list[dict]:
    path = HERE / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"no results yet at {path}")
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if mode is None or row.get("mode") == mode:
            out.append(row)
    return out


def spread(values: list[float]) -> float:
    """Peak-to-peak as a fraction of the median. The ROUND4 number."""
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    return (max(values) - min(values)) / median if median else 0.0


def by_arm(items: list[dict], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in items:
        if field in row:
            out.setdefault(row["arm"], []).append(float(row[field]))
    return out


def step1() -> None:
    for mode, label in (("short", "short"), ("long", "64k")):
        items = rows(mode)
        if not items:
            continue
        rates = by_arm(items, "profiler_tok_s")
        depths = by_arm(items, "mean_rows")
        print(f"\n## {label}\n")
        print("| arm | n | tok/s each | median | spread | mean rows each |")
        print("|---|---:|---|---:|---:|---|")
        for arm in sorted(rates):
            r = rates[arm]
            d = depths.get(arm, [])
            print(
                f"| `{arm}` | {len(r)} | {', '.join(f'{x:.1f}' for x in r)} | "
                f"{statistics.median(r):.1f} | {spread(r) * 100:.1f}% | "
                f"{', '.join(f'{x:.2f}' for x in d)} |"
            )
        adaptive = rates.get("adaptive", [])
        fixed = rates.get("fixed3", [])
        if adaptive and fixed:
            print(
                f"\nadaptive median {statistics.median(adaptive):.1f} against "
                f"fixed depth 3 at {statistics.median(fixed):.1f}; the round's "
                f"bar is a spread under 5% and a median not below fixed."
            )


def step2() -> None:
    for mode, label in (("short", "short"), ("long", "64k")):
        items = rows(mode)
        if not items:
            continue
        rates = by_arm(items, "profiler_tok_s")
        print(f"\n## {label}\n")
        print("| arm | n | median tok/s |")
        print("|---|---:|---:|")
        for arm in sorted(rates):
            print(f"| `{arm}` | {len(rates[arm])} | {statistics.median(rates[arm]):.1f} |")
    print("\n## memory either side of the prefill\n")
    print("| arm | pre active | pre cache | post active | post cache | released |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows("prefill"):
        memory = row.get("memory") or {}
        if not memory:
            continue
        print(
            f"| `{row['arm']}` | {memory.get('pre_active_mb', 0):.0f} | "
            f"{memory.get('pre_cache_mb', 0):.0f} | "
            f"{memory.get('post_active_mb', 0):.0f} | "
            f"{memory.get('post_cache_mb', 0):.0f} | "
            f"{memory.get('released_mb', 0):.0f} |"
        )


def step3() -> None:
    print("\n| planned chunk | chunks | full-size tok/s | whole-prompt in-chunk tok/s |")
    print("|---|---:|---:|---:|")
    for row in sorted(rows("prefill"), key=lambda r: r.get("arm", "")):
        table = row.get("by_size") or {}
        if not table:
            continue
        biggest = max(table, key=lambda k: int(k))
        print(
            f"| `{row['arm']}` | {row.get('chunks')} | "
            f"{table[biggest]['tok_s']} (n={table[biggest]['n']}, {biggest} tok) | "
            f"{row.get('in_chunk_tok_s')} |"
        )


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "step1"
    {"step1": step1, "step2": step2, "step3": step3}[which]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
