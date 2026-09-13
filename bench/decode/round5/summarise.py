#!/usr/bin/env python3
"""Turn ``results.jsonl`` into the tables ROUND5.md carries.

One reader for every step, because a step's question decides which columns
matter: step 1 is about the *spread* across repeats of one arm, which is the
one number ROUND4 could not report; step 2 is about the memory line either side
of the prefill next to the decode rate; step 3 is about per-chunk rate against
chunk size.

    summarise.py step1 | step2 | step3 | step4 | step5
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


def _chunk_key(row: dict) -> int:
    arm = str(row.get("arm", ""))
    digits = "".join(ch for ch in arm if ch.isdigit())
    return int(digits) if digits else 0


def step3() -> None:
    items = rows("prefill")
    print("\n## per-chunk rate against chunk size\n")
    print("| arm | n | chunks | full-size tok/s | in-chunk tok/s | end-to-end tok/s |")
    print("|---|---:|---:|---:|---:|---:|")
    grouped: dict[str, list[dict]] = {}
    for row in items:
        if row.get("by_size"):
            grouped.setdefault(str(row["arm"]), []).append(row)
    for arm, group in sorted(grouped.items(), key=lambda kv: _chunk_key(kv[1][0])):
        table = group[0]["by_size"]
        biggest = max(table, key=lambda k: int(k))
        full = [r["by_size"][biggest]["tok_s"] for r in group if biggest in r["by_size"]]
        inchunk = [r["in_chunk_tok_s"] for r in group]
        e2e = [r.get("end_to_end_tok_s", 0.0) for r in group]
        print(
            f"| `{arm}` | {len(group)} | {group[0].get('chunks')} | "
            f"{statistics.median(full):.1f} ({biggest} tok) | "
            f"{statistics.median(inchunk):.1f} | {statistics.median(e2e):.1f} |"
        )
    print("\n## per-chunk fixed cost, the wall between two forwards\n")
    print("| arm | median gap ms | after a snapshot | after a plain chunk | "
          "gap total s | gap share |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm, group in sorted(grouped.items(), key=lambda kv: _chunk_key(kv[1][0])):
        med = statistics.median([r.get("gap_ms_median", 0.0) for r in group])
        snap = statistics.median([r.get("gap_after_snapshot_ms", 0.0) for r in group])
        plain = statistics.median([r.get("gap_after_plain_ms", 0.0) for r in group])
        total = statistics.median([r.get("gap_ms_total", 0.0) for r in group])
        share = statistics.median([r.get("gap_fraction", 0.0) for r in group])
        print(
            f"| `{arm}` | {med:.2f} | {snap:.2f} | {plain:.2f} | "
            f"{total / 1000.0:.2f} | {share * 100:.1f}% |"
        )


def step4() -> None:
    """The pooled fp32 indexer bank, at 64k plain and drafted.

    The four arms are named ``bank{on,off}-{plain,draft}``: plain is
    ``speculation.enabled=false``, which is the width-1 decode where the
    indexer runs once a token, and draft is the production cycle.
    """
    items = [r for r in rows("long") if str(r.get("arm", "")).startswith("bank")]
    rates = by_arm(items, "profiler_tok_s")
    cycle = by_arm(items, "cycle_ms")
    print("\n## 64k decode\n")
    print("| arm | n | tok/s each | median | median cycle ms |")
    print("|---|---:|---|---:|---:|")
    for arm in sorted(rates):
        r = rates[arm]
        c = cycle.get(arm, [0.0])
        print(f"| `{arm}` | {len(r)} | {', '.join(f'{x:.1f}' for x in r)} | "
              f"{statistics.median(r):.1f} | {statistics.median(c):.2f} |")
    digests: dict[str, set[str]] = {}
    for row in rows("lossless"):
        digests.setdefault(row["arm"], set()).add(row["lossless_md5"])
    if digests:
        print("\n## the lossless digest, which has to be identical\n")
        print("| arm | digests |")
        print("|---|---|")
        for arm, seen in sorted(digests.items()):
            print(f"| `{arm}` | {', '.join(sorted(seen))} |")


def step5() -> None:
    """The production-config A/B: before this round against after it."""
    for mode, label, field in (
        ("short", "short decode", "profiler_tok_s"),
        ("long", "64k decode", "profiler_tok_s"),
        ("long", "cold 65k prefill", "prefill_tok_s"),
    ):
        items = rows(mode)
        if not items:
            continue
        rates = by_arm(items, field)
        print(f"\n## {label}\n")
        print("| arm | n | each | median |")
        print("|---|---:|---|---:|")
        for arm in sorted(rates):
            r = rates[arm]
            print(f"| `{arm}` | {len(r)} | {', '.join(f'{x:.1f}' for x in r)} | "
                  f"{statistics.median(r):.1f} |")
    digests: dict[str, set[str]] = {}
    for row in rows("lossless"):
        digests.setdefault(row["arm"], set()).add(row["lossless_md5"])
    if digests:
        print("\n## the 150-token lossless digest\n")
        for arm, seen in sorted(digests.items()):
            print(f"- `{arm}`: {', '.join(sorted(seen))}")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "step1"
    {"step1": step1, "step2": step2, "step3": step3, "step4": step4,
     "step5": step5}[which]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
