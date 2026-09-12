#!/usr/bin/env python3
"""Stage budget table for a decode-profile run, and A/B of two runs.

    analyze.py RUN [--ctx-min N] [--ctx-max N] [--k K] [--csv]
    analyze.py --compare RUN_A RUN_B

RUN is a JSON file written by patch.py or a directory of them (the newest is
used; pass --all to pool every file in the directory).  Times are medians over
the selected cycles.  GB/s is the stage's modelled bytes over its median wall
time, against the 718 GB/s read ceiling measured in AUDIT-2026-09-12.md.

Bytes per stage are modelled, not measured:
  lm_head        passes x head weight bytes
  bb_moe         distinct experts touched by M rows x bytes per expert x 48
  bb_attn        QSA weights + the whole KV/indexer cache read
  bb_gdn         GDN weights (the recurrent state is inside the cache figure)
  ngram_lookup   rows read x packed row stride
  verify forward dense weights + experts + KV
In wall mode these rates are dispatch-bound and meaningless where the stage is
not the thing the GPU is doing; read them only from a run recorded with
OMLX_DECODE_PROFILE_SYNC=1.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

CEILING = 718.0  # GB/s, read-only, measured


def load(path: Path, pool: bool = False) -> list[dict]:
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise SystemExit(f"no profile JSON under {path}")
        files = files if pool else files[-1:]
    else:
        files = [path]
    return [json.loads(f.read_text()) | {"_file": f.name} for f in files]


def cycles(payloads, ctx_min=0, ctx_max=None, k=None):
    out = []
    for p in payloads:
        for rec in p["records"]:
            if rec.get("ctx", 0) < ctx_min:
                continue
            if ctx_max is not None and rec.get("ctx", 0) > ctx_max:
                continue
            if k is not None and rec.get("k") != k:
                continue
            out.append(rec)
    return out


def med(xs):
    return st.median(xs) if xs else 0.0


def stage_bytes(rec: dict, key: str) -> float:
    b = rec.get("bytes", {})
    if key == "lm_head":
        return b.get("lm_head", 0.0)
    if key == "bb_moe":
        return b.get("experts", 0.0)
    if key == "bb_attn":
        return 0.40e9 + b.get("kv", 0.0)
    if key == "bb_gdn":
        return 1.25e9
    if key == "ngram_lookup":
        return b.get("ngram", 0.0)
    if key == "verify_dispatch":
        return b.get("dense", 0.0) + b.get("experts", 0.0) + b.get("kv", 0.0)
    return 0.0


def table(recs, title, csv=False):
    cyc = med([r["cycle_ms"] for r in recs])
    print(f"\n{title}")
    print(f"  cycles {len(recs)}  median cycle {cyc:.2f} ms  "
          f"median ctx {med([r.get('ctx', 0) for r in recs]):.0f}  "
          f"median k {med([r.get('k', 0) for r in recs]):.2f}  "
          f"median accepted {med([r.get('m', 0) for r in recs]):.2f}  "
          f"median M {med([r.get('M', 0) for r in recs]):.2f}")
    rows = []
    phases = ["pre", "verify_dispatch", "accept", "commit", "head_gap",
              "draft", "post"]
    for name in phases:
        v = med([r["phase_ms"].get(name, 0.0) for r in recs])
        rows.append(("phase", name, v, v / cyc * 100 if cyc else 0.0,
                     med([stage_bytes(r, name) for r in recs])))
    subkeys = sorted({k for r in recs for k in r["sub_ms"]})
    for name in subkeys:
        v = med([r["sub_ms"].get(name, 0.0) for r in recs])
        rows.append(("sub", name, v, v / cyc * 100 if cyc else 0.0,
                     med([stage_bytes(r, name) for r in recs])))
    synckeys = sorted({k for r in recs for k in r["sync_ms"]})
    for name in synckeys:
        v = med([r["sync_ms"].get(name, 0.0) for r in recs])
        n = med([r["sync_n"].get(name, 0) for r in recs])
        rows.append(("sync", f"{name} (n={n:.0f})", v,
                     v / cyc * 100 if cyc else 0.0, 0.0))
    if csv:
        print("kind,stage,ms,pct,gbs")
    else:
        print(f"  {'kind':5s} {'stage':28s} {'ms':>8s} {'% cycle':>8s} "
              f"{'GB/s':>8s} {'% ceil':>7s}")
    for kind, name, v, pct, byts in rows:
        gbs = (byts / 1e9) / (v / 1000.0) if v > 0 and byts > 0 else 0.0
        if csv:
            print(f"{kind},{name},{v:.4f},{pct:.2f},{gbs:.1f}")
        else:
            print(f"  {kind:5s} {name:28s} {v:8.3f} {pct:8.1f} "
                  f"{(f'{gbs:.0f}' if gbs else '-'):>8s} "
                  f"{(f'{gbs / CEILING * 100:.0f}' if gbs else '-'):>7s}")
    resid = [abs(sum(r["phase_ms"].values()) - r["cycle_ms"])
             / max(r["cycle_ms"], 1e-9) * 100 for r in recs]
    print(f"  per-cycle phase residual: median {med(resid):.4f}%, "
          f"max {max(resid):.4f}% (the phases are an exact partition)")
    syncs = med([sum(r["sync_n"].values()) for r in recs])
    print(f"  host syncs per cycle (eval/async_eval/synchronize/tolist/item): "
          f"{syncs:.1f}")
    tok = med([r.get("m", 0) + 1 for r in recs])
    print(f"  tokens per cycle {tok:.2f} -> {tok / (cyc / 1000.0):.1f} tok/s "
          f"at this cycle time")
    return {name: v for kind, name, v, _, _ in rows if kind != "sync"}


def compare(a, b, args):
    ra = cycles(load(a, args.all), args.ctx_min, args.ctx_max, args.k)
    rb = cycles(load(b, args.all), args.ctx_min, args.ctx_max, args.k)
    ma = table(ra, f"A: {a}", args.csv)
    mb = table(rb, f"B: {b}", args.csv)
    print(f"\n  {'stage':28s} {'A ms':>9s} {'B ms':>9s} {'delta':>9s} "
          f"{'B/A':>7s}")
    for name in sorted(set(ma) | set(mb), key=lambda n: -abs(
            mb.get(n, 0.0) - ma.get(n, 0.0))):
        x, y = ma.get(name, 0.0), mb.get(name, 0.0)
        if max(x, y) < 0.005:
            continue
        print(f"  {name:28s} {x:9.3f} {y:9.3f} {y - x:+9.3f} "
              f"{(y / x if x else 0):7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--ctx-min", type=int, default=0)
    ap.add_argument("--ctx-max", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--all", action="store_true",
                    help="pool every JSON in the directory")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1], args)
    if not args.run:
        ap.error("give a run file/directory or --compare A B")
    payloads = load(args.run, args.all)
    recs = cycles(payloads, args.ctx_min, args.ctx_max, args.k)
    if not recs:
        raise SystemExit("no cycles matched the filters")
    mode = "SYNC (device time)" if payloads[0].get("sync_mode") else "wall"
    table(recs, f"{args.run} [{mode}, {len(payloads)} request(s)]", args.csv)


if __name__ == "__main__":
    main()
