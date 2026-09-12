#!/usr/bin/env python3
"""Reconstruct the park/probe timeline of the round-3 workbench prose request
and price three alternative policies against it.

The request is e2e_cold.py's tail: "Explain the CAP theorem in detail with
examples.", max_tokens=300, greedy, issued 15 s after a 64k cold prefill on
the same instance. It is the ONLY request in the round-3 sweep that parks.

Inputs (read only):
  ~/inference-server/staging/server-plain.log.13*   the MTP[...] lines
  ~/inference-server/staging/round3.log             the per-config decode rate

Run: ~/inference-server/kdev/bin/python analyze_logs.py
"""

import re
import sys
from datetime import datetime
from pathlib import Path

STAGING = Path.home() / "inference-server" / "staging"
TS = "%Y-%m-%d %H:%M:%S,%f"

# log file -> config tag, from run-round3.sh's order and the timestamps
CONFIGS = [
    ("server-plain.log.130103", "head8"),
    ("server-plain.log.130359", "conf-depth"),
    ("server-plain.log.130639", "head8+conf-depth"),
    ("server-plain.log.130910", "gdn-scan"),
    ("server-plain.log.131140", "copy-lane"),
    ("server-plain.log.131413", "decode-all"),
]
TOTAL_TOKENS = 300
R_STD_FALLBACK = None

RE_FIN = re.compile(
    r"^(\S+ \S+) - \S+ - INFO - MTP\[7\] finish=(\S+) tokens=(\d+) cycles=(\d+) "
    r"tok/cycle=([\d.]+) accept=(\d+)/(\d+)"
)
RE_PARK = re.compile(r"^(\S+ \S+) - \S+ - INFO - MTP\[7\] parked for (\d+) standard")
RE_PROBE = re.compile(r"^(\S+ \S+) - \S+ - INFO - MTP\[7\] re-entry probe started")
RE_OK = re.compile(r"^(\S+ \S+) - \S+ - INFO - MTP\[7\] re-entry probe succeeded")
RE_RATE = re.compile(r"^\[(\S+)\] decode ([\d.]+) tok/s")


def decode_rates():
    out = {}
    for line in (STAGING / "round3.log").read_text(errors="replace").splitlines():
        m = RE_RATE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def events(path):
    ev = []
    for line in path.read_text(errors="replace").splitlines():
        for rx, kind in ((RE_FIN, "fin"), (RE_PARK, "park"),
                         (RE_PROBE, "probe"), (RE_OK, "ok")):
            m = rx.match(line)
            if m:
                ev.append((datetime.strptime(m.group(1), TS), kind, m))
                break
    return ev


def timeline(path, total_rate):
    """Split the 300-token request into MTP / standard / probe segments."""
    ev = events(path)
    wall = TOTAL_TOKENS / total_rate  # includes a ~40-token prefill, small
    segs = []          # (mode, tokens, seconds)
    mtp_runs = [(int(m.group(3)), m) for _, k, m in ev if k == "fin"]
    parks = [t for t, k, _ in ev if k == "park"]
    probes = [t for t, k, _ in ev if k == "probe"]
    fins = [(t, m) for t, k, m in ev if k == "fin"]
    if not fins:
        return None
    # segment 1: MTP from request start to the first park
    n1 = int(fins[0][1].group(3))
    t_park1 = fins[0][0]
    # standard stretch: park -> next probe start (or end of request)
    if probes:
        std_s = (probes[0] - t_park1).total_seconds()
        std_n = 128
    else:
        std_n = TOTAL_TOKENS - n1
        std_s = None
    tail = []
    for i, (t, m) in enumerate(fins[1:], start=1):
        n = int(m.group(3))
        start = probes[i - 1] if len(probes) >= i else None
        secs = (t - start).total_seconds() if start else None
        tail.append((n, secs, m.group(2)))
    # anything after the last logged segment finished in standard mode
    counted = n1 + std_n + sum(n for n, _, _ in tail)
    # seg-1 wall time is the residual
    known = (std_s or 0.0) + sum(s or 0.0 for _, s, _ in tail)
    s1 = wall - known
    if std_s is None and R_STD_FALLBACK:
        std_s = std_n / R_STD_FALLBACK
        s1 = wall - std_s
    segs.append(("mtp", n1, s1))
    segs.append(("std", std_n, std_s))
    for n, s, why in tail:
        segs.append(("probe", n, s))
    return segs, counted


def main():
    rates = decode_rates()
    rows = []
    # First pass: the configs whose park was followed by a probe give a
    # directly measured standard-decode rate (128 tokens between two log
    # lines). Configs whose request ended inside the park need that rate to
    # split their single wall-clock number, so take the median.
    measured = []
    for fname, tag in CONFIGS:
        p = STAGING / fname
        if not p.exists() or tag not in rates:
            continue
        out = timeline(p, rates[tag])
        if out and out[0][1][2]:
            measured.append(out[0][1][1] / out[0][1][2])
    global R_STD_FALLBACK
    R_STD_FALLBACK = sorted(measured)[len(measured) // 2] if measured else None
    print(f"{'config':18s} {'seg':6s} {'tok':>5s} {'s':>7s} {'tok/s':>7s}")
    for fname, tag in CONFIGS:
        p = STAGING / fname
        if not p.exists() or tag not in rates:
            continue
        out = timeline(p, rates[tag])
        if out is None:
            continue
        segs, counted = out
        for mode, n, s in segs:
            r = n / s if s else float("nan")
            print(f"{tag:18s} {mode:6s} {n:5d} {s if s else float('nan'):7.3f} {r:7.1f}")
        rows.append((tag, segs, counted, rates[tag]))
    print()
    project(rows)


def project(rows):
    print("Modelled tok/s for the same 300-token request")
    hdr = f"{'config':18s} {'stock':>7s} {'a:never':>8s} {'b:512/32':>9s} {'c:2 probes':>11s}"
    print(hdr)
    for tag, segs, counted, stock in rows:
        n1 = segs[0][1]
        s1 = segs[0][2]
        r_mtp = n1 / s1 if s1 and s1 > 0 else float("nan")
        std = [s for s in segs if s[0] == "std" and s[2]]
        r_std = (std[0][1] / std[0][2]) if std else None
        probes = [s for s in segs if s[0] == "probe"]
        if r_std is None:
            # no probe fired, so the standard stretch is the residual
            rest = TOTAL_TOKENS - n1
            r_std = rest / max(1e-6, TOTAL_TOKENS / stock - s1)
        # (a) never park: the whole request stays at the measured MTP rate
        a = r_mtp
        # (b) park at the same cycle, 512-token cooldown: no probe fits in
        #     the remaining budget, so the tail is pure standard decode
        rest = TOTAL_TOKENS - n1
        b = TOTAL_TOKENS / (n1 / r_mtp + rest / r_std)
        # (c) stock until two probes have failed, standard after that
        t = n1 / r_mtp
        used = n1
        for i, (_, n, s) in enumerate(probes):
            if i >= 2:
                break
            t += 128 / r_std + (s if s else n / r_mtp)
            used += 128 + n
        t += max(0, TOTAL_TOKENS - used) / r_std
        c = TOTAL_TOKENS / t
        print(f"{tag:18s} {stock:7.1f} {a:8.1f} {b:9.1f} {c:11.1f}")


if __name__ == "__main__":
    sys.exit(main())
