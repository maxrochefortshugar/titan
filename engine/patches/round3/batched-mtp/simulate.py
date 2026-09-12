#!/usr/bin/env python3
"""Expected throughput for fused batched MTP against plain batched decode.

Two models, both calibrated on measured numbers only.

1. Cycle cost T(R, B, ctx): one backbone forward carrying R rows spread over
   B sequences. Components and their sources:

     shared weights (36 GDN layers, dense projections, norms)
         constant in R; solved so T(1, 1, short) = 24.4 ms, the measured
         MTP-off decode step (kernels/REPORT.md, "MTP off: decode 41")
     expert gather, 48 layers
         0.061 ms/layer at 1 row and 300 GB/s, 490 at 4 rows, 549 at 8+
         (kernels/AUDIT-2026-09-12.md:95). Bytes grow with rows because
         top-10 of 512 experts means more rows touch more experts, so the
         win is the bandwidth ratio, not byte reuse.
     attention over the cache, 12 QSA layers
         sparse gathered arm: 1.42 ms/forward at L=1 and 2.83 at L=4, 65k
         (round3/qsa-verify/REPORT.md section 2), per sequence.
         dense arm: 2.82 at L=1 and 14.9 at L=4, same table.
     PLE n-gram host lookup
         2.3 ms/forward (AUDIT, derived from the rows patch moving decode
         47.9 -> 61.7 tok/s)
     vocabulary head, 8-bit
         measured 0.93 / 1.51 / 4.23 / 1.70 ms at M = 1 / 4 / 8 / 16. M=8
         lands in affine_qmv_wide, M=16 reaches affine_qmm_t_nax (AUDIT:95).
     hyper-connections
         2.0 ms/forward, launch bound

2. Lockstep queue dynamics. The batch cache has one row per sequence, so a
   forward covers every row or none: a subset cannot be advanced. Rows
   therefore step together and a row that accepts more than its peers builds
   an emit queue, capped by clamping its accepted count. Monte Carlo over
   per-row acceptance draws gives tokens per fused cycle.

Run: ~/inference-server/kdev/bin/python simulate.py
"""

import random

# --- measured anchors -------------------------------------------------------
GATHER_MS_PER_LAYER_1ROW = 0.061
GATHER_BW = {1: 300.0, 2: 390.0, 4: 490.0, 8: 549.0}
N_LAYERS = 48
PLE_MS = 2.3
HYPER_MS = 2.0
HEAD_MS = {1: 0.93, 2: 1.20, 4: 1.51, 6: 2.20, 8: 4.23, 12: 3.00, 16: 1.70}
QSA_SPARSE = (1.42, 0.47)  # ms at 1 row per sequence, ms per extra row
QSA_DENSE = (2.82, 4.03)
QSA_SHORT = (0.30, 0.08)  # 2k context, both arms cheap
DRAFT_CHAIN_MS = 2.4  # per row per cycle, depth 3 with the round-2 shortlist
DRAFT_STEP_MS = 0.8  # per extra draft step


def bw(rows):
    keys = sorted(GATHER_BW)
    if rows <= keys[0]:
        return GATHER_BW[keys[0]]
    if rows >= keys[-1]:
        return GATHER_BW[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= rows <= b:
            f = (rows - a) / (b - a)
            return GATHER_BW[a] + f * (GATHER_BW[b] - GATHER_BW[a])


def gather_ms(rows):
    return N_LAYERS * GATHER_MS_PER_LAYER_1ROW * rows * 300.0 / bw(rows)


def head_ms(rows):
    keys = sorted(HEAD_MS)
    if rows in HEAD_MS:
        return HEAD_MS[rows]
    if rows >= keys[-1]:
        return HEAD_MS[keys[-1]] * rows / keys[-1]
    for a, b in zip(keys, keys[1:]):
        if a < rows < b:
            f = (rows - a) / (b - a)
            return HEAD_MS[a] + f * (HEAD_MS[b] - HEAD_MS[a])


def qsa_ms(rows_per_seq, B, ctx, arm):
    base, per = {"sparse": QSA_SPARSE, "dense": QSA_DENSE, "short": QSA_SHORT}[arm]
    return B * (base + per * (rows_per_seq - 1))


# solve the constant so T(1,1,short) = 24.4
_SHARED = 24.4 - (
    gather_ms(1) + qsa_ms(1, 1, 0, "short") + PLE_MS + HYPER_MS + head_ms(1)
)


def T(rows_per_seq, B, arm):
    R = rows_per_seq * B
    return (
        _SHARED
        + gather_ms(R)
        + qsa_ms(rows_per_seq, B, 0, arm)
        + PLE_MS
        + 0.15 * R
        + HYPER_MS
        + head_ms(R)
    )


# --- acceptance profiles ----------------------------------------------------
# conditional accept at depth j; the "measured" row is round-2's fit to
# production's 1.91 tokens per cycle, "code" is copy-lane's repetitive profile
PROFILES = {
    "measured (1.91 tok/cycle)": [0.610, 0.705 * 0.610, 0.705**2 * 0.610],
    "code (repetitive)": [0.780, 0.900 * 0.780, 0.900**2 * 0.780],
    "prose (high entropy)": [0.420, 0.780 * 0.420, 0.780**2 * 0.420],
}


def draw_m(cond, k, rng):
    m = 0
    for j in range(k):
        if rng.random() < cond[j]:
            m += 1
        else:
            break
    return m


def lockstep_tokens_per_cycle(cond, k, B, cap=8, trials=200000, seed=7):
    """Mean emitted tokens per fused cycle under the lockstep queue rule."""
    rng = random.Random(seed)
    q = [0] * B
    emitted = 0
    cycles = 0
    calls = 0
    while cycles < trials:
        if min(q) == 0:
            cycles += 1
            for b in range(B):
                m = draw_m(cond, k, rng)
                m = min(m, max(0, cap - 1 - q[b]))
                q[b] += m + 1
        calls += 1
        for b in range(B):
            q[b] -= 1
            emitted += 1
    return emitted / cycles


def lockstep_tokens_per_cycle_mixed(conds, k, cap=8, trials=200000, seed=11):
    rng = random.Random(seed)
    B = len(conds)
    q = [0] * B
    emitted = cycles = 0
    while cycles < trials:
        if min(q) == 0:
            cycles += 1
            for b in range(B):
                m = draw_m(conds[b], k, rng)
                m = min(m, max(0, cap - 1 - q[b]))
                q[b] += m + 1
        for b in range(B):
            q[b] -= 1
            emitted += 1
    return emitted / cycles


def solo_tokens_per_cycle(cond, k, trials=200000, seed=7):
    rng = random.Random(seed)
    return 1 + sum(draw_m(cond, k, rng) for _ in range(trials)) / trials


# --- report -----------------------------------------------------------------
def main():
    print(f"cost model: shared constant {_SHARED:.2f} ms\n")

    print("forward cost T (ms), rows = B * (k+1)")
    print(f"{'B':>3} {'k':>3} {'rows':>5} {'short':>8} {'65k sparse':>11} "
          f"{'65k dense':>10}")
    for B in (1, 2, 4, 8):
        for k in (0, 1, 3):
            r = k + 1
            print(
                f"{B:>3} {k:>3} {B * r:>5} {T(r, B, 'short'):>8.1f} "
                f"{T(r, B, 'sparse'):>11.1f} {T(r, B, 'dense'):>10.1f}"
            )
    print()

    for pname, cond in PROFILES.items():
        print(f"--- profile: {pname}")
        solo3 = solo_tokens_per_cycle(cond, 3)
        print(f"    single stream, k=3: {solo3:.2f} tokens/cycle, "
              f"{1000 * solo3 / (T(4, 1, 'short') + DRAFT_CHAIN_MS):.1f} tok/s "
              f"short, "
              f"{1000 * solo3 / (T(4, 1, 'sparse') + DRAFT_CHAIN_MS):.1f} tok/s "
              f"at 65k")
        print(f"    {'B':>3} {'k':>3} {'tok/cyc':>8} {'cycle ms':>9} "
              f"{'aggregate':>10} {'per stream':>11} {'plain batched':>14}")
        for B in (2, 4, 8):
            plain = 1000 * B / T(1, B, "short")
            best = None
            for k in (0, 1, 2, 3):
                if k == 0:
                    tpc = float(B)
                else:
                    tpc = lockstep_tokens_per_cycle(cond, k, B, trials=40000)
                chain = 0.0 if k == 0 else B * (DRAFT_CHAIN_MS - (3 - k) * DRAFT_STEP_MS)
                cyc = T(k + 1, B, "short") + chain
                agg = 1000 * tpc / cyc
                mark = ""
                if best is None or agg > best[0]:
                    best = (agg, k)
                print(
                    f"    {B:>3} {k:>3} {tpc:>8.2f} {cyc:>9.1f} {agg:>10.1f} "
                    f"{agg / B:>11.1f} {plain:>14.1f}{mark}"
                )
            print(f"      best k={best[1]}: {best[0]:.1f} vs plain "
                  f"{plain:.1f} tok/s  ({best[0] / plain:.2f}x)")
        print()

    print("65k, sparse arm assumed preserved (per-sequence QSA loop):")
    cond = PROFILES["measured (1.91 tok/cycle)"]
    for B in (2, 4):
        plain = 1000 * B / T(1, B, "sparse")
        for k in (1, 3):
            tpc = lockstep_tokens_per_cycle(cond, k, B, trials=40000)
            cyc = T(k + 1, B, "sparse") + B * (DRAFT_CHAIN_MS - (3 - k) * DRAFT_STEP_MS)
            print(f"    B={B} k={k}: {1000 * tpc / cyc:.1f} vs plain {plain:.1f}")
    print("\n65k, dense arm (what the code does today with BatchQSAKVCache):")
    for B in (2, 4):
        plain = 1000 * B / T(1, B, "dense")
        for k in (1, 3):
            tpc = lockstep_tokens_per_cycle(cond, k, B, trials=40000)
            cyc = T(k + 1, B, "dense") + B * (DRAFT_CHAIN_MS - (3 - k) * DRAFT_STEP_MS)
            print(f"    B={B} k={k}: {1000 * tpc / cyc:.1f} vs plain {plain:.1f}")

    print("\nmin-acceptance collapse (why lockstep costs tokens), k=3:")
    print(f"    {'profile':>26} {'B=1':>7} {'B=2':>7} {'B=4':>7} {'B=8':>7}")
    for pname, cond in PROFILES.items():
        row = [solo_tokens_per_cycle(cond, 3)]
        for B in (2, 4, 8):
            row.append(lockstep_tokens_per_cycle(cond, 3, B, trials=40000) / B)
        print(f"    {pname:>26} " + " ".join(f"{v:>7.2f}" for v in row))

    print("\nheterogeneous rows (one code, one prose, rest measured), k=3:")
    for B in (2, 4):
        mix = [PROFILES["code (repetitive)"], PROFILES["prose (high entropy)"]]
        mix += [PROFILES["measured (1.91 tok/cycle)"]] * (B - 2)
        tpc = lockstep_tokens_per_cycle_mixed(mix, 3, trials=40000)
        cyc = T(4, B, "short") + B * DRAFT_CHAIN_MS
        plain = 1000 * B / T(1, B, "short")
        solo = sum(solo_tokens_per_cycle(c, 3) for c in mix)
        print(f"    B={B}: {tpc:.2f} tok/cycle (rows alone would give "
              f"{solo:.2f}), {1000 * tpc / cyc:.1f} vs plain {plain:.1f}")


if __name__ == "__main__":
    main()
