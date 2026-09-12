#!/usr/bin/env python3
"""Break-even arithmetic for fixed depth vs the per-cycle confidence gate.

Cost model from kernels/round2/mtp/REPORT.md section 3(c):

    C(0) = 24.4 ms   plain one-row forward (41 tok/s with MTP off)
    D_1  = 1.6  ms   one extra verify row; depth 1 reuses the fold's logits
    D_k             for k >= 2: verify row + head layer + lm_head
                    = 2.93 ms stock, 1.92 ms with the shortlist drafter
    C(k) = C(k-1) + D_k

Yield at fixed depth d with conditional acceptance a_1..a_d:

    E(d) = 1 + sum_{j=1..d} prod_{i<=j} a_i      tokens per cycle
    rate = 1000 * E(d) / C(d)                    tokens per second

Section 0 back-solves the acceptance profile from the two field
measurements we have (62 tok/s at depth 3, 56 tok/s at depth 4) and checks
the model reproduces both. Everything after uses the shortlist cost.

Run: ~/inference-server/kdev/bin/python breakeven.py
"""

import random

C0 = 24.4
D1 = 1.6
D_STOCK = 2.93
D_SHORT = 1.92


def cost(d: int, dk: float) -> float:
    c = C0
    for k in range(1, d + 1):
        c += D1 if k == 1 else dk
    return c


def geom(a1: float, r: float, n: int = 8):
    return [a1 * (r ** j) for j in range(n)]


def fixed(a, d: int, dk: float):
    e, run = 1.0, 1.0
    for j in range(d):
        run *= a[j]
        e += run
    return e, 1000.0 * e / cost(d, dk)


def adaptive_floor(j: int, expected: float, a, dk: float) -> float:
    """The floor patch.py computes with OMLX_MTP_CONF_ADAPT=1: deciding
    whether to take step j+1 with j drafts in hand, the floor on the running
    product is D_{j+1} * expected / (p[j] * C(j))."""
    c_j = cost(j, dk)
    d_marg = D1 if j + 1 == 1 else dk
    p_bar = a[j] if j < len(a) else a[-1]
    if p_bar <= 0:
        return 0.90
    return min(0.90, max(0.02, d_marg * expected / (p_bar * c_j)))


def gate(a, pmin=0.25, max_depth=5, dk=D_SHORT, seed=0, n=200000, adapt=True,
         sync_ms=0.0, min_depth=1, conc=6.0, over=1.0, pstep=0.0, mix=None,
         mix_w=0.5, ema=None):
    """Monte-Carlo tokens/s for the gate.

    Per cycle the step-j proposal probability q_j is drawn Beta with mean
    ``min(0.98, over * a_j)`` and concentration ``conc``; the draft is then
    accepted with probability ``q_j / over``, so realised acceptance still
    averages ``a_j`` while ``over`` > 1 models an over-confident head and
    ``conc`` controls how much spread the gate has to work with. The same
    draw drives the stop rule and the accept, which is the whole point: the
    gate only wins if confidence carries information about acceptance.
    """
    rng = random.Random(seed)
    tot_tok = 0.0
    tot_ms = 0.0
    hist = [0] * (max_depth + 1)
    for _ in range(n):
        # ``mix`` makes each cycle draw its regime: the gate sees the regime
        # through this cycle's confidences, a fixed depth cannot, and the
        # controller's acceptance EMA only sees the blend.
        cyc = a if (mix is None or rng.random() < mix_w) else mix
        floor_a = ema if ema is not None else cyc
        run, expected, k = 1.0, 1.0, 0
        confs = []
        for j in range(max_depth):
            m = min(0.98, over * (cyc[j] if j < len(cyc) else cyc[-1]))
            al = max(1e-3, m * conc)
            be = max(1e-3, (1.0 - m) * conc)
            q = rng.betavariate(al, be)
            confs.append(q)
            k = j + 1
            run *= q
            expected += run
            if k >= max_depth:
                break
            if k >= min_depth:
                if pstep > 0.0 and q < pstep:
                    break
                floor = adaptive_floor(k, expected, floor_a, dk) if adapt else pmin
                if run < floor:
                    break
        m_acc = 0
        for j in range(k):
            if rng.random() < min(1.0, confs[j] / over):
                m_acc += 1
            else:
                break
        hist[k] += 1
        tot_tok += m_acc + 1
        tot_ms += cost(k, dk) + sync_ms * max(0, k - min_depth)
    return tot_tok / n, 1000.0 * tot_tok / tot_ms, [h / n for h in hist]


# ---------------------------------------------------------------------------


def section0():
    print("=== 0. calibration against the two field measurements ===")
    print("  target: 62 tok/s at fixed depth 3, 56 tok/s at fixed depth 4, "
          "stock drafter")
    best = None
    for a1 in [x / 200 for x in range(80, 160)]:
        for r in [x / 200 for x in range(140, 200)]:
            a = geom(a1, r)
            _, r3 = fixed(a, 3, D_STOCK)
            _, r4 = fixed(a, 4, D_STOCK)
            err = (r3 - 62.0) ** 2 + (r4 - 56.0) ** 2
            if best is None or err < best[0]:
                best = (err, a1, r, r3, r4)
    _, a1, r, r3, r4 = best
    print(f"  best geometric fit a_j = {a1:.3f} * {r:.3f}^(j-1)  ->  "
          f"depth3 {r3:.1f} tok/s, depth4 {r4:.1f} tok/s")
    a = geom(a1, r)
    print("  profile: " + " ".join(f"a{j+1}={v:.3f}" for j, v in enumerate(a[:5])))
    e3, _ = fixed(a, 3, D_STOCK)
    print(f"  tokens/cycle at depth 3 = {e3:.3f}  (audit reports ~1.91)")
    return a


def main():
    fitted = section0()

    profiles = {
        "code, repetitive": geom(0.78, 0.90),
        "measured, mixed": fitted,
        "free prose": geom(0.42, 0.78),
    }

    for dk, tag in ((D_STOCK, "stock drafter, D_k=2.93"),
                    (D_SHORT, "shortlist drafter, D_k=1.92")):
        print(f"\n=== 1. fixed depth, tokens/cycle and tok/s ({tag}) ===")
        print(f"{'profile':<20}" + "".join(f"{'d=' + str(d):>17}"
                                           for d in range(0, 6)))
        for name, a in profiles.items():
            row = f"{name:<20}"
            for d in range(0, 6):
                e, rr = fixed(a, d, dk)
                row += f"{e:>7.2f}/{rr:>6.1f}  "
            print(row)

    dk = D_SHORT
    print(f"\n=== 2. gate vs fixed, shortlist cost, conc=6, calibrated head, "
          f"sync 0 ms ===")
    print(f"{'profile':<20}{'fix3':>8}{'fix4':>8}{'fix5':>8}"
          f"{'p.15':>8}{'p.25':>8}{'p.35':>8}{'adaptive':>10}{'mean k':>9}"
          f"   depth histogram k=0..5")
    for name, a in profiles.items():
        row = f"{name:<20}"
        for d in (3, 4, 5):
            row += f"{fixed(a, d, dk)[1]:>8.1f}"
        for pm in (0.15, 0.25, 0.35):
            row += f"{gate(a, pm, 5, dk, adapt=False)[1]:>8.1f}"
        _, rr, hist = gate(a, 0.25, 5, dk, adapt=True)
        row += f"{rr:>10.1f}{sum(i * h for i, h in enumerate(hist)):>9.2f}   "
        row += "[" + " ".join(f"{h:.2f}" for h in hist) + "]"
        print(row)

    print(f"\n=== 3. same, over-confident head (over=1.25) ===")
    print(f"{'profile':<20}{'fix3':>8}{'p.15':>8}{'p.25':>8}{'p.35':>8}"
          f"{'adaptive':>10}{'mean k':>9}")
    for name, a in profiles.items():
        row = f"{name:<20}{fixed(a, 3, dk)[1]:>8.1f}"
        for pm in (0.15, 0.25, 0.35):
            row += f"{gate(a, pm, 5, dk, adapt=False, over=1.25)[1]:>8.1f}"
        _, rr, hist = gate(a, 0.25, 5, dk, adapt=True, over=1.25)
        row += f"{rr:>10.1f}{sum(i * h for i, h in enumerate(hist)):>9.2f}"
        print(row)

    print(f"\n=== 4. cost of the host sync the gate needs (adaptive floor) ===")
    print(f"{'profile':<20}{'fix3':>8}" +
          "".join(f"{'sync ' + f'{s:.2f}':>11}"
                  for s in (0.0, 0.10, 0.20, 0.35, 0.50)))
    for name, a in profiles.items():
        row = f"{name:<20}{fixed(a, 3, dk)[1]:>8.1f}"
        for s in (0.0, 0.10, 0.20, 0.35, 0.50):
            row += f"{gate(a, 0.25, 5, dk, adapt=True, sync_ms=s)[1]:>11.1f}"
        print(row)

    print(f"\n=== 5. how much confidence spread the gate needs "
          f"(adaptive, sync 0.20) ===")
    print(f"{'profile':<20}{'fix3':>8}" +
          "".join(f"{'conc=' + str(int(c)):>10}" for c in (200, 60, 20, 6, 2)))
    for name, a in profiles.items():
        row = f"{name:<20}{fixed(a, 3, dk)[1]:>8.1f}"
        for c in (200.0, 60.0, 20.0, 6.0, 2.0):
            row += f"{gate(a, 0.25, 5, dk, adapt=True, conc=c, sync_ms=0.20)[1]:>10.1f}"
        print(row)

    print(f"\n=== 6. per-step floor (llama.cpp form) vs product floor, "
          f"sync 0.20 ===")
    print(f"{'profile':<20}{'fix3':>8}{'step.30':>9}{'step.50':>9}"
          f"{'step.70':>9}{'product adaptive':>18}")
    for name, a in profiles.items():
        row = f"{name:<20}{fixed(a, 3, dk)[1]:>8.1f}"
        for ps in (0.30, 0.50, 0.70):
            row += (f"{gate(a, 0.0, 5, dk, adapt=False, pstep=ps, sync_ms=0.20)[1]:>9.1f}")
        row += f"{gate(a, 0.25, 5, dk, adapt=True, sync_ms=0.20)[1]:>18.1f}"
        print(row)

    print("\n=== 7. mixed content: cycles alternate between two regimes ===")
    print("  a fixed depth must serve both; the acceptance EMA only sees the")
    print("  blend; the gate reads this cycle's own confidences.")
    code, prose = profiles["code, repetitive"], profiles["free prose"]
    print(f"{'mix (code share)':<20}" +
          "".join(f"{'fix' + str(d):>8}" for d in range(1, 6)) +
          f"{'gate p.25':>11}{'gate adapt':>12}{'mean k':>9}{'gain vs best fixed':>20}")
    for w in (0.25, 0.50, 0.75):
        blend = [w * c + (1 - w) * p for c, p in zip(code, prose)]
        row = f"{'code ' + str(int(w * 100)) + '%':<20}"
        fixed_rates = []
        for d in range(1, 6):
            # a fixed depth pays the blended acceptance
            e, _ = fixed(blend, d, dk)
            e_true = 0.0
            for src, ww in ((code, w), (prose, 1 - w)):
                e_true += ww * fixed(src, d, dk)[0]
            rr = 1000.0 * e_true / cost(d, dk)
            fixed_rates.append(rr)
            row += f"{rr:>8.1f}"
        r_p, r_a = None, None
        _, r_p, _ = gate(code, 0.25, 5, dk, adapt=False, sync_ms=0.20,
                         mix=prose, mix_w=w)
        _, r_a, hist = gate(code, 0.25, 5, dk, adapt=True, sync_ms=0.20,
                            mix=prose, mix_w=w, ema=blend)
        best_fixed = max(fixed_rates)
        row += (f"{r_p:>11.1f}{r_a:>12.1f}"
                f"{sum(i * h for i, h in enumerate(hist)):>9.2f}"
                f"{100 * (r_a / best_fixed - 1):>19.1f}%")
        print(row)

    print("\n=== 8. adaptive floors implied by the fitted profile "
          "(shortlist cost) ===")
    a = fitted
    exp_, run = 1.0, 1.0
    for j in range(0, 5):
        f = adaptive_floor(j, exp_, a, dk)
        print(f"  step {j+1}: C({j})={cost(j, dk):>5.2f}  "
              f"D={D1 if j == 0 else dk:.2f}  p[{j}]={a[j]:.3f}  "
              f"E={exp_:.3f}  floor={f:.3f}")
        run *= a[j]
        exp_ += run


if __name__ == "__main__":
    main()
