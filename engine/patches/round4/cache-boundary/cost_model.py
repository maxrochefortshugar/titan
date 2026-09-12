#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-turn latency model for the six-turn prefill_ab probe.

Terms, all in milliseconds:

  chunk(T, ctx) = F + T * (a + b * ctx)
      F = 73      fixed cost of one prefill forward launch
      a = 0.4488  marginal cost per token at an empty cache
      b = 7.63e-6 growth per token of context
      F and a solve AUDIT-2026-09-12.md section A: 992 ms for a 2048-token
      chunk at ctx 0, and 512-token chunks 22 percent worse per token.
      b is the same file's 992 -> 1472 ms slope over a 32k prefill.

  restore(cached) = r * cached
      r = 2.77e-3 ms per token, i.e. 0.10 ms per MB of restored KV at
      27.73 KB per token (a 512-token block file is 14,194,712 bytes).
      Calibrated on the five reconstruct times in workbench-fine-a.log:
      69.3/24576, 72.4/26112, 106.6/26112, 80.7/29184, 85.7/30720.

  snapshot = S per boundary snapshot emitted during the turn
      S = 210. From turn 6 of the workbench run, where both arms restore
      30720 and prefill 1513 tokens and the only difference is one extra
      chunk cut and one extra snapshot: 1.57 - 1.28 s = 290 ms, minus F.
      BoundarySnapshotSSDStore.save runs on the inference thread
      (scheduler.py:6761-6770): extract, mx.eval, and a 115.66 MB copy to
      host bytes, plus up to 2 s of backpressure against the 512 MB pending
      budget (boundary_snapshot_store.py:296-312).

  K   everything outside the scheduler: HTTP, tokenizing the prompt, the
      MTP decode of the 1-2 sampled tokens, the store dispatch. Fitted.

Run: ~/inference-server/kdev/bin/python cost_model.py
"""

from __future__ import annotations

F = 73.0
A = 0.4488
B = 7.63e-6
R = 2.77e-3
S = 210.0
COARSE = 2048
PROMPTS = [25_043, 26_481, 27_919, 29_357, 30_795, 32_233]
MEASURED_BASE = [16.35, 1.72, 2.47, 2.17, 1.82, 1.28]
MEASURED_FINE_4A = [17.62, 2.69, 1.66, 3.11, 1.64, 1.57]
# Arm-level drift: turn 1 is structurally identical in both arms (no fine cut,
# no extra snapshot, same chunks) yet the fine arm, which ran second against a
# cache directory holding the base arm's 5 GB, took 7.8 percent longer.
DRIFT = MEASURED_FINE_4A[0] / MEASURED_BASE[0]


def schedule(cached: int, total: int, fine: int, legacy: bool = False):
    """Return the chunk list [(tokens, ctx)] and the emitted boundaries."""
    chunks, emitted, ctx = [], [], cached
    while ctx < total:
        remaining = total - ctx
        step = min(COARSE, remaining)
        if step >= COARSE:
            nxt = ((ctx // COARSE) + 1) * COARSE
            n = min(step, nxt - ctx)
        else:
            end = ctx + step
            nxt = ((ctx // COARSE) + 1) * COARSE
            if fine and legacy:
                last_fine = (end // fine) * fine
                n = last_fine - ctx if last_fine > ctx else step
            elif fine and nxt < end:
                n = nxt - ctx
            elif fine:
                last_fine = (end // fine) * fine
                last_coarse = (end // COARSE) * COARSE
                gain = last_fine - max(ctx, last_coarse)
                n = last_fine - ctx if (last_fine > ctx and gain >= 384) else step
            else:
                n = min(step, nxt - ctx)
        chunks.append((n, ctx))
        ctx += n
        if ctx % (fine or COARSE) == 0:
            emitted.append(ctx)
    return chunks, emitted


def stored_from(emitted: list[int], cached: int, grid: int) -> int:
    """Longest store the split-GDN gate allows.

    A block that ends on the coarse grid needs a committed checkpoint or
    store_cache truncates the chain there (cache/prefix_cache.py:1285-1330),
    so the store can only reach a boundary whose every intervening coarse
    multiple was also emitted. This is what rejected 26624 in the workbench
    run and cost the round-4a arm its turn-3 store.
    """
    best = cached
    seen = set(emitted)
    for tc in sorted(emitted):
        if tc <= cached or tc % grid != 0:
            continue
        first = ((cached // COARSE) + 1) * COARSE
        if all(c in seen for c in range(first, tc + 1, COARSE)):
            best = tc
    return best


def run(fine: int, legacy: bool = False):
    """Walk the six turns, returning per-turn (cached, chunks, snapshots)."""
    grid = fine or COARSE
    cached, out = 0, []
    for total in PROMPTS:
        chunks, emitted = schedule(cached, total, fine, legacy)
        out.append((cached, chunks, len(emitted)))
        cached = stored_from(emitted, cached, grid)
    return out


def latency(turn, k: float) -> float:
    cached, chunks, nsnap = turn
    ms = k + R * cached + nsnap * S
    for n, ctx in chunks:
        ms += F + n * (A + B * ctx)
    return ms / 1000.0


def fit_k(turns, measured) -> float:
    resid = [
        (m - latency(t, 0.0)) for t, m in zip(turns[1:], measured[1:])
    ]
    return 1000.0 * sum(resid) / len(resid)


def table(name, turns, k, measured=None, drift=1.0):
    print(f"\n{name} (K = {k:.0f} ms, drift x{drift:.3f})")
    head = "turn  prompt  cached  suffix  chunks  snaps  model"
    print(head + ("  measured  error" if measured else ""))
    tot = 0.0
    for i, t in enumerate(turns):
        cached, chunks, nsnap = t
        lat = latency(t, k) * drift
        tot += lat
        row = (
            f"{i + 1:>4}  {PROMPTS[i]:>6}  {cached:>6}  {PROMPTS[i] - cached:>6}  "
            f"{len(chunks):>6}  {nsnap:>5}  {lat:>5.2f}"
        )
        if measured:
            row += f"  {measured[i]:>8.2f}  {lat - measured[i]:>+5.2f}"
        print(row)
    warm = [latency(t, k) * drift for t in turns[1:]]
    print(
        f"warm turns: total {sum(warm):.2f}s  median {sorted(warm)[len(warm) // 2]:.2f}s"
    )
    return warm


def main() -> None:
    stock = run(0)
    r4a = run(512, legacy=True)
    fine512 = run(512)
    fine1024 = run(1024)

    k = fit_k(stock, MEASURED_BASE)
    print("Calibration: K fitted on the five warm turns of the base arm.")
    table("stock, block 2048", stock, k, MEASURED_BASE)
    table("round 4a as measured", r4a, k, MEASURED_FINE_4A, drift=DRIFT)
    print("\nPrediction for the next workbench run (drift assumed 1.0):")
    w512 = table("fine 512, fixed", fine512, k)
    w1024 = table("fine 1024, fixed", fine1024, k)
    wstock = [latency(t, k) for t in stock[1:]]
    print(
        f"\nwarm-turn totals: stock {sum(wstock):.2f}s, "
        f"fine 512 {sum(w512):.2f}s, fine 1024 {sum(w1024):.2f}s"
    )

    print("\nSensitivity to the snapshot cost S (warm-turn total, seconds):")
    print("   S ms   stock   fine512   fine1024")
    global S
    s0 = S
    for s in (100, 210, 400, 600, 900):
        S = float(s)
        a = sum(latency(t, k) for t in stock[1:])
        b = sum(latency(t, k) for t in fine512[1:])
        c = sum(latency(t, k) for t in fine1024[1:])
        print(f"  {s:>5}  {a:>6.2f}   {b:>7.2f}   {c:>8.2f}")
    S = s0


if __name__ == "__main__":
    main()
