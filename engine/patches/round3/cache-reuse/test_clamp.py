# SPDX-License-Identifier: Apache-2.0
"""The patched clamp keeps 2048-token forwards and adds one fine stop in the tail.

Simulates the scheduler's prefill loop (scheduler.py:3672-3840) against the
stock rule and the patched one, over the real prompt lengths seen in the logs,
and reports chunk counts, snapshot points, and how many tokens the store keeps.

Run: ~/inference-server/kdev/bin/python test_clamp.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINE, COARSE = 512, 2048


def stock_clamp(chunk_tokens, *, cache_tokens, block_size):
    nb = ((cache_tokens // block_size) + 1) * block_size
    return max(1, min(chunk_tokens, nb - cache_tokens))


def emits(total, block_size, last):
    return total > 0 and total % block_size == 0 and last < total


def run(prompt, base, clamp, block_size, step=COARSE):
    """Return (chunk sizes, boundary snapshot points, tokens the store can keep)."""
    processed, chunks, snaps, last = 0, [], [], -1
    while processed < prompt - base:
        n = min(step, prompt - base - processed)
        n = clamp(n, cache_tokens=base + processed, block_size=block_size)
        processed += n
        chunks.append(n)
        total = base + processed
        if emits(total, block_size, last):
            snaps.append(total)
            last = total
    keep = max([s for s in snaps if s % block_size == 0] or [0])
    return chunks, snaps, keep


def main():
    spec = importlib.util.spec_from_file_location(
        "_cache_reuse_patch", os.path.join(HERE, "patch.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Exercise the clamp without importing omlx: rebuild it from the same rule.
    fine, coarse = FINE, COARSE

    def patched(chunk_tokens, *, cache_tokens, block_size):
        if block_size <= 0 or chunk_tokens <= 0:
            return max(1, chunk_tokens)
        if chunk_tokens >= coarse:
            return stock_clamp(chunk_tokens, cache_tokens=cache_tokens,
                               block_size=coarse)
        last_fine = ((cache_tokens + chunk_tokens) // fine) * fine
        return last_fine - cache_tokens if last_fine > cache_tokens else chunk_tokens

    # Real prompts from staging/clean-seq.log turns 1-5 plus a 64k cold one.
    cases = [(25043, 0), (26481, 24576), (27919, 24576), (29357, 26624),
             (30795, 28672), (64827, 0), (50843, 49152)]
    print(f"{'prompt':>7} {'cached':>7} | {'stock chunks':>12} {'kept':>6} "
          f"| {'fine chunks':>11} {'kept':>6} {'extra':>6} {'saved':>6}")
    total_saved = 0
    for prompt, base in cases:
        cs, _, ks = run(prompt, base, stock_clamp, COARSE)
        cf, _, kf = run(prompt, base, patched, FINE)
        assert sum(cs) == sum(cf) == prompt - base, (prompt, base)
        assert max(cf) <= COARSE and all(c > 0 for c in cf)
        body = [c for c in cf if c == COARSE]
        assert len(body) >= len([c for c in cs if c == COARSE]), "body chunks narrowed"
        saved = kf - ks
        total_saved += saved
        print(f"{prompt:>7} {base:>7} | {len(cs):>12} {ks:>6} "
              f"| {len(cf):>11} {kf:>6} {len(cf)-len(cs):>6} {saved:>6}")

    print(f"\nextra tokens kept across the seven cases: {total_saved}")
    print(f"extra forward launches per prompt: at most 1")

    # Degenerate inputs
    assert patched(0, cache_tokens=0, block_size=FINE) == 1
    assert patched(300, cache_tokens=0, block_size=FINE) == 300, "short prompt unsplit"
    assert patched(511, cache_tokens=1024, block_size=FINE) == 511
    assert patched(513, cache_tokens=1024, block_size=FINE) == 512
    assert patched(700, cache_tokens=1000, block_size=FINE) == 536, "off-boundary base"
    assert patched(4096, cache_tokens=0, block_size=FINE) == COARSE, "body stays coarse"
    assert patched(100, cache_tokens=0, block_size=0) == 100
    print("degenerate inputs OK")

    # Whole-session simulation: each turn appends the model's reply plus a tool
    # result to the previous prompt, and restores from whatever the previous
    # turn managed to store. Turn lengths from staging/clean-seq.log.
    print("\n5-turn session, prompt lengths 25043 26481 27919 29357 30795")
    print(f"{'turn':>4} {'prompt':>7} | {'stock cached':>12} {'recomputed':>10} "
          f"| {'fine cached':>11} {'recomputed':>10}")
    prompts = [25043, 26481, 27919, 29357, 30795]
    stored_s = stored_f = 0
    tot_s = tot_f = 0
    for i, p in enumerate(prompts, 1):
        base_s, base_f = min(stored_s, p), min(stored_f, p)
        _, _, ks = run(p, base_s, stock_clamp, COARSE)
        _, _, kf = run(p, base_f, patched, FINE)
        rs, rf = p - base_s, p - base_f
        tot_s += rs
        tot_f += rf
        print(f"{i:>4} {p:>7} | {base_s:>12} {rs:>10} | {base_f:>11} {rf:>10}")
        stored_s, stored_f = max(stored_s, ks), max(stored_f, kf)
    print(f"total prefilled tokens: stock {tot_s}, fine {tot_f} "
          f"({100 * (tot_s - tot_f) / tot_s:.1f}% fewer)")
    warm_s, warm_f = tot_s - prompts[0], tot_f - prompts[0]
    print(f"warm turns 2-5 only:    stock {warm_s}, fine {warm_f} "
          f"({100 * (warm_s - warm_f) / warm_s:.1f}% fewer)")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
