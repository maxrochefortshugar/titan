#!/usr/bin/env python3
"""Tune the depth policy's damping against the machine it will run on.

A fake, for the reason ROUND5 step 1 gives: the trade between probe duty and
convergence time cannot be read off a real run, because a real run gives one
number per configuration per two minutes and the interesting differences are a
few per cent. What it can be read off is a deterministic workload whose true
acceptance curve and true cycle cost are the ones step 1 measured, driven
through the real controller. No GPU, no model, no MLX.

Both workloads come from the step 1 arms at pinned depth 3, which is the only
arm that observes every position:

    short  P(>=1..3) = 637/721, 559/721, 483/721 over 721 cycles
    64k    P(>=1..3) =  82/109,  63/109,  46/109 over 109 cycles

and the cost line is fitted through the two widths step 1 actually ran, the
pinned arm's 3.97 and the adaptive arm's mean.

    python bench/decode/round5/depth_tune.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from titan.core.types import CycleProfile, SequenceId, VerifyOutcome  # noqa: E402
from titan.engine.decode_cycle import ExpectedValueDepthController  # noqa: E402


def profile(width: int, wall_ms: float, committed: int, drafted: int) -> CycleProfile:
    return CycleProfile(
        cycle=1, n_sequences=1, n_rows=width, draft_ms=0.0, verify_ms=0.0,
        accept_ms=0.0, sample_ms=0.0, detok_ms=0.0, ngram_wait_ms=0.0,
        host_syncs=1, tokens_committed=committed, tokens_drafted=drafted,
        wall_ms=wall_ms,
    )


class Workload:
    """A machine and a text, both deterministic.

    ``conditional[i]`` is P(draft i+1 sticks | the chain got that far), which is
    what the estimator is trying to learn. Acceptance is drawn from a fixed
    low-discrepancy sweep rather than a random number generator, so a sweep of
    twenty configurations is twenty comparable runs and not twenty samples.
    """

    def __init__(self, name: str, conditional, intercept: float, slope: float) -> None:
        self.name = name
        self.conditional = [float(p) for p in conditional]
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.step = 0

    def cost_ms(self, width: int) -> float:
        return self.intercept + self.slope * max(1, int(width))

    def accepted(self, depth: int) -> int:
        self.step += 1
        n = 0
        for position in range(1, int(depth) + 1):
            threshold = (self.step * 0.6180339887 + position * 0.31) % 1.0
            if threshold >= self.conditional[min(position, len(self.conditional)) - 1]:
                break
            n += 1
        return n

    def truth(self, depth: int) -> float:
        """Committed tokens per millisecond at a pinned depth. The target."""
        total, survival = 1.0, 1.0
        for position in range(1, depth + 1):
            survival *= self.conditional[min(position, len(self.conditional)) - 1]
            total += survival
        return total / self.cost_ms(depth + 1)

    def drive(self, policy, cycles: int) -> tuple[list[int], float]:
        """Run the policy. Returns the depths and the realised tokens per ms.

        Realised, not chosen: a probe run is in the denominator and in the
        numerator, which is the only honest way to price a duty cycle.
        """
        depths: list[int] = []
        committed = 0.0
        wall = 0.0
        for _ in range(cycles):
            depth = policy.next_depths_for([1])[0]
            depths.append(depth)
            n = self.accepted(depth)
            width = depth + 1
            ms = self.cost_ms(width)
            committed += 1.0 + n
            wall += ms
            outcomes = (
                [VerifyOutcome(SequenceId(1), accepted=tuple(range(n)), bonus=7,
                               n_drafted=depth)]
                if depth > 0
                else []
            )
            policy.record(profile(width, ms, 1 + n, depth), outcomes)
        return depths, committed / wall


SHORT = Workload("short", [0.8835, 0.8776, 0.8641], intercept=7.05, slope=7.50)
LONG = Workload("64k", [0.7523, 0.7683, 0.7302], intercept=24.86, slope=4.22)


def steer(policy, high: bool) -> None:
    """Put the policy at a resting point before it sees the workload."""
    for _ in range(120):
        if high:
            policy.record(
                profile(4, 20.0, 4, 3),
                [VerifyOutcome(SequenceId(1), accepted=(0, 1, 2), bonus=7, n_drafted=3)],
            )
        else:
            policy.record(
                profile(2, 60.0, 1, 1),
                [VerifyOutcome(SequenceId(1), accepted=(), bonus=7, n_drafted=1)],
            )


def run(workload: Workload, cycles: int = 900, **kwargs) -> dict:
    out = {}
    realised = []
    for start in ("fresh", "low", "high"):
        machine = Workload(workload.name, workload.conditional,
                           workload.intercept, workload.slope)
        policy = ExpectedValueDepthController(max_depth=3, min_depth=0, window=64,
                                              rows_budget=32, **kwargs)
        if start != "fresh":
            steer(policy, high=(start == "high"))
        depths, rate = machine.drive(policy, cycles)
        tail = depths[-300:]
        out[start] = {
            "mode": max(set(tail), key=tail.count),
            "mean": statistics.mean(tail),
            "rate": rate,
        }
        realised.append(rate)
    best = max(workload.truth(d) for d in range(0, 4))
    out["worst_of_three"] = min(realised)
    out["fraction_of_best"] = min(realised) / best
    out["spread"] = (max(realised) - min(realised)) / statistics.median(realised)
    return out


def main() -> int:
    for workload in (SHORT, LONG):
        print(f"\n## {workload.name}: what a pinned depth is worth\n")
        print("| depth | committed/ms | against the best |")
        print("|---:|---:|---:|")
        best = max(workload.truth(d) for d in range(0, 4))
        for depth in range(0, 4):
            value = workload.truth(depth)
            print(f"| {depth} | {value:.5f} | {value / best * 100:.1f}% |")

    grid = [
        ("shipped", dict(hysteresis=0.06, probe_every=48, probe_cycles=8)),
        ("h=0.02", dict(hysteresis=0.02, probe_every=48, probe_cycles=8)),
        ("h=0.00", dict(hysteresis=0.0, probe_every=48, probe_cycles=8)),
        ("h=0.02 probe 96/8", dict(hysteresis=0.02, probe_every=96, probe_cycles=8)),
        ("h=0.02 probe 96/6", dict(hysteresis=0.02, probe_every=96, probe_cycles=6)),
        ("h=0.02 probe 128/8", dict(hysteresis=0.02, probe_every=128, probe_cycles=8)),
        ("h=0.02 no probe", dict(hysteresis=0.02, probe_every=0, probe_cycles=0)),
        ("h=0.06 no probe", dict(hysteresis=0.06, probe_every=0, probe_cycles=0)),
    ]
    for workload in (SHORT, LONG):
        print(f"\n## {workload.name}: the policy against that ceiling\n")
        print("| setting | duty | mode from fresh/low/high | worst realised | "
              "of the best | spread |")
        print("|---|---:|---|---:|---:|---:|")
        for label, kwargs in grid:
            r = run(workload, **kwargs)
            duty = (kwargs["probe_cycles"] / kwargs["probe_every"]
                    if kwargs["probe_every"] else 0.0)
            modes = "/".join(str(r[s]["mode"]) for s in ("fresh", "low", "high"))
            print(
                f"| {label} | {duty * 100:.0f}% | {modes} | "
                f"{r['worst_of_three']:.5f} | {r['fraction_of_best'] * 100:.1f}% | "
                f"{r['spread'] * 100:.1f}% |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
