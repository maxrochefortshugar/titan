#!/usr/bin/env python3
"""Paired prefill benchmark for the three Flash-Next prefill patches.

Configurations, all in one process (loading the model twice is not affordable):

  base            stock oMLX: M5 gather_qmm reroute, MoE gate/up fusion,
                  sdpa256, PLE in mmap (SSD) mode
  +ple            base + the repacked PLE table (``rows`` mode: one contiguous
                  100 B read per row instead of three 16 KB pages)
  +norm           base + the bf16 grouped-RMSNorm kernel in prefill_forward
  +ple+norm       both
  +int8           base + the int8 x int4 routed-expert prefill kernel
                  (only if the 11 GB of tables fit under the memory budget)
  all             everything applicable

Protocol (profile-flashnext/REPORT.md section "Method"): the GPU power-manages
under sustained prefill (+33% over a dozen chunks), so every configuration is
measured as a **minimum of 3** after a **12 s cooldown**, and a **fresh
baseline is measured immediately before every variant**.  Deltas are quoted
against that paired baseline, never against a baseline from earlier in the run.

Memory: nothing here loads the 32 GB planar pack (``rows`` mode reads bounded
os.pread ranges out of layer1.rows.bin), and the int8 stage is skipped unless
the measured footprint plus its tables stays under MEM_BUDGET_GB.

Run: ``. ./_env.sh && $PY combined_bench.py [--stages chunk ple32k]``
"""
import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE must come first: kernels/moe-int8 also has a module named "patch", and it
# is loaded by path below rather than by name for exactly that reason.
sys.path.insert(0, os.path.expanduser("~/inference-server/profile-flashnext"))
sys.path.insert(0, HERE)

import patch as ple_patch          # noqa: E402
import norm_patch                  # noqa: E402

TOKENS = int(os.environ.get("BENCH_TOKENS", "2048"))
REPS = int(os.environ.get("BENCH_REPS", "3"))
COOL = float(os.environ.get("BENCH_COOL", "12"))
LONG_TOKENS = int(os.environ.get("BENCH_LONG_TOKENS", "32768"))
LONG_REPS = int(os.environ.get("BENCH_LONG_REPS", "2"))
MEM_BUDGET_GB = float(os.environ.get("BENCH_MEM_BUDGET_GB", "85"))
MIN_FREE_GB = float(os.environ.get("BENCH_MIN_FREE_GB", "12"))


# ------------------------------------------------------------------- memory
def rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576 if out else 0.0


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 16384
    d = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            d[k.strip()] = int(v)
    free = (d.get("Pages free", 0) + d.get("Pages speculative", 0)) * page / 1e9
    inactive = d.get("Pages inactive", 0) * page / 1e9
    return free, inactive, d.get("Pageins", 0)


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    for tok in out.split():
        pass
    try:
        part = out.split("used =")[1].split()[0]
        return float(part.rstrip("M")) / 1024 if part.endswith("M") else float(part.rstrip("G"))
    except Exception:  # noqa: BLE001
        return 0.0


_SWAP0 = None


def guard(where):
    """Abort cleanly rather than let the machine go to swap."""
    global _SWAP0
    free, inactive, _ = vm()
    swap = swap_used_gb()
    if _SWAP0 is None:
        _SWAP0 = swap
    if free + inactive < MIN_FREE_GB or swap - _SWAP0 > 2.0:
        raise MemoryError(f"{where}: free {free:.1f} + inactive {inactive:.1f} GB, "
                          f"swap +{swap - _SWAP0:.1f} GB -- stopping")
    return free, inactive


def mem_snapshot():
    free, inactive, _ = vm()
    return dict(rss_gb=rss_gb(), mlx_active_gb=mx.get_active_memory() / 1e9,
                mlx_peak_gb=mx.get_peak_memory() / 1e9,
                mlx_cache_gb=mx.get_cache_memory() / 1e9,
                sys_free_gb=free, sys_inactive_gb=inactive)


# -------------------------------------------------------- PLE lookup timing
PLE_T = {"calls": [], "rows": []}


def install_ple_timer(model):
    """Time the PLE embedding lookup in situ, the same way for every config.

    Wraps ``Qwen4ExpNGramEmbedding.__call__``, i.e. the n-gram hashing plus the
    table gather.  The hashing itself is lazy; it is forced inside the gather by
    ``_host_indices``'s ``mx.eval`` on both the stock and the packed ``rows``
    path, so this wall time is the gather plus ~0.3 ms of hash evaluation and is
    directly comparable between configurations.
    """
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpNGramEmbedding
    if getattr(Qwen4ExpNGramEmbedding, "_bench_timed", False):
        return
    original = Qwen4ExpNGramEmbedding.__call__

    def timed(self, input_ids, cache):
        t0 = time.perf_counter()
        out = original(self, input_ids, cache)
        mx.eval(out)
        PLE_T["calls"].append(time.perf_counter() - t0)
        PLE_T["rows"].append(int(getattr(self.ngram_embedding, "rows_read", 0)))
        return out

    Qwen4ExpNGramEmbedding.__call__ = timed
    Qwen4ExpNGramEmbedding._bench_timed = True


def ple_reset():
    PLE_T["calls"].clear()
    PLE_T["rows"].clear()


def ple_ms():
    return [t * 1e3 for t in PLE_T["calls"]]


# ------------------------------------------------------------------- config
class Config:
    """One combination of patches, applied and removed around a measurement."""

    def __init__(self, name, ple=False, norm=False, int8=False):
        self.name, self.ple, self.norm, self.int8 = name, ple, norm, int8

    def __repr__(self):
        return self.name


def apply(cfg, model, model_path):
    if cfg.ple:
        n = ple_patch.apply_ple_packed_patch(model, model_path, mode="rows", force=True)
        assert n == 1, f"expected 1 PLE layer patched, got {n}"
    if cfg.norm:
        assert norm_patch.apply_bf16_norm_patch(force=True)
    os.environ["OMLX_MOE_INT8_PREFILL"] = "1" if cfg.int8 else "0"


def unapply(cfg, model):
    if cfg.ple:
        ple_patch.remove_ple_packed_patch(model)
    if cfg.norm:
        norm_patch.remove_bf16_norm_patch()
    os.environ["OMLX_MOE_INT8_PREFILL"] = "0"


# --------------------------------------------------------------- measurement
def forward_body(lm, ids):
    cache = lm.make_cache()
    h = lm.model(ids, cache=cache)
    mx.eval(h)
    del h, cache


def measure(lm, ids, reps=REPS, cool=COOL, warm=1):
    """Minimum of ``reps`` whole-body prefills after a cooldown, with the
    per-chunk PLE lookup time of the fastest run."""
    mx.synchronize()
    mx.clear_cache()
    time.sleep(cool)
    for _ in range(warm):
        forward_body(lm, ids)
    mx.synchronize()
    best, best_ple, ts, ples = None, None, [], []
    for _ in range(reps):
        ple_reset()
        mx.synchronize()
        t0 = time.perf_counter()
        forward_body(lm, ids)
        mx.synchronize()
        dt = time.perf_counter() - t0
        p = sum(ple_ms())
        ts.append(dt * 1e3)
        ples.append(p)
        if best is None or dt * 1e3 < best:
            best, best_ple = dt * 1e3, p
    return dict(body_ms=best, all_ms=ts, ple_ms=best_ple, ple_all_ms=ples,
                tok_s=ids.shape[1] / (best / 1e3))


def paired(lm, ids, cfg, model, model_path, log):
    """Fresh baseline, then the variant, then report the delta between them."""
    guard(f"before {cfg.name}")
    base_cfg = Config("base")
    apply(base_cfg, model, model_path)
    b = measure(lm, ids)
    unapply(base_cfg, model)

    apply(cfg, model, model_path)
    v = measure(lm, ids)
    unapply(cfg, model)

    row = dict(config=cfg.name, baseline=b, variant=v,
               speedup=b["body_ms"] / v["body_ms"],
               body_delta_ms=b["body_ms"] - v["body_ms"],
               ple_delta_ms=b["ple_ms"] - v["ple_ms"],
               mem=mem_snapshot())
    print(f"  {cfg.name:12s} body {v['body_ms']:8.1f} ms "
          f"({v['tok_s']:7.1f} tok/s)  vs paired base {b['body_ms']:8.1f} ms "
          f"({b['tok_s']:7.1f} tok/s)  -> {row['speedup']:.3f}x "
          f"({row['body_delta_ms']:+.1f} ms) | PLE lookup "
          f"{v['ple_ms']:6.1f} vs {b['ple_ms']:6.1f} ms")
    log.append(row)
    return row


# ----------------------------------------------------------------- cold ids
def cold_ids_probe(lm, model, model_path, cfg, draws=3):
    """The gather cost on token ids whose rows are not in the file cache.

    Each draw is a fresh random chunk, so every draw is genuinely cold; the
    minimum over draws is reported, matching the protocol used elsewhere.
    """
    apply(cfg, model, model_path)
    body, lookup, rows = [], [], []
    for d in range(draws):
        guard(f"cold-ids {cfg.name} draw {d}")
        mx.random.seed(int(time.time() * 1000) % 2**31 + d * 7919)
        ids = mx.random.randint(1000, 60000, (1, TOKENS))
        mx.eval(ids)
        mx.synchronize()
        mx.clear_cache()
        time.sleep(COOL)
        ple_reset()
        t0 = time.perf_counter()
        forward_body(lm, ids)
        mx.synchronize()
        body.append((time.perf_counter() - t0) * 1e3)
        lookup.append(sum(ple_ms()))
        rows.append(sum(PLE_T["rows"]))
        del ids
        mx.clear_cache()
    unapply(cfg, model)
    out = dict(config=cfg.name, body_ms=min(body), lookup_ms=min(lookup),
               body_all=body, lookup_all=lookup, rows=rows[0])
    print(f"  {cfg.name:12s} cold-ids body {out['body_ms']:8.1f} ms, "
          f"PLE lookup {out['lookup_ms']:7.1f} ms  (draws: "
          + " ".join(f"{x:.0f}" for x in lookup) + ")")
    return out


# --------------------------------------------------------------- long prefill
def long_prefill(lm, ids, chunk=2048, lookahead=True):
    """oMLX's PromptProcessingBatch chunk loop, with the PLE gather-ahead."""
    cache = lm.make_cache()
    n = ids.shape[1]
    times = []
    ple_reset()
    t_all = time.perf_counter()
    for c in range(0, n, chunk):
        cur = ids[:, c:c + chunk]
        nxt = ids[:, c + chunk:c + 2 * chunk]
        t0 = time.perf_counter()
        if lookahead and nxt.shape[1]:
            lm.prefetch_ple(nxt, cur)
        h = lm.model(cur, cache=cache)
        mx.eval([e.state for e in cache])
        mx.eval(h)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        del h
        mx.clear_cache()
    total = (time.perf_counter() - t_all) * 1e3
    del cache
    mx.clear_cache()
    return dict(total_ms=total, tok_s=n / (total / 1e3), chunk_ms=times,
                ple_lookup_ms=sum(ple_ms()))


# ------------------------------------------------------------------- stages
def stage_chunk(lm, model, model_path, ids, results, int8_ok):
    print(f"\n=== per-chunk prefill, {TOKENS} tokens, warm table, "
          f"min of {REPS} after {COOL}s cooldown, paired baselines ===")
    log = []
    for cfg in (Config("+ple", ple=True),
                Config("+norm", norm=True),
                Config("+ple+norm", ple=True, norm=True)):
        paired(lm, ids, cfg, model, model_path, log)
    if int8_ok:
        for cfg in (Config("+int8", int8=True),
                    Config("all", ple=True, norm=True, int8=True)):
            paired(lm, ids, cfg, model, model_path, log)
    results["chunk"] = log


def stage_cold(lm, model, model_path, results):
    print(f"\n=== PLE table gather on cold ids ({TOKENS}-token chunk, "
          f"min of 3 fresh draws) ===")
    out = [cold_ids_probe(lm, model, model_path, Config("base")),
           cold_ids_probe(lm, model, model_path, Config("+ple", ple=True))]
    tables = ple_patch.packed_tables()
    if 1 in tables and tables[1].mode == "rows":
        t = tables[1]
        out[1]["pages_read"] = t.pages_read
        out[1]["pread_mb"] = t.pages_read * 16384 / 1e6
        out[1]["pread_ms_total"] = t.pread_seconds * 1e3
        print(f"    packed rows mode: {t.pages_read:,} pages "
              f"({out[1]['pread_mb']:.0f} MB) preaded in total, "
              f"{out[1]['pread_ms_total']:.0f} ms")
    results["cold_ids"] = out


def stage_long(lm, model, model_path, results, int8_ok):
    n = LONG_TOKENS
    print(f"\n=== {n}-token multi-chunk prefill (oMLX chunk loop with PLE "
          f"lookahead), page cache warm ===")
    mx.random.seed(4242)
    ids = mx.random.randint(1000, 60000, (1, n))
    mx.eval(ids)
    cfgs = [Config("base"),
            Config("all", ple=True, norm=True, int8=int8_ok)]
    print("  warming the page cache (one untimed pass per configuration)")
    for cfg in cfgs:
        apply(cfg, model, model_path)
        long_prefill(lm, ids)
        unapply(cfg, model)
        guard("long warmup")
    out = {}
    runs = {c.name: [] for c in cfgs}
    for r in range(LONG_REPS):
        for cfg in cfgs:
            guard(f"long {cfg.name} rep {r}")
            apply(cfg, model, model_path)
            mx.synchronize()
            time.sleep(COOL)
            res = long_prefill(lm, ids)
            unapply(cfg, model)
            runs[cfg.name].append(res)
            print(f"  rep {r} {cfg.name:6s} {res['total_ms']/1e3:7.2f} s "
                  f"{res['tok_s']:7.1f} tok/s  PLE lookup "
                  f"{res['ple_lookup_ms']:7.0f} ms  "
                  f"chunk1 {res['chunk_ms'][0]:.0f} ms, "
                  f"steady {np.median(res['chunk_ms'][1:]):.0f} ms")
    for name, rs in runs.items():
        best = min(rs, key=lambda r: r["total_ms"])
        out[name] = dict(best=best, all_total_ms=[r["total_ms"] for r in rs],
                         tok_s=best["tok_s"],
                         steady_chunk_ms=float(np.median(best["chunk_ms"][1:])))
    out["speedup"] = out["base"]["best"]["total_ms"] / out["all"]["best"]["total_ms"]
    print(f"  -> {out['base']['tok_s']:.1f} tok/s -> {out['all']['tok_s']:.1f} tok/s "
          f"({out['speedup']:.3f}x)")
    results["long"] = out
    del ids
    mx.clear_cache()


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*",
                    default=["chunk", "cold", "long"])
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--force-int8", action="store_true",
                    help="install the int8 MoE kernel even though its 11.3 GB "
                         "of tables push the footprint past the memory budget")
    args = ap.parse_args()

    os.environ["OMLX_MOE_INT8_PREFILL"] = "0"
    from load_omlx import load, MODEL

    results = {"model": MODEL, "tokens": TOKENS, "reps": REPS, "cool_s": COOL,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "mem_budget_gb": MEM_BUDGET_GB}
    print(f"loading {MODEL} (PLE mmap)")
    model, _ = load(ple_mode="mmap")
    lm = model.language_model
    install_ple_timer(model)
    base_mem = mem_snapshot()
    results["mem_after_load"] = base_mem
    print(f"  after load: RSS {base_mem['rss_gb']:.1f} GB, mlx active "
          f"{base_mem['mlx_active_gb']:.1f} GB, system free+inactive "
          f"{base_mem['sys_free_gb']+base_mem['sys_inactive_gb']:.1f} GB")

    mx.random.seed(7)
    ids = mx.random.randint(1000, 60000, (1, TOKENS))
    mx.eval(ids)

    # one untimed pass to warm the file cache for these ids and to compile
    forward_body(lm, ids)
    mx.synchronize()
    mx.clear_cache()
    warm_mem = mem_snapshot()
    results["mem_after_first_chunk"] = warm_mem
    print(f"  after one chunk: RSS {warm_mem['rss_gb']:.1f} GB, "
          f"mlx peak {warm_mem['mlx_peak_gb']:.1f} GB")

    # ---- int8 headroom decision
    int8_ok = False
    int8_note = ""
    if args.no_int8:
        int8_note = "disabled on the command line"
    else:
        # ps RSS badly under-reports this process: the 72.8 GB of weights are
        # file-backed mmap pages that macOS does not charge to the task, so RSS
        # reads ~16 GB.  mx.get_active_memory() is the honest live-array figure
        # and mx.get_peak_memory() covers the activation transient.
        footprint = max(warm_mem["rss_gb"], warm_mem["mlx_peak_gb"])
        projected = footprint + 11.3
        if projected > MEM_BUDGET_GB and not args.force_int8:
            int8_note = (f"SKIPPED: RSS {warm_mem['rss_gb']:.1f} GB + 11.3 GB of "
                         f"int8 tables = {projected:.1f} GB > the "
                         f"{MEM_BUDGET_GB:.0f} GB budget")
            print(f"  int8 MoE gather: {int8_note}")
        else:
            # both patch modules are literally named "patch"; load the MoE one
            # from its path so it cannot collide with ple-fix/patch.py
            spec = importlib.util.spec_from_file_location(
                "moe_int8_patch",
                os.path.expanduser("~/inference-server/kernels/moe-int8/patch.py"))
            moe = importlib.util.module_from_spec(spec)
            sys.modules["moe_int8_patch"] = moe
            spec.loader.exec_module(moe)
            globals()["moe_patch"] = moe
            print(f"  int8 MoE gather: projected {projected:.1f} GB <= "
                  f"{MEM_BUDGET_GB:.0f} GB budget; installing")
            moe.install()
            os.environ["OMLX_MOE_INT8_PREFILL"] = "1"
            t0 = time.time()
            n = moe.warmup(model)
            # force the tables to exist before reading RSS, and confirm the
            # shape gate actually routes on this model
            forward_body(lm, ids)
            mx.synchronize()
            routed = moe.stats()
            os.environ["OMLX_MOE_INT8_PREFILL"] = "0"
            print(f"    routing check: {routed}")
            results["int8_routing"] = routed
            after = mem_snapshot()
            results["mem_after_int8_warmup"] = after
            int8_note = (f"installed: {n} expert tensors prepared in "
                         f"{time.time()-t0:.1f}s, RSS "
                         f"{warm_mem['rss_gb']:.1f} -> {after['rss_gb']:.1f} GB")
            print(f"    {int8_note}")
            if after["rss_gb"] > MEM_BUDGET_GB:
                print("    over budget after warmup; dropping the tables")
                moe.clear_cache()
                moe.uninstall()
                int8_note += " -- then dropped, over budget"
            else:
                int8_ok = True
            guard("after int8 warmup")
    results["int8"] = dict(enabled=int8_ok, note=int8_note)

    try:
        if "chunk" in args.stages:
            stage_chunk(lm, model, MODEL, ids, results, int8_ok)
        if "cold" in args.stages:
            stage_cold(lm, model, MODEL, results)
        if "long" in args.stages:
            stage_long(lm, model, MODEL, results, int8_ok)
    except MemoryError as exc:
        print(f"\n!! {exc}")
        results["aborted"] = str(exc)
    finally:
        results["mem_final"] = mem_snapshot()
        if int8_ok:
            moe = globals().get("moe_patch")
            if moe is not None:
                moe.clear_cache()
                moe.uninstall()
        for t in ple_patch.packed_tables().values():
            t.close()
        gc.collect()
        mx.clear_cache()
        out = os.path.join(HERE, "combined_bench.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=1, default=str)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
