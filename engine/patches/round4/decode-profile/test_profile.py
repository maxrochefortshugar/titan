#!/usr/bin/env python3
"""Synthetic harness for the decode profiler.

Builds a miniature qwen4_exp-shaped model (6 decoder layers: 4 "GDN", 2
"attention", a MoE block each, one PLE layer with an n-gram table, a 512-row
vocabulary head and an MTP head) and a miniature ``batch_generator`` module
registered under the REAL module name, whose ``_run_verify_cycle_chain``
mirrors the shipped cycle order: backbone verify forward -> one host sync for
acceptance -> rollback -> draft chain.  The profiler is then installed exactly
as it would be in production and asked to profile that cycle.

Peak GPU is a few MB.  Run:  kdev/bin/python test_profile.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------- fake model
def build_model(mx, nn):
    H, V, E = 64, 512, 16

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.random.normal((E, H, H)).astype(mx.bfloat16)

        def __call__(self, x, target_verify=False):
            return (x @ self.w[0]) + (x @ self.w[1])

    class NGramTable:
        rows_read = 0

        def __call__(self, ids):
            self.rows_read = int(ids.size)
            return mx.zeros((*ids.shape, 8), dtype=mx.bfloat16)

    class NGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.ngram_embedding = NGramTable()

        def __call__(self, input_ids, cache):
            ids = mx.broadcast_to(input_ids[..., None], (*input_ids.shape, 4))
            emb = self.ngram_embedding(ids)
            return emb.reshape(*emb.shape[:-2], -1)

    class PLE(nn.Module):
        def __init__(self):
            super().__init__()
            self.ple_embedding = NGram()
            self.proj = nn.Linear(32, H, bias=False)

        def __call__(self, h, input_ids, cache, mask, target_verify=False):
            return self.proj(self.ple_embedding(input_ids, cache).astype(h.dtype))

    class Layer(nn.Module):
        def __init__(self, is_linear, with_ple):
            super().__init__()
            self.is_linear = is_linear
            self.mlp = MoE()
            self.a = nn.Linear(H, H, bias=False)
            if with_ple:
                self.ple = PLE()

        def __call__(self, h, input_ids, mask=None, cache=None, position_ids=None,
                     gdn_sink=None, target_verify=False):
            if "ple" in self:
                h = h + self.ple(h, input_ids, cache, mask)
            h = h + self.a(h)
            for _ in range(3 if self.is_linear else 1):
                h = h + 0.001 * self.mlp(h)
            return h

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(V, H)
            self.layers = [Layer(i % 3 != 2, i == 1) for i in range(6)]

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(H, V, bias=False)
            self.mtp_layer = nn.Linear(H, H, bias=False)

        def __call__(self, inputs, cache=None, return_hidden=False, n_confirmed=0):
            h = self.model.embed_tokens(inputs)
            for layer, c in zip(self.model.layers, cache or [None] * 6):
                h = layer(h, inputs, cache=c)
            return self.lm_head(h), h, ["gdn_state"]

        def mtp_forward(self, hidden, ids, mtp_cache, return_hidden=False,
                        logits_keep=0):
            h = self.mtp_layer(hidden) + self.model.embed_tokens(ids)
            if logits_keep and h.shape[1] > logits_keep:
                h = h[:, -logits_keep:]
            logits = self.lm_head(h)
            return (logits, h) if return_hidden else logits

        @staticmethod
        def _restore_ple_state(cache, snapshot, accepted):
            return None

        def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                       block_size):
            for c in caches:
                c.offset -= (block_size - accepted - 1)
            LM._restore_ple_state(caches[0], None, [accepted])
            return True

    return LM()


class Cache:
    def __init__(self, offset=4096):
        self.offset = offset

    def nbytes(self):
        return self.offset * 1024


# ------------------------------------------------- fake batch_generator module
_BG_SOURCE = """
import mlx.core as mx


def _call_backbone(model, inputs, cache, n_confirmed=0):
    return model(inputs, cache=cache, return_hidden=True, n_confirmed=n_confirmed)


def _chain_rollback(model, prompt_cache, accepted, num_drafts, gdn_states=None):
    return model.rollback_speculative_cache(
        prompt_cache, gdn_states, accepted, num_drafts + 1)


def _clear_rollback(prompt_cache):
    return None


def _chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf):
    m = gen_batch.model
    logits, h = m.mtp_forward(hidden_rows, committed[None, :], state.mtp_cache,
                              return_hidden=True, logits_keep=1)
    toks = []
    for j in range(state.depth):
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        toks.append(tok)
        if j + 1 == state.depth:
            break
        logits, h = m.mtp_forward(h[:, -1:], tok[None, :], state.mtp_cache,
                                  return_hidden=True)
    state.drafts = mx.concatenate(toks)
    mx.async_eval(state.drafts)


def _run_verify_cycle_chain(gen_batch, state):
    k = int(state.drafts.shape[0])
    inputs = mx.concatenate([state.next_main, state.drafts])
    logits, hidden, gdn_states = _call_backbone(
        gen_batch.model, inputs[None, :], gen_batch.prompt_cache, n_confirmed=1)
    rows = logits[0]
    targets = mx.argmax(rows, axis=-1).astype(mx.int32)
    matches = (targets[:k] == state.drafts.astype(mx.int32)).astype(mx.int32)
    m_arr = mx.cumprod(matches).sum().reshape(1)
    host = mx.concatenate([m_arr, targets]).tolist()   # the one host sync
    m = int(host[0])
    state.stats.accepts += m
    state.stats.cycles += 1
    if m == k:
        _clear_rollback(gen_batch.prompt_cache)
    else:
        _chain_rollback(gen_batch.model, gen_batch.prompt_cache, m, k, gdn_states)
    committed = mx.concatenate(
        [state.drafts[:m], mx.array([int(host[1 + m])], dtype=mx.uint32)])
    _chain_next_drafts(gen_batch, state, hidden[:, : m + 1], committed, None)
    state.next_main = committed[-1:]
    for c in gen_batch.prompt_cache:
        c.offset += m + 1


def _log_mtp_stats(uid, stats, finish_reason):
    return None
"""


def install_fake_bg(mx, model, depth=3):
    """Register a miniature batch_generator under the real module name.

    The body is exec'd into the module namespace so every intra-module call
    goes through a module-global lookup, exactly as the shipped file does and
    exactly what the profiler's function swaps rely on.
    """
    pkg = types.ModuleType("omlx")
    sub = types.ModuleType("omlx.patches")
    sub2 = types.ModuleType("omlx.patches.mlx_lm_mtp")
    bg = types.ModuleType("omlx.patches.mlx_lm_mtp.batch_generator")
    exec(compile(_BG_SOURCE, "<fake batch_generator>", "exec"), bg.__dict__)
    sub2.batch_generator = bg
    for name, mod in (("omlx", pkg), ("omlx.patches", sub),
                      ("omlx.patches.mlx_lm_mtp", sub2),
                      ("omlx.patches.mlx_lm_mtp.batch_generator", bg)):
        sys.modules[name] = mod
    return bg


class Stats:
    def __init__(self):
        self.accepts = 0
        self.cycles = 0


class State:
    def __init__(self, mx, depth):
        self.uid = "test-uid"
        self.depth = depth
        self.stats = Stats()
        self.mtp_cache = []
        self.next_main = mx.array([7], dtype=mx.uint32)
        self.drafts = mx.array([1, 2, 3], dtype=mx.uint32)


class Batch:
    def __init__(self, model, cache):
        self.model = model
        self.prompt_cache = cache


# ------------------------------------------------------------------- modes
def run_mode(mode, cycles=12):
    import importlib.util

    import mlx.core as mx
    import mlx.nn as nn

    outdir = HERE / "_testout" / mode
    os.environ["OMLX_DECODE_PROFILE_DIR"] = str(outdir)
    if mode != "off":
        os.environ["OMLX_DECODE_PROFILE"] = "1"
    os.environ["OMLX_DECODE_PROFILE_SYNC"] = "1" if mode == "sync" else "0"

    model = build_model(mx, nn)
    bg = install_fake_bg(mx, model)
    before = {
        "eval": mx.eval, "async_eval": mx.async_eval,
        "tolist": mx.array.tolist, "item": mx.array.item,
        "cycle": bg._run_verify_cycle_chain, "backbone": bg._call_backbone,
        "drafts": bg._chain_next_drafts, "log": bg._log_mtp_stats,
        "layer": type(model.model.layers[0]).__call__,
        "head": type(model.lm_head).__call__,
        "mtp": type(model).mtp_forward,
    }

    spec = importlib.util.spec_from_file_location("dprof", HERE / "patch.py")
    prof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prof)
    ok = prof.install()
    ok2 = prof.install()

    after_untouched = all(
        before[k] is {
            "eval": mx.eval, "async_eval": mx.async_eval,
            "tolist": mx.array.tolist, "item": mx.array.item,
            "cycle": bg._run_verify_cycle_chain, "backbone": bg._call_backbone,
            "drafts": bg._chain_next_drafts, "log": bg._log_mtp_stats,
            "layer": type(model.model.layers[0]).__call__,
            "head": type(model.lm_head).__call__,
            "mtp": type(model).mtp_forward,
        }[k]
        for k in before
    )

    state = State(mx, depth=3)
    batch = Batch(model, [Cache() for _ in range(6)])
    import time as _t
    t0 = _t.perf_counter()
    for _ in range(cycles):
        bg._run_verify_cycle_chain(batch, state)
    wall = (_t.perf_counter() - t0) * 1000.0
    bg._log_mtp_stats(state.uid, state.stats, "length")

    files = sorted(outdir.glob("*.json")) if outdir.is_dir() else []
    payload = json.loads(files[-1].read_text()) if files else None
    print(json.dumps({
        "mode": mode, "install": ok, "install_idempotent": ok2,
        "untouched": after_untouched, "wall_ms": wall,
        "cycles_recorded": (payload or {}).get("cycles", 0),
        "payload_file": files[-1].name if files else None,
        "median": (payload or {}).get("median"),
        "first_record": (payload or {}).get("records", [{}])[0],
    }))


# -------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("--mode"):
        return run_mode(sys.argv[2])
    py = sys.executable
    results = {}
    for mode in ("off", "on", "sync"):
        out = subprocess.run([py, __file__, "--mode", mode],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(out.stdout, out.stderr)
            raise SystemExit(f"mode {mode} produced no result")
        results[mode] = json.loads(line[-1])

    checks = []
    off = results["off"]
    checks.append(("[0] flag off: install() returns False",
                   off["install"] is False))
    checks.append(("[0] flag off: nothing wrapped", off["untouched"] is True))
    checks.append(("[0] flag off: no JSON written", off["payload_file"] is None))

    for mode in ("on", "sync"):
        r = results[mode]
        checks.append((f"[1] {mode}: install() True and idempotent",
                       r["install"] is True and r["install_idempotent"] is True))
        checks.append((f"[1] {mode}: wrappers installed",
                       r["untouched"] is False))
        checks.append((f"[2] {mode}: per-request JSON written",
                       r["payload_file"] is not None))
        rec = r["first_record"]
        phase_sum = sum(rec["phase_ms"].values())
        err = abs(phase_sum - rec["cycle_ms"]) / max(rec["cycle_ms"], 1e-9) * 100
        checks.append((f"[3] {mode}: phases sum to cycle wall "
                       f"({phase_sum:.3f} vs {rec['cycle_ms']:.3f} ms, "
                       f"{err:.4f}% off)", err < 1.0))
        sub = rec["sub_ms"]
        covered = (sub.get("bb_gdn", 0) + sub.get("bb_attn", 0)
                   + sub.get("lm_head", 0) + sub.get("embed", 0))
        cov = covered / max(rec["phase_ms"]["verify_dispatch"], 1e-9) * 100
        checks.append((f"[4] {mode}: layer+head cover the verify forward "
                       f"({cov:.1f}% of it)", 80.0 <= cov <= 100.5))
        n_layers = rec["sub_n"].get("bb_gdn", 0) + rec["sub_n"].get("bb_attn", 0)
        checks.append((f"[5] {mode}: all 6 layers timed ({n_layers})",
                       n_layers == 6))
        checks.append((f"[5] {mode}: MoE + PLE + n-gram split present",
                       {"bb_moe", "bb_ple", "ngram_lookup"} <= set(sub)))
        checks.append((f"[5] {mode}: n-gram rows counted "
                       f"({rec.get('rows_ngram')})", rec.get("rows_ngram", 0) > 0))
        checks.append((f"[6] {mode}: draft head split into step 1 and 2..k "
                       f"(n={rec['sub_n'].get('draft.mtp_head1')}/"
                       f"{rec['sub_n'].get('draft.mtp_headk')})",
                       rec["sub_n"].get("draft.mtp_head1") == 1
                       and rec["sub_n"].get("draft.mtp_headk") == 2))
        acc = [k for k in rec["sync_ms"] if k.startswith("accept.")]
        checks.append((f"[7] {mode}: the acceptance host sync is timed "
                       f"({acc})", any("tolist" in k for k in acc)))
        checks.append((f"[7] {mode}: bytes model populated",
                       rec["bytes"]["experts"] > 0 and rec["bytes"]["kv"] > 0))

    slow = results["sync"]["wall_ms"] / max(results["on"]["wall_ms"], 1e-9)
    checks.append((f"[8] sync mode is slower ({slow:.2f}x) and is "
                   "attribution-only", slow > 1.0))
    over = results["on"]["wall_ms"] / max(results["off"]["wall_ms"], 1e-9)
    print(f"\nfake-model wall per {12} cycles: off "
          f"{results['off']['wall_ms']:.1f} ms, on "
          f"{results['on']['wall_ms']:.1f} ms ({over:.2f}x), sync "
          f"{results['sync']['wall_ms']:.1f} ms ({slow:.2f}x)")
    print("median stage table (on):")
    med = results["on"]["median"]
    print("  cycle %.3f ms" % med["cycle_ms"])
    for k, v in sorted(med["phase_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    phase {k:16s} {v:7.3f} ms")
    for k, v in sorted(med["sub_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    sub   {k:16s} {v:7.3f} ms")
    for k, v in sorted(med["sync_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    sync  {k:24s} {v:7.3f} ms")
    print(f"    syncs/cycle {med['syncs_per_cycle']:.1f}")

    print()
    bad = 0
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
        bad += 0 if ok else 1
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
