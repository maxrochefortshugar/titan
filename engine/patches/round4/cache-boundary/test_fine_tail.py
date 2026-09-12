# SPDX-License-Identifier: Apache-2.0
"""End-to-end synthetic test for OMLX_CACHE_FINE_TAIL.

Drives the real oMLX cache classes (PagedCacheManager, PagedSSDCacheManager,
BlockAwarePrefixCache, BoundarySnapshotSSDStore and the scheduler's
_BoundarySnapshotProvider) with a fabricated two-layer cache: one ArraysCache
layer standing in for the 36 GDN layers and one KVCache layer standing in for
the 12 QSA layers. No model is loaded and no GDN kernel runs; tensors are a
few hundred KB.

Scenario per arm:
  prefill 10_167 tokens -> store -> new request of 13_084 tokens sharing the
  first 10_167 -> lookup -> reconstruct.

Asserts:
  (a) cached length is the fine-grid floor (9_728 patched vs 8_192 stock)
  (b) the restored GDN state and KV are bit-identical to a fresh prefill to
      that same boundary
  (c) the store keeps a boundary that the trailing-partial path can use, so
      no boundary_snapshot_unavailable / available_boundaries=0
  (d) prompts shorter than 2048 and exactly on a 2048 boundary are unchanged

Run:
  PYTHONPATH=/Applications/oMLX.app/Contents/Resources:\\
/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \\
    ~/inference-server/kdev/bin/python test_fine_tail.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.scheduler import (
    _BoundarySnapshotProvider,
    clamp_prefill_chunk_to_boundary,
    should_emit_prefill_boundary,
)

HERE = Path(__file__).resolve().parent
COARSE = 2048
FINE = 512
N_KV_HEADS, HEAD_DIM = 2, 8
GDN_DIM = 16

_spec = importlib.util.spec_from_file_location("cb_patch", HERE / "patch.py")
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)


# --------------------------------------------------------------------------
# A deterministic fake "model": every cache value is a pure function of the
# number of forwarded tokens, so a restored state can be compared bit for bit
# against a fresh prefill to the same length.
# --------------------------------------------------------------------------
def gdn_state(n: int) -> mx.array:
    """Stand-in for the recurrent state after n tokens. Deterministic in n."""
    idx = mx.arange(GDN_DIM * GDN_DIM, dtype=mx.float32).reshape(GDN_DIM, GDN_DIM)
    return mx.sin(idx * 0.013 + float(n) * 0.0007).astype(mx.float32)


def conv_state(n: int) -> mx.array:
    idx = mx.arange(3 * GDN_DIM, dtype=mx.float32).reshape(3, GDN_DIM)
    return mx.cos(idx * 0.021 + float(n) * 0.0011).astype(mx.bfloat16)


def kv_state(n: int) -> tuple[mx.array, mx.array]:
    """KV for tokens [0, n). Position t is a pure function of t."""
    t = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1)
    h = mx.arange(N_KV_HEADS, dtype=mx.float32).reshape(1, N_KV_HEADS, 1, 1)
    d = mx.arange(HEAD_DIM, dtype=mx.float32).reshape(1, 1, 1, HEAD_DIM)
    keys = mx.sin(t * 0.0003 + h * 0.7 + d * 0.05).astype(mx.bfloat16)
    values = mx.cos(t * 0.0004 + h * 0.9 + d * 0.03).astype(mx.bfloat16)
    return keys, values


def extracted_at(n: int) -> list[dict]:
    """The dict format store_cache and the snapshot store consume."""
    k, v = kv_state(n)
    return [
        {
            "cache_type": "ArraysCache",
            "class_name": "ArraysCache",
            "state": (gdn_state(n), conv_state(n)),
            "meta_state": (),
        },
        {
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "state": (k, v),
            "meta_state": (str(n),),
        },
    ]


class FakeModel:
    def make_cache(self):
        return [object(), object()]


# --------------------------------------------------------------------------
# Prefill simulation: the real clamp and emit predicates decide the chunk
# schedule and therefore which boundaries have a GDN snapshot.
# --------------------------------------------------------------------------
def prefill_schedule(
    total: int, block_size: int, step: int = COARSE, start: int = 0
) -> list[int]:
    """Return the token counts at which a boundary snapshot is emitted.

    ``start`` is the reused prefix length, so a warm turn schedules exactly the
    chunks the scheduler would run for its suffix.
    """
    import omlx.scheduler as sch

    clamp = sch.clamp_prefill_chunk_to_boundary
    emit = sch.should_emit_prefill_boundary
    processed, emitted, last = start, [], start - 1
    while processed < total:
        n = min(step, total - processed)
        n = clamp(n, cache_tokens=processed, block_size=block_size)
        processed += n
        if emit(
            total_tokens=processed, block_size=block_size, last_emitted_tokens=last
        ):
            emitted.append(processed)
            last = processed
    return emitted


class Arm:
    """One cache stack (stock or patched) on its own temp dir."""

    def __init__(self, root: Path, block_size: int):
        self.root = root
        self.block_size = block_size
        self.ssd = PagedSSDCacheManager(
            cache_dir=root / "blocks",
            max_size_bytes=4 << 30,
            expected_block_size_tokens=block_size,
            expected_kv_bytes_per_token=N_KV_HEADS * HEAD_DIM * 2 * 2,
            gdn_ssd_split_enabled=True,
        )
        self.paged = PagedCacheManager(
            block_size=block_size,
            max_blocks=100_000,
            model_name="synthetic-fine-tail",
            initial_blocks=256,
        )
        self.paged._paged_ssd_cache_manager = self.ssd
        self.cache = BlockAwarePrefixCache(
            FakeModel(), self.paged, self.ssd, gdn_ssd_split_enabled=True
        )
        self.snapshots = BoundarySnapshotSSDStore(root / "blocks")
        # Same wiring the scheduler does at scheduler.py:13554.
        self.cache.set_gdn_checkpoint_loader(
            self.snapshots.load_file,
            dequantization_counter=lambda: self.snapshots.gdn_state_dequantizations,
        )

    def close(self):
        try:
            self.snapshots.shutdown()
        except Exception:
            pass
        try:
            self.ssd.shutdown()
        except Exception:
            pass

    # -- the pieces the scheduler does around store_cache -------------------
    def prefill_and_store(self, rid: str, tokens: list[int]):
        """Emit snapshots on the real schedule, then store like the scheduler.

        Returns (stored_tokens, available_boundaries).
        """
        total = len(tokens)
        emitted = prefill_schedule(total, self.block_size)
        for tc in emitted:
            self.snapshots.save(
                rid,
                tc,
                [None, None],  # sliceable layers are skipped by the real path
                lambda _snap, _tc=tc: (extracted_at(_tc), None),
                block_size=self.block_size,
            )
        self.snapshots.flush() if hasattr(self.snapshots, "flush") else None

        # _get_boundary_store_override: only block-aligned snapshots count.
        valid = sorted(tc for tc in emitted if 0 < tc <= total and tc % self.block_size == 0)
        if not valid:
            return 0, 0
        latest = valid[-1]
        provider = _BoundarySnapshotProvider(
            store=self.snapshots,
            request_id=rid,
            valid_tcs=[tc for tc in valid if tc != latest],
            in_memory_snapshots={},
            paged_ssd_manager=self.ssd,
        )
        table = self.cache.store_cache(
            rid,
            tokens[:latest],
            extracted_at(latest),
            boundary_snapshots=provider,
            hot_cache_write_back=False,
        )
        return (table.num_tokens if table else 0), len(valid)

    def lookup_and_restore(self, rid: str, tokens: list[int]):
        table, _rest = self.cache.fetch_cache(rid, tokens)
        if table is None:
            return 0, None
        restored = self.cache.reconstruct_cache(table, promote_to_hot_cache=False)
        return table.num_tokens, restored


def bits_equal(a, b) -> bool:
    a, b = mx.array(a), mx.array(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(mx.all(a.view(mx.uint8) == b.view(mx.uint8)).item())


def state_of(cache_obj):
    st = getattr(cache_obj, "state", None)
    if st is None and hasattr(cache_obj, "_inner"):
        st = cache_obj._inner.state
    return st


def run_arm(
    name: str,
    block_size: int,
    patched: bool,
    prompt_a: int,
    prompt_b: int,
    keep_stock_commit: bool = False,
):
    """One arm. keep_stock_commit reproduces the round-3 attempt: fine block
    size and fine tail chunking, but the stock per-block checkpoint gate."""
    root = Path(tempfile.mkdtemp(prefix=f"fine-tail-{name}-"))
    if patched:
        os.environ["OMLX_CACHE_FINE_TAIL"] = str(FINE)
        os.environ["OMLX_CACHE_COARSE_CHUNK"] = str(COARSE)
        assert patch_mod.install(), "patch install failed"
        if keep_stock_commit:
            BlockAwarePrefixCache._commit_split_gdn_checkpoint = (
                BlockAwarePrefixCache._commit_split_gdn_checkpoint.__wrapped__
            )
    try:
        arm = Arm(root, block_size)
        tokens_a = [(i * 7919 + 13) % 100_000 for i in range(prompt_a)]
        tokens_b = tokens_a + [(i * 104_729 + 3) % 100_000 for i in range(prompt_b - prompt_a)]

        stored, boundaries = arm.prefill_and_store("req-a", tokens_a)
        cached, restored = arm.lookup_and_restore("req-b", tokens_b)
        result = {
            "stored": stored,
            "boundaries": boundaries,
            "cached": cached,
            "restored": restored,
        }
        arm.close()
        return result
    finally:
        if patched:
            patch_mod.uninstall()
            os.environ.pop("OMLX_CACHE_FINE_TAIL", None)
        shutil.rmtree(root, ignore_errors=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    PA, PB = 10_167, 13_084

    print(f"prompt A = {PA}, prompt B = {PB} (shares the first {PA})")

    print("\nstock (block_size=2048, no patch)")
    base = run_arm("stock", COARSE, False, PA, PB)
    print(f"  stored={base['stored']} boundaries={base['boundaries']} cached={base['cached']}")
    ok &= check("stock cached length is the 2048 floor", base["cached"] == 8192,
                f"got {base['cached']}")

    print("\nround-3 repro (block_size=512, fine tail chunk, stock checkpoint gate)")
    r3 = run_arm("r3", FINE, True, PA, PB, keep_stock_commit=True)
    print(f"  stored={r3['stored']} boundaries={r3['boundaries']} cached={r3['cached']}")
    ok &= check("round-3 store truncates and the warm turn caches nothing",
                r3["cached"] == 0, f"got {r3['cached']}")

    print("\npatched (block_size=512, OMLX_CACHE_FINE_TAIL=512)")
    fine = run_arm("fine", FINE, True, PA, PB)
    print(f"  stored={fine['stored']} boundaries={fine['boundaries']} cached={fine['cached']}")
    ok &= check("(a) cached length is the 512 floor of 10167", fine["cached"] == 9728,
                f"got {fine['cached']}")
    ok &= check("(a) that is 1536 tokens more than stock",
                fine["cached"] - base["cached"] == 1536)
    ok &= check("(c) a usable store boundary exists (no available_boundaries=0)",
                fine["boundaries"] > 0 and fine["stored"] == 9728,
                f"boundaries={fine['boundaries']} stored={fine['stored']}")

    # (b) bit-identical restore at the fine boundary.
    restored = fine["restored"]
    if restored is None:
        ok &= check("(b) restore returned cache objects", False)
    else:
        ref = extracted_at(9728)
        gdn_ok = bits_equal(state_of(restored[0])[0], ref[0]["state"][0]) and bits_equal(
            state_of(restored[0])[1], ref[0]["state"][1]
        )
        rk, rv = state_of(restored[1])
        ek, ev = ref[1]["state"]
        kv_ok = bits_equal(rk[:, :, :9728, :], ek) and bits_equal(rv[:, :, :9728, :], ev)
        ok &= check("(b) restored GDN state is bit-identical to a fresh prefill", gdn_ok)
        ok &= check("(b) restored KV is bit-identical to a fresh prefill", kv_ok,
                    f"restored kv len {rk.shape[2]}")

    # (d) nothing changes below 2048 or exactly on a 2048 boundary.
    print("\n(d) unchanged cases")
    for label, n in (("short prompt 1500", 1500), ("exact 2048", 2048), ("exact 4096", 4096)):
        s = prefill_schedule(n, COARSE)
        os.environ["OMLX_CACHE_FINE_TAIL"] = str(FINE)
        patch_mod.install()
        p = prefill_schedule(n, FINE)
        patch_mod.uninstall()
        os.environ.pop("OMLX_CACHE_FINE_TAIL", None)
        floor_s = (n // COARSE) * COARSE
        floor_p = (n // FINE) * FINE
        if n % COARSE == 0:
            ok &= check(f"{label}: same emitted boundaries", s == p, f"{s} vs {p}")
            ok &= check(f"{label}: same stored floor", floor_s == floor_p)
        else:
            # 1500 has no 2048 block at all; stock stores nothing either way.
            ok &= check(f"{label}: stock emits nothing", s == [], f"{s}")
            ok &= check(f"{label}: patched emits only the fine tail", p == [1024], f"{p}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
