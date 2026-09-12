#!/usr/bin/env python3
"""The four installs: preconditions, idempotency, and equivalence of the hooks.

Nothing here loads the model or needs mlx_vlm.  Peak allocation ~120 MB (one
248,320-row head at a cut-down width for item 1).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MTP = os.path.expanduser("~/inference-server/kernels/round2/mtp/patch.py")
V = 248320


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def item1():
    print("item 1: MTP shortlist fused top-K")
    mtp = load("round2_import_0", MTP)          # the name bootstrap.py uses
    import patch as P

    os.environ.pop("OMLX_MTP_SHORTLIST_FASTTOPK", None)
    assert P.install_mtp_fasttopk() is False, "must be off by default"
    print("  flag off                     -> install False   ok")

    os.environ["OMLX_MTP_SHORTLIST_FASTTOPK"] = "1"
    ok = P.install_mtp_fasttopk()
    print(f"  flag on                      -> install {ok}")
    assert ok
    assert P.install_mtp_fasttopk() is True
    print("  second call (idempotent)     -> install True    ok")

    stock = mtp._refresh_shortlist._omlx_original
    fast = mtp._refresh_shortlist

    # A head whose rows are cheap to gather: the real 248,320 rows, 256 wide.
    kw = 256 * 4 // 32
    head = types.SimpleNamespace(
        weight=mx.random.randint(0, 2 ** 31 - 1, (V, kw), dtype=mx.uint32),
        scales=(mx.random.normal((V, 4)) * 0.01).astype(mx.bfloat16),
        biases=(mx.random.normal((V, 4)) * 0.01).astype(mx.bfloat16),
        group_size=64,
        bits=4,
    )
    mx.eval(head.weight, head.scales, head.biases)

    allok = True
    for dtype in (mx.float32, mx.bfloat16):
        logits = (mx.random.normal((1, V)) * 6.0).astype(dtype)
        mx.eval(logits)
        for k in (512, 2048, 4096):
            a, b = mtp._Shortlist(), mtp._Shortlist()
            stock(mx, a, head, logits, k)
            fast(mx, b, head, logits, k)
            mx.eval(a.ids, b.ids)
            sa, sb = set(a.ids.tolist()), set(b.ids.tolist())
            same_ids = sa == sb
            va = mx.sort(logits.reshape(-1)[a.ids])
            vb = mx.sort(logits.reshape(-1)[b.ids])
            mx.eval(va, vb)
            same_vals = bool(mx.all(va == vb).item())
            rows = bool(mx.all(b.weight == head.weight[b.ids]).item())
            good = same_vals and rows and b.ids.dtype == a.ids.dtype
            allok &= good
            print(f"  {str(dtype).split('.')[-1]:<9} k={k:<5} ids equal={same_ids} "
                  f"values equal={same_vals} gathered rows ok={rows} "
                  f"{'PASS' if good else 'FAIL'}")
    print(f"  proxy stats: {P.mtp_fasttopk.stats() if hasattr(P, 'mtp_fasttopk') else ''}"
          f"{sys.modules['mtp_fasttopk'].stats()}")
    return allok


def item2():
    print("\nitem 2: PLE reader worker count")
    import ple_workers as W

    fake = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
    fake.__file__ = "<fake>"
    from concurrent.futures import ThreadPoolExecutor

    original = ThreadPoolExecutor(max_workers=48, thread_name_prefix="ple-io")
    fake._PLE_IO_POOL = original
    sys.modules["mlx_vlm.models.qwen4_exp.language"] = fake

    os.environ.pop("OMLX_PLE_READ_WORKERS", None)
    W._STATE.clear()
    assert W.install() is True and fake._PLE_IO_POOL is original
    print("  unset                        -> stock pool kept (48)   ok")

    W._STATE.clear()
    os.environ["OMLX_PLE_READ_WORKERS"] = "16"
    assert W.install() is True
    swapped = fake._PLE_IO_POOL is not original
    n = fake._PLE_IO_POOL._max_workers
    print(f"  OMLX_PLE_READ_WORKERS=16     -> swapped={swapped} max_workers={n}")
    assert W.install() is True, "idempotent"
    assert W.uninstall() is True and fake._PLE_IO_POOL is original
    print("  uninstall                    -> stock pool restored    ok")
    del sys.modules["mlx_vlm.models.qwen4_exp.language"]
    os.environ.pop("OMLX_PLE_READ_WORKERS", None)
    original.shutdown(wait=False)
    return swapped and n == 16


def item3():
    print("\nitem 3: MTP verify weighted sum")
    import wsum_verify as W
    import patch as P

    os.environ.pop("OMLX_WSUM_TOPK10_VERIFY", None)
    W._PATCHED = False
    assert P.install_wsum_verify() is False
    print("  flag off                     -> install False   ok")
    os.environ["OMLX_WSUM_TOPK10_VERIFY"] = "1"
    W._PATCHED = False
    got = P.install_wsum_verify()
    print(f"  flag on, mlx_vlm absent      -> install {got} (stock path kept)")
    print(f"  kernel self-check            -> {W._self_check()}")
    os.environ.pop("OMLX_WSUM_TOPK10_VERIFY", None)
    return W._self_check() and got is False


def item4():
    print("\nitem 4: int8 down-projection tables")
    import patch as P
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    import mlx.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            # 2 "layers", real widths, 4 experts instead of 512
            self.gate_up = QuantizedSwitchLinear(2560, 1280, 4, bits=4, group_size=64)
            self.down = QuantizedSwitchLinear(640, 2560, 4, bits=4, group_size=64)

    m = Tiny()
    got = P.down_table_bytes(m)
    want = 3 * 4 * (640 // 64) * 2560 * 2
    print(f"  down tables for 4 experts x1 down tensor: {got/1e6:.1f} MB "
          f"(expected {want/1e6:.1f} MB) {'ok' if got == want else 'FAIL'}")

    os.environ.pop(P.ENV_DOWN, None)
    assert P.install_moe_int8_down(m) is False
    os.environ[P.ENV_DOWN] = "1"
    os.environ["OMLX_MOE_INT8_PREFILL"] = "1"
    os.environ[P.ENV_DOWN_SOFT_GB] = "0.000001"      # force the guard to bite
    refused = P.install_moe_int8_down(m) is False
    kept = os.environ.get("OMLX_MOE_INT8_SKIP_DOWN")
    print(f"  soft-limit guard             -> refused={refused}, "
          f"OMLX_MOE_INT8_SKIP_DOWN untouched={kept is None}")
    for v in (P.ENV_DOWN, P.ENV_DOWN_SOFT_GB, "OMLX_MOE_INT8_PREFILL"):
        os.environ.pop(v, None)
    return got == want and refused and kept is None


def main():
    ok = item1()
    ok &= item2()
    ok &= item3()
    ok &= item4()
    print("\nALL PASS" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
