#!/usr/bin/env python3
"""Is the MTP verify MoE path leaking to separate gate/up gathers?

AUDIT-2026-09-12.md section D item 5 flags mlx_vlm/models/qwen3_5_moe/language.py:15-32
(_target_verify_switch_glu) as calling up_proj and gate_proj separately, i.e.
3 gathers per verify step instead of 2. oMLX's omlx/patches/qwen35_moe_gate_up.py
claims to close that at :146-167 (_make_patched_target_verify) via
_ensure_vlm_verify_patch(). This checks both halves synthetically:

  A. bit-exactness of the fused verify helper against the stock one at M=4
  B. the gather count and wall time, 3 gathers vs 2, at the verify shape

Run: ~/inference-server/kdev/bin/python test_verify_gate_up.py [--experts 128]
"""
import argparse, importlib.util, statistics as st, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

OMLX = Path("/Applications/oMLX.app/Contents/Resources/omlx/patches/qwen35_moe_gate_up.py")
CHAIN = 10


def load_gate_up():
    # the module imports omlx.scheduler for a cache drain helper; stub it
    import types
    for pkg in ("omlx", "omlx.patches"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
    sched = types.ModuleType("omlx.scheduler")
    sched._sync_and_clear_cache = lambda: None
    sys.modules["omlx.scheduler"] = sched
    spec = importlib.util.spec_from_file_location("omlx.patches.qwen35_moe_gate_up", OMLX)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["omlx.patches.qwen35_moe_gate_up"] = mod
    spec.loader.exec_module(mod)
    return mod


def stock_target_verify(switch_mlp, x, indices, target_verify):
    """Verbatim mlx_vlm 0.6.3 qwen3_5_moe/language.py:15-32."""
    if not (target_verify and x.ndim == 3 and x.shape[1] > 1):
        return switch_mlp(x, indices)
    B, T, D = x.shape
    k = indices.shape[-1]
    flat_x = mx.expand_dims(x.reshape(B * T, D), (-2, -3))
    flat_indices = indices.reshape(B * T, k)
    up = switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
    gate = switch_mlp.gate_proj(flat_x, flat_indices, sorted_indices=False)
    out = switch_mlp.down_proj(
        switch_mlp.activation(up, gate), flat_indices, sorted_indices=False)
    return out.squeeze(-2).reshape(B, T, k, -1)


def timeit(fn, iters=13, warm=3):
    def run():
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(*outs)
    for _ in range(warm):
        run()
    ts = []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); run(); mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128,
                    help="512 in the real model; lowered to stay under 2 GB")
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=64)
    a = ap.parse_args()

    gu = load_gate_up()
    mx.random.seed(3)
    mlp = SwitchGLU(a.dim, a.inter, a.experts)
    for name in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(mlp, name)
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.03).astype(mx.bfloat16)
    mx.eval(mlp.parameters())
    from mlx.nn.layers.quantized import quantize
    mlp = mlp
    import mlx.nn as nn
    nn.quantize(mlp, group_size=a.gs, bits=a.bits)
    mx.eval(mlp.parameters())

    x = (mx.random.normal((1, a.m, a.dim)) * 0.5).astype(mx.bfloat16)
    idx = mx.random.randint(0, a.experts, (1, a.m, a.topk))
    mx.eval(x, idx)

    ref = stock_target_verify(mlp, x, idx, True)
    mx.eval(ref)
    t_stock = timeit(lambda: stock_target_verify(mlp, x, idx, True))

    assert gu._can_fuse(mlp), "gate/up not fusable at this shape"
    gu._fuse_one(mlp)
    gu._ensure_call_patch()
    fused_fn = gu._make_patched_target_verify(stock_target_verify)
    got = fused_fn(mlp, x, idx, True)
    mx.eval(got)
    t_fused = timeit(lambda: fused_fn(mlp, x, idx, True))

    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    max_abs = float(mx.max(d).item())
    exact = bool(mx.all(got == ref).item())

    print(f"experts={a.experts} dim={a.dim} inter={a.inter} top_k={a.topk} "
          f"M={a.m} {a.bits}-bit gs{a.gs}")
    print(f"  A. bit-identical to the separate gate/up path : {exact} "
          f"(max abs {max_abs:.3e})")
    print(f"  B. 3 gathers (stock helper) : {t_stock*1e3:.3f} ms")
    print(f"     2 gathers (fused helper) : {t_fused*1e3:.3f} ms  "
          f"({t_stock/t_fused:.2f}x)")
    print("\nProduction state: omlx/engine/vlm.py:1978-1991 calls "
          "apply_qwen35_moe_gate_up_fusion on the loaded VLM model, which calls\n"
          "_ensure_vlm_verify_patch() (qwen35_moe_gate_up.py:164-177) and "
          "rebinds the module-global\n_target_verify_switch_glu that "
          "Qwen3_5MoeSparseMoeBlock.__call__ looks up at call time.\n"
          "The vendored qwen4_exp imports the CLASS "
          "(vendor/.../qwen4_exp/language.py:32), not the helper, so the\n"
          "rebind is live for qwen4_exp too. No leak, provided the fusion ran.")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
