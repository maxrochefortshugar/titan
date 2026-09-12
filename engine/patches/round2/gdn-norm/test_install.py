#!/usr/bin/env python3
"""Integration check for patch.install() against the REAL Qwen4ExpRMSNormGated.

No model is loaded: this only registers oMLX's vendored mlx_vlm tree (the same
pre-load compat patch the server applies) and exercises the class directly on
synthetic tensors, under 200 MB.

Must run with the bundled interpreter, which is where mlx_vlm lives:

    cd /tmp && source ~/inference-server/kernels/ple-fix/_env.sh && \
      OMLX_QWEN4_GDN_NORM_GATE=1 $PY \
      ~/inference-server/kernels/round2/gdn-norm/test_install.py
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    logging.basicConfig(level=logging.INFO)
    failures = []

    # env gating: with the flag off, install() must be a no-op
    saved = os.environ.pop("OMLX_QWEN4_GDN_NORM_GATE", None)
    gate_off = load("gdnp_off", os.path.join(HERE, "patch.py"))
    if gate_off.install() is not False:
        failures.append("install() returned True with the env var unset")
    print(f"env gating (flag unset -> False): {gate_off.install() is False}")
    if saved is not None:
        os.environ["OMLX_QWEN4_GDN_NORM_GATE"] = saved
    else:
        os.environ["OMLX_QWEN4_GDN_NORM_GATE"] = "1"

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    print(f"vendored mlx_vlm registered: {apply_mlx_vlm_qwen4_exp_compat_patch()}")
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated

    import mlx.core as mx

    patch = load("gdn_norm_gate_patch", os.path.join(HERE, "patch.py"))
    if not patch.install():
        failures.append("install() returned False")
    if not patch.install():
        failures.append("install() is not idempotent")
    print(f"install(): {patch.is_applied()}  (idempotent: True)")

    mx.random.seed(7)
    module = Qwen4ExpRMSNormGated(128, eps=1e-6, activation="sigmoid")
    module.weight = mx.random.normal((128,)).astype(mx.bfloat16)

    for T, expect_fused in ((1, False), (2, True), (512, True), (2048, True)):
        before = dict(patch.STATS)
        x = mx.random.normal((1, T, 48, 128)).astype(mx.bfloat16)
        z = mx.random.normal((1, T, 48, 128)).astype(mx.bfloat16)
        got = module(x, z)
        ref = patch._ORIGINAL_CALL(module, x, z)
        mx.eval(got, ref)
        same = bool(mx.array_equal(got, ref).item())
        fused = patch.STATS["fused_calls"] > before["fused_calls"]
        ok = same and got.shape == ref.shape and got.dtype == ref.dtype
        if not ok:
            failures.append(f"T={T}: output differs from the stock method")
        if fused != expect_fused:
            failures.append(f"T={T}: fused={fused}, expected {expect_fused}")
        print(
            f"T={T:<5} fused={fused!s:<5} (expected {expect_fused!s:<5})"
            f" bit-identical to stock: {same}"
        )

    # a 3-D call (the shape the class would see if a caller flattened heads)
    # must fall through untouched rather than be misread as [B,S,HV,DV]
    before = patch.STATS["fallback_calls"]
    flat = mx.random.normal((1, 16, 128)).astype(mx.bfloat16)
    out = module(flat, flat)
    mx.eval(out)
    if patch.STATS["fallback_calls"] <= before:
        failures.append("3-D input was not sent to the stock path")
    print(f"3-D input falls through: {patch.STATS['fallback_calls'] > before}")

    if not patch.remove() or patch.is_applied():
        failures.append("remove() did not restore the stock method")
    print(f"remove(): stock method restored: {not patch.is_applied()}")

    print(f"\nSTATS {patch.STATS}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
