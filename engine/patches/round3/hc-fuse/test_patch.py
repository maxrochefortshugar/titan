#!/usr/bin/env python3
"""Install contract and composition with the deployed bf16 norm patch.

Builds a stand-in ``mlx_vlm.models.qwen4_exp`` package around the app's real
hc_fused module, then checks: the gate, idempotence, removal, ordering against
norm_patch.py in both directions, and that an installed block still returns
bit-identical results.
"""
import importlib.util, os, sys, types
from pathlib import Path
import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import GatedResidual, load_hc_fused, make_input, ulp_stats

hc_fused = load_hc_fused()
# stand-in package tree so `from mlx_vlm.models.qwen4_exp import hc_fused` works
for name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.qwen4_exp"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["mlx_vlm.models.qwen4_exp"].hc_fused = hc_fused

spec = importlib.util.spec_from_file_location("hc2_patch", HERE / "patch.py")
patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
spec = importlib.util.spec_from_file_location(
    "ple_norm_patch", Path.home() / "inference-server/kernels/ple-fix/norm_patch.py")
norm_patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(norm_patch)

stock = hc_fused.prefill_forward
ok = []
os.environ.pop("OMLX_QWEN4_HC_FUSE2", None)
ok.append(("gate off -> False", patch.install() is False
           and hc_fused.prefill_forward is stock))

os.environ["OMLX_QWEN4_HC_FUSE2"] = "1"
ok.append(("install -> True", patch.install() is True))
ok.append(("rebound", getattr(hc_fused.prefill_forward, "_omlx_hc_fuse2", False)))
ok.append(("idempotent", patch.install() is True))

# norm_patch applied afterwards must not displace us
os.environ["OMLX_QWEN4_BF16_NORM"] = "1"
mine = hc_fused.prefill_forward
norm_patch.apply_bf16_norm_patch()
ok.append(("norm_patch defers to us", hc_fused.prefill_forward is mine))

# results still bit identical to the deployed path
mod = GatedResidual(bits=5, seed=2)
x = make_input(1024, seed=8)
dep = norm_patch._make_prefill_forward(hc_fused)
a = hc_fused.prefill_forward(mod, x)
b = dep(mod, x)
mx.eval(a, b)
ok.append(("installed output bit identical",
           ulp_stats(a[0], b[0])[2] == 0.0 and ulp_stats(a[2], b[2])[2] == 0.0))

ok.append(("remove restores stock", patch.remove() and hc_fused.prefill_forward is stock))

for name, good in ok:
    print(f"{'ok  ' if good else 'FAIL'} {name}")
print("PASS" if all(g for _, g in ok) else "FAIL")
raise SystemExit(0 if all(g for _, g in ok) else 1)
