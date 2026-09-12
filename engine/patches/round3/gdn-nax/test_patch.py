"""install() against stub mlx_vlm modules: routing, idempotence, fallthrough,
and the precedence rule versus gdn-scan."""
import sys, os, types, importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "gdn-scan"))
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_kernel
from test_exact import make_inputs, err

# --- stub mlx_vlm.models.qwen3_5.{gated_delta, language} -------------------
calls = {"orig": 0}


def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log) * nn.softplus(a + dt_bias))


def _compute_g_beta(A_log, a, b, dt_bias):
    return compute_g(A_log, a, dt_bias), mx.sigmoid(b)


def orig_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    calls["orig"] += 1
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    return gated_delta_kernel(q, k, v, g.astype(mx.float32), beta.astype(mx.float32), state)


pkg = types.ModuleType("mlx_vlm"); pkg.__path__ = []
mods = types.ModuleType("mlx_vlm.models"); mods.__path__ = []
q35 = types.ModuleType("mlx_vlm.models.qwen3_5"); q35.__path__ = []
gd = types.ModuleType("mlx_vlm.models.qwen3_5.gated_delta")
lang = types.ModuleType("mlx_vlm.models.qwen3_5.language")
gd._compute_g_beta = _compute_g_beta
gd.gated_delta_update = orig_update
lang.gated_delta_update = orig_update
for n, m in [("mlx_vlm", pkg), ("mlx_vlm.models", mods), ("mlx_vlm.models.qwen3_5", q35),
             ("mlx_vlm.models.qwen3_5.gated_delta", gd),
             ("mlx_vlm.models.qwen3_5.language", lang)]:
    sys.modules[n] = m

os.environ["OMLX_QWEN4_GDN_SCAN_NAX"] = "1"
os.environ["OMLX_QWEN4_GDN_SCAN"] = "1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nax_patch = load(os.path.join(_HERE, "patch.py"), "_t_nax_patch")
print("install():", nax_patch.install())
print("idempotent:", nax_patch.install())
print("OMLX_QWEN4_GDN_SCAN now:", os.environ.get("OMLX_QWEN4_GDN_SCAN"))

scan_patch = load(os.path.join(_HERE, "..", "gdn-scan", "patch.py"), "_t_scan_patch")
print("gdn-scan install() after NAX (expect False):", scan_patch.install())

# --- routed prefill --------------------------------------------------------
T = 512
q, k, v, g, beta, st = make_inputs(T)
Hv = v.shape[-2]
A_log = mx.zeros((Hv,)); dt_bias = mx.zeros((Hv,))
a = mx.random.normal((1, T, Hv)); b = mx.random.normal((1, T, Hv))
gg, bb = _compute_g_beta(A_log, a, b, dt_bias)
gg = gg.astype(mx.float32); bb = bb.astype(mx.float32)
q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
y_ref, s_ref = gated_delta_kernel(q32, k32, v32, gg, bb, st); mx.eval(y_ref, s_ref)

before = calls["orig"]
y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, st); mx.eval(y, s)
print(f"prefill routed (orig not called): {calls['orig'] == before}"
      f"  y rrmse {err(y, y_ref)[2]:.3e}  state rrmse {err(s, s_ref)[2]:.3e}")

before = calls["orig"]
y, s = gd.gated_delta_update(q[:, :1], k[:, :1], v[:, :1], a[:, :1], b[:, :1], A_log, dt_bias, st)
mx.eval(y, s)
print("T=1 falls through:", calls["orig"] == before + 1)
before = calls["orig"]
y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, st, mask=mx.ones((1, T), mx.bool_))
mx.eval(y, s)
print("masked falls through:", calls["orig"] == before + 1)
