# SPDX-License-Identifier: Apache-2.0
"""install()/uninstall(), the env gate, and precedence against the int8 patch."""
import importlib.util, os, sys
import mlx.core as mx

WS = "~/inference-server/kernels/round3/gather-ws/patch.py"
I8 = "~/inference-server/kernels/moe-int8/patch.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make(E=512, N=1280, Kd=2560, R=20480):
    w = (mx.random.normal((E, N, Kd)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    idx = mx.sort(mx.random.randint(0, E, (R,)).astype(mx.uint32))
    x = (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16)
    mx.eval(wq, s, b, idx, x)
    return wq, s, b, idx, x


def call(wq, s, b, idx, x):
    return mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                         group_size=64, bits=4, sorted_indices=True)


ws = load(WS, "ws_patch")
stock_fn = mx.gather_qmm
wq, s, b, idx, x = make()
ref = call(wq, s, b, idx, x); mx.eval(ref)

os.environ["OMLX_MOE_GATHER_WS"] = "0"
assert ws.install() is True
assert ws.install() is False, "install must be idempotent"
o = call(wq, s, b, idx, x); mx.eval(o)
assert ws.stats()["routed"] == 0, "gate off must not route"
print("gate off: not routed, bit-identical:",
      bool(mx.all(o.astype(mx.float32) == ref.astype(mx.float32)).item()))

os.environ["OMLX_MOE_GATHER_WS"] = "1"
o = call(wq, s, b, idx, x); mx.eval(o)
print("gate on : routed", ws.stats()["routed"],
      "bit-identical:", bool(mx.all(o.astype(mx.float32) == ref.astype(mx.float32)).item()))

# unsupported shapes must fall through: 8-bit, and a decode-sized row count
n0 = ws.stats()["fallback"]
w8 = mx.quantize((mx.random.normal((8, 128, 256)) * 0.02).astype(mx.bfloat16),
                 group_size=64, bits=8)
i8x = mx.sort(mx.random.randint(0, 8, (80,)).astype(mx.uint32))
x8 = (mx.random.normal((80, 1, 256)) * 0.5).astype(mx.bfloat16)
mx.eval(*w8, i8x, x8)
mx.eval(mx.gather_qmm(x8, w8[0], w8[1], w8[2], rhs_indices=i8x, transpose=True,
                      group_size=64, bits=8, sorted_indices=True))
print("8-bit tensor fell through:", ws.stats()["fallback"] > n0)

# precedence: int8 installed under us and enabled -> we hand the call down
i8 = load(I8, "i8_patch")
ws.uninstall()
assert i8.install() is True
assert ws.install() is True
os.environ["OMLX_MOE_INT8_PREFILL"] = "1"
n_to = ws.stats()["to_int8"]
o = call(wq, s, b, idx, x); mx.eval(o)
print("int8 enabled: handed down", ws.stats()["to_int8"] - n_to,
      "| int8 routed", i8.stats()["routed"])
os.environ["OMLX_MOE_INT8_PREFILL"] = "0"
n_r = ws.stats()["routed"]
o = call(wq, s, b, idx, x); mx.eval(o)
print("int8 disabled: ws routed", ws.stats()["routed"] - n_r)

ws.uninstall(); i8.uninstall()
print("uninstall restores stock:", mx.gather_qmm is stock_fn)
