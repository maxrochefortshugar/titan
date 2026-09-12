import os, sys, time
sys.path.insert(0,'~/inference-server/kernels/moe-int8')
os.environ["OMLX_MOE_INT8_PREFILL"]="1"
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
import patch, kernel as K

E, D, H, TOPK = 512, 2560, 640, 10
glu = SwitchGLU(D, H, E)
glu = glu  # quantize
nn.quantize(glu, group_size=64, bits=4)
mx.eval(glu.parameters())

def run(T):
    x = (mx.random.normal((T, D))*0.5).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (T, TOPK)).astype(mx.uint32)
    mx.eval(x, idx)
    ref = glu(x, idx); mx.eval(ref)
    patch.install()
    out = glu(x, idx); mx.eval(out)
    patch.uninstall()
    d = (out.astype(mx.float32)-ref.astype(mx.float32))
    rms = (mx.mean(ref.astype(mx.float32)**2).item())**0.5
    print(f"T={T:5d} routed={patch.stats()['routed']:3d} fallback={patch.stats()['fallback']:3d} "
          f"max abs {mx.abs(d).max().item():.5f}  RMS err {(mx.mean(d*d).item()**0.5)/rms*100:.4f}% of out RMS")

run(64)     # below min_tokens -> must fall back, bit-identical
run(512)
run(2048)
