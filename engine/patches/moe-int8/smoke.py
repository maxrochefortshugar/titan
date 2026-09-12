import mlx.core as mx, sys
sys.path.insert(0, '~/inference-server/kernels/moe-int8')
import kernel as K
E, N, Kd, R = 8, 128, 128, 40
w = (mx.random.normal((E, N, Kd))*0.05).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4)
idx = mx.sort(mx.random.randint(0, E, (R,)).astype(mx.uint32))
x = (mx.random.normal((R,1,Kd))*0.5).astype(mx.bfloat16)
mx.eval(wq,s,b,idx,x)
ref = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=4, sorted_indices=False)
out = K.gather_qmm_sorted(x, wq, s, b, idx)
mx.eval(ref,out)
d = (out.astype(mx.float32)-ref.astype(mx.float32))
print("shape", out.shape, "max abs", mx.abs(d).max().item(), "ref rms", (ref.astype(mx.float32)**2).mean().item()**0.5)
