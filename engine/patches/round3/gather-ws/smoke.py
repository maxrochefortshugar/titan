import sys, mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/round3/gather-ws')
import kernel as K
E, N, Kd, R = 8, 128, 256, 200
w = (mx.random.normal((E,N,Kd))*0.02).astype(mx.bfloat16)
wq,s,b = mx.quantize(w, group_size=64, bits=4)
idx = mx.sort(mx.random.randint(0,E,(R,)).astype(mx.uint32))
x = (mx.random.normal((R,1,Kd))*0.5).astype(mx.bfloat16)
mx.eval(wq,s,b,idx,x)
stock = mx.gather_qmm(x,wq,s,b,rhs_indices=idx,transpose=True,group_size=64,bits=4,sorted_indices=True).reshape(R,N)
ours = K.gather_qmm_sorted(x,wq,s,b,idx).reshape(R,N)
mx.eval(stock,ours)
d = (ours.astype(mx.float32)-stock.astype(mx.float32))
print("max abs", mx.abs(d).max().item(), "ref max", mx.abs(stock).max().item())
print("nz mismatch frac", (mx.abs(d)>0).astype(mx.float32).mean().item())
