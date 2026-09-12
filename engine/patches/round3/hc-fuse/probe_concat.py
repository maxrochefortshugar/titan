# Evidence for REPORT: a merged down|inject bank keeps the low-rank rows bit
# identical and moves the four injection rows.
import mlx.core as mx, mlx.nn as nn
def ob(a):
    u=a.astype(mx.bfloat16).view(mx.uint16).astype(mx.int32); s=u>>15; m=u&0x7fff
    return mx.where(s==1,-m,m)
def ulps(a,b):
    d=mx.abs(ob(a)-ob(b)).astype(mx.float32); return int(mx.max(d).item()), float(mx.mean(d).item())
K=10240; R=320; HC=4
for bits in (4,5,6,8):
  for M in (256, 2048):
    mx.random.seed(3)
    wd=(mx.random.normal((R,K))*0.02).astype(mx.bfloat16)
    wi=(mx.random.normal((HC,K))*0.02).astype(mx.bfloat16)
    dq,ds,db = mx.quantize(wd,64,bits); iq,isc,ib = mx.quantize(wi,64,bits)
    cq=mx.concatenate([dq,iq],0); cs=mx.concatenate([ds,isc],0); cb=mx.concatenate([db,ib],0)
    x=(mx.random.normal((M,K))).astype(mx.bfloat16)
    a=mx.quantized_matmul(x,dq,ds,db,transpose=True,group_size=64,bits=bits)
    b=mx.quantized_matmul(x,iq,isc,ib,transpose=True,group_size=64,bits=bits)
    c=mx.quantized_matmul(x,cq,cs,cb,transpose=True,group_size=64,bits=bits)
    mx.eval(a,b,c)
    print(f"bits={bits} M={M}: down rows {ulps(c[:,:R],a)}  inject rows {ulps(c[:,R:],b)}")
    del a,b,c,x,dq,ds,db,iq,isc,ib,cq,cs,cb,wd,wi
    mx.clear_cache()
