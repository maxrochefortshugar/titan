# Evidence for REPORT: mx.mean over a bf16 axis carries a bf16 accumulator.
import mlx.core as mx
HDR = open("kernel.py").read().split('_HEADER = r"""')[1].split('"""')[0]
SRC = r"""
    const uint i = thread_position_in_grid.x;
    const size_t b = (size_t)i * HC;
    float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    T bacc = T(0.0f);
    for (int g = 0; g < HC; ++g) {
        const float gate = hc2_sigmoid<T>(float(up[b+g]));
        a1 += float(T(gate * float(xn[b+g])));      // round product, fp32 acc
        a2 += gate * float(xn[b+g]);                // pure fp32
        bacc = T(float(bacc) + float(T(gate * float(xn[b+g]))));  // bf16 acc
    }
    o1[i] = T(a1 / float(HC));
    o2[i] = T(a2 / float(HC));
    o3[i] = T(float(bacc) / float(HC));
    o4[i] = T(a1 * (1.0f/float(HC)));
"""
k = mx.fast.metal_kernel(name="probe3", input_names=["up","xn"],
    output_names=["o1","o2","o3","o4"], header=HDR, source=SRC, ensure_row_contiguous=True)
n = 1<<15; HC=4
up = (mx.random.normal((n,HC))*3.0).astype(mx.bfloat16)
xn = (mx.random.normal((n,HC))*1.0).astype(mx.bfloat16)
o = k(inputs=[up,xn], template=[("T",mx.bfloat16),("HC",HC)], grid=(n,1,1), threadgroup=(256,1,1),
      output_shapes=[(n,)]*4, output_dtypes=[mx.bfloat16]*4)
ref = mx.mean(mx.sigmoid(up)*xn, axis=-1)
ref2 = (mx.sigmoid(up)*xn).sum(axis=-1)/4
mx.eval(o, ref, ref2)
def ob(a):
    u=a.astype(mx.bfloat16).view(mx.uint16).astype(mx.int32); s=u>>15; m=u&0x7fff
    return mx.where(s==1,-m,m)
def ulps(a,b):
    d=mx.abs(ob(a)-ob(b)).astype(mx.float32); return int(mx.max(d).item()), float(mx.mean(d).item())
for nm,v in zip(["round-prod fp32acc","pure fp32","bf16 acc","fp32acc *0.25"], o):
    print(f"{nm:22s}", ulps(v, ref))
print("sum/4 vs mean       ", ulps(ref2, ref))
