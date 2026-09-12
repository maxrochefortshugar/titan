import mlx.core as mx
H = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""
TM, TN, BK = 48, 16, 64
src = f"""
  threadgroup bfloat Ws[{TN}*{BK}];
  const uint tid = thread_position_in_threadgroup.x;
  for (uint i = tid; i < {TN}*{BK}; i += 32) Ws[i] = b[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr auto d = matmul2d_descriptor({TM}, {TN}, {BK}, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<d, metal::execution_simdgroup> op;
  auto tA = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(
      (device bfloat*)a, dextents<int32_t,2>({BK}, {TM}), array<int32_t,2>{{1, {BK}}});
  auto tB = tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>(
      (threadgroup bfloat*)Ws, dextents<int32_t,2>({BK}, {TN}), array<int32_t,2>{{1, {BK}}});
  auto ct = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  op.run(tA, tB, ct);
  uint C = ct.get_capacity();
  if (tid==0) cap[0] = C;
  for (uint i = 0; i < C; ++i) {{
     auto ix = ct.get_multidimensional_index(i);
     y[(tid*32+i)] = ct[i];
     mi[(tid*32+i)*2+0] = ix[0];
     mi[(tid*32+i)*2+1] = ix[1];
  }}
"""
k = mx.fast.metal_kernel(name="probe_tg", input_names=["a","b"], output_names=["y","mi","cap"], header=H, source=src)
a = (mx.random.normal((TM,BK))*0.5).astype(mx.bfloat16)
b = (mx.random.normal((TN,BK))*0.5).astype(mx.bfloat16)
y, mi, cap = k(inputs=[a,b], output_shapes=[(32*32,),(32*32*2,),(1,)],
               output_dtypes=[mx.float32, mx.int32, mx.int32], grid=(32,1,1), threadgroup=(32,1,1))
mx.eval(y,mi,cap)
C = cap.item(); print("capacity", C)
ref = (a.astype(mx.float32) @ b.astype(mx.float32).T)
mi = mi.tolist(); y=y.tolist()
err = 0.0
for lane in range(32):
    for i in range(C):
        n = mi[(lane*32+i)*2+0]; m = mi[(lane*32+i)*2+1]
        err = max(err, abs(y[lane*32+i] - ref[m,n].item()))
print("max abs err vs fp32 ref:", err)
print("lane0 idx sample", [(mi[i*2],mi[i*2+1]) for i in range(min(C,8))])
