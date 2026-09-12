import mlx.core as mx, numpy as np
HDR = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""
def mksrc(aptr,bptr,cptr,M,N,K,transb):
    bcast = "(device uchar*)" if "4b_format" in bptr else f"(device {bptr}*)"
    bext = f"dextents<int32_t,2>({K},{N})" if transb else f"dextents<int32_t,2>({N},{K})"
    return f"""
    constexpr auto desc = matmul2d_descriptor({M}, {N}, {K}, false, {str(transb).lower()}, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto tA = tensor<device {aptr}, dextents<int32_t,2>, tensor_inline>((device {aptr}*)A, dextents<int32_t,2>({K},{M}));
    auto tB = tensor<device {bptr}, dextents<int32_t,2>, tensor_inline>({bcast}B, {bext});
    auto tC = tensor<device {cptr}, dextents<int32_t,2>, tensor_inline>((device {cptr}*)out, dextents<int32_t,2>({N},{M}));
    op.run(tA, tB, tC);
"""
def go(name, aptr, bptr, cptr, ain, bin_, M, N, K, transb, odt, ref):
    try:
        k = mx.fast.metal_kernel(name=name, input_names=["A","B"], output_names=["out"],
                                 header=HDR, source=mksrc(aptr,bptr,cptr,M,N,K,transb))
        r = k(inputs=[ain, bin_], output_shapes=[(M*N,)], output_dtypes=[odt], grid=(32,1,1), threadgroup=(32,1,1))
        mx.eval(r)
        got = np.array(r[0]).reshape(M,N).astype(np.float64)
        print(f"{name:14s}: OK maxabserr={np.abs(got-ref).max():.4g}  got00={got[0,0]:.4f} ref00={ref[0,0]:.4f}")
    except Exception as e:
        print(f"{name:14s}: FAIL {type(e).__name__}: {str(e)[:700]}")

rng = np.random.default_rng(7); M,N,K = 16,32,64
A = rng.standard_normal((M,K)).astype(np.float32); B = rng.standard_normal((K,N)).astype(np.float32)
mA = mx.array(A).astype(mx.bfloat16); mB = mx.array(B).astype(mx.bfloat16)
Af = np.array(mA.astype(mx.float32)).astype(np.float64)
ref = Af @ np.array(mB.astype(mx.float32)).astype(np.float64)
go("bf16_nn","bfloat","bfloat","float", mA, mB, M,N,K, False, mx.float32, ref)
mBT = mx.array(np.ascontiguousarray(B.T)).astype(mx.bfloat16)
go("bf16_nt","bfloat","bfloat","float", mA, mBT, M,N,K, True, mx.float32, ref)

Ai = rng.integers(-100,100,(M,K)).astype(np.int8); Bi = rng.integers(-100,100,(K,N)).astype(np.int8)
go("i8_i8_nn","int8_t","int8_t","int32_t", mx.array(Ai), mx.array(Bi), M,N,K, False, mx.int32,
   (Ai.astype(np.int64)@Bi.astype(np.int64)).astype(np.float64))
BiT = np.ascontiguousarray(Bi.T)
go("i8_i8_nt","int8_t","int8_t","int32_t", mx.array(Ai), mx.array(BiT), M,N,K, True, mx.int32,
   (Ai.astype(np.int64)@Bi.astype(np.int64)).astype(np.float64))

Bu = rng.integers(0,16,(K,N)).astype(np.uint8)
pk = lambda X: (X[:, 0::2] | (X[:, 1::2] << 4)).astype(np.uint8)
go("i8_u4_nn","int8_t","metal::uint4b_format","int32_t", mx.array(Ai), mx.array(pk(Bu)), M,N,K, False, mx.int32,
   (Ai.astype(np.int64)@Bu.astype(np.int64)).astype(np.float64))
BuT = np.ascontiguousarray(Bu.T)
go("i8_u4_nt","int8_t","metal::uint4b_format","int32_t", mx.array(Ai), mx.array(pk(BuT)), M,N,K, True, mx.int32,
   (Ai.astype(np.int64)@Bu.astype(np.int64)).astype(np.float64))
go("u8_u4_nt","uint8_t","metal::uint4b_format","int32_t", mx.array(Ai.astype(np.uint8)), mx.array(pk(BuT)), M,N,K, True, mx.int32,
   (Ai.astype(np.uint8).astype(np.int64)@Bu.astype(np.int64)).astype(np.float64))
go("bf16_u4_nn","bfloat","metal::uint4b_format","float", mA, mx.array(pk(Bu)), M,N,K, False, mx.float32, Af@Bu.astype(np.float64))
go("bf16_u4_nt","bfloat","metal::uint4b_format","float", mA, mx.array(pk(BuT)), M,N,K, True, mx.float32, Af@Bu.astype(np.float64))
Bs = rng.integers(-8,8,(K,N)).astype(np.int8)
pks = lambda X: ((X[:,0::2].astype(np.uint8)&0xF) | ((X[:,1::2].astype(np.uint8)&0xF)<<4)).astype(np.uint8)
BsT = np.ascontiguousarray(Bs.T)
go("bf16_i4_nt","bfloat","metal::int4b_format","float", mA, mx.array(pks(BsT)), M,N,K, True, mx.float32, Af@Bs.astype(np.float64))
go("i8_i4_nt","int8_t","metal::int4b_format","int32_t", mx.array(Ai), mx.array(pks(BsT)), M,N,K, True, mx.int32,
   (Ai.astype(np.int64)@Bs.astype(np.int64)).astype(np.float64))
