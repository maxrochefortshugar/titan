"""Peak sweep: vary tile shape, scope, accumulators, occupancy."""
import time
import mlx.core as mx, numpy as np

HDR = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

def build(tag, atype, btype, ctype, astore, bstore, TM, TN, TK, NSG, NACC, loops, scope_sgs):
    ab = TM*TK
    bb = TK*TN//2 if "4b_format" in btype else TK*TN
    bcast = "(threadgroup uchar*)" if "4b_format" in btype else f"(threadgroup {btype}*)"
    scope = f"metal::execution_simdgroups<{scope_sgs}>" if scope_sgs > 1 else "metal::execution_simdgroup"
    accs = "".join(f"    auto acc{i} = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), {ctype}>();\n" for i in range(NACC))
    zero = "".join(f"acc{i}[i]=0; " for i in range(NACC))
    runs = "".join(f"        op.run(tA, tB, acc{i});\n" for i in range(NACC))
    sink = " + ".join(f"float(acc{i}[i])" for i in range(NACC))
    src = f"""
    threadgroup {astore} As[{ab}];
    threadgroup {bstore} Bs[{bb}];
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < {ab}; i += {32*NSG}) As[i] = ({astore})(A[i & 255]);
    for (uint i = tid; i < {bb}; i += {32*NSG}) Bs[i] = ({bstore})(Bp[i & 255]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor({TM}, {TN}, {TK}, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, {scope}> op;
    auto tA = tensor<threadgroup {atype}, dextents<int32_t,2>, tensor_inline>((threadgroup {atype}*)As, dextents<int32_t,2>({TK},{TM}));
    auto tB = tensor<threadgroup {btype}, dextents<int32_t,2>, tensor_inline>({bcast}Bs, dextents<int32_t,2>({TN},{TK}));
{accs}    #pragma unroll
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {{ {zero} }}
    for (uint l = 0; l < {loops}; ++l) {{
{runs}    }}
    float s = 0;
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) s += {sink};
    if (s == 1234.5678f) out[0] = s;
"""
    return mx.fast.metal_kernel(name=f"pk_{tag}", input_names=["A","Bp"], output_names=["out"], header=HDR, source=src)

def measure(**kw):
    ntg = kw.pop("ntg"); iters = kw.pop("iters", 5)
    k = build(**kw)
    NSG = kw["NSG"]
    A = mx.random.normal((256,)).astype(mx.float32); B = mx.random.normal((256,)).astype(mx.float32)
    call = lambda: k(inputs=[A,B], output_shapes=[(4,)], output_dtypes=[mx.float32],
                     grid=(32*NSG*ntg,1,1), threadgroup=(32*NSG,1,1))
    mx.eval(call()); mx.synchronize()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); outs=[call() for _ in range(10)]; mx.eval(outs); mx.synchronize()
        ts.append((time.perf_counter()-t0)/10)
    t=float(np.median(ts))
    # matmuls issued per threadgroup = loops*NACC, each covering TMxTNxTK, executed by scope_sgs simdgroups
    sgroups_per_tg = NSG
    tiles = ntg * (sgroups_per_tg // kw["scope_sgs"]) * kw["loops"] * kw["NACC"]
    macs = tiles * kw["TM"]*kw["TN"]*kw["TK"]
    return t, 2*macs/t/1e12

if __name__ == "__main__":
    CFG = dict(atype="bfloat", btype="bfloat", ctype="float", astore="bfloat", bstore="bfloat")
    print("--- bf16 sweep: tile / scope / accumulators / occupancy ---")
    for (TM,TN,TK,NSG,NACC,sgs,ntg) in [
        (16,32,64,8,4,1,640),(16,32,64,8,8,1,640),(16,32,64,4,8,1,640),
        (32,32,64,8,4,1,640),(16,64,64,8,4,1,640),(32,64,64,8,2,1,640),
        (64,32,64,8,4,4,640),(64,64,64,8,2,4,640),(32,32,32,8,8,1,640),
        (16,32,64,8,4,1,160),(16,32,64,8,4,1,2560),
    ]:
        loops = max(16, 2048*4//(NACC*max(1,TM*TN*TK//32768)))
        try:
            t,tf = measure(tag=f"s{TM}_{TN}_{TK}_{NSG}_{NACC}_{sgs}", TM=TM,TN=TN,TK=TK,NSG=NSG,NACC=NACC,
                           loops=loops, scope_sgs=sgs, ntg=ntg, **CFG)
            print(f"  tile {TM:3d}x{TN:3d}x{TK:3d} nsg={NSG} acc={NACC} scope={sgs} ntg={ntg:5d} loops={loops:5d}: {t*1e3:7.3f} ms  {tf:7.1f} TFLOP/s")
        except Exception as e:
            print(f"  tile {TM}x{TN}x{TK} nsg={NSG} acc={NACC} scope={sgs}: FAIL {str(e)[:200]}")
