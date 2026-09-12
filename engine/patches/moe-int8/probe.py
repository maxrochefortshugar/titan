import mlx.core as mx
H = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""
def probe(TM, TN, KK=64):
    src = f"""
    const uint lane = thread_position_in_threadgroup.x;
    constexpr auto d = matmul2d_descriptor({TM}, {TN}, {KK}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, metal::execution_simdgroup> op;
    auto ct = op.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>, int32_t>();
    uint C = ct.get_capacity();
    if (lane == 0) cap[0] = C;
    for (uint i = 0; i < C; ++i) {{
        auto ix = ct.get_multidimensional_index(i);
        out[(lane * 64 + i) * 2 + 0] = ix[0];
        out[(lane * 64 + i) * 2 + 1] = ix[1];
    }}
    """
    k = mx.fast.metal_kernel(name=f"probe{TM}_{TN}", input_names=[], output_names=["out","cap"],
                             header=H, source=src)
    out, cap = k(inputs=[], output_shapes=[(32*64*2,), (1,)], output_dtypes=[mx.int32, mx.int32],
                 grid=(32,1,1), threadgroup=(32,1,1))
    mx.eval(out, cap)
    return out.tolist(), cap.item()

for (tm, tn) in [(32,32),(16,32),(32,64),(64,32),(16,64),(8,32),(48,32)]:
    try:
        o, c = probe(tm, tn)
        per = [(o[(0*64+i)*2], o[(0*64+i)*2+1]) for i in range(c)]
        rows = sorted(set(m for n,m in per)); cols = sorted(set(n for n,m in per))
        # check slot-rank uniformity across lanes
        ok = True
        ref = [(cols.index(n), rows.index(m)) for n,m in per]
        for L in range(1,32):
            p = [(o[(L*64+i)*2], o[(L*64+i)*2+1]) for i in range(c)]
            r_ = sorted(set(m for n,m in p)); c_ = sorted(set(n for n,m in p))
            if [(c_.index(n), r_.index(m)) for n,m in p] != ref: ok=False; break
        print(f"TM={tm} TN={tn}: cap={c} distinct rows={len(rows)} cols={len(cols)} uniform={ok} lane0 rows={rows} cols={cols}")
    except Exception as e:
        print(f"TM={tm} TN={tn}: FAIL {str(e)[:160]}")
