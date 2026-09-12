# SPDX-License-Identifier: Apache-2.0
"""Ablation harness: builds the kernel with parts of the rescale removed.

Usage: ablate.py <name> <N> <K> <T> "[((TM,TN,SGM,SGN),mode), ...]"
modes: none (matmul only), foldnb (no affine bias term), fold (shipped), full.
Rounds are interleaved with stock so GPU contention hits both equally.
"""
"""Interleaved A/B: transposed [E,G,N] scale tables vs [E,N,G]."""
import time, sys
import mlx.core as mx
sys.path.insert(0,'~/inference-server/kernels/moe-int8')
import kernel as K
H=K._HEADER

def build(cfg,Kd,N,G,mode,trans,stageB):
    TM,TN,SGM,SGN=cfg
    C,NR,NC,ranks,rep_row,rep_col=K._probe_layout(TM,TN)
    TMT,TNT=TM*SGM,TN*SGN; NTH=32*SGM*SGN
    consts=(f"constant uchar RI[{C}] = {{{', '.join(str(r) for r,_ in ranks)}}};\n"
            f"constant uchar CI[{C}] = {{{', '.join(str(c) for _,c in ranks)}}};\n"
            f"constant uchar RREP[{NR}] = {{{', '.join(map(str,rep_row))}}};\n"
            f"constant uchar CREP[{NC}] = {{{', '.join(map(str,rep_col))}}};\n")
    off = f"(size_t)g*{N} + n0 + ncol[j]" if trans else f"(size_t)(n0+ncol[j])*{G} + g"
    if mode=="none":
        loads=""; body="acc[i] += float(ct[i]);"
    elif mode=="fold":
        loads=(f"#pragma unroll\nfor (uint j=0;j<{NC};++j) {{ size_t o={off}; sv[j]=float(sc_e[o]); bv[j]=float(bi_e[o]); qv[j]=sv[j]*(128.0f*float(qs_e[o])); }}\n"
               f"#pragma unroll\nfor (uint j=0;j<{NR};++j) {{ uint o=(m0+mrow[j])*{G}+g; xv[j]=xsc[o]; uv[j]=xrs[o]; }}")
        body=("acc[i] = fma(xv[RI[i]], fma(float(ct[i]), sv[CI[i]], -qv[CI[i]]), acc[i]);"
              "acc[i] = fma(uv[RI[i]], bv[CI[i]], acc[i]);")
    elif mode=="foldnb":
        loads=(f"#pragma unroll\nfor (uint j=0;j<{NC};++j) {{ size_t o={off}; sv[j]=float(sc_e[o]); qv[j]=sv[j]*(128.0f*float(qs_e[o])); }}\n"
               f"#pragma unroll\nfor (uint j=0;j<{NR};++j) {{ uint o=(m0+mrow[j])*{G}+g; xv[j]=xsc[o]; }}")
        body="acc[i] = fma(xv[RI[i]], fma(float(ct[i]), sv[CI[i]], -qv[CI[i]]), acc[i]);"
    else:
        loads=(f"#pragma unroll\nfor (uint j=0;j<{NC};++j) {{ size_t o={off}; sv[j]=float(sc_e[o]); bv[j]=float(bi_e[o]); qv[j]=128.0f*float(qs_e[o]); }}\n"
               f"#pragma unroll\nfor (uint j=0;j<{NR};++j) {{ uint o=(m0+mrow[j])*{G}+g; xv[j]=xsc[o]; uv[j]=xrs[o]; }}")
        body="acc[i] = fma(xv[RI[i]], (float(ct[i])-qv[CI[i]])*sv[CI[i]], acc[i]); acc[i] = fma(uv[RI[i]], bv[CI[i]], acc[i]);"
    if stageB:
        stage=f"""
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < {TNT*2}; p += {NTH}) {{
            uint row = p >> 1, hf = p & 1;
            ((threadgroup uint4*)Bs)[p] = *(const device uint4*)(bsrc + (size_t)row*{Kd//2} + hf*16);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
        btensor=f"""auto tB = tensor<threadgroup metal::uint4b_format, dextents<int32_t,2>, tensor_inline>(
            (threadgroup uchar*)Bs + sgn*{TN*32}, dextents<int32_t,2>(64,{TN}), array<int32_t,2>{{1,64}});"""
        ctA="threadgroup"; ctB="threadgroup"
    else:
        stage=""; ctA="device"; ctB="device"
        btensor=f"""auto tB = tensor<device metal::uint4b_format, dextents<int32_t,2>, tensor_inline>(
            (device uchar*)wq_e + ((size_t)n0*{Kd} + k0)/2, dextents<int32_t,2>(64,{TN}), array<int32_t,2>{{1,{Kd}}});"""
    return consts, f"""
    const uint t = threadgroup_position_in_grid.y;
    if (t >= ntiles[0]) return;
    const uint e = tile_exp[t]; const uint row0 = tile_row[t]; const uint rowend = offsets[e+1];
    const uint tid = thread_position_in_threadgroup.x; const uint sgid = tid/32;
    const uint sgm = sgid % {SGM}; const uint sgn = sgid / {SGM};
    const uint m0 = row0 + sgm*{TM};
    const uint n0tg = threadgroup_position_in_grid.x*{TNT};
    const uint n0 = n0tg + sgn*{TN};
    threadgroup uchar Bs[{TNT*32 if stageB else 4}];
    constexpr auto qdesc = matmul2d_descriptor({TM},{TN},64,false,true,false, matmul2d_descriptor::mode::multiply);
    matmul2d<qdesc, metal::execution_simdgroup> qop;
    auto ct0 = qop.get_destination_cooperative_tensor<
        tensor<device uchar, dextents<int32_t,2>, tensor_inline>,
        tensor<{ctB} metal::uint4b_format, dextents<int32_t,2>, tensor_inline>, int32_t>();
    ushort mrow[{NR}], ncol[{NC}];
    #pragma unroll
    for (uint j=0;j<{NR};++j) mrow[j]=ct0.get_multidimensional_index(RREP[j])[1];
    #pragma unroll
    for (uint j=0;j<{NC};++j) ncol[j]=ct0.get_multidimensional_index(CREP[j])[0];
    float acc[{C}];
    #pragma unroll
    for (uint i=0;i<{C};++i) acc[i]=0.0f;
    const device bfloat* sc_e=(const device bfloat*)scales+(size_t)e*{N}*{G};
    const device bfloat* bi_e=(const device bfloat*)biases+(size_t)e*{N}*{G};
    const device ushort* qs_e=(const device ushort*)qsum+(size_t)e*{N}*{G};
    const device uchar* wq_e=(const device uchar*)wq+(size_t)e*({N}*{Kd}/2);
    float sv[{NC}], bv[{NC}], qv[{NC}], xv[{NR}], uv[{NR}];
    (void)sv;(void)bv;(void)qv;(void)xv;(void)uv;
    for (uint g=0; g<{G}; ++g) {{
        const uint k0=g*64;
        const device uchar* bsrc = wq_e + ((size_t)n0tg*{Kd} + k0)/2; (void)bsrc;
        {stage}
        {loads}
        auto tA = tensor<device uchar, dextents<int32_t,2>, tensor_inline>(
            (device uchar*)xq + (size_t)m0*{Kd} + k0, dextents<int32_t,2>(64,{TM}), array<int32_t,2>{{1,{Kd}}});
        {btensor}
        auto ct = qop.get_destination_cooperative_tensor<decltype(tA),decltype(tB),int32_t>();
        qop.run(tA,tB,ct);
        #pragma unroll
        for (uint16_t i=0;i<{C};++i) {{ {body} }}
    }}
    #pragma unroll
    for (uint16_t i=0;i<{C};++i) {{
        uint m = m0+mrow[RI[i]];
        if (m < rowend) y[(size_t)m*{N}+n0+ncol[CI[i]]] = bfloat(acc[i]);
    }}
    """

E,TOPK=512,10
name,N,Kd,T=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
w=(mx.random.normal((E,N,Kd))*0.02).astype(mx.bfloat16)
wq,s,b=mx.quantize(w,group_size=64,bits=4); del w; mx.eval(wq,s,b)
R=T*TOPK; G=Kd//64
idx=mx.sort(mx.random.randint(0,E,(R,)).astype(mx.uint32))
x=(mx.random.normal((R,1,Kd))*0.5).astype(mx.bfloat16); mx.eval(idx,x)
qsum=K.compute_qsum(wq); mx.eval(qsum)
sT=mx.contiguous(mx.swapaxes(s,1,2)); bT=mx.contiguous(mx.swapaxes(b,1,2)); qT=mx.contiguous(mx.swapaxes(qsum,1,2))
mx.eval(sT,bT,qT); fl=2*R*N*Kd

import ast
CFGS=ast.literal_eval(sys.argv[5])
variants=[("stock",None)]
for cfg,mode in CFGS:
    TNT=cfg[1]*cfg[3]
    if N%TNT or 32*cfg[2]*cfg[3]>1024: continue
    variants.append((f"TM{cfg[0]} TN{cfg[1]} SGM{cfg[2]} SGN{cfg[3]} tile{cfg[0]*cfg[2]}x{TNT} {mode}",(cfg,mode,True,False)))
built={}
for nm,v in variants:
    if v is None: continue
    cfg,mode,trans,stageB=v
    TMT,TNT=cfg[0]*cfg[2],cfg[1]*cfg[3]; NTH=32*cfg[2]*cfg[3]
    maxt=K.max_tiles(R,E,TMT)
    offs,trow,texp,nt=K.build_tiles(idx,E,TMT,maxt); mx.eval(offs,trow,texp,nt)
    xq,xsc,xrs,xav=K.quantize_activations(x.reshape(R,Kd),TMT); mx.eval(xq,xsc,xrs,xav)
    c_,s_=build(cfg,Kd,N,G,mode,trans,stageB)
    kk=mx.fast.metal_kernel(name="v"+str(abs(hash(nm))%100000),
        input_names=["xq","xsc","xrs","wq","scales","biases","qsum","offsets","tile_row","tile_exp","ntiles"],
        output_names=["y"],header=H+c_,source=s_)
    S,B_,Q = (sT,bT,qT) if trans else (s,b,qsum)
    built[nm]=lambda kk=kk,xq=xq,xsc=xsc,xrs=xrs,S=S,B_=B_,Q=Q,offs=offs,trow=trow,texp=texp,nt=nt,NTH=NTH,TNT=TNT,maxt=maxt: \
        kk(inputs=[xq,xsc,xrs,wq,S,B_,Q,offs,trow,texp,nt],output_shapes=[(R,N)],output_dtypes=[mx.bfloat16],
           grid=(NTH*(N//TNT),maxt,1),threadgroup=(NTH,1,1))
built["stock"]=lambda: mx.gather_qmm(x,wq,s,b,rhs_indices=idx,transpose=True,group_size=64,bits=4,sorted_indices=True)

best={nm:1e9 for nm,_ in variants}
for rnd in range(9):
    for nm,_ in variants:
        f=built[nm]; o=f(); mx.eval(o)
        mx.synchronize(); t0=time.perf_counter()
        outs=[f() for _ in range(5)]; mx.eval(outs); mx.synchronize()
        best[nm]=min(best[nm],(time.perf_counter()-t0)/5)
st=best["stock"]
for nm,_ in variants:
    print(f"{nm:44s} min {best[nm]*1e3:7.3f} ms  {fl/best[nm]/1e12:5.1f} TOP/s  {st/best[nm]:.2f}x")
