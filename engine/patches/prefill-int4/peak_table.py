"""Focused peak: best config per precision, short runs."""
import time, mlx.core as mx, numpy as np
import peak_lib as peak2

CFGS = {
 "bf16 x bf16 -> f32":  dict(atype="bfloat", btype="bfloat", ctype="float", astore="bfloat", bstore="bfloat"),
 "f16  x f16  -> f32":  dict(atype="half", btype="half", ctype="float", astore="half", bstore="half"),
 "int8 x int8 -> i32":  dict(atype="int8_t", btype="int8_t", ctype="int32_t", astore="char", bstore="char"),
 "uint8x uint8-> i32":  dict(atype="uint8_t", btype="uint8_t", ctype="int32_t", astore="uchar", bstore="uchar"),
 "bf16 x uint4-> f32":  dict(atype="bfloat", btype="metal::uint4b_format", ctype="float", astore="bfloat", bstore="uchar"),
 "bf16 x int4 -> f32":  dict(atype="bfloat", btype="metal::int4b_format", ctype="float", astore="bfloat", bstore="uchar"),
 "int8 x int4 -> i32":  dict(atype="int8_t", btype="metal::int4b_format", ctype="int32_t", astore="char", bstore="uchar"),
 "uint8x uint4-> i32":  dict(atype="uint8_t", btype="metal::uint4b_format", ctype="int32_t", astore="uchar", bstore="uchar"),
 "f32  x f32  -> f32":  dict(atype="float", btype="float", ctype="float", astore="float", bstore="float"),
}
print(f"{'precision':22s} {'best tile':14s} {'ms':>8s} {'TFLOP/s':>9s}")
for name, cfg in CFGS.items():
    best = (0, None, 0)
    for (TM,TN,TK,NACC) in [(16,32,64,4),(16,32,64,2),(16,32,128,4),(32,32,64,2),(32,32,64,4),(16,32,128,2)]:
        try:
            t, tf = peak2.measure(tag=f"f{TM}_{TN}_{TK}_{NACC}_{abs(hash(name))%9999}", TM=TM,TN=TN,TK=TK,
                                  NSG=8, NACC=NACC, loops=384, scope_sgs=1, ntg=640, iters=3, **cfg)
            if tf > best[0]: best = (tf, f"{TM}x{TN}x{TK}/a{NACC}", t)
        except Exception as e:
            pass
    if best[1]:
        print(f"{name:22s} {best[1]:14s} {best[2]*1e3:8.2f} {best[0]:9.1f}")
    else:
        print(f"{name:22s} {'--':14s} {'':8s} {'unsupported':>9s}")
