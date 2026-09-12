# SPDX-License-Identifier: Apache-2.0
"""Accuracy of the int8 x int4 sorted-gather kernel.

Three references per case:
  stock  - mx.gather_qmm(..., sorted_indices=True), bf16 activations x 4-bit weights
  fp32   - fp32 activations x fp32-dequantized weights, accumulated per expert in fp32
  bf16w  - (random cases only) the original unquantized bf16 weights, i.e. what the
           4-bit format itself costs the model before this kernel is involved

Reported: max abs error, max relative error against the fp32 reference's own scale,
and RMS error as a fraction of the reference output RMS (the number that matters).
"""
import sys, json, struct
import mlx.core as mx
sys.path.insert(0, '~/inference-server/kernels/moe-int8')
import kernel as K

MODEL = "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"


def fp32_ref(x, wq, s, b, idx, offsets, N):
    """Per-expert fp32 reference; avoids materialising all experts at once."""
    R = x.shape[0]
    out = mx.zeros((R, N), dtype=mx.float32)
    xf = x.astype(mx.float32)
    parts = []
    for e in range(wq.shape[0]):
        lo, hi = offsets[e], offsets[e + 1]
        if hi <= lo:
            continue
        w = mx.dequantize(wq[e], s[e], b[e], group_size=64, bits=4).astype(mx.float32)
        parts.append((lo, hi, xf[lo:hi] @ w.T))
        if len(parts) % 64 == 0:
            mx.eval([p[2] for p in parts])
    mx.eval([p[2] for p in parts])
    out = mx.concatenate([p[2] for p in parts], axis=0)
    return out


def errs(a, ref):
    a = a.astype(mx.float32); ref = ref.astype(mx.float32)
    d = a - ref
    rms_ref = mx.sqrt(mx.mean(ref * ref)).item()
    return (mx.abs(d).max().item(),
            (mx.abs(d).max() / max(mx.abs(ref).max().item(), 1e-30)).item(),
            (mx.sqrt(mx.mean(d * d)).item() / rms_ref) * 100.0,
            rms_ref)


def run_case(label, wq, s, b, N, Kd, T, E, xgen, with_bf16w=None, topk=10):
    R = T * topk
    idx = mx.sort(mx.random.randint(0, E, (R,)).astype(mx.uint32))
    x = xgen(R, Kd)
    mx.eval(idx, x)
    offsets = mx.zeros((E + 1,), dtype=mx.uint32)
    o, *_ = K.build_tiles(idx, E, 48, K.max_tiles(R, E, 48))
    mx.eval(o); offs = o.tolist()

    stock = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                          group_size=64, bits=4, sorted_indices=True).reshape(R, N)
    ours = K.gather_qmm_sorted(x, wq, s, b, idx).reshape(R, N)
    mx.eval(stock, ours)
    ref = fp32_ref(x.reshape(R, Kd), wq, s, b, idx, offs, N)
    mx.eval(ref)

    rows = []
    for nm, arr in (("stock vs fp32", stock), ("ours  vs fp32", ours)):
        ma, mr, rp, rr = errs(arr, ref)
        rows.append((nm, ma, mr, rp))
    ma, mr, rp, _ = errs(ours, stock)
    rows.append(("ours  vs stock", ma, mr, rp))
    if with_bf16w is not None:
        wref = with_bf16w(x.reshape(R, Kd), idx, offs)
        mx.eval(wref)
        for nm, arr in (("q4    vs bf16 weights", ref), ("ours  vs bf16 weights", ours)):
            ma, mr, rp, _ = errs(arr, wref)
            rows.append((nm, ma, mr, rp))
    print(f"\n{label}  T={T} R={R} N={N} K={Kd} E={E}   (ref RMS {rows[0][3] if False else ''})")
    print(f"  {'comparison':24s} {'max abs':>10s} {'max rel':>9s} {'RMS err % of out RMS':>22s}")
    for nm, ma, mr, rp in rows:
        print(f"  {nm:24s} {ma:10.5f} {mr*100:8.4f}% {rp:21.4f}%")
    return rows


def gen_normal(R, Kd):
    return (mx.random.normal((R, 1, Kd)) * 0.5).astype(mx.bfloat16)


def gen_outlier(R, Kd):
    x = mx.random.normal((R, 1, Kd)) * 0.5
    mask = (mx.arange(Kd) % 128 == 0).astype(mx.float32)
    return (x * (1.0 + 9.0 * mask)).astype(mx.bfloat16)


def gen_heavy(R, Kd):
    x = mx.random.normal((R, 1, Kd))
    return (x * mx.exp(mx.random.normal((R, 1, 1)) * 1.5) * 0.3).astype(mx.bfloat16)


def main():
    E, TOPK = 512, 10
    print("=" * 88)
    print("A. random weights, gate_up geometry  [512, 1280, 2560]")
    for wname, scale in (("normal weights", 0.02), ("heavy-tailed weights", None)):
        if scale is None:
            w = (mx.random.normal((E, 1280, 2560)) *
                 mx.exp(mx.random.normal((E, 1280, 1)) * 1.2) * 0.02).astype(mx.bfloat16)
        else:
            w = (mx.random.normal((E, 1280, 2560)) * scale).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        mx.eval(wq, s, b)

        def bf16w_ref(xf, idx, offs, w=w):
            parts = []
            for e in range(E):
                lo, hi = offs[e], offs[e + 1]
                if hi <= lo:
                    continue
                parts.append(xf[lo:hi].astype(mx.float32) @ w[e].astype(mx.float32).T)
                if len(parts) % 64 == 0:
                    mx.eval(parts)
            mx.eval(parts)
            return mx.concatenate(parts, axis=0)

        for T in (512, 2048):
            run_case(f"{wname}", wq, s, b, 1280, 2560, T, E, gen_normal,
                     with_bf16w=bf16w_ref if T == 512 else None)
        run_case(f"{wname}, outlier-channel activations", wq, s, b, 1280, 2560, 512, E, gen_outlier)
        run_case(f"{wname}, heavy-tailed activations", wq, s, b, 1280, 2560, 512, E, gen_heavy)
        del w, wq, s, b
        mx.clear_cache()
        K.clear_cache()

    print("=" * 88)
    print("B. real checkpoint tensors (Qwen3.8-Flash-Next-oQ4e, layer 3)")
    idx_map = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    for proj, N, Kd in (("gate_proj", 640, 2560), ("down_proj", 2560, 640)):
        base = f"language_model.model.layers.3.mlp.switch_mlp.{proj}"
        shard = idx_map[base + ".weight"]
        d = mx.load(f"{MODEL}/{shard}")
        wq, s, b = d[base + ".weight"], d[base + ".scales"], d[base + ".biases"]
        mx.eval(wq, s, b)
        del d
        for T, gen, nm in ((512, gen_normal, "normal acts"),
                           (2048, gen_normal, "normal acts"),
                           (512, gen_outlier, "outlier-channel acts")):
            run_case(f"real {proj} [{E},{N},{Kd}], {nm}", wq, s, b, N, Kd, T, E, gen)
        del wq, s, b
        mx.clear_cache()
        K.clear_cache()


main()
