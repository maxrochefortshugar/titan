#!/usr/bin/env python3
"""Exactness of the packed PLE lookup and the bf16 grouped norm.

  1. PLE table gather: stock SSD path vs packed ``rows`` vs packed
     ``resident``, on the real n-gram ids of a 2048-token chunk.  All three
     read the same nibbles/scales/biases into the same ``mx.dequantize``, so
     the requirement is *bitwise* equality, not a tolerance.
  2. Whole PLE layer output with each backend.
  3. Grouped RMSNorm: ``hc_fused._kernel_norm`` vs ``Qwen4ExpRMSNorm`` on a
     real hyper-connection input and on the live weights, reported in bf16
     ULPs, plus the full ``prefill_forward`` output.

Run: ``. ./_env.sh && $PY test_exact.py``
"""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/inference-server/profile-flashnext"))

import patch as ple_patch          # noqa: E402
import norm_patch                  # noqa: E402

TOKENS = int(os.environ.get("TEST_TOKENS", "2048"))


def bitwise_equal(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _ordinal(x):
    """Monotone integer ordering of bf16 bit patterns, so that adjacent
    representable values differ by 1 across zero and across the sign bit."""
    u = x.view(mx.uint16).astype(mx.int32)
    sign = u >= 32768
    return mx.where(sign, 32768 - u, u + 32768)


def ulp_report(ref, got, name):
    """Difference in bf16 ULPs, counted on the sign-corrected bit ordering."""
    d = mx.abs(_ordinal(ref) - _ordinal(got))
    mx.eval(d)
    exact = int(mx.sum(d == 0).item())
    total = int(d.size)
    rf, gf = ref.astype(mx.float32), got.astype(mx.float32)
    denom = mx.maximum(mx.abs(rf), 1e-30)
    rel = mx.max(mx.abs(rf - gf) / denom).item()
    out = dict(name=name, elements=total, bitwise_identical=exact,
               fraction_identical=exact / total,
               max_ulp=int(mx.max(d).item()),
               mean_ulp=float(mx.mean(d.astype(mx.float32)).item()),
               max_abs=float(mx.max(mx.abs(rf - gf)).item()),
               max_rel=float(rel))
    out["within_1_ulp"] = float(mx.mean((d <= 1).astype(mx.float32)).item())
    print(f"  {name:38s} identical {100*exact/total:7.3f}%  "
          f"<=1ULP {100*out['within_1_ulp']:7.3f}%  max {out['max_ulp']} ULP  "
          f"mean {out['mean_ulp']:.4f} ULP  max|abs| {out['max_abs']:.3e}")
    return out


def main():
    from load_omlx import load, MODEL

    results = {"model": MODEL, "tokens": TOKENS}
    print(f"loading {MODEL} (ple mode mmap)")
    model, processor = load(ple_mode="mmap")
    lm = model.language_model
    layer = next(l for l in lm.model.layers if "ple" in l)
    ple = layer.ple
    emb = ple.ple_embedding
    table = emb.ngram_embedding

    mx.random.seed(11)
    ids = mx.random.randint(1000, 60000, (1, TOKENS))
    mx.eval(ids)
    ids64 = ids.astype(mx.int64)
    prev = mx.full((1, emb.context_len), emb.eos_token_id, dtype=mx.int64)
    hist = mx.concatenate([prev, ids64], axis=-1)
    ngram_ids = emb._ngram_indices(hist, TOKENS)
    mx.eval(ngram_ids)
    print(f"n-gram ids {tuple(ngram_ids.shape)} -> "
          f"{int(np.prod(ngram_ids.shape)):,} rows, "
          f"range [{int(mx.min(ngram_ids).item())}, {int(mx.max(ngram_ids).item())}]")

    hidden = (mx.random.normal((1, TOKENS, lm.args.hc_count * lm.args.hidden_size))
              * 0.02).astype(mx.bfloat16)
    mx.eval(hidden)

    # ---------------------------------------------------------- 1. stock
    print("\n[1] stock SSD gather (reference)")
    t0 = time.perf_counter()
    ref = table(ngram_ids)
    mx.eval(ref)
    print(f"  {tuple(ref.shape)} {ref.dtype}, {(time.perf_counter()-t0)*1e3:.1f} ms (cold)")
    ref_layer = ple(hidden, ids, None, None)
    mx.eval(ref_layer)

    results["backends"] = {}

    # ---------------------------------------------------------- 2. rows
    for mode in ("rows", "resident"):
        print(f"\n[2] packed '{mode}' mode")
        n = ple_patch.apply_ple_packed_patch(model, MODEL, mode=mode, force=True)
        tbl = ple_patch.packed_tables()[1]
        print(f"  patched {n} layer(s); resident bytes {tbl.bytes_resident/1e9:.2f} GB; "
              f"load {tbl.load_seconds:.1f}s; "
              f"mlx active {mx.get_active_memory()/1e9:.1f} GB")
        got = table(ngram_ids)
        mx.eval(got)
        same = bitwise_equal(ref, got)
        got_layer = ple(hidden, ids, None, None)
        mx.eval(got_layer)
        same_layer = bitwise_equal(ref_layer, got_layer)
        print(f"  table gather bitwise identical to stock: {same}")
        print(f"  PLE layer output bitwise identical to stock: {same_layer}")
        entry = dict(mode=mode, gather_bitwise_identical=same,
                     layer_bitwise_identical=same_layer,
                     resident_gb=tbl.bytes_resident / 1e9,
                     load_seconds=tbl.load_seconds)
        if not same:
            entry["gather_ulp"] = ulp_report(ref, got, f"gather[{mode}]")
        results["backends"][mode] = entry
        ple_patch.remove_ple_packed_patch(model)
        # stock must still work after removal
        back = table(ngram_ids)
        mx.eval(back)
        print(f"  stock restored and still bitwise identical: {bitwise_equal(ref, back)}")
        results["backends"][mode]["stock_restored"] = bitwise_equal(ref, back)
        if mode == "rows":
            print(f"  rows-mode pages read {tbl.pages_read:,} "
                  f"({tbl.pages_read*16384/1e6:.0f} MB), "
                  f"pread {tbl.pread_seconds*1e3:.0f} ms")
            results["backends"][mode].update(pages_read=tbl.pages_read,
                                             pread_ms=tbl.pread_seconds * 1e3)

    # ---------------------------------------------------------- 3. norm
    print("\n[3] grouped RMSNorm: bf16 Metal kernel vs the fp32 round trip")
    from mlx_vlm.models.qwen4_exp import hc_fused
    module = lm.model.layers[0].attn_hyper_connection
    hc, hid = module.hc_count, module.hidden_size
    rows, width = TOKENS, hc * hid
    ref_norm = module.hc_norm(hidden)
    mx.eval(ref_norm)
    got_norm = hc_fused._kernel_norm(
        module, hidden.reshape(rows, width), rows, hc, hid, hidden.dtype
    ).reshape(hidden.shape)
    mx.eval(got_norm)
    results["norm"] = {"hc_norm_vs_kernel": ulp_report(ref_norm, got_norm, "hc_norm")}

    # Which of the two is closer to the exact answer?  Ground truth in fp32
    # with an fp32 weight, rounded to bf16 exactly once.
    def truth(x, module):
        xf = x.astype(mx.float32)
        g = xf.reshape(*xf.shape[:-1], -1, module.hc_norm.group_size)
        inv = mx.rsqrt(mx.mean(g * g, axis=-1, keepdims=True) + module.hc_norm.eps)
        scale = (1.0 + module.hc_norm.weight.astype(mx.float32)).reshape(
            -1, module.hc_norm.group_size)
        return (g * inv * scale).reshape(x.shape).astype(mx.bfloat16)

    gt = truth(hidden, module)
    mx.eval(gt)
    print("  against an fp32 ground truth (lower is better):")
    results["norm"]["fp32_truth_vs_stock"] = ulp_report(gt, ref_norm, "truth vs stock fp32")
    results["norm"]["fp32_truth_vs_kernel"] = ulp_report(gt, got_norm, "truth vs bf16 kernel")

    # on a realistic magnitude spread as well
    wide = (mx.random.normal((1, TOKENS, width))
            * mx.exp(mx.random.normal((1, TOKENS, 1)) * 2)).astype(mx.bfloat16)
    mx.eval(wide)
    rn2 = module.hc_norm(wide)
    gn2 = hc_fused._kernel_norm(module, wide.reshape(rows, width), rows, hc, hid,
                                wide.dtype).reshape(wide.shape)
    mx.eval(rn2, gn2)
    results["norm"]["hc_norm_wide_dynamic_range"] = ulp_report(rn2, gn2, "hc_norm (wide)")

    print("\n  whole prefill_forward (norm + projections + gated mean):")
    ref_out = hc_fused.prefill_forward(module, hidden)
    mx.eval(ref_out)
    norm_patch.apply_bf16_norm_patch(force=True)
    got_out = hc_fused.prefill_forward(module, hidden)
    mx.eval(got_out)
    pieces = ("mixed", "hyper_input", "injection")
    rep = {}
    for i, nm in enumerate(pieces if isinstance(ref_out, tuple) else ("mixed",)):
        r = ref_out[i] if isinstance(ref_out, tuple) else ref_out
        g = got_out[i] if isinstance(got_out, tuple) else got_out
        rep[nm] = ulp_report(r, g, f"prefill_forward.{nm}")
    results["norm"]["prefill_forward"] = rep
    results["norm"]["kernel_calls"] = norm_patch.STATS["kernel_calls"]
    results["norm"]["fallback_calls"] = norm_patch.STATS["fallback_calls"]
    norm_patch.remove_bf16_norm_patch()

    # ---------------------------------------------------------- 4. body
    print("\n[4] whole-body effect")
    from profile import forward
    base_h = forward(lm, ids)
    mx.eval(base_h)

    print("  control: PLE packed only (must be bitwise identical)")
    ple_patch.apply_ple_packed_patch(model, MODEL, mode="resident", force=True)
    ple_h = forward(lm, ids)
    mx.eval(ple_h)
    ctrl = bitwise_equal(base_h, ple_h)
    print(f"    body hidden state bitwise identical: {ctrl}")
    results["body_control_ple_only"] = ctrl

    print("  + bf16 norm, on random token ids (worst case: the model is maximally "
          "unsure, so argmax is a coin flip between near-ties)")
    norm_patch.apply_bf16_norm_patch(force=True)
    new_h = forward(lm, ids)
    mx.eval(new_h)
    rand_body = ulp_report(base_h, new_h, "body hidden (random ids)")
    rb = lm.lm_head(base_h)
    rn = lm.lm_head(new_h)
    mx.eval(rb, rn)
    rand_agree = float(mx.mean((mx.argmax(rb, -1) == mx.argmax(rn, -1)).astype(mx.float32)).item())
    top2 = mx.sort(rb, axis=-1)[..., -2:]
    margin = (top2[..., 1] - top2[..., 0]).astype(mx.float32)
    print(f"    greedy argmax agrees {100*rand_agree:.2f}% over {TOKENS} positions; "
          f"median top-1 margin {float(mx.median(margin).item()):.4f}")
    norm_patch.remove_bf16_norm_patch()
    ple_patch.remove_ple_packed_patch(model)
    results["body_random_ids"] = dict(ulp=rand_body, argmax_agreement=rand_agree,
                                      median_top1_margin=float(mx.median(margin).item()))

    print("  on real text (the regime that matters)")
    try:
        text = (open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "patch.py")).read()
                + open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "repack.py")).read())
        tok = getattr(processor, "tokenizer", processor)
        real = mx.array([tok.encode(text)[:TOKENS]])
        mx.eval(real)
        if real.shape[1] < TOKENS:
            raise ValueError("not enough tokens")
        rb_h = forward(lm, real)
        rb_l = lm.lm_head(rb_h)
        mx.eval(rb_l)
        ple_patch.apply_ple_packed_patch(model, MODEL, mode="resident", force=True)
        norm_patch.apply_bf16_norm_patch(force=True)
        rn_h = forward(lm, real)
        rn_l = lm.lm_head(rn_h)
        mx.eval(rn_l)
        real_ulp = ulp_report(rb_h, rn_h, "body hidden (real text)")
        real_agree = float(mx.mean((mx.argmax(rb_l, -1) == mx.argmax(rn_l, -1))
                                   .astype(mx.float32)).item())
        t2 = mx.sort(rb_l, axis=-1)[..., -2:]
        m2 = (t2[..., 1] - t2[..., 0]).astype(mx.float32)
        disagree = (mx.argmax(rb_l, -1) != mx.argmax(rn_l, -1)).reshape(-1)
        mx.eval(disagree, m2)
        m2n = np.asarray(m2.reshape(-1).astype(mx.float32))
        dn = np.asarray(disagree)
        med_dis = float(np.median(m2n[dn])) if dn.any() else float("nan")
        med_agr = float(np.median(m2n[~dn])) if (~dn).any() else float("nan")
        print(f"    median top-1 margin where they disagreed {med_dis:.4f}, "
              f"where they agreed {med_agr:.4f}")
        print(f"    greedy argmax agrees {100*real_agree:.2f}% over {TOKENS} positions; "
              f"median top-1 margin {float(mx.median(m2).item()):.4f}")
        results["body_real_text"] = dict(ulp=real_ulp, argmax_agreement=real_agree,
                                         median_top1_margin=float(mx.median(m2).item()),
                                         median_margin_where_disagreed=med_dis,
                                         median_margin_where_agreed=med_agr)
        norm_patch.remove_bf16_norm_patch()
        ple_patch.remove_ple_packed_patch(model)
    except Exception as exc:  # noqa: BLE001
        print(f"    real-text check skipped: {type(exc).__name__}: {exc}")
        norm_patch.remove_bf16_norm_patch()
        ple_patch.remove_ple_packed_patch(model)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_exact.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {out}")

    ok = all(v["gather_bitwise_identical"] and v["layer_bitwise_identical"]
             and v["stock_restored"] for v in results["backends"].values())
    print("PLE LOOKUP BIT-EXACT" if ok else "PLE LOOKUP MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
