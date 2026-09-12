#!/usr/bin/env python3
"""Numerics validation for norm_patch.py, on real activations.

The patch swaps ``hc_fused.prefill_forward``'s grouped RMSNorm from
``Qwen4ExpRMSNorm`` (bf16 -> fp32 -> normalise -> * fp32 scale -> bf16) to
``hc_fused._kernel_norm`` (bf16 in, fp32 simd accumulation in registers, bf16
out).  The arithmetic is the same; the rounding order is not.  This script asks
three questions on the *real* hyper-connection activations of a 2048-token
chunk of real text:

  1. stock vs kernel, in bf16 ULPs, over all 96 call sites of a real forward.
  2. Which of the two is closer to an fp32 reference (fp32 throughout, rounded
     to bf16 exactly once)?  If the kernel is no further from the reference
     than the stock path is, the patch is not a numerical regression.
  3. Does greedy argmax through the full model agree on real text?

Memory: the model is loaded once (mmap PLE, ~73 GB).  Per call site four
[1, T, 10240] tensors are live (42 MB each) plus fp32 temporaries; everything
is evaluated and dropped inside the hook, and mx.clear_cache() runs between
stages.  No packed table is faulted in.

Run: ``. ./_env.sh && $PY norm_exact.py``
"""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/inference-server/profile-flashnext"))

import norm_patch  # noqa: E402

TOKENS = int(os.environ.get("TEST_TOKENS", "2048"))


# --------------------------------------------------------------- ULP helpers
def _ordinal(x):
    """Monotone integer ordering of bf16 bit patterns: adjacent representable
    values differ by 1, across zero and across the sign bit."""
    u = x.view(mx.uint16).astype(mx.int32)
    return mx.where(u >= 32768, 32768 - u, u + 32768)


class UlpAcc:
    """Streaming bf16-ULP accumulator so nothing large has to stay live."""

    def __init__(self, name):
        self.name = name
        self.n = 0
        self.exact = 0
        self.le1 = 0
        self.sum_ulp = 0.0
        self.max_ulp = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.per_call_max = []

    def add(self, ref, got):
        d = mx.abs(_ordinal(ref) - _ordinal(got))
        rf, gf = ref.astype(mx.float32), got.astype(mx.float32)
        ad = mx.abs(rf - gf)
        # relative error only where the reference is not a denormal-scale zero
        denom = mx.maximum(mx.abs(rf), 1e-20)
        rel = ad / denom
        stats = (mx.sum(d == 0), mx.sum(d <= 1), mx.sum(d.astype(mx.float32)),
                 mx.max(d), mx.max(ad), mx.max(rel))
        mx.eval(stats)
        self.n += int(d.size)
        self.exact += int(stats[0].item())
        self.le1 += int(stats[1].item())
        self.sum_ulp += float(stats[2].item())
        m = int(stats[3].item())
        self.per_call_max.append(m)
        self.max_ulp = max(self.max_ulp, m)
        self.max_abs = max(self.max_abs, float(stats[4].item()))
        self.max_rel = max(self.max_rel, float(stats[5].item()))

    def result(self):
        return dict(name=self.name, elements=self.n, call_sites=len(self.per_call_max),
                    bitwise_identical=self.exact,
                    fraction_identical=self.exact / max(self.n, 1),
                    fraction_within_1_ulp=self.le1 / max(self.n, 1),
                    max_ulp=self.max_ulp,
                    mean_ulp=self.sum_ulp / max(self.n, 1),
                    max_abs=self.max_abs, max_rel=self.max_rel,
                    worst_call_site_ulp=max(self.per_call_max or [0]))

    def show(self):
        r = self.result()
        print(f"  {self.name:34s} identical {100*r['fraction_identical']:7.3f}%  "
              f"<=1ULP {100*r['fraction_within_1_ulp']:7.3f}%  "
              f"max {r['max_ulp']} ULP  mean {r['mean_ulp']:.2e} ULP  "
              f"max|abs| {r['max_abs']:.3e}  ({r['call_sites']} call sites)")
        return r


def fp32_truth(x, module):
    """Grouped RMSNorm in fp32 throughout, rounded to bf16 exactly once."""
    xf = x.astype(mx.float32)
    gs = module.hc_norm.group_size
    g = xf.reshape(*xf.shape[:-1], -1, gs)
    inv = mx.rsqrt(mx.mean(g * g, axis=-1, keepdims=True) + module.hc_norm.eps)
    scale = (1.0 + module.hc_norm.weight.astype(mx.float32)).reshape(-1, gs)
    return (g * inv * scale).reshape(x.shape).astype(mx.bfloat16)


# ------------------------------------------------------------------ real text
def real_tokens(processor, n, which="prose"):
    paths = {
        "prose": [os.path.expanduser("~/inference-server/profile-flashnext/REPORT.md"),
                  os.path.expanduser("~/inference-server/kernels/moe-int8/REPORT.md")],
        "code": [os.path.join(HERE, "patch.py"),
                 os.path.join(HERE, "repack.py"),
                 os.path.expanduser("~/inference-server/profile-flashnext/profile.py")],
    }[which]
    text = "\n\n".join(open(p).read() for p in paths if os.path.isfile(p))
    tok = getattr(processor, "tokenizer", processor)
    ids = tok.encode(text)
    if len(ids) < n:
        raise ValueError(f"{which}: only {len(ids)} tokens available, need {n}")
    out = mx.array([ids[:n]])
    mx.eval(out)
    return out


def main():
    from load_omlx import load, MODEL
    from profile import forward

    results = {"model": MODEL, "tokens": TOKENS,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"loading {MODEL} (PLE mmap; the packed table is NOT faulted in)")
    model, processor = load(ple_mode="mmap")
    lm = model.language_model
    print(f"  mlx active {mx.get_active_memory()/1e9:.1f} GB")

    ids = real_tokens(processor, TOKENS, "prose")
    print(f"  real-text chunk: {tuple(ids.shape)}")

    # ------------------------------------------------ 1+2: every real call site
    print("\n[1] grouped RMSNorm on the real activations of every "
          "hyper-connection call site")
    from mlx_vlm.models.qwen4_exp import hc_fused

    stock_vs_kernel = UlpAcc("stock vs kernel")
    truth_vs_stock = UlpAcc("fp32 truth vs stock (current)")
    truth_vs_kernel = UlpAcc("fp32 truth vs kernel (patch)")
    magnitudes = []
    original_prefill = hc_fused.prefill_forward

    def probing_prefill(module, hyper_input):
        rows = hyper_input.shape[0] * hyper_input.shape[1]
        width = module.hc_count * module.hidden_size
        ref = module.hc_norm(hyper_input)
        ker = hc_fused._kernel_norm(module, hyper_input.reshape(rows, width),
                                    rows, module.hc_count, module.hidden_size,
                                    hyper_input.dtype).reshape(hyper_input.shape)
        gt = fp32_truth(hyper_input, module)
        mx.eval(ref, ker, gt)
        stock_vs_kernel.add(ref, ker)
        truth_vs_stock.add(gt, ref)
        truth_vs_kernel.add(gt, ker)
        rms = mx.sqrt(mx.mean(hyper_input.astype(mx.float32) ** 2))
        mx.eval(rms)
        magnitudes.append(float(rms.item()))
        del ref, ker, gt
        return original_prefill(module, hyper_input)

    hc_fused.prefill_forward = probing_prefill
    t0 = time.time()
    h = forward(lm, ids)
    mx.eval(h)
    hc_fused.prefill_forward = original_prefill
    del h
    mx.clear_cache()
    print(f"  instrumented forward: {time.time()-t0:.1f}s, "
          f"{len(magnitudes)} call sites, activation RMS "
          f"{min(magnitudes):.3g} .. {max(magnitudes):.3g}")
    results["norm_call_sites"] = {
        "stock_vs_kernel": stock_vs_kernel.show(),
        "fp32_truth_vs_stock": truth_vs_stock.show(),
        "fp32_truth_vs_kernel": truth_vs_kernel.show(),
        "activation_rms_min": min(magnitudes),
        "activation_rms_max": max(magnitudes),
    }
    a = truth_vs_stock.result()["mean_ulp"]
    b = truth_vs_kernel.result()["mean_ulp"]
    print(f"  -> mean distance from the fp32 reference: stock {a:.3e} ULP, "
          f"kernel {b:.3e} ULP  ({'kernel no worse' if b <= a * 1.5 else 'KERNEL WORSE'})")

    # --------------------------------- 3: the whole prefill_forward, real input
    print("\n[2] whole prefill_forward (norm + projections + gated mean) on a "
          "real call site")
    captured = {}

    def capture(module, hyper_input):
        if not captured:
            captured["module"] = module
            captured["x"] = mx.array(hyper_input)
            mx.eval(captured["x"])
        return original_prefill(module, hyper_input)

    hc_fused.prefill_forward = capture
    h = forward(lm, ids)
    mx.eval(h)
    hc_fused.prefill_forward = original_prefill
    del h
    mx.clear_cache()

    module, x = captured["module"], captured["x"]
    ref_out = original_prefill(module, x)
    mx.eval(ref_out)
    norm_patch.apply_bf16_norm_patch(force=True)
    got_out = hc_fused.prefill_forward(module, x)
    mx.eval(got_out)
    names = ("mixed", "hyper_input", "injection") if isinstance(ref_out, tuple) else ("mixed",)
    pf = {}
    for i, nm in enumerate(names):
        r = ref_out[i] if isinstance(ref_out, tuple) else ref_out
        g = got_out[i] if isinstance(got_out, tuple) else got_out
        acc = UlpAcc(f"prefill_forward.{nm}")
        acc.add(r, g)
        pf[nm] = acc.show()
        rf = r.astype(mx.float32)
        rms = float(mx.sqrt(mx.mean(rf * rf)).item())
        err = float(mx.sqrt(mx.mean((rf - g.astype(mx.float32)) ** 2)).item())
        pf[nm]["ref_rms"] = rms
        pf[nm]["rms_error_pct_of_rms"] = 100.0 * err / max(rms, 1e-30)
        print(f"      RMS error {100*err/max(rms,1e-30):.4f}% of output RMS "
              f"(output RMS {rms:.4g})")
    results["prefill_forward"] = pf
    results["prefill_forward_stats"] = dict(norm_patch.STATS)
    norm_patch.remove_bf16_norm_patch()
    del ref_out, got_out, x, captured
    mx.clear_cache()

    # --------------------------------------------- 4: argmax through the model
    print("\n[3] greedy argmax through the full model, real text")
    body = {}
    for which in ("prose", "code"):
        tid = real_tokens(processor, TOKENS, which)

        def logits_of(tokens):
            hh = forward(lm, tokens)
            ll = lm.lm_head(hh)
            mx.eval(hh, ll)
            return hh, ll

        base_h, base_l = logits_of(tid)
        base_arg = mx.argmax(base_l, axis=-1)
        top2 = mx.sort(base_l, axis=-1)[..., -2:]
        margin = (top2[..., 1] - top2[..., 0]).astype(mx.float32)
        base_max = mx.max(base_l.astype(mx.float32))
        mx.eval(base_arg, margin, base_max)
        base_lf = base_l.astype(mx.float32)
        mx.eval(base_lf)
        del base_l, top2
        mx.clear_cache()

        # control: the stock path is deterministic, so a repeat must be identical
        ctl_h, ctl_l = logits_of(tid)
        ctl_arg = mx.argmax(ctl_l, axis=-1)
        ctl_same = bool(mx.all(ctl_arg == base_arg).item())
        ctl_bits = bool(mx.all(ctl_h.view(mx.uint16) == base_h.view(mx.uint16)).item())
        del ctl_h, ctl_l, ctl_arg
        mx.clear_cache()

        norm_patch.apply_bf16_norm_patch(force=True)
        new_h, new_l = logits_of(tid)
        new_arg = mx.argmax(new_l, axis=-1)
        agree = (new_arg == base_arg)
        frac = float(mx.mean(agree.astype(mx.float32)).item())
        dlog = float(mx.max(mx.abs(base_lf - new_l.astype(mx.float32))).item())
        hacc = UlpAcc(f"body hidden ({which})")
        hacc.add(base_h, new_h)
        mx.eval(agree)
        mn = np.asarray(margin.reshape(-1))
        ag = np.asarray(agree.reshape(-1))
        med_dis = float(np.median(mn[~ag])) if (~ag).any() else float("nan")
        med_agr = float(np.median(mn[ag])) if ag.any() else float("nan")
        bf = base_h.astype(mx.float32)
        hrms = float(mx.sqrt(mx.mean(bf * bf)).item())
        herr = float(mx.sqrt(mx.mean((bf - new_h.astype(mx.float32)) ** 2)).item())
        norm_patch.remove_bf16_norm_patch()
        print(f"  {which}: control repeat bitwise identical {ctl_bits} / "
              f"argmax identical {ctl_same}")
        hstat = hacc.show()
        print(f"      body hidden RMS error {100*herr/hrms:.4f}% of RMS; "
              f"max |logit| delta {dlog:.4f} (peak logit "
              f"{float(base_max.item()):.2f})")
        print(f"      greedy argmax agrees {100*frac:.2f}% over {TOKENS} positions; "
              f"median top-1 margin {float(np.median(mn)):.4f}; "
              f"where they disagreed {med_dis:.4f}, where they agreed {med_agr:.4f}")
        body[which] = dict(control_bitwise_identical=ctl_bits,
                           control_argmax_identical=ctl_same,
                           hidden=hstat,
                           hidden_rms_error_pct=100 * herr / hrms,
                           max_abs_logit_delta=dlog,
                           peak_logit=float(base_max.item()),
                           argmax_agreement=frac,
                           median_top1_margin=float(np.median(mn)),
                           median_margin_where_disagreed=med_dis,
                           median_margin_where_agreed=med_agr,
                           disagreements=int((~ag).sum()))
        del base_h, base_lf, new_h, new_l, new_arg, base_arg, agree, margin
        mx.clear_cache()
    results["body"] = body

    out = os.path.join(HERE, "norm_exact.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {out}")

    r = results["norm_call_sites"]
    ok = (r["stock_vs_kernel"]["max_ulp"] <= 1
          and r["fp32_truth_vs_kernel"]["mean_ulp"]
          <= 1.5 * max(r["fp32_truth_vs_stock"]["mean_ulp"], 1e-12)
          and all(v["argmax_agreement"] >= 0.99 for v in body.values()))
    print("NORM PATCH: PASS" if ok else "NORM PATCH: REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
