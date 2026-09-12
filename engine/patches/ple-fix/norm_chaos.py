#!/usr/bin/env python3
"""Is the bf16 norm kernel's end-to-end divergence a regression, or chaos?

norm_exact.py showed the kernel matches the stock grouped RMSNorm to <= 1 bf16
ULP on every one of the 97 real call sites, with ~4 elements per million
differing at all -- and that the resulting body hidden state is nevertheless
12% RMS away from stock, with greedy argmax agreeing 93-97%.

Either the model amplifies *any* perturbation of that size (36 recurrent
gated-delta scans over 2048 steps will do that), or the kernel is wrong in a way
the per-call ULP check does not see.  Three body runs distinguish them:

  stock    the shipped fp32-round-trip norm (the reference)
  kernel   norm_patch's bf16 Metal kernel
  exact    the norm done in fp32 throughout and rounded to bf16 exactly once --
           i.e. *more* accurate than stock, differing from it by <= 1 ULP on
           ~1.6 elements per million
  dither   stock, plus a random +-1 ULP on a random 4-per-million of the norm's
           output elements: the same perturbation *density* as the kernel, but
           carrying no information at all

If `exact` and `dither` diverge from stock as much as `kernel` does, the 12% is
the model's Lyapunov behaviour and not a property of this patch.  It also means
"more accurate" and "different from stock" are the same thing here, so bitwise
agreement with stock cannot be the acceptance criterion.

Run: ``. ./_env.sh && $PY norm_chaos.py``
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

from norm_exact import UlpAcc, fp32_truth, real_tokens  # noqa: E402

TOKENS = int(os.environ.get("TEST_TOKENS", "2048"))
PROBE_EVERY = 8          # capture the hyper-connection input every N call sites


def rms(x):
    xf = x.astype(mx.float32)
    return float(mx.sqrt(mx.mean(xf * xf)).item())


def rel_rms(a, b):
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    num = float(mx.sqrt(mx.mean((af - bf) ** 2)).item())
    den = float(mx.sqrt(mx.mean(af * af)).item())
    return 100.0 * num / max(den, 1e-30)


def main():
    from load_omlx import load, MODEL
    from profile import forward

    print(f"loading {MODEL} (PLE mmap)")
    model, processor = load(ple_mode="mmap")
    from mlx_vlm.models.qwen4_exp import hc_fused
    lm = model.language_model
    original_prefill = hc_fused.prefill_forward
    results = {"model": MODEL, "tokens": TOKENS,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------------------------------------------------- norm variants
    def make(norm_fn, tag):
        import mlx.nn as nn

        def pf(module, hyper_input):
            hc, hidden = module.hc_count, module.hidden_size
            normed = norm_fn(module, hyper_input)
            mix = nn.silu(module.input_mix_weight_down(normed) / hc)
            mixed = hc_fused._tail(hc, hidden)(module.input_mix_weight_up(mix), normed)
            inject = (module.block_inject_weight
                      if "block_inject_weight" in module else None)
            injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
            if injection is None:
                return mixed
            return mixed, hyper_input, injection

        pf.tag = tag
        return pf

    def stock_norm(module, x):
        return module.hc_norm(x)

    def kernel_norm(module, x):
        rows = x.shape[0] * x.shape[1]
        return hc_fused._kernel_norm(module, x.reshape(rows, module.hc_count
                                                       * module.hidden_size),
                                     rows, module.hc_count, module.hidden_size,
                                     x.dtype).reshape(x.shape)

    def exact_norm(module, x):
        return fp32_truth(x, module)

    DITHER_RATE = 4.4e-6          # measured stock-vs-kernel disagreement density
    _dither_seed = [0]

    def dither_norm(module, x):
        y = module.hc_norm(x)
        _dither_seed[0] += 1
        mx.random.seed(90000 + _dither_seed[0])
        u = y.view(mx.uint16).astype(mx.uint32)
        hit = mx.random.uniform(shape=y.shape) < DITHER_RATE
        step = mx.where(mx.random.uniform(shape=y.shape) < 0.5, 1, -1).astype(mx.uint32)
        # only perturb normal, non-zero magnitudes so +-1 never crosses zero
        mag = u & 0x7FFF
        safe = hit & (mag > 0x0100) & (mag < 0x7F00)
        u2 = mx.where(safe, u + step, u)
        return u2.astype(mx.uint16).view(mx.bfloat16)

    VARIANTS = {
        "stock": stock_norm,
        "kernel": kernel_norm,
        "exact": exact_norm,
        "dither": dither_norm,
    }

    # --------------------------------------------------- run each, with probes
    def run(tag, text_ids):
        probes = []
        counter = [0]
        fn = make(VARIANTS[tag], tag)

        def probing(module, hyper_input):
            if counter[0] % PROBE_EVERY == 0:
                c = mx.array(hyper_input)
                mx.eval(c)
                probes.append(c)
            counter[0] += 1
            return fn(module, hyper_input)

        hc_fused.prefill_forward = probing
        h = forward(lm, text_ids)
        logits = lm.lm_head(h)
        mx.eval(h, logits)
        hc_fused.prefill_forward = original_prefill
        arg = mx.argmax(logits, axis=-1)
        top2 = mx.sort(logits, axis=-1)[..., -2:]
        margin = (top2[..., 1] - top2[..., 0]).astype(mx.float32)
        mx.eval(arg, margin)
        del logits, top2
        mx.clear_cache()
        return dict(h=h, arg=arg, margin=margin, probes=probes)

    out = {}
    for which in ("prose", "code"):
        print(f"\n=== {which} ===")
        ids = real_tokens(processor, TOKENS, which)
        runs = {}
        for tag in VARIANTS:
            t0 = time.time()
            runs[tag] = run(tag, ids)
            print(f"  ran {tag:7s} ({time.time()-t0:.1f}s)")

        ref = runs["stock"]
        rows = {}
        for tag in ("kernel", "exact", "dither"):
            r = runs[tag]
            agree = (r["arg"] == ref["arg"])
            frac = float(mx.mean(agree.astype(mx.float32)).item())
            acc = UlpAcc(f"{tag} vs stock")
            acc.add(ref["h"], r["h"])
            mx.eval(agree)
            mn = np.asarray(ref["margin"].reshape(-1))
            ag = np.asarray(agree.reshape(-1))
            growth = [rel_rms(a, b) for a, b in zip(ref["probes"], r["probes"])]
            rows[tag] = dict(
                hidden_rms_error_pct=rel_rms(ref["h"], r["h"]),
                hidden_ulp=acc.result(),
                argmax_agreement=frac,
                disagreements=int((~ag).sum()),
                median_margin_where_disagreed=(float(np.median(mn[~ag]))
                                               if (~ag).any() else float("nan")),
                median_margin_where_agreed=(float(np.median(mn[ag]))
                                            if ag.any() else float("nan")),
                divergence_by_call_site=growth,
            )
            print(f"  {tag:7s} vs stock: body hidden RMS error "
                  f"{rows[tag]['hidden_rms_error_pct']:6.2f}%, "
                  f"argmax agrees {100*frac:6.2f}% "
                  f"({rows[tag]['disagreements']} of {TOKENS}), "
                  f"median margin at disagreement "
                  f"{rows[tag]['median_margin_where_disagreed']:.3f}")
            print("          divergence by hyper-connection call site "
                  f"(every {PROBE_EVERY}): "
                  + " ".join(f"{g:.2g}" for g in growth))

        # kernel against the most accurate reference we can build
        ex = runs["exact"]
        rows["kernel_vs_exact"] = dict(
            hidden_rms_error_pct=rel_rms(ex["h"], runs["kernel"]["h"]),
            argmax_agreement=float(mx.mean((runs["kernel"]["arg"] == ex["arg"])
                                           .astype(mx.float32)).item()))
        rows["stock_vs_exact"] = dict(
            hidden_rms_error_pct=rel_rms(ex["h"], ref["h"]),
            argmax_agreement=float(mx.mean((ref["arg"] == ex["arg"])
                                           .astype(mx.float32)).item()))
        print(f"  against the fp32-exact norm: stock is "
              f"{rows['stock_vs_exact']['hidden_rms_error_pct']:.2f}% / "
              f"{100*rows['stock_vs_exact']['argmax_agreement']:.2f}% argmax, "
              f"kernel is {rows['kernel_vs_exact']['hidden_rms_error_pct']:.2f}% / "
              f"{100*rows['kernel_vs_exact']['argmax_agreement']:.2f}% argmax")
        out[which] = rows
        del runs, ref
        mx.clear_cache()

    results["variants"] = out
    results["dither_rate"] = DITHER_RATE
    path = os.path.join(HERE, "norm_chaos.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
