# SPDX-License-Identifier: Apache-2.0
"""Boundary snapshot size for Qwen3.8-Flash-Next, from the model config only.

No weights are read. Run with any python3.
"""
from __future__ import annotations

import json

CFG = "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/config.json"


def main() -> None:
    t = json.load(open(CFG))["text_config"]
    n_lin = sum(1 for x in t["layer_types"] if x == "linear_attention")
    n_full = len(t["layer_types"]) - n_lin
    kd, vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
    nk, nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    conv_dim = kd * nk * 2 + vd * nv
    conv_len = t["linear_conv_kernel_dim"] - 1

    rec = nv * kd * vd * 4                      # cache[1], fp32 (mamba_ssm_dtype)
    conv = conv_len * conv_dim * 2              # cache[0], bf16
    per_layer = rec + conv
    total = per_layer * n_lin
    ple = 2 * conv_len * conv_dim * 2           # ple layer carries 2 extra slots
    total += ple * len(t["ple_layer_ids"])

    kv_per_tok = 2 * t["num_key_value_heads"] * t["head_dim"] * 2 * n_full

    print(f"linear (GDN) layers      {n_lin}")
    print(f"full (QSA) layers        {n_full}")
    print(f"GDN recurrent state      [{nv}, {kd}, {vd}] fp32 = {rec/2**20:.2f} MiB/layer")
    print(f"GDN conv state           [{conv_len}, {conv_dim}] bf16 = {conv/2**20:.3f} MiB/layer")
    print(f"one boundary snapshot    {total/2**20:.1f} MiB (fp32 sidecars)")
    print(f"                         {(total - n_lin*rec//2)/2**20:.1f} MiB at bf16 sidecars")
    print(f"QSA KV per token         {kv_per_tok/1024:.1f} KiB "
          f"({kv_per_tok*2048/2**20:.0f} MiB per 2048-token block)")
    print()
    for blk in (2048, 1024, 512, 256):
        n64 = 65536 // blk
        print(f"block {blk:>4}: {n64:>3} snapshots over a 64k prompt = "
              f"{n64*total/2**30:.2f} GiB if all resident; "
              f"tail-only patch keeps 1 extra = {total/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
