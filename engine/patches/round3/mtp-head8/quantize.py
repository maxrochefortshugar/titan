"""Requantise the 4-bit mtp.* tensors from the bf16 originals to 8-bit group 64.

Reads the bf16 slabs fetched by HTTP range from Qwen/Qwen3.8-Flash-Next, writes
mtp-8bit.safetensors plus manifest.json, and reports per-tensor error of both
the new 8-bit and the deployed 4-bit against bf16.

Memory: every tensor is processed in expert chunks, peak stays near 400 MB.
The deployed checkpoint is read one tensor at a time by byte range; no shard is
ever opened as a whole.
"""

import argparse
import hashlib
import json
import os
import time

import mlx.core as mx
import numpy as np

import stio

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "orig")
DEPLOY = os.path.expanduser(
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
)
GS, BITS = 64, 8

# name -> (raw file, bf16 shape, deployed prefix, deployed shard, chunk axis)
PLAN = {
    "mtp.fc_embedding": ("fc_embedding", (2560, 2560), 1, None),
    "mtp.fc_hidden": ("fc_hidden", (2560, 2560), 1, None),
    "mtp.hyper_connection_mixer.input_mix_weight_down": (
        "mix_down",
        (320, 10240),
        1,
        None,
    ),
    "mtp.hyper_connection_mixer.input_mix_weight_up": (
        "mix_up",
        (10240, 320),
        1,
        None,
    ),
    "mtp.layers.0.mlp.switch_mlp.gate_proj": (
        "experts_gate_up_proj",
        (512, 640, 2560),
        21,
        "gate",
    ),
    "mtp.layers.0.mlp.switch_mlp.up_proj": (
        "experts_gate_up_proj",
        (512, 640, 2560),
        21,
        "up",
    ),
    "mtp.layers.0.mlp.switch_mlp.down_proj": (
        "experts_down_proj",
        (512, 2560, 640),
        21,
        None,
    ),
}

SHARD = {
    1: "model-00001-of-00021.safetensors",
    21: "model-00021-of-00021.safetensors",
}
_HDRS = {}


def deployed(name, shard):
    path = os.path.join(DEPLOY, SHARD[shard])
    if shard not in _HDRS:
        _HDRS[shard] = stio.read_header(path)
    hdr, start = _HDRS[shard]
    w = stio.read_tensor(path, hdr, start, name + ".weight")
    s = stio.read_tensor(path, hdr, start, name + ".scales")
    b = stio.read_tensor(path, hdr, start, name + ".biases")
    return w, s, b


def deployed_meta(name, shard):
    cfg = json.load(open(os.path.join(DEPLOY, "config.json")))["quantization"]
    ov = cfg.get(name, cfg)
    return ov["bits"], ov["group_size"]


def load_bf16_chunk(raw, shape, half, lo, hi):
    """Load experts [lo, hi) (or the whole 2-D tensor when experts is None)."""
    path = os.path.join(ORIG, raw + ".bf16.raw")
    if len(shape) == 2:
        return stio.read_raw_bf16(path, shape)
    E, O, I = shape
    if half is None:
        per = O * I
        return stio.read_raw_bf16(
            path, (hi - lo, O, I), offset=lo * per * 2, count=(hi - lo) * per
        )
    # gate_up_proj is [E, 2*O, I]; gate is the first O rows, up the rest.
    per = 2 * O * I
    slab = stio.read_raw_bf16(
        path, (hi - lo, 2 * O, I), offset=lo * per * 2, count=(hi - lo) * per
    )
    return slab[:, :O, :] if half == "gate" else slab[:, O:, :]


def errs(a, b):
    """max abs error and relative Frobenius error of a against reference b."""
    d = (a.astype(mx.float32) - b.astype(mx.float32)).astype(mx.float32)
    return (
        float(mx.max(mx.abs(d))),
        float(mx.sum(d * d)),
        float(mx.sum(b.astype(mx.float32) ** 2)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma separated tensor names")
    ap.add_argument("--chunk", type=int, default=32, help="experts per chunk")
    ap.add_argument("--out", default="mtp-8bit.safetensors")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    names = list(PLAN)
    if args.only:
        want = args.only.split(",")
        names = [n for n in names if any(w in n for w in want)]

    spec = []
    for n in names:
        raw, shape, _, half = PLAN[n]
        O, I = shape[-2], shape[-1]
        lead = list(shape[:-2])
        spec += [
            (n + ".weight", lead + [O, I * BITS // 32], mx.uint32),
            (n + ".scales", lead + [O, I // GS], mx.bfloat16),
            (n + ".biases", lead + [O, I // GS], mx.bfloat16),
        ]

    out = os.path.join(HERE, args.out)
    w = stio.StreamWriter(
        out,
        spec,
        metadata={
            "source": "Qwen/Qwen3.8-Flash-Next bf16",
            "bits": str(BITS),
            "group_size": str(GS),
            "mode": "affine",
        },
    )

    manifest = {"tensors": {}, "bits": BITS, "group_size": GS, "mode": "affine"}
    report = []
    hasher = {}

    for n in names:
        raw, shape, shard, half = PLAN[n]
        t0 = time.time()
        E = shape[0] if len(shape) == 3 else 1
        step = args.chunk if len(shape) == 3 else E
        parts = []  # (w, s, b) per chunk, kept only to stream out in order
        s8_num = s8_den = 0.0
        s8_max = 0.0
        h = hashlib.sha256()
        chunks = []
        for lo in range(0, E, step):
            hi = min(lo + step, E)
            bf = load_bf16_chunk(raw, shape, half, lo, hi)
            qw, qs, qb = mx.quantize(bf, group_size=GS, bits=BITS, mode="affine")
            mx.eval(qw, qs, qb)
            dq = mx.dequantize(qw, qs, qb, group_size=GS, bits=BITS, mode="affine")
            m, num, den = errs(dq, bf)
            s8_max = max(s8_max, m)
            s8_num += num
            s8_den += den
            chunks.append((np.array(qw), np.array(qs.view(mx.uint16)),
                           np.array(qb.view(mx.uint16))))
            del bf, dq, qw, qs, qb
            mx.clear_cache()
        for i, key in enumerate([".weight", ".scales", ".biases"]):
            for c in chunks:
                w.f.write(c[i].tobytes())
                h.update(c[i].tobytes())
            w._i += 1
            w._cur = n + key
        hasher[n] = h.hexdigest()
        del chunks
        mx.clear_cache()

        rel8 = (s8_num / s8_den) ** 0.5
        row = {"name": n, "max8": s8_max, "rel8": rel8, "secs": time.time() - t0}

        if not args.no_compare:
            # deployed 4-bit against bf16, sampled on the first chunk for the
            # 3-D expert tensors so the comparison never holds the full tensor.
            db, dg = deployed_meta(n, shard)
            dw, ds, dbi = deployed(n, shard)
            if len(shape) == 3:
                dw, ds, dbi = dw[:step], ds[:step], dbi[:step]
            d4 = mx.dequantize(dw, ds, dbi, group_size=dg, bits=db, mode="affine")
            bf = load_bf16_chunk(raw, shape, half, 0, min(step, E))
            m4, n4, dd4 = errs(d4, bf)
            row.update(
                {
                    "max4": m4,
                    "rel4": (n4 / dd4) ** 0.5,
                    "dep_bits": db,
                    "dep_group": dg,
                    "sampled": len(shape) == 3,
                }
            )
            del d4, bf, dw, ds, dbi
            mx.clear_cache()

        report.append(row)
        print(json.dumps(row), flush=True)

        O, I = shape[-2], shape[-1]
        manifest["tensors"][n] = {
            "shape": list(shape),
            "bits": BITS,
            "group_size": GS,
            "mode": "affine",
            "weight_shape": list(shape[:-2]) + [O, I * BITS // 32],
            "scales_shape": list(shape[:-2]) + [O, I // GS],
            "sha256": hasher[n],
            "max_abs_err_8bit": row["max8"],
            "rel_fro_err_8bit": row["rel8"],
            "max_abs_err_4bit": row.get("max4"),
            "rel_fro_err_4bit": row.get("rel4"),
        }

    w.close()
    manifest["file"] = os.path.basename(out)
    manifest["file_bytes"] = os.path.getsize(out)
    manifest["file_sha256"] = _sha_file(out)
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=2)
    print("wrote", out, manifest["file_bytes"], "bytes")


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
