#!/usr/bin/env python3
"""Offline repack of the Qwen4-Exp PLE n-gram table into read-friendly layouts.

The shipped oQ4e checkpoint stores the 128 PLE shards as three separate
safetensors tensors per shard::

    ...ngram_embedding.shards.<s>.weight   U32  [2500012, 20]   (160 dims, 4 bit)
    ...ngram_embedding.shards.<s>.scales   BF16 [2500012,  5]   (group_size 32)
    ...ngram_embedding.shards.<s>.biases   BF16 [2500012,  5]

so one logical 160-dim row (80 B of nibbles + 10 B scales + 10 B biases = 100 B)
is spread over three regions of a 5 GB file.  oMLX's SSD offload preads whole
16 KB pages, so one row costs three pages: 32768 rows per 2048-token chunk read
~1.5 GB for 3.3 MB of payload (460x amplification, 963 ms, GPU idle).

This tool writes two derived layouts under ``<model>/ple-packed/``:

``rows``    one interleaved file per PLE layer, row stride 100 B,
            ``[weight(80) | scales(10) | biases(10)]``.  One row is one
            contiguous read, so the page count drops ~3x.

``planar``  three files per PLE layer holding the shard-concatenated
            ``weight`` / ``scales`` / ``biases`` planes back to back.  This is
            the layout ``nn.QuantizedEmbedding`` wants, so the whole 32 GB
            table can be faulted into one resident mx.array trio and the lookup
            becomes a single device gather.  (It is also what
            ``ShardedEmbedding.fuse_quantized_shards`` would build, without
            needing a 32 GB temporary next to a 100 GB process.)

Global row index == ``shard_offsets[s] + local_row``, matching
``DiskBackedShardedEmbedding``/``ShardedEmbedding`` exactly, so a packed row is
substitutable for the runtime's per-shard row with no index remapping.

Usage::

    ./repack.py                     # write both layouts, then verify
    ./repack.py --layout planar     # planar only (resident mode)
    ./repack.py --verify-only       # re-check an existing pack
    ./repack.py --verify-rows 200000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(
    os.environ.get(
        "PROFILE_MODEL",
        "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
    )
)
MARKER = ".ngram_embedding."
PACK_VERSION = 1

DTYPE_MAP = {
    "U32": (np.dtype("<u4"), 4),
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<u2"), 2),
    "F32": (np.dtype("<u4"), 4),
}


# --------------------------------------------------------------- safetensors
class SafeTensorFile:
    """Minimal read-only safetensors reader over a numpy memmap."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(header_size))
        self.data_start = 8 + header_size
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def entry(self, key):
        return self.header[key]

    def shape(self, key):
        return tuple(self.header[key]["shape"])

    def dtype(self, key):
        return str(self.header[key]["dtype"])

    def view(self, key) -> np.ndarray:
        """A 2-D uint8 view of the tensor's raw bytes: [rows, row_bytes]."""
        entry = self.header[key]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        np_dtype, item = DTYPE_MAP[str(entry["dtype"])]
        if len(shape) != 2 or (end - start) != math.prod(shape) * item:
            raise ValueError(f"unexpected layout for {key}: {entry}")
        row_bytes = shape[1] * item
        base = self.data_start + start
        return self._mm[base : base + shape[0] * row_bytes].reshape(shape[0], row_bytes)


# --------------------------------------------------------------- discovery
def discover(model: Path):
    """Return {layer_index: {'shards': [(weight_key, scales_key, biases_key, file)],
    'rows': n, 'dims': d, 'bits': b, 'group_size': g, 'weight_scale_key': k}}."""
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    layers: dict[int, dict] = {}
    for key in weight_map:
        if MARKER not in key:
            continue
        head, _, tail = key.partition(MARKER)
        parts_head = head.split(".")
        layer_index = int(parts_head[parts_head.index("layers") + 1])
        layers.setdefault(layer_index, {"prefix": head + MARKER.rstrip("."),
                                        "shards": {}, "weight_scale_key": None})
        if tail == "weight_scale":
            layers[layer_index]["weight_scale_key"] = key
            continue
        parts = tail.split(".")
        if parts[0] == "shards":
            shard_index, family = int(parts[1]), parts[2]
        elif parts[0].startswith("shard_"):
            shard_index, family = int(parts[0][6:]), parts[1]
        else:
            continue
        layers[layer_index]["shards"].setdefault(shard_index, {})[family] = key

    out = {}
    files: dict[str, SafeTensorFile] = {}

    def reader(key) -> SafeTensorFile:
        name = weight_map[key]
        if name not in files:
            files[name] = SafeTensorFile(model / name)
        return files[name]

    for layer_index, info in sorted(layers.items()):
        shards = info["shards"]
        specs, offsets, total = [], [], 0
        bits = group_size = dims = None
        for shard_index in sorted(shards):
            fam = shards[shard_index]
            wk, sk, bk = fam["weight"], fam.get("scales"), fam.get("biases")
            if sk is None or bk is None:
                raise ValueError(f"layer {layer_index} shard {shard_index} is not affine-packed")
            wr, sr = reader(wk), reader(sk)
            w_shape, s_shape = wr.shape(wk), sr.shape(sk)
            b_shape = reader(bk).shape(bk)
            if wr.dtype(wk) != "U32":
                raise TypeError(f"{wk} is {wr.dtype(wk)}, expected U32")
            if s_shape != b_shape or s_shape[0] != w_shape[0]:
                raise ValueError(f"shape mismatch on layer {layer_index} shard {shard_index}")
            packed_bits = w_shape[1] * 32
            this_dims = None
            for candidate_gs in (32, 64, 128):
                d = s_shape[1] * candidate_gs
                if packed_bits % d == 0 and packed_bits // d in (2, 3, 4, 5, 6, 8):
                    this_dims, this_gs = d, candidate_gs
                    break
            if this_dims is None:
                raise ValueError(f"cannot infer layout for {wk}")
            this_bits = packed_bits // this_dims
            if bits is None:
                bits, group_size, dims = this_bits, this_gs, this_dims
            elif (this_bits, this_gs, this_dims) != (bits, group_size, dims):
                raise ValueError("PLE shards disagree on bits/group_size/dims")
            specs.append(dict(shard=shard_index, weight=wk, scales=sk, biases=bk,
                              rows=int(w_shape[0]),
                              weight_bytes=int(w_shape[1]) * 4,
                              scale_bytes=int(s_shape[1]) * 2))
            offsets.append(total)
            total += int(w_shape[0])
        out[layer_index] = dict(
            prefix=info["prefix"], weight_scale_key=info["weight_scale_key"],
            specs=specs, shard_offsets=offsets, rows=total, dims=dims,
            bits=bits, group_size=group_size,
            weight_bytes=specs[0]["weight_bytes"], scale_bytes=specs[0]["scale_bytes"],
        )
    return out, files, weight_map


# --------------------------------------------------------------- writing
def _fmt(n):
    return f"{n / 1e9:.2f} GB"


def repack_layer(layer_index, info, files, weight_map, model: Path, out_dir: Path,
                 layouts, block_rows, progress_every=8):
    dims, bits, gs = info["dims"], info["bits"], info["group_size"]
    wb, sb = info["weight_bytes"], info["scale_bytes"]
    stride = wb + 2 * sb
    total_rows = info["rows"]
    print(f"  layer {layer_index}: {total_rows:,} rows x {dims} dims, {bits}-bit "
          f"group {gs}; row = {wb}+{sb}+{sb} = {stride} B")

    handles = {}
    if "rows" in layouts:
        handles["rows"] = open(out_dir / f"layer{layer_index}.rows.bin", "wb", buffering=0)
    if "planar" in layouts:
        for fam in ("weight", "scales", "biases"):
            ext = "u32" if fam == "weight" else "bf16"
            handles[fam] = open(out_dir / f"layer{layer_index}.{fam}.{ext}", "wb", buffering=0)

    t0 = time.time()
    written = 0
    for si, spec in enumerate(info["specs"]):
        wv = files[weight_map[spec["weight"]]].view(spec["weight"])
        sv = files[weight_map[spec["scales"]]].view(spec["scales"])
        bv = files[weight_map[spec["biases"]]].view(spec["biases"])
        n = spec["rows"]
        for start in range(0, n, block_rows):
            end = min(start + block_rows, n)
            w = np.ascontiguousarray(wv[start:end])
            s = np.ascontiguousarray(sv[start:end])
            b = np.ascontiguousarray(bv[start:end])
            if "rows" in handles:
                buf = np.empty((end - start, stride), dtype=np.uint8)
                buf[:, :wb] = w
                buf[:, wb:wb + sb] = s
                buf[:, wb + sb:] = b
                handles["rows"].write(buf.tobytes())
            if "weight" in handles:
                handles["weight"].write(w.tobytes())
                handles["scales"].write(s.tobytes())
                handles["biases"].write(b.tobytes())
            written += end - start
        if (si + 1) % progress_every == 0 or si + 1 == len(info["specs"]):
            dt = time.time() - t0
            rate = written * stride / dt / 1e9 if dt else 0
            print(f"    shard {si+1:3d}/{len(info['specs'])}  {written:,} rows  "
                  f"{dt:6.1f}s  {rate:5.2f} GB/s in")
    for h in handles.values():
        h.close()
    return dict(rows=total_rows, stride=stride, seconds=time.time() - t0)


def weight_scale_value(info, files, weight_map, model: Path):
    key = info["weight_scale_key"]
    if key is None:
        return 1.0
    f = files[weight_map[key]]
    entry = f.entry(key)
    start, end = entry["data_offsets"]
    raw = np.memmap(model / weight_map[key], dtype=np.uint8, mode="r")[
        f.data_start + start : f.data_start + end
    ]
    u16 = raw.view("<u2")
    return float((u16.astype(np.uint32) << np.uint32(16)).view(np.float32)[0])


# --------------------------------------------------------------- verify
def verify(model: Path, out_dir: Path, manifest, files, weight_map, n_rows, seed=0):
    """Bit-exact comparison of packed rows against the original shards."""
    rng = np.random.default_rng(seed)
    report = {}
    for layer_key, entry in manifest["layers"].items():
        layer_index = int(layer_key)
        info = entry["_info"]
        total, wb, sb = info["rows"], info["weight_bytes"], info["scale_bytes"]
        stride = wb + 2 * sb
        offsets = np.asarray(info["shard_offsets"], dtype=np.int64)

        # always include the boundaries: first/last row of every shard, plus
        # the global first/last, plus a uniform random sample.
        edges = np.concatenate([offsets, offsets + np.asarray(
            [s["rows"] for s in info["specs"]], dtype=np.int64) - 1])
        sample = rng.integers(0, total, size=max(0, n_rows - edges.size), dtype=np.int64)
        rows = np.unique(np.concatenate([edges, sample, np.array([0, total - 1])]))
        rng.shuffle(rows)

        # reference bytes, straight from the source safetensors
        shard = np.searchsorted(offsets, rows, side="right") - 1
        ref = np.empty((rows.size, stride), dtype=np.uint8)
        for si in np.unique(shard):
            spec = info["specs"][int(si)]
            pos = np.flatnonzero(shard == si)
            local = rows[pos] - offsets[si]
            order = np.argsort(local)          # memmap fancy-index likes sorted
            lo = local[order]
            ref[pos[order], :wb] = files[weight_map[spec["weight"]]].view(spec["weight"])[lo]
            ref[pos[order], wb:wb + sb] = files[weight_map[spec["scales"]]].view(spec["scales"])[lo]
            ref[pos[order], wb + sb:] = files[weight_map[spec["biases"]]].view(spec["biases"])[lo]

        res = {"rows_checked": int(rows.size)}
        if "rows" in entry["layouts"]:
            mm = np.memmap(out_dir / entry["layouts"]["rows"]["file"],
                           dtype=np.uint8, mode="r").reshape(total, stride)
            got = mm[rows]
            bad = int(np.count_nonzero(np.any(got != ref, axis=1)))
            res["rows_layout_mismatched_rows"] = bad
            res["rows_layout_exact"] = bad == 0
            del mm
        if "planar" in entry["layouts"]:
            pl = entry["layouts"]["planar"]
            w = np.memmap(out_dir / pl["weight"], dtype=np.uint8, mode="r").reshape(total, wb)
            s = np.memmap(out_dir / pl["scales"], dtype=np.uint8, mode="r").reshape(total, sb)
            b = np.memmap(out_dir / pl["biases"], dtype=np.uint8, mode="r").reshape(total, sb)
            bad_w = int(np.count_nonzero(np.any(w[rows] != ref[:, :wb], axis=1)))
            bad_s = int(np.count_nonzero(np.any(s[rows] != ref[:, wb:wb + sb], axis=1)))
            bad_b = int(np.count_nonzero(np.any(b[rows] != ref[:, wb + sb:], axis=1)))
            res.update(planar_mismatched_weight=bad_w, planar_mismatched_scales=bad_s,
                       planar_mismatched_biases=bad_b,
                       planar_exact=(bad_w == bad_s == bad_b == 0))
            del w, s, b
        report[layer_key] = res
    return report


def verify_full(model: Path, out_dir: Path, manifest, files, weight_map, block_rows):
    """Every byte of every packed layout compared with its source shard."""
    report = {}
    for layer_key, entry in manifest["layers"].items():
        info = entry["_info"]
        total, wb, sb = info["rows"], info["weight_bytes"], info["scale_bytes"]
        stride = wb + 2 * sb
        maps = {}
        if "rows" in entry["layouts"]:
            maps["rows"] = np.memmap(out_dir / entry["layouts"]["rows"]["file"],
                                     dtype=np.uint8, mode="r").reshape(total, stride)
        if "planar" in entry["layouts"]:
            pl = entry["layouts"]["planar"]
            maps["weight"] = np.memmap(out_dir / pl["weight"], dtype=np.uint8,
                                       mode="r").reshape(total, wb)
            maps["scales"] = np.memmap(out_dir / pl["scales"], dtype=np.uint8,
                                       mode="r").reshape(total, sb)
            maps["biases"] = np.memmap(out_dir / pl["biases"], dtype=np.uint8,
                                       mode="r").reshape(total, sb)
        bad = {k: 0 for k in maps}
        t0, done = time.time(), 0
        for si, spec in enumerate(info["specs"]):
            base = info["shard_offsets"][si]
            srcs = {f: files[weight_map[spec[f]]].view(spec[f])
                    for f in ("weight", "scales", "biases")}
            for start in range(0, spec["rows"], block_rows):
                end = min(start + block_rows, spec["rows"])
                g0, g1 = base + start, base + end
                w = srcs["weight"][start:end]
                s = srcs["scales"][start:end]
                b = srcs["biases"][start:end]
                if "rows" in maps:
                    chunk = maps["rows"][g0:g1]
                    if not (np.array_equal(chunk[:, :wb], w)
                            and np.array_equal(chunk[:, wb:wb + sb], s)
                            and np.array_equal(chunk[:, wb + sb:], b)):
                        bad["rows"] += int(np.count_nonzero(np.any(
                            chunk != np.concatenate([w, s, b], axis=1), axis=1)))
                for fam, src in (("weight", w), ("scales", s), ("biases", b)):
                    if fam in maps and not np.array_equal(maps[fam][g0:g1], src):
                        bad[fam] += int(np.count_nonzero(
                            np.any(maps[fam][g0:g1] != src, axis=1)))
                done += end - start
            if (si + 1) % 16 == 0:
                print(f"    full-verify shard {si+1:3d}/{len(info['specs'])}  "
                      f"{done:,} rows  {time.time()-t0:6.1f}s")
        report[layer_key] = {"rows_checked": int(total), "mismatched_rows": bad,
                             "exact": all(v == 0 for v in bad.values()),
                             "seconds": round(time.time() - t0, 1)}
        del maps
    return report


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <model>/ple-packed")
    ap.add_argument("--layout", choices=["rows", "planar", "both"], default="both")
    ap.add_argument("--block-rows", type=int, default=1 << 18)
    ap.add_argument("--verify-rows", type=int, default=200_000)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-full", action="store_true",
                    help="compare every byte instead of a random sample")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    model = args.model.expanduser().resolve()
    out_dir = (args.out or model / "ple-packed").expanduser()
    layouts = ("rows", "planar") if args.layout == "both" else (args.layout,)

    print(f"model    {model}")
    print(f"out      {out_dir}")
    info_all, files, weight_map = discover(model)
    if not info_all:
        sys.exit("no PLE n-gram embedding tensors found")

    planned = 0
    for layer_index, info in info_all.items():
        stride = info["weight_bytes"] + 2 * info["scale_bytes"]
        if "rows" in layouts:
            planned += info["rows"] * stride
        if "planar" in layouts:
            planned += info["rows"] * stride
    free = os.statvfs(out_dir.parent if out_dir.exists() else model).f_bavail * \
        os.statvfs(model).f_frsize
    print(f"layouts  {', '.join(layouts)}   planned {_fmt(planned)}, free {_fmt(free)}")
    if not args.verify_only and planned > free * 0.95:
        sys.exit("not enough free space")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        for k, v in manifest["layers"].items():
            v["_info"] = info_all[int(k)]
    else:
        manifest = {"pack_version": PACK_VERSION, "model": str(model),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "layers": {}}
        for layer_index, info in info_all.items():
            stats = repack_layer(layer_index, info, files, weight_map, model,
                                 out_dir, layouts, args.block_rows)
            entry = {
                "rows": info["rows"], "dims": info["dims"], "bits": info["bits"],
                "group_size": info["group_size"], "mode": "affine",
                "weight_bytes": info["weight_bytes"], "scale_bytes": info["scale_bytes"],
                "row_stride": stats["stride"],
                "shard_offsets": info["shard_offsets"],
                "shard_rows": [s["rows"] for s in info["specs"]],
                "weight_scale": weight_scale_value(info, files, weight_map, model),
                "prefix": info["prefix"],
                "seconds": round(stats["seconds"], 1),
                "layouts": {},
            }
            if "rows" in layouts:
                entry["layouts"]["rows"] = {"file": f"layer{layer_index}.rows.bin",
                                            "stride": stats["stride"],
                                            "order": ["weight", "scales", "biases"]}
            if "planar" in layouts:
                entry["layouts"]["planar"] = {
                    "weight": f"layer{layer_index}.weight.u32",
                    "scales": f"layer{layer_index}.scales.bf16",
                    "biases": f"layer{layer_index}.biases.bf16",
                    "weight_dtype": "uint32", "scale_dtype": "bfloat16",
                    "weight_cols": info["weight_bytes"] // 4,
                    "scale_cols": info["scale_bytes"] // 2,
                }
            entry["_info"] = info
            manifest["layers"][str(layer_index)] = entry
        payload = {k: ({kk: vv for kk, vv in v.items() if kk != "_info"}
                       if k == "layers" else v)
                   for k, v in manifest.items()}
        payload["layers"] = {k: {kk: vv for kk, vv in v.items() if kk != "_info"}
                             for k, v in manifest["layers"].items()}
        manifest_path.write_text(json.dumps(payload, indent=1))
        print(f"wrote {manifest_path}")

    if not args.no_verify:
        t0 = time.time()
        if args.verify_full:
            print("\nfull verification: every packed byte against its source shard")
            report = verify_full(model, out_dir, manifest, files, weight_map,
                                 args.block_rows)
        else:
            print(f"\nverifying {args.verify_rows:,} rows per layer against the source shards")
            report = verify(model, out_dir, manifest, files, weight_map, args.verify_rows)
        print(json.dumps(report, indent=1))
        print(f"verify took {time.time()-t0:.1f}s")
        ok = all(v.get("rows_layout_exact", True) and v.get("planar_exact", True)
                 and v.get("exact", True) for v in report.values())
        name = "verify_full.json" if args.verify_full else "verify.json"
        (out_dir / name).write_text(json.dumps(report, indent=1))
        print("BIT-EXACT" if ok else "MISMATCH")
        if not ok:
            sys.exit(1)

    for layer_key, entry in manifest["layers"].items():
        for name in ("rows",):
            if name in entry["layouts"]:
                p = out_dir / entry["layouts"][name]["file"]
                print(f"  {p.name:28s} {_fmt(p.stat().st_size)}")
        if "planar" in entry["layouts"]:
            for fam in ("weight", "scales", "biases"):
                p = out_dir / entry["layouts"]["planar"][fam]
                print(f"  {p.name:28s} {_fmt(p.stat().st_size)}")


if __name__ == "__main__":
    main()
