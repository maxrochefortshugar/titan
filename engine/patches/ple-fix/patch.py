#!/usr/bin/env python3
"""oMLX patch: serve the Qwen4-Exp PLE n-gram table from the repacked layout.

Stock oMLX (``qwen4_ple_ssd_offload``) gathers each 160-dim PLE row from three
separate safetensors tensors, preading whole 16 KB pages: ~93,000 pages and
1.5 GB of SSD traffic for the 3.3 MB a 2048-token chunk actually needs, 963 ms
with the GPU idle.  ``repack.py`` has already written a row-contiguous and a
plane-contiguous copy of the same bytes; this module swaps the lookup over.

Two modes, selected by ``OMLX_PLE_PACKED_MODE`` (default ``auto``):

``resident``  fault the three 32 GB planes into mx.arrays once, then every
              lookup is ``mx.take`` + ``mx.dequantize`` on the device.  This is
              what ``fuse_resident_ple_embeddings`` would do, except that it
              refuses to run below 192 GB of physical memory *and* needs a
              32 GB temporary next to the live shards; reading a pre-fused
              plane off disk needs neither, so the 192 GB gate never applies.
              Also removes the one host sync (``mx.eval`` on the n-gram ids)
              that stock oMLX does in the middle of every chunk.

``rows``      keep the table on SSD but read each row as one contiguous 100 B
              slice of ``layer1.rows.bin`` instead of three pages.  Roughly a
              3x cut in pages and SSD bytes.  Use this when the machine cannot
              spare the ~32 GB.

``auto``      resident when the table plus the already-loaded model fits under
              ``OMLX_PLE_PACKED_HEADROOM`` (default 12 GB spare), else rows.

Enable with ``OMLX_PLE_PACKED=1`` and call :func:`apply_ple_packed_patch`
*after* the model is loaded (it needs the live ``DiskBackedShardedEmbedding``
instances).  The patch is a no-op, returning 0, whenever the env toggle is off,
the pack is missing or stale, or the runtime is not in ``mmap`` PLE mode.

Numerics are bit-identical to stock: the same packed nibbles, scales and biases
go into the same ``mx.dequantize(..., mode="affine")`` and the same
``* weight_scale``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_APPLIED: dict[int, "PackedPLETable"] = {}
_PAGE = os.sysconf("SC_PAGE_SIZE")


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return _env_flag("OMLX_PLE_PACKED")


def pack_dir(model_path) -> Path:
    override = os.environ.get("OMLX_PLE_PACKED_DIR")
    if override:
        return Path(override).expanduser()
    return Path(model_path).expanduser() / "ple-packed"


# --------------------------------------------------------------------- table
class PackedPLETable:
    """The repacked table for one PLE layer, in whichever mode was chosen."""

    def __init__(self, directory: Path, entry: dict, mode: str):
        self.dir = directory
        self.rows = int(entry["rows"])
        self.dims = int(entry["dims"])
        self.bits = int(entry["bits"])
        self.group_size = int(entry["group_size"])
        self.weight_bytes = int(entry["weight_bytes"])
        self.scale_bytes = int(entry["scale_bytes"])
        self.stride = int(entry["row_stride"])
        self.shard_offsets = tuple(entry["shard_offsets"])
        self.mode = mode
        self.layouts = entry["layouts"]
        self.load_seconds = 0.0
        self.bytes_resident = 0
        # instrumentation, read by bench scripts
        self.last_rows = 0
        self.last_lookup_s = 0.0
        self.lookups = 0

        if mode == "resident":
            self._load_resident()
        elif mode == "rows":
            self._load_rows()
        else:
            raise ValueError(f"unknown packed PLE mode {mode!r}")

    # ---- resident -----------------------------------------------------
    def _load_resident(self):
        import mlx.core as mx

        pl = self.layouts["planar"]
        t0 = time.time()
        w_cols, s_cols = int(pl["weight_cols"]), int(pl["scale_cols"])
        wm = np.memmap(self.dir / pl["weight"], dtype="<u4", mode="r").reshape(self.rows, w_cols)
        sm = np.memmap(self.dir / pl["scales"], dtype="<u2", mode="r").reshape(self.rows, s_cols)
        bm = np.memmap(self.dir / pl["biases"], dtype="<u2", mode="r").reshape(self.rows, s_cols)
        self.weight = mx.array(wm)
        mx.eval(self.weight)
        del wm
        self.scales = mx.array(sm).view(mx.bfloat16)
        mx.eval(self.scales)
        del sm
        self.biases = mx.array(bm).view(mx.bfloat16)
        mx.eval(self.biases)
        del bm
        self.load_seconds = time.time() - t0
        self.bytes_resident = (self.weight.nbytes + self.scales.nbytes
                               + self.biases.nbytes)
        logger.info("PLE packed: %.1f GB resident in %.1fs",
                    self.bytes_resident / 1e9, self.load_seconds)

    def gather_device(self, indices):
        """indices: mx.array of any shape. Returns dequantized bf16 rows.

        No host sync: the ids stay on the device the whole way through.
        """
        import mlx.core as mx

        flat = indices.reshape(-1).astype(mx.uint32)
        w = mx.take(self.weight, flat, axis=0)
        s = mx.take(self.scales, flat, axis=0)
        b = mx.take(self.biases, flat, axis=0)
        return mx.dequantize(w, s, b, group_size=self.group_size,
                             bits=self.bits, mode="affine")

    # ---- rows (SSD, one contiguous read per row) -----------------------
    def _load_rows(self):
        path = self.dir / self.layouts["rows"]["file"]
        self._rows_path = path
        self._rows_file = path.open("rb")
        self._rows_mm = np.memmap(path, dtype=np.uint8, mode="r").reshape(
            self.rows, self.stride)
        self._seen_pages = bytearray(1 + (path.stat().st_size - 1) // _PAGE)
        self.bytes_resident = 0
        self.pages_read = 0
        self.pread_seconds = 0.0

    def _prefetch_pages(self, host: np.ndarray):
        """One page per row now, not three tensors' worth."""
        offsets = host.astype(np.int64) * self.stride
        pages = np.unique(np.concatenate(
            (offsets // _PAGE, (offsets + self.stride - 1) // _PAGE)))
        seen = np.frombuffer(self._seen_pages, dtype=np.uint8)
        fresh = pages[seen[pages] == 0]
        if fresh.size == 0:
            return
        t0 = time.perf_counter()
        fd = self._rows_file.fileno()
        from mlx_vlm.models.qwen4_exp.language import _PLE_IO_POOL

        def touch(page):
            offset = int(page) * _PAGE
            remaining = _PAGE
            while remaining > 0:
                chunk = os.pread(fd, remaining, offset + (_PAGE - remaining))
                if not chunk:
                    break
                remaining -= len(chunk)

        list(_PLE_IO_POOL.map(touch, (int(p) for p in fresh.tolist())))
        for page in fresh.tolist():
            self._seen_pages[page] = 1
        self.pages_read += int(fresh.size)
        self.pread_seconds += time.perf_counter() - t0

    def assemble_host(self, host: np.ndarray):
        """Copy the requested rows out of the packed mapping as three planes."""
        if host.size > 8:
            self._prefetch_pages(host)
        raw = np.array(self._rows_mm[np.asarray(host, dtype=np.intp)], copy=True)
        wb, sb = self.weight_bytes, self.scale_bytes
        return (raw[:, :wb].copy().view("<u4"),
                raw[:, wb:wb + sb].copy().view("<u2"),
                raw[:, wb + sb:].copy().view("<u2"))

    @staticmethod
    def _to_mx_bf16(u16: np.ndarray):
        import mlx.core as mx
        return mx.array(u16).view(mx.bfloat16)

    def dequantize_host(self, planes):
        import mlx.core as mx
        w, s, b = planes
        return mx.dequantize(mx.array(w), self._to_mx_bf16(s), self._to_mx_bf16(b),
                             group_size=self.group_size, bits=self.bits,
                             mode="affine")

    def close(self):
        if self.mode == "rows":
            self._rows_mm = None
            try:
                self._rows_file.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------- manifest
def load_manifest(model_path):
    directory = pack_dir(model_path)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no packed PLE table at {directory}")
    manifest = json.loads(path.read_text())
    for entry in manifest["layers"].values():
        for layout in entry["layouts"].values():
            names = ([layout["file"]] if "file" in layout
                     else [layout[f] for f in ("weight", "scales", "biases")])
            for name in names:
                if not (directory / name).is_file():
                    raise FileNotFoundError(f"packed PLE file missing: {name}")
    return directory, manifest


def choose_mode(entry, requested=None):
    requested = (requested or os.environ.get("OMLX_PLE_PACKED_MODE", "auto")).lower()
    if requested in {"resident", "rows"}:
        return requested
    if requested != "auto":
        raise ValueError(f"unknown OMLX_PLE_PACKED_MODE {requested!r}")
    if "planar" not in entry["layouts"]:
        return "rows"
    import mlx.core as mx

    need = entry["rows"] * entry["row_stride"]
    headroom = float(os.environ.get("OMLX_PLE_PACKED_HEADROOM_GB", "12")) * 1e9
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    budget = min(physical, mx.device_info()["max_recommended_working_set_size"])
    used = mx.get_active_memory()
    if used + need + headroom <= budget:
        return "resident"
    logger.info("PLE packed: falling back to rows mode "
                "(active %.1f GB + table %.1f GB + headroom %.1f GB > %.1f GB)",
                used / 1e9, need / 1e9, headroom / 1e9, budget / 1e9)
    return "rows"


# ------------------------------------------------------------------- install
def _install_on_embedding(embedding, table):
    """Rebind one DiskBackedShardedEmbedding instance to the packed table."""
    import mlx.core as mx

    if tuple(embedding.shard_offsets[:-1]) != tuple(table.shard_offsets):
        raise ValueError("packed shard offsets do not match the live embedding; "
                         "re-run repack.py against this checkpoint")
    if int(embedding.dims) != table.dims:
        raise ValueError(f"packed dims {table.dims} != runtime dims {embedding.dims}")

    embedding._packed = table
    embedding._packed_original_call = type(embedding).__call__

    if table.mode == "resident":
        def packed_call(self, indices):
            t0 = time.perf_counter()
            shape = indices.shape
            values = self._packed.gather_device(indices)
            values = values.astype(mx.bfloat16) * self.weight_scale
            out = values.reshape(*shape, self._packed.dims)
            self.rows_read = int(np.prod(shape)) if shape else 0
            self.last_uploads = 0
            self.last_prefetch_hit = True
            self._packed.last_rows = self.rows_read
            self._packed.last_lookup_s = time.perf_counter() - t0
            self._packed.lookups += 1
            return out

        def packed_prefetch(self, indices):
            return None            # nothing to prefetch: the table is resident

    else:
        def packed_call(self, indices):
            t0 = time.perf_counter()
            shape = indices.shape
            host = self._host_indices(indices)
            if host.size == 0:
                return mx.zeros((*shape, self._packed.dims), dtype=mx.bfloat16)
            pending = self._pending.pop(host.tobytes(), None)
            if pending is not None:
                planes = pending.result()
                self.last_prefetch_hit = True
            else:
                self.last_prefetch_hit = False
                planes = self._packed.assemble_host(host)
            values = self._packed.dequantize_host(planes)
            values = values.astype(mx.bfloat16) * self.weight_scale
            out = values.reshape(*shape, self._packed.dims)
            self.rows_read = int(host.size)
            self.last_uploads = 3
            self._packed.last_rows = self.rows_read
            self._packed.last_lookup_s = time.perf_counter() - t0
            self._packed.lookups += 1
            return out

        def packed_prefetch(self, indices):
            host = self._host_indices(indices)
            if host.size == 0:
                return
            with self._prefetch_lock:
                if self._prefetch_closed:
                    return
                key = host.tobytes()
                if key in self._pending:
                    return
                while len(self._pending) >= 2:
                    self._pending.pop(next(iter(self._pending))).cancel()
                self._pending[key] = self._prefetch_executor.submit(
                    self._packed.assemble_host, host)

    # bind per instance so untouched embeddings keep the stock path
    embedding.__dict__["__call__"] = packed_call.__get__(embedding)
    cls = type(embedding)
    if not getattr(cls, "_omlx_ple_packed", False):
        stock_call, stock_prefetch = cls.__call__, cls.prefetch

        def dispatch_call(self, indices):
            fn = self.__dict__.get("__call__")
            return fn(indices) if fn is not None else stock_call(self, indices)

        def dispatch_prefetch(self, indices):
            fn = self.__dict__.get("_packed_prefetch")
            return fn(indices) if fn is not None else stock_prefetch(self, indices)

        cls.__call__ = dispatch_call
        cls.prefetch = dispatch_prefetch
        cls._omlx_ple_packed = True
        cls._omlx_stock_call = stock_call
        cls._omlx_stock_prefetch = stock_prefetch
    embedding._packed_prefetch = packed_prefetch.__get__(embedding)
    # queued work from whichever path was active has the wrong payload shape
    with embedding._prefetch_lock:
        for pending in embedding._pending.values():
            future = pending[1] if isinstance(pending, tuple) else pending
            future.cancel()
        embedding._pending.clear()
    # The stock mmap readers stay open on purpose: paired ablations flip back to
    # the stock path, and an idle mmap costs nothing but reclaimable page cache.


def _ple_embeddings(model):
    language_model = getattr(model, "language_model", model)
    layers = getattr(getattr(language_model, "model", None), "layers", ())
    for index, layer in enumerate(layers):
        ple = getattr(layer, "ple", None)
        embedding = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        if embedding is not None:
            yield index, embedding


def apply_ple_packed_patch(model, model_path=None, *, mode=None, force=False) -> int:
    """Install the packed PLE lookup. Returns the number of layers patched."""
    if not force and not _enabled():
        return 0
    from mlx_vlm.models.qwen4_exp import language as L

    if model_path is None:
        model_path = L._PLE_RUNTIME_MODEL_PATH
    if model_path is None:
        model_path = os.environ.get("PROFILE_MODEL")
    if model_path is None:
        logger.warning("PLE packed: no model path; skipping")
        return 0

    try:
        directory, manifest = load_manifest(model_path)
    except FileNotFoundError as exc:
        logger.warning("PLE packed: %s; leaving the stock SSD path in place", exc)
        return 0

    patched = 0
    for layer_index, embedding in _ple_embeddings(model):
        if not isinstance(embedding, L.DiskBackedShardedEmbedding):
            logger.info("PLE packed: layer %d is %s, not the SSD-backed table; "
                        "skipping", layer_index, type(embedding).__name__)
            continue
        entry = manifest["layers"].get(str(layer_index))
        if entry is None:
            logger.warning("PLE packed: no packed layer %d in the manifest", layer_index)
            continue
        chosen = choose_mode(entry, mode)
        table = _APPLIED.get(layer_index)
        if table is None or table.mode != chosen:
            table = PackedPLETable(directory, entry, chosen)
            _APPLIED[layer_index] = table
        _install_on_embedding(embedding, table)
        patched += 1
        logger.info("PLE packed: layer %d -> %s mode", layer_index, chosen)
    return patched


def remove_ple_packed_patch(model) -> int:
    """Restore the stock lookup (keeps any resident buffers alive for re-use)."""
    removed = 0
    for _, embedding in _ple_embeddings(model):
        if embedding.__dict__.pop("__call__", None) is not None:
            embedding.__dict__.pop("_packed_prefetch", None)
            with embedding._prefetch_lock:
                for pending in embedding._pending.values():
                    future = pending[1] if isinstance(pending, tuple) else pending
                    future.cancel()
                embedding._pending.clear()
            removed += 1
    return removed


def packed_tables():
    return dict(_APPLIED)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="inspect the packed PLE table")
    ap.add_argument("--model", default=os.environ.get(
        "PROFILE_MODEL",
        "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
    args = ap.parse_args()
    directory, manifest = load_manifest(args.model)
    print(f"pack dir {directory}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "shard_offsets"}
                      for k, v in manifest["layers"].items()}, indent=1))
