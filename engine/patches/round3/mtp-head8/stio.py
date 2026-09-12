"""Header-only safetensors readers plus a streaming writer.

Nothing here ever opens a whole shard. Tensors are read by byte range from the
local file, one at a time, so peak memory stays at one tensor.
"""

import json
import struct

import mlx.core as mx
import numpy as np

_NP = {
    "BF16": np.uint16,  # viewed as bfloat16 after the fact
    "F16": np.float16,
    "F32": np.float32,
    "U32": np.uint32,
    "U8": np.uint8,
    "I32": np.int32,
}

_MX = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "U32": mx.uint32}


def read_header(path):
    """Return (header_dict, data_start). Reads only the header bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_tensor(path, hdr, data_start, name):
    """Read one tensor by byte range and return an mx.array."""
    e = hdr[name]
    a, b = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        raw = f.read(b - a)
    dt = e["dtype"]
    arr = np.frombuffer(raw, dtype=_NP[dt]).reshape(e["shape"])
    out = mx.array(arr)
    if dt == "BF16":
        out = out.view(mx.bfloat16)
    return out


def read_raw_bf16(path, shape, offset=0, count=None):
    """Read a bf16 slab written by the range downloader."""
    n = int(np.prod(shape)) if count is None else count
    arr = np.fromfile(path, dtype=np.uint16, count=n, offset=offset)
    return mx.array(arr.reshape(shape)).view(mx.bfloat16)


_ST_DT = {mx.uint32: "U32", mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32"}
_ITEM = {mx.uint32: 4, mx.bfloat16: 2, mx.float16: 2, mx.float32: 4}


class StreamWriter:
    """Write a safetensors file tensor by tensor without holding all of it.

    Shapes and dtypes must be declared up front so the header can be written
    first; ``write`` then appends each tensor's bytes in declaration order.
    """

    def __init__(self, path, spec, metadata=None):
        # spec: list of (name, shape, mx dtype)
        self.path = path
        self.order = [s[0] for s in spec]
        hdr = {}
        if metadata:
            hdr["__metadata__"] = metadata
        off = 0
        for name, shape, dtype in spec:
            nbytes = int(np.prod(shape)) * _ITEM[dtype]
            hdr[name] = {
                "dtype": _ST_DT[dtype],
                "shape": list(shape),
                "data_offsets": [off, off + nbytes],
            }
            off += nbytes
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        pad = (-(len(blob) + 8)) % 8
        blob += b" " * pad
        self.f = open(path, "wb")
        self.f.write(struct.pack("<Q", len(blob)))
        self.f.write(blob)
        self.hdr = hdr
        self._i = 0
        self._cur = None

    def write(self, name, chunk):
        """Append bytes for ``name``. Call repeatedly to stream a big tensor."""
        if name != self._cur:
            if self._i >= len(self.order) or self.order[self._i] != name:
                exp = self.order[self._i] if self._i < len(self.order) else "<end>"
                raise ValueError(f"out of order: got {name}, expected {exp}")
            self._cur = name
            self._i += 1
        a = np.array(chunk.view(mx.uint16) if chunk.dtype == mx.bfloat16 else chunk)
        self.f.write(a.tobytes())

    def close(self):
        self.f.close()
