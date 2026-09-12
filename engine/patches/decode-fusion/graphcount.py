"""Count GPU primitives in an MLX graph via export_to_dot (proxy for launches)."""
import io, re, collections
import mlx.core as mx

# Primitives that are metadata-only in MLX (no kernel dispatched).
_FREE = {"Reshape", "Broadcast", "AsStrided", "Squeeze", "ExpandDims", "StopGradient",
         "Transpose", "Depends", "Split"}


def primitives(*outputs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *outputs)
    txt = buf.getvalue()
    names = re.findall(r'label ="([^"]+)"', txt)
    prims = [n for n in names if n and not n[0].isdigit() and "array" not in n]
    return prims


def count(*outputs):
    prims = primitives(*outputs)
    c = collections.Counter(prims)
    launches = sum(v for k, v in c.items() if k not in _FREE)
    return launches, c
