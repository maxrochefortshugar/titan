# SPDX-License-Identifier: MIT
"""Exactness measures shared by the kernel tests.

Every test in this directory compares one op's fast implementation against its
reference at the exactness class the overlay report established. Two measures
are used and they are not interchangeable:

``ulp_distance``
    bit-pattern distance in the storage dtype, under the standard total order
    on floats. This is the right measure for elementwise work, where a relative
    error is meaningless wherever terms cancel.
``rrmse``
    ``||x - ref|| / ||ref||``. The right measure for the chunked scan, where
    the question is whether the state drifts, and for the int8 gather, whose
    error is a quantisation floor rather than a rounding difference.

Shapes are small on purpose: the workbench may be using the GPU, so nothing
here allocates more than a few hundred MB, and the real Flash-Next shapes live
behind ``-m slow``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# float dtype -> (mlx view dtype, width in bits, signed type wide enough to
# hold the ordered key without wrapping)
_ORDERED = {
    mx.bfloat16: (mx.uint16, 16, np.int32),
    mx.float16: (mx.uint16, 16, np.int32),
    mx.float32: (mx.uint32, 32, np.int64),
}


def _total_order(x: mx.array) -> np.ndarray:
    """Map floats onto a monotonically ordered signed integer.

    Positive floats keep their bit pattern; negative floats are reflected below
    zero, so the distance between two adjacent representable values is one in
    both half-lines and across zero.
    """
    view, width, itype = _ORDERED[x.dtype]
    bits = np.array(x.view(view), copy=True).astype(itype)
    sign_bit = itype(1) << itype(width - 1)
    return np.where(bits >= sign_bit, sign_bit - 1 - bits, bits)


def ulp_distance(got: mx.array, ref: mx.array) -> np.ndarray:
    """Elementwise bit-pattern distance. Both arrays must share a dtype."""
    assert got.dtype == ref.dtype, f"{got.dtype} vs {ref.dtype}"
    mx.eval(got, ref)
    return np.abs(_total_order(got) - _total_order(ref))


def max_ulp(got: mx.array, ref: mx.array) -> int:
    return int(ulp_distance(got, ref).max())


def bit_identical(got: mx.array, ref: mx.array) -> bool:
    mx.eval(got, ref)
    return bool(mx.all(got == ref).item())


def rrmse(got: mx.array, ref: mx.array) -> float:
    g = got.astype(mx.float32)
    r = ref.astype(mx.float32)
    return float(mx.sqrt(mx.mean(mx.square(g - r)) / (mx.mean(mx.square(r)) + 1e-30)))


def rel_rms(got: mx.array, ref: mx.array) -> float:
    """Error RMS as a fraction of the reference RMS. Same as rrmse; named
    separately where a report quotes it as a percentage of output RMS."""
    return rrmse(got, ref)
