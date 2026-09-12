# SPDX-License-Identifier: Apache-2.0
"""PR #4020's NAX (C=16 cooperative-tensor) GDN kernel with split-precision matmuls.

matmul2d(relaxed_precision=true) rounds both operands to an 11-bit significand
and accumulates in fp32 (probe_format.py, probe_accum.py). Each matmul site can
be switched to a three-pass hi/lo split that recovers about 22 bits, selected by
a bitmask template argument so the cost is paid only where it buys accuracy.

Bits: 0 KK^T, 1 WY Neumann iteration, 2 W panel, 3 U, 4 W S^T, 5 Q K^T,
      6 Q S^T, 7 out += (QK^T tri) delta, 8 state update.
"""

from __future__ import annotations

import os

import mlx.core as mx

import naxhdr

_HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 16
SUPPORTED_HEADS = {(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)}

SITES = ["KKt", "wy", "wpanel", "u", "ws", "qkt", "qs", "out", "state"]


def mask(**modes) -> int:
    """Build the site mask. modes: site name -> 0 none, 1 three-pass,
    2 correct the left operand only, 3 correct the right operand only."""
    m = 0
    for name, mode in modes.items():
        m |= (mode & 3) << (2 * SITES.index(name))
    return m


# Accuracy-first default: three-pass at every site whose operands are not
# already bf16-exact. Cheaper masks are measurably worse (see REPORT.md).
DEFAULT = mask(wy=1, wpanel=1, u=1, ws=1, qs=1, out=1, state=1)
# best partial: same but a two-pass correction of W at the W S^T site, 1.8x
# cheaper and 30x less accurate
FAST = mask(wy=1, wpanel=1, u=1, ws=2, qs=1, out=1, state=1)
DEFAULT_SPLIT = int(os.environ.get("OMLX_QWEN4_GDN_NAX_SPLIT", str(DEFAULT)), 0)

_KERNELS: dict[int, object] = {}


def _kernel(split: int, bfops: int = 0):
    key = (split, bfops)
    if key not in _KERNELS:
        header = naxhdr.header(open(os.path.join(_HERE, "_nax2_macros.h")).read())
        body = open(os.path.join(_HERE, "_nax2_body.metal")).read()
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"gdn_nax_split_{split:06x}_{bfops}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            header=header,
            source=body,
        )
    return _KERNELS[key]


def supported(q, k, v, g, beta, state) -> bool:
    if q.ndim != 4 or v.ndim != 4 or g.ndim != 3:
        return False
    _B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    return (
        Dk == 128
        and Dv == 128
        and (Hk, Hv) in SUPPORTED_HEADS
        and q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and (state is None or state.dtype == mx.float32)
        and T > 1
    )


def gated_delta_fused_nax2(q, k, v, g, beta, state=None, split: int | None = None,
                           bfops: int = 0):
    if split is None:
        split = DEFAULT_SPLIT
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    return _kernel(split, bfops)(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", in_dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("C", CHUNK),
            ("SPLIT", split),
            ("BFOPS", bfops),
        ],
        grid=(32, Dv // CHUNK, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[in_dtype, mx.float32],
    )
