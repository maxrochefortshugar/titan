# SPDX-License-Identifier: Apache-2.0
"""PR #4020's NAX (C=16, cooperative-tensor matmul2d) GDN chunk kernel, JIT'd.

Ported from ml-explore/mlx pull/4020 commit c7e1a2a,
``mlx/backend/metal/kernels/gated_delta_update_nax.h`` :: ``gated_delta_fused_nax``.

mlx's JIT metal compiler has no include search path, so mlx's own steel NAX
header (``steel/gemm/nax.h``) is flattened by recursively inlining its
``mlx/...`` includes from mlx's shipped ``include/`` tree, then pasted into the
metal_kernel ``header=`` argument together with the PR's macro block. The
``[[kernel]]`` signature comes from metal_kernel; the body is verbatim.
"""

from __future__ import annotations

import os
import re

import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 16
SUPPORTED_HEADS = {(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)}


def _mlx_include() -> str:
    import mlx.core  # noqa: F401

    return os.path.join(os.path.dirname(os.path.abspath(mx.__file__)), "include")


def _flatten(path: str, inc: str, seen: set) -> str:
    p = os.path.normpath(path)
    if p in seen:
        return ""
    seen.add(p)
    out = []
    with open(p) as fh:
        for line in fh:
            m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
            if m:
                out.append(_flatten(os.path.join(inc, m.group(1)), inc, seen))
            else:
                out.append(line)
    return "".join(out)


def _header() -> str:
    inc = _mlx_include()
    steel = _flatten(
        os.path.join(inc, "mlx/backend/metal/kernels/steel/gemm/nax.h"), inc, set()
    )
    macros = open(os.path.join(_HERE, "_nax_macros.h")).read()
    if not RELAXED:
        # matmul2d_descriptor's 6th argument is relaxed_precision. The PR leaves
        # it true (as mlx's own steel NAX GEMM does); false costs speed but
        # keeps the fp32 accumulator honest.
        macros = macros.replace("transpose_b, true, Mode)", "transpose_b, false, Mode)")
    return (
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "#include <metal_tensor>\n"
        "using namespace metal;\n"
        "using namespace mpp;\n"
        "using namespace mpp::tensor_ops;\n"
        + steel
        + macros
    )


RELAXED = os.environ.get("OMLX_QWEN4_GDN_NAX_RELAXED", "1") == "1"
_KERNEL = None


def available() -> bool:
    return os.path.exists(os.path.join(_HERE, "_nax_body.metal")) and os.path.exists(
        os.path.join(_HERE, "_nax_macros.h")
    )


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        body = open(os.path.join(_HERE, "_nax_body.metal")).read()
        _KERNEL = mx.fast.metal_kernel(
            name="pr4020_gated_delta_fused_nax" + ("" if RELAXED else "_precise"),
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            header=_header(),
            source=body,
        )
    return _KERNEL


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


def gated_delta_fused_nax(q, k, v, g, beta, state=None):
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    return _kernel()(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", in_dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("C", CHUNK),
        ],
        grid=(32, Dv // CHUNK, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[in_dtype, mx.float32],
    )
