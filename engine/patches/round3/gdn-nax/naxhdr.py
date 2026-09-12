# SPDX-License-Identifier: Apache-2.0
"""Shared header builder for NAX probes and kernel variants."""
from __future__ import annotations
import os, re
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(os.path.dirname(HERE), "gdn-scan")


def mlx_include() -> str:
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


def steel_nax() -> str:
    inc = mlx_include()
    return _flatten(os.path.join(inc, "mlx/backend/metal/kernels/steel/gemm/nax.h"), inc, set())


PRELUDE = (
    "#include <metal_stdlib>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "#include <metal_tensor>\n"
    "using namespace metal;\n"
    "using namespace mpp;\n"
    "using namespace mpp::tensor_ops;\n"
)


def header(extra: str = "") -> str:
    return PRELUDE + steel_nax() + extra
