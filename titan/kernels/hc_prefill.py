# SPDX-License-Identifier: MIT
"""Fused hyper-connection block for the prefill path.

Ported from ``engine/patches/round3/hc-fuse/kernel.py``.

The block mixes ``hc_count`` residual streams of ``hidden_size`` each. Written
out in plain ops it is roughly eleven Metal launches over a [T, hc*hidden]
bf16 stream::

    normed = grouped_rmsnorm(x)
    mix    = silu(down(normed) / hc)
    inj    = 2 * sigmoid(inject(normed) / hc)
    up_out = up(mix)
    mixed  = mean_g sigmoid(up_out)_g * normed_g

The fused implementation keeps the two quantized matmuls on MLX (they reach the
tensor units at M = 2048 and a hand written replacement would fall back to the
fp32 ALU rate) and fuses everything around them: the grouped norm runs on the
bf16 kernel, one epilogue kernel splits the projection results and applies both
activations, and one tail kernel applies the gate sigmoid, multiplies by the
normed stream and averages the streams where the [T, hc*hidden] up-projection
result is produced. Five launches instead of eleven.

Exactness: held the norm fixed, the fused block is bit-identical to the plain
chain on 99.998% of elements at the real shape, rrmse 5.5e-6, with the residual
ULP distance concentrated where the stream mean cancels. Getting that close
took reproducing
MLX's rounding rather than working in fp32: ``Sigmoid`` is the stable
``y = 1/(1+exp(|x|)); x<0 ? y : 1-y`` form whose every bf16 step rounds through
``bf16_math.h``'s fast intrinsics, and ``mx.mean`` over a bf16 axis carries a
bf16 accumulator (a sequential bf16 sum then one divide, which the tail kernel
matches exactly). The residue is the sigmoid: a literal transcription of MLX's
own expression into a JIT kernel still lands 1 ULP away on roughly 0.08% of
inputs, so ``mx.sigmoid`` is not bit-reproducible from ``mx.fast.metal_kernel``
on this build, and 0 ULP is not available against a plain-MLX reference. The
overlay measured 0 ULP because its reference was oMLX's own fused tail, another
JIT kernel, rather than the op chain.

On top of that the block inherits the norm's <= 1 ULP, which the stream mean
amplifies where the four terms cancel: mean 0.006 to 0.018 ULP end to end, with
large maxima only at cancellation.

Two options are off by default because both move results by more than 1 ULP:

``fp32_mean``
    accumulate the stream mean in fp32 instead of matching MLX's bf16 reduce.
    Strictly more accurate, and up to 11 ULP away from the plain chain.
``bank``
    merge the down and injection projections into one matmul over a
    concatenated weight bank, saving a 42 MB read. The low-rank rows come out
    bit identical; the ``hc`` injection rows do not (a different reduction
    order over K), 2 to 4 ULP. The bank is an explicit object the caller builds
    with :func:`build_bank` and owns; nothing is cached behind the caller's
    back.

State: the quantized projections and the concatenated bank are values the
caller passes in (:class:`QuantLinear`, :class:`HCWeights`, :class:`ConcatBank`).
There is no module-level weight cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from titan.kernels import grouped_rmsnorm_bf16 as gnorm
from titan.kernels.registry import KernelOp, ShapeClass, shape_class

__all__ = ["ConcatBank", "HCWeights", "OP", "QuantLinear", "build_bank",
           "chain_from_normed", "key", "metal", "reference", "supports"]


# ---------------------------------------------------------------------------
# caller-owned state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantLinear:
    """An affine-quantized projection: ``y = x @ dequant(weight).T``."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int = 4
    group_size: int = 64

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x, self.weight, self.scales, self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits,
        )

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])


@dataclass(frozen=True)
class ConcatBank:
    """The merged down|inject weight bank. Built by :func:`build_bank`."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int


@dataclass(frozen=True)
class HCWeights:
    """Everything one hyper-connection block needs, supplied by the caller."""

    norm_weight: mx.array          # [hc_count * hidden_size]
    down: QuantLinear              # [hc_lowrank, width]
    up: QuantLinear                # [width, hc_lowrank]
    hc_count: int
    hidden_size: int
    eps: float = 1e-6
    inject: Optional[QuantLinear] = None   # [hc_count, width]
    bank: Optional[ConcatBank] = None      # optional merged down|inject bank

    @property
    def width(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def hc_lowrank(self) -> int:
        return self.down.out_features


def build_bank(down: QuantLinear, inject: QuantLinear) -> ConcatBank:
    """Concatenate the two projections into one bank, or raise if incompatible."""
    if down.bits != inject.bits or down.group_size != inject.group_size:
        raise ValueError("down and inject banks disagree on bits/group_size")
    if down.weight.dtype != inject.weight.dtype:
        raise ValueError("down and inject banks disagree on weight dtype")
    if down.weight.shape[1] != inject.weight.shape[1]:
        raise ValueError("down and inject banks disagree on packed width")
    bank = ConcatBank(
        weight=mx.concatenate([down.weight, inject.weight], axis=0),
        scales=mx.concatenate([down.scales, inject.scales], axis=0),
        biases=mx.concatenate([down.biases, inject.biases], axis=0),
        bits=int(down.bits),
        group_size=int(down.group_size),
    )
    mx.eval(bank.weight, bank.scales, bank.biases)
    return bank


# ---------------------------------------------------------------------------
# Metal sources
# ---------------------------------------------------------------------------

# MLX's Sigmoid is the stable form, and with T = bfloat16_t every intermediate
# rounds to bf16 after each step. Reproducing that rounding exactly is what
# keeps the fused block bit-identical to the elementwise chain.
_HEADER = r"""
using namespace metal;

template <typename T>
inline float hc_sigmoid(float x) {
    // MLX's Sigmoid (unary_ops.h) is y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y,
    // and with T = bfloat16_t every step goes through bf16_math.h, which casts
    // to float, calls the FAST intrinsic, and rounds back to bf16. Both halves
    // matter: metal::exp on a float is the precise one and lands 2 ULP away.
    const float e = float(T(metal::fast::exp(metal::abs(x))));
    const float d = float(T(1.0f + e));
    const float y = float(T(metal::fast::divide(1.0f, d)));
    return (x < 0.0f) ? y : float(T(1.0f - y));
}
"""

# Both projection results, both activations, one launch.
_EPI_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    const int width = R + (INJ ? HC : 0);
    const int c = int(idx) % width;
    const int row = int(idx) / width;
    const float raw = (c < R) ? float(d[(size_t)row * R + c])
                              : float(j[(size_t)row * HC + (c - R)]);
    const float v = float(T(raw / float(HC)));
    const float s = hc_sigmoid<T>(v);
    if (c < R) {
        act[(size_t)row * R + c] = T(v * s);
    } else {
        inj[(size_t)row * HC + (c - R)] = T(2.0f * s);
    }
"""

# Concat variant: one [rows, R + HC] result to split.
_EPI_COMB_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    const int c = int(idx) % (R + HC);
    const int row = int(idx) / (R + HC);
    const float v = float(T(float(comb[idx]) / float(HC)));
    const float s = hc_sigmoid<T>(v);
    if (c < R) {
        act[(size_t)row * R + c] = T(v * s);
    } else {
        inj[(size_t)row * HC + (c - R)] = T(2.0f * s);
    }
"""

# Gate, stream product and the mean over the streams in one pass.
# ACC32 = 0 reproduces MLX's reduction, which carries a bf16 accumulator for a
# bf16 input, and is therefore bit identical to the plain tail.
_TAIL_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    if (idx >= (uint)TOTAL) return;
    // VEC consecutive hidden lanes per thread: 2 * VEC bytes per load keeps the
    // simdgroup's requests wide over the two large reads.
    const int h = int(idx % (uint)(H / VEC)) * VEC;
    const int row = int(idx / (uint)(H / VEC));
    const size_t base = (size_t)row * (size_t)(H * HC) + (size_t)h;
    float acc[VEC];
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;
    for (int g = 0; g < HC; ++g) {
        const size_t o = base + (size_t)(g * H);
        for (int v = 0; v < VEC; ++v) {
            const float gate = hc_sigmoid<T>(float(up[o + v]));
            const float prod = float(T(gate * float(xn[o + v])));
            acc[v] = ACC32 ? (acc[v] + prod) : float(T(acc[v] + prod));
        }
    }
    const size_t ob = (size_t)row * (size_t)H + (size_t)h;
    for (int v = 0; v < VEC; ++v) mixed[ob + v] = T(acc[v] / float(HC));
"""

_KERNELS: dict[str, object] = {}


def _kernel(name, input_names, output_names, source):
    k = _KERNELS.get(name)
    if k is None:
        k = mx.fast.metal_kernel(
            name=name, input_names=input_names, output_names=output_names,
            header=_HEADER, source=source, ensure_row_contiguous=True,
        )
        _KERNELS[name] = k
    return k


def _epilogue(d, j, rows, lowrank, hc, dtype):
    inj_flag = 0 if j is None else 1
    total = rows * (lowrank + (hc if inj_flag else 0))
    tg = 256
    out = _kernel("titan_hc_epilogue", ["d", "j"], ["act", "inj"], _EPI_SOURCE)(
        inputs=[d, d if j is None else j],
        template=[("T", dtype), ("R", lowrank), ("HC", hc), ("INJ", inj_flag),
                  ("TOTAL", total)],
        grid=((total + tg - 1) // tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )
    return (out[0], None) if j is None else (out[0], out[1])


def _epilogue_bank(comb, rows, lowrank, hc, dtype):
    total = rows * (lowrank + hc)
    tg = 256
    act, inj = _kernel(
        "titan_hc_epilogue_bank", ["comb"], ["act", "inj"], _EPI_COMB_SOURCE
    )(
        inputs=[comb],
        template=[("T", dtype), ("R", lowrank), ("HC", hc), ("TOTAL", total)],
        grid=((total + tg - 1) // tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )
    return act, inj


def _tail(up_out, normed, rows, hc, hidden, dtype, acc32=False, vec=4):
    vec = vec if hidden % vec == 0 else 1
    total = rows * (hidden // vec)
    tg = 256
    return _kernel("titan_hc_tail", ["up", "xn"], ["mixed"], _TAIL_SOURCE)(
        inputs=[up_out, normed],
        template=[("T", dtype), ("H", hidden), ("HC", hc), ("VEC", vec),
                  ("ACC32", 1 if acc32 else 0), ("TOTAL", total)],
        grid=((total + tg - 1) // tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[dtype],
    )[0]


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def chain_from_normed(normed, w: HCWeights, *, fp32_mean=False):
    """Everything after the norm, in plain MLX ops. ``normed``: [rows, width].

    Exposed so a test can hold the norm fixed and check that the fused block is
    bit-identical over the part it claims to be bit-identical over.
    """
    rows = normed.shape[0]
    hc, hidden = w.hc_count, w.hidden_size
    dtype = normed.dtype
    mix_raw = (w.down(normed) / hc).astype(dtype)
    mix = (mix_raw * mx.sigmoid(mix_raw)).astype(dtype)
    up_out = w.up(mix)
    gated = (mx.sigmoid(up_out).astype(dtype) * normed).astype(dtype)
    acc = gated.reshape(rows, hc, hidden)
    if fp32_mean:
        mixed = mx.mean(acc.astype(mx.float32), axis=1).astype(dtype)
    else:
        mixed = mx.mean(acc, axis=1).astype(dtype)
    if w.inject is None:
        return mixed, None
    raw = (w.inject(normed) / hc).astype(dtype)
    injection = (2 * mx.sigmoid(raw)).astype(dtype)
    return mixed, injection


def reference(x, w: HCWeights, *, fp32_mean=False, use_bank=False):
    """The plain-MLX block. ``x``: [B, S, hc*hidden]. Returns the same shape
    contract as :func:`metal`: ``mixed`` alone with no injection, else
    ``(mixed, x, injection)``."""
    batch, seq = x.shape[0], x.shape[1]
    rows = batch * seq
    flat = x.reshape(rows, w.width)
    normed = gnorm.reference(flat, w.norm_weight, groups=w.hc_count, eps=w.eps)
    mixed, injection = chain_from_normed(normed, w, fp32_mean=fp32_mean)
    mixed = mixed.reshape(batch, seq, w.hidden_size)
    if injection is None:
        return mixed
    return mixed, x, injection.reshape(batch, seq, w.hc_count)


def metal(x, w: HCWeights, *, fp32_mean=False, use_bank=False):
    """Five launches. Same signature and return contract as :func:`reference`."""
    hc, hidden, lowrank = w.hc_count, w.hidden_size, w.hc_lowrank
    dtype = x.dtype
    batch, seq = x.shape[0], x.shape[1]
    rows = batch * seq
    flat = x.reshape(rows, w.width)

    normed = gnorm.metal(flat, w.norm_weight, groups=hc, eps=w.eps)

    if use_bank:
        if w.bank is None:
            raise ValueError("use_bank=True needs HCWeights.bank; call build_bank")
        b = w.bank
        comb = mx.quantized_matmul(
            normed, b.weight, b.scales, b.biases,
            transpose=True, group_size=b.group_size, bits=b.bits,
        )
        act, inj = _epilogue_bank(comb, rows, lowrank, hc, dtype)
    else:
        act, inj = _epilogue(
            w.down(normed),
            None if w.inject is None else w.inject(normed),
            rows, lowrank, hc, dtype,
        )

    up_out = w.up(act)
    mixed = _tail(up_out, normed, rows, hc, hidden, dtype, acc32=fp32_mean)
    mixed = mixed.reshape(batch, seq, hidden)
    if w.inject is None and not use_bank:
        return mixed
    return mixed, x, inj.reshape(batch, seq, hc)


def key(x, w: HCWeights, *, fp32_mean=False, use_bank=False) -> ShapeClass:
    return shape_class(
        x, w.norm_weight, w.down.weight, w.up.weight,
        extra=(w.hc_count, w.hidden_size, w.hc_lowrank, w.down.bits, w.up.bits,
               w.inject is not None, bool(fp32_mean), bool(use_bank)),
    )


def supports(k: ShapeClass) -> bool:
    if k.device != "gpu" or len(k.shapes) < 4 or len(k.extra) != 8:
        return False
    if k.dtypes[0] not in (mx.bfloat16, mx.float16):
        return False
    xs = k.shapes[0]
    hc, hidden = k.extra[0], k.extra[1]
    if len(xs) != 3 or xs[2] != hc * hidden:
        return False
    # the tail vectorises over the hidden axis, the epilogue over rows*width
    return hidden % 4 == 0 and hc >= 1


OP = KernelOp(
    name="hc_prefill",
    aliases=("hyper_connection_block",),
    reference_fn=reference,
    fast_fn=metal,
    key=key,
    supports_key=supports,
    tolerance=2.0,
    shapes=(
        {"T": 8, "hc": 4, "hidden": 256, "lowrank": 32, "bits": 4},
        {"T": 64, "hc": 4, "hidden": 256, "lowrank": 32, "bits": 4},
        {"T": 2048, "hc": 4, "hidden": 2560, "lowrank": 320, "bits": 5},
    ),
    exactness="99.998% bit-identical downstream of the norm (rrmse 5.5e-6); the norm itself <= 1 ULP",
    source="engine/patches/round3/hc-fuse/REPORT.md",
)
