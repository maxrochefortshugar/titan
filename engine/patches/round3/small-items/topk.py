# SPDX-License-Identifier: Apache-2.0
"""Fused single-row top-K over a very wide logits vector (JIT Metal).

The MTP shortlist drafter (``kernels/round2/mtp/patch.py:105``
``_refresh_shortlist``) selects K candidate rows out of the 248,320-wide
lm_head logits once per decode cycle.  ``mx.topk`` / ``mx.argpartition`` cost
0.27 to 0.38 ms there, against 0.027 ms for ``mx.argmax`` over the same array,
because MLX's partition sorts far more than it has to.

This is a three-launch radix select over the order-preserving uint32 key of
each float:

  1. ``topk_hist``   2048-bin histogram of the top 11 key bits, threadgroup
                     private histograms combined with one device atomic per
                     non-empty bin.
  2. ``topk_split``  every threadgroup redundantly suffix-scans that histogram
                     to find the boundary bucket b1 and the count n_hi of
                     elements strictly above it, then emits those n_hi winners
                     directly and compacts the boundary bucket into a
                     candidate list.
  3. ``topk_refine`` one threadgroup selects the remaining K - n_hi winners out
                     of the candidate list exactly, with an 11-bit and then a
                     10-bit in-threadgroup histogram, so all 32 key bits are
                     resolved.

The result is a correct top-K for any input: no tolerance, no fallback.  Ties
at the boundary value are broken arbitrarily, exactly as ``mx.topk`` breaks
them arbitrarily, so the two agree as sets whenever the K-th largest value is
unique and always agree as multisets of values.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

// Order-preserving map float -> uint32 (total order on non-NaN floats).
inline uint fkey(float f) {
    uint b = as_type<uint>(f);
    return (b & 0x80000000u) ? (~b) : (b | 0x80000000u);
}

// Smallest bucket b such that sum_{c >= b} h[c] >= want.
// res[0] = b, res[1] = sum_{c > b} h[c].  Two-level so no thread walks all
// NB bins serially.  All threads must reach this call.
inline void pick_bucket(threadgroup uint *h,
                        threadgroup uint *part,
                        threadgroup uint *res,
                        uint NB, uint PARTS, uint want,
                        uint lid, uint nthreads) {
    const uint BS = NB / PARTS;
    for (uint p = lid; p < PARTS; p += nthreads) {
        uint s = 0;
        for (uint b = p * BS; b < (p + 1) * BS; ++b) { s += h[b]; }
        part[p] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint cum = 0;
        uint blk = 0;
        for (uint p = PARTS; p-- > 0; ) {
            if (cum + part[p] >= want) { blk = p; break; }
            cum += part[p];
            if (p == 0) { blk = 0; }
        }
        uint b = (blk + 1) * BS - 1;
        res[0] = blk * BS;
        res[1] = cum;
        while (true) {
            if (cum + h[b] >= want) { res[0] = b; res[1] = cum; break; }
            cum += h[b];
            if (b == blk * BS) { res[0] = b; res[1] = cum - h[b]; break; }
            --b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# ---------------------------------------------------------------- pass 1
_SRC_HIST = r"""
    threadgroup atomic_uint hs[NBINS];
    const uint lid = thread_position_in_threadgroup.x;
    const uint ntg = threads_per_threadgroup.x;
    for (uint b = lid; b < uint(NBINS); b += ntg) {
        atomic_store_explicit(&hs[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint gid = thread_position_in_grid.x;
    for (uint i = gid; i < uint(N); i += uint(NTHREADS)) {
        const uint key = fkey(float(x[i]));
        atomic_fetch_add_explicit(&hs[key >> 21], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device atomic_uint *hg = (device atomic_uint *)hist;
    for (uint b = lid; b < uint(NBINS); b += ntg) {
        const uint v = atomic_load_explicit(&hs[b], memory_order_relaxed);
        if (v != 0u) {
            atomic_fetch_add_explicit(&hg[b], v, memory_order_relaxed);
        }
    }
"""

# ---------------------------------------------------------------- pass 2
_SRC_SPLIT = r"""
    threadgroup uint h[NBINS];
    threadgroup uint part[PARTS];
    threadgroup uint res[2];
    const uint lid = thread_position_in_threadgroup.x;
    const uint ntg = threads_per_threadgroup.x;
    for (uint b = lid; b < uint(NBINS); b += ntg) { h[b] = hist[b]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pick_bucket(h, part, res, uint(NBINS), uint(PARTS), uint(K), lid, ntg);
    const uint b1 = res[0];
    const uint nhi = res[1];

    if (lid == 0 && threadgroup_position_in_grid.x == 0) {
        ctr[2] = b1;
        ctr[3] = nhi;
    }

    device atomic_uint *c_hi = (device atomic_uint *)ctr;
    device atomic_uint *c_cd = ((device atomic_uint *)ctr) + 1;
    const uint gid = thread_position_in_grid.x;
    for (uint i = gid; i < uint(N); i += uint(NTHREADS)) {
        const T v = x[i];
        const uint pre = fkey(float(v)) >> 21;
        if (pre > b1) {
            const uint p = atomic_fetch_add_explicit(c_hi, 1u, memory_order_relaxed);
            if (p < uint(K)) { oidx[p] = i; oval[p] = v; }
        } else if (pre == b1) {
            const uint p = atomic_fetch_add_explicit(c_cd, 1u, memory_order_relaxed);
            cand[p] = i;
        }
    }
"""

# ---------------------------------------------------------------- pass 3
_SRC_REFINE = r"""
    threadgroup uint h2[2048];
    threadgroup uint h3[1024];
    threadgroup uint part[PARTS];
    threadgroup uint res[2];
    threadgroup atomic_uint wa;      // writers above b2
    threadgroup atomic_uint wb;      // writers at b2, above b3
    threadgroup atomic_uint wc;      // writers at the boundary value (capped)
    const uint lid = thread_position_in_threadgroup.x;
    const uint ntg = threads_per_threadgroup.x;

    const uint nhi  = ctr[3];
    const uint C    = ctr[1];
    const uint take = uint(K) - nhi;

    for (uint i = lid; i < nhi; i += ntg) { oidx[i] = iidx[i]; oval[i] = ival[i]; }

    for (uint b = lid; b < 2048u; b += ntg) { h2[b] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint *a2 = (threadgroup atomic_uint *)h2;
    for (uint j = lid; j < C; j += ntg) {
        const uint key = fkey(float(x[cand[j]]));
        atomic_fetch_add_explicit(&a2[(key >> 10) & 2047u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pick_bucket(h2, part, res, 2048u, uint(PARTS), take, lid, ntg);
    const uint b2 = res[0];
    const uint nhi2 = res[1];
    const uint take2 = (take > nhi2) ? (take - nhi2) : 0u;

    for (uint b = lid; b < 1024u; b += ntg) { h3[b] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint *a3 = (threadgroup atomic_uint *)h3;
    for (uint j = lid; j < C; j += ntg) {
        const uint key = fkey(float(x[cand[j]]));
        if (((key >> 10) & 2047u) == b2) {
            atomic_fetch_add_explicit(&a3[key & 1023u], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pick_bucket(h3, part, res, 1024u, uint(PARTS), take2, lid, ntg);
    const uint b3 = res[0];
    const uint nhi3 = res[1];
    const uint take3 = (take2 > nhi3) ? (take2 - nhi3) : 0u;

    if (lid == 0) {
        atomic_store_explicit(&wa, nhi, memory_order_relaxed);
        atomic_store_explicit(&wb, nhi + nhi2, memory_order_relaxed);
        atomic_store_explicit(&wc, nhi + nhi2 + nhi3, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint limit = nhi + nhi2 + nhi3 + take3;
    for (uint j = lid; j < C; j += ntg) {
        const uint i = cand[j];
        const T v = x[i];
        const uint key = fkey(float(v));
        const uint mid = (key >> 10) & 2047u;
        const uint low = key & 1023u;
        uint p = 0xffffffffu;
        if (mid > b2) {
            p = atomic_fetch_add_explicit(&wa, 1u, memory_order_relaxed);
        } else if (mid == b2 && low > b3) {
            p = atomic_fetch_add_explicit(&wb, 1u, memory_order_relaxed);
        } else if (mid == b2 && low == b3) {
            p = atomic_fetch_add_explicit(&wc, 1u, memory_order_relaxed);
            if (p >= limit) { p = 0xffffffffu; }
        }
        if (p < uint(K)) { oidx[p] = i; oval[p] = v; }
    }
"""

_NBINS = 2048
_PARTS = 256
_TG = 256
_GROUPS = 32
_CACHE: dict = {}


def _kernels():
    k = _CACHE.get("k")
    if k is None:
        k = (
            mx.fast.metal_kernel(
                name="omlx_topk_hist",
                input_names=["x"],
                output_names=["hist"],
                header=_HEADER,
                source=_SRC_HIST,
            ),
            mx.fast.metal_kernel(
                name="omlx_topk_split",
                input_names=["x", "hist"],
                output_names=["oidx", "oval", "cand", "ctr"],
                header=_HEADER,
                source=_SRC_SPLIT,
            ),
            mx.fast.metal_kernel(
                name="omlx_topk_refine",
                input_names=["x", "cand", "ctr", "iidx", "ival"],
                output_names=["oidx", "oval"],
                header=_HEADER,
                source=_SRC_REFINE,
            ),
        )
        _CACHE["k"] = k
    return k


def fast_topk(x: mx.array, k: int):
    """Top-k values and indices of a single row.

    ``x``  one row of any float dtype, shape [V] or [1, V].
    Returns ``(values, indices)``: values in the input dtype, indices uint32,
    both shape [k], in no particular order (the same guarantee ``mx.topk``
    gives).
    """
    row = x.reshape(-1)
    v = int(row.size)
    k = max(1, min(int(k), v))
    kh, ks, kr = _kernels()
    nthreads = _TG * _GROUPS

    (hist,) = kh(
        inputs=[row],
        template=[("T", row.dtype), ("N", v), ("NTHREADS", nthreads),
                  ("NBINS", _NBINS)],
        grid=(nthreads, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(_NBINS,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )
    oidx, oval, cand, ctr = ks(
        inputs=[row, hist],
        template=[("T", row.dtype), ("N", v), ("NTHREADS", nthreads),
                  ("NBINS", _NBINS), ("PARTS", _PARTS), ("K", k)],
        grid=(nthreads, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(k,), (k,), (v,), (4,)],
        output_dtypes=[mx.uint32, row.dtype, mx.uint32, mx.uint32],
        init_value=0,
    )
    fidx, fval = kr(
        inputs=[row, cand, ctr, oidx, oval],
        template=[("T", row.dtype), ("K", k), ("PARTS", _PARTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, row.dtype],
        init_value=0,
    )
    return fval, fidx


def self_check() -> bool:
    """Compile the kernels and verify them against mx.topk at two shapes."""
    for v, k in ((4096, 64), (248320, 2048)):
        a = mx.random.normal((v,)).astype(mx.float32)
        mx.eval(a)
        val, idx = fast_topk(a, k)
        ref = mx.sort(mx.topk(a, k))
        got = mx.sort(val)
        mx.eval(val, idx, ref, got)
        if not bool(mx.all(got == ref).item()):
            return False
        if int(mx.min(mx.abs(a[idx] - val)).item()) != 0:
            return False
        if len(set(idx.tolist())) != k:
            return False
    return True


__all__ = ["fast_topk", "self_check"]
