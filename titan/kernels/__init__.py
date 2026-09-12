"""Kernels: the op registry and our Metal implementations.

Every op is registered here with a reference implementation in plain mlx ops and
an optional ``mx.fast.metal_kernel`` fast path. The registry decides which one
runs; the exactness test decides whether the fast one is allowed to exist.

House rules, carried over from the overlay because they worked:

- Bit-identical, or within one bf16 ULP, against the reference. Anything looser
  is a different model and is not deployed.
- Every op keeps its exactness test in ``tests/kernels/test_<op>.py``, and the
  test runs over the shape list the op declares, not over a shape the author
  liked.
- Every op keeps a microbenchmark, and a kernel is judged in situ, on a paired
  A/B with a cooldown. Isolated microbenchmarks overstated small-kernel launch
  cost by roughly 40x and cost a round of work.
- A fast path that raises or meets an unsupported shape falls back and counts.

Ops the engine expects to find registered, from the overlay's measured set:

    ngram_gather            packed contiguous row read, bit-exact
    rms_norm_grouped        bf16 with fp32 accumulate, within 1 ULP
    gdn_norm_gate           fused grouped norm plus sigmoid gate, bit-identical
    gdn_scan_chunked        chunked delta scan, state error ~6e-7
    moe_weighted_sum        fused top_k=10 weighted sum, bit-identical
    moe_gather_gate_up      int8 x int4 expert gather for gate_up
    hyper_connection_block  fused prefill hyper-connection, bit-identical
    qsa_sparse_decode       sparse attention for batched decode, 1 ULP
    mtp_shortlist           top-K draft head for draft steps 2..k
    verify_accept           in-graph acceptance: compare, cumprod, sum

``verify_accept`` is the decode cycle's own op and is not in this package yet;
everything else on that list is registered, under both the name above and the
module name it lives in. The op table, the exactness class each op was measured
at, and the source report it came from are in ``README.md`` next door.

Nothing here imports oMLX, mlx_vlm or mlx_lm, and nothing monkeypatches. State
that has to persist between calls (weight tables, tile tables, reader pools,
memory maps) lives in objects the caller constructs and owns; the only cache
the library keeps is the compiled-kernel handle.
"""

from titan.kernels import registry
from titan.kernels.registry import build_registry, reference_only

__all__ = ["build_registry", "reference_only", "registry"]
