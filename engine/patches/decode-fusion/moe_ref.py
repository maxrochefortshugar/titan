"""Faithful stock MoE routed-MLP block for Qwen3.8-Flash-Next shapes.

Mirrors mlx_lm.models.qwen3_moe.Qwen3MoeSparseMoeBlock + SwitchGLU, with the
oMLX gate+up fusion (qwen35_moe_gate_up.py) already applied, which is what the
production path actually runs.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear, SwitchGLU, _gather_sort, _scatter_unsort)
from mlx_lm.models.activations import swiglu

HIDDEN = 2560
INTER = 640
NEXP = 512
TOPK = 10


def _quantized_switch_linear(E, out_dims, in_dims, gs=64, bits=4, chunk=64, seed=0):
    """Build a QuantizedSwitchLinear without materialising the full fp weight."""
    ql = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(ql)
    ws, ss, bs = [], [], []
    key = mx.random.key(seed)
    scale = math.sqrt(1.0 / in_dims)
    for i in range(0, E, chunk):
        n = min(chunk, E - i)
        sub = mx.random.uniform(-scale, scale, (n, out_dims, in_dims),
                                key=mx.random.split(key, E // chunk + 1)[i // chunk]
                                ).astype(mx.bfloat16)
        q, s, b = mx.quantize(sub, gs, bits)
        mx.eval(q, s, b)
        del sub
        ws.append(q); ss.append(s); bs.append(b)
        mx.clear_cache()
    ql.weight = mx.concatenate(ws, axis=0)
    ql.scales = mx.concatenate(ss, axis=0)
    ql.biases = mx.concatenate(bs, axis=0)
    mx.eval(ql.weight, ql.scales, ql.biases)
    del ws, ss, bs
    mx.clear_cache()
    ql.group_size, ql.bits, ql.mode = gs, bits, "affine"
    ql.freeze()
    return ql


class FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU with the oMLX gate_up_proj fusion baked in (stock kernels)."""

    def __init__(self, E=NEXP, hidden=HIDDEN, inter=INTER, gs=64, bits=4, seed=0):
        super().__init__()
        self.gate_up_proj = _quantized_switch_linear(E, 2 * inter, hidden, gs, bits, seed=seed)
        self.down_proj = _quantized_switch_linear(E, hidden, inter, gs, bits, seed=seed + 1)
        self.activation = None

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
        x = self.down_proj(swiglu(x_gate, x_up), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SparseMoeBlock(nn.Module):
    """Qwen3MoeSparseMoeBlock, routed experts only (no shared expert)."""

    def __init__(self, E=NEXP, hidden=HIDDEN, inter=INTER, top_k=TOPK,
                 gs=64, bits=4, seed=0, quantized_gate=True):
        super().__init__()
        self.num_experts, self.top_k, self.norm_topk_prob = E, top_k, True
        g = nn.Linear(hidden, E, bias=False)
        g.weight = (mx.random.normal((E, hidden)) * 0.02).astype(mx.bfloat16)
        if quantized_gate:
            g = g.to_quantized(group_size=gs, bits=bits)
        self.gate = g
        self.switch_mlp = FusedGateUpSwitchGLU(E, hidden, inter, gs, bits, seed)
        ln = nn.RMSNorm(hidden)
        ln.weight = ln.weight.astype(mx.bfloat16)
        self.input_layernorm = ln

    def route(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return inds, scores

    def __call__(self, x):
        inds, scores = self.route(x)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
