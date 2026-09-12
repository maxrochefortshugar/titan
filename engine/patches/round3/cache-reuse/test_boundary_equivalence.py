# SPDX-License-Identifier: Apache-2.0
"""Resuming a hybrid prefill from a finer boundary reproduces a full prefill.

Synthetic two-layer stack at Flash-Next shapes: one GatedDeltaNet layer driven
by the production ``gated_delta_update`` kernel (recurrent state + causal conv
state) and one QSA-style attention layer with a plain KV cache, followed by a
linear head so the comparison lands on logits rather than hidden states.

Three runs over the same 2048-token chunk:
  A  full     one forward over all 2048 tokens
  B  split    1536 then 512, carrying live state across the split
  C  resume   1536, snapshot the state, drop everything, restore, then 512

C is what a 512-aligned boundary snapshot does at restore time. It must match A
on the final 512 tokens' logits. B isolates the pure re-association error of
splitting the scan from any snapshot round-trip error.

No model weights are loaded and nothing above ~200 MB is allocated.
Run: PYTHONPATH=/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
     ~/inference-server/kdev/bin/python test_boundary_equivalence.py
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update

# Flash-Next text config
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K = 128
HEAD_V = 128
CONV_K = 4
CONV_DIM = HEAD_K * NUM_K_HEADS * 2 + HEAD_V * NUM_V_HEADS  # 10240
HIDDEN = 2560
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HEAD_DIM = 256

T = 2048
FINE = 512
COARSE = 2048
SPLIT = 1536  # last 512-multiple inside the chunk

mx.random.seed(0)


def _rand(*shape, dtype=mx.bfloat16, scale=1.0):
    return (mx.random.normal(shape) * scale).astype(dtype)


class Weights:
    def __init__(self):
        self.conv_w = _rand(CONV_DIM, CONV_K, dtype=mx.float32, scale=0.25)
        self.A_log = mx.random.uniform(shape=(NUM_V_HEADS,)).astype(mx.float32) * -1.0
        self.dt_bias = (mx.random.uniform(shape=(NUM_V_HEADS,)) - 0.5).astype(mx.float32)
        self.wq = _rand(HIDDEN, QSA_HEADS * QSA_HEAD_DIM, scale=0.02)
        self.wk = _rand(HIDDEN, QSA_KV_HEADS * QSA_HEAD_DIM, scale=0.02)
        self.wv = _rand(HIDDEN, QSA_KV_HEADS * QSA_HEAD_DIM, scale=0.02)
        self.wo = _rand(QSA_HEADS * QSA_HEAD_DIM, HIDDEN, scale=0.02)
        self.head = _rand(HIDDEN, 4096, scale=0.02)


W = Weights()


def causal_conv(x, conv_state):
    """Depthwise causal conv1d. x is [B, T, C]; conv_state is [B, CONV_K-1, C]."""
    padded = mx.concatenate([conv_state, x], axis=1)
    new_state = mx.contiguous(padded[:, -(CONV_K - 1) :, :])
    acc = None
    for i in range(CONV_K):
        sl = padded[:, i : i + x.shape[1], :]
        term = sl * W.conv_w[:, i]
        acc = term if acc is None else acc + term
    return acc, new_state


def gdn_layer(x, state, conv_state):
    """One GDN layer over x [B, T, HIDDEN] -> [B, T, HIDDEN], carrying state."""
    B, L, _ = x.shape
    proj = mx.concatenate([x] * (CONV_DIM // HIDDEN) + [x[..., : CONV_DIM % HIDDEN]], axis=-1)
    conv_out, conv_state = causal_conv(proj, conv_state)
    conv_out = mx.tanh(conv_out * 0.5)
    qk = HEAD_K * NUM_K_HEADS
    q = conv_out[..., :qk].reshape(B, L, NUM_K_HEADS, HEAD_K)
    k = conv_out[..., qk : 2 * qk].reshape(B, L, NUM_K_HEADS, HEAD_K)
    v = conv_out[..., 2 * qk :].reshape(B, L, NUM_V_HEADS, HEAD_V)
    a = mx.sigmoid(v[..., 0].astype(mx.float32))          # [B, L, NUM_V_HEADS]
    b = mx.sigmoid(v[..., 1].astype(mx.float32)) * 0.5
    out, state = gated_delta_update(
        q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16),
        a, b, W.A_log, W.dt_bias, state=state, mask=None,
    )
    out = out.reshape(B, L, NUM_V_HEADS * HEAD_V)
    out = out[..., :HIDDEN] + out[..., HIDDEN : 2 * HIDDEN]
    return out.astype(mx.bfloat16), state, conv_state


def qsa_layer(x, kv):
    B, L, _ = x.shape
    q = (x @ W.wq).reshape(B, L, QSA_HEADS, QSA_HEAD_DIM).transpose(0, 2, 1, 3)
    k = (x @ W.wk).reshape(B, L, QSA_KV_HEADS, QSA_HEAD_DIM).transpose(0, 2, 1, 3)
    v = (x @ W.wv).reshape(B, L, QSA_KV_HEADS, QSA_HEAD_DIM).transpose(0, 2, 1, 3)
    if kv is not None:
        k = mx.concatenate([kv[0], k], axis=2)
        v = mx.concatenate([kv[1], v], axis=2)
    kv_new = (k, v)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(QSA_HEAD_DIM), mask="causal"
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, L, QSA_HEADS * QSA_HEAD_DIM)
    return (out @ W.wo).astype(mx.bfloat16), kv_new


def forward(x, state, conv_state, kv):
    h, state, conv_state = gdn_layer(x, state, conv_state)
    x = x + h
    h, kv = qsa_layer(x, kv)
    x = x + h
    return (x @ W.head), state, conv_state, kv


def fresh():
    return None, mx.zeros((1, CONV_K - 1, CONV_DIM), dtype=mx.float32), None


def snapshot(state, conv_state, kv):
    """Serialize the way a boundary snapshot does: fp32 recurrent, bf16 conv/KV."""
    return (
        mx.array(state.astype(mx.float32)),
        mx.array(conv_state.astype(mx.float32)),
        (mx.array(kv[0]), mx.array(kv[1])),
    )


def report(a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return (
        f"max abs {float(mx.max(d)):.3e}  mean abs {float(mx.mean(d)):.3e}  "
        f"differing bf16 elements {100 * float(mx.mean((d > 0).astype(mx.float32))):.2f}%  "
        f"rel RMS {float(mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(a.astype(mx.float32) ** 2))):.3e}"
    )


def main():
    x = _rand(1, T, HIDDEN, scale=1.0)

    # A: one unsplit forward over the whole chunk
    st, cs, kv = fresh()
    logits_a, _, _, _ = forward(x, st, cs, kv)
    mx.eval(logits_a)

    def split_run(at, roundtrip):
        st, cs, kv = fresh()
        _, st, cs, kv = forward(x[:, :at, :], st, cs, kv)
        mx.eval(st, cs, kv[0], kv[1])
        if roundtrip:
            st, cs, kv = snapshot(st, cs, kv)
            mx.eval(st, cs, kv[0], kv[1])
        tail, _, _, _ = forward(x[:, at:, :], st, cs, kv)
        mx.eval(tail)
        return tail

    print(f"shapes: T={T} fine={FINE} coarse={COARSE} gdn heads k/v "
          f"{NUM_K_HEADS}/{NUM_V_HEADS} dim {HEAD_K}/{HEAD_V}, logits [1,*,4096] bf16")
    print()
    print("1. Re-chunking error: splitting one 2048 forward at various points,")
    print("   each measured on the last 512 tokens against the unsplit run A.")
    ref = logits_a[:, SPLIT:, :]
    for at in (1024, 1280, SPLIT):
        tail = split_run(at, roundtrip=False)[:, SPLIT - at :, :]
        print("   split@%-5d %s" % (at, report(ref, tail)))
    print()
    print("2. Snapshot round trip at the 512-aligned boundary: serialize the GDN")
    print("   recurrent state and conv state to fp32 and the KV to its stored")
    print("   dtype, drop the live cache, restore, then run the tail.")
    b = split_run(SPLIT, roundtrip=False)
    c = split_run(SPLIT, roundtrip=True)
    print("   resume vs live-carry   %s" % report(b, c))
    print()

    exact = float(mx.max(mx.abs(b.astype(mx.float32) - c.astype(mx.float32)))) == 0.0
    scale = float(mx.max(mx.abs(ref.astype(mx.float32))))
    rech = float(mx.max(mx.abs(ref.astype(mx.float32)
                               - split_run(SPLIT, False).astype(mx.float32))))
    base = float(mx.max(mx.abs(ref.astype(mx.float32)
                               - split_run(1024, False)[:, SPLIT - 1024 :, :]
                               .astype(mx.float32))))
    print(f"logit scale {scale:.3f}")
    print(f"re-chunk at the fine boundary  max abs {rech:.3e} = {100*rech/scale:.3f}% of scale")
    print(f"re-chunk at an arbitrary point max abs {base:.3e} = {100*base/scale:.3f}% of scale")
    print(f"snapshot round trip bit-exact  {exact}")
    ok = exact and rech <= 2.0 * max(base, 1e-9)
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
