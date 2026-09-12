# ROUND2: the second decode round, measured

M5 Max, 128 GB, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`, greedy, one model
instance at a time on 127.0.0.1:8085. Written as the measurements were taken,
so the negative results are here with the positive ones and in the order they
happened.

The baseline this round starts from is `bench/decode/README.md`: short 54.6
tok/s with MTP off and 82.9 with the `omlx` chain at fixed depth 3; 64k 45.7
off and 52.0 to 54.3 on. The oMLX targets on the same machine are 84 to 91
short and 71 at 64k. Short was already at target when this round opened. 64k
was not, and that is what ROUND2 is about.

## How to reproduce

`run_round2.sh` starts a server, runs one mode, and stops everything it
started. One mode per invocation on purpose: arms have to alternate on a warm
GPU, because thermal and page-cache drift on this machine moves a decode number
by more than most of what this round is looking for.

    bash bench/decode/run_round2.sh <arm> <short|long|sweep|lossless> [--set ...]

Results append to `~/inference-server/staging/round2/results.jsonl`.

Two profiler fields are new this round and both are in `GET /metrics`:

- `position_acceptance`, P(draft *i* accepted | the chain reached *i*),
  reconstructed from the cycle profiles. It is the curve the long-context
  defect shows up in first, and nothing was reporting it.
- `?window=n`, which limits the decode summary to the last *n* cycles. The
  counters stay cumulative, so a caller that knows how many cycles its own
  request cost can read that request's numbers back without resetting
  anything. The context sweep needs it: seven contexts in one process.

## Reading the numbers

`accepted/cycle`, `acceptance` and the position curve are the steady figures
and are what the arguments below rest on. `tok/s` and `cycle ms` on this
machine drift by a factor that swamps most of the effects here (see step 1c),
so a throughput claim is only made where an arm was measured repeatedly inside
one process.

---

## Step 1. The MTP head at long context

### 1a. Head position alignment: no effect, and it cannot have one

**Hypothesis.** The head's KV cache starts empty at the first decode cycle, so
its RoPE offset is the count of decoded tokens while the trunk attends at
64,779 plus that count. Put the head back on the trunk's position line and
acceptance should recover.

**Where the positions come from.** `Qwen4ExpMTPModule.__call__` passed
`position_ids=None` into its one decoder layer, and with nothing passed
`Qwen3_5Attention` builds `arange(cache.offset, cache.offset + L)` from the
cache it was handed. For the head that cache is the head's own, so the
hypothesis is exactly right about the mechanism: at 64k the head really is
reading positions 0, 1, 2 while the trunk reads 64,779, 64,780, 64,781. oMLX
does not seed an offset either; what it does instead is fold the whole prompt
into the head cache during prefill (`patches/mlx_lm_mtp/prompt_priming.py`), so
its head's offset *is* the prompt length as a side effect of having the
prompt's keys.

**What was built.** `position_ids` threaded through the vendored MTP module,
`TitanQwenFlashNext.mtp_step(..., position_offset=)` building the same
`[3, 1, T]` mRoPE planes the attention builds for itself, and the drafter
computing the constant that turns a head offset into a trunk position.
`speculation.mtp_head_align_positions` selects it. Passing nothing is still the
identical graph, which is what makes the arm honest.

**Result: byte-identical drafts.** 64k, fixed depth 3, two arms in the same
session:

| arm | accepted/cycle | acceptance | position acceptance | accepted histogram |
|---|---|---|---|---|
| base | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 35 / 32 / 15 / 39 |
| positions aligned to the trunk | 1.479 | 50.1% | 0.723 / 0.454 / 0.328 | 35 / 32 / 15 / 39 |

Not close. Identical, to the cycle.

**Why.** RoPE is a relative encoding. Shifting every query and every key by the
same constant leaves every `q · k` unchanged, so an absolute position offset
applied to a cache the head attends to *in its entirety* is a no-op by
construction. Confirmed off the GPU as well, on a synthetic Qwen4-Exp with a
real head: an offset of 64,779 moves the head's logits by 1.0e-5 and moves no
argmax.

So the hypothesis is half right and the half it gets right is the half that
matters. The head's problem at 64k is not where it thinks it is. It is that its
cache holds a few hundred decoded tokens and nothing of the 64,779 the answer
depends on.

**Kept:** nothing. `mtp_head_align_positions` stays at its default of false and
is documented as a measured no-op; the `position_ids` parameter on the vendored
module stays, because it costs nothing and the next thing that wants to move
the head's positions should not have to re-derive the plumbing.

### 1b. Prompt priming: fold the prompt into the head during prefill

*(measurements in progress)*

### 1c. What the throughput numbers are worth on this machine

*(measurements in progress)*
