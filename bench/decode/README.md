# W4.1: the MTP drafter, measured

What the drafter and the depth policy are worth on this machine, arm by arm.
Four arms, two contexts, one lossless check. Every number below came from the
commands in this file, one model instance at a time, on 127.0.0.1:8085.

## Running it

The server runs from a tree containing only HEAD plus the W4.1 change, so that
the concurrent cache and scheduler work in the repo cannot move these numbers:

    cd ~/Engineering/titan
    rm -rf /tmp/w41src && mkdir -p /tmp/w41src && git archive HEAD | tar -x -C /tmp/w41src
    cp titan/adapters/mlx/{drafter,model,backend}.py /tmp/w41src/titan/adapters/mlx/
    cp titan/engine/decode_cycle.py /tmp/w41src/titan/engine/
    # plus the two hunks in titan/config/{schema,wiring}.py that add
    # speculation.mtp_chain, speculation.draft_p_min, build_drafter and
    # build_verifier

Then one arm at a time. `run_arms.sh` starts the server, waits for `/health`,
runs the lossless probe and the short arm, restarts for the 64k arm, and stops
everything it started:

    bash bench/decode/run_arms.sh a_mtp_off  --set speculation.enabled=false
    bash bench/decode/run_arms.sh b_chain_d3 --set speculation.enabled=true \
        --set speculation.adaptive_depth=false --set speculation.mtp_depth_max=3 \
        --set 'speculation.mtp_chain="head_output"'
    bash bench/decode/run_arms.sh c_omlx_d3  --set speculation.enabled=true \
        --set speculation.adaptive_depth=false --set speculation.mtp_depth_max=3 \
        --set 'speculation.mtp_chain="omlx"'
    bash bench/decode/run_arms.sh d_ev_pmin  --set speculation.enabled=true \
        --set speculation.adaptive_depth=true --set speculation.mtp_depth_max=6 \
        --set 'speculation.mtp_chain="head_output"' --set speculation.draft_p_min=0.1

Results land in `~/inference-server/staging/w41/results.jsonl`, completions in
`lossless-<arm>.txt`, server logs in `serve-<arm>.log`.

The measurement itself is `mtp_ab.py`, which also runs standalone against an
already-running server:

    python bench/decode/mtp_ab.py --arm b_chain_d3 --mode short
    python bench/decode/mtp_ab.py --arm b_chain_d3 --mode long --words 11000
    python bench/decode/mtp_ab.py --arm b_chain_d3 --mode lossless

`short` is the four decode-heavy prompts from `staging/decode_bench.py`, 600
greedy tokens each. `long` is one seeded 64k cold prompt built the way
`staging/e2e_cold.py --words 11000` builds one, followed by 300 tokens of
decode from that context. `lossless` is one fixed prompt whose completion has
to be identical across arms; `--lossless-tokens 150` is the length at which
every arm agrees byte for byte, and 400 is the length at which the block-width
numerics start to show. Each short arm was run twice, from a fresh process, and
both runs are in the table.

Two rates are reported and they answer different questions. The wall rate is
what a client sees, with the prefill share subtracted at the measured cold
rate. The profiler rate is committed tokens over the summed wall time of the
decode cycles themselves, straight from `GET /metrics`, and it is the number a
drafter can move.

## The table

M5 Max, 2026-09-12, one instance at a time, `Qwen3.8-Flash-Next-oQ4e-mtp:no-think`,
greedy. Short is four prompts at 600 tokens; 64k is one seeded cold prompt of
64,779 tokens followed by 300 tokens of decode. Two independent runs of every
short arm; both are shown, and the spread between them is the honest error bar
on this machine (thermal drift alone was measured at 10 to 16% elsewhere).
Accepted per cycle and cycle time are far steadier than tok/s and are the
numbers to argue from.

### Short prompts, 4 x 600 tokens

| arm | wall tok/s (median) | profiler tok/s | accepted/cycle | acceptance | mean rows | cycle ms | draft ms | verify ms |
|---|---|---|---|---|---|---|---|---|
| a MTP off | 53.3 | 54.6 | 0.00 | - | 1.00 | 18.3 | 0.0 | 16.4 |
| b chain `head_output`, fixed depth 3 | 79.9 / 83.5 | 75.9 / 79.1 | 2.12 / 2.12 | 71.4% | 3.97 | 41.1 / 39.4 | 6.6 / 6.0 | 30.2 / 29.2 |
| c chain `omlx`, fixed depth 3 | 84.3 / 85.2 | 82.2 / 82.9 | 2.24 / 2.26 | 75.5% | 3.97 | 39.5 / 39.3 | 6.1 / 6.1 | 29.1 / 29.0 |
| d EV depth (max 6) + p_min 0.1 | 79.9 / 82.1 | 79.2 / 81.3 | 1.68 / 1.63 | 80.2% | 3.09 / 2.99 | 33.8 / 32.3 | 4.5 / 4.1 | 25.5 / 24.4 |

### 64k prompt, 300 tokens of decode

| arm | wall s | profiler tok/s | accepted/cycle | acceptance | mean rows | cycle ms | draft ms | verify ms |
|---|---|---|---|---|---|---|---|---|
| a MTP off | 85.1 | 45.7 | 0.00 | - | 1.00 | 21.9 | 0.0 | 19.4 |
| b chain `head_output`, fixed depth 3 | 92.7 | 53.1 | 1.42 | 48.4% | 3.94 | 45.5 | 6.3 | 35.0 |
| c chain `omlx`, fixed depth 3 | 97.9 | 52.0 | 1.48 | 50.1% | 3.95 | 47.7 | 7.4 | 36.0 |
| d EV depth (max 6) + p_min 0.1 | 90.8 | 54.3 | 1.10 | 55.7% | 2.97 | 38.6 | 5.1 | 29.7 |

Wall seconds at 64k are dominated by the 45-second prefill and are not a decode
measurement; the profiler rate is.

### What the table says

**The drafter is worth about 1.5x at short context.** 54.6 to 82.9 tok/s on the
profiler rate, 53.3 to 85.2 on the wall rate, which lands in the 84 to 91 band
oMLX gets on the same machine at a comparable acceptance.

**The oMLX chain form beats the vLLM one here, and it reproduced.** Accepted
per cycle 2.24 and 2.26 against 2.12 and 2.12, acceptance 75.5% against 71.4%,
throughput ahead in both runs. EAGLE 3.1's argument for feeding the post-norm
head output predicts the opposite, so the prediction does not hold on this
checkpoint's hyper-connection head, where the pre-mixer streams are the head's
native width and the post-mixer output has to be lifted back into it. The code
default stays `head_output` as specified; the config on this machine should say
`mtp_chain = "omlx"` until something explains the gap.

**The EV policy buys its throughput by spending fewer rows.** It settles around
depth 2 rather than 3 (mean rows 3.0 against 4.0), which takes 7 ms off the
cycle and raises acceptance to 80% while lowering accepted tokens per cycle to
1.65. Net at short context it is level with the fixed depth 3 and ahead of the
`head_output` fixed arm it shares a chain form with, at three quarters of the
row budget. At 64k it is the fastest arm on the board, 54.3 against 53.1 and
52.0 for the fixed arms and 45.7 for no speculation: +19% over MTP off, where
the fixed depth-3 arms manage +16% and +14%.

**Acceptance at 64k is 1.1 to 1.5 accepted tokens a cycle, not 1.** Better than
the median of 1 the integration report measured, and still less than half the
short-context figure and a long way from the 4.07 Qwen's own report claims at
four steps. The long-context defect is real and this work does not close it.

**The draft chain costs about 2 ms a step, which is 45% of a verify row.** The
head is one layer out of 49 and should be nearer 2%. The unembedding is the
reason: every chain step runs the full 2560 x 248,320 head, roughly 320 MB of
4-bit weights read per step, so depth 3 spends about 1 GB of bandwidth on
draft logits alone. That is the largest single piece of the drafter that is
not the verify.

## The lossless check

Speculation is exact by construction here: `verify` commits `accepted + bonus`,
which is what greedy decoding would have produced without any drafting, and the
drafter's numerics cannot reach the output. The check is that the arms actually
agree on the same prompt at temperature 0:

    cd ~/inference-server/staging/w41
    md5 short150-a.txt short150-b.txt short150-c.txt short150-d.txt
    diff lossless-a_mtp_off.txt lossless-b_chain_d3.txt

## What the lossless check found

At 150 tokens all four arms are byte-identical:

    md5 short150-{a,b,c,d}.txt   ->   e54dbfb37cac5d18d6c942f8e407cf4b, four times

At 400 tokens the three speculative arms agree with each other and part from
MTP off at character 1294 of 1604, roughly 330 tokens in, one word into a
paragraph that then rewords itself. `b` and `c` are byte-identical to each
other over the whole 400 tokens, and `d` follows them for 1475 characters.

That pattern is the answer to the question the check is asking. Two different
drafters, proposing different tokens at different acceptance rates, produced
exactly the same output: the drafter cannot reach what is committed, which is
the invariant. What moves the output is the *width* of the verify block, and
only after a few hundred tokens. `titan/adapters/mlx/model.py` states why: the
padded batched-QSA arm can differ from the unbatched reference by a single
bf16 ULP, and 36 recurrent GDN layers amplify one ULP into a visible divergence
over a few thousand tokens. The arms that share a width (`b`, `c`) agree
exactly; the arms that do not (`a` at width 1, `d` at width 3) diverge, and `d`
diverges from the width-4 arms later than either diverges from width 1.

So speculation here is lossless with respect to the drafter, and exact with
respect to the target model only up to the numerics of the block width the
target was run at. That is a property of the verify forward, not of this work,
and it is the reason the parity harness measures divergence rather than
asserting it away.
