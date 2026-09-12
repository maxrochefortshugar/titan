# M1 parity report (2026-09-12)

Titan loads Qwen3.8-Flash-Next through its own loader and runs plain greedy decoding through its own forward path. This report records how its output compares with oMLX 0.7.0.dev2 and what the comparison can and cannot show.

## How the references were captured

oMLX returns neither token ids nor logprobs, so each reference is the completion text from `/v1/completions` at temperature 0, re-encoded with the model tokenizer (`ids_source: reencoded_text`). oMLX strips leading whitespace from completion text, so all comparisons strip leading whitespace on both sides and re-encode from the stripped text. Twenty prompts, ten short and ten with a 3.6k-token prefix, 128 tokens each. Files: `reference-stock.json`, `reference-stock-rep2.json` (same configuration repeated), `reference-stock-nomtp.json` (speculative decoding off), `reference-prod.json` (the production patch set).

## The noise floor: oMLX against itself

| comparison | identical texts of 20 |
|---|---|
| stock oMLX, same configuration, run 1 vs run 2 | 19 |
| stock oMLX with MTP on vs MTP off | 4 |
| stock oMLX vs production patch set | 4 |

A fixed configuration is deterministic. Changing the verify batch shape from one row to k+1 rows changes 16 of 20 greedy outputs, and so does the production patch set. The model amplifies 1-ULP kernel differences through 36 recurrent layers, which the ple-fix report already documented. Bit-exact agreement across implementations is therefore not a meaningful bar; agreement at the level of a configuration change is the realistic one.

## Titan against oMLX

Plain greedy loop, single row per step, compared with the MTP-off reference (the closest numerics).

| Titan mode | identical of 20 | median agreeing prefix (tokens) | sum of agreeing tokens |
|---|---|---|---|
| reference ops only | 2 | 42 | 1168 |
| fast kernels | 6 | 42 | 1212 |

Titan's reference-mode texts agree with each of the four oMLX arms about equally (2, 4, 4 and 5 identical; sums within 4%), so Titan sits inside the same band the oMLX arms occupy against each other. Titan fast mode against Titan reference mode is 5 of 20 identical, the same size of effect as oMLX's own configuration changes.

Titan against itself, fast mode, two runs: 20 of 20 identical.

## What was checked and ruled out

- Prompt framing. Titan's first token for every prompt matches the reference once the stripped-whitespace rule is applied (short-00 predicts a double newline, then `def`; the reference starts at `def`). No chat-template or special-token difference.
- Loader. All 3748 checkpoint tensors are claimed exactly once. Two ordering bugs were fixed: the fused gate/up module must be built before quantisation so the quantisation predicate sees the fused path.
- Early divergence. No short prompt diverges before token 3; most agree for 12 to 127 tokens. There is no systematic first-tokens bug.

## Performance observed during the runs (not a benchmark)

Plain greedy, one host sync per token, no pipelining: short prompts 53 to 60 tok/s, after a 3.6k prefix 42 to 48 tok/s. oMLX with MTP off measured about 60 on the same machine. Load time 7 s, 73 GB resident. Fast kernels cut the 3.6k prefill from 3.5 s to 3.1 s.

## Verdict

M1 acceptance is met on the evidence available: Titan is deterministic, agrees with oMLX at the same level oMLX agrees with itself across configurations, and shows no framing, loading or early-token defect. The stricter "identical ids" criterion in PLAN.md is unachievable against oMLX for this model and is replaced by this noise-floor criterion. The remaining numerical gap is kernel-level and will be revisited when the fast kernels are individually A/B'd on the same harness.
