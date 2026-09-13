#!/bin/bash
# ROUND5 step 2: does a prefill kernel cost a decode through what it leaves
# behind? Three arms, two repeats, alternated.
#
#   control   the seven-kernel default
#   prefill   the default plus gdn_chunk_scan and grouped_rmsnorm_bf16
#   released  the same, with mx.clear_cache() when the prompt is in
#
# Every arm reads its own prefill_done memory line, which is the sample
# ROUND4 section 5 called cheap and did not take.
#
# kernels.enabled is a TOML array and the override goes through the TOML
# parser, so it is spelled as one. A comma-separated string is rejected at
# parse time, which is the config module doing its job.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"
SEVEN='"moe_weighted_sum","gdn_norm_gate","hc_prefill","topk_radix","moe_gather_ws","verify_accept","qsa_gathered_attention"'
PREFILL="$SEVEN,\"gdn_chunk_scan\",\"grouped_rmsnorm_bf16\""

for r in 0 1; do
  if [ "$r" -eq 0 ]; then order="control prefill released"; else order="released prefill control"; fi
  for a in $order; do
    case "$a" in
      control)  bash "$ARM" "$a" short,prefill,long,lossless "$r" \
                  --set "kernels.enabled=[$SEVEN]" ;;
      prefill)  bash "$ARM" "$a" short,prefill,long,lossless "$r" \
                  --set "kernels.enabled=[$PREFILL]" ;;
      released) bash "$ARM" "$a" short,prefill,long,lossless "$r" \
                  --set "kernels.enabled=[$PREFILL]" \
                  --set 'scheduler.release_after_prefill=true' ;;
    esac
  done
done
echo "step2 done"
