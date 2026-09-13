#!/bin/bash
# The best subset, against the two arms it has to beat, and against itself
# with the prefill-oriented ops put back.
#
# The per-op pass at pinned depth splits the eleven ops cleanly. Six pay for
# themselves at both contexts, one is neutral, and four cost throughput:
# ``moe_gather_int8`` and ``gdn_chunk_scan`` by about ten and fourteen per cent
# at 64k, ``grouped_rmsnorm_bf16`` by nine, and ``ple_packed_lookup`` by four
# and a half at both contexts.
#
# Two of the four losers were built for prefill, so a decode-only bisect would
# drop them for the wrong reason -- which is why there are two subsets here.
# ``decode`` is the six winners plus the neutral one. ``prefill`` is that plus
# ``gdn_chunk_scan`` and ``grouped_rmsnorm_bf16``, the two prefill ops with the
# best cold-prefill column of the single-op arms. The question between them is
# whether the cold 65k prefill pays for a 64k decode arm that is ten per cent
# slower. ``moe_gather_int8`` is in neither: it is the worst 64k arm measured
# and it also has the worst cold-prefill number of the four, so nothing is
# being traded for it.
set -uo pipefail
cd "$(dirname "$0")/../../.."
HERE="bench/decode/round4"

WIN='"moe_weighted_sum","gdn_norm_gate","hc_prefill","topk_radix","moe_gather_ws","verify_accept","qsa_gathered_attention"'
DECODE="[$WIN]"
PREFILL="[$WIN,\"gdn_chunk_scan\",\"grouped_rmsnorm_bf16\"]"

FIXED=(--set speculation.adaptive_depth=false
       --set speculation.mtp_depth_min=3
       --set speculation.mtp_depth_max=3)

run() {  # run <repeat> <name> <extra flags...>
  local repeat="$1"; shift
  local name="$1"; shift
  bash "$HERE/arm.sh" "$name" short,long,lossless "$repeat" "$@" \
    >> "$HERE/subset.log" 2>&1
  echo "[$(date +%T)] $name repeat=$repeat done"
}

# Pinned depth: the attribution pass, with a contemporaneous control beside it
# so the subset is not being compared with a table taken an hour earlier.
fixed_pass() {
  local repeat="$1"
  run "$repeat" fd_subset_decode  "${FIXED[@]}" --set "kernels.enabled=$DECODE"
  run "$repeat" fd_subset_prefill "${FIXED[@]}" --set "kernels.enabled=$PREFILL"
  run "$repeat" fd_all_on2        "${FIXED[@]}"
  run "$repeat" fd_refonly2       "${FIXED[@]}" --set kernels.reference_only=true
}
fixed_pass_reverse() {
  local repeat="$1"
  run "$repeat" fd_refonly2       "${FIXED[@]}" --set kernels.reference_only=true
  run "$repeat" fd_all_on2        "${FIXED[@]}"
  run "$repeat" fd_subset_prefill "${FIXED[@]}" --set "kernels.enabled=$PREFILL"
  run "$repeat" fd_subset_decode  "${FIXED[@]}" --set "kernels.enabled=$DECODE"
}

fixed_pass 0
fixed_pass_reverse 1

# The production configuration: adaptive depth back on. These carry the
# variance section 1a describes, which is why there are three repeats each and
# why they are reported as delivered numbers rather than as attribution.
for repeat in 0 1 2; do
  if [ $((repeat % 2)) -eq 0 ]; then
    run "$repeat" prod_subset --set "kernels.enabled=$DECODE"
    run "$repeat" prod_all_on
  else
    run "$repeat" prod_all_on
    run "$repeat" prod_subset --set "kernels.enabled=$DECODE"
  fi
done
echo "[$(date +%T)] subset pass complete"
