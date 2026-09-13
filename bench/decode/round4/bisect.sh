#!/bin/bash
# The ROUND4 kernel bisect: the control, every fast op on its own, and the
# production default, at short context and at 64k, twice each.
#
# Paired alternation is the ordering, not a flag. The arm list is run forwards
# for repeat 0 and backwards for repeat 1, so an arm at position i in the first
# pass sits at position 2N-i in the second and any drift that is monotone in
# time cancels in the pair. ROUND2 step 1c measured that drift at 10 to 16% on
# this machine, which is larger than most of what this is looking for.
set -uo pipefail
cd "$(dirname "$0")/../../.."
HERE="bench/decode/round4"
MODES="${MODES:-short,long,lossless}"

ARMS=(
  "refonly|--set kernels.reference_only=true"
  "all_on|"
  "only_moe_gather_int8|--set kernels.enabled=[\"moe_gather_int8\"]"
  "only_moe_weighted_sum|--set kernels.enabled=[\"moe_weighted_sum\"]"
  "only_gdn_norm_gate|--set kernels.enabled=[\"gdn_norm_gate\"]"
  "only_gdn_chunk_scan|--set kernels.enabled=[\"gdn_chunk_scan\"]"
  "only_hc_prefill|--set kernels.enabled=[\"hc_prefill\"]"
  "only_grouped_rmsnorm_bf16|--set kernels.enabled=[\"grouped_rmsnorm_bf16\"]"
  "only_topk_radix|--set kernels.enabled=[\"topk_radix\"]"
  "only_ple_packed_lookup|--set kernels.enabled=[\"ple_packed_lookup\"]"
  "only_qsa_gathered_attention|--set kernels.enabled=[\"qsa_gathered_attention\"]"
  "only_moe_gather_ws|--set kernels.enabled=[\"moe_gather_ws\"]"
  "only_verify_accept|--set kernels.enabled=[\"verify_accept\"]"
)

run_arm() {  # run_arm <repeat> <entry>
  local repeat="$1" entry="$2"
  local name="${entry%%|*}" flags="${entry#*|}"
  # The lossless check is a property of the arm, not of the repeat, so it runs
  # once per arm rather than twice.
  local modes="$MODES"
  [ "$repeat" = "0" ] || modes="${MODES/,lossless/}"
  # shellcheck disable=SC2086
  bash "$HERE/arm.sh" "$name" "$modes" "$repeat" $flags \
    >> "$HERE/bisect.log" 2>&1
  echo "[$(date +%T)] $name repeat=$repeat done"
}

for i in "${!ARMS[@]}"; do
  run_arm 0 "${ARMS[$i]}"
done
for (( i=${#ARMS[@]}-1; i>=0; i-- )); do
  run_arm 1 "${ARMS[$i]}"
done
echo "[$(date +%T)] bisect complete"
