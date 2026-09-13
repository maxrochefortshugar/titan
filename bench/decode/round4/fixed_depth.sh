#!/bin/bash
# The bisect again, at both contexts, with the drafter's depth policy nailed
# down. Arms are named ``fd_*`` and land in the same results.jsonl.
#
# Why this exists. The main bisect runs the production configuration, which has
# ``speculation.adaptive_depth = true``, and that policy picks a depth by
# argmax over ``E[committed | k] / cycle_ms(k + 1)`` -- a *measured* cycle-time
# model. So the depth an arm settles on depends on how fast that arm ran, and
# accepted-per-cycle is downstream of speed rather than only of numerics. Two
# arms whose kernels are bit-identical can still land on different depths and
# report different throughput.
#
# Fixing the depth removes the feedback loop and leaves the kernel. Read this
# table for *which op costs what*; read the main table for what a production
# instance would actually deliver, control loop and all.
set -uo pipefail
cd "$(dirname "$0")/../../.."
HERE="bench/decode/round4"
FIXED=(--set speculation.adaptive_depth=false
       --set speculation.mtp_depth_min=3
       --set speculation.mtp_depth_max=3)

ARMS=(
  "fd_refonly|--set kernels.reference_only=true"
  "fd_all_on|"
  "fd_moe_gather_int8|--set kernels.enabled=[\"moe_gather_int8\"]"
  "fd_moe_weighted_sum|--set kernels.enabled=[\"moe_weighted_sum\"]"
  "fd_gdn_norm_gate|--set kernels.enabled=[\"gdn_norm_gate\"]"
  "fd_gdn_chunk_scan|--set kernels.enabled=[\"gdn_chunk_scan\"]"
  "fd_hc_prefill|--set kernels.enabled=[\"hc_prefill\"]"
  "fd_grouped_rmsnorm_bf16|--set kernels.enabled=[\"grouped_rmsnorm_bf16\"]"
  "fd_topk_radix|--set kernels.enabled=[\"topk_radix\"]"
  "fd_ple_packed_lookup|--set kernels.enabled=[\"ple_packed_lookup\"]"
  "fd_qsa_gathered_attention|--set kernels.enabled=[\"qsa_gathered_attention\"]"
  "fd_moe_gather_ws|--set kernels.enabled=[\"moe_gather_ws\"]"
  "fd_verify_accept|--set kernels.enabled=[\"verify_accept\"]"
)

run_arm() {
  local repeat="$1" entry="$2"
  local name="${entry%%|*}" flags="${entry#*|}"
  # shellcheck disable=SC2086
  bash "$HERE/arm.sh" "$name" "${MODES:-short,long}" "$repeat" "${FIXED[@]}" $flags \
    >> "$HERE/fixed_depth.log" 2>&1
  echo "[$(date +%T)] $name repeat=$repeat done"
}

for i in "${!ARMS[@]}"; do run_arm 0 "${ARMS[$i]}"; done
for (( i=${#ARMS[@]}-1; i>=0; i-- )); do run_arm 1 "${ARMS[$i]}"; done
echo "[$(date +%T)] fixed-depth pass complete"
