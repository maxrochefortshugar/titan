#!/bin/bash
# ROUND3, the plain-decode arms: same two trees, speculation off.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export TITAN_OUT="$HOME/inference-server/staging/round3"
BEFORE="$HOME/inference-server/staging/round3/before"
AFTER="$HOME/inference-server/staging/round3/after"
run() {
  TITAN_SRC="$1" bash bench/decode/run_round2.sh "$2" "$3" "${@:4}" \
    >> "bench/decode/round3/ab-$2-$3.log" 2>&1
  echo "[$(date +%T)] $2 $3 done"
}
OFF=(--set speculation.enabled=false)
for mode in short long; do
  run "$BEFORE" r3_before_plain "$mode" "${OFF[@]}"
  run "$AFTER"  r3_after_plain  "$mode" "${OFF[@]}"
  run "$AFTER"  r3_after_plain  "$mode" "${OFF[@]}"
  run "$BEFORE" r3_before_plain "$mode" "${OFF[@]}"
done
# Does kernels.reference_only reach the forward? Before the fix it did not, so
# the "before" tree should measure the same with it on as with it off, and the
# "after" tree should not.
run "$BEFORE" r3_refonly_before short --set kernels.reference_only=true
run "$AFTER"  r3_refonly_after  short --set kernels.reference_only=true
echo "[$(date +%T)] all done"
