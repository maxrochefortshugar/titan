#!/bin/bash
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
run "$AFTER"  r3_after_plain  long "${OFF[@]}"
run "$BEFORE" r3_before_plain long "${OFF[@]}"
run "$BEFORE" r3_before_plain long "${OFF[@]}"
run "$AFTER"  r3_after_plain  long "${OFF[@]}"
echo "[$(date +%T)] all done"
