#!/bin/bash
# ROUND3 A/B: HEAD against HEAD-plus-this-round, alternated on a warm GPU.
#
# Two source trees, both built from HEAD by ``git archive`` so the concurrent
# engine and config work in the repo cannot move these numbers; the "after"
# tree is that plus the four files this round changed. One arm per server, one
# server at a time, order A B B A within each mode, because thermal drift on
# this machine is larger than most of what is being measured (ROUND2 step 1c).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export TITAN_OUT="$HOME/inference-server/staging/round3"
BEFORE="$HOME/inference-server/staging/round3/before"
AFTER="$HOME/inference-server/staging/round3/after"

run() {   # run <tree> <arm> <mode>
  TITAN_SRC="$1" bash bench/decode/run_round2.sh "$2" "$3" \
    >> "bench/decode/round3/ab-$2-$3.log" 2>&1
  echo "[$(date +%T)] $2 $3 done"
}

for mode in short long; do
  run "$BEFORE" r3_before "$mode"
  run "$AFTER"  r3_after  "$mode"
  run "$AFTER"  r3_after  "$mode"
  run "$BEFORE" r3_before "$mode"
done
run "$BEFORE" r3_before lossless
run "$AFTER"  r3_after  lossless
echo "[$(date +%T)] all done"
