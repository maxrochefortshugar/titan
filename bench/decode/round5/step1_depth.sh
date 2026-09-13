#!/bin/bash
# ROUND5 step 1: does the depth policy converge to the same depth twice?
#
# Four repeats of one configuration, short and 64k, plus two repeats of pinned
# depth 3 as the floor the adaptive median has to clear. Each repeat is its own
# server lifetime, which is the point: the policy starts from nothing every
# time and has to arrive somewhere it also arrived last time.
#
# Ordering is ROUND4's paired alternation for its reason -- drift on this
# machine was measured at 10 to 16% and is monotone in time -- so the adaptive
# repeats are spread across the pass rather than run back to back.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"
FIXED='--set speculation.adaptive_depth=false --set speculation.mtp_depth_min=3 --set speculation.mtp_depth_max=3'

for r in 0 1 2 3; do
  bash "$ARM" "adaptive" short,long "$r"
  if [ "$r" -eq 0 ] || [ "$r" -eq 2 ]; then
    bash "$ARM" "fixed3" short,long "$((r / 2))" $FIXED
  fi
done
echo "step1 done"
