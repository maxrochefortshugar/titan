#!/bin/bash
# ROUND5 step 1: does the depth policy converge to the same depth twice?
#
# Three arms, four repeats each, short and 64k in every repeat.
#
#   adaptive   the converging expected-value policy this round ships
#   fixed3     pinned depth 3, the floor the adaptive median has to clear
#   legacy     ROUND4's round(mean_accepted) + 1, the policy that measured
#              93.5 and 79.0 tok/s an hour apart
#
# Each repeat is its own server lifetime, which is the point: the policy starts
# from nothing every time and has to arrive somewhere it also arrived last
# time. Ordering is ROUND4's paired alternation for its reason -- drift on this
# machine was measured at 10 to 16% and is monotone in time -- so the three
# arms rotate rather than running in blocks, and no arm sits at the same
# position in the pass twice.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"

ADAPTIVE='--set speculation.adaptive_depth=true --set speculation.depth_policy=expected_value'
FIXED='--set speculation.adaptive_depth=false --set speculation.mtp_depth_min=3 --set speculation.mtp_depth_max=3'
LEGACY='--set speculation.adaptive_depth=true --set speculation.depth_policy=mean_accepted'

run_arm() {
  case "$1" in
    adaptive) bash "$ARM" adaptive short,long "$2" $ADAPTIVE ;;
    fixed3)   bash "$ARM" fixed3   short,long "$2" $FIXED ;;
    legacy)   bash "$ARM" legacy   short,long "$2" $LEGACY ;;
  esac
}

for r in 0 1 2 3; do
  case "$r" in
    0) order="adaptive fixed3 legacy" ;;
    1) order="legacy adaptive fixed3" ;;
    2) order="fixed3 legacy adaptive" ;;
    3) order="adaptive legacy fixed3" ;;
  esac
  for a in $order; do run_arm "$a" "$r"; done
done
echo "step1 done"
