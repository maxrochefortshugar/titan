#!/bin/bash
# ROUND5 step 1, the one re-measure the round allows.
#
# The sweep in depth_tune.py, priced at the horizons the arms actually have
# (912 cycles short, 133 at 64k) and read off the fresh-start column because
# every arm here is a fresh process, says the shipped hysteresis of 6% is the
# single thing to change. It is wider than the margin between neighbouring
# depths on both workloads -- about 5% at 64k and under 1% at short -- so the
# policy cannot leave whatever depth the seed cost table put it in, and at
# short that is depth 1. Three per cent is inside both margins and outside the
# noise the hysteresis exists to damp.
#
# Four repeats of the tuned policy against two fresh repeats of pinned depth 3.
# The pinned arm is measured again rather than compared against the four from
# an hour ago, because drift on this machine is monotone in time and that is
# the whole reason ROUND4 alternates arms.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"
TUNED='--set speculation.adaptive_depth=true --set speculation.depth_policy=expected_value --set speculation.depth_hysteresis=0.03'
FIXED='--set speculation.adaptive_depth=false --set speculation.mtp_depth_min=3 --set speculation.mtp_depth_max=3'

for r in 0 1 2 3; do
  bash "$ARM" "tuned" short,long "$r" $TUNED
  if [ "$r" -eq 0 ] || [ "$r" -eq 2 ]; then
    bash "$ARM" "fixed3b" short,long "$((r / 2))" $FIXED
  fi
done
echo "step1b done"
