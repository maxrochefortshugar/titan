#!/bin/bash
# ROUND5 step 3a: per-chunk prefill rate against chunk size, on the real model.
#
# One cold 65k prompt per arm, four tokens of decode, and the scheduler's own
# prefill_chunk events read back for the rate table. 256 needs the block grid
# moved with it: the planner refuses a chunk that is not a multiple of the
# block, and the block grid is what a stored block is measured in.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"

for r in 0 1; do
  if [ "$r" -eq 0 ]; then sizes="256 512 1024 2048 4096"; else sizes="4096 2048 1024 512 256"; fi
  for c in $sizes; do
    block=512
    [ "$c" -lt 512 ] && block="$c"
    bash "$ARM" "chunk$c" prefill "$r" \
      --set "scheduler.prefill_chunk=$c" --set "cache.block_tokens=$block"
  done
done
echo "step3a done"
