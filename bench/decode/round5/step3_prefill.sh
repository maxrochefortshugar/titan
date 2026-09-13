#!/bin/bash
# ROUND5 step 3: the prefill round, against oMLX's 1600 tok/s cold on 65k.
#
# 3a  per-chunk prefill rate against chunk size. One cold 65k prompt per arm,
#     four tokens of decode, and the scheduler's own prefill_chunk events read
#     back for the rate table. 256 needs the block grid moved with it: the
#     planner refuses a chunk that is not a multiple of the block, and the
#     block grid is what a stored block is measured in.
#
# 3b  the per-chunk fixed cost, which needs no arm of its own: every chunk
#     event carries the profiler's clock, so the wall between one forward
#     returning and the next one starting is a subtraction. probe.py reports
#     it split by whether the previous chunk staged a snapshot.
#
# 3c  moe_gather_int8 as a prefill-only op. ROUND4 measured it at -3.7% on a
#     64k decode and never separated that from what it does to a prefill; the
#     op runs over 20,480 rows on a 2048-token chunk and over ten on a
#     width-1 decode, so the two phases are not the same measurement. Its
#     activation quantisation is 0.65% of output RMS, so this arm's answer is
#     not byte-identical and the digest is recorded rather than asserted.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"
SEVEN='"moe_weighted_sum","gdn_norm_gate","hc_prefill","topk_radix","moe_gather_ws","verify_accept","qsa_gathered_attention"'

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

for r in 0 1; do
  if [ "$r" -eq 0 ]; then order="int8off int8prefill"; else order="int8prefill int8off"; fi
  for a in $order; do
    case "$a" in
      int8off)     bash "$ARM" "$a" prefill,long,lossless "$r" ;;
      int8prefill) bash "$ARM" "$a" prefill,long,lossless "$r" \
                     --set "kernels.enabled=[$SEVEN,\"moe_gather_int8\"]" \
                     --set 'kernels.prefill_only=["moe_gather_int8"]' ;;
    esac
  done
done
echo "step3c done"
