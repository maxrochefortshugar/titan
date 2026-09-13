#!/bin/bash
# ROUND5 step 4: the fp32 pooled indexer bank, at 64k.
#
# Four arms, two repeats, alternated: the bank on and off, each with the
# drafted production cycle and with speculation off. Speculation off is the
# width-1 decode, where the indexer runs once per committed token and the cast
# the bank removes is at its largest share of a step; drafted is what
# production runs. A win at one and not the other is a real answer and the
# reason both are here.
#
# Widening bf16 to float32 is lossless, so the 150-token digest is not a
# tolerance check: it has to be the same md5 in all four arms, and a difference
# is a bug in the bank's offset bookkeeping rather than a numerical trade.
set -uo pipefail
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
ARM="$SRC/bench/decode/round5/arm.sh"
ON='--set kernels.forward_paths_on=["qsa_pooled_bank_f32"]'
PLAIN='--set speculation.enabled=false'

for r in 0 1; do
  if [ "$r" -eq 0 ]; then
    order="bankoff-draft bankon-draft bankoff-plain bankon-plain"
  else
    order="bankon-plain bankoff-plain bankon-draft bankoff-draft"
  fi
  for a in $order; do
    case "$a" in
      bankoff-draft) bash "$ARM" "$a" long,lossless "$r" ;;
      bankon-draft)  bash "$ARM" "$a" long,lossless "$r" $ON ;;
      bankoff-plain) bash "$ARM" "$a" long,lossless "$r" $PLAIN ;;
      bankon-plain)  bash "$ARM" "$a" long,lossless "$r" $PLAIN $ON ;;
    esac
  done
done
echo "step4 done"
