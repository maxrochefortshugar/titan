#!/bin/bash
# Kill the workbench server if swap use climbs past 3 GB or free memory stays under 1.5 GB for 3 checks (kernel panic guard).
low=0; base=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); echo "$(date +%T) baseline swap ${base}M"
while true; do
  used=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); free=$(vm_stat | awk '/Pages free/{print $3*16384/1e9}')
  if (( $(echo "$used - $base > 3072" | bc) )); then echo "$(date +%T) SWAP grew ${base}M -> ${used}M -> killing workbench"; for p in $(lsof -nP -tiTCP:8084 -sTCP:LISTEN); do kill -9 $p; done; fi
  lvl=$(memory_pressure -Q 2>/dev/null | awk "/free percentage/{print \$NF+0}"); if [ -n "$lvl" ] && [ "$lvl" -lt 5 ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 3 ]; then echo "$(date +%T) PRESSURE free%=${lvl} x3 -> killing workbench"; for p in $(lsof -nP -tiTCP:8084 -sTCP:LISTEN); do kill -9 $p; done; low=0; fi
  sleep 5
done
