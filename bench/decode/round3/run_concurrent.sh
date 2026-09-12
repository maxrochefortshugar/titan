#!/bin/bash
# Start one server from <tree>, run the two-stream check, stop everything.
# Same port, same locks, same drain wait as run_round2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.."
SRC="$1"; TAG="$2"; shift 2
PY="/Users/maxshugar/inference-server/kdev/bin/python"
CONFIG="$HOME/.config/titan/titan.toml"
LOG="bench/decode/round3/serve-concurrent-$TAG.log"

wait_for_memory() {
  for _ in $(seq 1 90); do
    if ! pgrep -f "titan.cli serve" >/dev/null 2>&1; then
      avail=$(vm_stat | awk -v ps=$(sysctl -n hw.pagesize) '/Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*ps/1073741824}')
      [ "${avail:-0}" -ge 85 ] && return 0
    fi
    sleep 2
  done
  echo "memory not reclaimed"; return 1
}
stop() {
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill $pids 2>/dev/null
  for _ in $(seq 1 60); do
    lsof -nP -iTCP:8085 -sTCP:LISTEN -t >/dev/null 2>&1 || break
    sleep 1
  done
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  pkill -f 'titan.cli serve' 2>/dev/null
  wait_for_memory
}
trap stop EXIT
stop

( cd "$SRC" && PYTHONPATH="$SRC" "$PY" -m titan.cli serve --config "$CONFIG" \
    --set 'cache.ssd_dir=""' --set 'cache.ram_tier_mb=1.0' "$@" ) >"$LOG" 2>&1 &
for _ in $(seq 1 300); do
  curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
       http://127.0.0.1:8085/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
     http://127.0.0.1:8085/health >/dev/null || { echo "[$TAG] never came up"; tail -20 "$LOG"; exit 1; }
echo "[$TAG] up"
"$PY" bench/decode/round3/concurrent_check.py --context 2600 --tokens 60
echo "[$TAG] exit $?"
stop
