#!/bin/bash
# One arm of the MTP drafter A/B: start Titan on 8085, measure, stop it.
#
# One model instance at a time and one port, 8085. Production owns 8083 and the
# workbench owns 8084, and the config refuses either at parse time, but the
# kill below is written against 8085 explicitly rather than against every
# python process on the machine.
#
#   run_arms.sh <arm> <serve-overrides...>
#
# Arms, and what each is measuring, are in README.md.
set -uo pipefail

ARM="$1"; shift
SRC="${TITAN_SRC:-/tmp/w41src}"
PY="${TITAN_PY:-/Users/maxshugar/inference-server/kdev/bin/python}"
CONFIG="${TITAN_CONFIG_FILE:-$HOME/.config/titan/titan.toml}"
OUT="${TITAN_OUT:-$HOME/inference-server/staging/w41}"
WORDS="${WORDS:-11000}"
mkdir -p "$OUT"


# Wait until every titan serve process has exited and macOS has actually
# reclaimed its image. Closing the port is not the same as releasing 73 GB:
# starting the next arm before this returns overlaps two model images and swaps.
wait_for_memory() {
  local need_gb=${1:-85}
  for _ in $(seq 1 90); do
    if ! pgrep -f "titan.cli serve" >/dev/null 2>&1; then
      local avail
      avail=$(vm_stat | awk -v ps=$(sysctl -n hw.pagesize) '/Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*ps/1073741824}')
      [ "${avail:-0}" -ge "$need_gb" ] && return 0
    fi
    sleep 2
  done
  echo "memory not reclaimed after 180s (titan still running or available below ${need_gb} GB)"; return 1
}

stop() {
  local pids
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill $pids 2>/dev/null
  for _ in $(seq 1 40); do
    lsof -nP -iTCP:8085 -sTCP:LISTEN -t >/dev/null 2>&1 || return 0
    sleep 1
  done
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 2
  pkill -f 'titan.cli serve' 2>/dev/null; wait_for_memory 85
}

trap stop EXIT
stop

LOG="$OUT/serve-$ARM.log"
CACHE="$OUT/cache-$ARM"
rm -rf "$CACHE"; mkdir -p "$CACHE"

( cd "$SRC" && PYTHONPATH="$SRC" "$PY" -m titan.cli serve --config "$CONFIG" \
    --set "cache.ssd_dir=\"$CACHE\"" "$@" ) >"$LOG" 2>&1 &

for _ in $(seq 1 300); do
  curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
       http://127.0.0.1:8085/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
     http://127.0.0.1:8085/health >/dev/null || { echo "[$ARM] server never came up"; tail -25 "$LOG"; exit 1; }

echo "[$ARM] up"
"$PY" bench/decode/mtp_ab.py --arm "$ARM" --mode lossless > "$OUT/lossless-$ARM.txt"
"$PY" bench/decode/mtp_ab.py --arm "$ARM" --mode short --out "$OUT/results.jsonl"
stop
sleep 5

# The long arm gets its own process so its 64k acceptance history is not the
# tail of a short-prompt one: the depth policy's estimates are per sequence,
# but the cost table it prices against is not.
rm -rf "$CACHE"; mkdir -p "$CACHE"
( cd "$SRC" && PYTHONPATH="$SRC" "$PY" -m titan.cli serve --config "$CONFIG" \
    --set "cache.ssd_dir=\"$CACHE\"" "$@" ) >>"$LOG" 2>&1 &
for _ in $(seq 1 300); do
  curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
       http://127.0.0.1:8085/health >/dev/null 2>&1 && break
  sleep 2
done
"$PY" bench/decode/mtp_ab.py --arm "$ARM" --mode long --words "$WORDS" \
      --out "$OUT/results.jsonl"
stop
echo "[$ARM] done"
