#!/bin/bash
# One arm of a ROUND2 measurement: start Titan on 8085, run one mode, stop it.
#
# Same rules as run_arms.sh, and the same single port. What is different is
# that this one runs a single mode per invocation, so the caller can alternate
# arms on a warm GPU rather than running an arm's whole battery back to back:
# thermal drift on this machine was measured at 10 to 16%, which is larger than
# most of what ROUND2 is looking for, and alternating is the only way an A/B
# survives it.
#
#   run_round2.sh <arm> <mode> [serve-overrides...]
#
# mode is one of mtp_ab.py's: short, long, lossless, sweep.
set -uo pipefail

ARM="$1"; shift
MODE="$1"; shift
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
PY="${TITAN_PY:-/Users/maxshugar/inference-server/kdev/bin/python}"
CONFIG="${TITAN_CONFIG_FILE:-$HOME/.config/titan/titan.toml}"
OUT="${TITAN_OUT:-$HOME/inference-server/staging/round2}"
WORDS="${WORDS:-11000}"
CONTEXTS="${CONTEXTS:-8192,16384,32768,49152,63488,64779,67584}"
SWEEP_TOKENS="${SWEEP_TOKENS:-200}"
LOSSLESS_TOKENS="${LOSSLESS_TOKENS:-150}"
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

# Closing the port is not the same as the process being gone, and the process
# being gone is not the same as the 73 GB being back. An earlier version of
# this returned as soon as the port closed, and the arm after it lost a race
# with the outgoing server's flock on serve-8085.lock: it printed a config
# error, never came up, and then sat in the health-check loop for 800 seconds
# looking like a slow model load. Every exit path from here now ends at
# wait_for_memory.
stop() {
  local pids
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill $pids 2>/dev/null
  for _ in $(seq 1 60); do
    lsof -nP -iTCP:8085 -sTCP:LISTEN -t >/dev/null 2>&1 || break
    sleep 1
  done
  pids=$(lsof -nP -iTCP:8085 -sTCP:LISTEN -t 2>/dev/null)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  pkill -f 'titan.cli serve' 2>/dev/null
  wait_for_memory 85
}

trap stop EXIT
stop

LOG="$OUT/serve-$ARM-$MODE.log"
CACHE="$OUT/cache"
rm -rf "$CACHE"; mkdir -p "$CACHE"

# The prefix store is off for a decode measurement, and this is not a detail.
# A 64k prompt is about 4.5 GB of blocks and boundary snapshots, the store
# serialises on the loop thread, and one pump was measured holding it for 12.0
# seconds inside a 300-token decode. With the store on, the same arm measured
# 48 ms and 198 ms a cycle in two runs that drafted identical tokens: the
# variance is the writer, not the drafter. TITAN_CACHE=on puts it back for a
# measurement that is actually about the cache.
CACHE_ARGS=(--set 'cache.ssd_dir=""' --set 'cache.ram_tier_mb=1.0')
if [ "${TITAN_CACHE:-off}" = "on" ]; then
  CACHE_ARGS=(--set "cache.ssd_dir=\"$CACHE\"")
fi

( cd "$SRC" && PYTHONPATH="$SRC" "$PY" -m titan.cli serve --config "$CONFIG" \
    "${CACHE_ARGS[@]}" "$@" ) >"$LOG" 2>&1 &

for _ in $(seq 1 400); do
  curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
       http://127.0.0.1:8085/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
     http://127.0.0.1:8085/health >/dev/null || {
  echo "[$ARM/$MODE] server never came up"; tail -30 "$LOG"; exit 1; }

echo "[$ARM/$MODE] up"
case "$MODE" in
  lossless)
    "$PY" "$SRC/bench/decode/mtp_ab.py" --arm "$ARM" --mode lossless \
      --lossless-tokens "$LOSSLESS_TOKENS" > "$OUT/lossless-$ARM.txt" ;;
  sweep)
    "$PY" "$SRC/bench/decode/mtp_ab.py" --arm "$ARM" --mode sweep \
      --contexts "$CONTEXTS" --sweep-tokens "$SWEEP_TOKENS" \
      --out "$OUT/results.jsonl" ;;
  long)
    "$PY" "$SRC/bench/decode/mtp_ab.py" --arm "$ARM" --mode long \
      --words "$WORDS" --out "$OUT/results.jsonl" ;;
  *)
    "$PY" "$SRC/bench/decode/mtp_ab.py" --arm "$ARM" --mode short \
      --out "$OUT/results.jsonl" ;;
esac
stop
echo "[$ARM/$MODE] done"
