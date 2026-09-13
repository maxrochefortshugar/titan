#!/bin/bash
# One arm of the ROUND4 kernel bisect: start Titan on 8085 with one kernel
# configuration, run every mode against it, stop it.
#
# The stop/wait logic is ``run_round2.sh``'s, unchanged and for its reasons:
# closing the port is not the process exiting and the process exiting is not
# the 73 GB coming back, and starting the next arm before it does overlaps two
# model images and swaps. What is different is that this runs all the modes
# inside one server lifetime, because a kernel configuration is fixed at
# startup and a bisect of eleven ops times two contexts times two repeats is
# fifty-two model loads if each mode gets its own.
#
#   arm.sh <arm> <modes> <repeat> [serve-overrides...]
set -uo pipefail

ARM="$1"; shift
MODES="$1"; shift
REPEAT="$1"; shift
SRC="${TITAN_SRC:-$HOME/Engineering/titan}"
PY="${TITAN_PY:-/Users/maxshugar/inference-server/kdev/bin/python}"
CONFIG="${TITAN_CONFIG_FILE:-$HOME/.config/titan/titan.toml}"
OUT="${TITAN_OUT:-$SRC/bench/decode/round5}"
mkdir -p "$OUT"

# 80 GB, which is what the loader's headroom gate actually asks for now. The
# gate used to sum all 21 safetensors shards (99 GB) and add 8, so the harness
# was made to wait for 107 to avoid launching into a refusal. At HEAD the need
# is the plan's resident bytes instead, 68.4 GB plus 8, and the serve
# preflight's own bar is model.weights_gb plus 8. Waiting for 80 clears both
# with a little to spare and no longer parks the harness above what this
# machine has free with Docker resident.
wait_for_memory() {
  local need_gb=${1:-80}
  for _ in $(seq 1 90); do
    if ! pgrep -f "titan.cli serve" >/dev/null 2>&1; then
      local avail
      avail=$(vm_stat | awk -v ps=$(sysctl -n hw.pagesize) '/Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*ps/1073741824}')
      [ "${avail:-0}" -ge "$need_gb" ] && return 0
    fi
    sleep 2
  done
  echo "memory not reclaimed after 180s"; return 1
}

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
  wait_for_memory 80
}

trap stop EXIT
stop

LOG="$OUT/serve-$ARM-r$REPEAT.log"

# The prefix store stays off, for run_round2.sh's reason: a 64k prompt is about
# 4.5 GB of blocks, the store serialises on the loop thread, and one pump was
# measured holding it for 12 seconds inside a 300-token decode. This bisect is
# about kernels, so the writer is not in it.
( cd "$SRC" && PYTHONPATH="$SRC" "$PY" -m titan.cli serve --config "$CONFIG" \
    --set 'cache.ssd_dir=""' --set 'cache.ram_tier_mb=1.0' "$@" ) >"$LOG" 2>&1 &

for _ in $(seq 1 400); do
  curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
       http://127.0.0.1:8085/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS -H "Authorization: Bearer $(cat ~/.omlx/api_key.txt)" \
     http://127.0.0.1:8085/health >/dev/null || {
  echo "[$ARM r$REPEAT] server never came up"; tail -30 "$LOG"; exit 1; }

echo "[$ARM r$REPEAT] up ($MODES)"
"$PY" "$SRC/bench/decode/round5/probe.py" --arm "$ARM" --modes "$MODES" \
  --repeat "$REPEAT" --note "$*" --out "$OUT/results.jsonl"
STATUS=$?
stop
echo "[$ARM r$REPEAT] done status=$STATUS"
exit $STATUS
