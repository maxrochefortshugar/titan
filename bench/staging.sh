#!/bin/bash
# usage: staging.sh start plain|patched   |   staging.sh stop
S=~/inference-server/staging; R=/Applications/oMLX.app/Contents/Resources
stop_srv(){ for p in $(lsof -nP -tiTCP:8084 -sTCP:LISTEN 2>/dev/null); do kill $p; done; pkill -f "bootstrap.py serve" 2>/dev/null; for i in $(seq 1 30); do lsof -nP -iTCP:8084 -sTCP:LISTEN >/dev/null 2>&1 || return 0; sleep 1; done; for p in $(lsof -nP -tiTCP:8084 -sTCP:LISTEN 2>/dev/null); do kill -9 $p; done; sleep 2; }
case "$1" in
  stop) stop_srv; echo stopped;;
  start)
    stop_srv; MODE=${2:-plain}; [ "$MODE" = patched ] && export OMLX_SMALLM_QMM=1 || export OMLX_SMALLM_QMM=0; export OMLX_PREFILL_STEP=${OMLX_PREFILL_STEP:-0} OMLX_ARRAYS_CACHE_BLOCK=${OMLX_ARRAYS_CACHE_BLOCK:-0} OMLX_MOE_INT8_PREFILL=${OMLX_MOE_INT8_PREFILL:-0} OMLX_PLE_PACKED=${OMLX_PLE_PACKED:-0} OMLX_PLE_PACKED_MODE=${OMLX_PLE_PACKED_MODE:-rows} OMLX_QWEN4_BF16_NORM=${OMLX_QWEN4_BF16_NORM:-0}
    export OMLX_ROUND2_PATCHES=${OMLX_ROUND2_PATCHES:-} OMLX_QWEN4_GDN_NORM_GATE=${OMLX_QWEN4_GDN_NORM_GATE:-0} OMLX_MLXVLM_PATH=${OMLX_MLXVLM_PATH:-} OMLX_BASE_PATH=$S/omlx-home PYTHONHOME=$R/Python/cpython-3.11 PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages
    cp -f $S/server-$MODE.log $S/server-$MODE.log.$(date +%H%M%S) 2>/dev/null
    nohup $R/Python/cpython-3.11/bin/python3 $S/bootstrap.py serve --model-dir $S/models --host 127.0.0.1 --port 8084 \
      --max-concurrent-requests ${OMLX_MAX_CONC:-4} --hot-cache-max-size 4GB --memory-guard-gb ${OMLX_GUARD_GB:-110} --paged-ssd-cache-dir $S/omlx-home/cache \
      ${OMLX_EXTRA_ARGS:-} --api-key "$(cat $S/omlx-home/api_key.txt)" > $S/server-$MODE.log 2>&1 &
    for i in $(seq 1 60); do sleep 2; curl -s -m 2 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat $S/omlx-home/api_key.txt)" http://127.0.0.1:8084/v1/models | grep -q 200 && { echo "staging ($MODE) up on 8084 after $((i*2))s"; grep -m1 "small-M" $S/server-$MODE.log; exit 0; }; done
    echo "staging failed to start"; tail -5 $S/server-$MODE.log; exit 1;;
esac
