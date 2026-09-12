#!/bin/sh
# Wait for the GPU to be free, then run the three microbenchmarks.
# Usage: sh run_benches.sh [max_wait_seconds]
cd "$(dirname "$0")" || exit 1
PY="$HOME/inference-server/kdev/bin/python"
FLAG="$HOME/inference-server/staging/GPU_FREE"
MAX=${1:-3600}
W=0
while [ ! -f "$FLAG" ]; do
    [ "$W" -ge "$MAX" ] && { echo "GPU_FREE never appeared after ${MAX}s"; exit 2; }
    sleep 20
    W=$((W + 20))
done
echo "GPU_FREE seen after ${W}s: $(date)"

echo "=== bench_topk.py (float32) ==="
"$PY" bench_topk.py --dtype float32
echo
echo "=== bench_topk.py (bfloat16) ==="
"$PY" bench_topk.py --dtype bfloat16
echo
echo "=== bench_wsum_verify.py ==="
"$PY" bench_wsum_verify.py
echo
echo "=== bench_ple_workers.py (F_NOCACHE, real rows.bin) ==="
"$PY" bench_ple_workers.py --json ple_workers.json
echo
echo "done: $(date)"
