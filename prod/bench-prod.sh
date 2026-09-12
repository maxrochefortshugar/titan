#!/bin/bash
# Production benchmark: cold 65k prefill x2 (same probe as the workbench A/B), then the agentic suite (no-think) at 32k.
cd ~/inference-server; export OMLX_URL=http://127.0.0.1:8083 OMLX_KEY_FILE=~/.omlx/api_key.txt
echo "##### prod cold prefill #####"; kdev/bin/python staging/e2e_cold.py --tag prod --reps 2 --cool 45
echo "##### prod agentic suite (no-think) #####"; sleep 30; python3 bench.py Qwen3.8-Flash-Next-oQ4e-mtp --no-think --long-tokens 32000 --out prod/bench-prod-$(date +%Y%m%d).json
echo "##### prod agentic suite via no-think alias #####"; sleep 30; python3 bench.py Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think --long-tokens 32000 --out prod/bench-prod-alias-$(date +%Y%m%d).json 2>&1 | python3 -c "import json,sys; t=sys.stdin.read(); d=json.loads(t[t.index('{'):]); print({k:d[k] for k in ('short','long_uncached','long_cached','coding_pass','tool_call')})"
echo BENCHDONE
