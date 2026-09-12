#!/bin/bash
K=$(cat ~/.omlx/api_key.txt)
curl -s -X POST http://127.0.0.1:8083/admin/api/login -H 'Content-Type: application/json' -d "{\"api_key\":\"$K\"}" -c /tmp/omlx-ck.txt -o /dev/null
curl -s -b /tmp/omlx-ck.txt -X PUT http://127.0.0.1:8083/admin/api/models/Qwen3.8-Flash-Next-oQ4e-mtp/settings -H 'Content-Type: application/json' -d '{"is_pinned": true}' -o /dev/null -w "pin: %{http_code}\n"
echo "loading..."; curl -s -m 900 http://127.0.0.1:8083/v1/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{"model":"Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think","messages":[{"role":"user","content":"Say ready."}],"max_tokens":8}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print("reply:", d["choices"][0]["message"]["content"].strip())'
echo "--- patch lines"; grep -E "prod-patch\] (PLE packed|bf16|post-load|import-time|moe-int8)" /var/log/omlx/stderr.log | tail -n 6 | cut -c1-150
sysctl -n vm.swapusage
echo "--- benchmark"; ~/inference-server/prod/bench-prod.sh
