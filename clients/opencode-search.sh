#!/bin/bash
# Adds a web search tool to opencode: the SearXNG instance on the M5 Max (docker, tailnet-only via Tailscale Serve on port 8443),
# through the mcp-searxng MCP server run locally with npx. Run: bash opencode-search.sh   (needs node/npx on this Mac)
set -e; command -v npx >/dev/null || { echo "npx not found: install node first (brew install node)"; exit 1; }
python3 - <<'PY'
import json,os
p=os.path.expanduser('~/.config/opencode/opencode.json'); s=json.load(open(p)) if os.path.exists(p) else {"$schema":"https://opencode.ai/config.json"}
s.setdefault("mcp",{})["searxng"]={"type":"local","command":["npx","-y","mcp-searxng"],"environment":{"SEARXNG_URL":"https://<server>.<tailnet>.ts.net:8443"},"enabled":True}
json.dump(s,open(p,'w'),indent=2); print("opencode: mcp server 'searxng' added (tools: searxng_web_search, web_url_read)")
PY
curl -s -m 10 -o /dev/null -w "SearXNG reachable over the tailnet: HTTP %{http_code} (200 = good)\n" "https://<server>.<tailnet>.ts.net:8443/search?q=test&format=json"
echo "Check with: opencode mcp list"
