#!/bin/bash
# M4 setup, opencode only. Removes the earlier Claude Code wiring (claude-local wrapper, its env file, PATH line) and
# writes the opencode provider for the M5 Max with Flash-Next as the default model and thinking variants. Run: bash opencode-only.sh
set -e
echo "== removing Claude Code wiring"
rm -f ~/.local/bin/claude-local ~/.config/m5max/env.json; rmdir ~/.config/m5max 2>/dev/null || true
sed -i '' '/# m5max claude-local/d' ~/.zshrc 2>/dev/null || true
# per-project pins written by 'claude-local pin' live in <repo>/.claude/settings.local.json; list them so you can delete the ones you no longer want
find ~ -maxdepth 4 -path '*/.claude/settings.local.json' -newer ~/.zshrc 2>/dev/null | xargs -I{} sh -c 'grep -lq "m5-128gb" "{}" && echo "  pin found: {}"' 2>/dev/null || true
echo "== writing opencode provider"
mkdir -p ~/.config/opencode
SERVER="https://<server>.<tailnet>.ts.net" KEY="${OMLX_API_KEY}" python3 - <<'PY'
import json,os
e=os.environ; p=os.path.expanduser('~/.config/opencode/opencode.json')
s=json.load(open(p)) if os.path.exists(p) else {}
s["$schema"]="https://opencode.ai/config.json"
s.setdefault('provider',{})['m5max']={"npm":"@ai-sdk/openai-compatible","name":"M5 Max (oMLX)",
  "options":{"baseURL":e['SERVER']+"/v1","apiKey":e['KEY']},
  "models":{"Qwen3.8-Flash-Next-oQ4e-mtp":{"name":"Qwen3.8 Flash-Next",
              "options":{"reasoningEffort":"medium"},
              "variants":{"low":{"reasoningEffort":"low"},"medium":{"reasoningEffort":"medium"},"xhigh":{"reasoningEffort":"xhigh"},
                          "off":{"chat_template_kwargs":{"enable_thinking":False}}}},
            "Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think":{"name":"Qwen3.8 Flash-Next (no thinking)"}}}
s["model"]="m5max/Qwen3.8-Flash-Next-oQ4e-mtp"
s["small_model"]="m5max/Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think"
json.dump(s,open(p,'w'),indent=2); print("opencode: provider m5max, default model", s["model"], "written to", p)
PY
curl -s -m 8 -o /dev/null -w "Server reachable: HTTP %{http_code} (200 = good)\n" -H "Authorization: Bearer ${OMLX_API_KEY}" "https://<server>.<tailnet>.ts.net/v1/models" || echo "Server not reachable: is Tailscale connected?"
echo "Done. In opencode: ctrl+t cycles thinking (low, medium, xhigh, off); ctrl+x then m switches model."
