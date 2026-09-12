# Client configuration (opencode only, 2026-09-12)

Claude Code wiring was removed on 2026-09-12 at Max's request; the old sections are in archive/client-configs-with-claude.md.

## opencode

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "m5max": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "M5 Max (oMLX)",
      "options": {
        "baseURL": "https://<server>.<tailnet>.ts.net/v1",
        "apiKey": "PASTE_KEY_HERE"
      },
      "models": {
        "Qwen3.8-Flash-Next-oQ4e-mtp": { "name": "Qwen3.8 Flash-Next" },
        "Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think": { "name": "Flash-Next fast (no thinking)" }
      }
    }
  }
}
```

## qwen-code

```bash
export OPENAI_BASE_URL="https://<server>.<tailnet>.ts.net/v1"
export OPENAI_API_KEY="PASTE_KEY_HERE"
export OPENAI_MODEL="Qwen3.8-Flash-Next-oQ4e-mtp"
```

## Sanity check from any tailnet machine

```bash
curl -s https://<server>.<tailnet>.ts.net/v1/models -H "Authorization: Bearer PASTE_KEY_HERE"
```

Admin panel (benchmarks, model settings, cache stats): `https://<server>.<tailnet>.ts.net/admin`


Setup script for the M4: m4/opencode-only.sh (removes the Claude wiring, writes the provider with a medium default and low/medium/xhigh/off variants; ctrl+t cycles them, ctrl+x then m switches model).

## Web search (SearXNG on the M5, 2026-09-12)

SearXNG runs in docker on the server (127.0.0.1:8888) and is published tailnet-only at `https://<server>.<tailnet>.ts.net:8443` via Tailscale Serve (`tailscale serve --https=8443 off` to withdraw). Two consumers:
- oMLX's own web search (Responses API and Anthropic `web_search` tool types, plus `POST /v1/web/search` and `/v1/web/fetch`) now uses it: `web_search_provider=searxng`, url `http://127.0.0.1:8888`.
- opencode gets a real search tool through the `mcp-searxng` MCP server run locally with npx (`m4/opencode-search.sh`), tools `searxng_web_search` and `web_url_read`. Verify with `opencode mcp list`.

## Sampling (2026-09-12)

opencode sends no temperature and top_p 1.0 by default, so the server's defaults decide. Server model defaults for Flash-Next are now temperature 0.7, top_p 0.8, top_k 20 (Qwen's recommendation for this family); opencode additionally sends temperature 0.7 via the model options (m4/opencode-sampling.sh). Before this, sessions ran at temperature 1.0 with top_p 1.0 and no top_k, which produced rambling or truncated-looking answers on long outputs.
