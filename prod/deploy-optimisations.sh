#!/bin/bash
# Switch the production oMLX daemon to the patched launcher. Run: bash ~/inference-server/prod/deploy-optimisations.sh
# Undo: edit the plist back to ~/.omlx/bin/omlx (see harden-server.sh step 8) and reload the same way.
set -e
H=$HOME; KEY=$(cat "$H/.omlx/api_key.txt"); P=/Library/LaunchDaemons/io.titan.omlx.plist
sudo cp "$P" "$H/inference-server/prod/io.titan.omlx.plist.bak-$(date +%Y%m%d-%H%M%S)" && sudo chown "$USER" "$H"/inference-server/prod/*.bak-*
sudo tee "$P" >/dev/null <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.titan.omlx</string>
  <key>UserName</key><string>$USER</string>
  <key>ProgramArguments</key><array>
    <string>$H/inference-server/prod/run-omlx.sh</string>
    <string>serve</string>
    <string>--model-dir</string><string>$H/Engineering/MLX/_models</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8083</string>
    <string>--max-concurrent-requests</string><string>4</string>
    <string>--memory-guard-gb</string><string>100</string>
    <string>--api-key</string><string>$KEY</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>HOME</key><string>$H</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>WorkingDirectory</key><string>$H</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>/var/log/omlx/stdout.log</string>
  <key>StandardErrorPath</key><string>/var/log/omlx/stderr.log</string>
  <key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>8192</integer></dict>
</dict></plist>
PL
sudo chown root:wheel "$P"; sudo chmod 600 "$P"
sudo launchctl bootout system/io.titan.omlx 2>/dev/null || true
for i in $(seq 1 30); do launchctl print system/io.titan.omlx >/dev/null 2>&1 || break; sleep 1; done
sudo launchctl bootstrap system "$P"
echo "waiting for the server..."
for i in $(seq 1 60); do sleep 2; curl -s -m 2 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" http://127.0.0.1:8083/v1/models | grep -q 200 && break; done
grep -m1 "prod-patch" /var/log/omlx/stderr.log || echo "WARNING: patch hooks not logged"
# re-pin Flash-Next so it loads at boot, then trigger the load and confirm both patches engaged
curl -s -X POST http://127.0.0.1:8083/admin/api/login -H 'Content-Type: application/json' -d "{\"api_key\":\"$KEY\"}" -c /tmp/omlx-ck.txt -o /dev/null
curl -s -b /tmp/omlx-ck.txt -X PUT http://127.0.0.1:8083/admin/api/models/Qwen3.8-Flash-Next-oQ4e-mtp/settings -H 'Content-Type: application/json' -d '{"is_pinned": true}' -o /dev/null -w "pin: %{http_code}\n"
echo "loading Flash-Next (about 90 s)..."
curl -s -m 600 http://127.0.0.1:8083/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think","messages":[{"role":"user","content":"Say ready."}],"max_tokens":8}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print("reply:", d["choices"][0]["message"]["content"].strip())'
grep -E "prod-patch\] (PLE packed|bf16)" /var/log/omlx/stderr.log | tail -2
echo DONE
