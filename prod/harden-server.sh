#!/bin/bash
# Turns this MacBook Pro into an unattended inference server.
# Everything here needs sudo. Each step is reversible; the undo is noted.
# Run:  bash harden-server.sh
set -u
USER_NAME="maxshugar"
HOME_DIR="/Users/$USER_NAME"
OMLX_KEY_FILE="$HOME_DIR/.omlx/api_key.txt"

echo "== 1. Power: never sleep, display off after 5 min, restart after power loss"
# undo: sudo pmset -a restoredefaults
sudo pmset -a sleep 0 disksleep 0 displaysleep 5 powernap 0 standby 0 \
  hibernatemode 0 autorestart 1 womp 1 tcpkeepalive 1 ttyskeepawake 1 autopoweroff 0

echo "== 2. Remote Login (SSH) on. Restrict to your user in System Settings > Sharing."
sudo systemsetup -setremotelogin on 2>/dev/null

echo "== 3. macOS updates: download but never auto-install or auto-restart"
# undo: set these back to 1
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate CriticalUpdateInstall -bool false
sudo defaults write /Library/Preferences/com.apple.commerce AutoUpdate -bool false

echo "== 4. Spotlight indexing off (it was crawling 300 GB of model files)"
# undo: sudo mdutil -a -i on
sudo mdutil -a -i off >/dev/null
sudo mdutil -a -E >/dev/null

echo "== 5. Time Machine off, local snapshots removed"
sudo tmutil disable 2>/dev/null
for s in $(tmutil listlocalsnapshots / 2>/dev/null | sed -n 's/.*\.\([0-9-]*\)$/\1/p'); do sudo tmutil deletelocalsnapshots "$s" >/dev/null; done

echo "== 6. Remove the Thunderbolt network services EXO created"
for svc in "EXO Thunderbolt 1" "EXO Thunderbolt 2" "EXO Thunderbolt 3"; do
  sudo networksetup -removenetworkservice "$svc" 2>/dev/null && echo "   removed $svc"
done

echo "== 7. GPU wired memory limit: 118 GB (default cap is ~96 GB on a 128 GB Mac)"
# Applied now and at every boot. undo: sudo sysctl iogpu.wired_limit_mb=0 and delete the plist
sudo sysctl iogpu.wired_limit_mb=120832
sudo tee /Library/LaunchDaemons/io.titan.iogpu-wired-limit.plist >/dev/null <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.titan.iogpu-wired-limit</string>
  <key>ProgramArguments</key><array>
    <string>/usr/sbin/sysctl</string><string>iogpu.wired_limit_mb=120832</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
EOF
sudo chown root:wheel /Library/LaunchDaemons/io.titan.iogpu-wired-limit.plist
sudo chmod 644 /Library/LaunchDaemons/io.titan.iogpu-wired-limit.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/io.titan.iogpu-wired-limit.plist 2>/dev/null

echo "== 8. oMLX as a system daemon (starts at boot, before anyone logs in)"
# The oMLX menu-bar app must NOT also auto-start the server, or the port collides.
# undo: sudo launchctl bootout system/io.titan.omlx && sudo rm the plist
OMLX_KEY=$(cat "$OMLX_KEY_FILE")
sudo mkdir -p /var/log/omlx && sudo chown "$USER_NAME" /var/log/omlx
sudo tee /Library/LaunchDaemons/io.titan.omlx.plist >/dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.titan.omlx</string>
  <key>UserName</key><string>$USER_NAME</string>
  <key>ProgramArguments</key><array>
    <string>$HOME_DIR/.omlx/bin/omlx</string>
    <string>serve</string>
    <string>--model-dir</string><string>$HOME_DIR/Engineering/MLX/_models</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8083</string>
    <string>--max-concurrent-requests</string><string>4</string>
    <string>--hot-cache-max-size</string><string>16GB</string>
    <string>--memory-guard-gb</string><string>110</string>
    <string>--api-key</string><string>$OMLX_KEY</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>HOME</key><string>$HOME_DIR</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>WorkingDirectory</key><string>$HOME_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>/var/log/omlx/stdout.log</string>
  <key>StandardErrorPath</key><string>/var/log/omlx/stderr.log</string>
  <key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>8192</integer></dict>
</dict></plist>
EOF
sudo chown root:wheel /Library/LaunchDaemons/io.titan.omlx.plist
sudo chmod 600 /Library/LaunchDaemons/io.titan.omlx.plist   # contains the API key
sudo tee /etc/newsyslog.d/omlx.conf >/dev/null <<EOF
# logfilename              [owner:group]      mode count size when flags
/var/log/omlx/stdout.log   $USER_NAME:wheel   644  7     20480 *   NJ
/var/log/omlx/stderr.log   $USER_NAME:wheel   644  7     20480 *   NJ
EOF
# Hand over from the app-managed server to the daemon
"$HOME_DIR/.omlx/bin/omlx" stop --timeout 60 >/dev/null 2>&1
python3 - <<EOF
import json; p="$HOME_DIR/.omlx/settings.json"; s=json.load(open(p))
s["server"]["auto_start_on_launch"]=False; json.dump(s,open(p,"w"),indent=2)
EOF
sudo launchctl bootstrap system /Library/LaunchDaemons/io.titan.omlx.plist
sleep 5
sudo launchctl print system/io.titan.omlx | grep -E "state|pid" | head -3

echo
echo "== Result"
pmset -g | grep -E "^ (sleep|displaysleep|autorestart|powernap|hibernatemode)"
sysctl iogpu.wired_limit_mb
mdutil -s / | tail -1
echo "oMLX daemon: $(curl -s -m 5 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $OMLX_KEY" http://127.0.0.1:8083/v1/models) (200 = serving)"
echo
echo "Still to do by hand in System Settings (no CLI exists):"
echo "  Battery > Charging > Charge Limit 80%     Battery > Options: keep Automatic power mode (not High Power)"
echo "  General > Sharing > Remote Login (i): only your user;  Screen Sharing on, your user only"
echo "  General > Software Update > Automatic Updates: confirm 'Install macOS updates' is off after every update"
echo "  Focus: Do Not Disturb all day.   Apple Account: sign out of iCloud on this machine (stops photo/media analysis daemons)"
