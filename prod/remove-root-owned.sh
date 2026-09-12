#!/bin/bash
# Removes the root-owned leftovers that need your password.
# Everything here was approved on 2026-09-11: GarageBand, iMovie, Xcode,
# the three Creator Studio apps, the EXO network daemon, Draw Things sandbox.
set -u

echo "== Repointing developer tools to Command Line Tools (keeps brew/git working)"
sudo xcode-select -s /Library/Developer/CommandLineTools || exit 1

echo "== Unloading and removing the EXO network daemon"
sudo launchctl bootout system /Library/LaunchDaemons/io.exo.networksetup.plist 2>/dev/null
sudo rm -f /Library/LaunchDaemons/io.exo.networksetup.plist

echo "== Removing apps"
sudo rm -rf \
  "/Applications/GarageBand.app" \
  "/Applications/iMovie.app" \
  "/Applications/Xcode.app" \
  "/Applications/Keynote Creator Studio.app" \
  "/Applications/Numbers Creator Studio.app" \
  "/Applications/Pages Creator Studio.app"

echo "== Removing GarageBand/iMovie sound libraries and Xcode leftovers"
sudo rm -rf "/Library/Application Support/GarageBand" "/Library/Audio/Apple Loops" \
  "/Library/Application Support/Logic" "$HOME/Library/Developer/Xcode" \
  "$HOME/Library/Caches/com.apple.dt.Xcode"

echo "== Removing the Draw Things sandbox container"
sudo rm -rf "$HOME/Library/Containers/com.liuliu.draw-things"

echo
echo "== Result"
xcode-select -p
ls /Applications
df -h / | tail -1
