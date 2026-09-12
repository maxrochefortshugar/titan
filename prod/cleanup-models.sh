#!/bin/bash
# Removes duplicate and orphaned model stores now that oMLX is the only server.
# Frees roughly 330 GB. Review before running; re-downloading is slow.
# Run:  bash cleanup-models.sh
set -eu
M="$HOME/Engineering/MLX/_models"

echo "== Keep: everything in $M plus z-lab DFlash2 drafter in the HF cache"
echo "== Move Qwen3-Coder-Next 8-bit (the boring, always-works fallback) from LM Studio into the oMLX model dir"
if [ -d "$HOME/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-8bit" ]; then
  mv "$HOME/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-8bit" "$M/Qwen3-Coder-Next-8bit"
  rm -rf "$M/Qwen3-Coder-Next-4bit"        # superseded by the 8-bit copy
fi

echo "== Remove the rest of LM Studio's store (Qwen3.6-27B 8-bit, Qwen3-4B duplicate)"
rm -rf "$HOME/.lmstudio"

echo "== Remove mlx-serve's store (Qwen3.8-27B 8-bit dup, Muse-Glimmer, 64 GB disk KV cache, chat history)"
rm -rf "$HOME/.mlx-serve"

echo "== Remove the MTPLX-runtime builds from the HF cache (built for a different runtime; oMLX has its own MTP build)"
rm -rf "$HOME/.cache/huggingface/hub/models--Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
rm -rf "$HOME/.cache/huggingface/hub/models--Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality"
rm -rf "$HOME/.cache/huggingface/hub/models--Jundot--Qwen3.8-Flash-Next-oQ4e-mtp"     # empty stub
rm -rf "$HOME/.cache/huggingface/hub/models--ddalcu--Qwen3.8-27B-MLX-Serve-4bit"      # empty stub
rm -rf "$HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2"             # empty stub
rm -rf "$HOME/.cache/huggingface/hub/datasets--MariusHobbhahn--swe-bench-verified-mini"

echo "== Remove the empty Qwen3.6-27B-MTP-5bit stub and the duplicate Qwen3-4B base model"
rm -rf "$M/Qwen3.6-27B-MTP-5bit" "$M/Qwen3-4B-4bit"

echo "== Clear stale caches"
rm -rf "$HOME/Library/Caches/Homebrew" "$HOME/.cache/uv" "$HOME/.cache/codex-runtimes" "$HOME/.npm/_cacache"
brew cleanup -s >/dev/null 2>&1 || true

echo
echo "== Models now served by oMLX:"
ls "$M"
df -h / | tail -1
echo "Restart the server so it rescans:  sudo launchctl kickstart -k system/io.titan.omlx   (or: omlx restart if still app-managed)"
