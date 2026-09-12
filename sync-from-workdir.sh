#!/bin/bash
# Sync the working tree (~/inference-server) into this repo with private strings scrubbed.
# The literal -> placeholder pairs live OUTSIDE the repo in ~/.config/titan/scrub.env (one "literal|placeholder" per line).
set -e; T=~/Engineering/titan; W=~/inference-server; E=~/.config/titan/scrub.env
[ -f "$E" ] || { echo "missing $E"; exit 1; }
rsync -a --delete --exclude='__pycache__' --exclude='*.log' --exclude='*.log.*' --exclude='wheels' --exclude='mlx-src' --exclude='orig' --exclude='corpus' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.u32' --exclude='*.bf16' --exclude='*.npz' --exclude='*.json.bak*' $W/kernels/ $T/engine/patches/
cp $W/prod/bootstrap.py $T/engine/bootstrap/; cp $W/prod/run-omlx.sh $W/prod/deploy-optimisations.sh $W/prod/verify-and-bench.sh $W/prod/bench-prod.sh $W/harden-server.sh $W/cleanup-models.sh $W/remove-root-owned.sh $T/prod/
for f in bench.py kbench.py; do cp $W/$f $T/bench/; done; for f in e2e_cold.py decode_bench.py prefill_ab.py concurrency_sweep.py mtp_stats.py bootstrap.py staging.sh memwatch.sh; do cp $W/staging/$f $T/bench/ 2>/dev/null || true; done
cp $W/m4/opencode-only.sh $W/m4/opencode-search.sh $W/m4/opencode-sampling.sh $T/clients/; cp $W/IMPROVEMENTS.md $W/inference-server-plan.md $W/client-configs.md $W/kernel-notes.md $W/kernels-report.md $T/docs/plan/; cp $W/research/*.md $T/docs/research/; cp $W/kernels/REPORT.md $W/kernels/AUDIT-2026-09-12.md $W/kernels/RESEARCH-2026-09-12.md $W/kernels/COMMON.md $T/docs/kernels/
cd $T; SEDARGS=(); while IFS='|' read -r lit ph; do [ -z "$lit" ] && continue; case "$lit" in \#*) continue;; esac; esc=$(printf '%s' "$lit" | sed 's/[.[\*^$/]/\\&/g'); SEDARGS+=(-e "s/$esc/$ph/g"); done < "$E"
{ git ls-files -o -m --exclude-standard; git ls-files; } | sort -u | while read -r f; do [ -f "$f" ] || continue; case "$f" in sync-from-workdir.sh|LICENSE|*.png|*.jpg) continue;; esac; sed -i '' "${SEDARGS[@]}" "$f" 2>/dev/null || true; done
echo "== residual scan (expects nothing)"; while IFS='|' read -r lit ph; do [ -z "$lit" ] && continue; case "$lit" in \#*|/Users/*) continue;; esac; git grep -nF -- "$lit" -- . ':!sync-from-workdir.sh' | head -2; done < "$E"; echo "(end)"; git status --short | wc -l
