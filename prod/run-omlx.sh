#!/bin/sh
# oMLX production launcher with kernel patches. Called by /Library/LaunchDaemons/io.titan.omlx.plist.
# Toggle patches here (no sudo needed afterwards: `sudo launchctl kickstart -k system/io.titan.omlx` restarts with the new values).
R=/Applications/oMLX.app/Contents/Resources
export OMLX_PLE_PACKED=1                 # streamed packed n-gram table (bit-exact, 3 page reads -> 1 per row)
export OMLX_PLE_PACKED_MODE=rows         # NEVER "resident" on the 128 GB machine (kernel panic 2026-09-12)
export OMLX_PLE_PACKED_DIR="$HOME/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed"
export OMLX_QWEN4_BF16_NORM=1            # bf16 grouped RMSNorm (<=1 ULP)
# Round 2 (2026-09-12, measured on the workbench: 65k cold prefill 1441 -> 1581 tok/s, decode unchanged)
R2="$HOME/inference-server/kernels/round2"
export OMLX_QWEN4_GDN_NORM_GATE=1        # fused GDN grouped norm + sigmoid gate at prefill (bit-identical)
export OMLX_WSUM_TOPK10=1 OMLX_WSUM_TOPK10_MIN_TOKENS=64   # fused MoE weighted sum for top_k=10, prefill only (bit-identical); decode keeps stock
export OMLX_MTP_SHORTLIST_DRAFT=1        # MTP draft steps 2..3 on a top-K shortlist head (verified output identical)
export OMLX_MOE_INT8_PREFILL=1 OMLX_MOE_INT8_SKIP_DOWN=1   # int8 x int4 expert gather at prefill, gate_up only (~7.5 GB of tables)
export OMLX_ROUND2_IMPORT_PATCHES="$R2/mtp/patch.py:install_shortlist_draft"
# Round 4 (2026-09-12): exact, zero-memory extras, +3% prefill paired on the workbench
R3="$HOME/inference-server/kernels/round3"
export OMLX_QWEN4_HC_FUSE2=1             # fused prefill hyper-connection block (bit-identical, 11 -> 6 launches)
export OMLX_QWEN4_GDN_SCAN=1             # chunked GDN prefill scan from mlx PR #4020, C=8 (state rrmse 6e-7)
export OMLX_WSUM_TOPK10_VERIFY=1         # fused weighted sum on the MTP verify layout (bit-identical)
export OMLX_QSA_BATCHED_SPARSE=1 OMLX_QSA_GATHER_MIN_CTX=8192   # sparse QSA arm for batched decode (1 ULP): +22-28% per stream at 68k with 2-4 streams
export OMLX_ROUND2_PATCHES="$R2/gdn-norm/patch.py,$R2/wsum10/patch.py,$R3/hc-fuse/patch.py,$R3/gdn-scan/patch.py,$R3/small-items/patch.py:install_wsum_verify,$R3/qsa-batched/patch.py"
export PYTHONHOME="$R/Python/cpython-3.11" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages"
BP="$HOME/Library/Application Support/oMLX/base-path"
[ -r "$BP" ] && IFS= read -r OMLX_BASE_PATH < "$BP" && [ -n "$OMLX_BASE_PATH" ] && export OMLX_BASE_PATH
# Memory guard: the plist passes 100; 110 is needed so concurrent requests are not serialised by the soft limit (85% of ceiling) once the int8 tables are resident (2026-09-12).
GUARD_GB=${OMLX_GUARD_GB:-110}
set -- $(printf '%s\n' "$@" | sed "s/^100$/$GUARD_GB/" | tr '\n' ' ')
exec "$R/Python/cpython-3.11/bin/python3" "$HOME/inference-server/prod/bootstrap.py" "$@"
