#!/bin/bash
# Sets sane sampling for Flash-Next in opencode (temperature 0.7, top_p 0.8; the server supplies top_k 20). Run: bash opencode-sampling.sh
python3 - <<'PY'
import json,os
p=os.path.expanduser('~/.config/opencode/opencode.json'); s=json.load(open(p))
for mid in ("Qwen3.8-Flash-Next-oQ4e-mtp","Qwen3.8-Flash-Next-oQ4e-mtp:flash-next-no-think"):
    s['provider']['m5max']['models'][mid].setdefault('options',{}).update({"temperature":0.7,"topP":0.8})
json.dump(s,open(p,'w'),indent=2); print("opencode: sampling set (temperature 0.7, top_p 0.8) for both Flash-Next entries")
PY
