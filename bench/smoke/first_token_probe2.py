import sys, os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/parity"))
import mlx.core as mx
from titan.adapters.mlx.loader import load_model
from titan.adapters.mlx.model import TitanQwenFlashNext
from titan.adapters.mlx.state import ModelState
from prompts import prompts
from transformers import AutoTokenizer
model_dir = Path(os.path.expanduser("~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
ref = json.load(open(ROOT / "bench/parity/reference-stock-nomtp.json"))["records"]
tok = AutoTokenizer.from_pretrained(str(model_dir))
model, _ = load_model(model_dir); m = TitanQwenFlashNext(model)
def first(ids, n=4):
    st = ModelState.new(model); res = m.prefill(ids, st, want_logits=True); lg = res.logits[0, -1].astype(mx.float32)
    top = mx.argsort(-lg)[:n].tolist(); return top, [round(float(lg[t]), 1) for t in top]
def chat(p, **kw): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True, **kw)
for i in (0, 1):
    p = prompts()[i]["prompt"]; r = ref[i]["token_ids"][:4]
    print(f"== {ref[i]['id']} ref {r} {tok.decode(r)!r}")
    t = chat(p); print("  template tail:", repr(t[-40:]))
    fr = {"chat_noopener": t[: t.rfind("<think>")] if t.rstrip().endswith("<think>") else t.replace("<think>\n", ""),
          "chat_nothink": chat(p, enable_thinking=False),
          "chat_low": chat(p, reasoning_effort="low"),
          "raw_im_start": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
          "raw_nl": p + "\n\n"}
    for name, s in fr.items():
        ids = tok.encode(s, add_special_tokens=False); top, vals = first(ids)
        print(f"  {name:<14} n={len(ids):<4} {top} {vals} {[tok.decode([x]) for x in top]}  tail={s[-25:]!r}", flush=True)
print("PROBEDONE")
