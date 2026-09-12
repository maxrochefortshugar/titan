"""Which prompt framing reproduces the oMLX reference's first token? Raw ids, raw with specials, chat template."""
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
def first(ids, n=5):
    st = ModelState.new(model); res = m.prefill(ids, st, want_logits=True); lg = res.logits[0, -1].astype(mx.float32)
    top = mx.argsort(-lg)[:n].tolist(); return top, [round(float(lg[t]), 2) for t in top]
for i in (0, 1, 10):
    p = prompts()[i]["prompt"]; r = ref[i]["token_ids"][:6]
    print(f"== {ref[i]['id']} reference first ids {r} -> {tok.decode(r)!r}")
    for name, ids in (("raw", tok.encode(p, add_special_tokens=False)), ("raw+special", tok.encode(p, add_special_tokens=True)),
                      ("chat", tok.encode(tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True), add_special_tokens=False))):
        top, vals = first(ids); print(f"  {name:<12} n={len(ids):<5} top5 {top} {vals} -> {[tok.decode([t]) for t in top]}", flush=True)
print("PROBEDONE")
