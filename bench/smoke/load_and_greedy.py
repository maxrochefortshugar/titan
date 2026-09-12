"""First real-model smoke test: load through Titan's loader, prefill a short prompt, greedy-decode a few tokens.

Runs alone on the GPU. Prints load time, memory, and the decoded text so the loader and the forward path can be judged before the parity harness.
"""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
from titan.adapters.mlx.loader import load_model, plan_summary
from titan.adapters.mlx.model import TitanQwenFlashNext
from titan.adapters.mlx.state import ModelState

model_dir = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
t0 = time.perf_counter()
model, plan = load_model(model_dir)
mx.eval(model.parameters())
print(f"loaded in {time.perf_counter()-t0:.1f}s; active {mx.get_active_memory()/1e9:.1f} GB peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
print(plan_summary(plan)[:400], flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(model_dir))
msgs = [{"role": "user", "content": "Write a Python function that merges two sorted lists."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(prompt, add_special_tokens=False)
m = TitanQwenFlashNext(model)
state = ModelState.new(model)
t1 = time.perf_counter()
res = m.prefill(ids, state, want_logits=True)
tokn = int(mx.argmax(res.logits[0, -1]).item())
print(f"prefill {len(ids)} tok in {time.perf_counter()-t1:.2f}s", flush=True)
out = [tokn]; t2 = time.perf_counter()
for _ in range(n - 1):
    logits = m.decode([tokn], state)
    tokn = int(mx.argmax(logits[0]).item()); out.append(tokn)
dt = time.perf_counter() - t2
print(f"decode {len(out)} tok in {dt:.2f}s -> {len(out)/dt:.1f} tok/s (plain greedy, no MTP)", flush=True)
print("TEXT:", repr(tok.decode(out)), flush=True)
print(f"peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
print("SMOKEDONE")
