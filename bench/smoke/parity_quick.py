"""Quick M1 parity: Titan plain greedy (reference ops) against a captured oMLX reference.

Self-contained: uses the transformers tokenizer so it does not depend on the tokenizer adapter.
Prints agreeing token prefix and agreeing character prefix per prompt.
"""
import sys, time, os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/parity"))
import mlx.core as mx
from titan.adapters.mlx.loader import load_model
from titan.adapters.mlx.model import TitanQwenFlashNext
from titan.adapters.mlx.state import ModelState
from prompts import prompts
from transformers import AutoTokenizer

ref_path = Path(sys.argv[1]); model_dir = Path(os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
ref = json.load(open(ref_path)); max_tokens = ref["max_tokens"]
tok = AutoTokenizer.from_pretrained(str(model_dir))
cfg = json.load(open(model_dir / "config.json")); eos = cfg.get("eos_token_id"); eos = set(eos if isinstance(eos, list) else [eos])
t0 = time.perf_counter(); model, plan = load_model(model_dir); mx.eval(model.parameters()); print(f"loaded {time.perf_counter()-t0:.1f}s", flush=True)
m = TitanQwenFlashNext(model)
rows = []; texts = {}
for item, rec in zip(prompts(), ref["records"]):
    assert item["id"] == rec["id"]
    ids = tok.encode(item["prompt"], add_special_tokens=False)
    state = ModelState.new(model); t1 = time.perf_counter()
    res = m.prefill(ids, state, want_logits=True); t = int(mx.argmax(res.logits[0, -1]).item()); tp = time.perf_counter() - t1
    out = [t]
    while len(out) < max_tokens and t not in eos:
        t = int(mx.argmax(m.decode([t], state)[0]).item()); out.append(t)
    dt = time.perf_counter() - t1 - tp
    # oMLX strips leading whitespace from completion text, so compare stripped text and
    # re-encode both sides from the stripped text so token boundaries are comparable.
    text = tok.decode(out).lstrip(); rt = rec["text"].lstrip()
    r = tok.encode(rt, add_special_tokens=False); o2 = tok.encode(text, add_special_tokens=False); n = 0
    for a, b in zip(r, o2):
        if a != b: break
        n += 1
    c = 0
    for a, b in zip(rt, text):
        if a != b: break
        c += 1
    rows.append((rec["id"], n, len(r), c, len(rt), text.startswith(rt) or rt.startswith(text)))
    texts[rec["id"]] = text
    print(f"{rec['id']:<10} ids {n:>3}/{len(r):<3} chars {c:>4}/{len(rt):<4} {'EXACT' if text == rt else ''}  prefill {len(ids)} tok {tp:.1f}s decode {len(out)/dt:.1f} tok/s", flush=True)
ex = sum(1 for x in rows if x[5]); print(f"SUMMARY exact {ex}/{len(rows)} median agreeing ids {sorted(x[1] for x in rows)[len(rows)//2]}")
json.dump({"texts": texts, "rows": [dict(id=a, agree_ids=b, ref_len=c, agree_chars=d, ref_chars=e, exact=f) for a,b,c,d,e,f in rows]}, open(ROOT / f"bench/smoke/parity-{ref_path.stem}.json", "w"), indent=1)
print("PARITYDONE")
