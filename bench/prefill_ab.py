#!/usr/bin/env python3
"""Cold-prefill and multi-turn cache test against the staging server (8084, Qwen3.6-35B-A3B-4bit).
Usage: prefill_ab.py --tag <label> [--tokens 32000] [--turns 6]"""
import argparse, json, os, random, time, urllib.request, urllib.error
URL="http://127.0.0.1:8084/v1/chat/completions"; MODEL=os.environ.get("BENCH_MODEL","Qwen3.8-Flash-Next-oQ4e-mtp")
KEY=open(os.path.expanduser("~/inference-server/staging/omlx-home/api_key.txt")).read().strip()
def call(msgs, mt=32):
    r=urllib.request.Request(URL,data=json.dumps({"model":MODEL,"messages":msgs,"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode(),headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    t=time.time()
    try:
        d=json.load(urllib.request.urlopen(r,timeout=900))
    except urllib.error.HTTPError as e:
        body=e.read().decode(errors="replace")[:300]; print(f"HTTP {e.code}: {body}"); d={"usage":{"prompt_tokens":0,"completion_tokens":0},"choices":[{"message":{"content":""}}],"error":body}
    if "usage" not in d: print("no usage in response:", str(d)[:300]); d["usage"]={"prompt_tokens":0,"completion_tokens":0}
    return d,time.time()-t
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",required=True); ap.add_argument("--tokens",type=int,default=32000); ap.add_argument("--turns",type=int,default=6); a=ap.parse_args()
    call([{"role":"user","content":"hi"}],4)
    salt=random.randint(0,10**9); n=a.tokens//22
    fill="\n".join(f"def f{salt}_{i}(x):\n    # step {i%7}\n    return x*{i}+{(i*7)%13}" for i in range(n))
    d,w=call([{"role":"user","content":fill+"\n\nWhich function multiplies by 777? Name only."}]); u=d["usage"]
    print(f"[{a.tag}] cold prefill: {u['prompt_tokens']} tok in {w:.1f}s -> {u['prompt_tokens']/max(0.05,w-0.3):.0f} tok/s")
    # multi-turn: grow the conversation by ~600 tokens per turn, measure per-turn latency (cache reuse + re-prefill)
    msgs=[{"role":"system","content":"You are a code reviewer."},{"role":"user","content":fill[:len(fill)//2]+"\n\nSay OK."}]
    lat=[]
    for t in range(a.turns):
        d,w=call(msgs,16); msgs.append({"role":"assistant","content":d["choices"][0]["message"].get("content") or "OK"})
        msgs.append({"role":"user","content":"\n".join(f"def t{salt}_{t}_{i}(x): return x+{i}" for i in range(60))+"\n\nSay OK."})
        lat.append(w); print(f"[{a.tag}] turn {t+1}: {w:.2f}s (prompt {d['usage']['prompt_tokens']}, cached {d['usage'].get('prompt_tokens_details',{}).get('cached_tokens')})")
    print(f"[{a.tag}] multi-turn median latency {sorted(lat)[len(lat)//2]:.2f}s over {a.turns} turns")
if __name__=="__main__": main()
