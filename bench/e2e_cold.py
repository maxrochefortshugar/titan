#!/usr/bin/env python3
"""Cold-text 32k prefill (fresh random ids every call) + short decode, against 8084. Usage: e2e_cold.py --tag X [--reps 2] [--cool 45]"""
import argparse,json,os,random,time,urllib.request,urllib.error,statistics as st
URL=os.environ.get("OMLX_URL","http://127.0.0.1:8084")+"/v1/chat/completions"; MODEL="Qwen3.8-Flash-Next-oQ4e-mtp"; KEY=open(os.path.expanduser(os.environ.get("OMLX_KEY_FILE","~/inference-server/staging/omlx-home/api_key.txt"))).read().strip()
WORDS="alpha beta gamma delta kernel tensor buffer stride lambda socket thread mutex cache page index router expert gather scatter norm residual".split()
def prompt(n):
    r=random.Random(); return " ".join(f"{r.choice(WORDS)}{r.randint(0,99999)}" for _ in range(n))
def call(text,mt):
    q=urllib.request.Request(URL,data=json.dumps({"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":mt,"temperature":0}).encode(),headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    t=time.time(); 
    try: d=json.load(urllib.request.urlopen(q,timeout=1200))
    except urllib.error.HTTPError as e: print("HTTP",e.code,e.read()[:200]); return None,0
    return d,time.time()-t
a=argparse.ArgumentParser(); a.add_argument("--tag",required=True); a.add_argument("--reps",type=int,default=2); a.add_argument("--cool",type=int,default=45); a.add_argument("--words",type=int,default=11000); a=a.parse_args()
call("hi",4); rates=[]
for i in range(a.reps):
    time.sleep(a.cool); d,w=call(prompt(a.words)+"\n\nReply with the single word OK.",8)
    if d: u=d["usage"]; r=u["prompt_tokens"]/max(0.05,w-0.4); rates.append(r); print(f"[{a.tag}] rep{i+1}: {u['prompt_tokens']} cold tok in {w:.1f}s -> {r:.0f} tok/s")
time.sleep(15); d,w=call("Explain the CAP theorem in detail with examples.",300); print(f"[{a.tag}] decode {d['usage']['completion_tokens']/w:.1f} tok/s")
print(f"[{a.tag}] RESULT cold prefill median {st.median(rates):.0f} tok/s")
