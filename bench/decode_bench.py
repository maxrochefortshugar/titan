#!/usr/bin/env python3
"""Decode-heavy probe: several long greedy generations, mixed regimes. Reports tok/s per prompt and overall.
Env: OMLX_URL (default http://127.0.0.1:8084), OMLX_KEY_FILE, BENCH_MODEL. Usage: decode_bench.py --tag X [--tokens 600] [--think 0|1]"""
import argparse, json, os, time, urllib.request, statistics as st
URL=os.environ.get("OMLX_URL","http://127.0.0.1:8084")+"/v1/chat/completions"; MODEL=os.environ.get("BENCH_MODEL","Qwen3.8-Flash-Next-oQ4e-mtp")
KEY=open(os.path.expanduser(os.environ.get("OMLX_KEY_FILE","~/inference-server/staging/omlx-home/api_key.txt"))).read().strip()
CODE="\n".join(f"def step_{i}(x, cfg):\n    # stage {i}: scale then offset\n    y = x * cfg.get('scale_{i}', {i%7+1})\n    return y + cfg.get('offset_{i}', {i%13})" for i in range(40))
PROMPTS={
 "edit":  f"Here is a Python file:\n```python\n{CODE}\n```\nRewrite the whole file so every function has a docstring and type hints. Output the complete file, nothing else.",
 "code":  "Write a complete Python module implementing an LRU cache with TTL expiry, a background sweeper thread, thread safety, and a small test suite using unittest. Output code only.",
 "prose": "Explain in detail how TCP congestion control evolved from Tahoe to BBR, covering the motivation for each algorithm and the failure modes it addressed.",
 "json":  "Produce a JSON array of 60 objects describing fictional cities, each with fields name, country, population, founded, mayor, and three landmarks. Output JSON only.",
}
def call(text, mt, think):
    body={"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":bool(think)}}
    q=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(q,timeout=1800)); w=time.time()-t
    u=d["usage"]; return u["prompt_tokens"], u["completion_tokens"], w
a=argparse.ArgumentParser(); a.add_argument("--tag",required=True); a.add_argument("--tokens",type=int,default=600); a.add_argument("--think",type=int,default=0); a=a.parse_args()
call("hi",4,0); rates=[]; tot_c=tot_w=0
for name,p in PROMPTS.items():
    pt,ct,w=call(p,a.tokens,a.think); dec=w-pt/1400.0  # subtract approximate prefill time
    r=ct/max(0.05,dec); rates.append(r); tot_c+=ct; tot_w+=dec
    print(f"[{a.tag}] {name:5s}: {ct} tok in {dec:.1f}s -> {r:.1f} tok/s (prompt {pt})", flush=True)
print(f"[{a.tag}] DECODE median {st.median(rates):.1f} tok/s, aggregate {tot_c/tot_w:.1f} tok/s over {tot_c} tok")
