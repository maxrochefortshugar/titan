#!/usr/bin/env python3
"""Summarise oMLX MTP log lines (accept rate, tok/cycle, depth attempts) from a server log, optionally after a byte offset."""
import re,sys,statistics as st
log=sys.argv[1]; start=int(sys.argv[2]) if len(sys.argv)>2 else 0
txt=open(log,errors="ignore").read()[start:]
acc=[];tpc=[];d=[0,0,0,0,0,0];n=[0,0,0,0,0,0]
for m in re.finditer(r"MTP\[\d+\].*?tok/cycle=([\d.]+) accept=(\d+)/(\d+) \(([\d.]+)%\) depth\[(.*?)\]",txt):
    tpc.append(float(m.group(1))); acc.append(float(m.group(4)))
    for dm in re.finditer(r"d(\d)=(\d+)/(\d+)",m.group(5)):
        i=int(dm.group(1)); d[i]+=int(dm.group(2)); n[i]+=int(dm.group(3))
if tpc: print(f"MTP requests={len(tpc)} tok/cycle median={st.median(tpc):.2f} accept median={st.median(acc):.1f}% per-depth acc/att: "+" ".join(f"d{i}={d[i]}/{n[i]}" for i in range(1,6) if n[i]))
else: print("no MTP lines")
