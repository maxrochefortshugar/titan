#!/usr/bin/env python3
"""Benchmark an oMLX model for agentic coding: speed at short and long context, plus a small
correctness probe (coding tasks verified by running their tests, and a tool-call formatting check).

Usage: python3 bench.py <model_id> [--long-tokens 32000] [--out results.json]
Speed numbers come from oMLX's own usage block (total_time) and the server log tok/s lines.
"""
import argparse, json, os, subprocess, sys, tempfile, textwrap, time, urllib.request

BASE = os.environ.get("OMLX_URL", "http://127.0.0.1:8083")
KEY = open(os.path.expanduser("~/.omlx/api_key.txt")).read().strip()

NO_THINK = False

def chat(model, messages, max_tokens=512, tools=None, temperature=0.0):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    d["_wall"] = time.time() - t0
    return d

def strip_think(s):
    if "</think>" in s:
        s = s.split("</think>")[-1]
    return s.strip()

def code_block(s):
    s = strip_think(s)
    if "```" in s:
        parts = s.split("```")
        for p in parts[1::2]:
            p = p.split("\n", 1)
            return p[1] if len(p) > 1 and p[0].strip() in ("python", "py", "") else "\n".join(p)
    return s

# --- correctness probes: (prompt, test code appended and executed) ---
TASKS = [
    ("Write a Python function `lru(capacity)` returning an object with get(key) and put(key, value) methods "
     "implementing an LRU cache in O(1). Code only, no explanation.",
     "c=lru(2);c.put(1,1);c.put(2,2);assert c.get(1)==1;c.put(3,3);assert c.get(2)==-1 or c.get(2) is None;c.put(4,4);assert c.get(1) in (-1,None);assert c.get(3)==3;assert c.get(4)==4"),
    ("Write a Python function `parse_duration(s)` that parses strings like '1h30m', '45s', '2h', '90m10s' "
     "into total seconds as an int. Raise ValueError on invalid input. Code only.",
     "assert parse_duration('1h30m')==5400;assert parse_duration('45s')==45;assert parse_duration('2h')==7200;assert parse_duration('90m10s')==5410\ntry:\n parse_duration('abc');raise SystemExit('no error')\nexcept ValueError: pass"),
    ("Write a Python function `topo_sort(graph)` where graph is a dict node->list of dependencies. "
     "Return a list ordering dependencies before dependents. Raise ValueError on a cycle. Code only.",
     "o=topo_sort({'a':['b','c'],'b':['c'],'c':[]});assert o.index('c')<o.index('b')<o.index('a')\ntry:\n topo_sort({'x':['y'],'y':['x']});raise SystemExit('no cycle error')\nexcept ValueError: pass"),
    ("Write a Python function `merge_intervals(iv)` taking a list of [start,end] pairs and returning merged, "
     "sorted intervals. Code only.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]];assert merge_intervals([[1,4],[4,5]])==[[1,5]];assert merge_intervals([])==[]"),
    ("Write a Python function `apply_patch(text, patch)` where patch is a list of (old, new) string pairs that "
     "must each match exactly once in text; apply them all and return the result. Raise ValueError if any old "
     "string matches zero or more than one time. Code only.",
     "assert apply_patch('a b c','')=='a b c' if False else True\nassert apply_patch('foo bar baz',[('bar','qux')])=='foo qux baz'\ntry:\n apply_patch('a a',[('a','b')]);raise SystemExit('dup ok')\nexcept ValueError: pass\ntry:\n apply_patch('a',[('z','b')]);raise SystemExit('missing ok')\nexcept ValueError: pass"),
]

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from disk",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write content to a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]

def run_task(model, prompt, test):
    d = chat(model, [{"role": "user", "content": prompt}], max_tokens=16384)
    msg = d["choices"][0]["message"]; code = code_block(msg.get("content") or "")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test + "\nprint('OK')\n")
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0 and "OK" in r.stdout
    except subprocess.TimeoutExpired:
        ok = False
    return ok, d["usage"], d["_wall"], len((msg.get("reasoning_content") or ""))//4, d["choices"][0].get("finish_reason")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--long-tokens", type=int, default=32000)
    ap.add_argument("--out", default=None); ap.add_argument("--no-think", action="store_true")
    a = ap.parse_args()
    global NO_THINK; NO_THINK = a.no_think
    res = {"model": a.model, "thinking": not a.no_think, "when": time.strftime("%Y-%m-%d %H:%M")}

    # warm load
    chat(a.model, [{"role": "user", "content": "hi"}], max_tokens=8)

    # 1. short-context decode
    d = chat(a.model, [{"role": "user", "content": "Explain how a B-tree differs from a B+tree, in detail."}], max_tokens=400)
    res["short"] = {"completion_tokens": d["usage"]["completion_tokens"], "wall_s": round(d["_wall"], 2),
                    "decode_tok_s": round(d["usage"]["completion_tokens"] / d["_wall"], 1)}

    # 2. long-context prefill (uncached) then decode; a synthetic code-ish document
    filler = "\n".join(f"def func_{i}(x):\n    # helper number {i} does step {i%7}\n    return x * {i} + {i%13}" for i in range(a.long_tokens // 22))
    msgs = [{"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": filler + "\n\nWhich function multiplies by 4242? Answer with the function name only."}]
    d = chat(a.model, msgs, max_tokens=1024)
    pt, ct = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
    res["long_uncached"] = {"prompt_tokens": pt, "wall_s": round(d["_wall"], 1),
                            "approx_prefill_tok_s": round(pt / max(0.01, d["_wall"] - ct / max(1, res["short"]["decode_tok_s"])), 0),
                            "answer": strip_think(d["choices"][0]["message"].get("content") or "")[:60]}
    # 3. same prefix again, new question -> prefix cache hit
    msgs[1] = {"role": "user", "content": filler + "\n\nWhich function multiplies by 1234? Answer with the function name only."}
    d = chat(a.model, msgs, max_tokens=1024)
    res["long_cached"] = {"cached_tokens": d["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
                          "wall_s": round(d["_wall"], 1), "answer": strip_think(d["choices"][0]["message"].get("content") or "")[:60]}

    # 4. correctness probes
    passes, details = 0, []
    for prompt, test in TASKS:
        ok, usage, wall, rtok, fr = run_task(a.model, prompt, test)
        passes += ok; details.append({"ok": ok, "tokens": usage["completion_tokens"], "approx_reasoning_tokens": rtok, "finish": fr, "wall_s": round(wall, 1)})
    res["coding_pass"] = f"{passes}/{len(TASKS)}"; res["coding_detail"] = details

    # 5. tool-call formatting
    d = chat(a.model, [{"role": "user", "content": "Read the file src/config.yaml and tell me what it contains. Use the tools."}], max_tokens=512, tools=TOOLS)
    m = d["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    ok = bool(tc) and tc[0]["function"]["name"] == "read_file" and "config.yaml" in tc[0]["function"]["arguments"]
    res["tool_call"] = {"ok": ok, "finish_reason": d["choices"][0].get("finish_reason"), "calls": [c["function"] for c in tc][:2]}

    # 6. tool call after a long prompt (qwen3.5-family parsers have reported unclosed tool-call envelopes past ~20k tokens)
    d = chat(a.model, [{"role": "system", "content": "You are a coding agent."},
                       {"role": "user", "content": filler[: len(filler)//2] + "\n\nNow read the file src/config.yaml using the tools."}],
             max_tokens=1024, tools=TOOLS)
    m = d["choices"][0]["message"]; tc = m.get("tool_calls") or []
    res["tool_call_long"] = {"prompt_tokens": d["usage"]["prompt_tokens"],
        "ok": bool(tc) and tc[0]["function"]["name"] == "read_file" and "config.yaml" in tc[0]["function"]["arguments"],
        "finish_reason": d["choices"][0].get("finish_reason"), "leaked_markup": "<tool_call" in (m.get("content") or "")}

    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
