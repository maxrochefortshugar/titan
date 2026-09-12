#!/usr/bin/env python3
"""Build a realistic token stream: public-domain prose + local source code.

Tokenized with the model's own tokenizer.json (the `tokenizers` fast tokenizer,
no model weights loaded).  Documents are joined with the eos token, 248044, so
the n-gram shift-right logic sees real document boundaries.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "corpus"
MODEL = Path(os.environ.get(
    "PROFILE_MODEL",
    "~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
EOS = 248044

BOOKS = {
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "war_and_peace": "https://www.gutenberg.org/files/2600/2600-0.txt",
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "origin_of_species": "https://www.gutenberg.org/files/2009/2009-h/2009-h.htm",
}

CODE_ROOTS = [
    Path("/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/"
         "lib/python3.11/site-packages/mlx"),
    Path("/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/"
         "lib/python3.11/site-packages/mlx_lm"),
    Path("/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/lib/python3.11"),
]
CODE_SUFFIXES = {".py"}


def fetch_books() -> list[str]:
    CACHE.mkdir(exist_ok=True)
    out = []
    for name, url in BOOKS.items():
        path = CACHE / f"{name}.txt"
        if not path.is_file():
            try:
                subprocess.run(["curl", "-sSL", "-m", "60", "-o", str(path), url],
                               check=True)
            except subprocess.CalledProcessError:
                continue
        text = path.read_text(errors="replace")
        if path.suffix == ".txt" and "<" in text[:200]:
            text = re.sub(r"<[^>]+>", " ", text)
        if len(text) > 2000:
            out.append(text)
    return out


def collect_code(limit_bytes=6_000_000) -> list[str]:
    docs, total = [], 0
    for root in CODE_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in CODE_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if len(text) < 400:
                continue
            docs.append(text)
            total += len(text)
            if total >= limit_bytes:
                return docs
    return docs


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL / "tokenizer.json"))


def build(force=False) -> dict:
    CACHE.mkdir(exist_ok=True)
    npz = CACHE / "tokens.npz"
    meta_path = CACHE / "tokens.json"
    if npz.is_file() and not force:
        data = np.load(npz)
        return {"tokens": data["tokens"], "prose_len": int(data["prose_len"]),
                "meta": json.loads(meta_path.read_text())}

    tok = tokenizer()
    prose_docs = fetch_books()
    code_docs = collect_code()
    prose_ids, code_ids = [], []
    for doc in prose_docs:
        prose_ids.append(np.array(tok.encode(doc).ids, dtype=np.int64))
        prose_ids.append(np.array([EOS], dtype=np.int64))
    for doc in code_docs:
        code_ids.append(np.array(tok.encode(doc).ids, dtype=np.int64))
        code_ids.append(np.array([EOS], dtype=np.int64))
    prose = np.concatenate(prose_ids) if prose_ids else np.zeros(0, np.int64)
    code = np.concatenate(code_ids) if code_ids else np.zeros(0, np.int64)
    tokens = np.concatenate([prose, code])
    meta = {
        "prose_documents": len(prose_docs),
        "code_documents": len(code_docs),
        "prose_tokens": int(prose.size),
        "code_tokens": int(code.size),
        "total_tokens": int(tokens.size),
        "distinct_token_ids": int(np.unique(tokens).size),
        "vocab_size": 248320,
    }
    np.savez(npz, tokens=tokens, prose_len=np.int64(prose.size))
    meta_path.write_text(json.dumps(meta, indent=1))
    return {"tokens": tokens, "prose_len": int(prose.size), "meta": meta}


if __name__ == "__main__":
    info = build(force=bool(os.environ.get("FORCE")))
    print(json.dumps(info["meta"], indent=1))
