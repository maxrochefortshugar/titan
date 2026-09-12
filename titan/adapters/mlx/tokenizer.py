"""Tokenizer adapter with streaming-safe incremental detokenisation.

The held-back-suffix rule is the whole job: a byte-level BPE can turn an already
emitted string into a different one when the next token arrives, so the adapter
keeps the shortest suffix that could still change and releases it on the next
call. Getting this wrong shows up as mojibake in a client, or worse, as a stop
string that never matches because it straddled two emits.
"""

from __future__ import annotations

__all__: list[str] = []
