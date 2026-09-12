"""The twenty fixed parity prompts.

Ten short prompts and ten that carry a ~4k-token prefix, because the two
regimes exercise different code: a short prompt never leaves the first prefill
chunk and decodes against a small QSA window, while a 4k prefix crosses chunk
boundaries, fills the n-gram history, and puts the GDN recurrence far enough
along that a one-ULP difference has had room to grow.

The prefix is generated, not pasted, so this file stays readable and the corpus
is identical on both sides of the comparison.
"""

from __future__ import annotations

SHORT_PROMPTS: tuple[str, ...] = (
    "Write a Python function that returns the nth triangular number.",
    "Explain the difference between a mutex and a semaphore in two sentences.",
    "What does the `-p` flag do in `mkdir -p`?",
    "Rewrite this sentence in the active voice: The report was written by the team.",
    "Give me a regex that matches an ISO 8601 date.",
    "In Rust, why does the borrow checker reject two mutable references?",
    "Summarise what a bloom filter is for, in one paragraph.",
    "Convert 98.6 degrees Fahrenheit to Celsius and show the arithmetic.",
    "List three reasons a TCP connection might hang without an error.",
    "What is the time complexity of heapsort, and why?",
)

PREFIX_TOPICS: tuple[str, ...] = (
    "a build system that caches by content hash",
    "an inventory service backed by an append-only log",
    "a text editor's undo stack",
    "a rate limiter shared across processes",
    "a scheduler that admits requests under a memory budget",
)


def long_prefix(target_words: int = 3000) -> str:
    """A deterministic filler prefix of roughly ``target_words`` words.

    Roughly 4k tokens at the usual English ratio.  Deterministic so that the
    reference capture and the Titan run see byte-identical prompts.
    """
    lines: list[str] = [
        "The following is an internal design note. Read it, then answer the "
        "question at the end.",
        "",
    ]
    words = 0
    section = 0
    while words < target_words:
        topic = PREFIX_TOPICS[section % len(PREFIX_TOPICS)]
        paragraph = (
            f"Section {section + 1}. We considered {topic}. "
            f"The first design kept every entry in memory, which was simple and "
            f"wrong at the sizes we care about. The second design spilled cold "
            f"entries to disk behind a small index, which cost one seek per miss "
            f"and held the resident set flat. Measurements at section "
            f"{section + 1} showed the miss rate settling near "
            f"{(section * 7) % 40 + 5} percent, and the tail latency dominated by "
            f"the write-behind queue rather than by the reads. We kept the second "
            f"design and wrote down the crossover so the next person does not "
            f"rediscover it."
        )
        lines.append(paragraph)
        lines.append("")
        words += len(paragraph.split())
        section += 1
    return "\n".join(lines)


LONG_QUESTIONS: tuple[str, ...] = (
    "Which design was kept, and what dominated the tail latency?",
    "Summarise the note in three bullet points.",
    "What was the miss rate in section 3?",
    "Name one thing the note says the next person should not rediscover.",
    "Write the note's conclusion as a single sentence.",
    "What was wrong with the first design?",
    "How many sections does the note have?",
    "Quote the sentence that mentions the write-behind queue.",
    "What would you measure next, given this note?",
    "Rewrite section 1 for a reader who has never seen a cache.",
)


def prompts() -> list[dict]:
    """The twenty prompts, each with a stable id."""
    out: list[dict] = []
    for index, text in enumerate(SHORT_PROMPTS):
        out.append({"id": f"short-{index:02d}", "kind": "short", "prompt": text})
    prefix = long_prefix()
    for index, question in enumerate(LONG_QUESTIONS):
        out.append(
            {
                "id": f"long-{index:02d}",
                "kind": "long",
                "prompt": f"{prefix}\n\nQuestion: {question}",
            }
        )
    return out
