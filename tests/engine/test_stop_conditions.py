"""Stop conditions: EOS ids, stop strings across emits, and the token budget.

A stop string is matched on text, not on ids, so it can straddle any number of
tokens and it can land in the middle of one. Two things have to hold at once:
the client never sees a character of it, and the tokens after it are discarded
so the state the prefix cache is about to store does not contain them.
"""

from __future__ import annotations

import pytest

from titan.core.types import FinishReason, SequencePhase, SequenceState
from titan.engine.decode_cycle import (
    MTPDecodeCycle,
    PlainDecodeCycle,
    TextEmitter,
)

from tests.engine.conftest import (
    FakeBackend,
    FakeTokenizer,
    ScriptedDrafter,
    make_request,
)
from tests.engine.test_greedy_loop import decoding_sequence
from tests.engine.test_mtp_parity import perfect


def scripted_backend(stream, vocab=64):
    """A backend whose greedy continuation is exactly ``stream``."""
    order = list(stream)

    def next_token(context, _vocab):
        index = len(context) - 3  # the prompt below is three tokens long
        return order[index] if 0 <= index < len(order) else order[-1]

    return FakeBackend(vocab=vocab, next_token=next_token)


PIECES = {
    20: "Hello",
    21: " wor",
    22: "ld",
    23: "<|im_",
    24: "end|>",
    25: " tail",
    26: "STOP",
    0: "",
}


def emit_all(events):
    return "".join(event.text for event in events)


# ---------------------------------------------------------------------------
# EOS
# ---------------------------------------------------------------------------


def test_eos_ends_the_turn_and_is_not_emitted():
    backend = scripted_backend([20, 21, 0, 25])
    request = make_request((1, 2, 3), max_tokens=10, eos=(0,))
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer(PIECES))
    events = []
    while sequence.finish_reason is None:
        events.extend(cycle.run([sequence]).events)
    assert emit_all(events) == "Hello wor"
    assert sequence.finish_reason is FinishReason.STOP
    assert sequence.committed == 2
    assert 0 not in sequence.tokens


def test_eos_inside_an_accepted_chain_discards_the_rest():
    """An MTP cycle that accepts past an EOS keeps nothing past it."""
    backend = scripted_backend([20, 0, 21, 22])
    request = make_request((1, 2, 3), max_tokens=10, eos=(0,))
    sequence = decoding_sequence(backend, request)
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(PIECES),
        drafter=ScriptedDrafter(perfect(scripted_backend([20, 0, 21, 22]))),
        max_depth=3,
    )
    events = cycle.run([sequence]).events
    assert emit_all(events) == "Hello"
    assert sequence.finish_reason is FinishReason.STOP
    assert sequence.tokens[3:] == [20]


# ---------------------------------------------------------------------------
# stop strings
# ---------------------------------------------------------------------------


def test_a_stop_string_split_across_two_emits_is_never_leaked():
    """``<|im_end|>`` arrives as two tokens; neither half may reach the client."""
    backend = scripted_backend([20, 23, 24, 25])
    request = make_request((1, 2, 3), max_tokens=10, stop_strings=("<|im_end|>",))
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer(PIECES))
    events = []
    while sequence.finish_reason is None:
        events.extend(cycle.run([sequence]).events)
    text = emit_all(events)
    assert text == "Hello"
    assert "<|im_" not in text
    assert sequence.finish_reason is FinishReason.STOP
    assert sequence.tokens[3:] == [20]


def test_text_is_held_back_while_a_suffix_could_still_become_a_stop():
    """The half-token is withheld at the moment it arrives, not retracted later."""
    backend = scripted_backend([20, 23, 25, 25])
    request = make_request((1, 2, 3), max_tokens=3, stop_strings=("<|im_end|>",))
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer(PIECES))

    first = cycle.run([sequence]).events
    assert emit_all(first) == "Hello"
    second = cycle.run([sequence]).events
    # "<|im_" is a live prefix of the stop string, so nothing of it goes out.
    assert emit_all(second) == ""
    third = cycle.run([sequence]).events
    # " tail" settles it: the suffix can no longer become the stop string.
    assert emit_all(third) == "<|im_ tail"


def test_a_stop_string_inside_one_token_cuts_at_the_token():
    """Text is cut at the stop string; tokens are cut at the token boundary.

    They disagree by design when a stop string starts mid-token. The client
    gets every character before the stop, which is what a client asking for a
    stop string means; the state keeps whole tokens only, because half a token
    is not a thing a KV cache can hold.
    """
    backend = scripted_backend([20, 26, 25, 25])
    request = make_request((1, 2, 3), max_tokens=10, stop_strings=("TO",))
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer(PIECES))
    events = []
    while sequence.finish_reason is None:
        events.extend(cycle.run([sequence]).events)
    assert emit_all(events) == "HelloS"
    # The stop starts inside token 26, so the token is dropped whole: a partial
    # token cannot be committed to the state.
    assert sequence.tokens[3:] == [20]


def test_several_stop_strings_take_the_earliest():
    backend = scripted_backend([20, 21, 22, 25])
    request = make_request((1, 2, 3), max_tokens=10, stop_strings=("ld", "wor"))
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer(PIECES))
    events = []
    while sequence.finish_reason is None:
        events.extend(cycle.run([sequence]).events)
    assert emit_all(events) == "Hello "
    assert sequence.finish_reason is FinishReason.STOP


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------


def test_max_tokens_counts_accepted_tokens_not_cycles():
    backend = FakeBackend()
    request = make_request((11, 12, 13), max_tokens=7)
    sequence = decoding_sequence(backend, request)
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(),
        drafter=ScriptedDrafter(perfect(FakeBackend())),
        max_depth=4,
    )
    while sequence.finish_reason is None:
        cycle.run([sequence])
    assert sequence.committed == 7
    assert sequence.finish_reason is FinishReason.LENGTH
    assert backend.state_length(sequence.state) == len(sequence.tokens) - 1


def test_max_total_tokens_ends_the_turn_too():
    backend = FakeBackend()
    request = make_request((11, 12, 13), max_tokens=100, max_total_tokens=9)
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=FakeTokenizer())
    while sequence.finish_reason is None:
        cycle.run([sequence])
    assert len(sequence.tokens) == 9
    assert sequence.finish_reason is FinishReason.LENGTH


# ---------------------------------------------------------------------------
# the emitter on its own
# ---------------------------------------------------------------------------


def test_the_emitter_reports_how_many_tokens_survive():
    tokenizer = FakeTokenizer(PIECES)
    request = make_request(stop_strings=("world",))
    emitter = TextEmitter(tokenizer, 1, request.stop)
    result = emitter.push([20, 21, 22, 25], budget_left=10)
    assert result.text == "Hello "
    assert result.keep == 1
    assert result.finish_reason is FinishReason.STOP


def test_the_emitter_holds_nothing_back_without_stop_strings():
    tokenizer = FakeTokenizer(PIECES)
    request = make_request()
    emitter = TextEmitter(tokenizer, 1, request.stop)
    assert emitter.push([20, 21, 22], budget_left=10).text == "Hello world"
