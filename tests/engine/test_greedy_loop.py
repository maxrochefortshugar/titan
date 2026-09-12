"""W1.9: the plain greedy loop, and what it is allowed to depend on.

The claim under test is D16's: greedy uses no RNG and no batch-dependent state,
and its output may not depend on padding width, batch composition or how many
tokens a cycle accepted. Here that reduces to one assertion made several ways:
the loop reproduces the backend's own argmax sequence, token for token.
"""

from __future__ import annotations

import pytest

from titan.core.types import FinishReason, SequencePhase, SequenceState
from titan.engine.decode_cycle import PlainDecodeCycle

from tests.engine.conftest import FakeBackend, FakeTokenizer, make_request


def decoding_sequence(backend: FakeBackend, request, sequence_id: int = 1) -> SequenceState:
    """A sequence in the state the scheduler hands the cycle: prefilled up to
    the last prompt token, which is pending."""
    prompt = list(request.prompt_tokens)
    state = backend.open_state(sequence_id, len(prompt) + request.stop.max_tokens)
    backend.prefill(state, prompt[:-1], snapshot=True)
    return SequenceState(
        sequence_id=sequence_id,
        request=request,
        phase=SequencePhase.DECODING,
        state=state,
        tokens=list(prompt),
        prompt_len=len(prompt),
        prefill_position=len(prompt) - 1,
    )


def test_plain_loop_reproduces_the_argmax_sequence(backend, tokenizer, clock):
    request = make_request((11, 12, 13), max_tokens=24)
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)

    for _ in range(24):
        if sequence.finish_reason is not None:
            break
        cycle.run([sequence])

    expected = backend.greedy_continuation(request.prompt_tokens, 24)
    assert sequence.tokens[3:] == expected
    assert sequence.committed == 24
    assert sequence.finish_reason is FinishReason.LENGTH


def test_one_token_per_cycle_and_one_host_sync(backend, tokenizer, clock):
    request = make_request((5, 6), max_tokens=8)
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)

    result = cycle.run([sequence])
    assert result.profile.host_syncs == 1
    assert result.profile.tokens_committed == 1
    assert result.profile.tokens_drafted == 0
    assert result.profile.n_rows == 1
    assert len(result.events) == 1
    assert result.events[0].token_ids == (sequence.tokens[-1],)


def test_the_state_holds_every_token_but_the_pending_one(backend, tokenizer, clock):
    """The decode invariant: ``state_length == len(tokens) - 1``, always."""
    request = make_request((7, 8, 9), max_tokens=6)
    sequence = decoding_sequence(backend, request)
    cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)
    for _ in range(6):
        cycle.run([sequence])
        assert backend.state_length(sequence.state) == len(sequence.tokens) - 1
        assert backend.state_tokens(sequence.state) == sequence.tokens[:-1]


def test_batch_composition_does_not_change_a_sequence(backend, tokenizer, clock):
    """Two sequences decoded together produce what each produces alone."""
    solo_backend = FakeBackend()
    requests = [
        make_request((11, 12, 13), request_id="a", max_tokens=10),
        make_request((21, 22), request_id="b", max_tokens=10),
    ]
    alone = []
    for index, request in enumerate(requests):
        sequence = decoding_sequence(solo_backend, request, sequence_id=index + 1)
        cycle = PlainDecodeCycle(
            backend=solo_backend, tokenizer=FakeTokenizer(), clock=clock
        )
        for _ in range(10):
            cycle.run([sequence])
        alone.append(list(sequence.tokens))

    together_backend = FakeBackend()
    sequences = [
        decoding_sequence(together_backend, request, sequence_id=index + 1)
        for index, request in enumerate(requests)
    ]
    cycle = PlainDecodeCycle(
        backend=together_backend, tokenizer=FakeTokenizer(), clock=clock
    )
    for _ in range(10):
        cycle.run(sequences)
    assert [list(s.tokens) for s in sequences] == alone


def test_an_empty_batch_is_a_cheap_no_op(backend, tokenizer, clock):
    cycle = PlainDecodeCycle(backend=backend, tokenizer=tokenizer, clock=clock)
    result = cycle.run([])
    assert result.events == ()
    assert result.profile.n_sequences == 0
    assert backend.verify_calls == []
