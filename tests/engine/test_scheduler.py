"""The loop: turn ordering, chunked prefill, retirement, cancellation, usage.

Everything here drives ``EngineLoop.step`` directly, one turn at a time, which
is the same code ``run_forever`` calls. What is being checked is the order of a
turn and what each stage leaves behind, because that ordering is the whole
scheduling contract: prefill and decode never share a turn, state handles close
at a turn boundary and never inside a cycle, and a finished sequence is offered
to the cache only when the state actually matches the tokens.
"""

from __future__ import annotations

import pytest

from titan.core.types import FinishReason, SequencePhase, StreamEnd, TokenEvent
from titan.engine.admission import AdmissionConfig
from titan.engine.decode_cycle import NullProfiler, PlainDecodeCycle
from titan.engine.scheduler import EngineLoop

from tests.engine.conftest import (
    FakeBackend,
    FakeCache,
    FakeClock,
    FakeTokenizer,
    make_request,
)


class Sink:
    """Stands in for the asyncio queue the engine bridges to."""

    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)

    @property
    def tokens(self) -> list[int]:
        out: list[int] = []
        for event in self.events:
            if isinstance(event, TokenEvent):
                out.extend(event.token_ids)
        return out

    @property
    def end(self) -> StreamEnd | None:
        for event in self.events:
            if isinstance(event, StreamEnd):
                return event
        return None


def build_loop(*, cache=None, config=None, backend=None, tokenizer=None, profiler=None, **kwargs):
    backend = backend or FakeBackend()
    tokenizer = tokenizer or FakeTokenizer()
    clock = FakeClock()
    config = config or AdmissionConfig(prefill_chunk_tokens=2048, block_tokens=512)
    profiler = profiler or NullProfiler()
    cycle = kwargs.pop("cycle", None) or PlainDecodeCycle(
        backend=backend, tokenizer=tokenizer, clock=clock, profiler=profiler
    )
    loop = EngineLoop(
        backend=backend,
        tokenizer=tokenizer,
        cycle=cycle,
        config=config,
        cache=cache,
        clock=clock,
        profiler=profiler,
        **kwargs,
    )
    return loop, backend, tokenizer


def drain(loop, limit: int = 500) -> int:
    turns = 0
    while turns < limit and loop.step():
        turns += 1
    return turns


# ---------------------------------------------------------------------------
# turn ordering
# ---------------------------------------------------------------------------


def test_a_turn_runs_one_prefill_chunk_and_no_decode():
    loop, backend, _tokenizer = build_loop()
    sink = Sink()
    loop.submit(make_request(tuple(range(5000)), max_tokens=2), sink)

    loop.step()  # drain the command queue, admit, and run the first chunk
    assert backend.prefill_calls == [(1, 2048, True)]
    assert backend.verify_calls == []
    assert loop.live[0].phase is SequencePhase.PREFILLING

    loop.step()
    assert [call[1] for call in backend.prefill_calls] == [2048, 2048]
    assert backend.verify_calls == []

    loop.step()
    # The block floor of the 4999-token prefill is 4608, and it ends a chunk of
    # its own so the terminal snapshot has somewhere to be staged.
    assert [call[1] for call in backend.prefill_calls] == [2048, 2048, 512]
    assert loop.live[0].phase is SequencePhase.PREFILLING
    assert backend.verify_calls == []

    loop.step()
    # 4999 tokens of prefill: the last chunk is short and ends the phase.
    assert [call[1] for call in backend.prefill_calls] == [2048, 2048, 512, 391]
    assert loop.live[0].phase is SequencePhase.DECODING
    assert backend.verify_calls == []

    loop.step()
    assert backend.verify_calls == [1]


def test_snapshots_are_staged_on_the_grid_and_at_the_prefill_end():
    loop, backend, _tokenizer = build_loop()
    loop.submit(make_request(tuple(range(5000)), max_tokens=1), Sink())
    drain(loop)
    staged = [call for call in backend.prefill_calls if call[2]]
    # 2048, 4096 are the grid multiples; 4608 is the block floor of the 4999
    # token prefill, which is where the terminal snapshot can be restored from.
    assert [call[1] for call in staged] == [2048, 2048, 512]
    assert len(staged) == 3


def test_a_short_prompt_needs_no_prefill_at_all():
    """A one-token prompt is entirely pending: the first cycle consumes it."""
    loop, backend, _tokenizer = build_loop()
    sink = Sink()
    loop.submit(make_request((7,), max_tokens=3), sink)
    drain(loop)
    assert backend.prefill_calls == []
    assert len(sink.tokens) == 3
    assert sink.end.finish_reason is FinishReason.LENGTH


def test_prefill_never_shares_a_turn_with_decode():
    """One sequence decoding, another arriving: the arrival's chunks run in
    their own turns, and no turn does both."""
    loop, backend, _tokenizer = build_loop()
    first, second = Sink(), Sink()
    loop.submit(make_request((1, 2, 3), request_id="a", max_tokens=8), first)
    drain(loop, limit=2)
    loop.submit(make_request(tuple(range(3000)), request_id="b", max_tokens=2), second)

    for _ in range(12):
        before_prefill = len(backend.prefill_calls)
        before_verify = len(backend.verify_calls)
        loop.step()
        did_prefill = len(backend.prefill_calls) > before_prefill
        did_decode = len(backend.verify_calls) > before_verify
        assert not (did_prefill and did_decode)


def test_decode_batches_every_decoding_sequence_into_one_forward():
    loop, backend, _tokenizer = build_loop()
    sinks = [Sink() for _ in range(3)]
    for index, sink in enumerate(sinks):
        loop.submit(make_request((1, 2, 3), request_id=f"r{index}", max_tokens=4), sink)
    drain(loop)
    assert all(len(sink.tokens) == 4 for sink in sinks)
    # Three sequences, four tokens each, one verify per cycle: at most six
    # verify calls even counting the ragged tail.
    assert len(backend.verify_calls) <= 6
    assert loop.stats().decode_cycles <= 6


# ---------------------------------------------------------------------------
# usage and retirement
# ---------------------------------------------------------------------------


def test_usage_counts_prompt_cached_and_completion_tokens():
    config = AdmissionConfig()
    cache = FakeCache(matched=2048, config=config)
    loop, backend, _tokenizer = build_loop(cache=cache, config=config)
    sink = Sink()
    loop.submit(make_request(tuple(range(3000)), max_tokens=5), sink)
    drain(loop)
    end = sink.end
    assert end.prompt_tokens == 3000
    assert end.cached_tokens == 2048
    assert end.completion_tokens == 5
    assert len(sink.tokens) == 5
    assert end.finish_reason is FinishReason.LENGTH


def test_the_stream_ends_exactly_once_and_last():
    loop, _backend, _tokenizer = build_loop()
    sink = Sink()
    loop.submit(make_request((1, 2, 3), max_tokens=4), sink)
    drain(loop)
    ends = [e for e in sink.events if isinstance(e, StreamEnd)]
    assert len(ends) == 1
    assert isinstance(sink.events[-1], StreamEnd)


def test_a_finished_sequence_frees_its_state_and_offers_its_prefix():
    config = AdmissionConfig()
    cache = FakeCache(config=config)
    loop, backend, _tokenizer = build_loop(cache=cache, config=config)
    sink = Sink()
    loop.submit(make_request((1, 2, 3), max_tokens=4), sink)
    drain(loop)
    assert backend.open_handles == set()
    assert loop.live == ()
    covered, boundaries = cache.stores[0]
    assert covered == 6  # three prompt tokens plus four generated, less pending
    # Nothing staged a snapshot: a three-token prompt has no block end in it,
    # so the prefix is offered with no resumable point rather than with one the
    # store would have to drop.
    assert boundaries == ()


def test_the_offered_boundaries_are_the_ones_something_staged():
    """The plan's snapshot positions, and nothing invented on the way out. A
    boundary no snapshot sits at is one the store rounds down, fails to export
    and counts as a truncated chain."""
    config = AdmissionConfig()
    cache = FakeCache(config=config)
    loop, _backend, _tokenizer = build_loop(cache=cache, config=config)
    loop.submit(make_request(tuple(range(3000)), max_tokens=2), Sink())
    drain(loop)
    covered, boundaries = cache.stores[0]
    assert covered == 3001
    assert boundaries == (2048, 2560)


def test_a_refused_truncation_records_no_boundary():
    """D9 as a runtime rule: a lookup may never report a length it cannot
    restore, and no more than that.

    The stop string spans two tokens, so the trim reaches back into a block the
    verify already closed. A backend that refuses that truncation is right to --
    it is below the staged snapshot -- and what the loop owes the cache is a
    prefix with no resumable point, not silence. The blocks of the prefix that
    survives are still the KV for those same token ids.
    """
    from titan.core.errors import StateError

    stream = [20, 23, 24, 25]

    class RefusingBackend(FakeBackend):
        def truncate_state(self, state, length):
            raise StateError("below the last snapshot")

    def next_token(context, _vocab):
        index = len(context) - 3
        return stream[index] if 0 <= index < len(stream) else stream[-1]

    config = AdmissionConfig()
    cache = FakeCache(config=config)
    profiler = NullProfiler()
    loop, backend, _tokenizer = build_loop(
        cache=cache,
        config=config,
        backend=RefusingBackend(next_token=next_token),
        tokenizer=FakeTokenizer({20: "Hello", 23: "<|im_", 24: "end|>", 25: "!"}),
        profiler=profiler,
    )
    sink = Sink()
    loop.submit(
        make_request((1, 2, 3), max_tokens=10, stop_strings=("<|im_end|>",)), sink
    )
    drain(loop)

    assert "".join(e.text for e in sink.events if isinstance(e, TokenEvent)) == "Hello"
    assert sink.end.finish_reason is FinishReason.STOP
    # The refused truncation left the state longer than the sequence. What is
    # offered is trimmed to what the state actually backs, and no boundary is
    # recorded, so a later lookup cannot report a length it cannot restore.
    assert [boundaries for _covered, boundaries in cache.stores] == [()]
    names = [name for name, _fields in profiler.events]
    assert "truncate_refused" in names


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_cancelling_a_running_sequence_frees_its_state():
    loop, backend, tokenizer = build_loop()
    sink = Sink()
    request = make_request((1, 2, 3), max_tokens=100)
    loop.submit(request, sink)
    loop.step()
    assert backend.open_handles

    loop.cancel(request.request_id)
    loop.step()
    assert backend.open_handles == set()
    assert loop.live == ()
    assert sink.end.finish_reason is FinishReason.ABORT
    assert 1 in tokenizer.flushed  # the detokenisation stream is dropped too


def test_cancelling_a_waiting_request_never_allocates():
    loop, backend, _tokenizer = build_loop()
    sink = Sink()
    request = make_request((1, 2, 3))
    loop.submit(request, sink)
    loop.cancel(request.request_id)
    drain(loop)
    assert backend.open_handles == set()
    assert sink.end.finish_reason is FinishReason.ABORT
    assert loop.stats().admitted == 0


def test_shutdown_aborts_what_is_still_running():
    loop, backend, _tokenizer = build_loop()
    sink = Sink()
    loop.submit(make_request((1, 2, 3), max_tokens=1000), sink)
    for _ in range(3):
        loop.step()
    loop.shutdown(drain_timeout_s=0.0)
    assert backend.open_handles == set()
    assert sink.end.finish_reason is FinishReason.ABORT


# ---------------------------------------------------------------------------
# admission through the loop
# ---------------------------------------------------------------------------


def test_the_seat_count_holds_sequences_back_without_blocking_the_queue():
    config = AdmissionConfig(max_sequences=2)
    loop, _backend, _tokenizer = build_loop(config=config)
    sinks = [Sink() for _ in range(4)]
    for index, sink in enumerate(sinks):
        loop.submit(make_request((1, 2, 3), request_id=f"r{index}", max_tokens=2), sink)
    loop.step()
    assert len(loop.live) == 2
    drain(loop)
    assert all(len(sink.tokens) == 2 for sink in sinks)


def test_a_request_over_the_guard_is_refused_as_an_error_event():
    config = AdmissionConfig(memory_guard_gb=78.0, weights_gb=78.0)
    loop, _backend, _tokenizer = build_loop(config=config)
    sink = Sink()
    loop.submit(make_request((1, 2, 3)), sink)
    for _ in range(4):
        loop.step()
    # The guard refuses it at the policy gate, so it waits rather than erroring:
    # nothing running will ever free the weights, but nothing is lost either.
    assert loop.live == ()
    assert loop.stats().queue_depth == 1
    assert sink.events == []


def test_a_full_queue_answers_with_an_error_stream_end():
    config = AdmissionConfig(queue_depth=1)
    loop, _backend, _tokenizer = build_loop(config=config)
    first, second = Sink(), Sink()
    loop.submit(make_request((1, 2, 3), request_id="a"), first)
    loop.submit(make_request((1, 2, 3), request_id="b"), second)
    loop.step()
    assert second.end is not None
    assert second.end.finish_reason is FinishReason.ERROR
    assert "queue is full" in second.end.error


def test_stats_report_what_the_loop_did():
    loop, _backend, _tokenizer = build_loop()
    loop.submit(make_request(tuple(range(3000)), max_tokens=4), Sink())
    drain(loop)
    stats = loop.stats()
    assert stats.admitted == 1
    assert stats.prefill_chunks == 3
    assert stats.decode_cycles == 4
    assert stats.tokens_out == 4
    assert stats.mean_rows_per_cycle == 1.0
    assert stats.queue_depth == 0


# ---------------------------------------------------------------------------
# the prompt-end boundary
# ---------------------------------------------------------------------------


def test_the_first_decode_cycle_stages_the_end_of_the_prompt():
    """Prefill stops one token short of the prompt, so the boundary a follow-up
    turn would resume from does not exist when prefill ends. The cycle that
    consumes the last prompt token is the only place it can be staged, and it
    costs one copy of the recurrent state and no forward pass."""
    config = AdmissionConfig()
    cache = FakeCache(config=config)
    loop, backend, _tokenizer = build_loop(cache=cache, config=config)
    # 3072 is a block multiple, so the last prefill chunk stops at 2560 and the
    # prompt end is a block above the deepest thing prefill could stage.
    loop.submit(make_request(tuple(range(3072)), max_tokens=2), Sink())
    drain(loop)
    assert (1, 3072) in backend.staged
    covered, boundaries = cache.stores[0]
    assert boundaries[-1] == 3072


def test_a_prompt_whose_end_prefill_already_covered_stages_nothing_extra():
    """A second snapshot at the same rounded position buys nothing, and it is
    110 MiB of recurrent state to hold until the sequence retires."""
    config = AdmissionConfig()
    cache = FakeCache(config=config)
    loop, backend, _tokenizer = build_loop(cache=cache, config=config)
    loop.submit(make_request(tuple(range(3000)), max_tokens=2), Sink())
    drain(loop)
    assert backend.staged == []
    _covered, boundaries = cache.stores[0]
    assert boundaries[-1] == 2560


def test_the_cycle_that_ends_the_prompt_drafts_nothing_when_it_must_stage():
    """A verify block wider than one column lands the state past the prompt
    end, and the boundary is then unreachable without a forward pass. So the
    first cycle spends no drafts, once, and only when the boundary is owed."""
    from titan.engine.decode_cycle import MTPDecodeCycle

    from tests.engine.conftest import ScriptedDrafter
    from tests.engine.test_mtp_parity import perfect

    config = AdmissionConfig()
    cache = FakeCache(config=config)
    backend = FakeBackend()
    cycle = MTPDecodeCycle(
        backend=backend,
        tokenizer=FakeTokenizer(),
        drafter=ScriptedDrafter(perfect(backend)),
        max_depth=3,
    )
    loop, _backend, _tokenizer = build_loop(
        cache=cache, config=config, backend=backend, cycle=cycle
    )
    loop.submit(make_request(tuple(range(3072)), max_tokens=8), Sink())
    drain(loop)
    assert backend.verify_calls[0] == 1
    assert max(backend.verify_calls) > 1
    assert (1, 3072) in backend.staged


def test_a_short_prompt_has_no_prompt_end_boundary_to_stage():
    config = AdmissionConfig()
    loop, backend, _tokenizer = build_loop(config=config)
    loop.submit(make_request((1, 2, 3), max_tokens=3), Sink())
    drain(loop)
    assert backend.staged == []


def test_an_unaligned_prompt_end_is_left_to_the_prefill_plan():
    """A restore point has to be a block end, so the store rounds every
    boundary down to the grid. Staging one at an unaligned prompt end produces
    a snapshot the store looks for at the rounded position, does not find, and
    drops: work, a 110 MiB copy, and a counted chain truncation for nothing."""
    config = AdmissionConfig()
    cache = FakeCache(config=config)
    loop, backend, _tokenizer = build_loop(cache=cache, config=config)
    loop.submit(make_request(tuple(range(3001)), max_tokens=2), Sink())
    drain(loop)
    assert backend.staged == []


def test_every_prefill_chunk_is_handed_the_token_that_follows_it():
    """A draft head folds ``(hidden[t], token[t+1])``, so the pair at a chunk's
    last position is the only one the chunk cannot form from its own tokens.

    The token is always in range: prefill plans stop one short of the prompt,
    which is the decode invariant, so no chunk is ever the end of the list.
    """
    loop, backend, _tokenizer = build_loop()
    prompt = tuple(range(5000))
    loop.submit(make_request(prompt, max_tokens=2), Sink())
    for _ in range(4):
        loop.step()

    ends = [2048, 4096, 4608, 4999]
    assert backend.prefill_next_tokens == [prompt[e] for e in ends]
    assert None not in backend.prefill_next_tokens
