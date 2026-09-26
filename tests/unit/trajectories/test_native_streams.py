from __future__ import annotations

from copy import deepcopy
import math
import pickle
import struct
from typing import Any, cast

from openai.types.chat import ChatCompletion
import pytest
from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def record(exchange: tr.ChatCompletionsExchange) -> Any:
    return cast(Any, exchange.response.choices[0])


def example() -> tuple[tr.Trajectory, Any]:
    tokenizer = _CharacterTemplateTokenizer()
    first = _chat_exchange(tokenizer._encode("turn 0"), tokenizer._encode("rawanswer§"))
    record(first).finish_reason = "length"
    second = _chat_exchange(
        tokenizer._encode("turn 0answer§turn 1"), tokenizer._encode("answer§"), offset=1
    )
    third = _chat_exchange(
        tokenizer._encode("turn 0answer§turn 1answer§turn 2"),
        tokenizer._encode("raw terminal tool output"),
        offset=2,
    )
    payload = third.response.model_dump(mode="python")
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "public",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    third.response = ChatCompletion.model_validate(payload)
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second, third]),
        reward=3.0,
    )
    return trajectory, tokenizer


def test_nonnested_streams_cover_native_sources_and_roles() -> None:
    trajectory, tokenizer = example()
    before = trajectory.model_dump_json()
    histories = trajectory.histories()
    assert len(histories) == 2
    result, traces = module._tokenize_trajectory_with_trace(
        trajectory, tokenizer=tokenizer
    )
    assert result.trajectory is trajectory
    assert result.reward == trajectory.reward
    assert len(result.histories) == 3
    sources = trajectory.exchanges.chat_completions
    expected = [
        record(sources[0]),
        record(sources[0]),
        record(sources[-1]),
    ]
    for history, choice in zip(result.histories, expected, strict=True):
        assert history.tokens == [*choice.prompt_token_ids, *choice.token_ids]
    final = result.histories[-1]
    assert final.flags[len("turn 0")] & tr.TokenFlag.ASSISTANT
    assert not final.flags[len("turn 0")] & (tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT)
    assert final.flags[len("turn 0answer")] & tr.TokenFlag.STOP
    for exchange in sources:
        choice = record(exchange)
        matches = [
            (value, trace)
            for value, trace in zip(result.histories, traces, strict=True)
            if any(
                getattr(source, "exchange", None) is exchange
                for source in trace.sources.values()
            )
        ]
        assert matches
        for value, trace in matches:
            source = next(
                source
                for source in trace.sources.values()
                if getattr(source, "exchange", None) is exchange
            )
            assert module._complete_source_is_represented(
                source,
                choice.prompt_token_ids,
                choice.token_ids,
                [entry.logprob for entry in choice.logprobs.content],
                [(value, trace)],
            )
    assert trajectory.model_dump_json() == before


def test_old_single_history_refusal_is_retained() -> None:
    trajectory, tokenizer = example()
    history = trajectory.histories()[-1]
    assert isinstance(history, tr.ChatCompletionsHistory)
    with pytest.raises(ValueError):
        history.tokenize(tokenizer=tokenizer)


def test_explicit_rendering_does_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory, tokenizer = example()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("explicit rendering cannot split native streams")

    monkeypatch.setattr(module, "_native_history_streams", forbidden)
    with pytest.raises(ValueError):
        trajectory.tokenize(
            tokenizer=tokenizer, multi_history=True, chat_template="override"
        )


def test_public_regression_requires_new_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer = example()
    monkeypatch.setattr(module, "_native_history_streams", lambda history: [history])
    with pytest.raises(ValueError, match="sampled content boundary"):
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)


def finite32(value: float) -> bool:
    try:
        return math.isfinite(struct.unpack("!f", struct.pack("!f", value))[0])
    except OverflowError:
        return False


def native_terms(trajectory: tr.Trajectory) -> list[tuple[Any, float]]:
    """Independent prefix-edge oracle: claim BEFORE finite filtering."""
    seen = set()
    terms = []
    for history in trajectory.histories():
        assert isinstance(history, tr.ChatCompletionsHistory)
        for source in history.message_sources:
            if source is None or source.choice_index is None:
                continue
            assert isinstance(source.exchange, tr.ChatCompletionsExchange)
            choice: Any = next(
                c
                for c in source.exchange.response.choices
                if c.index == source.choice_index
            )
            for index, entry in enumerate(choice.logprobs.content):
                key = (
                    history.model,
                    tuple([*choice.prompt_token_ids, *choice.token_ids[: index + 1]]),
                )
                if key not in seen:
                    seen.add(key)
                    if finite32(entry.logprob):
                        terms.append((key, entry.logprob))
    return terms


def result_terms(result: Any) -> list[tuple[Any, float]]:
    terms = []
    for history, mask in zip(
        result.histories,
        tr.first_occurrence_masks(result.histories, where=tr.TokenFlag.SAMPLED),
        strict=True,
    ):
        for index, (selected, logprob) in enumerate(
            zip(mask, history.logprobs, strict=True)
        ):
            if selected and finite32(logprob):
                terms.append(
                    ((history.model, tuple(history.tokens[: index + 1])), logprob)
                )
    return terms


@pytest.mark.parametrize("first_logprob", [-0.4, math.nan, 1e100])
def test_ordered_native_objective_and_compact_roundtrip(first_logprob: float) -> None:
    trajectory, tokenizer = example()
    first = trajectory.exchanges.chat_completions[0]
    record(first).logprobs.content[0].logprob = first_logprob
    later = _chat_exchange(
        list(record(first).prompt_token_ids),
        list(record(first).token_ids),
        offset=4,
    )
    later.request["messages"] = [{"role": "user", "content": "other branch"}]
    record(later).finish_reason = "length"
    record(later).logprobs.content[0].logprob = -0.1
    other_model = _chat_exchange([1], [2], model="other/model", offset=5)
    other_model.request["messages"] = [{"role": "user", "content": "other model"}]
    other_model.response.choices[0].finish_reason = "length"
    trajectory.exchanges.chat_completions.extend([later, other_model])
    before = trajectory.model_dump_json()
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert result_terms(result) == native_terms(trajectory)
    for restored in [
        pickle.loads(pickle.dumps(result)),
        tr.compact_validate(
            tr.compact_dump(result), type=tr.TokenizedMultiHistoryTrajectory
        ),
    ]:
        assert result_terms(restored) == native_terms(trajectory)
        assert restored.model_dump_json() == result.model_dump_json()
    assert trajectory.model_dump_json() == before


@pytest.mark.parametrize(
    "change",
    [
        "missing_ids",
        "missing_lp",
        "edited",
        "context",
        "unsupported_finish",
        "reordered",
    ],
)
def test_incomplete_or_edited_history_does_not_authorize_splitting(change: str) -> None:
    trajectory, _ = example()
    history = trajectory.histories()[-1]
    assert isinstance(history, tr.ChatCompletionsHistory)
    if change == "missing_ids":
        record(trajectory.exchanges.chat_completions[1]).model_extra.pop(
            "prompt_token_ids"
        )
    elif change == "missing_lp":
        trajectory.exchanges.chat_completions[1].response.choices[0].logprobs = None
    elif change == "edited":
        history.messages[-1]["content"] = "edited response"
        history.message_sources[-1] = None
    elif change == "context":
        history.chat_template_kwargs = {"enable_thinking": False}
    elif change == "unsupported_finish":
        trajectory.exchanges.chat_completions[1].response.choices[
            0
        ].finish_reason = "content_filter"
    else:
        history.messages[1], history.messages[3] = (
            history.messages[3],
            history.messages[1],
        )
        history.message_sources[1], history.message_sources[3] = (
            history.message_sources[3],
            history.message_sources[1],
        )
    assert module._native_history_streams(history) == [history]


@pytest.mark.parametrize(
    "change", ["missing_source", "prompt", "lp", "stop", "extra_stop", "flags"]
)
def test_scoped_final_guard_rejects_corrupt_results(change: str) -> None:
    trajectory, tokenizer = example()
    result, traces = module._tokenize_trajectory_with_trace(
        trajectory, tokenizer=tokenizer
    )
    value, trace = result.histories[-1], traces[-1]
    builder = module._TraceBuilder(trace=trace, tokenizer=tokenizer)
    sampled = [i for i, flag in enumerate(value.flags) if flag & tr.TokenFlag.SAMPLED]
    if change == "missing_source":
        trace.sources.pop(next(iter(trace.sources)))
    elif change == "prompt":
        value.tokens[0] += 1
    elif change == "lp":
        value.logprobs[sampled[0]] += 1
    elif change == "stop":
        stop = next(i for i in sampled if value.flags[i] & tr.TokenFlag.STOP)
        value.flags[stop] &= ~tr.TokenFlag.STOP
    elif change == "extra_stop":
        value.flags[sampled[-1]] |= tr.TokenFlag.STOP
    else:
        value.flags[sampled[0]] &= ~tr.TokenFlag.SAMPLED
    with pytest.raises((ValueError, AssertionError)):
        module._require_native_stream(value.history, value, builder)


def test_foreign_planning_exception_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer = example()
    failure = module._NativeHistoryStreams(trajectory.histories())

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(tokenizer, "apply_chat_template", fail)
    with pytest.raises(module._NativeHistoryStreams) as caught:
        trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert caught.value is failure


def test_callback_mutation_of_earlier_source_invalidates_scoped_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer = example()
    original = tokenizer.apply_chat_template
    count = 0

    def mutate(messages: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if len(messages) > 3:
            record(trajectory.exchanges.chat_completions[0]).logprobs.content[
                0
            ].logprob = -99.0
        return original(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", mutate)
    with pytest.raises(ValueError, match="inventory changed"):
        trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert count


def test_stop_encoder_cannot_change_already_checked_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer = example()
    second = trajectory.exchanges.chat_completions[1]
    record(second).model_extra["stop_reason"] = "§"
    result, traces = module._tokenize_trajectory_with_trace(
        trajectory, tokenizer=tokenizer
    )
    value, trace = result.histories[-1], traces[-1]
    original = tokenizer.__class__.__call__

    def mutate(self: Any, text: str, **kwargs: Any) -> Any:
        if text == "§":
            record(second).logprobs.content[0].logprob = -123.0
        return original(self, text, **kwargs)

    monkeypatch.setattr(tokenizer.__class__, "__call__", mutate)
    with pytest.raises(ValueError, match="changed while proving STOP"):
        module._require_native_stream(
            value.history, value, module._TraceBuilder(trace=trace, tokenizer=tokenizer)
        )


@pytest.mark.parametrize(
    "change", ["request", "request_order", "scoped_context", "source_identity"]
)
def test_stop_encoder_cannot_change_role_proof_context(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    trajectory, tokenizer = example()
    second = trajectory.exchanges.chat_completions[1]
    record(second).model_extra["stop_reason"] = "§"
    result, traces = module._tokenize_trajectory_with_trace(
        trajectory, tokenizer=tokenizer
    )
    value, trace = result.histories[-1], traces[-1]
    history = value.history
    assert isinstance(history, tr.ChatCompletionsHistory)
    original = tokenizer.__class__.__call__

    def mutate(self: Any, text: str, **kwargs: Any) -> Any:
        if text == "§":
            if change == "request":
                second.request["chat_template_kwargs"] = {"changed": True}
            elif change == "request_order":
                cast(Any, second.request["messages"])[0] = dict(
                    reversed(list(second.request["messages"][0].items()))
                )
            elif change == "scoped_context":
                history.chat_template_kwargs = {"changed": True}
            else:
                source = history.message_sources[3]
                assert source is not None
                history.message_sources[3] = source.model_copy(
                    update={"exchange": second.model_copy(deep=True)}
                )
        return original(self, text, **kwargs)

    monkeypatch.setattr(tokenizer.__class__, "__call__", mutate)
    with pytest.raises(ValueError, match="context changed while proving STOP"):
        module._require_native_stream(
            history, value, module._TraceBuilder(trace=trace, tokenizer=tokenizer)
        )


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("change", ["request", "logprob", "stop_reason"])
def test_later_stop_encoder_cannot_change_an_already_certified_stream(
    monkeypatch: pytest.MonkeyPatch, private: bool, change: str
) -> None:
    trajectory, tokenizer = example()
    first, second, _ = trajectory.exchanges.chat_completions
    record(first).finish_reason = "stop"
    record(first).model_extra["stop_reason"] = "§"
    record(second).model_extra["stop_reason"] = "§"
    original_guard = module._require_native_stream
    original_encode = tokenizer.__class__.__call__
    armed = False
    changed = False

    def guard(history: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal armed
        # Final scope validation supplies planning snapshots; earlier assembly
        # checks do not. Arm only the later stream's actual STOP encoder.
        armed = len(args) >= 4 and any(
            source is not None and source.exchange is second
            for source in history.message_sources
        )
        try:
            original_guard(history, *args, **kwargs)
        finally:
            armed = False

    def encode(self: Any, text: str, **kwargs: Any) -> Any:
        nonlocal changed
        if armed and text == "§":
            changed = True
            if change == "request":
                first.request["chat_template_kwargs"] = {"changed": True}
            elif change == "logprob":
                record(first).logprobs.content[0].logprob = -123.0
            else:
                record(first).model_extra["stop_reason"] = "!"
        return original_encode(self, text, **kwargs)

    monkeypatch.setattr(module, "_require_native_stream", guard)
    monkeypatch.setattr(tokenizer.__class__, "__call__", encode)
    with pytest.raises(ValueError, match="during final STOP validation"):
        if private:
            module._tokenize_trajectory_with_trace(trajectory, tokenizer=tokenizer)
        else:
            trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert changed
