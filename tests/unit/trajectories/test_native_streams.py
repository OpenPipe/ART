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


@pytest.fixture(autouse=True)
def isolate_prefix_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_WARNED_PREFIX_RETOKENIZATION", False)


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


def test_native_stream_preserves_interior_request_assistant_roles() -> None:
    trajectory, tokenizer = example()
    third = trajectory.exchanges.chat_completions[-1]
    third.request["messages"][-1:-1] = [
        {"role": "user", "content": "intermediate"},
        {"role": "assistant", "content": "request-only"},
    ]
    prefix = "turn 0answer§turn 1answer§intermediate"
    record(third).prompt_token_ids = tokenizer._encode(prefix + "request-only§turn 2")
    before = trajectory.model_dump_json()
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    final = result.histories[-1]
    assert final.tokens == [*record(third).prompt_token_ids, *record(third).token_ids]
    assert result_terms(result) == native_terms(trajectory)
    assert trajectory.model_dump_json() == before
    context = slice(len(prefix), len(prefix + "request-only§"))
    assert all(math.isnan(value) for value in final.logprobs[context])
    assert final.flags[context] == [
        *([tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT] * len("request-only")),
        tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.STOP,
    ]


def test_existing_native_shortcut_precedes_stream_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer = example()
    record(trajectory.exchanges.chat_completions[0]).finish_reason = "stop"
    render = tokenizer.apply_chat_template

    def changed_renderer(messages: Any, **kwargs: Any) -> Any:
        copied = deepcopy(messages)
        for message in copied:
            if message.get("role") == "assistant":
                message["content"] = "different-renderer:" + (
                    message.get("content") or ""
                )
        return render(copied, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", changed_renderer)
    before = trajectory.model_dump_json()
    with monkeypatch.context() as old_route:
        old_route.setattr(module, "_native_history_streams", lambda history: [history])
        existing = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert result.model_dump_json() == existing.model_dump_json()
    assert result_terms(result) == native_terms(trajectory)
    assert trajectory.model_dump_json() == before


def test_existing_late_native_role_proof_precedes_stream_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTemplateTokenizer()
    messages = [
        {"role": "user", "content": "intro"},
        {"role": "assistant", "content": "history"},
        {"role": "user", "content": "turn0"},
    ]
    prompt = "introhistory§turn0"
    first = _chat_exchange(tokenizer._encode(prompt), tokenizer._encode("rawanswer§"))
    first.request["messages"] = deepcopy(messages)
    record(first).message.content = "answer"
    second = _chat_exchange(
        tokenizer._encode(prompt + "answer§turn1"),
        tokenizer._encode("terminal§"),
        offset=1,
    )
    second.request["messages"] = [
        *deepcopy(messages),
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "turn1"},
    ]
    record(second).message.content = "terminal"
    for exchange in (first, second):
        record(exchange).finish_reason = "stop"
        record(exchange).model_extra["stop_reason"] = tokenizer.eos_token_id
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second]), reward=2.0
    )
    render = tokenizer.apply_chat_template

    def changed_renderer(selected: Any, **kwargs: Any) -> Any:
        copied = deepcopy(selected)
        for message in copied:
            if (
                message.get("role") == "assistant"
                and message.get("content") == "answer"
            ):
                message["content"] = "!answer"
        return render(copied, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", changed_renderer)
    before = trajectory.model_dump_json()
    with monkeypatch.context() as old_route:
        old_route.setattr(module, "_native_history_streams", lambda history: [history])
        existing = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert result.model_dump_json() == existing.model_dump_json()
    assert result_terms(result) == native_terms(trajectory)
    assert trajectory.model_dump_json() == before


@pytest.mark.parametrize("middle_kind", ["tool", "reasoning"])
def test_native_stream_keeps_complete_nonterminal_structured_output(
    middle_kind: str,
) -> None:
    trajectory, tokenizer = example()
    _, second, third = trajectory.exchanges.chat_completions
    middle = _chat_exchange(
        list(record(second).prompt_token_ids),
        tokenizer._encode("different native middle body§"),
        offset=1,
    )
    data = middle.response.model_dump(mode="python")
    message = data["choices"][0]["message"]
    if middle_kind == "tool":
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "middle-tool",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key": 1}'},
            }
        ]
    else:
        message["reasoning_content"] = "recorded structured reasoning"
    middle.response = ChatCompletion.model_validate(data)
    trajectory.exchanges.chat_completions[1] = middle
    third.request["messages"][3] = record(middle).message.model_dump(
        mode="python", exclude_none=True
    )
    record(third).prompt_token_ids = [
        *record(middle).prompt_token_ids,
        *record(middle).token_ids,
        *tokenizer._encode("turn 2"),
    ]
    before = trajectory.model_dump_json()
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert result.histories[-1].tokens == [
        *record(third).prompt_token_ids,
        *record(third).token_ids,
    ]
    assert result_terms(result) == native_terms(trajectory)
    assert trajectory.model_dump_json() == before


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
    "change",
    ["missing_source", "prompt", "lp", "stop", "extra_stop", "flags", "extra_sample"],
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
    elif change == "extra_sample":
        value.flags[0] |= tr.TokenFlag.SAMPLED
        value.logprobs[0] = -0.5
        trace.source_keys[0] = trace.source_keys[sampled[0]]
        trace.validate(value)  # Coherent trace membership is not complete coverage.
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
    with pytest.raises(
        ValueError, match="Sampled source changed during tokenization callback"
    ):
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
@pytest.mark.parametrize(
    "change",
    [
        "request",
        "logprob",
        "stop_reason",
        "unscoped_logprob",
        "unscoped_request_role",
        "unscoped_request_tools",
        "unscoped_history_role",
        "expanded_original_role",
    ],
)
def test_later_stop_encoder_cannot_change_an_already_certified_stream(
    monkeypatch: pytest.MonkeyPatch, private: bool, change: str
) -> None:
    trajectory, tokenizer = example()
    first, second, third = trajectory.exchanges.chat_completions
    record(first).finish_reason = "stop"
    record(first).model_extra["stop_reason"] = "§"
    record(second).model_extra["stop_reason"] = "§"
    earlier = first
    if change.startswith("unscoped_"):
        earlier = _chat_exchange(
            tokenizer._encode("separatehistorical§again"),
            tokenizer._encode("answer§"),
            offset=-1,
        )
        earlier.request["messages"] = [
            {"role": "user", "content": "separate"},
            {"role": "assistant", "content": "historical"},
            {"role": "user", "content": "again"},
        ]
        trajectory.exchanges.chat_completions.insert(0, earlier)
    # Request-owned roles make the old sampled-only boundary shortcut decline,
    # so the later stream's validation callback is actually reached.
    for exchange in (second, third):
        exchange.request["messages"][2:2] = [
            {"role": "user", "content": "bridge"},
            {"role": "assistant", "content": "request-only"},
        ]
        record(exchange).prompt_token_ids[
            len("turn 0answer§") : len("turn 0answer§")
        ] = tokenizer._encode("bridgerequest-only§")
    original_guard = module._require_native_stream
    original_encode = tokenizer.__class__.__call__
    original_history = module.tokenize_history
    original_planner = module._native_history_streams
    completed = []
    expanded = []
    armed = False
    changed = False

    def plan_streams(history: Any) -> Any:
        if any(
            source is not None and source.exchange is second
            for source in history.message_sources
        ):
            expanded.append(history)
        return original_planner(history)

    def history_call(history: Any, *args: Any, **kwargs: Any) -> Any:
        value = original_history(history, *args, **kwargs)
        if change.startswith("unscoped_") and any(
            source is not None and source.exchange is earlier
            for source in history.message_sources
        ):
            assert value.flags[len("separate")] & tr.TokenFlag.ASSISTANT
            assert not value.flags[len("separate")] & tr.TokenFlag.SAMPLED
            completed.append(value)
        return value

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
            elif change in {"logprob", "unscoped_logprob"}:
                record(earlier).logprobs.content[0].logprob = -123.0
            elif change == "unscoped_request_role":
                earlier.request["messages"][1]["role"] = "user"
            elif change == "unscoped_request_tools":
                earlier.request["tools"] = [
                    {"type": "function", "function": {"name": "changed"}}
                ]
            elif change == "unscoped_history_role":
                assert len(completed) == 1
                completed[0].history.messages[1]["role"] = "user"
            elif change == "expanded_original_role":
                assert len(expanded) == 1
                expanded[0].messages[0]["role"] = "assistant"
            else:
                record(first).model_extra["stop_reason"] = "!"
        return original_encode(self, text, **kwargs)

    monkeypatch.setattr(module, "_require_native_stream", guard)
    monkeypatch.setattr(module, "tokenize_history", history_call)
    monkeypatch.setattr(module, "_native_history_streams", plan_streams)
    monkeypatch.setattr(tokenizer.__class__, "__call__", encode)
    if change == "unscoped_logprob":
        message = "Sampled source changed during tokenization callback"
    elif change.startswith("unscoped_") or change == "expanded_original_role":
        message = "Tokenization context changed during tokenization callback"
    else:
        message = "during final STOP validation"
    with pytest.raises(ValueError, match=message):
        if private:
            module._tokenize_trajectory_with_trace(trajectory, tokenizer=tokenizer)
        else:
            trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
        assert changed
    assert changed
