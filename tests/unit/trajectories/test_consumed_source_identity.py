from __future__ import annotations

import copy
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange, _completion_exchange, _message_exchange

import art.trajectories as tr
import art.trajectories._tokenize as core


def test_complete_native_messages_accept_opaque_unused_metadata():
    exchange = _message_exchange(
        cast(
            Any,
            {
                "model": "test/model",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
        prompt_token_ids=[1],
        token_ids=[2],
        logprobs=[-0.2],
    )
    opaque = object()
    exchange.request["metadata"] = cast(Any, {"opaque": opaque})
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    actual = value.tokenize()
    assert actual.tokens == [1, 2]
    assert actual.logprobs[-1] == -0.2
    assert actual.flags[-1] & tr.TokenFlag.SAMPLED
    assert exchange.request["metadata"]["opaque"] is opaque


@pytest.mark.parametrize("mutate", [False, True])
def test_consumed_equal_key_distinct_source_objects_remain_guarded(mutate):
    first = _chat_exchange([1], [2])
    second = copy.deepcopy(first)
    key = core._exchange_sampled_source_key(first)
    assert key == core._exchange_sampled_source_key(second)
    assert second is not first
    builder = core._TraceBuilder()
    builder.consume_sources({key: first})
    builder.consume_sources({key: second})
    if mutate:
        assert second.response.choices[0].logprobs is not None
        assert second.response.choices[0].logprobs.content
        second.response.choices[0].logprobs.content[0].logprob = -9.0
    assert builder.validate_sources is not None
    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            builder.validate_sources(None)
    else:
        builder.validate_sources(None)


def test_complete_native_completions_accept_opaque_unused_metadata():
    exchange = _completion_exchange()
    opaque = object()
    exchange.request["metadata"] = cast(Any, {"opaque": opaque})
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(completions=[exchange]))
    actual = value.tokenize()
    assert actual.tokens == [1, 2]
    assert actual.logprobs[-1] == -0.2
    assert actual.flags[-1] & tr.TokenFlag.SAMPLED
    assert exchange.request["metadata"]["opaque"] is opaque


def test_opaque_exchange_still_refuses_before_prompt_callback():
    exchange = _message_exchange(
        cast(
            Any,
            {
                "model": "test/model",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
        token_ids=[2],
        logprobs=[-0.2],
    )
    exchange.request["metadata"] = cast(Any, {"opaque": object()})
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(True)
            return [1]

    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    with pytest.raises(ValueError, match="context cannot be checked"):
        value.tokenize(tokenizer=cast(Any, Tokenizer()))
    assert not calls


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("field", ["logprob", "request"])
def test_aliases_in_one_consumption_keep_each_original(selected, which, field):
    first = _chat_exchange([1], [2])
    second = copy.deepcopy(first)
    key = core._exchange_sampled_source_key(first)
    builder = core._TraceBuilder()
    builder.consume_sources([(key, first), (key, second)])
    source = (first, second)[which]
    if field == "request":
        source.request["messages"][0]["content"] = "edited"
    else:
        assert source.response.choices[0].logprobs is not None
        assert source.response.choices[0].logprobs.content
        source.response.choices[0].logprobs.content[0].logprob = -9.0
    assert builder.validate_sources is not None
    with pytest.raises(
        ValueError, match="[Ss]ampled source changed|[Cc]ontext changed"
    ):
        builder.validate_sources(key if selected else None)


def test_alias_extension_does_not_bless_earlier_mutation():
    first = _chat_exchange([1], [2])
    second = copy.deepcopy(first)
    key = core._exchange_sampled_source_key(first)
    builder = core._TraceBuilder()
    builder.consume_sources({key: first})
    assert first.response.choices[0].logprobs is not None
    assert first.response.choices[0].logprobs.content
    first.response.choices[0].logprobs.content[0].logprob = -9.0
    with pytest.raises(ValueError, match="[Ss]ampled source changed"):
        builder.consume_sources({key: second})


def test_opaque_allowance_never_disables_native_evidence_validation():
    exchange = _chat_exchange([1], [2])
    exchange.request["metadata"] = cast(Any, {"opaque": object()})
    key = core._exchange_sampled_source_key(exchange)
    validate = core._sampled_source_validator({key: exchange})
    validate(None, require_supported_context=False)
    assert exchange.response.choices[0].logprobs is not None
    assert exchange.response.choices[0].logprobs.content
    exchange.response.choices[0].logprobs.content[0].logprob = -9.0
    with pytest.raises(ValueError, match="[Ss]ampled source changed"):
        validate(None, require_supported_context=False)


def test_callback_free_multiple_histories_keep_opaque_metadata():
    first = _completion_exchange()
    second = copy.deepcopy(first)
    second.request["model"] = "other/model"
    opaque = object()
    for exchange in (first, second):
        exchange.request["metadata"] = cast(Any, {"opaque": opaque})
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(completions=[first, second]))
    actual = value.tokenize(multi_history=True)
    assert len(actual.histories) == 2
    assert [history.tokens for history in actual.histories] == [[1, 2], [1, 2]]
    assert [history.logprobs[-1] for history in actual.histories] == [-0.2, -0.2]
    assert all(history.flags[-1] & tr.TokenFlag.SAMPLED for history in actual.histories)
    assert first.request["metadata"]["opaque"] is opaque
    assert second.request["metadata"]["opaque"] is opaque


def test_supported_context_is_checked_without_opaque_requirement():
    source = _chat_exchange([1], [2])
    key = core._exchange_sampled_source_key(source)
    validate = core._sampled_source_validator({key: source})
    source.request["messages"][0]["content"] = "edited"
    with pytest.raises(ValueError, match="[Cc]ontext changed"):
        validate(None, require_supported_context=False)
