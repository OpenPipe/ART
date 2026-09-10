from __future__ import annotations

import pytest
from test_tokenize import (
    _chat_exchange,
    _completion_exchange,
    _message_exchange,
    _response_exchange,
)

import art
from art.trajectories import MessagesRequest, TrajectoryExchanges
from art.trajectories import TokenFlag as F
from art.trajectories import _tokenize as native


def captured(protocol):
    if protocol == "chat":
        return _chat_exchange([1], [2])
    if protocol == "completions":
        return _completion_exchange(prompt=[1])
    if protocol == "messages":
        return _message_exchange(
            MessagesRequest(
                model="test/model",
                max_tokens=16,
                messages=[{"role": "user", "content": "question"}],
            ),
            prompt_token_ids=[1],
            token_ids=[2],
            logprobs=[-0.2],
        )
    return _response_exchange("response-0", 2, prompt_token_ids=[1])


def change_prompt(exchange, protocol, prompt):
    if protocol in {"chat", "completions"}:
        exchange.response.choices[0].model_extra["prompt_token_ids"] = prompt
    elif protocol == "messages":
        exchange.response.model_extra["prompt_token_ids"] = prompt
    else:
        exchange.response.model_extra["token_generations"][0]["prompt_token_ids"] = (
            prompt
        )


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "completions"])
def test_same_response_and_output_binds_distinct_captured_prompt(protocol):
    original = captured(protocol)
    changed = original.model_copy(deep=True)
    change_prompt(changed, protocol, [9])
    assert original.response.id == changed.response.id
    assert (original.start_time, original.end_time) == (
        changed.start_time,
        changed.end_time,
    )
    assert native._exchange_sampled_source_key(
        original
    ) != native._exchange_sampled_source_key(changed)


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "completions"])
def test_identical_capture_roundtrip_retains_source_identity(protocol):
    original = captured(protocol)
    restored = type(original).model_validate_json(original.model_dump_json())
    assert native._exchange_sampled_source_key(
        original
    ) == native._exchange_sampled_source_key(restored)


@pytest.mark.parametrize("protocol", ["chat", "completions"])
@pytest.mark.parametrize("choice_prompt", ["missing", "null"])
def test_choice_without_prompt_binds_authoritative_response_fallback(
    protocol, choice_prompt
):
    original = captured(protocol)
    extra = original.response.choices[0].model_extra
    if choice_prompt == "missing":
        extra.pop("prompt_token_ids")
    else:
        extra["prompt_token_ids"] = None
    original.response.model_extra["prompt_token_ids"] = [1]
    changed = original.model_copy(deep=True)
    changed.response.model_extra["prompt_token_ids"] = [9]
    assert native._exchange_sampled_source_key(
        original
    ) != native._exchange_sampled_source_key(changed)


@pytest.mark.parametrize("protocol", ["chat", "completions"])
def test_selected_choice_ignores_unrelated_choices_and_shadowed_fallback(protocol):
    original = captured(protocol)
    original.response.choices[0].index = 7
    original.response.model_extra["prompt_token_ids"] = [900]
    key = native._exchange_sampled_source_key(original)
    changed = original.model_copy(deep=True)
    unrelated = changed.response.choices[0].model_copy(deep=True)
    unrelated.index = 3
    unrelated.model_extra["prompt_token_ids"] = [800]
    changed.response.choices.insert(0, unrelated)
    changed.response.model_extra["prompt_token_ids"] = [901]
    assert (
        native._source_key(
            changed, protocol=key.protocol, index=7, prompt_index=key.prompt_index
        )
        == key
    )
    unrelated.model_extra["prompt_token_ids"] = [801]
    assert (
        native._source_key(
            changed, protocol=key.protocol, index=7, prompt_index=key.prompt_index
        )
        == key
    )


@pytest.mark.parametrize("multi_history", [False, True])
def test_reused_response_identity_preserves_both_sampled_causal_prefixes(multi_history):
    first = _chat_exchange([1], [2])
    second = _chat_exchange([1, 2, 3], [2], offset=1)
    second.response.id = first.response.id
    second.response.created = first.response.created
    second.start_time = first.start_time
    second.end_time = first.end_time
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )
    result = trajectory.tokenize(multi_history=multi_history)
    histories = (
        result.histories
        if isinstance(result, art.trajectories.TokenizedMultiHistoryTrajectory)
        else [result]
    )
    selected = {
        (tuple(history.tokens[: index + 1]), history.logprobs[index])
        for history in histories
        for index, flag in enumerate(history.flags)
        if flag & F.SAMPLED
    }
    assert selected == {((1, 2), -0.2), ((1, 2, 3, 2), -0.2)}
    assert (
        sum(
            sum(mask)
            for mask in art.trajectories.first_occurrence_masks(
                histories, where=F.SAMPLED
            )
        )
        == 2
    )
    restored = art.Trajectory.model_validate_json(
        trajectory.model_dump_json()
    ).tokenize(multi_history=multi_history)
    restored_histories = (
        restored.histories
        if isinstance(restored, art.trajectories.TokenizedMultiHistoryTrajectory)
        else [restored]
    )
    assert [(h.tokens, h.flags) for h in restored_histories] == [
        (h.tokens, h.flags) for h in histories
    ]
