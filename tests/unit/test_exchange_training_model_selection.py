from datetime import datetime, timedelta
from typing import cast

import numpy as np
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import pytest
from transformers import PreTrainedTokenizerBase

import art
from art.openai import ART_MOE_ROUTING_METADATA_KEY
from art.preprocessing.tokenize import _chat_choice_trace, tokenize_trajectory_groups
from art.tinker_native.data import trajectory_groups_to_datums
from art.trajectories import (
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    ChatCompletionsMessageSource,
    ChatCompletionsRequest,
)
from art.trajectories._tokenize import _training_model_pattern


def _exchange(model: str, output_token: int) -> ChatCompletionsExchange:
    response = ChatCompletion.model_validate(
        {
            "id": f"chatcmpl-{output_token}",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": [1],
                    "token_ids": [output_token],
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{output_token}",
                                "logprob": -0.1,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
        }
    )
    start = datetime(2026, 1, 1)
    return ChatCompletionsExchange(
        request=ChatCompletionsRequest(
            model=model,
            messages=[{"role": "user", "content": "question"}],
        ),
        response=response,
        start_time=start,
        end_time=start + timedelta(seconds=1),
    )


def _routed_exchange(
    *,
    prompt_token_ids: list[int],
    output_token: int,
    messages: list[ChatCompletionMessageParam],
    content: str,
) -> ChatCompletionsExchange:
    exchange = _exchange("policy", output_token)
    exchange.request["messages"] = messages
    choice = exchange.response.choices[0]
    choice.message.content = content
    extra = choice.model_extra
    assert extra is not None
    extra["prompt_token_ids"] = prompt_token_ids
    extra[ART_MOE_ROUTING_METADATA_KEY] = {
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": [output_token],
        "routed_experts": np.asarray(
            [[[10]]] * len(prompt_token_ids) + [[[output_token * 10]]],
            dtype=np.int32,
        ),
    }
    return exchange


def _group() -> art.TrajectoryGroup:
    trajectories = [
        art.Trajectory(
            exchanges=art.TrajectoryExchanges(
                chat_completions=[
                    _exchange("policy", 2),
                    _exchange("judge", 3),
                ]
            ),
            reward=reward,
        )
        for reward in (1.0, 0.0)
    ]
    return art.TrajectoryGroup(trajectories=trajectories)


def _versioned_group() -> art.TrajectoryGroup:
    return art.TrajectoryGroup(
        trajectories=[
            art.Trajectory(
                exchanges=art.TrajectoryExchanges(
                    chat_completions=[
                        _exchange("policy@12", 2),
                        _exchange("judge@4", 3),
                        _exchange("policy@13", 4),
                    ]
                ),
                reward=reward,
            )
            for reward in (1.0, 0.0)
        ]
    )


class _Tokenizer:
    name_or_path = "base/model"


def test_preprocessing_requires_model_selection() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, _Tokenizer())
    with pytest.raises(ValueError, match="exactly one model"):
        list(
            tokenize_trajectory_groups(
                tokenizer,
                [_group()],
                allow_training_without_logprobs=False,
                scale_rewards=False,
            )
        )

    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [_group()],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            model="policy",
        )
    )
    assert len(results) == 2
    assert all(result.token_ids == [1, 2] for result in results)


def test_tinker_requires_model_selection() -> None:
    with pytest.raises(ValueError, match="exactly one model"):
        trajectory_groups_to_datums([_group()], renderer=None, tokenizer=None)

    datums = trajectory_groups_to_datums(
        [_group()],
        renderer=None,
        tokenizer=None,
        model="policy",
    )
    assert len(datums) == 2
    assert all(datum.model_input.to_ints() == [1] for datum in datums)


def test_training_model_patterns_select_all_policy_versions() -> None:
    group = _versioned_group()
    results = list(
        tokenize_trajectory_groups(
            cast(PreTrainedTokenizerBase, _Tokenizer()),
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            model="policy@*",
        )
    )
    assert len(results) == 4
    assert {result.token_ids[-1] for result in results} == {2, 4}

    datums = trajectory_groups_to_datums(
        [group],
        renderer=None,
        tokenizer=None,
        model="policy@*",
    )
    assert len(datums) == 4

    trajectory = group.trajectories[0]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize(model="policy@*")
    tokenized = trajectory.tokenize(model="policy@*", multi_history=True)
    assert [history.model for history in tokenized.histories] == [
        "policy@12",
        "policy@13",
    ]


@pytest.mark.parametrize(
    ("model", "pattern"),
    [
        ("policy@12", "policy@*"),
        (
            "wandb-artifact:///entity/project/run:step12",
            "wandb-artifact:///entity/project/run:step*",
        ),
        ("policy:active", "policy:active"),
        ("base/model", "base/model"),
    ],
)
def test_automatic_training_model_pattern(model: str, pattern: str) -> None:
    assert _training_model_pattern(model) == pattern


def test_preprocessing_preserves_adjacent_choice_boundaries_for_moe() -> None:
    exchanges = [
        _routed_exchange(
            prompt_token_ids=[1],
            output_token=2,
            messages=[{"role": "user", "content": "question"}],
            content="first",
        ),
        _routed_exchange(
            prompt_token_ids=[1, 2],
            output_token=3,
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "again"},
            ],
            content="second",
        ),
    ]
    group = art.TrajectoryGroup(
        trajectories=[
            art.Trajectory(
                exchanges=art.TrajectoryExchanges(
                    chat_completions=exchanges,
                ),
                reward=reward,
            )
            for reward in (1.0, 0.0)
        ]
    )

    results = list(
        tokenize_trajectory_groups(
            cast(PreTrainedTokenizerBase, _Tokenizer()),
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            model="policy",
        )
    )

    assert len(results) == 2
    assert all(result.choice_offsets == [1, 2] for result in results)
    assert all(result.moe_routed_experts is not None for result in results)


def test_preprocessing_preserves_moe_routes_for_reasoning_stripped_suffix() -> None:
    first = _routed_exchange(
        prompt_token_ids=[1],
        output_token=2,
        messages=[{"role": "user", "content": "one"}],
        content="first",
    )
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "first",
        "reasoning": "thought-one",
    }
    first_data["choices"][0]["token_ids"] = [2, 101, 102, 9]
    first_data["choices"][0]["logprobs"]["content"] = [
        {
            "token": f"token_id:{token}",
            "logprob": -token / 10,
            "bytes": [],
            "top_logprobs": [],
        }
        for token in [2, 101, 102, 9]
    ]
    first.response = ChatCompletion.model_validate(first_data)
    first_extra = first.response.choices[0].model_extra
    assert first_extra is not None
    first_extra[ART_MOE_ROUTING_METADATA_KEY] = {
        "prompt_token_ids": [1],
        "completion_token_ids": [2, 101, 102, 9],
        "routed_experts": np.asarray(
            [[[10]], [[20]], [[1010]], [[1020]], [[90]]], dtype=np.int32
        ),
    }

    second = _routed_exchange(
        prompt_token_ids=[1, 101, 102, 9, 4],
        output_token=5,
        messages=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ],
        content="second",
    )
    second_data = second.response.model_dump(mode="python")
    second_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "second",
        "reasoning": "thought-two",
    }
    second_data["choices"][0]["token_ids"] = [5, 6]
    second_data["choices"][0]["logprobs"]["content"] = [
        {
            "token": f"token_id:{token}",
            "logprob": -token / 10,
            "bytes": [],
            "top_logprobs": [],
        }
        for token in [5, 6]
    ]
    second.response = ChatCompletion.model_validate(second_data)
    second_extra = second.response.choices[0].model_extra
    assert second_extra is not None
    second_extra[ART_MOE_ROUTING_METADATA_KEY] = {
        "prompt_token_ids": [1, 101, 102, 9, 4],
        "completion_token_ids": [5, 6],
        "routed_experts": np.asarray(
            [[[10]], [[1010]], [[1020]], [[90]], [[40]], [[50]], [[60]]],
            dtype=np.int32,
        ),
    }

    class Tokenizer(_Tokenizer):
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {
                "one": [1],
                "first": [500],
                "two": [4],
                "thought-two": [5],
                "second": [6],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, object]], **kwargs: object
        ) -> list[int]:
            del kwargs
            content = [message.get("content") for message in messages]
            return (
                [1, 2, 101, 102, 9]
                if content == ["one", "first"]
                else [1, 500, 9, 4, 5, 6]
            )

    group = art.TrajectoryGroup(
        trajectories=[
            art.Trajectory(
                exchanges=art.TrajectoryExchanges(
                    chat_completions=[first, second],
                ),
                reward=reward,
            )
            for reward in (1.0, 0.0)
        ]
    )

    results = list(
        tokenize_trajectory_groups(
            cast(PreTrainedTokenizerBase, Tokenizer()),
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            shuffle_group_trajectories=False,
            drop_zero_advantage_trajectories=False,
            model="policy",
        )
    )

    stripped = [result for result in results if result.token_ids[1] == 101]
    assert len(stripped) == 2
    assert all(result.choice_offsets == [1, 5] for result in stripped)
    expected_routes = np.asarray(
        [[[10]], [[1010]], [[1020]], [[90]], [[40]], [[50]], [[60]]],
        dtype=np.int32,
    )
    for result in stripped:
        assert isinstance(result.moe_routed_experts, np.ndarray)
        assert np.array_equal(result.moe_routed_experts, expected_routes)


def test_ambiguous_non_moe_suffix_falls_back_to_sampled_spans() -> None:
    first = _exchange("policy", 2)
    first_extra = first.response.choices[0].model_extra
    assert first_extra is not None
    first_extra["token_ids"] = [2, 11]
    second = _exchange("policy", 11)
    history = ChatCompletionsHistory(
        model="policy",
        messages=[],
        message_sources=[
            ChatCompletionsMessageSource(exchange=first, choice_index=0),
            ChatCompletionsMessageSource(exchange=second, choice_index=0),
        ],
    )

    assert (
        _chat_choice_trace(
            history,
            [1, 11, 2, 11],
            [
                art.TokenFlag.EXACT,
                art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
                art.TokenFlag.EXACT,
                art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
            ],
        )
        is None
    )


def test_preprocessing_rejects_partial_choice_evidence_before_moe_routes() -> None:
    first = _routed_exchange(
        prompt_token_ids=[1],
        output_token=2,
        messages=[{"role": "user", "content": "question"}],
        content="first",
    )
    first_extra = first.response.choices[0].model_extra
    assert first_extra is not None
    first_extra.pop("token_ids")
    first_extra.pop(ART_MOE_ROUTING_METADATA_KEY)
    second = _routed_exchange(
        prompt_token_ids=[1, 2],
        output_token=3,
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "again"},
        ],
        content="second",
    )
    group = art.TrajectoryGroup(
        trajectories=[
            art.Trajectory(
                exchanges=art.TrajectoryExchanges(chat_completions=[first, second]),
                reward=reward,
            )
            for reward in (1.0, 0.0)
        ]
    )

    with pytest.raises(RuntimeError, match="every sourced choice"):
        list(
            tokenize_trajectory_groups(
                cast(PreTrainedTokenizerBase, _Tokenizer()),
                [group],
                allow_training_without_logprobs=False,
                scale_rewards=False,
                model="policy",
            )
        )
