from datetime import datetime, timedelta
from typing import cast

import numpy as np
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import pytest
from transformers import PreTrainedTokenizerBase

import art
from art.openai import ART_MOE_ROUTING_METADATA_KEY
from art.preprocessing.tokenize import tokenize_trajectory_groups
from art.tinker_native.data import trajectory_groups_to_datums
from art.trajectories import ChatCompletionsExchange, ChatCompletionsRequest


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
