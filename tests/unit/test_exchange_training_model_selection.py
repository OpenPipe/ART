from __future__ import annotations

from datetime import datetime, timedelta
from typing import SupportsIndex, cast, overload

import numpy as np
from openai.types import Completion
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
    CompletionsExchange,
    CompletionsRequest,
)
from art.trajectories._selection import (
    automatic_training_model_selector,
    resolve_training_model,
)


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
    with pytest.raises(ValueError, match="exactly one concrete model"):
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
    with pytest.raises(ValueError, match="exactly one concrete model"):
        trajectory_groups_to_datums([_group()], renderer=None, tokenizer=None)

    datums = trajectory_groups_to_datums(
        [_group()],
        renderer=None,
        tokenizer=None,
        model="policy",
    )
    assert len(datums) == 2
    assert all(datum.model_input.to_ints() == [1] for datum in datums)


def test_training_rejects_multiple_concrete_policy_versions() -> None:
    group = _versioned_group()
    with pytest.raises(ValueError, match="exactly one concrete model"):
        list(
            tokenize_trajectory_groups(
                cast(PreTrainedTokenizerBase, _Tokenizer()),
                [group],
                allow_training_without_logprobs=False,
                scale_rewards=False,
                model="policy@*",
            )
        )

    with pytest.raises(ValueError, match="exactly one concrete model"):
        trajectory_groups_to_datums(
            [group],
            renderer=None,
            tokenizer=None,
            model="policy@*",
        )

    trajectory = group.trajectories[0]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize(model="policy@*")
    tokenized = trajectory.tokenize(model="policy@*", multi_history=True)
    assert [history.model for history in tokenized.histories] == [
        "policy@12",
        "policy@13",
    ]


@pytest.mark.parametrize(
    ("model", "matches", "misses"),
    [
        ("policy@12", ("policy@0", "policy@12"), ("policy@x", "policy@12x")),
        (
            "wandb-artifact:///entity/project/run:step12",
            (
                "wandb-artifact:///entity/project/run:step0",
                "wandb-artifact:///entity/project/run:step12",
            ),
            ("wandb-artifact:///entity/project/run:stepx",),
        ),
        ("policy:active", ("policy:active",), ("policy:active2",)),
        ("base/model", ("base/model",), ("base/model@1",)),
    ],
)
def test_automatic_training_model_selector(
    model: str, matches: tuple[str, ...], misses: tuple[str, ...]
) -> None:
    selector = automatic_training_model_selector(model)
    assert all(selector.matches(candidate) for candidate in matches)
    assert not any(selector.matches(candidate) for candidate in misses)


def test_automatic_training_model_selector_treats_family_metacharacters_literally() -> (
    None
):
    selector = automatic_training_model_selector("policy[blue]*@12")
    assert selector.matches("policy[blue]*@13")
    assert not selector.matches("policyb@13")
    assert not selector.matches("policy[blue]anything@13")


def test_automatic_training_selector_rejects_multiple_numeric_steps() -> None:
    trajectory = _versioned_group().trajectories[0]
    selector = automatic_training_model_selector("policy@12")
    with pytest.raises(ValueError, match="exactly one concrete model"):
        resolve_training_model(trajectory, selector)


def test_public_training_selector_prefers_exact_model_over_glob_interpretation() -> (
    None
):
    trajectory = art.Trajectory(
        exchanges=art.TrajectoryExchanges(
            chat_completions=[
                _exchange("policy[*]", 2),
                _exchange("policyx", 3),
            ]
        )
    )
    assert resolve_training_model(trajectory, "policy[*]") == "policy[*]"


def test_training_selector_rejects_zero_matches() -> None:
    with pytest.raises(ValueError, match="no exchanges"):
        resolve_training_model(_group().trajectories[0], "missing*")


def test_training_selector_rejects_mixed_protocols() -> None:
    start = datetime(2026, 1, 1)
    completion = CompletionsExchange(
        request=CompletionsRequest(model="policy", prompt="question"),
        response=Completion.model_validate(
            {
                "id": "cmpl",
                "object": "text_completion",
                "created": 1,
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "text": "answer",
                    }
                ],
            }
        ),
        start_time=start,
        end_time=start + timedelta(seconds=1),
    )
    trajectory = art.Trajectory(
        exchanges=art.TrajectoryExchanges(
            chat_completions=[_exchange("policy", 2)],
            completions=[completion],
        )
    )
    with pytest.raises(ValueError, match="mixed protocols"):
        resolve_training_model(trajectory, "policy")


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


def test_chat_choice_trace_anchors_retained_suffix_at_its_prompt_boundary() -> None:
    first = _exchange("policy", 8)
    first_extra = first.response.choices[0].model_extra
    assert first_extra is not None
    first_extra["prompt_token_ids"] = [1]
    first_extra["token_ids"] = [7, 8]
    second = _exchange("policy", 8)
    second_extra = second.response.choices[0].model_extra
    assert second_extra is not None
    second_extra["prompt_token_ids"] = [1, 8, 9]
    second_extra["token_ids"] = [7, 8]
    history = ChatCompletionsHistory(
        model="policy",
        messages=[],
        message_sources=[
            ChatCompletionsMessageSource(exchange=first, choice_index=0),
            ChatCompletionsMessageSource(exchange=second, choice_index=0),
        ],
    )

    trace = _chat_choice_trace(
        history,
        [1, 8, 9, 7, 8],
        [
            art.TokenFlag.EXACT,
            art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
            art.TokenFlag.EXACT,
            art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
            art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        ],
    )

    assert trace is not None
    assert trace.offsets == [1, 3]
    assert trace.lengths == [1, 2]


def test_chat_choice_trace_does_bounded_work_for_a_retained_suffix() -> None:
    class CountingList(list[int]):
        slice_reads = 0

        @overload
        def __getitem__(self, key: SupportsIndex, /) -> int: ...

        @overload
        def __getitem__(self, key: slice[SupportsIndex | None], /) -> list[int]: ...

        def __getitem__(
            self, key: SupportsIndex | slice[SupportsIndex | None], /
        ) -> int | list[int]:
            if isinstance(key, slice):
                self.slice_reads += 1
            return super().__getitem__(key)

    exchange = _exchange("policy", 7)
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    extra["prompt_token_ids"] = [1]
    extra["token_ids"] = [*([8] * 510), 7]
    history = ChatCompletionsHistory(
        model="policy",
        messages=[],
        message_sources=[
            ChatCompletionsMessageSource(exchange=exchange, choice_index=0)
        ],
    )
    token_ids = CountingList([1, 7, *([0] * 510)])
    flags = [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        *([art.TokenFlag.EXACT] * 510),
    ]

    trace = _chat_choice_trace(history, token_ids, flags)

    assert trace is not None
    assert trace.offsets == [1]
    assert trace.lengths == [1]
    assert token_ids.slice_reads < 10


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
