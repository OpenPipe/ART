from datetime import datetime
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
import pytest

import art
from art.trajectories import ChatCompletionsExchange, ChatCompletionsRequest

pytest.importorskip("tinker")

from art.tinker_native.data import trajectory_groups_to_datums  # noqa: E402


def _exchange(
    prompt: list[int], output: list[int], *, logprobs: bool = True
) -> ChatCompletionsExchange:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "question"}
    ]
    if len(prompt) > 1:
        messages.extend(
            [
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "next"},
            ]
        )
    response = ChatCompletion.model_validate(
        {
            "id": "chat-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": prompt,
                    "token_ids": output,
                    "logprobs": (
                        {
                            "content": [
                                {
                                    "token": f"token_id:{token}",
                                    "logprob": -token / 10,
                                    "top_logprobs": [],
                                }
                                for token in output
                            ]
                        }
                        if logprobs
                        else None
                    ),
                }
            ],
        }
    )
    return ChatCompletionsExchange(
        request=ChatCompletionsRequest(model="test/model", messages=messages),
        response=response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


def _trajectory(reward: float, *, logprobs: bool = True) -> art.Trajectory:
    return art.Trajectory(
        exchanges=art.TrajectoryExchanges(
            chat_completions=[
                _exchange([10], [20], logprobs=logprobs),
                _exchange([10, 20, 11], [21], logprobs=logprobs),
            ]
        ),
        reward=reward,
    )


def test_exchange_trajectory_builds_masked_multiturn_datum() -> None:
    datums = trajectory_groups_to_datums(
        [art.TrajectoryGroup([_trajectory(1), _trajectory(-1)])],
        renderer=None,
        tokenizer=None,
        normalize_advantages=False,
    )

    assert len(datums) == 2
    datum = datums[0]
    assert datum.model_input.to_ints() == [10, 20, 11]
    assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == [20, 11, 21]
    assert datum.loss_fn_inputs["logprobs"].to_torch().tolist() == pytest.approx(
        [-2.0, 0.0, -2.1]
    )
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [1, 0, 1]
    assert datum.loss_fn_inputs["mask"].to_torch().tolist() == [1, 0, 1]


def test_exchange_trajectory_requires_assistant_logprobs() -> None:
    with pytest.raises(ValueError, match="requires logprobs"):
        trajectory_groups_to_datums(
            [art.TrajectoryGroup([_trajectory(1, logprobs=False), _trajectory(-1)])],
            renderer=None,
            tokenizer=None,
            normalize_advantages=False,
        )


def test_legacy_trajectory_still_uses_history_conversion() -> None:
    class Prompt:
        def to_ints(self) -> list[int]:
            return [10]

    class Renderer:
        def build_generation_prompt(self, _messages: list[dict[str, Any]]) -> Prompt:
            return Prompt()

    class Tokenizer:
        def convert_tokens_to_ids(self, _token: str) -> int:
            return 20

    choice = Choice.model_validate(
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "answer"},
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:20",
                        "logprob": -2.0,
                        "top_logprobs": [],
                    }
                ]
            },
        }
    )

    def trajectory(reward: float) -> art.Trajectory:
        return art.Trajectory(
            messages_and_choices=[{"role": "user", "content": "question"}, choice],
            reward=reward,
        )

    datums = trajectory_groups_to_datums(
        [art.TrajectoryGroup([trajectory(1), trajectory(-1)])],
        renderer=Renderer(),
        tokenizer=Tokenizer(),
        normalize_advantages=False,
    )

    assert len(datums) == 2
    assert datums[0].model_input.to_ints() == [10]
    assert datums[0].loss_fn_inputs["target_tokens"].to_torch().tolist() == [20]
