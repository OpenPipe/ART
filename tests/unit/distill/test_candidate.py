from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any

from openai.types.chat import ChatCompletion
from pydantic import ValidationError
import pytest

from art.distill.candidate import CandidateTrainingBatch, snapshot_groups
from art.distill.capture import last_generation
from art.distill.types import Example, GenerationPart
from art.trajectories import (
    ChatCompletionsExchange,
    TokenFlag,
    Trajectory,
    TrajectoryExchanges,
    TrajectoryGroup,
)


def _trajectory(
    identity: int,
    *,
    reward: float = 0.0,
    completion: tuple[int, ...] = (3, 4),
    completion_logprobs: tuple[float, ...] | None = (-0.25, -0.5),
) -> Trajectory:
    choice: dict[str, Any] = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": f"answer-{identity}"},
        "prompt_token_ids": [identity + 1, identity + 2],
        "token_ids": list(completion),
    }
    if completion_logprobs is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": f"token_id:{token_id}",
                    "logprob": logprob,
                    "bytes": None,
                    "top_logprobs": [],
                }
                for token_id, logprob in zip(
                    completion,
                    completion_logprobs,
                    strict=True,
                )
            ]
        }
    start = datetime(2026, 1, 1) + timedelta(seconds=identity)
    exchange = ChatCompletionsExchange(
        request={
            "model": "student",
            "messages": [{"role": "user", "content": f"question-{identity}"}],
        },
        response=ChatCompletion.model_validate(
            {
                "id": f"response-{identity}",
                "object": "chat.completion",
                "created": identity,
                "model": "student",
                "choices": [choice],
            }
        ),
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )
    return Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange]),
        reward=reward,
        initial_policy_version=7,
        final_policy_version=7,
    )


def _group(*trajectories: Trajectory) -> TrajectoryGroup:
    return TrajectoryGroup(trajectories)


def test_snapshot_is_deterministic_order_sensitive_and_detached() -> None:
    first = _trajectory(10, reward=1.0)
    second = _trajectory(20, reward=3.0)
    first.metadata["api_key"] = "secret"
    first.logs.append("authorization: secret")
    group = _group(first, second)
    group.metadata["private"] = {"password": "secret"}
    group.logs.append("secret")

    candidate = snapshot_groups([group])
    copied = snapshot_groups([group.model_copy(deep=True)])
    reversed_candidate = snapshot_groups([_group(second, first)])

    assert candidate == copied
    assert candidate.batch_id == copied.batch_id
    assert candidate.batch_id != reversed_candidate.batch_id
    serialized = candidate.model_dump_json()
    assert "secret" not in serialized
    assert "metadata" not in serialized
    assert "logs" not in serialized

    first.reward = 100
    first.metadata["api_key"] = "changed"
    assert candidate.groups[0].trajectories[0].reward == 1.0
    assert candidate.groups[0].trajectories[0].token_ids == (11, 12, 3, 4)


def test_snapshot_preserves_exact_tokens_flags_offsets_and_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_tokenizer_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact Chat Completions path must not load a tokenizer")

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        unexpected_tokenizer_load,
    )
    snapshot = snapshot_groups([_group(_trajectory(10))]).groups[0].trajectories[0]

    assert snapshot.token_ids == (11, 12, 3, 4)
    assert snapshot.logprobs == (None, None, -0.25, -0.5)
    assert snapshot.token_flags == (
        int(TokenFlag.EXACT),
        int(TokenFlag.EXACT),
        int(TokenFlag.EXACT | TokenFlag.SAMPLED),
        int(TokenFlag.EXACT | TokenFlag.SAMPLED),
    )
    assert snapshot.generations[0].trajectory_token_start == 2
    assert snapshot.generations[0].continuation_token_ids == (3, 4)


def test_nonfinite_or_unavailable_logprobs_become_none() -> None:
    trajectory = _trajectory(
        10,
        completion_logprobs=(float("nan"), float("-inf")),
    )

    snapshot = snapshot_groups([_group(trajectory)]).groups[0].trajectories[0]

    assert snapshot.logprobs == (None, None, None, None)


def test_group_advantages_are_raw_reward_minus_group_mean() -> None:
    candidate = snapshot_groups(
        [_group(_trajectory(10, reward=1.0), _trajectory(20, reward=4.0))]
    )

    trajectories = candidate.groups[0].trajectories
    assert [item.reward for item in trajectories] == [1.0, 4.0]
    assert [item.advantage for item in trajectories] == [-1.5, 1.5]


def test_zero_variance_and_zero_advantage_trajectories_are_retained() -> None:
    candidate = snapshot_groups(
        [_group(_trajectory(10, reward=2.0), _trajectory(20, reward=2.0))]
    )

    assert len(candidate.groups[0].trajectories) == 2
    assert [item.advantage for item in candidate.groups[0].trajectories] == [
        0.0,
        0.0,
    ]


def test_examples_must_be_unique_and_owned_by_snapshot() -> None:
    trajectory = _trajectory(10)
    generation = last_generation(trajectory)
    example = Example.create(
        generation=generation,
        teacher_view=generation.context,
        parts={GenerationPart.ASSISTANT_TEXT},
    )
    candidate = snapshot_groups([_group(trajectory)], examples=[example])

    assert candidate.generation(generation.generation_id) == generation
    with pytest.raises(KeyError):
        candidate.generation("missing")
    with pytest.raises(ValidationError, match="at most once"):
        CandidateTrainingBatch.create(
            groups=candidate.groups,
            examples=(example, example),
        )

    foreign_generation = last_generation(_trajectory(20))
    foreign = Example.create(
        generation=foreign_generation,
        teacher_view=foreign_generation.context,
        parts={GenerationPart.ASSISTANT_TEXT},
    )
    with pytest.raises(ValidationError, match="does not belong"):
        CandidateTrainingBatch.create(
            groups=candidate.groups,
            examples=(foreign,),
        )


def test_duplicate_trajectory_identity_is_rejected() -> None:
    trajectory = _trajectory(10)

    with pytest.raises(ValidationError, match="trajectory fingerprints"):
        snapshot_groups([_group(trajectory, trajectory.model_copy(deep=True))])


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([], "at least one group"),
        ([_group()], "must contain trajectories"),
    ],
)
def test_empty_inputs_are_rejected(
    groups: list[TrajectoryGroup],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        snapshot_groups(groups)


def test_missing_exact_capture_and_nonfinite_reward_are_rejected() -> None:
    missing_exact = _trajectory(10)
    extra = missing_exact.exchanges.chat_completions[0].response.choices[0].model_extra
    assert extra is not None
    extra.pop("token_ids")

    with pytest.raises(ValueError, match="exact prompt and completion token IDs"):
        snapshot_groups([_group(missing_exact)])
    with pytest.raises(ValueError, match="rewards must be finite"):
        snapshot_groups([_group(_trajectory(20, reward=math.inf))])
