import math

import art


def _history() -> art.TokenizedHistory:
    return art.TokenizedHistory(
        model="policy",
        token_ids=[1, 2],
        logprobs=[math.nan, -0.25],
        flags=[art.TokenFlag.EXACT, art.TokenFlag.EXACT | art.TokenFlag.SAMPLED],
    )


def _assert_history_round_trip(
    restored: art.TokenizedHistory, expected: art.TokenizedHistory
) -> None:
    assert restored.model == expected.model
    assert restored.token_ids == expected.token_ids
    assert restored.flags == expected.flags
    assert math.isnan(restored.logprobs[0])
    assert restored.logprobs[1:] == expected.logprobs[1:]


def test_tokenized_history_nan_json_round_trip() -> None:
    value = _history()
    payload = value.model_dump_json()
    assert '"NaN"' in payload
    restored = art.TokenizedHistory.model_validate_json(payload)
    _assert_history_round_trip(restored, value)


def test_tokenized_trajectory_nan_json_round_trip() -> None:
    value = art.TokenizedTrajectory(
        **_history().model_dump(),
        reward=1.0,
        metrics={"count": 1},
        metadata={"source": "test"},
    )
    restored = art.TokenizedTrajectory.model_validate_json(value.model_dump_json())
    _assert_history_round_trip(restored, value)
    assert restored.reward == value.reward
    assert restored.metrics == value.metrics
    assert restored.metadata == value.metadata


def test_nested_tokenized_models_nan_json_round_trip() -> None:
    trajectory = art.TokenizedMultiHistoryTrajectory(
        histories=[_history()],
        reward=1.0,
        metrics={},
        metadata={},
    )
    group = art.TokenizedTrajectoryGroup[art.TokenizedMultiHistoryTrajectory](
        trajectories=[trajectory],
        metrics={},
        metadata={},
    )
    restored = art.TokenizedTrajectoryGroup[
        art.TokenizedMultiHistoryTrajectory
    ].model_validate_json(group.model_dump_json())
    restored_trajectory = restored.trajectories[0]
    _assert_history_round_trip(restored_trajectory.histories[0], _history())
    assert restored_trajectory.reward == trajectory.reward
    assert restored.metrics == group.metrics
    assert restored.metadata == group.metadata


def test_public_group_tokenization_nan_json_round_trip() -> None:
    from datetime import datetime

    from openai.types.chat import ChatCompletion

    from art.trajectories import (
        ChatCompletionsExchange,
        ChatCompletionsRequest,
        TrajectoryExchanges,
    )

    exchange = ChatCompletionsExchange(
        request=ChatCompletionsRequest(
            model="policy",
            messages=[{"role": "user", "content": "question"}],
        ),
        response=ChatCompletion.model_validate(
            {
                "id": "chat",
                "object": "chat.completion",
                "created": 0,
                "model": "policy",
                "choices": [
                    {
                        "index": index,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                        "prompt_token_ids": [1],
                        "token_ids": [token_id],
                        "logprobs": {
                            "content": [
                                {
                                    "token": f"token_id:{token_id}",
                                    "logprob": -0.1 * token_id,
                                    "bytes": [],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    }
                    for index, (text, token_id) in enumerate(
                        (("left", 2), ("right", 3))
                    )
                ],
            }
        ),
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
    )
    group = art.TrajectoryGroup(
        [art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[exchange]))]
    )

    single_exchange = exchange.model_copy(
        update={
            "response": exchange.response.model_copy(
                update={"choices": [exchange.response.choices[0]]}
            )
        }
    )
    single = art.TrajectoryGroup(
        [
            art.Trajectory(
                exchanges=TrajectoryExchanges(chat_completions=[single_exchange])
            )
        ]
    ).tokenize()
    single_json = single.model_dump_json()
    assert '"NaN"' in single_json
    art.TokenizedTrajectoryGroup[art.TokenizedTrajectory].model_validate_json(
        single_json
    )

    multi = group.tokenize(multi_history=True)
    multi_json = multi.model_dump_json()
    assert '"NaN"' in multi_json
    art.TokenizedTrajectoryGroup[
        art.TokenizedMultiHistoryTrajectory
    ].model_validate_json(multi_json)
