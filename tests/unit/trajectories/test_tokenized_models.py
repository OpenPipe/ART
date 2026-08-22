import math
import pickle

import pytest

import art
import art.trajectories as tr


def _history() -> tr.TokenizedHistory:
    return tr.TokenizedHistory(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=[1, 2],
        logprobs=[math.nan, -0.25],
        flags=[
            tr.TokenFlag.EXACT,
            tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED,
        ],
    )


def _assert_history_round_trip(
    restored: tr.TokenizedHistory,
    expected: tr.TokenizedHistory,
) -> None:
    assert restored.model == expected.model
    assert restored.tokens == expected.tokens
    assert restored.flags == expected.flags
    assert math.isnan(restored.logprobs[0])
    assert restored.logprobs[1:] == expected.logprobs[1:]


def test_tokenized_history_nan_json_round_trip() -> None:
    value = _history()
    payload = value.model_dump_json()
    assert '"NaN"' in payload
    restored = tr.TokenizedHistory.model_validate_json(payload)
    _assert_history_round_trip(restored, value)


def test_tokenized_trajectory_nan_json_round_trip() -> None:
    trajectory = art.Trajectory(
        reward=1.0, metrics={"count": 1}, metadata={"source": "test"}
    )
    value = tr.TokenizedTrajectory(
        **_history().model_dump(),
        trajectory=trajectory,
    )
    restored = tr.TokenizedTrajectory.model_validate_json(value.model_dump_json())
    _assert_history_round_trip(restored, value)
    assert restored.reward == value.reward
    assert restored.metrics == value.metrics
    assert restored.metadata == value.metadata


def test_nested_tokenized_models_nan_json_round_trip() -> None:
    source = art.Trajectory(reward=1.0)
    trajectory = tr.TokenizedMultiHistoryTrajectory(
        trajectory=source,
        histories=[_history()],
    )
    group = tr.TokenizedTrajectoryGroup[tr.TokenizedMultiHistoryTrajectory](
        trajectory_group=art.TrajectoryGroup([source]),
        trajectories=[trajectory],
    )
    restored = tr.TokenizedTrajectoryGroup[
        tr.TokenizedMultiHistoryTrajectory
    ].model_validate_json(group.model_dump_json())
    restored_trajectory = restored.trajectories[0]
    _assert_history_round_trip(restored_trajectory.histories[0], _history())
    assert restored_trajectory.reward == trajectory.reward
    assert restored.metrics == group.metrics
    assert restored.metadata == group.metadata


def test_tokenized_group_skips_equality_dump_for_identical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = art.Trajectory()
    tokenized = tr.TokenizedTrajectory(**_history().model_dump(), trajectory=source)

    def unexpected_dump(*_: object, **__: object) -> object:
        raise AssertionError("identical source trajectories must not be dumped")

    monkeypatch.setattr(art.Trajectory, "model_dump", unexpected_dump)
    group = tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory](
        trajectory_group=art.TrajectoryGroup([source]),
        trajectories=[tokenized],
    )

    assert group.trajectories[0].trajectory is source


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
    single_value = single.trajectories[0]
    assert isinstance(single_value.history, tr.ChatCompletionsHistory)
    assert single.trajectory_group.trajectories[0] is single_value.trajectory
    assert single_value.history.message_sources[0] is not None
    assert (
        single_value.history.message_sources[0].exchange
        is single_value.trajectory.exchanges.chat_completions[0]
    )
    single_json = single.model_dump_json()
    assert '"NaN"' in single_json
    tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory].model_validate_json(single_json)

    multi = group.tokenize(multi_history=True)
    multi_json = multi.model_dump_json()
    assert '"NaN"' in multi_json
    tr.TokenizedTrajectoryGroup[tr.TokenizedMultiHistoryTrajectory].model_validate_json(
        multi_json
    )


def test_tokenized_compact_round_trips_retain_source_references() -> None:
    from datetime import datetime

    from openai.types.chat import ChatCompletion

    from art.trajectories import ChatCompletionsExchange, TrajectoryExchanges

    exchange = ChatCompletionsExchange(
        request={
            "model": "policy",
            "messages": [{"role": "user", "content": "question"}],
        },
        response=ChatCompletion.model_validate(
            {
                "id": "chat",
                "object": "chat.completion",
                "created": 0,
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                        "prompt_token_ids": [1],
                        "token_ids": [2],
                    }
                ],
            }
        ),
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
    )
    source_group = art.TrajectoryGroup(
        [art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[exchange]))]
    )
    tokenized_group = source_group.tokenize()
    tokenized = tokenized_group.trajectories[0]
    history = tokenized.history
    assert isinstance(history, tr.ChatCompletionsHistory)

    restored_history = art.trajectories.compact_validate(
        history.tokenize().compact_dump(), type=tr.TokenizedHistory
    )
    assert isinstance(restored_history.history, tr.ChatCompletionsHistory)
    first_source = restored_history.history.message_sources[0]
    last_source = restored_history.history.message_sources[-1]
    assert first_source is not None and last_source is not None
    assert first_source.exchange is last_source.exchange

    restored = art.trajectories.compact_validate(
        tokenized.compact_dump(), type=tr.TokenizedTrajectory
    )
    assert isinstance(restored.history, tr.ChatCompletionsHistory)
    restored_source = restored.history.message_sources[0]
    assert restored_source is not None
    assert restored_source.exchange is restored.trajectory.exchanges.chat_completions[0]

    pickled = pickle.loads(pickle.dumps(tokenized))
    assert isinstance(pickled.history, tr.ChatCompletionsHistory)
    pickled_source = pickled.history.message_sources[0]
    assert pickled_source is not None
    assert pickled_source.exchange is pickled.trajectory.exchanges.chat_completions[0]

    tokenized_multi = tr.TokenizedMultiHistoryTrajectory(
        trajectory=tokenized.trajectory,
        histories=[
            tr.TokenizedHistory(
                history=tokenized.history,
                model=tokenized.model,
                tokens=tokenized.tokens,
                logprobs=tokenized.logprobs,
                flags=tokenized.flags,
            )
        ],
    )
    restored_multi = art.trajectories.compact_validate(
        tokenized_multi.compact_dump(), type=tr.TokenizedMultiHistoryTrajectory
    )
    assert isinstance(restored_multi.histories[0].history, tr.ChatCompletionsHistory)
    restored_multi_source = restored_multi.histories[0].history.message_sources[0]
    assert restored_multi_source is not None
    assert (
        restored_multi_source.exchange
        is restored_multi.trajectory.exchanges.chat_completions[0]
    )

    restored_group = art.trajectories.compact_validate(
        tokenized_group.compact_dump(),
        type=tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory],
    )
    restored_child = restored_group.trajectories[0]
    assert isinstance(restored_child, tr.TokenizedTrajectory)
    assert isinstance(restored_child.history, tr.ChatCompletionsHistory)
    assert restored_child.trajectory is restored_group.trajectory_group.trajectories[0]
    restored_group_source = restored_child.history.message_sources[0]
    assert restored_group_source is not None
    assert (
        restored_group_source.exchange
        is restored_group.trajectory_group.trajectories[0].exchanges.chat_completions[0]
    )

    payload = art.trajectories.compact_dump([tokenized])
    restored_list = art.trajectories.compact_validate(
        payload, type=list[tr.TokenizedTrajectory]
    )
    assert restored_list[0].model_dump() == tokenized.model_dump()
    history_payload = art.trajectories.compact_dump([history.tokenize()])
    assert (
        len(
            art.trajectories.compact_validate(
                history_payload, type=list[tr.TokenizedHistory]
            )
        )
        == 1
    )
    multi_payload = art.trajectories.compact_dump([tokenized_multi])
    assert (
        len(
            art.trajectories.compact_validate(
                multi_payload, type=list[tr.TokenizedMultiHistoryTrajectory]
            )
        )
        == 1
    )
    group_payload = art.trajectories.compact_dump([tokenized_group])
    assert (
        len(
            art.trajectories.compact_validate(
                group_payload,
                type=list[tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory]],
            )
        )
        == 1
    )
