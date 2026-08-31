from __future__ import annotations

from datetime import datetime
import json
import math
import pickle
import subprocess
import sys

from openai.types.chat import ChatCompletion
import pytest

import art
import art.trajectories as tr

torch = pytest.importorskip("torch")


def _tokenized_history() -> tr.TokenizedHistory:
    return tr.TokenizedHistory(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=[1, 2],
        logprobs=[math.nan, -0.25],
        flags=[
            tr.TokenFlag.EXACT,
            tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED | tr.TokenFlag.STOP,
        ],
    )


def test_tensorized_history_rejects_sampled_without_exact() -> None:
    assert not hasattr(tr.TensorizedHistory, "exact_mask")
    assert not hasattr(tr.TensorizedHistory, "sampled_mask")
    assert not hasattr(tr.TensorizedHistory, "assistant_mask")
    assert not hasattr(tr.TensorizedHistory, "stop_mask")
    with pytest.raises(ValueError, match="SAMPLED tokens must also be EXACT"):
        tr.TensorizedHistory(
            history=tr.LegacyHistory(messages_and_choices=[]),
            model="policy",
            tokens=[1],
            logprobs=[math.nan],
            flags=[tr.TokenFlag.SAMPLED],
        )


def _trajectory() -> art.Trajectory:
    exchange = tr.ChatCompletionsExchange(
        request=tr.ChatCompletionsRequest(
            model="policy", messages=[{"role": "user", "content": "question"}]
        ),
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
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:2",
                                    "logprob": -0.25,
                                    "bytes": [],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
    )
    return art.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))


def test_ordinary_trajectory_import_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; import art.trajectories; "
                "assert 'art.trajectories.tensors' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_torch_reports_tensor_extra() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; import art.trajectories as t; "
                "v=t.TokenizedHistory(history=t.LegacyHistory(messages_and_choices=[]),"
                "model='m',tokens=[1],logprobs=[0.],flags=[t.TokenFlag.EXACT]); "
                "v.tensorize()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "openpipe-art[tensors]" in result.stderr


def test_history_tensorize_uses_canonical_tensors_and_moves_in_place() -> None:
    value = _tokenized_history().tensorize()

    assert isinstance(value, tr.TensorizedHistory)
    assert value.tokens.dtype is torch.int64
    assert value.logprobs.dtype is torch.float32
    assert value.flags.dtype is torch.int32
    assert value.tokens.device.type == "cpu"
    assert value.tokens.is_contiguous()
    assert value.to("cpu") is value
    assert "tokenized" not in type(value).model_fields


def test_direct_tensorization_matches_tokenized_conversion() -> None:
    trajectory = _trajectory()
    direct = trajectory.tensorize()
    converted = trajectory.tokenize().tensorize()
    history_view = trajectory.history()
    assert isinstance(history_view, tr.ChatCompletionsHistory)
    history = history_view.tensorize()
    group = art.TrajectoryGroup([trajectory]).tensorize()

    assert torch.equal(direct.tokens, converted.tokens)
    assert torch.allclose(direct.logprobs, converted.logprobs, equal_nan=True)
    assert torch.equal(direct.flags, converted.flags)
    assert torch.equal(history.tokens, direct.tokens)
    assert torch.equal(group.trajectories[0].tokens, direct.tokens)
    assert direct.trajectory is trajectory
    assert group.trajectory_group.trajectories[0] is trajectory

    choice = trajectory.exchanges.chat_completions[0].response.choices[0]
    legacy = art.Trajectory(messages_and_choices=[choice])
    direct_legacy = legacy.tensorize(model="policy")
    converted_legacy = legacy.tokenize(model="policy").tensorize()
    assert torch.equal(direct_legacy.tokens, converted_legacy.tokens)


def test_tensorized_history_validates_shapes_lengths_and_json() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        tr.TensorizedHistory(
            history=tr.LegacyHistory(messages_and_choices=[]),
            model="policy",
            tokens=[[1]],
            logprobs=[0.0],
            flags=[1],
        )
    with pytest.raises(ValueError, match="differ in length"):
        tr.TensorizedHistory(
            history=tr.LegacyHistory(messages_and_choices=[]),
            model="policy",
            tokens=[1, 2],
            logprobs=[0.0],
            flags=[1],
        )

    value = _tokenized_history().tensorize()
    payload = value.model_dump_json(warnings="error")
    assert '"NaN"' in payload
    assert "device" not in payload
    restored = tr.TensorizedHistory.model_validate_json(payload)
    assert torch.equal(restored.tokens, value.tokens)
    assert torch.equal(restored.flags, value.flags)
    assert torch.allclose(restored.logprobs, value.logprobs, equal_nan=True)
    assert isinstance(value.model_dump()["tokens"], torch.Tensor)


def test_trajectory_and_group_tensorize_retain_mutable_sources() -> None:
    trajectory = art.Trajectory(
        reward=1.0,
        metrics={"score": 1},
        metadata={"split": "train"},
    )
    tokenized = tr.TokenizedTrajectory(
        **_tokenized_history().model_dump(), trajectory=trajectory
    )
    tokenized.reward = 1.5
    tokenized.metrics = {"score": 1.5}
    tokenized.metadata = {"split": "train"}
    assert trajectory.reward == 1.5
    assert tokenized.metrics is trajectory.metrics
    assert tokenized.metadata is trajectory.metadata
    tensorized = tokenized.tensorize()

    assert tensorized.trajectory is trajectory
    assert tensorized.reward == 1.5
    assert tensorized.metrics is trajectory.metrics
    assert tensorized.metadata is trajectory.metadata
    tensorized.reward = 2.0
    trajectory_metrics: dict[str, float | int | bool] = {"score": 2}
    tensorized.metrics = trajectory_metrics
    tensorized.metadata["epoch"] = 3
    assert trajectory.reward == 2.0
    assert trajectory.metrics == {"score": 2}
    assert trajectory.metadata == {"split": "train", "epoch": 3}
    assert set(tensorized.model_dump()) == {
        "history",
        "model",
        "tokens",
        "logprobs",
        "flags",
        "trajectory",
    }

    source_group = art.TrajectoryGroup(
        [trajectory], metrics={"batch": 1}, metadata={"name": "group"}
    )
    tokenized_group = tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory](
        trajectory_group=source_group,
        trajectories=[tokenized],
    )
    tensorized_group = tokenized_group.tensorize()
    assert tensorized_group.trajectory_group is source_group
    assert tensorized_group.trajectories[0].trajectory is trajectory
    group_metrics: dict[str, float | int | bool] = {"batch": 2}
    tensorized_group.metrics = group_metrics
    tensorized_group.metadata = {"name": "updated"}
    assert source_group.metrics == {"batch": 2}
    assert source_group.metadata == {"name": "updated"}


def test_tensorized_compact_round_trips_inferred_and_typed() -> None:
    value = _tokenized_history().tensorize()
    payload = value.compact_dump()

    assert payload["kind"] == "tensorized_history"
    assert "device" not in json.dumps(payload)
    inferred = tr.compact_validate(payload)
    typed = tr.compact_validate(payload, type=tr.TensorizedHistory)
    for restored in (inferred, typed):
        assert isinstance(restored, tr.TensorizedHistory)
        assert torch.equal(restored.tokens, value.tokens)
        assert torch.allclose(restored.logprobs, value.logprobs, equal_nan=True)

    collection = tr.compact_dump([value])
    restored_values = tr.compact_validate(
        collection, type=list[tr.TensorizedHistory], device="cpu"
    )
    assert len(restored_values) == 1
    assert restored_values[0].tokens.device.type == "cpu"
    with pytest.raises(ValueError, match="expects kind"):
        tr.compact_validate(payload, type=tr.TensorizedTrajectory)
    with pytest.raises(ValueError, match="only valid for tensorized"):
        tr.compact_validate(art.Trajectory().compact_dump(), device="cpu")


def test_tensorized_trajectory_multi_history_and_group_compact_round_trips() -> None:
    trajectory = _trajectory()
    tokenized = trajectory.tokenize()
    tensorized = tokenized.tensorize()
    multi = tr.TokenizedMultiHistoryTrajectory(
        trajectory=trajectory,
        histories=[
            tr.TokenizedHistory(
                history=tokenized.history,
                model=tokenized.model,
                tokens=tokenized.tokens,
                logprobs=tokenized.logprobs,
                flags=tokenized.flags,
            )
        ],
    ).tensorize()
    group = tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory](
        trajectory_group=art.TrajectoryGroup([trajectory]),
        trajectories=[tokenized],
    ).tensorize()

    restored_trajectory = tr.compact_validate(
        tensorized.compact_dump(), type=tr.TensorizedTrajectory
    )
    restored_multi = tr.compact_validate(
        multi.compact_dump(), type=tr.TensorizedMultiHistoryTrajectory
    )
    restored_group = tr.compact_validate(
        group.compact_dump(),
        type=tr.TensorizedTrajectoryGroup[tr.TensorizedTrajectory],
    )

    assert restored_trajectory.trajectory.exchanges.chat_completions
    assert len(restored_multi.histories) == 1
    assert (
        restored_group.trajectories[0].trajectory
        is (restored_group.trajectory_group.trajectories[0])
    )
    assert restored_group.to("cpu") is restored_group


def test_tensorized_pickle_retains_sources_without_tokenized_intermediate() -> None:
    trajectory = art.Trajectory()
    tokenized = tr.TokenizedTrajectory(
        **_tokenized_history().model_dump(), trajectory=trajectory
    )
    tensorized = tokenized.tensorize()
    restored = pickle.loads(pickle.dumps(tensorized))

    assert isinstance(restored, tr.TensorizedTrajectory)
    assert restored.trajectory is not trajectory
    assert restored.history is not tokenized.history
    assert not hasattr(restored, "tokenized")
    assert torch.equal(restored.tokens, tensorized.tokens)


def test_tensorized_group_skips_equality_dump_for_identical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = art.Trajectory()
    tokenized = tr.TokenizedTrajectory(
        **_tokenized_history().model_dump(), trajectory=source
    )
    tensorized = tokenized.tensorize()

    def unexpected_dump(*_: object, **__: object) -> object:
        raise AssertionError("identical source trajectories must not be dumped")

    monkeypatch.setattr(art.Trajectory, "model_dump", unexpected_dump)
    group = tr.TensorizedTrajectoryGroup[tr.TensorizedTrajectory](
        trajectory_group=art.TrajectoryGroup([source]),
        trajectories=[tensorized],
    )

    assert group.trajectories[0].trajectory is source


def test_tensorized_group_pydantic_and_cloudpickle_round_trips() -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    trajectory = _trajectory()
    group = art.TrajectoryGroup(
        [trajectory], metrics={"batch": 1}, metadata={"name": "group"}
    ).tensorize()
    cls = tr.TensorizedTrajectoryGroup[tr.TensorizedTrajectory]

    restored = cls.model_validate_json(group.model_dump_json(warnings="error"))
    cloud_restored = cloudpickle.loads(cloudpickle.dumps(group))
    for value in (restored, cloud_restored):
        child = value.trajectories[0]
        assert child.trajectory is value.trajectory_group.trajectories[0]
        assert child.metrics is child.trajectory.metrics
        assert value.metrics is value.trajectory_group.metrics
        assert torch.equal(child.tokens, group.trajectories[0].tokens)
