import importlib
from types import SimpleNamespace

import pytest

import art

ruler_module = importlib.import_module("art.rewards.ruler")


@pytest.fixture
def scored_by_ruler(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ruler(message_lists, **kwargs):
        return [
            SimpleNamespace(
                score=0.25 * (index + 1), explanation=f"explanation {index}"
            )
            for index in range(len(message_lists))
        ]

    monkeypatch.setattr(ruler_module, "ruler", fake_ruler)


def trajectory(identity: int) -> art.Trajectory:
    t = art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": f"question {identity}"},
            {"role": "assistant", "content": f"answer {identity}"},
        ],
        reward=float(identity),
        initial_policy_version=3,
        final_policy_version=5,
        metrics={"turns": 1},
        metadata={"identity": identity},
        logs=[f"rollout {identity}"],
    )
    t._policy_token_counts = {3: 2, 4: 1}
    return t


async def test_ruler_score_group_keeps_group_and_trajectory_provenance(
    scored_by_ruler: None,
) -> None:
    original = [trajectory(1), trajectory(2)]
    group = art.TrajectoryGroup(
        original,
        exceptions=[RuntimeError("third rollout failed")],
        metadata={"scenario_id": "s-7"},
        metrics={"group_latency": 1.5},
        logs=["group log"],
    )

    scored = await ruler_module.ruler_score_group(group, "judge")

    assert scored is not None
    assert [t.reward for t in scored.trajectories] == [0.25, 0.5]
    assert [t.metrics["independent_reward"] for t in scored.trajectories] == [1.0, 2.0]
    assert [e.message for e in scored.exceptions] == ["third rollout failed"]
    assert scored.metadata == {"scenario_id": "s-7"}
    assert scored.metrics == {"group_latency": 1.5}
    assert scored.logs == ["group log"]
    for before, after in zip(original, scored.trajectories, strict=True):
        assert after.initial_policy_version == 3
        assert after.final_policy_version == 5
        assert after._policy_token_counts == {3: 2, 4: 1}
        assert after.start_time == before.start_time
        assert after.metadata == before.metadata
    # The input group is left as it was.
    assert [t.reward for t in group.trajectories] == [1.0, 2.0]
    assert "ruler_score" not in group.trajectories[0].metrics
