import pytest

from art import Trajectory, TrajectoryGroup
from art.utils.benchmarking.load_trajectories import load_trajectories
from art.utils.trajectory_logging import write_trajectory_groups_parquet


@pytest.mark.asyncio
async def test_load_trajectories_group_columns(tmp_path):
    project_name = "proj"
    model_name = "model"
    traj_dir = tmp_path / project_name / "models" / model_name / "trajectories" / "val"
    traj_dir.mkdir(parents=True)

    groups = [
        TrajectoryGroup(
            trajectories=[
                Trajectory(
                    reward=1.0,
                    messages_and_choices=[{"role": "user", "content": "hi"}],
                )
            ],
            metadata={"scenario_id": "abc"},
            metrics={"judge_score": 0.9},
            logs=["group log"],
            exceptions=[],
        )
    ]
    write_trajectory_groups_parquet(groups, traj_dir / "0000.parquet")

    df = await load_trajectories(
        project_name=project_name,
        models=[model_name],
        art_path=str(tmp_path),
    )

    assert "group_metric_judge_score" in df.columns
    assert "group_metadata_scenario_id" in df.columns
    assert df["group_metric_judge_score"][0] == 0.9
    assert df["group_metadata_scenario_id"][0] == "abc"
