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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("art_folder", "project_name", "model_name", "split"),
    [
        ("runs", "project", "model", "val"),
        ("researcher's runs", "project", "model", "val"),
        ("runs", "agent's benchmark", "model", "val"),
        ("runs", "project", "model's run", "val"),
        ("runs", "project", "model", "held-out's split"),
    ],
)
async def test_load_trajectories_quoted_paths(
    tmp_path, art_folder, project_name, model_name, split
):
    art_path = tmp_path / art_folder
    for name, reward in [(model_name, 0.75), ("control", 0.25)]:
        traj_dir = art_path / project_name / "models" / name / "trajectories" / split
        traj_dir.mkdir(parents=True)
        write_trajectory_groups_parquet(
            [
                TrajectoryGroup(
                    trajectories=[
                        Trajectory(
                            reward=reward,
                            metrics={"success": reward},
                            messages_and_choices=[{"role": "user", "content": "hi"}],
                        )
                    ]
                )
            ],
            traj_dir / "0003.parquet",
        )

    df = await load_trajectories(project_name, art_path=str(art_path))

    assert df.height == 2
    assert df.sort("model").select(
        "model", "split", "step", "reward", "metric_success"
    ).to_dicts() == [
        {
            "model": name,
            "split": split,
            "step": 3,
            "reward": reward,
            "metric_success": reward,
        }
        for name, reward in sorted([(model_name, 0.75), ("control", 0.25)])
    ]
