from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import art
from art.distributed import PackingRequest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "multinode"


def test_packing_request_from_public_groups() -> None:
    model = art.TrainableModel(
        name="packing-public-api", project="test", base_model="not-loaded"
    )
    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "Respond with maybe."},
                    {"role": "assistant", "content": "maybe"},
                ],
                reward=1.0,
                initial_policy_version=3,
            )
        ],
        metadata={"split": "smoke"},
    )

    request = PackingRequest.from_groups(
        model,
        [group],
        packed_sequence_length=128,
        allow_training_without_logprobs=True,
        group_ids=("maybe",),
        min_source_version=3,
        max_source_version=3,
    )

    assert request.model.build().base_model == "not-loaded"
    assert request.trajectory_groups[0].build().model_dump(mode="json") == (
        group.model_dump(mode="json")
    )
    assert request.group_ids == ("maybe",)
    assert request.min_source_version == request.max_source_version == 3


def test_distributed_package_import_is_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import art.distributed as distributed
assert "art.distributed.art_runtime" not in sys.modules
from art.distributed import ArtRuntime, ClusterSpec, PackingRequest, compile_topology
assert all(value is not None for value in (
    ArtRuntime, ClusterSpec, PackingRequest, compile_topology
))
assert "monarch" not in sys.modules
assert "PackingRequest" in distributed.__all__
""",
        ],
        check=True,
    )


def test_documented_rollout_is_installed_and_bounded() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(EXAMPLE_DIR), environment.get("PYTHONPATH", ""))
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import art
from art.distributed import InstalledAsyncCallable
import program

async def check():
    reference = InstalledAsyncCallable.from_callable(program.rollout)
    assert (reference.module, reference.qualname) == ("program", "rollout")
    model = art.TrainableModel(
        name="documented-rollout", project="test", base_model="not-loaded"
    )
    trajectory = await program.rollout(model, "maybe", None)
    assert trajectory.reward == 1.0
    assert trajectory.metadata["answer"] == "maybe"

asyncio.run(check())
""",
        ],
        check=True,
        env=environment,
    )
