from __future__ import annotations

import asyncio
import os
import socket

import art
from art.distributed import (
    ArtLaunchContext,
    ArtRuntime,
    InstalledAsyncCallable,
    compile_topology,
)

REWARDS = {"yes": 0.5, "no": 0.75, "maybe": 1.0}


async def rollout(
    _model: art.TrainableModel, answer: str, _config: None
) -> art.Trajectory:
    messages: art.MessagesAndChoices = [
        {"role": "user", "content": f"Respond with {answer}."},
        {"role": "assistant", "content": answer},
    ]
    return art.Trajectory(
        messages_and_choices=messages,
        reward=REWARDS[answer],
        metadata={
            "answer": answer,
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
        },
    )


async def main(launch: ArtLaunchContext) -> None:
    host_count = launch.host_count
    runtime = await ArtRuntime.start(
        launch.host_mesh,
        compile_topology(
            cluster=launch.homogeneous_cluster(
                cpu_slots=1,
                startup_timeout_s=90,
                rpc_timeout_s=30,
            )
        ),
    )
    try:
        workers = tuple(range(host_count))
        executor = runtime.rollout_executor(
            InstalledAsyncCallable.from_callable(rollout),
            target_workers=host_count,
        )
        executor.set_workers(workers)
        model = art.TrainableModel(
            name="multinode-smoke",
            project="art",
            base_model="not-loaded",
            run_name="multinode-smoke",
        )
        trajectories: list[art.Trajectory] = []
        for answer in REWARDS:
            trajectories.extend(
                await asyncio.gather(
                    *(
                        executor.run(worker, rollout, model, answer, None)
                        for worker in workers
                    )
                )
            )
        answers = [str(trajectory.metadata["answer"]) for trajectory in trajectories]
        expected = [answer for answer in REWARDS for _ in workers]
        placements = {
            (trajectory.metadata["hostname"], trajectory.metadata["process_id"])
            for trajectory in trajectories
        }
        if answers != expected or len(placements) != host_count:
            raise RuntimeError(
                f"distributed rollout mismatch: answers={answers}, placements={placements}"
            )
        print(
            f"ART_MULTINODE_SMOKE_PASS hosts={host_count} answers={answers}",
            flush=True,
        )
    finally:
        await runtime.close()
