from typing import Awaitable, Iterable

import asyncio
import tqdm

from art.gather import GatherContext, set_gather_context, wrap_group_awaitable
import art


async def gather_trajectory_groups_by_index(
    grouped_trajectory_awaitables: Iterable[Awaitable[tuple[art.Trajectory, ...]]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float = 0,
    trajectories_per_rollout: int = 1,
) -> list[art.TrajectoryGroup]:
    grouped_trajectory_awaitables = list(grouped_trajectory_awaitables)
    context = GatherContext(
        pbar=None,
        pbar_total_completion_tokens=pbar_total_completion_tokens,
        max_exceptions=max_exceptions,
    )

    with set_gather_context(context):
        future = asyncio.gather(
            *[wrap_group_awaitable(g) for g in grouped_trajectory_awaitables]
        )
        total = (
            sum(
                getattr(g, "_num_trajectories", 1)
                for g in grouped_trajectory_awaitables
            )
            * trajectories_per_rollout
        )
        context.pbar = tqdm.tqdm(desc=pbar_desc, total=total)
        all_trajectory_tuples = await future

    if context.pbar is not None:
        context.pbar.close()

    # Group trajectories by position in tuple
    num_groups = len(all_trajectory_tuples[0])
    grouped_trajectories: list[list[art.Trajectory]] = [[] for _ in range(num_groups)]

    for traj_tuple in all_trajectory_tuples:
        for i, traj in enumerate(traj_tuple):
            grouped_trajectories[i].append(traj)

    return [art.TrajectoryGroup(trajs) for trajs in grouped_trajectories]
