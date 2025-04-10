from typing import Callable, Coroutine, Any

import art
from art.types import Trajectory
from openai import AsyncOpenAI


async def benchmark_rollout(
    client: AsyncOpenAI,
    model: str,
    num_rollouts: int,
    rollout: Callable[[AsyncOpenAI, str, int, bool], Coroutine[Any, Any, Trajectory]],
) -> float:
    trajectory_groups = await art.gather_trajectories(
        (rollout(client, model, i, False) for i in range(num_rollouts)),
        pbar_desc="Benchmarking rollout",
    )

    trajectory_group_rewards = []

    for group in trajectory_groups:
        total_reward = sum(trajectory.reward for trajectory in group)
        trajectory_group_rewards.append(total_reward / len(group))

    average_reward = sum(trajectory_group_rewards) / len(trajectory_group_rewards)

    print(f"Average reward for {model}: {average_reward}")

    return average_reward
