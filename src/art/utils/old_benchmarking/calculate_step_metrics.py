import numpy as np

from art.trajectories import TrajectoryGroup


# calculate the average standard deviation of rewards within groups
def calculate_step_std_dev(trajectory_groups: list[TrajectoryGroup]) -> float:
    std_devs = []
    for group in trajectory_groups:
        group_rewards = []

        for trajectory in group.trajectories:
            if isinstance(trajectory, BaseException):
                continue
            group_rewards.append(trajectory.reward)

        if len(group_rewards) > 1:
            std_devs.append(np.std(group_rewards))

    if len(std_devs) == 0:
        return 0

    return sum(std_devs) / len(std_devs)
