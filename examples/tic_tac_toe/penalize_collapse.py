from typing import Dict, List
from art.trajectories import Trajectory, TrajectoryGroup
import math


def penalize_collapse(
    trajectory_group: TrajectoryGroup, max_penalty: float = -1, message_depth: int = 1
) -> None:
    """
    Applies penalties to trajectories that have common initial outputs.
    Common outputs get a large negative reward (penalty), while rare outputs get close to zero penalty.
    """
    output_to_trajectory_dict: Dict[str, List[Trajectory]] = {}

    for trajectory in trajectory_group.trajectories:
        output_key = ""
        assistant_messages = list(
            filter(lambda m: m["role"] == "assistant", trajectory.messages())
        )
        for i in range(message_depth):
            if len(assistant_messages) > i:
                output_key += assistant_messages[i]["content"]
            else:
                break
        if output_key not in output_to_trajectory_dict:
            output_to_trajectory_dict[output_key] = []
        output_to_trajectory_dict[output_key].append(trajectory)

    total_trajectories = len(trajectory_group.trajectories)

    print("Assigning collapse penalties")
    for output_key, trajectories in output_to_trajectory_dict.items():
        # Calculate frequency ratio (0 to 1)
        frequency_ratio = len(trajectories) / total_trajectories
        penalty = max_penalty * frequency_ratio**2

        print(output_key, len(trajectories), penalty)
        for trajectory in trajectories:
            trajectory.reward = trajectory.reward + penalty
