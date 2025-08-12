import json
from typing import Any, cast

import yaml

from art import Trajectory, TrajectoryGroup
from art.trajectories import History
from art.types import Choice, Message, MessageOrChoice


# serialize trajectory groups to a jsonl string
def serialize_trajectory_groups(trajectory_groups: list[TrajectoryGroup]) -> str:
    group_dicts = [
        trajectory_group_to_dict(trajectory_group)
        for trajectory_group in trajectory_groups
    ]

    return "\n".join(json.dumps(group_dict) for group_dict in group_dicts)


def trajectory_group_to_dict(trajectory_group: TrajectoryGroup) -> dict[str, Any]:
    trajectory_dicts = []
    for trajectory in trajectory_group.trajectories:
        if not isinstance(trajectory, Trajectory):
            # remove exceptions
            continue
        trajectory_dicts.append(trajectory_to_dict(trajectory))

    return {
        "trajectories": trajectory_dicts,
    }


def history_to_dict(history: History) -> dict[str, Any]:
    messages_and_choices = [
        message_or_choice_to_dict(message_or_choice)
        for message_or_choice in history.messages_and_choices
    ]

    return {"messages_and_choices": messages_and_choices, "tools": history.tools}


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    messages_and_choices = [
        message_or_choice_to_dict(message_or_choice)
        for message_or_choice in trajectory.messages_and_choices
    ]

    return {
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "metadata": trajectory.metadata,
        "messages_and_choices": messages_and_choices,
        "tools": trajectory.tools,
        "additional_histories": (
            [history_to_dict(h) for h in trajectory.additional_histories]
            if trajectory.additional_histories
            else trajectory.additional_histories
        ),
        "logs": trajectory.logs,
    }


def message_or_choice_to_dict(message_or_choice: MessageOrChoice) -> dict[str, Any]:
    # messages are sometimes stored as dicts, so we need to handle both cases
    item_dict = (
        message_or_choice
        if isinstance(message_or_choice, dict)
        else message_or_choice.to_dict()
    )

    if "logprobs" in item_dict:
        # item is a choice with logprobs, remove the logprobs
        item_dict.pop("logprobs")

    return dict(item_dict)


def deserialize_trajectory_groups(serialized: str) -> list[TrajectoryGroup]:
    # Try to parse as JSONL first (new format)
    try:
        loaded_groups = [
            json.loads(line) for line in serialized.strip().split("\n") if line
        ]
    except json.JSONDecodeError:
        # Fall back to YAML parsing (old format)
        loaded_groups = yaml.load(serialized, Loader=yaml.SafeLoader)
    return [dict_to_trajectory_group(group) for group in loaded_groups]


def dict_to_trajectory_group(dict: dict[str, Any]) -> TrajectoryGroup:
    return TrajectoryGroup(
        trajectories=[
            dict_to_trajectory(trajectory) for trajectory in dict["trajectories"]
        ],
        exceptions=[],
    )


def dict_to_trajectory(dict: dict[str, Any]) -> Trajectory:
    return Trajectory(
        messages_and_choices=[
            dict_to_message_or_choice(message_or_choice)
            for message_or_choice in dict["messages_and_choices"]
        ],
        reward=dict["reward"],
        metrics=dict["metrics"],
        metadata=dict["metadata"],
        logs=dict["logs"],
    )


def dict_to_message_or_choice(dict: dict[str, Any]) -> MessageOrChoice:
    if "message" in dict:
        return Choice(**dict)
    else:
        return cast(Message, dict)
