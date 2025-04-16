import yaml
from typing import Any

from art import TrajectoryGroup, Trajectory
from art.types import Choice, Message, MessageOrChoice
from openai.types.chat.chat_completion_message_param import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

# serialize trajectory groups to a yaml string
def serialize_trajectory_groups(trajectory_groups: list[TrajectoryGroup]) -> str:
    group_dicts = [trajectory_group_to_dict(trajectory_group) for trajectory_group in trajectory_groups]

    return yaml.dump(group_dicts)

def trajectory_group_to_dict(trajectory_group: TrajectoryGroup) -> dict[str, Any]:

    trajectory_dicts = []
    for trajectory in trajectory_group.trajectories:
        if isinstance(trajectory, BaseException):
            continue
        trajectory_dicts.append(trajectory_to_dict(trajectory))

    return {
        "trajectories": trajectory_dicts,
        "metadata": trajectory_group.metadata,
        "exceptions": trajectory_group.exceptions,
    }

def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    messages_and_choices = [message_or_choice_to_dict(message_or_choice) for message_or_choice in trajectory.messages_and_choices]

    return {
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "metadata": trajectory.metadata,
        "messages_and_choices": messages_and_choices,
        "logs": trajectory.logs,
    }

def message_or_choice_to_dict(message_or_choice: Message | Choice) -> dict[str, Any]:
    # messages are sometimes stored as dicts, so we need to handle both cases
    item_dict = message_or_choice if isinstance(message_or_choice, dict) else message_or_choice.to_dict()

    if "message" in item_dict:
        # item is a choice, remove the logprobs
        item_dict.pop("logprobs")
        return item_dict
    else:
        # item is a message, return the item as is
        return item_dict

def deserialize_trajectory_groups(serialized: str) -> list[TrajectoryGroup]:
    loaded_groups = yaml.load(serialized, Loader=yaml.SafeLoader)
    return [dict_to_trajectory_group(group) for group in loaded_groups]

def dict_to_trajectory_group(dict: dict[str, Any]) -> TrajectoryGroup:
    return TrajectoryGroup(
        trajectories=[dict_to_trajectory(trajectory) for trajectory in dict["trajectories"]],
        metadata=dict["metadata"],
        exceptions=dict["exceptions"],
    )

def dict_to_trajectory(dict: dict[str, Any]) -> Trajectory:
    return Trajectory(
        messages_and_choices=[dict_to_message_or_choice(message_or_choice) for message_or_choice in dict["messages_and_choices"]],
        reward=dict["reward"],
        metrics=dict["metrics"],
        metadata=dict["metadata"],
        logs=dict["logs"],
    )



def dict_to_message_or_choice(dict: dict[str, Any]) -> MessageOrChoice:
    if "message" in dict:
        return Choice(**dict)
    else:
        # cannot be instantiated as just "Message" because it is a union type alias
        if dict["role"] == "system":
            return ChatCompletionSystemMessageParam(**dict)
        elif dict["role"] == "user":
            return ChatCompletionUserMessageParam(**dict)
        elif dict["role"] == "assistant":
            return ChatCompletionAssistantMessageParam(**dict)
        else:
            raise ValueError(f"Unknown message role: {dict['role']}")
