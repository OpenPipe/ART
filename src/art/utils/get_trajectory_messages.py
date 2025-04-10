from ..trajectories import Trajectory
from ..types import Messages


def get_trajectory_messages(trajectory: Trajectory) -> Messages:
    messages: Messages = []
    for item in trajectory.messages_and_choices:

        # if item is not a dict, convert it to a dict
        if not isinstance(item, dict):
            item = item.to_dict()

        # check if item is a choice
        if "message" in item:
            messages.append(
                {"role": "assistant", "content": item["message"]["content"]}
            )
        else:
            # otherwise it's a message
            messages.append(item)
    return messages
