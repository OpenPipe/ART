from typing import List
from art.types import MessagesAndChoices, Message


def to_messages(
    messages_and_choices: MessagesAndChoices,
) -> List[Message]:
    messages = []
    for message in messages_and_choices:
        if isinstance(message, dict):
            if "message" in message:
                messages.append(message["message"])
            else:
                messages.append(message)
        else:
            messages.append(message.message)
    return messages
