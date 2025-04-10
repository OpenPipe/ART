from ..types import Message


def format_message(message: Message) -> str:
    """Format a message into a readable string."""
    # Format the role and content
    role = message["role"].capitalize()
    content = message.get("content", message.get("refusal", "")) or ""

    # Format any tool calls
    tool_calls_text = "\n" if content else ""
    tool_calls_text += "\n".join(
        f"{tool_call['function']['name']}({tool_call['function']['arguments']})"
        for tool_call in message.get("tool_calls") or []
    )

    # Combine all parts
    formatted_message = f"{role}:\n{content}{tool_calls_text}"
    return formatted_message
