from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


def format_message(message: ChatCompletionMessageParam) -> str:
    """Format a message into a readable string."""
    # Format the role and content
    role = message["role"].capitalize()
    content = message.get("content", message.get("refusal", "")) or ""

    # Format any tool calls
    tool_calls = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_calls.append(f"{function.get('name')}({function.get('arguments')})")
    tool_calls_text = ("\n" if content else "") + "\n".join(tool_calls)

    # Combine all parts
    formatted_message = f"{role}:\n{content}{tool_calls_text}"
    return formatted_message
