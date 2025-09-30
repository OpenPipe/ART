import json
from typing import List, Union

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from smolagents import ChatMessage
from smolagents.models import get_clean_message_list, tool_role_conversions

Message = ChatCompletionMessageParam
MessagesAndChoices = List[Union[Message, Choice]]


def create_choice_from_message(msg: ChatMessage) -> Choice:
    """Convert a smolagents ChatMessage with token_usage to OpenAI Choice format."""
    tool_calls = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": json.dumps(tc.function.arguments)
                    if isinstance(tc.function.arguments, dict)
                    else tc.function.arguments,
                }
            })

    # Extract content - handle both str and list[dict] formats
    content = msg.content
    if isinstance(content, list):
        # Convert list format to string
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        content = "\n".join(text_parts) if text_parts else ""
    elif content is None:
        content = ""

    return Choice(
        message=ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        index=0,
        finish_reason="stop",
        logprobs=None,
    )


def convert_smolagents_input_messages(messages: List[ChatMessage]) -> list:
    """Convert input messages to OpenAI Message format.

    Returns a list of dicts compatible with OpenAI message format.
    """
    messages_for_clean: list = messages  # type: ignore

    # Use smolagents' utility to convert messages to clean dict format
    clean_messages = get_clean_message_list(
        messages_for_clean,
        role_conversions=tool_role_conversions,
        convert_images_to_image_urls=False,
        flatten_messages_as_text=False,
    )

    return clean_messages


def convert_smolagents_output_to_choice(output: ChatMessage) -> Choice:
    """Convert model output message to OpenAI Choice format.

    The output should be the result from Model.generate().
    """
    return create_choice_from_message(output)
