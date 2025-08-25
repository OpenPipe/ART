import json
from typing import List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

Message = ChatCompletionMessageParam
MessagesAndChoices = List[Union[Message, Choice]]

role_to_param = {
    "user": ChatCompletionUserMessageParam,
    "assistant": ChatCompletionAssistantMessageParam,
    "system": ChatCompletionSystemMessageParam,
    "tool": ChatCompletionToolMessageParam,
    "function": ChatCompletionFunctionMessageParam,
    "developer": ChatCompletionDeveloperMessageParam,
}


def make_message_param(role: str, **kwargs) -> ChatCompletionMessageParam:
    cls = role_to_param.get(role)
    if cls is None:
        raise ValueError(f"Unsupported role: {role}")
    return cls(**kwargs)


def langchain_msg_to_openai(msg: BaseMessage):
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    elif isinstance(msg, FunctionMessage):
        role = "function"
    else:
        raise TypeError(f"Unsupported LangChain message type: {type(msg)}")

    content = msg.content
    if not content:
        content = ""

    result = {"role": role, "content": content}

    # Handle tool calls or function call if present
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls

    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        result["tool_call_id"] = tool_call_id

    function_call = getattr(msg, "function_call", None)
    if function_call:
        result["function_call"] = function_call

    return result


def convert_langgraph_messages(messages: List[object]) -> MessagesAndChoices:
    converted = []

    for msg in messages:
        response_metadata = getattr(msg, "response_metadata")
        if response_metadata and "logprobs" in response_metadata:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call["function"] = {
                        "arguments": json.dumps(tool_call["args"]),
                        "name": tool_call["name"],
                    }
                    tool_call["type"] = "function"

            converted.append(
                Choice(
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=getattr(msg, "content"),
                        tool_calls=tool_calls,
                    ),
                    index=0,
                    **response_metadata,
                )
            )
        elif isinstance(msg, BaseMessage):
            converted.append(langchain_msg_to_openai(msg))
        elif isinstance(msg, dict):
            converted.append(msg)
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

    return converted
