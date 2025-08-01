from typing import List, Union
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    FunctionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
import json

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


def langchain_msg_to_openai(msg: BaseMessage) -> Message:
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

    result = {"role": role, "content": content}

    # Handle tool calls or function call if present
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "function_call") and msg.function_call:
        result["function_call"] = msg.function_call

    return result


def convert_langgraph_messages(messages: List[object]) -> MessagesAndChoices:
    converted: MessagesAndChoices = []

    for msg in messages:
        if hasattr(msg, "response_metadata") and "logprobs" in msg.response_metadata:
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call["function"] = {
                        "arguments": json.dumps(tool_call["args"]),
                        "name": tool_call["name"],
                    }
                    tool_call["type"] = "function"

            converted.append(
                Choice(
                    message=ChatCompletionAssistantMessageParam(
                        role="assistant", content=msg.content, tool_calls=msg.tool_calls
                    ),
                    index=0,
                    **msg.response_metadata,
                )
            )
        elif isinstance(msg, BaseMessage):
            converted.append(langchain_msg_to_openai(msg))
        elif isinstance(msg, dict):
            converted.append(msg)
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

    return converted
