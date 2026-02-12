from litellm.types.utils import (
    ChoiceLogprobs as LitellmChoiceLogprobs,
)
from litellm.types.utils import (
    Choices,
    StreamingChoices,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)


def convert_litellm_choice_to_openai(
    litellm_choice: Choices | StreamingChoices,
) -> Choice:
    assert isinstance(litellm_choice, Choices), (
        "Only non-streaming choices are currently supported"
    )
    litellm_message = litellm_choice.message
    assert litellm_message.role == "assistant", "Only assistant messages are supported"

    # Convert tool calls if they exist
    openai_tool_calls = None
    if litellm_message.tool_calls:
        openai_tool_calls = []
        for tool_call in litellm_message.tool_calls:
            assert tool_call.type == "function", (
                "Only function tool calls are supported"
            )
            assert tool_call.function.name is not None, (
                "Tool call function name must not be None"
            )
            openai_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.id,
                    type=tool_call.type,  # ty:ignore[invalid-argument-type]
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )

    openai_message = ChatCompletionMessage(
        content=litellm_message.content,
        role=litellm_message.role,
        tool_calls=openai_tool_calls,
    )

    # Convert logprobs if they exist
    openai_logprobs = None
    if hasattr(litellm_choice, "logprobs") and litellm_choice.logprobs is not None:
        assert litellm_choice.logprobs.content is not None
        assert isinstance(litellm_choice.logprobs, LitellmChoiceLogprobs)

        converted_logprobs: list[ChatCompletionTokenLogprob] = []
        for logprob in litellm_choice.logprobs.content:
            top_logprobs = [
                TopLogprob(
                    token=logprob.token,
                    logprob=logprob.logprob,
                    bytes=logprob.bytes,
                )
                for logprob in logprob.top_logprobs
            ]
            converted_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=logprob.token,
                    bytes=logprob.bytes,
                    logprob=logprob.logprob,
                    top_logprobs=top_logprobs,
                )
            )
        openai_logprobs = ChoiceLogprobs(content=converted_logprobs)

    # Create the OpenAI Choice object
    openai_choice = Choice(
        message=openai_message,
        finish_reason=litellm_choice.finish_reason,  # type: ignore
        index=litellm_choice.index,
        logprobs=openai_logprobs,
    )

    return openai_choice
