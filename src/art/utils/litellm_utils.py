from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from litellm.types.utils import Choices, ChoiceLogprobs as LitellmChoiceLogprobs


def convert_litellm_choice_to_openai(litellm_choice: Choices) -> Choice:
    litellm_message = litellm_choice.message
    assert litellm_message.role == "assistant", "Only assistant messages are supported"

    assert litellm_message.tool_calls is None, (
        "The conversion function doesn't yet support tool calls"
    )

    openai_message = ChatCompletionMessage(
        content=litellm_message.content,
        role=litellm_message.role,
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
