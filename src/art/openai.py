import openai
from openai import AsyncStream
from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import Any, Callable

from .gather import get_gather_context


def patch_openai(client: openai.AsyncOpenAI) -> openai.AsyncOpenAI:
    create = client.chat.completions.create

    def report_usage(chat_completion: ChatCompletion) -> None:
        context = get_gather_context()
        if chat_completion.usage is not None:
            context.metric_sums["prompt_tokens"] += chat_completion.usage.prompt_tokens
            context.metric_divisors["prompt_tokens"] += 1

    async def create_patched(*args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream:
        return_stream = kwargs.get("stream", False)
        context = get_gather_context()
        if context.pbar_total_completion_tokens:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return_value = await create(*args, **kwargs)
        if isinstance(return_value, ChatCompletion):
            report_usage(return_value)
            return return_value
        assert isinstance(return_value, AsyncStream)
        if return_stream:
            return return_value

        def on_chunk(chunk: ChatCompletionChunk, _: ChatCompletion) -> None:
            context = get_gather_context()
            if context.pbar_total_completion_tokens:
                context.metric_sums["total_completion_tokens"] += sum(
                    len(choice.logprobs.content or choice.logprobs.refusal or [])
                    for choice in chunk.choices
                    if choice.logprobs
                    and (choice.logprobs.content or choice.logprobs.refusal)
                )
                context.update_pbar(n=0)

        chat_completion = await consume_chat_completion_stream(return_value, on_chunk)
        report_usage(chat_completion)
        return chat_completion

    client.chat.completions.create = create_patched  # type: ignore
    return client


async def consume_chat_completion_stream(
    stream: AsyncStream[ChatCompletionChunk],
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], Any] | None = None,
) -> ChatCompletion:
    """Consume a chat completion stream and build a complete ChatCompletion object.

    This function processes a stream of ChatCompletionChunks, constructing a complete
    ChatCompletion object as if it was returned from a non-streaming API call.
    Works with any OpenAI-compatible API implementation.

    Args:
        stream: An AsyncStream of ChatCompletionChunk objects.
        on_chunk: Optional callback that receives each chunk and the current state of the
            ChatCompletion. If the callback raises StopIteration, the stream will close early.

    Returns:
        A complete ChatCompletion object built from the streamed chunks.

    Raises:
        AssertionError: If no chat completion object could be created.
    """
    chat_completion: ChatCompletion | None = None
    async for chunk in stream:
        if chat_completion is None:
            chat_completion = ChatCompletion(
                id=chunk.id,
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=choice.index,
                        logprobs=(ChoiceLogprobs() if choice.logprobs else None),
                        message=ChatCompletionMessage(role="assistant"),
                    )
                    for choice in chunk.choices
                ],
                created=chunk.created,
                model=chunk.model,
                object="chat.completion",
            )
        for choice, chunk_choice in zip(chat_completion.choices, chunk.choices):
            choice.finish_reason = chunk_choice.finish_reason or "stop"
            if chunk_choice.logprobs:
                if choice.logprobs is None:
                    choice.logprobs = ChoiceLogprobs()
                if chunk_choice.logprobs.content:
                    if choice.logprobs.content is None:
                        choice.logprobs.content = []
                    choice.logprobs.content.extend(chunk_choice.logprobs.content)
                if chunk_choice.logprobs.refusal:
                    if choice.logprobs.refusal is None:
                        choice.logprobs.refusal = []
                    choice.logprobs.refusal.extend(chunk_choice.logprobs.refusal)
            if chunk_choice.delta.content:
                if choice.message.content is None:
                    choice.message.content = ""
                choice.message.content += chunk_choice.delta.content
            if chunk_choice.delta.refusal:
                if choice.message.refusal is None:
                    choice.message.refusal = ""
                choice.message.refusal += chunk_choice.delta.refusal
            if chunk_choice.delta.function_call:
                if choice.message.function_call is None:
                    choice.message.function_call = FunctionCall(arguments="", name="")
                choice.message.function_call.name += (
                    chunk_choice.delta.function_call.name or ""
                )
                choice.message.function_call.arguments += (
                    chunk_choice.delta.function_call.arguments or ""
                )
            if chunk_choice.delta.tool_calls:
                if choice.message.tool_calls is None:
                    choice.message.tool_calls = []
                for tool_call in chunk_choice.delta.tool_calls:
                    while not tool_call.index in range(len(choice.message.tool_calls)):
                        choice.message.tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id="",
                                function=Function(arguments="", name=""),
                                type="function",
                            )
                        )
                    if tool_call.id:
                        choice.message.tool_calls[tool_call.index].id = tool_call.id
                    if tool_call.function:
                        if tool_call.function.name:
                            choice.message.tool_calls[tool_call.index].function.name = (
                                tool_call.function.name
                            )
                        if tool_call.function.arguments:
                            choice.message.tool_calls[
                                tool_call.index
                            ].function.arguments += tool_call.function.arguments
            if getattr(chunk_choice.delta, "reasoning", None):
                if not hasattr(choice.message, "reasoning"):
                    setattr(choice.message, "reasoning", "")
                setattr(
                    choice.message,
                    "reasoning",
                    getattr(choice.message, "reasoning")
                    + getattr(chunk_choice.delta, "reasoning"),
                )
        chat_completion.service_tier = chunk.service_tier
        chat_completion.system_fingerprint = chunk.system_fingerprint
        chat_completion.usage = chunk.usage
        if on_chunk:
            try:
                on_chunk(chunk, chat_completion)
            except StopIteration:
                await stream.close()
                break
    assert chat_completion is not None
    return chat_completion
