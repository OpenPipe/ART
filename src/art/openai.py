from typing import Any, Callable, cast

from openai import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

from .preprocessing.policy_spans import POLICY_TOKEN_SPANS_KEY


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
            chat_completion = init_chat_completion(chunk)
        update_chat_completion(chat_completion, chunk)
        if on_chunk:
            try:
                on_chunk(chunk, chat_completion)
            except StopIteration:
                await stream.close()
                break
    assert chat_completion is not None
    return chat_completion


def consume_sync_chat_completion_stream(
    stream: Stream[ChatCompletionChunk],
) -> ChatCompletion:
    chat_completion: ChatCompletion | None = None
    for chunk in stream:
        if chat_completion is None:
            chat_completion = init_chat_completion(chunk)
        update_chat_completion(chat_completion, chunk)
    assert chat_completion is not None
    return chat_completion


def init_chat_completion(chunk: ChatCompletionChunk) -> ChatCompletion:
    return ChatCompletion(
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


def update_chat_completion(
    chat_completion: ChatCompletion, chunk: ChatCompletionChunk
) -> None:
    chat_completion_extra = cast(dict[str, Any], chat_completion.model_extra)
    prompt_token_ids = getattr(chunk, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        chat_completion_extra["prompt_token_ids"] = prompt_token_ids
    completion_prompt_token_ids = chat_completion_extra.get("prompt_token_ids")
    for choice, chunk_choice in zip(chat_completion.choices, chunk.choices):
        choice_extra = cast(dict[str, Any], choice.model_extra)
        if completion_prompt_token_ids is not None:
            choice_extra["prompt_token_ids"] = completion_prompt_token_ids
        token_ids = getattr(chunk_choice, "token_ids", None)
        if token_ids:
            choice_extra["token_ids"] = [
                *choice_extra.get("token_ids", []),
                *token_ids,
            ]
        policy_token_spans = getattr(chunk_choice, POLICY_TOKEN_SPANS_KEY, None)
        if policy_token_spans:
            choice_extra[POLICY_TOKEN_SPANS_KEY] = [
                *choice_extra.get(POLICY_TOKEN_SPANS_KEY, []),
                *policy_token_spans,
            ]
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
            for tool_call_delta in chunk_choice.delta.tool_calls:
                while tool_call_delta.index not in range(
                    len(choice.message.tool_calls)
                ):
                    choice.message.tool_calls.append(
                        ChatCompletionMessageFunctionToolCall(
                            id="",
                            function=Function(arguments="", name=""),
                            type="function",
                        )
                    )
                if tool_call_delta.id:
                    choice.message.tool_calls[
                        tool_call_delta.index
                    ].id = tool_call_delta.id
                if tool_call_delta.function:
                    tool_call = choice.message.tool_calls[tool_call_delta.index]
                    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                    if tool_call_delta.function.name:
                        tool_call.function.name = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_call.function.arguments += (
                            tool_call_delta.function.arguments
                        )
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
