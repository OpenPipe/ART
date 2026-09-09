from typing import Any, Callable, cast

from openai import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChatCompletionChunkChoice,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

from .preprocessing.policy_spans import POLICY_TOKEN_SPANS_KEY

ART_MOE_ROUTING_METADATA_KEY = "art_moe_routing"


class IncompleteChatCompletionStreamError(ValueError):
    pass


async def consume_chat_completion_stream(
    stream: AsyncStream[ChatCompletionChunk],
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], Any] | None = None,
    *,
    require_usage: bool = False,
) -> ChatCompletion:
    """Consume a chat completion stream and build a complete ChatCompletion object.

    This function processes a stream of ChatCompletionChunks, constructing a complete
    ChatCompletion object as if it was returned from a non-streaming API call.
    Works with any OpenAI-compatible API implementation.

    Args:
        stream: An AsyncStream of ChatCompletionChunk objects.
        on_chunk: Optional callback that receives each chunk and the current state of the
            ChatCompletion. If the callback raises StopIteration, the stream will close early.
        require_usage: Reject a completed stream without a usage trailer.

    Returns:
        A complete ChatCompletion object built from the streamed chunks.

    Raises:
        ValueError: If the stream has no choices or ends before a choice is terminal.
    """
    chat_completion: ChatCompletion | None = None
    terminal_choices: set[int] = set()
    stopped_early = False
    try:
        async for chunk in stream:
            if _is_empty_stream_prologue(chunk):
                continue
            if chat_completion is None:
                chat_completion = init_chat_completion(chunk)
            update_chat_completion(chat_completion, chunk)
            terminal_choices.update(
                choice.index
                for choice in chunk.choices
                if choice.finish_reason is not None
            )
            if on_chunk:
                try:
                    on_chunk(chunk, chat_completion)
                except StopIteration:
                    await stream.close()
                    stopped_early = True
                    break
        return _validate_and_finalize_chat_completion(
            chat_completion,
            terminal_choices,
            allow_incomplete=stopped_early,
            require_usage=require_usage,
        )
    except BaseException:
        await stream.close()
        raise


def _is_empty_stream_prologue(chunk: ChatCompletionChunk) -> bool:
    return (
        not chunk.choices
        and chunk.id == ""
        and chunk.object == ""
        and chunk.model == ""
        and chunk.usage is None
    )


def consume_sync_chat_completion_stream(
    stream: Stream[ChatCompletionChunk],
) -> ChatCompletion:
    chat_completion: ChatCompletion | None = None
    terminal_choices: set[int] = set()
    try:
        for chunk in stream:
            if _is_empty_stream_prologue(chunk):
                continue
            if chat_completion is None:
                chat_completion = init_chat_completion(chunk)
            update_chat_completion(chat_completion, chunk)
            terminal_choices.update(
                choice.index
                for choice in chunk.choices
                if choice.finish_reason is not None
            )
        return _validate_and_finalize_chat_completion(
            chat_completion,
            terminal_choices,
        )
    except BaseException:
        stream.close()
        raise


def _validate_and_finalize_chat_completion(
    chat_completion: ChatCompletion | None,
    terminal_choices: set[int],
    *,
    allow_incomplete: bool = False,
    require_usage: bool = False,
) -> ChatCompletion:
    if chat_completion is None or not chat_completion.choices:
        raise IncompleteChatCompletionStreamError(
            "Chat Completions stream returned no choices"
        )
    if not allow_incomplete:
        missing = {
            choice.index for choice in chat_completion.choices
        } - terminal_choices
        if missing:
            raise IncompleteChatCompletionStreamError(
                "Chat Completions stream ended before choices "
                f"{sorted(missing)} became terminal"
            )
    if require_usage and chat_completion.usage is None:
        raise IncompleteChatCompletionStreamError(
            "Chat Completions stream ended before its usage trailer"
        )
    return finalize_chat_completion(chat_completion)


def init_chat_completion(chunk: ChatCompletionChunk) -> ChatCompletion:
    return ChatCompletion(
        id=chunk.id,
        choices=[_init_choice(choice) for choice in chunk.choices],
        created=chunk.created,
        model=chunk.model,
        object="chat.completion",
    )


def _init_choice(chunk_choice: ChatCompletionChunkChoice) -> Choice:
    return Choice(
        finish_reason=chunk_choice.finish_reason or "stop",
        index=chunk_choice.index,
        logprobs=(ChoiceLogprobs() if chunk_choice.logprobs else None),
        message=ChatCompletionMessage(role="assistant"),
    )


def finalize_chat_completion(chat_completion: ChatCompletion) -> ChatCompletion:
    prompt_token_ids = (chat_completion.model_extra or {}).get("prompt_token_ids")
    if prompt_token_ids is not None:
        for choice in chat_completion.choices:
            cast(dict[str, Any], choice.model_extra)["prompt_token_ids"] = (
                prompt_token_ids
            )
    return chat_completion


def update_chat_completion(
    chat_completion: ChatCompletion, chunk: ChatCompletionChunk
) -> None:
    chat_completion_extra = cast(dict[str, Any], chat_completion.model_extra)
    prompt_token_ids = getattr(chunk, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        chat_completion_extra["prompt_token_ids"] = prompt_token_ids
    completion_prompt_token_ids = chat_completion_extra.get("prompt_token_ids")
    choices = {choice.index: choice for choice in chat_completion.choices}
    if completion_prompt_token_ids is not None:
        for choice in choices.values():
            cast(dict[str, Any], choice.model_extra)["prompt_token_ids"] = (
                completion_prompt_token_ids
            )
    for chunk_choice in chunk.choices:
        choice = choices.get(chunk_choice.index)
        if choice is None:
            choice = _init_choice(chunk_choice)
            choices[choice.index] = choice
            chat_completion.choices.append(choice)
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
        if chunk_choice.finish_reason is not None:
            choice.finish_reason = chunk_choice.finish_reason
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
        if chunk_choice.delta.content is not None:
            if choice.message.content is None:
                choice.message.content = ""
            choice.message.content += chunk_choice.delta.content
        if chunk_choice.delta.refusal is not None:
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
                if tool_call_delta.index < 0:
                    raise ValueError(
                        "Tool call stream index must be non-negative, got "
                        f"{tool_call_delta.index}"
                    )
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
                    ].id += tool_call_delta.id
                if tool_call_delta.function:
                    tool_call = choice.message.tool_calls[tool_call_delta.index]
                    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                    if tool_call_delta.function.name:
                        tool_call.function.name += tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_call.function.arguments += (
                            tool_call_delta.function.arguments
                        )
        for field in ("reasoning", "reasoning_content"):
            value = getattr(chunk_choice.delta, field, None)
            if value is None:
                continue
            setattr(
                choice.message,
                field,
                (getattr(choice.message, field, None) or "") + value,
            )
    chat_completion.choices.sort(key=lambda choice: choice.index)
    if chunk.service_tier is not None:
        chat_completion.service_tier = chunk.service_tier
    if chunk.system_fingerprint is not None:
        chat_completion.system_fingerprint = chunk.system_fingerprint
    if chunk.usage is not None:
        chat_completion.usage = chunk.usage
