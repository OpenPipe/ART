"""Monkey patches and modifications for vLLM."""

from typing import Any


def subclass_chat_completion_request() -> None:
    """
    Subclass ChatCompletionRequest so that logprobs are always returned.
    """
    import vllm.entrypoints.openai.protocol

    class ChatCompletionRequest(vllm.entrypoints.openai.protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0

    vllm.entrypoints.openai.protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]


def patch_listen_for_disconnect() -> None:
    async def patched_listen_for_disconnect(request):
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    break
        except UnboundLocalError:
            pass

    # Replace the original function
    import vllm.entrypoints.utils

    vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect  # ty:ignore[invalid-assignment]


def patch_tool_parser_manager() -> None:
    """
    Patch ToolParserManager to support streaming tool call logprobs.
    """
    from vllm.entrypoints.openai.protocol import DeltaMessage
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    get_tool_parser = ToolParserManager.get_tool_parser

    def patched_get_tool_parser(name: str) -> type:
        tool_parser_class = get_tool_parser(name)
        original = tool_parser_class.extract_tool_calls_streaming

        def patch(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return original(*args, **kwargs) or DeltaMessage()

        tool_parser_class.extract_tool_calls_streaming = patch  # ty:ignore[invalid-assignment]
        return tool_parser_class

    ToolParserManager.get_tool_parser = patched_get_tool_parser  # ty:ignore[invalid-assignment]
