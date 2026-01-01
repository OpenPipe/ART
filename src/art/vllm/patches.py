"""Monkey patches and modifications for vLLM."""

from typing import Any


def subclass_chat_completion_request() -> None:
    """
    Subclass ChatCompletionRequest so that logprobs are always returned.
    """
    import vllm.entrypoints.openai.protocol

    class ChatCompletionRequest(vllm.entrypoints.openai.protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0

    vllm.entrypoints.openai.protocol.ChatCompletionRequest = ChatCompletionRequest


def patch_lora_request() -> None:
    """
    Patches the vLLM LoRARequest type to have attributes Unsloth expects and the Unsloth LoRARequest type to have attributes vLLM expects.
    """
    from unsloth_zoo.vllm_lora_request import LoRARequest as UnslothLoRARequest
    from vllm.lora.request import LoRARequest

    LoRARequest.lora_tensors = {}  # type: ignore
    LoRARequest.lora_embeddings = {}  # type: ignore
    UnslothLoRARequest.tensorizer_config_dict = None  # type: ignore


def patch_get_lora_tokenizer_async() -> None:
    import vllm.transformers_utils.tokenizer
    import vllm.transformers_utils.tokenizer_group

    async def patch(*_: Any, **__: Any) -> None:
        return None

    vllm.transformers_utils.tokenizer.get_lora_tokenizer_async = patch  # type: ignore
    vllm.transformers_utils.tokenizer_group.get_lora_tokenizer_async = (  # type: ignore
        patch
    )

    async def patch2(self, *args: Any, **kwargs: Any) -> None:
        return self.tokenizer

    vllm.transformers_utils.tokenizer_group.TokenizerGroup.get_lora_tokenizer_async = (
        patch2  # type: ignore
    )


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

    vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect


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

        tool_parser_class.extract_tool_calls_streaming = patch
        return tool_parser_class

    ToolParserManager.get_tool_parser = patched_get_tool_parser
