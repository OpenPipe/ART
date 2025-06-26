"""Monkey patches and modifications for vLLM."""

import ctypes
import torch
from typing import Any
from vllm.worker.multi_step_model_runner import MultiStepModelRunner


def patch_allocator() -> None:
    """
    Patch the vLLM CuMemAllocator to specifically focus on offloading/discarding
    the KV cache.
    """
    from vllm.device_allocator.cumem import (
        create_and_map,
        CuMemAllocator,
        libcudart,
        unmap_and_release,
    )
    from vllm.utils import is_pin_memory_available

    allocator = CuMemAllocator.get_instance()

    def sleep(offload_tags: tuple[str, ...] | str | None = None) -> None:
        # In this version of vLLM (0.7.3) one tag is provided for sleep level 1
        # and no tags are provided for sleep level 2, so we can reverse-engineer
        # the sleep level from the tags
        sleep_level = 1 if offload_tags else 2
        # We reinterpret the sleep levels as follows:
        # Sleep level 1: offload kv cache to CPU memory (or disk)
        if sleep_level == 1:
            offload_to = "cpu"
            # TODO: Check if there is sufficient CPU memory, otherwise offload to disk
        # Sleep level 2: discard kv cache
        else:
            offload_to = "none"

        override_tags = getattr(allocator, "_override_tags", {"kv_cache"})
        for ptr, data in allocator.pointer_to_data.items():
            if data.tag not in override_tags:
                continue
            handle = data.handle
            size_in_bytes = handle[1]
            if offload_to != "none" or data.tag == "weights":
                if offload_to == "disk" and data.tag != "weights":
                    cpu_backup_tensor = torch.from_file(
                        f"/tmp/kv-cache-{ptr}.pt",
                        size=size_in_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                        shared=True,
                    )
                else:
                    cpu_backup_tensor = torch.empty(
                        size_in_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=is_pin_memory_available(),
                    )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(
                    ctypes.c_void_p(cpu_ptr), ctypes.c_void_p(ptr), size_in_bytes
                )
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

    def wake_up(tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.
        """
        override_tags = getattr(allocator, "_override_tags", {"kv_cache"})
        for ptr, data in allocator.pointer_to_data.items():
            if data.tag not in override_tags:
                continue
            create_and_map(data.handle)
            if data.cpu_backup_tensor is not None:
                cpu_backup_tensor = data.cpu_backup_tensor
                if cpu_backup_tensor is not None:
                    size_in_bytes = (
                        cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                    )
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(
                        ctypes.c_void_p(ptr),
                        ctypes.c_void_p(cpu_ptr),
                        size_in_bytes,
                    )
                    data.cpu_backup_tensor = None

    allocator.sleep = sleep
    allocator.wake_up = wake_up


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
    """
    Patches an Unsloth patch that causes issues with vLLM.

    Specifically, Unsloth patches get_lora_tokenizer_async with a non-async function, which causes issues.
    """
    import vllm.transformers_utils.tokenizer
    import vllm.transformers_utils.tokenizer_group

    async def _return_nothing(*_, **__) -> None:
        return None

    async def get_self_lora_tokenizer_async(self, *args, **kwargs):
        return self.tokenizer

    vllm.transformers_utils.tokenizer.get_lora_tokenizer_async = _return_nothing  # type: ignore
    vllm.transformers_utils.tokenizer_group.get_lora_tokenizer_async = (  # type: ignore
        _return_nothing
    )
    vllm.transformers_utils.tokenizer_group.TokenizerGroup.get_lora_tokenizer_async = get_self_lora_tokenizer_async  # type: ignore


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
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParserManager,
    )

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


def patch_multi_step_model_runner(runner: MultiStepModelRunner) -> None:
    """
    Patches a MultiStepModelRunner to support LoRA adapters.
    """
    base_runner = runner._base_model_runner
    runner.set_active_loras = base_runner.set_active_loras
    runner.add_lora = base_runner.add_lora
    runner.remove_lora = base_runner.remove_lora
    runner.pin_lora = base_runner.pin_lora
    runner.list_loras = base_runner.list_loras
