"""Monkey patches and modifications for vLLM."""

import ctypes
from typing import Any

import torch


def patch_allocator() -> None:
    """
    Patch the vLLM CuMemAllocator to specifically focus on offloading/discarding
    the KV cache.
    """
    import gc
    import logging

    from vllm.device_allocator.cumem import (
        CuMemAllocator,
        create_and_map,
        libcudart,
        unmap_and_release,
    )

    logger = logging.getLogger(__name__)

    # is_pin_memory_available was moved in vLLM 0.13.0
    try:
        from vllm.utils.platform_utils import is_pin_memory_available
    except ImportError:
        from vllm.utils import is_pin_memory_available

    allocator = CuMemAllocator.get_instance()

    # Save original methods
    _original_sleep = allocator.sleep
    _original_wake_up = allocator.wake_up

    def patched_sleep(
        self: CuMemAllocator, offload_tags: tuple[str, ...] | str | None = None
    ) -> None:
        """
        Enhanced sleep that respects _override_tags for controlling what gets offloaded.

        The override_tags attribute allows the caller to specify which memory pools
        should be offloaded (backed up to CPU) vs discarded. This enables:
        - Offloading both weights and KV cache for full memory recovery
        - Selective offloading based on outstanding requests
        """
        # Check for override tags set by do_sleep
        override_tags = getattr(self, "_override_tags", None)

        if override_tags is None:
            # No override, use original behavior
            return _original_sleep(offload_tags)

        # Determine sleep level from offload_tags
        # In vLLM 0.13.0: offload_tags=("weights",) for level 1, offload_tags=() for level 2
        sleep_level = 1 if offload_tags else 2
        offload_to = "cpu" if sleep_level == 1 else "none"

        import datetime

        with open("/tmp/patch_allocator_debug.log", "a") as f:
            f.write(
                f"{datetime.datetime.now()}: [patched sleep] offload_tags={offload_tags}, "
                f"sleep_level={sleep_level}, override_tags={override_tags}\n"
            )
            f.flush()

        total_bytes = 0
        backup_bytes = 0
        tag_counts: dict[str, int] = {}

        for ptr, data in self.pointer_to_data.items():
            tag_counts[data.tag] = tag_counts.get(data.tag, 0) + 1
            if data.tag not in override_tags:
                continue
            handle = data.handle
            size_in_bytes = handle[1]
            total_bytes += size_in_bytes
            # Always backup weights; backup KV cache only at level 1
            if offload_to != "none" or data.tag == "weights":
                backup_bytes += size_in_bytes
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

        with open("/tmp/patch_allocator_debug.log", "a") as f:
            f.write(
                f"{datetime.datetime.now()}: [patched sleep] freed {total_bytes / 1024**3:.2f} GiB, "
                f"backed up {backup_bytes / 1024**3:.2f} GiB, "
                f"tag_counts={tag_counts}\n"
            )
            f.flush()

        gc.collect()
        torch.cuda.empty_cache()

    def patched_wake_up(self: CuMemAllocator, tags: list[str] | None = None) -> None:
        """
        Enhanced wake_up that respects _override_tags for controlling what gets restored.
        """
        override_tags = getattr(self, "_override_tags", None)

        if override_tags is None:
            # No override, use original behavior
            return _original_wake_up(tags)

        import datetime

        with open("/tmp/patch_allocator_debug.log", "a") as f:
            f.write(
                f"{datetime.datetime.now()}: [patched wake_up] tags={tags}, override_tags={override_tags}\n"
            )
            f.flush()

        restored_bytes = 0
        allocated_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            if data.tag not in override_tags:
                continue
            create_and_map(data.handle)
            allocated_bytes += data.handle[1]
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
                    restored_bytes += size_in_bytes

        with open("/tmp/patch_allocator_debug.log", "a") as f:
            f.write(
                f"{datetime.datetime.now()}: [patched wake_up] allocated {allocated_bytes / 1024**3:.2f} GiB, "
                f"restored {restored_bytes / 1024**3:.2f} GiB from backup\n"
            )
            f.flush()

    # Bind methods to allocator instance
    import types

    allocator.sleep = types.MethodType(patched_sleep, allocator)
    allocator.wake_up = types.MethodType(patched_wake_up, allocator)

    # Write to a file for debugging since worker stdout/stderr may not be visible
    with open("/tmp/patch_allocator_debug.log", "a") as f:
        import datetime

        f.write(
            f"{datetime.datetime.now()}: Patched allocator.sleep and allocator.wake_up\n"
        )
        f.flush()


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

    # ToolParserManager was moved in vLLM 0.13.0
    try:
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    except ImportError:
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
