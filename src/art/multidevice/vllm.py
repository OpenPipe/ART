from argparse import Namespace
import asyncio
from contextlib import asynccontextmanager
import ctypes
import logging
from openai import AsyncOpenAI
import os
import torch
from typing import Any, AsyncIterator, Callable, Coroutine, TYPE_CHECKING, Optional
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.lora.request import LoRARequest
from vllm.logger import _DATE_FORMAT, _FORMAT
from vllm.utils import FlexibleArgumentParser
from uvicorn.config import LOGGING_CONFIG

from ..dev.openai_server import OpenAIServerConfig

if TYPE_CHECKING:
    from .state import vLLMState


async def openai_server_task(
    state: "vLLMState",
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    """
    Starts an asyncio task that runs an OpenAI-compatible server.

    Args:
        state: The state of the vLLM engine.
        config: The configuration for the OpenAI-compatible server.

    Returns:
        A running asyncio task for the OpenAI-compatible server. Cancel the task
        to stop the server.
    """
    # We must subclass ChatCompletionRequest before importing api_server
    # or logprobs will not always be returned
    subclass_chat_completion_request()
    from vllm.entrypoints.openai import api_server

    patch_lora_request()
    patch_get_lora_tokenizer_async()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    patch_multi_step_model_runner(state)
    set_vllm_log_file(config.get("log_file", "vllm.log"))

    @asynccontextmanager
    async def build_async_engine_client(
        _: Namespace,
    ) -> AsyncIterator[EngineClient]:
        yield state.async_engine

    api_server.build_async_engine_client = build_async_engine_client
    openai_server_task = asyncio.create_task(_openai_server_coroutine(config))
    server_args = config.get("server_args", {})
    client = AsyncOpenAI(
        api_key=server_args.get("api_key"),
        base_url=f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1",
    )

    async def test_client() -> None:
        while True:
            try:
                async for _ in client.models.list():
                    return
            except:
                pass

    test_client_task = asyncio.create_task(test_client())
    try:
        done, _ = await asyncio.wait(
            [openai_server_task, test_client_task],
            timeout=10.0,
            return_when="FIRST_COMPLETED",
        )
        if not done:
            raise TimeoutError("Unable to reach OpenAI-compatible server in time.")
        for task in done:
            task.result()

        return openai_server_task
    except Exception:
        openai_server_task.cancel()
        test_client_task.cancel()
        raise


def _openai_server_coroutine(
    config: OpenAIServerConfig,
) -> Coroutine[Any, Any, None]:
    from vllm.entrypoints.openai import api_server

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    engine_args = config.get("engine_args", {})
    server_args = config.get("server_args", {})
    args = [
        *[
            f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
            for args in [engine_args, server_args]
            for key, value in args.items()
            for item in (value if isinstance(value, list) else [value])
            if item is not None
        ],
    ]
    namespace = parser.parse_args(args)
    validate_parsed_serve_args(namespace)
    return api_server.run_server(
        namespace,
        log_config=get_uvicorn_logging_config(config.get("log_file", "vllm.log")),
    )


def create_engine_pause_and_resume_functions(
    engine: AsyncLLMEngine,
) -> tuple[
    Callable[[], Coroutine[Any, Any, None]], Callable[[], Coroutine[Any, Any, None]]
]:
    """
    Patches the vLLM engine and returns a pair of functions for pausing and resuming
    request processing respectively.
    """
    _engine_step = engine.engine_step
    resume_event = asyncio.Event()
    resume_event.set()
    engine_step_event = asyncio.Event()

    async def engine_step(virtual_engine: int) -> bool:
        engine_step_event.set()
        await resume_event.wait()
        return await _engine_step(virtual_engine)

    engine.engine_step = engine_step

    async def pause_engine() -> None:
        resume_event.clear()
        if engine.engine.has_unfinished_requests():
            engine_step_event.clear()
            await engine_step_event.wait()

    async def resume_engine() -> None:
        resume_event.set()

    return pause_engine, resume_engine


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

        for ptr, data in allocator.pointer_to_data.items():
            if data.tag != "kv_cache":
                continue
            handle = data.handle
            size_in_bytes = handle[1]
            if offload_to != "none":
                if offload_to == "disk":
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

    def wake_up(tags: Optional[list[str]] = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.
        """
        for ptr, data in allocator.pointer_to_data.items():
            if data.tag != "kv_cache":
                continue
            if tags is None or data.tag in tags:
                create_and_map(data.handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(
                            ctypes.c_void_p(ptr), ctypes.c_void_p(cpu_ptr), size_in_bytes
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
    Patches the vLLM LoRARequest type to have attributes Unsloth expects.
    """
    LoRARequest.lora_tensors = {}  # type: ignore
    LoRARequest.lora_embeddings = {}  # type: ignore


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
    vllm.transformers_utils.tokenizer_group.get_lora_tokenizer_async = (
        _return_nothing  # type: ignore
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


def patch_multi_step_model_runner(state: "vLLMState") -> None:
    """
    Patches the vLLM multi-step model runner to support LoRA adapters.
    """
    model_runner = state.multi_step_model_runner  # type: ignore
    if not hasattr(model_runner, "_base_model_runner"):
        return
    base_model_runner = model_runner._base_model_runner
    model_runner.set_active_loras = base_model_runner.set_active_loras
    model_runner.add_lora = base_model_runner.add_lora
    model_runner.remove_lora = base_model_runner.remove_lora
    model_runner.pin_lora = base_model_runner.pin_lora
    model_runner.list_loras = base_model_runner.list_loras


def get_uvicorn_logging_config(path: str) -> dict[str, Any]:
    """
    Returns a Uvicorn logging config that writes to the given path.
    """
    return {
        **LOGGING_CONFIG,
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
            "access": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
        },
    }


def set_vllm_log_file(path: str) -> None:
    """
    Sets the vLLM log file to the given path.
    """

    # Create directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get the vLLM logger
    vllm_logger = logging.getLogger("vllm")

    # Remove existing handlers
    for handler in vllm_logger.handlers[:]:
        vllm_logger.removeHandler(handler)

    # Create a file handler
    file_handler = logging.FileHandler(path)

    # Use the same formatter as vLLM's default
    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    vllm_logger.addHandler(file_handler)

    # Set log level to filter out DEBUG messages
    vllm_logger.setLevel(logging.INFO)
