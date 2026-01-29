"""OpenAI-compatible server functionality for vLLM."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncIterator, Coroutine, cast

from openai import AsyncOpenAI
from uvicorn.config import LOGGING_CONFIG
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import _DATE_FORMAT, _FORMAT
from vllm.logging_utils import NewLineFormatter
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ..dev.openai_server import OpenAIServerConfig

_openai_serving_models: Any | None = None


async def openai_server_task(
    engine: EngineClient,
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    """
    Starts an asyncio task that runs an OpenAI-compatible server.

    Args:
        engine: The vLLM engine client.
        config: The configuration for the OpenAI-compatible server.

    Returns:
        A running asyncio task for the OpenAI-compatible server. Cancel the task
        to stop the server.
    """
    # Import patches before importing api_server
    from .patches import (
        patch_listen_for_disconnect,
        patch_tool_parser_manager,
        subclass_chat_completion_request,
    )

    # We must subclass ChatCompletionRequest before importing api_server
    # or logprobs will not always be returned
    subclass_chat_completion_request()
    # Capture the OpenAIServingModels instance so dynamically added LoRAs
    # are reflected in the model list.
    from vllm.entrypoints.openai import api_server, serving_models

    serving_models_any = cast(Any, serving_models)
    if not getattr(serving_models_any, "_art_openai_serving_models_patched", False):
        serving_models_any._art_openai_serving_models_patched = True
        original_init = serving_models.OpenAIServingModels.__init__

        def _init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            global _openai_serving_models
            _openai_serving_models = self

        serving_models.OpenAIServingModels.__init__ = _init  # ty:ignore[invalid-assignment]

    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    set_vllm_log_file(config.get("log_file", "vllm.log"))

    # Patch engine.add_lora to ensure lora_tensors attribute exists
    # This is needed for compatibility with Unsloth
    add_lora = engine.add_lora

    async def _add_lora(lora_request) -> bool:
        # Ensure lora_tensors attribute exists on the request
        if not hasattr(lora_request, "lora_tensors"):
            # For msgspec.Struct, we need to create a new instance with the attribute
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                lora_name=lora_request.lora_name,
                lora_int_id=lora_request.lora_int_id,
                lora_path=lora_request.lora_path,
                long_lora_max_len=getattr(lora_request, "long_lora_max_len", None),
                base_model_name=getattr(lora_request, "base_model_name", None),
            )
        added = await add_lora(lora_request)
        if added and _openai_serving_models is not None:
            _openai_serving_models.lora_requests[lora_request.lora_name] = lora_request
        return added

    engine.add_lora = _add_lora  # ty:ignore[invalid-assignment]

    @asynccontextmanager
    async def build_async_engine_client(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[EngineClient]:
        yield engine

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
            except:  # noqa: E722
                await asyncio.sleep(0.1)

    test_client_task = asyncio.create_task(test_client())
    try:
        timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 30.0))
        done, _ = await asyncio.wait(
            [openai_server_task, test_client_task],
            timeout=timeout,
            return_when="FIRST_COMPLETED",
        )
        if not done:
            raise TimeoutError(
                f"Unable to reach OpenAI-compatible server within {timeout} seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable."
            )
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
    assert namespace is not None
    validate_parsed_serve_args(namespace)
    return api_server.run_server(
        namespace,
        log_config=get_uvicorn_logging_config(config.get("log_file", "vllm.log")),
    )


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

    # Use vLLM's NewLineFormatter which adds the fileinfo field
    formatter = NewLineFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    vllm_logger.addHandler(file_handler)

    # Set log level to filter out DEBUG messages
    vllm_logger.setLevel(logging.INFO)
