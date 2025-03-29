from argparse import Namespace
import asyncio
from contextlib import asynccontextmanager
from functools import partial
import logging
import multiprocessing
import os
from peft.peft_model import PeftModel
import re
from typing import AsyncIterator, Coroutine, Any
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.serving_models import LoRARequest  # type: ignore
from vllm.logger import _DATE_FORMAT, _FORMAT
from vllm.utils import get_open_zmq_ipc_path, FlexibleArgumentParser
from typing import cast, Literal, TypedDict

from .. import UVICORN_LOGGING_CONFIG_PATH
from .async_multiprocessing_engine import MQAsyncLLMEngine
from ..types import BaseModel


# Unsloth expects these attributes to be present
LoRARequest.lora_tensors = {}  # type: ignore
LoRARequest.lora_embeddings = {}  # type: ignore


def max_concurrent_tokens(path: str) -> int:
    with open(path, "r") as f:
        matches = re.findall(
            r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x",
            f.read(),
        )
        return int(int(matches[-1][0]) * float(matches[-1][1]))


def openai_server_task2(
    model: PeftModel,
    model_name: str,
    base_model: BaseModel,
    tool_use: bool,
    lora_path: str,
) -> asyncio.Task[None]:
    patch_get_lora_tokenizer_async()
    patch_multi_step_model_runner(model)

    # Select random path for IPC.
    ipc_path = get_open_zmq_ipc_path()
    print("Multiprocessing frontend to use %s for IPC Path.", ipc_path)

    engine = MQAsyncLLMEngine(
        ipc_path=ipc_path,
        async_engine=model.vllm_engine,
    )

    # Start client in separate process (provides the OpenAI API server).
    # the current process might have CUDA context,
    # so maybe we need to spawn a new process
    context = multiprocessing.get_context("spawn")

    client_process = context.Process(
        target=openai_server_target,
        args=(ipc_path, os.getpid(), model_name, base_model, tool_use, lora_path),
    )

    async def openai_server_task() -> None:
        engine_task = asyncio.create_task(engine.run())
        try:
            client_process.start()
            await engine_task
        finally:
            engine_task.cancel()
            client_process.terminate()

    return asyncio.create_task(openai_server_task())


def openai_server_target(
    ipc_path: str,
    engine_pid: int,
    model_name: str,
    base_model: BaseModel,
    tool_use: bool,
    lora_path: str,
) -> None:
    patch_listen_for_disconnect()

    engine_args = AsyncEngineArgs(
        model=base_model,
        num_scheduler_steps=16,
        served_model_name=base_model,
    )

    @asynccontextmanager
    async def build_async_engine_client(
        _: Namespace,
    ) -> AsyncIterator[EngineClient]:
        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_config = engine_args.create_engine_config()
        build_client = partial(MQLLMEngineClient, ipc_path, engine_config, engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client
        )
        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    pass

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Close all open connections to the backend
            mq_engine_client.close()

    api_server.build_async_engine_client = build_async_engine_client
    asyncio.run(_openai_server_coroutine(model_name, base_model, tool_use, lora_path))


def openai_server_task(
    model: PeftModel,
    model_name: str,
    base_model: BaseModel,
    tool_use: bool,
    lora_path: str,
) -> asyncio.Task[None]:
    patch_get_lora_tokenizer_async()
    patch_listen_for_disconnect()
    patch_multi_step_model_runner(model)

    @asynccontextmanager
    async def build_async_engine_client(
        _: Namespace,
    ) -> AsyncIterator[EngineClient]:
        yield getattr(model, "vllm_engine")

    api_server.build_async_engine_client = build_async_engine_client

    return asyncio.create_task(
        _openai_server_coroutine(model_name, base_model, tool_use, lora_path)
    )


def _openai_server_coroutine(
    model_name: str,
    base_model: BaseModel,
    tool_use: bool,
    lora_path: str,
) -> Coroutine[Any, Any, None]:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    engine_args = cast(
        dict,
        cast(type[AsyncEngineArgs], dict)(
            disable_log_requests=True,
            model=base_model,
            num_scheduler_steps=16,
            served_model_name=base_model,
        ),
    )
    server_args = ServerArgs(
        api_key="default",
        lora_modules=[f'{{"name": "{model_name}", "path": "{lora_path}"}}'],
        return_tokens_as_token_ids=True,
    )
    if tool_use:
        server_args["enable_auto_tool_choice"] = True
        server_args["tool_call_parser"] = "hermes"
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
    return api_server.run_server(namespace, log_config=UVICORN_LOGGING_CONFIG_PATH)


def patch_get_lora_tokenizer_async() -> None:
    """
    Patches an Unsloth patch that causes issues with vLLM.

    Specifically, Unsloth patches get_lora_tokenizer_async with a non-async function, which causes issues.
    """
    import vllm.transformers_utils.tokenizer_group.tokenizer_group

    async def _return_nothing(*_, **__) -> None:
        return None

    vllm.transformers_utils.tokenizer_group.tokenizer_group.get_lora_tokenizer_async = _return_nothing  # type: ignore


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


def patch_multi_step_model_runner(model: PeftModel) -> None:
    """
    Patches the vLLM multi-step model runner to support LoRA adapters.
    """
    model_runner = model.vllm_engine.engine.model_executor.driver_worker.model_runner  # type: ignore
    if not hasattr(model_runner, "_base_model_runner"):
        return
    base_model_runner = model_runner._base_model_runner
    model_runner.set_active_loras = base_model_runner.set_active_loras
    model_runner.add_lora = base_model_runner.add_lora
    model_runner.remove_lora = base_model_runner.remove_lora
    model_runner.pin_lora = base_model_runner.pin_lora
    model_runner.list_loras = base_model_runner.list_loras


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


class ServerArgs(TypedDict, total=False):
    """Arguments for the vLLM OpenAI-compatible server, not including AsyncEngineArgs.

    Fields:
        host: Host name for the server. If None, will listen on all available interfaces (0.0.0.0).
        port: Port number for the server to listen on.
        uvicorn_log_level: Log level for the uvicorn server.
        allow_credentials: Whether to allow credentials in CORS requests.
        allowed_origins: JSON string that will be parsed into a list of allowed CORS origins. Defaults to ["*"].
        allowed_methods: JSON string that will be parsed into a list of allowed HTTP methods. Defaults to ["*"].
        allowed_headers: JSON string that will be parsed into a list of allowed HTTP headers. Defaults to ["*"].
        api_key: API key required for authentication. If None, no authentication is required.
        lora_modules: List of LoRA module configurations. Each string is either in 'name=path' format or a JSON object.
        prompt_adapters: List of prompt adapter configurations. Each string is in 'name=path' format.
        chat_template: Path to chat template file or the template in single-line form.
        chat_template_content_format: Format to render message content within chat templates.
        response_role: Role name to return when request.add_generation_prompt is true.
        ssl_keyfile: Path to SSL key file for HTTPS.
        ssl_certfile: Path to SSL certificate file for HTTPS.
        ssl_ca_certs: Path to CA certificates file for HTTPS.
        enable_ssl_refresh: Whether to refresh SSL context when certificate files change.
        ssl_cert_reqs: SSL certificate requirements (see stdlib ssl module's CERT_* constants).
        root_path: FastAPI root_path when app is behind a path-based routing proxy.
        middleware: List of additional ASGI middleware to apply to the app. Each string is an import path.
        return_tokens_as_token_ids: When max_logprobs is specified, represent tokens as 'token_id:{token_id}' strings.
        disable_frontend_multiprocessing: Run OpenAI frontend server in same process as model serving engine.
        enable_request_id_headers: Add X-Request-Id header to responses (may impact performance at high QPS).
        enable_auto_tool_choice: Enable automatic tool choice for supported models.
        tool_call_parser: Parser to use for model-generated tool calls into OpenAI API format.
        tool_parser_plugin: Plugin for registering custom tool call parsers.
        max_log_len: Maximum number of prompt characters or prompt ID numbers to print in logs.
        disable_fastapi_docs: Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint.
        enable_prompt_tokens_details: Include prompt_tokens_details in usage statistics.
        enable_server_load_tracking: Track server_load_metrics in the app state.
        enable_reasoning: Enable reasoning capabilities for supported models.
        reasoning_parser: Parser to use for model-generated reasoning into OpenAI API format.
    """

    host: str | None
    port: int
    uvicorn_log_level: Literal["debug", "info", "warning", "error", "critical", "trace"]
    allow_credentials: bool
    allowed_origins: str  # JSON string that will be parsed into list[str]
    allowed_methods: str  # JSON string that will be parsed into list[str]
    allowed_headers: str  # JSON string that will be parsed into list[str]
    api_key: str | None
    lora_modules: list[str] | None  # Each string is either 'name=path' or a JSON object
    prompt_adapters: list[str] | None  # Each string is in 'name=path' format
    chat_template: str | None
    chat_template_content_format: Literal["auto", "string", "openai"]
    response_role: str | None
    ssl_keyfile: str | None
    ssl_certfile: str | None
    ssl_ca_certs: str | None
    enable_ssl_refresh: bool
    ssl_cert_reqs: int
    root_path: str | None
    middleware: list[str]  # Each string is an import path
    return_tokens_as_token_ids: bool
    disable_frontend_multiprocessing: bool
    enable_request_id_headers: bool
    enable_auto_tool_choice: bool
    tool_call_parser: str | None
    tool_parser_plugin: str
    max_log_len: int | None
    disable_fastapi_docs: bool
    enable_prompt_tokens_details: bool
    enable_server_load_tracking: bool
    enable_reasoning: bool
    reasoning_parser: str | None
