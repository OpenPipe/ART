from argparse import Namespace
import asyncio
from contextlib import asynccontextmanager
from functools import partial
import logging
import multiprocessing
from openai import AsyncOpenAI
import os
from typing import AsyncIterator, Coroutine, Any
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.serving_models import LoRARequest  # type: ignore
from vllm.logger import _DATE_FORMAT, _FORMAT
from vllm.utils import get_open_zmq_ipc_path, FlexibleArgumentParser
from typing import TYPE_CHECKING
from uvicorn.config import LOGGING_CONFIG

from .async_multiprocessing_engine import MQAsyncLLMEngine
from ..config.openai_server import OpenAIServerConfig

if TYPE_CHECKING:
    from .state import vLLMState

# Unsloth expects these attributes to be present
LoRARequest.lora_tensors = {}  # type: ignore
LoRARequest.lora_embeddings = {}  # type: ignore


async def openai_server_task(
    state: "vLLMState",
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    patch_get_lora_tokenizer_async()
    patch_listen_for_disconnect()
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
                async for model in client.models.list():
                    return
            except:
                pass

    try:
        test_client_task = asyncio.create_task(test_client())

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
    except Exception as e:
        openai_server_task.cancel()
        test_client_task.cancel()
        raise


def _openai_server_coroutine(
    config: OpenAIServerConfig,
) -> Coroutine[Any, Any, None]:
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


def mp_openai_server_task(
    state: "vLLMState",
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    patch_get_lora_tokenizer_async()
    patch_multi_step_model_runner(state)

    # Select random path for IPC.
    ipc_path = get_open_zmq_ipc_path()
    print("Multiprocessing frontend to use %s for IPC Path.", ipc_path)

    engine = MQAsyncLLMEngine(
        ipc_path=ipc_path,
        async_engine=state.async_engine,
    )

    # Start client in separate process (provides the OpenAI API server).
    # the current process might have CUDA context,
    # so maybe we need to spawn a new process
    context = multiprocessing.get_context("spawn")

    client_process = context.Process(
        target=openai_server_target,
        args=(ipc_path, os.getpid(), config),
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
    config: OpenAIServerConfig,
) -> None:
    patch_listen_for_disconnect()

    @asynccontextmanager
    async def build_async_engine_client(
        _: Namespace,
    ) -> AsyncIterator[EngineClient]:
        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_config = AsyncEngineArgs(
            **config.get("engine_args", {})
        ).create_engine_config()
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
    asyncio.run(_openai_server_coroutine(config))
