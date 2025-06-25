"""Engine and worker management for vLLM."""

import asyncio
import cloudpickle
import contextlib
import contextvars
from dataclasses import replace
import time
from typing import Any, Callable, cast, Coroutine, Generator, ParamSpec, TypeVar
import vllm
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import Worker

from .patches import patch_allocator


async def get_llm(args: vllm.AsyncEngineArgs) -> AsyncLLM:
    """
    Create an AsyncLLM engine with model download and patches applied.

    Args:
        args: The engine arguments including model name and configuration.

    Returns:
        A configured AsyncLLM instance.
    """
    # Download model
    process = await asyncio.create_subprocess_shell(
        f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {args.model}"
    )
    await process.wait()
    # Make sure we are using the v1 engine
    import vllm.envs as envs

    envs.VLLM_USE_V1 = "1"
    # Create engine
    llm = AsyncLLM.from_engine_args(
        replace(
            args,
            worker_extension_cls=f"{WorkerExtension.__module__}.{WorkerExtension.__qualname__}",
            enable_sleep_mode=True,
        )
    )
    await run_on_workers(llm, patch_allocator)
    return llm


def create_engine_pause_and_resume_functions(
    engine: AsyncLLMEngine,
) -> tuple[
    Callable[[], Coroutine[Any, Any, None]], Callable[[], Coroutine[Any, Any, None]]
]:
    """
    Patches the vLLM engine and returns a pair of functions for pausing and resuming
    request processing respectively.

    Args:
        engine: The AsyncLLMEngine to patch.

    Returns:
        A tuple of (pause_engine, resume_engine) async functions.
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


P = ParamSpec("P")
R = TypeVar("R")


async def run_on_workers(
    llm: AsyncLLM, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> list[R]:
    """
    Run a function on all workers in a distributed setup.

    Args:
        llm: The AsyncLLM instance with workers.
        func: The function to run on each worker.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        List of results from each worker.
    """
    return await llm.collective_rpc(
        "run", args=(cloudpickle.dumps(func), *args), kwargs=kwargs
    )


# Context variable to hold the current worker
_worker: contextvars.ContextVar["ExtendedWorker"] = contextvars.ContextVar("worker")


def get_worker() -> "ExtendedWorker":
    """Get the current worker instance"""
    return _worker.get()


class WorkerExtension:
    """Extension for running arbitrary functions on vLLM workers."""

    def run(self, pickled_func: bytes, *args: Any, **kwargs: Any) -> Any:
        func = cloudpickle.loads(pickled_func)
        token = _worker.set(cast(ExtendedWorker, self))
        try:
            return func(*args, **kwargs)
        finally:
            _worker.reset(token)

    @contextlib.contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        from vllm.v1.worker.gpu_worker import logger

        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        logger.info(f"{name}: {end_time - start_time:.2f} seconds")


class ExtendedWorker(Worker, WorkerExtension):
    pass
