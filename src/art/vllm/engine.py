"""Engine and worker management for vLLM."""

import asyncio
import contextlib
import contextvars
from dataclasses import replace
import os
import time
from typing import Any, Callable, Generator, ParamSpec, TypeVar, cast

import cloudpickle
import vllm
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import Worker


async def get_llm(args: vllm.AsyncEngineArgs) -> AsyncLLM:  # ty:ignore[unresolved-attribute]
    """
    Create an AsyncLLM engine with model download and patches applied.

    Args:
        args: The engine arguments including model name and configuration.

    Returns:
        A configured AsyncLLM instance.
    """
    # Download model only if it's not a local path
    if not os.path.exists(args.model):
        process = await asyncio.create_subprocess_shell(
            f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {args.model}"
        )
        await process.wait()

    llm = AsyncLLM.from_engine_args(
        replace(
            args,
            worker_extension_cls=f"{WorkerExtension.__module__}.{WorkerExtension.__qualname__}",
            enable_sleep_mode=True,
        )
    )
    return llm


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
