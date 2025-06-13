import asyncio
import cloudpickle
import contextvars
import ctypes
from dataclasses import replace
import torch
from typing import Any, Callable, cast, ParamSpec, TypeVar
import vllm
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import Worker


async def get_llm(args: vllm.AsyncEngineArgs) -> AsyncLLM:
    # Download model
    process = await asyncio.create_subprocess_shell(
        f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {args.model}"
    )
    await process.wait()
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


def patch_allocator() -> None:
    """
    Patch the vLLM CuMemAllocator to specifically focus on offloading/discarding
    the KV cache.
    """
    import gc
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

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(tags: list[str] | None = None) -> None:
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
                            ctypes.c_void_p(ptr),
                            ctypes.c_void_p(cpu_ptr),
                            size_in_bytes,
                        )
                        data.cpu_backup_tensor = None

    allocator.sleep = sleep
    allocator.wake_up = wake_up


P = ParamSpec("P")
R = TypeVar("R")


async def run_on_workers(
    llm: AsyncLLM, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> list[R]:
    return await llm.collective_rpc(
        "run", args=(cloudpickle.dumps(func), *args), kwargs=kwargs
    )


# Context variable to hold the current worker
_worker: contextvars.ContextVar[Worker] = contextvars.ContextVar("worker")


def get_worker() -> Worker:
    """Get the current worker instance"""
    return _worker.get()


class WorkerExtension:
    def run(self, pickled_func: bytes, *args: Any, **kwargs: Any) -> Any:
        func = cloudpickle.loads(pickled_func)
        token = _worker.set(cast(Worker, self))
        try:
            return func(*args, **kwargs)
        finally:
            _worker.reset(token)
