"""Sleep function for vLLM workers in DecoupledUnslothService.

This is in a separate module to avoid importing unsloth in the worker process.
"""

import os
import time
import logging
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker.gpu_worker import logger
from ..vllm import get_worker


def sleep(*, level: int, pids_path: str, profile: bool) -> None:
    """
    Put the worker to sleep until signaled to wake up.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        profile: Whether to profile
    """
    with open(pids_path, "a") as f:
        f.write(f"{os.getpid()}\n")
    worker = get_worker()
    allocator = CuMemAllocator.get_instance()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        setattr(allocator, "_override_tags", {"weights", "kv_cache"})
        with worker.time("sleep"):
            worker.sleep(level)
        with open(pids_path, "a") as f:
            f.write(f"{os.getpid()}\n")

        # Wait for the signal to wake up
        while os.path.exists(pids_path):
            time.sleep(1)

        with worker.time("wake_up"):
            worker.wake_up()
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")
