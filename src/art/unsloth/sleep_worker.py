"""Sleep function for vLLM workers in DecoupledUnslothService.

This is in a separate module to avoid importing unsloth in the worker process.
"""
import os
import time
import logging
from safetensors.torch import load_file
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker.gpu_worker import logger
from ..vllm import get_worker

def sleep(*, level: int, pids_path: str, weights_path: str, profile: bool) -> None:
    """
    Put the worker to sleep until the new model weights are loaded.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        weights_path: The path to the weights file.
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

        # Wait for weights to be saved by the training process
        while True:
            if os.path.exists(weights_path):
                # Load the new weights
                with worker.time("load_file"):
                    weights = load_file(weights_path)
                break
            elif not os.path.exists(pids_path):
                # no pids file indicates we can wake up without new weights
                weights = None
                break
            else:
                time.sleep(1)
                continue

        with worker.time("wake_up"):
            worker.wake_up()

        if weights is not None:
            with worker.time("load_weights"):
                worker.model_runner.model.load_weights(weights.items())  # type: ignore
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")