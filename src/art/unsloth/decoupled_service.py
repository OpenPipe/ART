import asyncio
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import json
import logging
import os
from pathlib import Path
from safetensors.torch import save_file
import time
import torch
from typing import AsyncIterator
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev
from ..local.pack import DiskPackedTensors
from .. import types
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers


@dataclass
class DecoupledUnslothService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        await openai_server_task(
            engine=await self.llm,
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                base_model=self.get_last_checkpoint_dir() or self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                config=config,
            ),
        )

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        llm = await self.llm
        pids_path = f"{self.output_dir}/pids.txt"
        # reset the pids file
        with open(pids_path, "w") as f:
            f.write("")
        weights_path = "/dev/shm/weights.safetensors"
        # remove the weights file if it exists
        Path(weights_path).unlink(missing_ok=True)
        
        # start putting the workers to sleep
        sleep_task = asyncio.create_task(
            run_on_workers(
                llm,
                sleep,
                level=1,
                pids_path=pids_path,
                weights_path=weights_path,
                profile=verbose,
            )
        )
        # wait for the workers to write their pids twice, indicating that they are asleep
        while True:
            pids = Counter(open(pids_path).read().splitlines())
            if set(pids.values()) == {2}:
                break
            await asyncio.sleep(0.25)
        
        # acquire the train process and queue
        train_process = await self.train_process
        train_queue = await self.train_queue
        
        # write the batch info for communication with the train process
        batch_info = {
            "disk_packed_tensors": disk_packed_tensors,
            "config": config.model_dump(),
            "_config": _config,
        }
        
        # Send batch info to train process via stdin
        import json
        batch_json = json.dumps(batch_info) + "\n"
        assert train_process.stdin is not None
        train_process.stdin.write(batch_json.encode())
        await train_process.stdin.drain()
        
        # consume the batch gradient step results
        num_gradient_steps = -1
        while num_gradient_steps != 0:
            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(train_queue.get()),
                    asyncio.create_task(train_process.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                result = task.result()
                if isinstance(result, dict):
                    result["num_gradient_steps"] = int(result.get("num_gradient_steps", 1))
                    if num_gradient_steps == -1:
                        num_gradient_steps = result["num_gradient_steps"]
                    yield result
                else:
                    raise RuntimeError(
                        f"Train process exited early. See {self.output_dir}/logs/train.log for details."
                    )
            num_gradient_steps -= 1
        
        # wait for the workers to wake up
        await sleep_task
        # remove the weights file
        Path(weights_path).unlink(missing_ok=True)

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        # Ensure we use vLLM V1 engine
        os.environ["VLLM_USE_V1"] = "1"
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**self.config.get("engine_args", {})))  # type: ignore
        )

    @cached_property
    def train_queue(self) -> asyncio.Task[asyncio.Queue[dict[str, float]]]:
        return asyncio.create_task(self.get_train_queue())

    @cached_property
    def train_process(self) -> asyncio.Task[asyncio.subprocess.Process]:
        return asyncio.create_task(self.get_train_process())

    async def get_train_process(self) -> asyncio.subprocess.Process:
        # Migrate existing checkpoints to new structure if needed
        from ..local.checkpoints import migrate_checkpoints_to_new_structure
        migrate_checkpoints_to_new_structure(self.output_dir)
        
        checkpoint_dir = self.get_last_checkpoint_dir() or self.base_model
        
        # Create the training script command
        program_and_args = [
            "python",
            "-m",
            "art.unsloth.train_process",
            "--base-model", self.base_model,
            "--checkpoint-dir", checkpoint_dir,
            "--output-dir", self.output_dir,
            "--config", json.dumps(self.config),
        ]
        
        return await asyncio.subprocess.create_subprocess_exec(
            *program_and_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def get_train_queue(self) -> asyncio.Queue[dict[str, float]]:
        process = await self.train_process
        queue = asyncio.Queue()

        async def read(reader: asyncio.StreamReader) -> None:
            async for line in reader:
                line_str = line.decode("utf-8")
                with open(f"{self.output_dir}/logs/train.log", "a") as f:
                    f.write(line_str)
                line_str = line_str.strip()
                
                # Parse training metrics from the output
                if line_str.startswith("{") and line_str.endswith("}"):
                    try:
                        metrics = json.loads(line_str)
                        if isinstance(metrics, dict) and any(k in metrics for k in ["loss", "reward", "kl"]):
                            await queue.put(metrics)
                    except json.JSONDecodeError:
                        pass

        assert process.stdout and process.stderr
        asyncio.create_task(read(process.stdout))
        asyncio.create_task(read(process.stderr))
        return queue

    def get_last_checkpoint_dir(self) -> str | None:
        from ..local.checkpoints import get_last_checkpoint_dir
        return get_last_checkpoint_dir(self.output_dir)


def sleep(
    *, level: int, pids_path: str, weights_path: str, profile: bool
) -> None:
    """
    Put the worker to sleep until the new model weights are loaded.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        weights_path: The path to the weights file.
        profile: Whether to profile
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.worker.gpu_worker import logger

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
                from safetensors.torch import load_file
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