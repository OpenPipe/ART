import asyncio
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import glob
import os
from safetensors.torch import load_file
import time
import torch
import torchtune
from typing import AsyncIterator
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .batch import Batch
from .. import dev
from ..local.pack import DiskPackedTensors
from .. import types
from ..utils.get_model_step import get_step_from_dir
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers


@dataclass
class TorchtuneService:
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
        sleep_level = 1 if llm.output_processor.has_unfinished_requests() else 2
        pids_path = f"{self.output_dir}/pids.txt"
        state_dict_path = "/dev/shm/state_dict.safetensors"
        with open(pids_path, "w") as f:
            f.write("")
        if os.path.exists(state_dict_path):
            os.remove(state_dict_path)

        def sleep() -> None:
            import logging
            from vllm.device_allocator.cumem import CuMemAllocator
            from vllm.v1.worker.gpu_worker import logger

            with open(pids_path, "a") as f:
                f.write(f"{os.getpid()}\n")
            worker = get_worker()
            if not (verbose and worker.rank == 0):
                logger.setLevel(logging.CRITICAL)
            allocator = CuMemAllocator.get_instance()
            setattr(allocator, "_override_tags", {"weights", "kv_cache"})
            with worker.time("sleep"):
                worker.sleep(sleep_level)
            with open(pids_path, "a") as f:
                f.write(f"{os.getpid()}\n")
            while True:
                try:
                    with worker.time("load_state_dict"):
                        state_dict = load_file(state_dict_path)
                    break
                except FileNotFoundError:
                    time.sleep(1)
                    continue
                except Exception as e:
                    if "MetadataIncompleteBuffer" in str(
                        e
                    ) or "InvalidHeaderLength" in str(e):
                        time.sleep(1)
                        continue
                    raise e
            with worker.time("wake_up"):
                worker.wake_up()
            delattr(allocator, "_override_tags")
            with worker.time("load_weights"):
                worker.model_runner.model.load_weights(state_dict.items())  # type: ignore
            logger.setLevel(logging.INFO)

        sleep_task = asyncio.create_task(run_on_workers(llm, sleep))
        while True:
            pids = Counter(open(pids_path).read().splitlines())
            if set(pids.values()) == {2}:
                break
            await asyncio.sleep(0.25)

        train_process = await self.train_process
        train_queue = await self.train_queue
        with open(f"{self.output_dir}/batches.jsonl", "a") as f:
            f.write(
                Batch(
                    disk_packed_tensors=disk_packed_tensors,
                    config=config,
                    dev_config=_config,
                ).model_dump_json()
                + "\n"
            )
        num_gradient_steps = -1
        while num_gradient_steps != 0:
            done, _ = await asyncio.wait(
                [train_queue.get(), train_process.wait()],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                result = task.result()
                if isinstance(result, dict):
                    result["num_gradient_steps"] = int(result["num_gradient_steps"])
                    if num_gradient_steps == -1:
                        num_gradient_steps = result["num_gradient_steps"]
                    yield result
                else:
                    raise RuntimeError(
                        f"Train process exited early. See {self.output_dir}/logs/train.log for details."
                    )
            num_gradient_steps -= 1
        await sleep_task
        os.remove(pids_path)
        os.remove(state_dict_path)

    @property
    def torchtune_args(self) -> dev.TorchtuneArgs:
        torchtune_args = self.config.get("torchtune_args")
        assert (
            torchtune_args is not None
        ), 'TorchtuneService created without config["torchtune_args"]'
        return torchtune_args

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
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
        if os.path.exists(f"{self.output_dir}/batches.jsonl"):
            os.remove(f"{self.output_dir}/batches.jsonl")
        checkpoint_dir = await self.get_checkpoint_dir()
        torchtune_args = self.torchtune_args

        # Get the list of safetensor files
        safetensor_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        checkpoint_files = [os.path.basename(f) for f in safetensor_files]
        checkpoint_files_str = "[" + ", ".join(f'"{f}"' for f in checkpoint_files) + "]"

        program_and_args = [
            "python",  # Use Python interpreter
            f"{os.path.dirname(torchtune.__file__)}/_cli/tune.py",
            "run",
            "--nproc-per-node",
            str(torch.cuda.device_count()),
            "art.torchtune.recipe.FullFinetuneRecipeDistributed",
            "--config",
            f"{os.path.dirname(__file__)}/config.yaml",
            f"model._component_=torchtune.models.{torchtune_args['model'].split('_')[0]}.{torchtune_args['model']}",
            f"checkpointer.checkpoint_dir={checkpoint_dir}",
            f"checkpointer.checkpoint_files={checkpoint_files_str}",
            f"checkpointer.model_type={torchtune_args['model_type']}",
            f"tensor_parallel_dim={torchtune_args.get('tensor_parallel_dim', 1)}",
            f"context_parallel_dim={torchtune_args.get('context_parallel_dim', 1)}",
            f"output_dir={self.output_dir}",
            "metric_logger._component_=torchtune.training.metric_logging.StdoutLogger",
            "metric_logger.log_dir=null",
        ]
        return await asyncio.subprocess.create_subprocess_exec(
            *program_and_args,
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
                if line_str.startswith("Step ") and " | " in line_str:
                    parts = line_str.split(" | ", 1)
                    metrics: dict[str, float] = {}
                    if len(parts) > 1:
                        for metric in parts[1].split():
                            if ":" in metric:
                                name, value = metric.split(":", 1)
                                try:
                                    metrics[name] = float(value)
                                except ValueError:
                                    # Skip non-numeric values to match the return type
                                    pass
                    await queue.put(metrics)

        assert process.stdout and process.stderr
        asyncio.create_task(read(process.stdout))
        asyncio.create_task(read(process.stderr))
        return queue

    async def get_checkpoint_dir(self) -> str:
        # Use the last of any existing checkpoints to resume training
        if last_checkpoint_dir := self.get_last_checkpoint_dir():
            return last_checkpoint_dir
        # Check if self.base_model is a directory
        if os.path.isdir(self.base_model):
            return self.base_model
        # Otherwise, assume it's a HuggingFace model id and download it
        process = await asyncio.subprocess.create_subprocess_exec(
            "huggingface-cli",
            "download",
            self.base_model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return stdout.decode("utf-8").splitlines()[-1].strip()

    def get_last_checkpoint_dir(self) -> str | None:
        dir = f"{self.output_dir}/{get_step_from_dir(self.output_dir):04d}"
        return dir if os.path.isdir(dir) else None
