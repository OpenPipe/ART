import asyncio
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import signal
import torch
from typing import AsyncIterator
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .batch import Batch
from .. import dev
from ..local.checkpoints import get_step_from_dir
from ..local.pack import DiskPackedTensors
from .. import types
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers


@dataclass
class TorchtuneService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str

    @lru_cache(maxsize=1)
    def llm(self) -> asyncio.Task[AsyncLLM]:
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**self.config.get("engine_args", {})))  # type: ignore
        )

    @lru_cache(maxsize=1)
    def train_process(self) -> asyncio.Task[asyncio.subprocess.Process]:
        import torchtune

        program_and_args = [
            f"{os.path.dirname(torchtune.__file__)}/_cli/tune.py",
            "run",
            "--nproc-per-node",
            str(torch.cuda.device_count()),
            f"{os.path.dirname(__file__)}/recipe.py",
            "--config",
            f"{os.path.dirname(__file__)}/config.yaml",
            "metric_logger._component_=torchtune.training.metric_logging.StdoutLogger",
            "metric_logger.log_dir=null",
        ]

        print(program_and_args)

        return asyncio.create_task(
            asyncio.subprocess.create_subprocess_exec(
                *program_and_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        await openai_server_task(
            engine=await self.llm(),
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                # TODO: Choose the base model to be the latest version of the model
                base_model=self.base_model,
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
        llm = await self.llm()
        pids_path = f"{self.output_dir}/pids.txt"
        with open(pids_path, "w") as f:
            f.write("")

        def sleep() -> None:
            from vllm.device_allocator.cumem import CuMemAllocator

            with open(pids_path, "a") as f:
                f.write(f"{os.getpid()}\n")
            worker = get_worker()
            allocator = CuMemAllocator.get_instance()
            setattr(allocator, "_override_tags", {"weights", "kv_cache"})
            worker.sleep()
            with open(pids_path, "a") as f:
                f.write(f"{os.getpid()}\n")
            os.kill(os.getpid(), signal.SIGSTOP)
            worker.wake_up()
            delattr(allocator, "_override_tags")

        sleep_task = asyncio.create_task(run_on_workers(llm, sleep))
        while True:
            pids = Counter(open(pids_path).read().splitlines())
            if set(pids.values()) == {2}:
                break
            await asyncio.sleep(0.25)

        with open(f"{self.output_dir}/batches.jsonl", "a") as f:
            f.write(
                Batch(
                    disk_packed_tensors=disk_packed_tensors,
                    config=config,
                    dev_config=_config,
                ).model_dump_json()
                + "\n"
            )

        train_process = await self.train_process()
        assert train_process.stdout is not None
        num_steps = math.ceil(
            disk_packed_tensors["num_sequences"] / torch.cuda.device_count()
        )
        async for line in train_process.stdout:
            line_str = line.decode("utf-8")
            with open(f"{self.output_dir}/logs/train.log", "a") as f:
                f.write(line_str)
            line_str = line_str.strip()

            # Look for lines in format: "Step {step} | {name}:{value} {name}:{value} ..."
            if line_str.startswith("Step ") and " | " in line_str:
                parts = line_str.split(" | ", 1)
                step = int(parts[0].split()[1])

                metrics: dict[str, float] = {"step": float(step)}

                # Parse metrics from the second part
                if len(parts) > 1:
                    for metric in parts[1].split():
                        if ":" in metric:
                            name, value = metric.split(":", 1)
                            try:
                                metrics[name] = float(value)
                            except ValueError:
                                # Skip non-numeric values to match the return type
                                pass

                yield metrics
                num_steps -= 1
                if num_steps == 0:
                    break

        await sleep_task
