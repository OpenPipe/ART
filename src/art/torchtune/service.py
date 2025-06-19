import asyncio
from collections import Counter
from dataclasses import dataclass
import math
import os
import signal
import torch
from typing import AsyncIterator
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

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

    @property
    def llm_task(self) -> asyncio.Task[AsyncLLM]:
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**self.config.get("engine_args", {})))  # type: ignore
        )

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        await openai_server_task(
            engine=await self.llm_task,
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
        llm = await self.llm_task
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

        num_steps = math.ceil(
            disk_packed_tensors["num_sequences"] / torch.cuda.device_count()
        )
        for _ in range(num_steps):
            yield {"loss": 0.0}
        os.makedirs(
            f"{self.output_dir}/{get_step_from_dir(self.output_dir)+1:04d}",
            exist_ok=True,
        )
