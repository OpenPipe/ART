import asyncio
from dataclasses import dataclass
import math
import os
import torch
from typing import AsyncIterator
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev
from ..local.checkpoints import get_step_from_dir
from ..local.pack import DiskPackedTensors
from .. import types
from ..vllm import get_llm, openai_server_task


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
        num_steps = math.ceil(
            disk_packed_tensors["num_sequences"] / torch.cuda.device_count()
        )
        for _ in range(num_steps):
            yield {"loss": 0.0}
        os.makedirs(
            f"{self.output_dir}/{get_step_from_dir(self.output_dir)+1:04d}",
            exist_ok=True,
        )
