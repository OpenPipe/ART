import asyncio
from dataclasses import dataclass
from typing import AsyncIterator
from vllm import AsyncEngineArgs

from .. import dev
from ..local.pack import DiskPackedTensors
from .. import types
from ..vllm import get_llm, openai_server_task


@dataclass
class TorchtuneService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        config = dev.openai_server.get_openai_server_config(
            self.model_name,
            # TODO: Choose the base model to be the latest version of the model
            base_model=self.base_model,
            log_file=self.output_dir,
            config=config,
        )
        # TODO: Update the types for EngineArgs
        llm = await get_llm(AsyncEngineArgs(**config.get("engine_args", {})))  # type: ignore
        await openai_server_task(
            engine=llm,
            config=config,
        )

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        await asyncio.sleep(1)
        yield {"loss": 0.0}
