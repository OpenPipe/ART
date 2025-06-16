import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from .. import dev
from ..local.pack import DiskPackedTensors
from .. import types


@dataclass
class TorchtuneService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> None: ...

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        await asyncio.sleep(1)
        yield {"loss": 0.0}
