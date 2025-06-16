from typing import AsyncIterator, Protocol, runtime_checkable

from .pack import DiskPackedTensors
from .. import dev
from .. import types


@runtime_checkable
class ModelService(Protocol):
    def __init__(
        self,
        model_name: str,
        base_model: str,
        config: dev.InternalModelConfig,
        output_dir: str,
    ):
        pass

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> None: ...

    def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]: ...
