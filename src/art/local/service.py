from typing import AsyncIterator, Protocol, runtime_checkable

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch


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
    ) -> tuple[str, int]: ...

    async def vllm_engine_is_sleeping(self) -> bool: ...

    def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]: ...

    def train_sft(
        self,
        batches: list[SFTBatch],
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train using SFT on pre-computed batches.

        Args:
            batches: List of SFTBatch objects to train on.
            verbose: Whether to print detailed logs.

        Yields:
            Dictionary containing training metrics for each batch.
        """
        ...
