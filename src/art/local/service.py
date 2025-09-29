from typing import AsyncIterator, Protocol, runtime_checkable

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors


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

    async def vllm_engine_is_sleeping(self) -> bool: ...

    def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]: ...

    async def set_temperature(self, temperature: float) -> None:
        self.config.setdefault("trainer_args", {})["temperature"] = temperature
        self.state.trainer.args.temperature = temperature  # type: ignore
        await self.start_openai_server(
            dev.OpenAIServerConfig(
                engine_args=dev.EngineArgs(
                    override_generation_config={"temperature": temperature}
                )
            )
        )

