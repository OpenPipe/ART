from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Protocol, TypeAlias

from . import dev
from .trajectories import TrajectoryGroup
from .types import TrainConfig, TrainResult

if TYPE_CHECKING:
    from .model import Model, TrainableModel

# Type aliases for models with any config/state type (for backend method signatures)
AnyModel: TypeAlias = "Model[Any, Any]"
AnyTrainableModel: TypeAlias = "TrainableModel[Any, Any]"


class Backend(Protocol):
    """Protocol for backend implementations."""

    def _model_inference_name(
        self, model: AnyModel, step: int | None = None
    ) -> str: ...

    async def close(self) -> None: ...

    async def register(self, model: AnyModel) -> None: ...

    async def _get_step(self, model: AnyTrainableModel) -> int: ...

    async def _delete_checkpoint_files(
        self, model: AnyTrainableModel, steps_to_keep: list[int]
    ) -> None: ...

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]: ...

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult: ...

    def _train_model(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]: ...
