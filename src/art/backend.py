from contextlib import AbstractAsyncContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Protocol,
    TypeAlias,
)

from . import dev
from .trajectories import Trajectory
from .types import TrainResult, TrainSFTConfig

if TYPE_CHECKING:
    from .model import Model, TrainableModel
    from .pipeline_trainer.checkpoint_retention import (
        CheckpointInfo,
        CheckpointRetentionPlan,
        CheckpointRetentionStrategy,
    )

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

    async def _list_checkpoint_infos(
        self, model: AnyTrainableModel
    ) -> list["CheckpointInfo"]: ...

    def default_checkpoint_retention_strategy(
        self,
    ) -> "CheckpointRetentionStrategy | None": ...

    async def _apply_checkpoint_retention(
        self, model: AnyTrainableModel, plan: "CheckpointRetentionPlan"
    ) -> None: ...

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]: ...

    def exact_adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AbstractAsyncContextManager[None]: ...

    # Backends intentionally expose backend-specific optional training arguments.
    # Callable[..., ...] preserves that extensibility without falsely requiring
    # every implementation to accept every other backend's keyword arguments.
    @property
    def train(self) -> Callable[..., Coroutine[Any, Any, TrainResult]]: ...

    def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]: ...
