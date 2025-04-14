from dataclasses import dataclass
from openai import AsyncOpenAI
from typing import cast, Iterable, TYPE_CHECKING

from . import dev
from .openai import patch_openai
from .trajectories import Trajectory, TrajectoryGroup
from .types import BaseModel, TrainConfig


if TYPE_CHECKING:
    from .local.api import LocalAPI


@dataclass
class Model:
    name: str
    project: str
    base_model: BaseModel
    _api: "LocalAPI"
    _config: dev.ModelConfig | None = None

    async def openai_client(
        self,
        _config: dev.OpenAIServerConfig | None = None,
    ) -> AsyncOpenAI:
        """
        Get a client to an OpenAI-compatible inference service for this model.

        Args:
            _config: An OpenAIServerConfig object. May be subject to breaking changes at any time.
                Use at your own risk.

        Returns:
            An asynchronous OpenAI client.
        """
        client = await self._api._get_openai_client(self, _config)
        return patch_openai(client)

    async def get_step(self) -> int:
        """
        Get the model's current training step.
        """
        return await self._api._get_step(self)

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        await self._api._delete_checkpoints(self, best_checkpoint_metric)

    async def log(
        self,
        trajectories: Iterable[Trajectory] | Iterable[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectories or trajectory groups.

        Args:
            trajectories: A batch of trajectories or trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        if any(isinstance(t, Trajectory) for t in trajectories):
            trajectory_groups = [
                TrajectoryGroup(cast(Iterable[Trajectory], trajectories))
            ]
        else:
            trajectory_groups = cast(list[TrajectoryGroup], list(trajectories))
        await self._api._log(
            self,
            trajectory_groups,
            split,
        )

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        await self._api._train_model(
            self, list(trajectory_groups), config, _config or {}
        )
