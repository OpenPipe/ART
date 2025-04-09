from dataclasses import dataclass
from openai import AsyncOpenAI
from typing import Iterable, TYPE_CHECKING

from .config.model import ModelConfig
from .config.openai_server import OpenAIServerConfig
from .openai import patch_openai
from .types import BaseModel, Trajectory, TuneConfig, Verbosity


if TYPE_CHECKING:
    from .local.api import LocalAPI


@dataclass
class Model:
    api: "LocalAPI"
    name: str
    base_model: BaseModel
    _config: ModelConfig | None = None

    async def openai_client(self) -> AsyncOpenAI:
        """
        OpenAI client to a managed inference service.

        Returns:
            An asynchronous OpenAI client.
        """
        return await self._openai_client(_config=None)

    async def _openai_client(
        self,
        _config: OpenAIServerConfig | None = None,
    ) -> AsyncOpenAI:
        """
        Private method for an OpenAI client to a managed inference service.

        Args:
            _config: An OpenAIServerConfig object. May be subject to breaking changes at any time.
                Use at your own risk.

        Returns:
            An asynchronous OpenAI client.
        """
        client = await self.api._get_openai_client(self, _config)
        return patch_openai(client)

    async def get_iteration(self) -> int:
        """
        Get the model's current training iteration.
        """
        return await self.api._get_iteration(self)

    async def clear_iterations(
        self,
        benchmark: str = "val/reward",
        benchmark_smoothing: float = 1.0,
        verbosity: Verbosity = 1,
    ) -> None:
        """
        Delete all but the latest and best iteration checkpoints.

        Args:
            benchmark: The benchmark to use to determine the best iteration.
            benchmark_smoothing: Smoothing factor (0-1) that controls how much to reduce
                variance when determining the best iteration. Defaults to 1.0 (no smoothing).
            verbosity: Verbosity level.
        """
        await self.api._clear_iterations(
            self, benchmark, benchmark_smoothing, verbosity
        )

    async def log(
        self,
        trajectory_groups: Iterable[Iterable[Trajectory | BaseException]],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        await self.api._log(self, [list(group) for group in trajectory_groups], split)

    async def tune(
        self,
        trajectory_groups: Iterable[Iterable[Trajectory | BaseException]],
        config: TuneConfig = TuneConfig(),
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration with options for the optimizer,
                loss, other hyperparameters, logging, etc.
        """
        await self.api._tune_model(
            self, [list(group) for group in trajectory_groups], config
        )
