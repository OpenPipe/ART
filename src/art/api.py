import httpx

from . import dev
from .model import Model, TrainableModel
from .trajectories import TrajectoryGroup
from .types import TrainConfig


class API:
    def __init__(self, *, base_url: str) -> None:
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url)

    async def register(
        self,
        model: Model,
    ) -> None:
        """
        Registers a model with the API for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        response = await self._client.post(
            "/register", json={"model": model.model_dump()}
        )
        response.raise_for_status()

    async def _get_step(self, model: TrainableModel) -> int:
        response = await self._client.post(
            "/_get_step", json={"model": model.model_dump()}
        )
        response.raise_for_status()
        return response.json()

    async def _delete_checkpoints(
        self,
        model: TrainableModel,
        benchmark: str,
        benchmark_smoothing: float = 1.0,
    ) -> None:
        response = await self._client.post(
            "/_delete_checkpoints",
            json={
                "model": model.model_dump(),
                "benchmark": benchmark,
                "benchmark_smoothing": benchmark_smoothing,
            },
        )
        response.raise_for_status()

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        response = await self._client.post(
            "/_prepare_backend_for_training",
            json={"model": model.model_dump(), "config": config},
        )
        response.raise_for_status()
        return tuple(response.json())

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        response = await self._client.post(
            "/_log",
            json={
                "model": model.model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "split": split,
            },
        )
        response.raise_for_status()

    async def _train_model(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        _config: dev.TrainConfig,
    ) -> None:
        response = await self._client.post(
            "/_train_model",
            json={
                "model": model.model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "config": config,
                "_config": _config,
            },
        )
        response.raise_for_status()
