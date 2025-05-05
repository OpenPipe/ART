import httpx
import json
from tqdm import auto as tqdm
from typing import AsyncIterator, TYPE_CHECKING

from art.utils import log_http_errors

from . import dev
from .trajectories import TrajectoryGroup
from .types import TrainConfig

if TYPE_CHECKING:
    from .model import Model, TrainableModel


class Backend:
    def __init__(
        self,
        *,
        base_url: str = "http://0.0.0.0:7999",
    ) -> None:
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url)

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        response = await self._client.post("/register", json=model.model_dump())
        response.raise_for_status()

    async def _get_step(self, model: "TrainableModel") -> int:
        response = await self._client.post("/_get_step", json=model.model_dump())
        response.raise_for_status()
        return response.json()

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        response = await self._client.post(
            "/_delete_checkpoints",
            json=model.model_dump(),
            params={"benchmark": benchmark, "benchmark_smoothing": benchmark_smoothing},
        )
        response.raise_for_status()

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        response = await self._client.post(
            "/_prepare_backend_for_training",
            json={"model": model.model_dump(), "config": config},
            timeout=600,
        )
        response.raise_for_status()
        [base_url, api_key] = tuple(response.json())

        return [base_url, api_key]

    async def _log(
        self,
        model: "Model",
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
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        async with self._client.stream(
            "POST",
            "/_train_model",
            json={
                "model": model.model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "config": config.model_dump(),
                "dev_config": dev_config,
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            pbar: tqdm.tqdm | None = None
            async for line in response.aiter_lines():
                result = json.loads(line)
                yield result
                num_gradient_steps = result.pop("num_gradient_steps")
                if pbar is None:
                    pbar = tqdm.tqdm(total=num_gradient_steps, desc="train")
                pbar.update(1)
                pbar.set_postfix(result)
            if pbar is not None:
                pbar.close()

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    @log_http_errors
    async def _experimental_pull_from_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Download the model directory from S3 into file system where the LocalBackend is running. Right now this can be used to pull trajectory logs for processing or model checkpoints."""
        response = await self._client.post(
            "/_experimental_pull_from_s3",
            json={
                "model": model.model_dump(),
                "s3_bucket": s3_bucket,
                "prefix": prefix,
                "verbose": verbose,
                "delete": delete,
            },
            timeout=600,
        )
        response.raise_for_status()

    @log_http_errors
    async def _experimental_push_to_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Upload the model directory from the file system where the LocalBackend is running to S3."""
        response = await self._client.post(
            "/_experimental_push_to_s3",
            json={
                "model": model.model_dump(),
                "s3_bucket": s3_bucket,
                "prefix": prefix,
                "verbose": verbose,
                "delete": delete,
            },
            timeout=600,
        )
        response.raise_for_status()
