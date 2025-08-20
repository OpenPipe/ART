import json
from typing import TYPE_CHECKING, AsyncIterator, Literal

import httpx
from tqdm import auto as tqdm

from art.utils import log_http_errors
from art.utils.deploy_model import LoRADeploymentJob, LoRADeploymentProvider

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

    async def close(self) -> None:
        """
        If running vLLM in a separate process, this will kill that process and close the communication threads.
        """
        response = await self._client.post("/close", timeout=None)
        response.raise_for_status()

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        response = await self._client.post("/register", json=model.safe_model_dump())
        response.raise_for_status()

    async def _get_step(self, model: "TrainableModel") -> int:
        response = await self._client.post("/_get_step", json=model.safe_model_dump())
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
            json=model.safe_model_dump(),
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
            json={"model": model.safe_model_dump(), "config": config},
            timeout=600,
        )
        response.raise_for_status()
        base_url, api_key = tuple(response.json())
        return base_url, api_key

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        response = await self._client.post(
            "/_log",
            json={
                "model": model.safe_model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "split": split,
            },
            timeout=None,
        )
        response.raise_for_status()

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        async with self._client.stream(
            "POST",
            "/_train_model",
            json={
                "model": model.safe_model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "config": config.model_dump(),
                "dev_config": dev_config,
                "verbose": verbose,
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
        only_step: int | Literal["latest"] | None = None,
    ) -> None:
        """Download the model directory from S3 into file system where the LocalBackend is running. Right now this can be used to pull trajectory logs for processing or model checkpoints.

        Args:
            only_step: If specified, only pull this specific step. Can be an int for a specific step,
                      or "latest" to pull only the latest checkpoint. If None, pulls all steps.
        """
        response = await self._client.post(
            "/_experimental_pull_from_s3",
            json={
                "model": model.safe_model_dump(),
                "s3_bucket": s3_bucket,
                "prefix": prefix,
                "verbose": verbose,
                "delete": delete,
                "only_step": only_step,
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
                "model": model.safe_model_dump(),
                "s3_bucket": s3_bucket,
                "prefix": prefix,
                "verbose": verbose,
                "delete": delete,
            },
            timeout=600,
        )
        response.raise_for_status()

    @log_http_errors
    async def _experimental_fork_checkpoint(
        self,
        model: "Model",
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        """Fork a checkpoint from another model to initialize this model.

        Args:
            model: The model to fork to.
            from_model: The name of the model to fork from.
            from_project: The project of the model to fork from. Defaults to model.project.
            from_s3_bucket: Optional S3 bucket to pull the checkpoint from. If provided,
                will pull from S3 first. Otherwise, will fork from local disk.
            not_after_step: Optional step number. If provided, will copy the last saved
                checkpoint that is <= this step. Otherwise, copies the latest checkpoint.
            verbose: Whether to print verbose output.
            prefix: Optional S3 prefix for the bucket.
        """
        response = await self._client.post(
            "/_experimental_fork_checkpoint",
            json={
                "model": model.safe_model_dump(),
                "from_model": from_model,
                "from_project": from_project,
                "from_s3_bucket": from_s3_bucket,
                "not_after_step": not_after_step,
                "verbose": verbose,
                "prefix": prefix,
            },
            timeout=600,
        )
        response.raise_for_status()

    @log_http_errors
    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: "TrainableModel",
        step: int | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        pull_s3: bool = True,
        wait_for_completion: bool = True,
    ) -> LoRADeploymentJob:
        """
        Deploy the model's latest checkpoint to a hosted inference endpoint.

        Together is currently the only supported provider. See link for supported base models:
        https://docs.together.ai/docs/lora-inference#supported-base-models
        """
        response = await self._client.post(
            "/_experimental_deploy",
            json={
                "deploy_to": deploy_to,
                "model": model.safe_model_dump(),
                "step": step,
                "s3_bucket": s3_bucket,
                "prefix": prefix,
                "verbose": verbose,
                "pull_s3": pull_s3,
                "wait_for_completion": wait_for_completion,
            },
            timeout=600,
        )
        response.raise_for_status()
        return LoRADeploymentJob(**response.json())
