import httpx
import json
from tqdm import auto as tqdm
from typing import AsyncIterator, TYPE_CHECKING
import os

from art.errors import UnsupportedLoRADeploymentProviderError
from art.utils import get_model_step, log_http_errors
from art.utils.deploy_model import (
    LoRADeploymentJob,
    check_together_job_status,
    deploy_together,
    find_existing_together_job_id,
    wait_for_together_job,
)
from art.utils.output_dirs import get_default_art_path
from art.utils.s3 import archive_and_presign_step_url, pull_model_from_s3

from . import dev
from .trajectories import TrajectoryGroup
from .types import TrainConfig

if TYPE_CHECKING:
    from .model import Model, TrainableModel
    from art.utils.deploy_model import LoRADeploymentProvider


class Backend:
    def __init__(
        self,
        *,
        base_url: str = "http://0.0.0.0:7999",
        path: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url)
        self._path = path or get_default_art_path()
        os.makedirs(self._path, exist_ok=True)

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
                "model": model.model_dump(),
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
                "model": model.model_dump(),
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

    @log_http_errors
    async def _experimental_deploy_remote(
        self,
        deploy_to: "LoRADeploymentProvider",
        model: "Model",
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
                "model": model.model_dump(),
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

    # shared between LocalBackend and remote backends
    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: TrainableModel,
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
        if pull_s3:
            # pull the latest step from S3
            await pull_model_from_s3(
                model_name=model.name,
                project=model.project,
                step=step,
                s3_bucket=s3_bucket,
                prefix=prefix,
                verbose=verbose,
                art_path=self._path,
            )

        if step is None:
            step = get_model_step(model, self._path)

        presigned_url = await archive_and_presign_step_url(
            model_name=model.name,
            project=model.project,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
        )

        if deploy_to == LoRADeploymentProvider.TOGETHER:
            existing_job_id = await find_existing_together_job_id(model, step)
            existing_job = None
            if existing_job_id is not None:
                existing_job = await check_together_job_status(
                    existing_job_id, verbose=verbose
                )

            if not existing_job or existing_job.status == "Failed":
                deployment_result = await deploy_together(
                    model=model,
                    presigned_url=presigned_url,
                    step=step,
                    verbose=verbose,
                )
                job_id = deployment_result["data"]["job_id"]
            else:
                job_id = existing_job_id
                print(
                    f"Previous deployment for {model.name} at step {step} has status '{existing_job.status}', skipping redployment"
                )

            if wait_for_completion:
                return await wait_for_together_job(job_id, verbose=verbose)
            else:
                return await check_together_job_status(job_id, verbose=verbose)

        raise UnsupportedLoRADeploymentProviderError(
            f"Unsupported deployment option: {deploy_to}"
        )
