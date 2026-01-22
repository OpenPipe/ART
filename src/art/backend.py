import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Literal, TypeAlias
import warnings

import httpx
from tqdm import auto as tqdm

from art.utils import log_http_errors
from art.utils.deployment import (
    DeploymentResult,
    Provider,
    TogetherDeploymentConfig,
    WandbDeploymentConfig,
)

from . import dev
from .trajectories import Trajectory, TrajectoryGroup
from .types import SFTConfig, TrainConfig, TrainResult

if TYPE_CHECKING:
    from .model import Model, TrainableModel

# Type aliases for models with any config/state type (for backend method signatures)
AnyModel: TypeAlias = "Model[Any, Any]"
AnyTrainableModel: TypeAlias = "TrainableModel[Any, Any]"


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
        model: AnyModel,
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        response = await self._client.post("/register", json=model.safe_model_dump())
        response.raise_for_status()

    async def _get_step(self, model: AnyTrainableModel) -> int:
        response = await self._client.post("/_get_step", json=model.safe_model_dump())
        response.raise_for_status()
        return response.json()

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        response = await self._client.post(
            "/_delete_checkpoint_files",
            json={"model": model.safe_model_dump(), "steps_to_keep": steps_to_keep},
        )
        response.raise_for_status()

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
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

    def _model_inference_name(self, model: AnyModel, step: int | None = None) -> str:
        """Return the inference name for a model checkpoint.

        Override in subclasses to provide backend-specific naming.
        Default implementation returns model.name with optional @step suffix.
        """
        base_name = model.inference_model_name or model.name
        if step is not None:
            return f"{base_name}@{step}"
        return base_name

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        """Train the model on the given trajectory groups.

        This method is not implemented in the base Backend class. Use
        LocalBackend, ServerlessBackend, or TinkerBackend directly for training.

        Raises:
            NotImplementedError: Always raised. Use a concrete backend instead.
        """
        raise NotImplementedError(
            "The base Backend class does not support the train() method. "
            "Use LocalBackend, ServerlessBackend, or TinkerBackend directly. "
            "If you are using the 'art run' server, consider using LocalBackend "
            "in-process instead."
        )

    async def _train_model(
        self,
        model: AnyTrainableModel,
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

    async def _train_sft(
        self,
        model: "TrainableModel",
        trajectories: Iterable[Trajectory],
        config: SFTConfig,
        dev_config: dev.SFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        raise NotImplementedError(
            "SFT training is not yet implemented. "
            "This method will be available in a future release."
        )
        # This yield is unreachable but makes this an async generator
        yield  # type: ignore

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    @log_http_errors
    async def _experimental_pull_from_s3(
        self,
        model: AnyModel,
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
    ) -> None:
        """Download the model directory from S3 into file system where the LocalBackend is running. Right now this can be used to pull trajectory logs for processing or model checkpoints.

        .. deprecated::
            This method is deprecated. Use `_experimental_pull_model_checkpoint` instead.

        Args:
            only_step: If specified, only pull this specific step. Can be an int for a specific step,
                      or "latest" to pull only the latest checkpoint. If None, pulls all steps.
        """
        warnings.warn(
            "_experimental_pull_from_s3 is deprecated. Use _experimental_pull_model_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        model: AnyModel,
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
        model: AnyModel,
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
