import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Literal

from art.client import Client
from art.utils.deploy_model import LoRADeploymentJob, LoRADeploymentProvider

from .. import dev
from ..backend import Backend
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class ServerlessBackend(Backend):
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        super().__init__(base_url=str(client.base_url))
        self._client = client

    async def close(self) -> None:
        await self._client.close()

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        from art import TrainableModel

        if not isinstance(model, TrainableModel):
            print(
                "Registering a non-trainable model with the WandB backend is not supported."
            )
            return

        client_model = await self._client.models.create(
            entity=model.entity,
            project=model.project,
            name=model.name,
            base_model=model.base_model,
            return_existing=True,
        )
        model.id = client_model.id
        model.entity = client_model.entity

    def _model_inference_name(self, model: "TrainableModel") -> str:
        assert model.entity is not None, "Model entity is required"
        return f"{model.entity}/{model.project}/{model.name}"

    async def _get_step(self, model: "TrainableModel") -> int:
        assert model.id is not None, "Model ID is required"
        checkpoint = await self._client.checkpoints.retrieve(
            model_id=model.id, step="latest"
        )
        return checkpoint.step

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        raise NotImplementedError

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        return str(self._base_url), self._client.api_key

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        raise NotImplementedError

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        assert model.id is not None, "Model ID is required"
        training_job = await self._client.training_jobs.create(
            model_id=model.id,
            trajectory_groups=trajectory_groups,
        )
        while training_job.status != "COMPLETED":
            await asyncio.sleep(1)
            training_job = await self._client.training_jobs.retrieve(training_job.id)
            yield {"num_gradient_steps": 1}

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

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
        raise NotImplementedError

    async def _experimental_push_to_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
