import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Literal
import warnings

from openai._types import NOT_GIVEN
from tqdm import auto as tqdm

from art.serverless.client import Client, ExperimentalTrainingConfig

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
                "Registering a non-trainable model with the Serverless backend is not supported."
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

    async def delete(
        self,
        model: "Model",
    ) -> None:
        """
        Deletes a model from the Backend.

        Args:
            model: An art.Model instance to delete.
        """
        from art import TrainableModel

        if not isinstance(model, TrainableModel):
            print(
                "Deleting a non-trainable model from the Serverless backend is not supported."
            )
            return
        assert model.id is not None, "Model ID is required"
        await self._client.models.delete(model_id=model.id)

    def _model_inference_name(self, model: "TrainableModel") -> str:
        assert model.entity is not None, "Model entity is required"
        return f"wandb-artifact:///{model.entity}/{model.project}/{model.name}"

    async def _get_step(self, model: "Model") -> int:
        if model.trainable:
            assert model.id is not None, "Model ID is required"
            async for checkpoint in self._client.models.checkpoints.list(
                limit=1, order="desc", model_id=model.id
            ):
                return checkpoint.step
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        # TODO: potentially implement benchmark smoothing
        assert model.id is not None, "Model ID is required"
        benchmark_values: dict[int, float] = {}
        async for checkpoint in self._client.models.checkpoints.list(model_id=model.id):
            benchmark_values[checkpoint.step] = checkpoint.metrics.get(
                benchmark, -float("inf")
            )
        max_step = max(benchmark_values.keys())
        max_benchmark_value = max(benchmark_values.values())
        if steps_to_delete := [
            step
            for step, benchmark_value in benchmark_values.items()
            if step != max_step and benchmark_value != max_benchmark_value
        ]:
            await self._client.models.checkpoints.delete(
                model_id=model.id,
                steps=steps_to_delete,
            )

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
        # TODO: log trajectories to local file system?
        if not model.trainable:
            print(f"Model {model.name} is not trainable; skipping logging.")
            return
        assert model.id is not None, "Model ID is required"
        await self._client.models.log(
            model_id=model.id, trajectory_groups=trajectory_groups, split=split
        )

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
            experimental_config=ExperimentalTrainingConfig(
                advantage_balance=dev_config.get("advantage_balance"),
                epsilon=dev_config.get("epsilon"),
                epsilon_high=dev_config.get("epsilon_high"),
                importance_sampling_level=dev_config.get("importance_sampling_level"),
                kimi_k2_tau=dev_config.get("kimi_k2_tau"),
                learning_rate=config.learning_rate,
                mask_prob_ratio=dev_config.get("mask_prob_ratio"),
                max_negative_advantage_importance_sampling_weight=dev_config.get(
                    "max_negative_advantage_importance_sampling_weight"
                ),
                ppo=dev_config.get("ppo"),
                precalculate_logprobs=dev_config.get("precalculate_logprobs"),
                scale_rewards=dev_config.get("scale_rewards"),
            ),
        )
        after: str | None = None
        num_sequences: int | None = None
        pbar: tqdm.tqdm | None = None
        while True:
            await asyncio.sleep(1)
            async for event in self._client.training_jobs.events.list(
                training_job_id=training_job.id, after=after or NOT_GIVEN
            ):
                if event.type == "gradient_step":
                    assert pbar is not None and num_sequences is not None
                    pbar.update(1)
                    pbar.set_postfix(event.data)
                    yield {**event.data, "num_gradient_steps": num_sequences}
                elif event.type == "training_started":
                    num_sequences = event.data["num_sequences"]
                    if pbar is None:
                        pbar = tqdm.tqdm(total=num_sequences, desc="train")
                    continue
                elif event.type == "training_ended":
                    return
                elif event.type == "training_failed":
                    error_message = event.data.get(
                        "error_message", "Training failed with an unknown error"
                    )
                    raise RuntimeError(f"Training job failed: {error_message}")
                after = event.id

    # ------------------------------------------------------------------
    # Experimental support for S3 and checkpoints
    # ------------------------------------------------------------------

    async def _experimental_pull_model_checkpoint(
        self,
        model: "TrainableModel",
        *,
        step: int | Literal["latest"] | None = None,
        local_path: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Pull a model checkpoint from W&B artifacts to a local path.

        For ServerlessBackend, this downloads the checkpoint from W&B artifact storage.

        Args:
            model: The model to pull checkpoint for.
            step: The step to pull. Can be an int for a specific step,
                 or "latest" to pull the latest checkpoint. If None, pulls latest.
            local_path: Local directory to save the checkpoint. If None, uses temporary directory.
            verbose: Whether to print verbose output.

        Returns:
            Path to the local checkpoint directory.
        """
        import os
        import tempfile

        import wandb

        assert model.id is not None, "Model ID is required"

        # If entity is not set, use the user's default entity from W&B
        api = wandb.Api(api_key=self._client.api_key)
        if model.entity is None:
            model.entity = api.default_entity
            if verbose:
                print(f"Using default W&B entity: {model.entity}")

        # Determine which step to use
        resolved_step: int
        if step is None or step == "latest":
            # Get latest checkpoint from API
            async for checkpoint in self._client.models.checkpoints.list(
                limit=1, order="desc", model_id=model.id
            ):
                resolved_step = checkpoint.step
                break
            else:
                raise ValueError(f"No checkpoints found for model {model.name}")
        else:
            resolved_step = step

        if verbose:
            print(f"Downloading checkpoint step {resolved_step} from W&B artifacts...")

        # Download from W&B artifacts
        # The artifact name follows the pattern: {entity}/{project}/{model_name}:step{step}
        artifact_name = (
            f"{model.entity}/{model.project}/{model.name}:step{resolved_step}"
        )

        # Use wandb API to download (api was already created above for entity lookup)
        artifact = api.artifact(artifact_name, type="lora")

        # Determine download path
        if local_path is None:
            # Create a temporary directory that won't be cleaned up automatically
            checkpoint_dir = os.path.join(
                tempfile.gettempdir(),
                "art_checkpoints",
                model.project,
                model.name,
                f"{resolved_step:04d}",
            )
        else:
            # Custom location - copy directly to local_path
            checkpoint_dir = local_path

        # Download artifact
        os.makedirs(checkpoint_dir, exist_ok=True)
        artifact.download(root=checkpoint_dir)
        if verbose:
            print(f"Downloaded checkpoint to {checkpoint_dir}")

        return checkpoint_dir

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
        """Deprecated. Use `_experimental_pull_model_checkpoint` instead."""
        warnings.warn(
            "_experimental_pull_from_s3 is deprecated. Use _experimental_pull_model_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
