import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Iterable, Literal
import warnings

from openai._types import NOT_GIVEN
from tqdm import auto as tqdm

from art.serverless.client import Client, ExperimentalTrainingConfig

from .. import dev
from ..backend import AnyTrainableModel, Backend
from ..trajectories import TrajectoryGroup
from ..types import ServerlessTrainResult, TrainConfig
from ..utils.record_provenance import record_provenance

if TYPE_CHECKING:
    import wandb

    from ..model import Model, TrainableModel


def _extract_step_from_wandb_artifact(artifact: "wandb.Artifact") -> int | None:
    """Extract step number from a W&B artifact's aliases."""
    for alias in artifact.aliases:
        if alias.startswith("step"):
            try:
                return int(alias[4:])
            except ValueError:
                pass
    return None


class ServerlessBackend(Backend):
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        self._base_url = str(client.base_url)
        self._client = client

    async def close(self) -> None:
        await self._client.close()  # ty:ignore[possibly-missing-attribute]

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
        client_model = await self._client.models.create(  # ty:ignore[possibly-missing-attribute]
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
        await self._client.models.delete(model_id=model.id)  # ty:ignore[possibly-missing-attribute]

    def _model_inference_name(self, model: "Model", step: int | None = None) -> str:
        """Return the inference name for a model checkpoint.

        Args:
            model: The model.
            step: If provided, returns name for specific checkpoint using
                  W&B artifact versioning (e.g., :step5). If None, returns
                  name for latest checkpoint (default, backwards compatible).
        """
        assert model.entity is not None, "Model entity is required"
        base_name = f"wandb-artifact:///{model.entity}/{model.project}/{model.name}"
        if step is not None:
            return f"{base_name}:step{step}"
        return base_name

    async def _get_step(self, model: "Model") -> int:
        if model.trainable:
            assert model.id is not None, "Model ID is required"
            async for checkpoint in self._client.models.checkpoints.list(  # ty:ignore[possibly-missing-attribute]
                limit=1, order="desc", model_id=model.id
            ):
                return checkpoint.step
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        """Delete checkpoint files, keeping only the specified steps."""
        assert model.id is not None, "Model ID is required"
        # Get all checkpoint steps
        all_steps: list[int] = []
        async for checkpoint in self._client.models.checkpoints.list(model_id=model.id):  # ty:ignore[possibly-missing-attribute]
            all_steps.append(checkpoint.step)
        # Delete all steps not in steps_to_keep
        if steps_to_delete := [step for step in all_steps if step not in steps_to_keep]:
            await self._client.models.checkpoints.delete(  # ty:ignore[possibly-missing-attribute]
                model_id=model.id,
                steps=steps_to_delete,
            )

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        return str(self._base_url), self._client.api_key  # ty:ignore[possibly-missing-attribute]

    # Note: _log() method has been moved to the Model class (frontend)
    # Trajectories are now saved locally by the Model.log() method

    async def train(  # type: ignore[override]
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training parameters
        learning_rate: float = 5e-6,
        beta: float = 0.0,
        # RL algorithm settings
        ppo: bool = False,
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        # Advantage computation
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        # Importance sampling
        importance_sampling_level: Literal[
            "token", "sequence", "average", "geometric_average"
        ] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        # Experimental parameters
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
        # Verbosity
        verbose: bool = False,
    ) -> ServerlessTrainResult:
        """Train the model on the given trajectory groups.

        This is the recommended way to train models. Unlike model.train(), this
        method does NOT automatically log trajectories or metrics. Call model.log()
        explicitly before and/or after training if you want to log data.

        Args:
            model: The trainable model to train.
            trajectory_groups: Batches of trajectories to train on.
            learning_rate: Learning rate for training. Defaults to 5e-6.
            beta: KL penalty coefficient. Defaults to 0.0.
            ppo: Whether to use PPO clipping. Defaults to False.
            epsilon: Clip epsilon for importance sampling. Defaults based on ppo.
            epsilon_high: Asymmetric upper clip bound. Defaults to epsilon.
            advantage_balance: Balance between negative and positive advantages
                in range [-1.0, 1.0]. Defaults to 0.0 (balanced).
            scale_rewards: Whether to scale rewards by standard deviation.
                Defaults to True.
            importance_sampling_level: Level at which to compute importance
                sampling weights. Defaults to "token".
            max_negative_advantage_importance_sampling_weight: Maximum weight
                for negative advantage samples.
            mask_prob_ratio: Whether to mask probability ratios. Defaults to False.
            kimi_k2_tau: Tau parameter for Kimi K2 algorithm.
            precalculate_logprobs: Whether to precalculate logprobs.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            ServerlessTrainResult with step number, training metrics, and artifact name.

        Example:
            # Before (deprecated):
            await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))

            # After (recommended):
            await model.log(trajectory_groups, split="train")
            result = await backend.train(model, trajectory_groups, learning_rate=5e-6)
            # Optionally log training metrics:
            # await model.log(metrics=result.metrics, step=result.step)
        """
        groups_list = list(trajectory_groups)

        # Record provenance in W&B
        wandb_run = model._get_wandb_run()
        if wandb_run is not None:
            record_provenance(wandb_run, "serverless-rl")

        # Build config objects from explicit kwargs
        config = TrainConfig(learning_rate=learning_rate, beta=beta)
        dev_config: dev.TrainConfig = {
            "advantage_balance": advantage_balance,
            "importance_sampling_level": importance_sampling_level,
            "mask_prob_ratio": mask_prob_ratio,
            "ppo": ppo,
            "precalculate_logprobs": precalculate_logprobs,
            "scale_rewards": scale_rewards,
        }
        # Only include optional fields if they're set
        if epsilon is not None:
            dev_config["epsilon"] = epsilon
        if epsilon_high is not None:
            dev_config["epsilon_high"] = epsilon_high
        if max_negative_advantage_importance_sampling_weight is not None:
            dev_config["max_negative_advantage_importance_sampling_weight"] = (
                max_negative_advantage_importance_sampling_weight
            )
        if kimi_k2_tau is not None:
            dev_config["kimi_k2_tau"] = kimi_k2_tau

        # Collect metrics from training
        training_metrics: list[dict[str, float]] = []
        async for metrics in self._train_model(
            model, groups_list, config, dev_config, verbose
        ):
            training_metrics.append(metrics)

        # Aggregate metrics
        avg_metrics: dict[str, float] = {}
        if training_metrics:
            avg_metrics = {
                k: sum(d.get(k, 0) for d in training_metrics)
                / sum(1 for d in training_metrics if k in d)
                for k in {k for d in training_metrics for k in d}
                if k != "num_gradient_steps"
            }

        # Get step and artifact name
        step = await self._get_step(model)
        artifact_name: str | None = None
        if model.entity is not None:
            artifact_name = f"{model.entity}/{model.project}/{model.name}:step{step}"

        return ServerlessTrainResult(
            step=step,
            metrics=avg_metrics,
            artifact_name=artifact_name,
        )

    async def _train_model(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        assert model.id is not None, "Model ID is required"
        training_job = await self._client.training_jobs.create(  # ty:ignore[possibly-missing-attribute]
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
            async for event in self._client.training_jobs.events.list(  # ty:ignore[possibly-missing-attribute]
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
        api = wandb.Api(api_key=self._client.api_key)  # ty:ignore[possibly-missing-attribute]
        if model.entity is None:
            model.entity = api.default_entity
            if verbose:
                print(f"Using default W&B entity: {model.entity}")

        # Determine which step to use
        resolved_step: int
        if step is None or step == "latest":
            # Get latest checkpoint from API
            async for checkpoint in self._client.models.checkpoints.list(  # ty:ignore[possibly-missing-attribute]
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
        """Push model checkpoints from W&B artifacts to S3.

        Downloads checkpoint(s) from W&B and uploads them to S3.

        Args:
            model: The model whose checkpoints to push.
            s3_bucket: S3 bucket name. If None, uses BACKUP_BUCKET env var.
            prefix: Optional S3 prefix path.
            verbose: Whether to print verbose output.
            delete: Whether to delete files from S3 that don't exist in source.
        """
        from art.utils.s3 import build_s3_path, ensure_bucket_exists, s3_sync

        assert model.id is not None, "Model ID is required"

        # Get all checkpoint steps
        steps: list[int] = []
        async for checkpoint in self._client.models.checkpoints.list(  # ty:ignore[possibly-missing-attribute]
            model_id=model.id, order="asc"
        ):
            steps.append(checkpoint.step)

        if not steps:
            if verbose:
                print("No checkpoints found to push.")
            return

        await ensure_bucket_exists(s3_bucket)

        for step in steps:
            if verbose:
                print(f"Pushing checkpoint step {step} to S3...")

            # Pull from W&B to local temp dir
            checkpoint_dir = await self._experimental_pull_model_checkpoint(
                model,  # type: ignore[arg-type]
                step=step,
                verbose=verbose,
            )

            # Push to S3
            s3_path = build_s3_path(
                model_name=model.name,
                project=model.project,
                step=step,
                s3_bucket=s3_bucket,
                prefix=prefix,
            )
            await s3_sync(checkpoint_dir, s3_path, verbose=verbose, delete=delete)

        if verbose:
            print(f"Successfully pushed {len(steps)} checkpoint(s) to S3.")

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

        Pulls the source checkpoint from W&B artifacts (or S3 if from_s3_bucket
        is provided) and uploads it as a W&B artifact for the destination model.

        Note: This uploads the artifact directly to W&B. The ServerlessBackend's
        checkpoint tracking may not immediately reflect the forked checkpoint
        until the next training step.

        Args:
            model: The destination model to fork to.
            from_model: The name of the source model to fork from.
            from_project: The project of the source model. Defaults to model.project.
            from_s3_bucket: Optional S3 bucket to pull the checkpoint from.
            not_after_step: If provided, uses the latest checkpoint <= this step.
            verbose: Whether to print verbose output.
            prefix: Optional S3 prefix for bucket operations.
        """
        import os
        import tempfile

        import wandb

        from_project = from_project or model.project

        if from_s3_bucket is not None:
            # Pull from S3
            from art.utils.s3 import build_s3_path, ensure_bucket_exists, s3_sync
            from art.utils.s3_checkpoint_utils import (
                get_checkpoint_step_not_after_from_s3,
                get_latest_checkpoint_step_from_s3,
            )

            if not_after_step is None:
                target_step = await get_latest_checkpoint_step_from_s3(
                    model_name=from_model,
                    project=from_project,
                    s3_bucket=from_s3_bucket,
                    prefix=prefix,
                )
            else:
                target_step = await get_checkpoint_step_not_after_from_s3(
                    model_name=from_model,
                    project=from_project,
                    not_after_step=not_after_step,
                    s3_bucket=from_s3_bucket,
                    prefix=prefix,
                )

            if target_step is None:
                raise ValueError(
                    f"No suitable checkpoint found in S3 for model {from_model}"
                )

            if verbose:
                print(f"Pulling checkpoint step {target_step} from S3...")

            checkpoint_dir = os.path.join(
                tempfile.gettempdir(),
                "art_fork_checkpoints",
                from_project,
                from_model,
                f"{target_step:04d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            s3_path = build_s3_path(
                model_name=from_model,
                project=from_project,
                step=target_step,
                s3_bucket=from_s3_bucket,
                prefix=prefix,
            )
            await ensure_bucket_exists(from_s3_bucket)
            await s3_sync(s3_path, checkpoint_dir, verbose=verbose)
            selected_step = target_step
        else:
            # Pull from W&B artifacts
            api = wandb.Api(api_key=self._client.api_key)  # ty:ignore[possibly-missing-attribute]
            from_entity = model.entity or api.default_entity

            # Iterate all artifact versions to find the best step.
            # We avoid relying on the W&B `:latest` alias because it
            # may not correspond to the highest training step.
            collection_path = f"{from_entity}/{from_project}/{from_model}"
            versions = api.artifacts("lora", collection_path)

            best_step: int | None = None
            best_artifact = None
            for version in versions:
                step_num = _extract_step_from_wandb_artifact(version)
                if step_num is None:
                    continue
                if not_after_step is not None and step_num > not_after_step:
                    continue
                if best_step is None or step_num > best_step:
                    best_step = step_num
                    best_artifact = version

            if best_step is None or best_artifact is None:
                if not_after_step is not None:
                    raise ValueError(
                        f"No checkpoints found not after step {not_after_step} "
                        f"for model {from_model}"
                    )
                raise ValueError(f"No checkpoints found for model {from_model}")
            selected_step = best_step
            artifact = best_artifact

            checkpoint_dir = os.path.join(
                tempfile.gettempdir(),
                "art_fork_checkpoints",
                from_project,
                from_model,
                f"{selected_step:04d}" if selected_step is not None else "latest",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            artifact.download(root=checkpoint_dir)

            if verbose:
                print(f"Downloaded source checkpoint step {selected_step} from W&B")

        # Upload as W&B artifact for the destination model
        assert model.entity is not None, "Model entity is required"

        if verbose:
            print(f"Uploading forked checkpoint as W&B artifact for {model.name}...")

        wandb.login(key=self._client.api_key)  # ty:ignore[possibly-missing-attribute]
        run = wandb.init(
            project=model.project,
            entity=model.entity,
            job_type="checkpoint-fork",
            name=f"fork-{from_model}-to-{model.name}",
            settings=wandb.Settings(silent=True),
        )
        assert run is not None

        dest_artifact = wandb.Artifact(name=model.name, type="lora")
        dest_artifact.add_dir(checkpoint_dir)
        aliases = ["latest"]
        if selected_step is not None:
            aliases.insert(0, f"step{selected_step}")
        run.log_artifact(dest_artifact, aliases=aliases)
        run.finish()

        if verbose:
            print(
                f"Successfully forked checkpoint from {from_model} "
                f"(step {selected_step}) to {model.name}"
            )
