import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import secrets
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Literal, cast
import warnings

from art.adapter_leases import pin_inference_step, pinned_inference_step
from art.serverless.client import Client

from .. import dev
from .._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
)
from ..backend import AnyTrainableModel
from ..metrics_taxonomy import (
    TRAIN_GRADIENT_STEPS_KEY,
    build_training_summary_metrics,
    summarize_trajectory_groups,
)
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import (
    ServerlessTrainResult,
    TrainConfig,
    TrainSFTConfig,
)
from ..utils import wandb_sdk
from ..utils.record_provenance import record_provenance

if TYPE_CHECKING:
    from wandb.sdk.artifacts.artifact import Artifact

    from ..model import Model, TrainableModel
    from ..training import PackedInputCaptureRef, TrainingRunSpec
    from .native_training import RemoteTrainingClient


def _extract_step_from_wandb_artifact(artifact: "Artifact") -> int | None:
    """Extract step number from a W&B artifact's aliases."""
    for alias in artifact.aliases:
        if alias.startswith("step"):
            try:
                return int(alias[4:])
            except ValueError:
                pass
    return None


def _training_run_spec(model: AnyTrainableModel) -> "TrainingRunSpec":
    from art.megatron.lora_config import default_lora_rank_for_handler
    from art.megatron.model_support import (
        default_target_modules_for_model,
        get_model_support_handler,
    )
    from art.training import AdapterSpec, TrainingRunSpec

    configured = model.lora_config or {}
    allow_unvalidated = bool(
        (model._internal_config or {}).get("allow_unvalidated_arch", False)
    )
    handler = get_model_support_handler(
        model.base_model,
        allow_unvalidated_arch=allow_unvalidated,
    )
    return TrainingRunSpec(
        base_model=model.base_model,
        adapter=AdapterSpec(
            rank=int(configured.get("rank") or default_lora_rank_for_handler(handler)),
            target_modules=tuple(
                configured.get("target_modules")
                or default_target_modules_for_model(
                    model.base_model,
                    allow_unvalidated_arch=allow_unvalidated,
                )
            ),
        ),
        seed=configured.get("random_state"),
    )


class ServerlessBackend:
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        self._base_url = str(client.base_url)
        self._client = client
        self._training_clients: dict[str, RemoteTrainingClient] = {}
        self._native_steps: dict[str, int] = {}
        self._native_artifacts: dict[str, str] = {}
        self._retain_next_inputs: set[str] = set()
        self._retained_inputs: dict[str, list["PackedInputCaptureRef"]] = {}

    def logs_sft_metrics_remotely(self) -> bool:
        return False

    def pipeline_autotuner_inference_observer(self) -> Literal["rollout_supply"]:
        return "rollout_supply"

    async def close(self) -> None:
        try:
            await asyncio.gather(
                *(client.close() for client in self._training_clients.values())
            )
        finally:
            self._training_clients.clear()
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
            name=model._storage_name(),
            base_model=model.base_model,
            return_existing=True,
        )
        model.id = client_model.id
        model.entity = client_model.entity
        model.run_id = client_model.run_id

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
                  name for the pinned checkpoint when running inside an
                  adapter_lease, otherwise latest checkpoint.
        """
        assert model.entity is not None, "Model entity is required"
        if step is None:
            step = pinned_inference_step(model._storage_name())
        base_name = (
            f"wandb-artifact:///{model.entity}/{model.project}/{model._storage_name()}"
        )
        if step is not None:
            return f"{base_name}:step{step}"
        return base_name

    @asynccontextmanager
    async def adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        async with pin_inference_step(model._storage_name(), step):
            yield

    @asynccontextmanager
    async def exact_adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        async with pin_inference_step(model._storage_name(), step):
            yield

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

    async def create_training_client(
        self,
        *,
        request_id: str,
        run_name: str,
        spec: "TrainingRunSpec",
        poll_interval_s: float = 0.1,
    ) -> "RemoteTrainingClient":
        """Resolve one durable native run and retain its operation identities."""

        from .native_training import RemoteTrainingClient

        return await RemoteTrainingClient.resolve(
            self._client.training_runs,  # ty:ignore[possibly-missing-attribute]
            request_id=request_id,
            run_name=run_name,
            spec=spec,
            poll_interval_s=poll_interval_s,
        )

    async def training_client(self, model: AnyTrainableModel) -> "RemoteTrainingClient":
        """Return the one native sequenced client owned by this model run."""

        key = model._storage_name()
        client = self._training_clients.get(key)
        if client is not None:
            return client
        spec = _training_run_spec(model)
        request_id = (
            "resolve-"
            + hashlib.sha256(
                json.dumps(
                    {
                        "project": model.project,
                        "entity": model.entity,
                        "run_name": model.run_name,
                        "spec": spec.model_dump(mode="json"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        client = await self.create_training_client(
            request_id=request_id,
            run_name=model.run_name,
            spec=spec,
        )
        self._training_clients[key] = client
        model.run_id = client.run_id
        return client

    def retain_next_packed_input(self, model: AnyTrainableModel) -> None:
        """Keep the next exact packed input alive for a later matched replay."""

        self._retain_next_inputs.add(model._storage_name())

    def retained_packed_inputs(
        self, model: AnyTrainableModel
    ) -> tuple["PackedInputCaptureRef", ...]:
        return tuple(self._retained_inputs.get(model._storage_name(), ()))

    # Note: _log() method has been moved to the Model class (frontend)
    # Trajectories are now saved locally by the Model.log() method

    async def train(  # type: ignore[override]
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training parameters
        learning_rate: float = 5e-6,
        loss_fn: Literal["cispo", "ppo"] | None = None,
        loss_fn_config: dict | None = None,
        normalize_advantages: bool = True,
        adam_params: object | None = None,
        # KL-penalized advantage adjustment
        kl_penalty_coef: float = 0.0,
        kl_penalty_reference_step: int | None = None,
        kl_penalty_source: Literal["current_learner", "sample"] | None = None,
        kl_penalty_step_lag: int | None = None,
        kl_ref_adapter_path: str | None = None,
        # RL algorithm settings
        ppo: bool | None = None,
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
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        truncated_importance_sampling: float | None = None,
        scale_learning_rate_by_reward_std_dev: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        packed_sequence_length: int | None = None,
        num_trajectories_learning_rate_multiplier_power: float = 0.0,
        # Checkpoint behavior
        save_checkpoint: bool = True,
        optimizer_save_interval: int = 5,
        # Verbosity
        verbose: bool = False,
    ) -> ServerlessTrainResult:
        """Train the model on the given trajectory groups.

        This method does NOT automatically log trajectories or metrics. Call
        model.log() explicitly before and/or after training if you want to log
        data.

        Args:
            model: The trainable model to train.
            trajectory_groups: Batches of trajectories to train on.
            learning_rate: Learning rate for training. Defaults to 5e-6.
            loss_fn: RL loss function. ServerlessBackend supports "cispo" and
                "ppo". If unset, the legacy ppo argument is used.
            loss_fn_config: Additional loss-function config. Not supported by
                ServerlessBackend.
            normalize_advantages: Backward-compatible alias for reward std scaling.
                When False, ServerlessBackend centers rewards but does not divide
                by group reward std dev.
            adam_params: Custom optimizer params. Not supported by
                ServerlessBackend.
            kl_penalty_coef: Coefficient for KL-penalized advantage adjustment.
                Defaults to 0.0 (disabled).
            kl_penalty_reference_step: Checkpoint step of the training model to
                use as the KL reference. When omitted, the backend may use
                kl_ref_adapter_path or its default reference policy.
            kl_penalty_source: Which policy's logprobs to compare against the
                reference policy. When omitted, defaults to "sample" if KL is
                enabled and "current_learner" otherwise.
            kl_penalty_step_lag: Moving KL reference lag. The serverless
                backend resolves this as max(0, current_step - lag). Mutually
                exclusive with kl_penalty_reference_step.
            kl_ref_adapter_path: Direct filesystem path to a LoRA adapter
                checkpoint to use as the KL reference.
            ppo: Legacy flag for PPO clipping. Prefer loss_fn="ppo".
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
            allow_training_without_logprobs: Allow training even when no logprobs
                are available. Defaults to False.
            plot_tensors: Whether to plot training tensors for debugging.
                Defaults to False.
            truncated_importance_sampling: Truncation threshold for importance
                sampling weights.
            scale_learning_rate_by_reward_std_dev: Whether to scale learning rate
                by reward standard deviation. Defaults to False.
            logprob_calculation_chunk_size: Chunk size for logprob calculation.
                Defaults to 1024.
            packed_sequence_length: Packed sequence length to use for training.
            num_trajectories_learning_rate_multiplier_power: Power for learning
                rate multiplier based on number of trajectories.
            save_checkpoint: Accepted for PipelineTrainer compatibility. Serverless
                training currently always saves a trainable checkpoint for the next
                inference step.
            optimizer_save_interval: Accepted for PipelineTrainer compatibility;
                serverless training owns optimizer checkpoint cadence.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            ServerlessTrainResult with step number, training metrics, and artifact name.

        Example:
            await model.log(trajectory_groups, split="train")
            result = await backend.train(model, trajectory_groups, learning_rate=5e-6)
            # Optionally log training metrics:
            # await model.log(metrics=result.metrics, step=result.step)
        """
        groups_list = list(trajectory_groups)
        if loss_fn is None:
            resolved_loss_fn: Literal["cispo", "ppo"] = "ppo" if ppo else "cispo"
        else:
            resolved_loss_fn = loss_fn
            if ppo is not None and ppo != (loss_fn == "ppo"):
                raise ValueError("ServerlessBackend got conflicting loss_fn and ppo.")
        if resolved_loss_fn not in {"cispo", "ppo"}:
            raise ValueError(
                "ServerlessBackend only supports loss_fn='cispo' or 'ppo'."
            )
        if loss_fn_config is not None:
            raise ValueError("ServerlessBackend requires loss_fn_config=None.")
        if not normalize_advantages:
            scale_rewards = False
        if adam_params is not None:
            raise ValueError("ServerlessBackend requires adam_params=None.")
        if kl_penalty_reference_step is not None and kl_penalty_reference_step < 0:
            raise ValueError("kl_penalty_reference_step must be >= 0.")
        if kl_penalty_step_lag is not None:
            if kl_penalty_step_lag < 1:
                raise ValueError("kl_penalty_step_lag must be >= 1.")
            if kl_penalty_reference_step is not None:
                raise ValueError(
                    "Only one of kl_penalty_reference_step and "
                    "kl_penalty_step_lag may be set."
                )
        resolved_kl_penalty_source: Literal["current_learner", "sample"] = (
            kl_penalty_source
            if kl_penalty_source is not None
            else ("sample" if kl_penalty_coef > 0.0 else "current_learner")
        )
        _ = save_checkpoint

        config, dev_config = build_rl_train_configs(
            learning_rate=learning_rate,
            advantage_balance=advantage_balance,
            scale_rewards=scale_rewards,
            importance_sampling_level=importance_sampling_level,
            mask_prob_ratio=mask_prob_ratio,
            ppo=resolved_loss_fn == "ppo",
            precalculate_logprobs=precalculate_logprobs,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            max_negative_advantage_importance_sampling_weight=max_negative_advantage_importance_sampling_weight,
            kimi_k2_tau=kimi_k2_tau,
            kl_penalty_coef=kl_penalty_coef,
            kl_penalty_source=resolved_kl_penalty_source,
            allow_training_without_logprobs=allow_training_without_logprobs,
            plot_tensors=plot_tensors,
            truncated_importance_sampling=truncated_importance_sampling,
            scale_learning_rate_by_reward_std_dev=scale_learning_rate_by_reward_std_dev,
            logprob_calculation_chunk_size=logprob_calculation_chunk_size,
            packed_sequence_length=packed_sequence_length,
            num_trajectories_learning_rate_multiplier_power=num_trajectories_learning_rate_multiplier_power,
            kl_ref_adapter_path=kl_ref_adapter_path,
            optimizer_save_interval=optimizer_save_interval,
        )
        if kl_penalty_reference_step is not None:
            dev_config["kl_penalty_reference_step"] = kl_penalty_reference_step
        if kl_penalty_step_lag is not None:
            dev_config["kl_penalty_step_lag"] = kl_penalty_step_lag

        # Collect metrics from training
        training_metrics: list[dict[str, float]] = []
        trainer_started = time.monotonic()
        async for metrics in self._train_model(
            model, groups_list, config, dev_config, verbose
        ):
            training_metrics.append(metrics)

        avg_metrics = aggregate_rl_training_metrics(
            training_metrics=training_metrics,
            trajectory_groups=groups_list,
            trainer_started=trainer_started,
        )

        key = model._storage_name()
        step = self._native_steps.get(key)
        if step is None:
            step = await self._get_step(model)
        artifact_name: str | None = self._native_artifacts.get(key)
        if artifact_name is None and model.entity is not None:
            artifact_name = (
                f"{model.entity}/{model.project}/{model._storage_name()}:step{step}"
            )

        # Record provenance on the latest W&B artifact
        wandb_run = model._get_wandb_run()
        if wandb_run is not None:
            record_provenance(wandb_run, "serverless-rl")

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
        from art.distributed.trajectory_store import TrajectoryGroupBundle
        from art.training import (
            AdamConfig,
            ForwardBackwardRequest,
            LossConfig,
            OptimStepRequest,
            RlTrajectoryBatch,
            SamplerPublication,
            SaveWeightsForSamplerRequest,
        )

        summary = summarize_trajectory_groups(trajectory_groups)
        base_metrics = build_training_summary_metrics(
            summary,
            include_trainable_groups=True,
        )
        if not trajectory_groups or not any(
            group.trajectories for group in trajectory_groups
        ):
            raise ValueError("native training requires at least one trajectory")
        client = await self.training_client(model)
        versions = [
            version
            for group in trajectory_groups
            for trajectory in group.trajectories
            for version in (
                trajectory.initial_policy_version,
                trajectory.final_policy_version,
            )
            if version is not None
        ]
        current_version = client.projected_learner_version
        batch = RlTrajectoryBatch(
            groups=tuple(
                TrajectoryGroupBundle.from_group(group) for group in trajectory_groups
            ),
            min_source_version=min(versions, default=current_version),
            max_source_version=max(versions, default=current_version),
        )
        sequence_id = client.next_sequence_id
        request_id = secrets.token_hex(16)
        key = model._storage_name()
        retain_input = key in self._retain_next_inputs
        forward = await client.forward_backward(
            ForwardBackwardRequest(
                run_id=client.run_id,
                request_id=f"fb-{request_id}",
                sequence_id=sequence_id,
                batch=batch,
                loss=LossConfig(
                    name="ppo" if dev_config.get("ppo") else "cispo",
                    normalize_advantages=bool(dev_config.get("scale_rewards", True)),
                    values=cast(
                        dict[str, Any],
                        {
                            **config.model_dump(mode="python"),
                            **dict(dev_config),
                        },
                    ),
                ),
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in trajectory_groups
                ),
                return_token_logprobs=False,
                retain_packed_input=retain_input,
            )
        )
        self._retain_next_inputs.discard(key)
        forward_result = await forward.result()
        capture = forward_result.packed_input_capture
        if retain_input:
            if capture is None or capture.content_sha256 is None:
                raise RuntimeError("retained packed input has no exact content digest")
            self._retained_inputs.setdefault(key, []).append(capture)
        if forward_result.packing.group_shapes:
            if len(forward_result.packing.group_shapes) != len(trajectory_groups):
                raise RuntimeError("packed-group shapes changed cardinality")
            for group, shape in zip(
                trajectory_groups,
                forward_result.packing.group_shapes,
                strict=True,
            ):
                group._packed_group_shape = shape

        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id=client.run_id,
                request_id=f"optim-{request_id}",
                sequence_id=client.next_sequence_id,
                optimizer=AdamConfig(learning_rate=config.learning_rate),
            )
        )
        optimizer_result = await optimizer.result()
        publication_mode = (
            "in_flight_lora"
            if (model._internal_config or {}).get("rollout_weight_update_mode")
            == "in_flight_lora"
            else "versioned_lora"
        )
        publication = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=f"publish-{request_id}",
                sequence_id=client.next_sequence_id,
                checkpoint_name=f"step-{optimizer_result.checkpoint.learner_version}",
                publication=SamplerPublication(
                    mode=publication_mode,
                    model_alias=model.name,
                ),
            )
        )
        publication_result = await publication.result()
        self._native_steps[key] = optimizer_result.checkpoint.learner_version
        self._native_artifacts[key] = publication_result.lora
        if verbose:
            print(
                "Native training operations: "
                f"{forward.ref.operation_id}, {optimizer.ref.operation_id}, "
                f"{publication.ref.operation_id}"
            )
        yield {
            **base_metrics,
            **forward_result.metrics,
            **optimizer_result.metrics,
            TRAIN_GRADIENT_STEPS_KEY: float(forward_result.packing.packed_sequences),
        }

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Lower raw supervised trajectories through native split commands."""
        del dev_config

        from art.training import (
            AdamConfig,
            ForwardBackwardRequest,
            LossConfig,
            OptimStepRequest,
            SamplerPublication,
            SaveWeightsForSamplerRequest,
            SupervisedTrajectoryBatch,
        )

        from ..utils.sft import resolve_sft_batch_size

        values = list(trajectories)
        if not values:
            return
        batch_size = resolve_sft_batch_size(
            batch_size=config.batch_size,
            default_batch_size=2,
        )
        batches = [
            values[index : index + batch_size]
            for index in range(0, len(values), batch_size)
        ]
        learning_rates = (
            [float(value) for value in config.learning_rate]
            if isinstance(config.learning_rate, list)
            else [float(config.learning_rate)] * len(batches)
        )
        if len(learning_rates) != len(batches):
            raise ValueError("SFT learning-rate schedule must match batch count")

        client = await self.training_client(model)
        rows: list[dict[str, float]] = []
        operation_ids: list[str] = []
        optimizer_result: Any | None = None
        for batch, learning_rate in zip(batches, learning_rates, strict=True):
            request_id = secrets.token_hex(16)
            forward = await client.forward_backward(
                ForwardBackwardRequest(
                    run_id=client.run_id,
                    request_id=f"fb-{request_id}",
                    sequence_id=client.next_sequence_id,
                    batch=SupervisedTrajectoryBatch(
                        trajectories=tuple(batch),
                        assistant_turns=config.assistant_turns,
                    ),
                    loss=LossConfig(name="cross_entropy"),
                    return_token_logprobs=False,
                )
            )
            operation_ids.append(forward.ref.operation_id)
            forward_result = await forward.result()
            if not forward_result.produced_gradient:
                await forward.cancel()
                continue

            optimizer = await client.optim_step(
                OptimStepRequest(
                    run_id=client.run_id,
                    request_id=f"optim-{request_id}",
                    sequence_id=client.next_sequence_id,
                    optimizer=AdamConfig(learning_rate=learning_rate),
                )
            )
            operation_ids.append(optimizer.ref.operation_id)
            optimizer_result = await optimizer.result()
            rows.append(
                {
                    **forward_result.metrics,
                    **optimizer_result.metrics,
                    "data/step_num_trajectories": float(len(batch)),
                    "data/step_trainable_assistant_tokens": float(
                        forward_result.packing.trainable_assistant_tokens
                    ),
                }
            )

        if optimizer_result is None:
            return
        publication_mode = (
            "in_flight_lora"
            if (model._internal_config or {}).get("rollout_weight_update_mode")
            == "in_flight_lora"
            else "versioned_lora"
        )
        publication = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=f"publish-{secrets.token_hex(16)}",
                sequence_id=client.next_sequence_id,
                checkpoint_name=f"step-{optimizer_result.checkpoint.learner_version}",
                publication=SamplerPublication(
                    mode=publication_mode,
                    model_alias=model.name,
                ),
            )
        )
        operation_ids.append(publication.ref.operation_id)
        publication_result = await publication.result()
        key = model._storage_name()
        self._native_steps[key] = optimizer_result.checkpoint.learner_version
        self._native_artifacts[key] = publication_result.lora
        wandb_run = model._get_wandb_run()
        if wandb_run is not None:
            record_provenance(wandb_run, "serverless-sft")
        if verbose:
            print(f"Native SFT operations: {', '.join(operation_ids)}")
        for row in rows:
            row[TRAIN_GRADIENT_STEPS_KEY] = float(len(rows))
            yield row

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

        assert model.id is not None, "Model ID is required"

        # If entity is not set, use the user's default entity from W&B
        api = wandb_sdk.api(api_key=self._client.api_key)
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
                raise ValueError(
                    f"No checkpoints found for model {model._storage_name()}"
                )
        else:
            resolved_step = step

        if verbose:
            print(f"Downloading checkpoint step {resolved_step} from W&B artifacts...")

        # Download from W&B artifacts
        # The artifact name follows the pattern: {entity}/{project}/{model_name}:step{step}
        artifact_name = f"{model.entity}/{model.project}/{model._storage_name()}:step{resolved_step}"

        # Use wandb API to download (api was already created above for entity lookup)
        artifact = api.artifact(artifact_name, type="lora")

        # Determine download path
        if local_path is None:
            # Create a temporary directory that won't be cleaned up automatically
            checkpoint_dir = os.path.join(
                tempfile.gettempdir(),
                "art_checkpoints",
                model.project,
                model._storage_name(),
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
        model: "TrainableModel",
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
                model,
                step=step,
                verbose=verbose,
            )

            # Push to S3
            s3_path = build_s3_path(
                model_name=model._storage_name(),
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
            api = wandb_sdk.api(api_key=self._client.api_key)
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
            print(
                "Uploading forked checkpoint as W&B artifact for "
                f"{model._storage_name()}..."
            )

        wandb_sdk.login(key=self._client.api_key)
        run = wandb_sdk.init(
            project=model.project,
            entity=model.entity,
            job_type="checkpoint-fork",
            name=f"fork-{from_model}-to-{model._storage_name()}",
            settings=wandb_sdk.settings(silent=True),
        )
        assert run is not None

        dest_artifact = wandb_sdk.artifact(name=model._storage_name(), type="lora")
        dest_artifact.add_dir(checkpoint_dir)
        aliases = ["latest"]
        if selected_step is not None:
            aliases.insert(0, f"step{selected_step}")
        run.log_artifact(dest_artifact, aliases=aliases)
        run.finish()

        # Copy provenance from the source model's W&B run to the destination model
        api = wandb_sdk.api(api_key=self._client.api_key)
        try:
            source_run = api.run(f"{model.entity}/{from_project}/{from_model}")
            source_provenance = source_run.config.get("wandb.provenance")
            if source_provenance is not None:
                dest_run = model._get_wandb_run()
                if dest_run is not None:
                    dest_run.config.update(
                        {"wandb.provenance": list(source_provenance)}
                    )
        except Exception:
            pass  # Source run may not exist (e.g., S3-only models)

        if verbose:
            print(
                f"Successfully forked checkpoint from {from_model} "
                f"(step {selected_step}) to {model._storage_name()}"
            )
