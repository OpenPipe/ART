from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterable
from contextlib import asynccontextmanager
import hashlib
import os
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from art._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
    merge_gradient_step_metrics,
    should_save_optimizer_state,
)
from art._source_revision import art_source_revision
from art.adapter_leases import pin_inference_target, pinned_inference_name
from art.backend import AnyModel, AnyTrainableModel
from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    DistributedTrajectorySelection,
)
from art.distributed.trajectory_store import TrajectoryGroupBundle, TrajectoryGroupRef
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.serving_capabilities import discover_serving_capabilities
from art.training.client import TrainingOperation
from art.training.contracts import (
    COMMAND_CONTRACT_VERSION,
    PACKING_CONTRACT_VERSION,
    AdamConfig,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    LossConfig,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    RlTrajectoryBatch,
    SamplerPublication,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
    SupervisedTrajectoryBatch,
)
from art.trajectories import Trajectory, TrajectoryGroup
from art.types import ServerlessTrainResult, TrainConfig, TrainSFTConfig
from art.utils.lifecycle import complete_task

from .. import dev
from .client import (
    RemoteTrainingClient,
    RemoteTrainingServiceClient,
    RouteObjectPublisher,
)
from .contracts import (
    AdapterSpec,
    ApplyCheckpointRetentionRequest,
    CheckpointRevision,
    CreateTrainingRunRequest,
    ReleaseRouteObjectsRequest,
    RemoteRlBatchRef,
    RemoteRlGroupRef,
    TrainingRunSpec,
)
from .data_plane import encode_trajectory_group

if TYPE_CHECKING:
    from art.model import Model, TrainableModel
    from art.pipeline_trainer.checkpoint_retention import (
        CheckpointRetentionPlan,
        CheckpointRetentionStrategy,
    )


class SamplerManager(Protocol):
    async def publish(
        self,
        model: AnyTrainableModel,
        weights: SamplerWeightsResult,
        publication: SamplerPublication,
    ) -> dict[str, float] | None: ...

    async def remove(
        self, model: AnyTrainableModel, publication: SamplerPublication
    ) -> None: ...


class _RemotePipelineBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    batch: RemoteRlBatchRef
    forward: SkipValidation[TrainingOperation[ForwardBackwardResult]] = Field(
        exclude=True
    )
    loss: LossConfig
    forward_submit_s: float = Field(ge=0)
    groups: tuple[Any, ...]
    selections: tuple[Any, ...]
    generation_id: str = Field(min_length=1)


class _StagedPipelineGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote: RemoteRlGroupRef
    byte_stream_receive_s: float = Field(ge=0)
    encode_s: float = Field(ge=0)
    upload_s: float = Field(ge=0)
    wall_s: float = Field(ge=0)


class _PendingServerlessTrain(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: int = Field(ge=1)
    completion: SkipValidation[asyncio.Task[ServerlessTrainResult]] = Field(
        exclude=True
    )

    async def result(self) -> ServerlessTrainResult:
        return await self.completion


class _ServerlessTrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    learning_rate: float = 5e-6
    loss_fn: Literal["cispo", "ppo"] = "cispo"
    loss_fn_config: dict[str, Any] | None = None
    normalize_advantages: bool = True
    adam_params: object | None = None
    kl_penalty_coef: float = 0.0
    kl_penalty_reference_step: int | None = None
    kl_ref_adapter_path: str | None = None
    kl_penalty_source: Literal["current_learner", "sample"] = "current_learner"
    epsilon: float | None = None
    epsilon_high: float | None = None
    advantage_balance: float = 0.0
    scale_rewards: bool = True
    importance_sampling_level: Literal[
        "token", "sequence", "average", "geometric_average"
    ] = "token"
    max_negative_advantage_importance_sampling_weight: float | None = None
    mask_prob_ratio: bool = False
    kimi_k2_tau: float | None = None
    precalculate_logprobs: bool = False
    allow_training_without_logprobs: bool = False
    plot_tensors: bool = False
    truncated_importance_sampling: float | None = None
    scale_learning_rate_by_reward_std_dev: bool = False
    logprob_calculation_chunk_size: int = 1024
    packed_sequence_length: int | None = None
    num_trajectories_learning_rate_multiplier_power: float = 0.0
    save_checkpoint: bool = True
    optimizer_save_interval: int = Field(default=5, ge=1)
    final_training_step: int | None = Field(default=None, ge=1)
    grad_accumulation_sequences: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_supported_options(self) -> "_ServerlessTrainSettings":
        if self.loss_fn_config is not None or self.adam_params is not None:
            raise ValueError("custom loss and optimizer objects are not supported")
        if self.kl_ref_adapter_path is not None:
            raise ValueError("remote training does not accept client filesystem paths")
        if self.kl_penalty_reference_step is not None:
            raise NotImplementedError(
                "remote KL checkpoint resolution is not implemented"
            )
        return self

    def resolve(self) -> tuple[TrainConfig, LossConfig]:
        config, values = build_rl_train_configs(
            learning_rate=self.learning_rate,
            advantage_balance=self.advantage_balance,
            scale_rewards=self.scale_rewards and self.normalize_advantages,
            importance_sampling_level=self.importance_sampling_level,
            mask_prob_ratio=self.mask_prob_ratio,
            ppo=self.loss_fn == "ppo",
            precalculate_logprobs=self.precalculate_logprobs,
            epsilon=self.epsilon,
            epsilon_high=self.epsilon_high,
            max_negative_advantage_importance_sampling_weight=(
                self.max_negative_advantage_importance_sampling_weight
            ),
            kimi_k2_tau=self.kimi_k2_tau,
            kl_penalty_coef=self.kl_penalty_coef,
            kl_penalty_source=self.kl_penalty_source,
            allow_training_without_logprobs=self.allow_training_without_logprobs,
            plot_tensors=self.plot_tensors,
            truncated_importance_sampling=self.truncated_importance_sampling,
            scale_learning_rate_by_reward_std_dev=(
                self.scale_learning_rate_by_reward_std_dev
            ),
            logprob_calculation_chunk_size=self.logprob_calculation_chunk_size,
            packed_sequence_length=self.packed_sequence_length,
            num_trajectories_learning_rate_multiplier_power=(
                self.num_trajectories_learning_rate_multiplier_power
            ),
            optimizer_save_interval=self.optimizer_save_interval,
            final_training_step=self.final_training_step,
            grad_accumulation_sequences=self.grad_accumulation_sequences,
        )
        return config, LossConfig(
            name=self.loss_fn,
            normalize_advantages=self.scale_rewards and self.normalize_advantages,
            values={
                **values,
                "grad_accumulation_sequences": config.grad_accumulation_sequences,
            },
        )


def _serverless_train_settings(values: dict[str, Any]) -> _ServerlessTrainSettings:
    fields = _ServerlessTrainSettings.model_fields
    return _ServerlessTrainSettings.model_validate(
        {name: value for name, value in values.items() if name in fields}
    )


class ServerlessBackend:
    """ART backend backed by the sequenced Remote Training command service."""

    def __init__(
        self,
        *,
        training_base_url: str,
        inference_base_url: str,
        sampler_manager: SamplerManager,
        route_object_publisher: RouteObjectPublisher | None = None,
        api_key: str | None = None,
        inference_api_key: str | None = None,
        checkpoint: str | None = None,
        restore_optimizer: bool = False,
        enable_expert_replay: bool = True,
        close_timeout_s: float = 20.0,
    ) -> None:
        api_key = api_key or os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise ValueError("ServerlessBackend requires api_key or WANDB_API_KEY")
        if not inference_base_url:
            raise ValueError("ServerlessBackend requires an inference_base_url")
        if restore_optimizer and checkpoint is None:
            raise ValueError("restore_optimizer requires checkpoint")
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")
        if enable_expert_replay and route_object_publisher is None:
            raise ValueError("expert replay requires a binary route object publisher")
        self._service = RemoteTrainingServiceClient(
            api_key=api_key,
            base_url=training_base_url,
        )
        self._inference_base_url = inference_base_url
        self._inference_api_key = inference_api_key or api_key
        self._checkpoint = checkpoint
        self._restore_optimizer = restore_optimizer
        self._sampler_manager = sampler_manager
        self._route_object_publisher = route_object_publisher
        self._enable_expert_replay = enable_expert_replay
        self._close_timeout_s = close_timeout_s
        self._clients: dict[tuple[str | None, str, str, str], RemoteTrainingClient] = {}
        self._register_lock = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._background_failures: list[BaseException] = []
        self._published_initial_models: set[tuple[str | None, str, str, str]] = set()
        self._sampler_results: dict[
            tuple[tuple[str | None, str, str, str], int], SamplerWeightsResult
        ] = {}
        self._exact_adapter_leases: dict[
            tuple[tuple[str | None, str, str, str], int], int
        ] = {}
        self._exact_adapter_lock = asyncio.Lock()
        self._staged_pipeline_groups: dict[
            tuple[str, str], asyncio.Task[_StagedPipelineGroup]
        ] = {}
        self._closed = False

    async def register(self, model: "Model") -> None:
        from art.model import TrainableModel

        if not isinstance(model, TrainableModel):
            raise TypeError("ServerlessBackend only supports trainable models")
        key = self._model_key(model)
        async with self._register_lock:
            if self._closed:
                raise RuntimeError("ServerlessBackend is closed")
            if key in self._clients:
                raise RuntimeError("model is already registered")
            await self._check_capabilities(model)
            client = await RemoteTrainingClient.create(
                self._service,
                CreateTrainingRunRequest(
                    spec=TrainingRunSpec(
                        run_name=model.run_name,
                        base_model=model.base_model,
                        adapter=_adapter_spec(model),
                        seed=int((model.lora_config or {}).get("random_state", 3407)),
                        packing_contract_version=PACKING_CONTRACT_VERSION,
                        art_version=art_source_revision(),
                        metadata={
                            "project": model.project,
                            "model_alias": model.name,
                            **({"entity": model.entity} if model.entity else {}),
                        },
                    ),
                    checkpoint=self._checkpoint,
                    restore_optimizer=self._restore_optimizer,
                ),
                route_publisher=self._route_object_publisher,
                close_timeout_s=self._close_timeout_s,
            )
            self._clients[key] = client
            model.id = client.run_id
            model.run_id = client.run_id

    async def training_client(self, model: AnyTrainableModel) -> RemoteTrainingClient:
        try:
            return self._clients[self._model_key(model)]
        except KeyError as error:
            raise RuntimeError(
                "model is not registered with ServerlessBackend"
            ) from error

    async def _check_capabilities(self, model: AnyTrainableModel) -> None:
        capabilities = await self._service.capabilities()
        if capabilities.command_contract_version != COMMAND_CONTRACT_VERSION:
            raise RuntimeError("remote command contract version does not match ART")
        if PACKING_CONTRACT_VERSION not in capabilities.packing_contract_versions:
            raise RuntimeError("remote service does not support ART's packing contract")
        if "bfloat16" not in capabilities.supported_dtypes:
            raise RuntimeError("remote service does not support bfloat16 training")
        if not {"cross_entropy", "cispo", "ppo"}.issubset(
            capabilities.supported_losses
        ):
            raise RuntimeError("remote service does not support ART's named losses")
        if _adapter_spec(model).rank > capabilities.max_lora_rank:
            raise RuntimeError("requested LoRA rank exceeds remote service capacity")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        try:
            async with asyncio.timeout(self._close_timeout_s):
                failures.extend(await self._discard_all_staged_pipeline_groups())
                results = await asyncio.gather(
                    *(client.close() for client in self._clients.values()),
                    return_exceptions=True,
                )
                failures.extend(
                    result for result in results if isinstance(result, BaseException)
                )
                await self._drain_background()
        except BaseException as error:
            failures.append(error)
            await asyncio.gather(
                *(client.abort_result_waiters() for client in self._clients.values()),
                return_exceptions=True,
            )
        finally:
            await asyncio.gather(
                *(client.close_event_observer() for client in self._clients.values()),
                return_exceptions=True,
            )
            await self._service.close()
        if failures:
            raise BaseExceptionGroup("ServerlessBackend shutdown failed", failures)

    async def delete(self, model: "Model") -> None:
        from art.model import TrainableModel

        if not isinstance(model, TrainableModel):
            raise TypeError("ServerlessBackend only supports trainable models")
        client = await self.training_client(model)
        failures = await self._discard_staged_pipeline_run(client.run_id)
        if failures:
            raise BaseExceptionGroup("staged pipeline input cleanup failed", failures)
        await client.close()
        await client.wait_closed()
        await client.close_event_observer()
        checkpoints = await self._service.list_checkpoints(client.run_id)
        for checkpoint in checkpoints.checkpoints:
            await self._service.delete_checkpoint(
                client.run_id, checkpoint.checkpoint_id
            )
        self._clients.pop(self._model_key(model))

    def logs_sft_metrics_remotely(self) -> bool:
        return False

    def pipeline_autotuner_inference_observer(self) -> Literal["rollout_supply"]:
        return "rollout_supply"

    def _model_inference_name(self, model: AnyModel, step: int | None = None) -> str:
        if name := pinned_inference_name(model.name, step):
            return name
        return model.name if step is None else f"{model.name}@{step}"

    @asynccontextmanager
    async def adapter_lease(
        self, model: AnyTrainableModel, step: int
    ) -> AsyncIterator[None]:
        async with pin_inference_target(
            model.name, step=step, inference_name=model.name
        ):
            yield

    @asynccontextmanager
    async def exact_adapter_lease(
        self, model: AnyTrainableModel, step: int
    ) -> AsyncIterator[None]:
        key = self._sampler_key(model, step)
        inference_name = f"{model.name}@{step}"
        publication = SamplerPublication(
            mode="versioned_lora", model_alias=inference_name
        )
        async with self._exact_adapter_lock:
            weights = self._sampler_results.get(key)
            if weights is None:
                raise RuntimeError(
                    f"exact sampler weights for learner step {step} are unavailable"
                )
            leases = self._exact_adapter_leases.get(key, 0)
            if leases == 0:
                await self._sampler_manager.publish(model, weights, publication)
            self._exact_adapter_leases[key] = leases + 1
        try:
            async with pin_inference_target(
                model.name, step=step, inference_name=inference_name
            ):
                yield
        finally:
            async with self._exact_adapter_lock:
                leases = self._exact_adapter_leases[key]
                if leases == 1:
                    del self._exact_adapter_leases[key]
                    await self._sampler_manager.remove(model, publication)
                else:
                    self._exact_adapter_leases[key] = leases - 1

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        if config is not None:
            raise ValueError("remote inference does not accept a local server config")
        root_url = _inference_root_url(self._inference_base_url)
        capabilities = await discover_serving_capabilities(
            base_url=root_url,
            headers={"Authorization": f"Bearer {self._inference_api_key}"},
            allow_openai_compatible=False,
        )
        capabilities.require(
            "in_flight_lora_updates", operation="remote pipeline weight publication"
        )
        capabilities.require(
            "policy_token_spans", operation="remote pipeline policy provenance"
        )
        object.__setattr__(model, "_serving_capabilities", capabilities)
        object.__setattr__(model, "_inference_connection_errors_are_fatal", True)
        if self._model_uses_expert_replay(model):
            capabilities.require(
                "binary_routed_experts", operation="remote MoE routing replay"
            )
            object.__setattr__(
                model, "_art_binary_routes_base_url", f"{root_url}/art/v1"
            )
        model.inference_base_url = self._inference_base_url
        model.inference_api_key = self._inference_api_key
        model.inference_model_name = self._model_inference_name(model)
        await self._publish_initial_sampler(model)
        return self._inference_base_url, self._inference_api_key

    async def _publish_initial_sampler(self, model: AnyTrainableModel) -> None:
        key = self._model_key(model)
        if key in self._published_initial_models:
            return
        client = await self.training_client(model)
        step = client.projected_learner_version
        operation = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=client.next_sequence_id,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(
                    mode="in_flight_lora", model_alias=model.name
                ),
            )
        )
        result = await operation.result()
        await self._publish_active_sampler(model, result)
        self._published_initial_models.add(key)

    def _sampler_key(
        self, model: AnyTrainableModel, step: int
    ) -> tuple[tuple[str | None, str, str, str], int]:
        return self._model_key(model), step

    def _remember_sampler_result(
        self, model: AnyTrainableModel, result: SamplerWeightsResult
    ) -> None:
        self._sampler_results[
            self._sampler_key(model, result.checkpoint.learner_version)
        ] = result

    async def _publish_active_sampler(
        self, model: AnyTrainableModel, result: SamplerWeightsResult
    ) -> dict[str, float] | None:
        self._remember_sampler_result(model, result)
        return await self._sampler_manager.publish(
            model,
            result,
            SamplerPublication(mode="in_flight_lora", model_alias=model.name),
        )

    def _model_uses_expert_replay(self, model: AnyTrainableModel) -> bool:
        if not self._enable_expert_replay:
            return False
        from art.megatron.model_support import model_uses_expert_parallel

        return model_uses_expert_parallel(
            model.base_model,
            allow_unvalidated_arch=bool(
                (model._internal_config or {}).get("allow_unvalidated_arch")
            ),
        )

    async def _get_step(self, model: AnyTrainableModel) -> int:
        client = await self.training_client(model)
        return (await self._service.get_run(client.run_id)).committed_learner_version

    async def _delete_checkpoint_files(
        self, model: AnyTrainableModel, steps_to_keep: list[int]
    ) -> None:
        keep = set(steps_to_keep)
        async with self._exact_adapter_lock:
            self._validate_sampler_retention(model, keep)
            client = await self.training_client(model)
            page = await self._service.list_checkpoints(client.run_id)
            for checkpoint in page.checkpoints:
                if (
                    checkpoint.learner_version not in keep
                    and checkpoint.checkpoint_id != page.current_checkpoint_id
                ):
                    await self._service.delete_checkpoint(
                        client.run_id, checkpoint.checkpoint_id
                    )
            self._forget_sampler_results(model, keep)

    def _forget_sampler_results(
        self, model: AnyTrainableModel, retain_steps: set[int]
    ) -> None:
        self._validate_sampler_retention(model, retain_steps)
        model_key = self._model_key(model)
        forgotten = [
            key
            for key in self._sampler_results
            if key[0] == model_key and key[1] not in retain_steps
        ]
        for key in forgotten:
            del self._sampler_results[key]

    def _validate_sampler_retention(
        self, model: AnyTrainableModel, retain_steps: set[int]
    ) -> None:
        model_key = self._model_key(model)
        leased = [
            step
            for (key, step), leases in self._exact_adapter_leases.items()
            if key == model_key and step not in retain_steps and leases > 0
        ]
        if leased:
            raise RuntimeError(
                f"checkpoint retention selected active exact adapter steps {leased}"
            )

    async def _list_checkpoint_infos(self, model: AnyTrainableModel):
        from art.pipeline_trainer.checkpoint_retention import CheckpointInfo

        client = await self.training_client(model)
        page = await self._service.list_checkpoints(client.run_id)
        return [
            CheckpointInfo(
                step=checkpoint.learner_version,
                created_at=checkpoint.created_at,
            )
            for checkpoint in page.checkpoints
            if checkpoint.state == "ready"
        ]

    def default_checkpoint_retention_strategy(
        self,
    ) -> "CheckpointRetentionStrategy | None":
        from art.pipeline_trainer.checkpoint_retention import keep_recent_and_periodic

        return keep_recent_and_periodic()

    async def _apply_checkpoint_retention(
        self, model: AnyTrainableModel, plan: "CheckpointRetentionPlan"
    ) -> None:
        retained = set(plan.retain_steps)
        async with self._exact_adapter_lock:
            self._validate_sampler_retention(model, retained)
            client = await self.training_client(model)
            page = await self._service.list_checkpoints(client.run_id)
            ready = tuple(
                checkpoint
                for checkpoint in page.checkpoints
                if checkpoint.state == "ready"
            )
            actual_steps = {checkpoint.learner_version for checkpoint in ready}
            if actual_steps != plan.observed_steps:
                raise RuntimeError("remote checkpoint catalog changed during retention")
            retain_checkpoint_ids = {
                checkpoint.checkpoint_id
                for checkpoint in ready
                if checkpoint.learner_version in retained
            }
            if page.current_checkpoint_id is not None:
                retain_checkpoint_ids.add(page.current_checkpoint_id)
            await self._service.apply_checkpoint_retention(
                client.run_id,
                ApplyCheckpointRetentionRequest(
                    observed=tuple(
                        CheckpointRevision(
                            checkpoint_id=checkpoint.checkpoint_id,
                            revision=checkpoint.revision,
                        )
                        for checkpoint in ready
                    ),
                    retain_checkpoint_ids=tuple(sorted(retain_checkpoint_ids)),
                    archive_checkpoint_ids=tuple(
                        checkpoint.checkpoint_id
                        for checkpoint in ready
                        if checkpoint.learner_version in plan.archive_steps
                    ),
                ),
            )
            self._forget_sampler_results(model, retained)

    async def train(  # type: ignore[override]
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float = 5e-6,
        loss_fn: Literal["cispo", "ppo"] = "cispo",
        loss_fn_config: dict | None = None,
        normalize_advantages: bool = True,
        adam_params: object | None = None,
        kl_penalty_coef: float = 0.0,
        kl_penalty_reference_step: int | None = None,
        kl_ref_adapter_path: str | None = None,
        kl_penalty_source: Literal["current_learner", "sample"] = "current_learner",
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        importance_sampling_level: Literal[
            "token", "sequence", "average", "geometric_average"
        ] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        truncated_importance_sampling: float | None = None,
        scale_learning_rate_by_reward_std_dev: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        packed_sequence_length: int | None = None,
        num_trajectories_learning_rate_multiplier_power: float = 0.0,
        save_checkpoint: bool = True,
        optimizer_save_interval: int = 5,
        final_training_step: int | None = None,
        grad_accumulation_sequences: int | None = None,
        verbose: bool = False,
    ) -> ServerlessTrainResult:
        del verbose
        self._raise_background_failures()
        settings = _serverless_train_settings(locals())
        groups = list(trajectory_groups)
        return await (await self._start_train(model, groups, settings)).result()

    async def start_pipeline_train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> _PendingServerlessTrain:
        self._raise_background_failures()
        return await self._start_train(
            model,
            list(trajectory_groups),
            _ServerlessTrainSettings.model_validate(kwargs),
        )

    async def _start_train(
        self,
        model: AnyTrainableModel,
        groups: list[TrajectoryGroup],
        settings: _ServerlessTrainSettings,
    ) -> _PendingServerlessTrain:
        if not groups:
            raise ValueError("trajectory_groups must not be empty")
        config, loss = settings.resolve()
        client = await self.training_client(model)
        prepared = groups[0]._prepared_training_batch
        if prepared is not None:
            if (
                not isinstance(prepared, _RemotePipelineBatch)
                or len(prepared.groups) != len(groups)
                or any(
                    actual is not expected
                    for actual, expected in zip(prepared.groups, groups, strict=True)
                )
                or any(
                    group._prepared_training_batch is not prepared for group in groups
                )
            ):
                raise RuntimeError("remote pipeline batch preparation is inconsistent")
        elif any(group._prepared_training_batch is not None for group in groups):
            raise RuntimeError("remote pipeline batch preparation is inconsistent")
        elif any(group._distributed_lease is not None for group in groups):
            raise RuntimeError(
                "distributed trajectory batches require remote pipeline preparation"
            )
        started = time.monotonic()
        if prepared is None:
            request = ForwardBackwardRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=client.next_sequence_id,
                batch=RlTrajectoryBatch.from_groups(
                    groups, default_source_version=client.projected_learner_version
                ),
                loss=loss,
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in groups
                ),
            )
            submit_started = time.monotonic()
            forward = await client.forward_backward(request)
            forward_submit_s = time.monotonic() - submit_started
        else:
            if prepared.loss != loss:
                raise RuntimeError("prepared remote F/B configuration changed")
            forward = prepared.forward
            forward_submit_s = prepared.forward_submit_s

        optimizer = None
        try:
            sequence = client.next_sequence_id
            optimizer = await client.optim_step(
                OptimStepRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence,
                    optimizer=AdamConfig(learning_rate=settings.learning_rate),
                )
            )
            step = optimizer.ref.reserved_output_learner_version
            if step is None:
                raise RuntimeError("optimizer did not reserve a learner version")
            sampler = await client.save_weights_for_sampler(
                SaveWeightsForSamplerRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence + 1,
                    checkpoint_name=f"step-{step}",
                    publication=SamplerPublication(
                        mode="in_flight_lora", model_alias=model.name
                    ),
                )
            )
            if settings.save_checkpoint or should_save_optimizer_state(step, config):
                state = await client.save_state(
                    SaveStateRequest(
                        run_id=client.run_id,
                        request_id=uuid.uuid4().hex,
                        sequence_id=sequence + 2,
                        checkpoint_name=f"step-{step}",
                    )
                )
                self._track(state.result(), f"remote-state-{step}")
        except BaseException as primary:
            cleanup: list[Coroutine[Any, Any, None]] = []
            if optimizer is None:
                cleanup.append(forward.cancel())
            if prepared is not None:
                for group in groups:
                    group._prepared_training_batch = None
                cleanup.append(
                    self._release_remote_pipeline_batch(
                        prepared,
                        disposition=("discarded" if optimizer is None else "consumed"),
                    )
                )
            failures = [
                result
                for result in await asyncio.gather(*cleanup, return_exceptions=True)
                if isinstance(result, BaseException)
            ]
            if failures:
                raise BaseExceptionGroup(
                    "remote train admission and cleanup failed", [primary, *failures]
                ) from None
            raise
        release: asyncio.Task[None] | None = None
        if prepared is not None:
            release = asyncio.create_task(
                self._release_remote_pipeline_batch(prepared, disposition="consumed"),
                name=f"remote-pipeline-release-{client.run_id}-{step}",
            )
            for group in groups:
                group._prepared_training_batch = None

        async def publish_sampler() -> tuple[
            SamplerWeightsResult, dict[str, float] | None
        ]:
            sampler_result = await sampler.result()
            return sampler_result, await self._publish_active_sampler(
                model, sampler_result
            )

        publication = asyncio.create_task(
            publish_sampler(), name=f"remote-sampler-publication-{client.run_id}-{step}"
        )

        async def complete() -> ServerlessTrainResult:
            try:
                results = await asyncio.gather(
                    forward.result(),
                    optimizer.result(),
                    publication,
                    return_exceptions=True,
                )
                failures = [
                    result for result in results if isinstance(result, BaseException)
                ]
                if failures:
                    raise BaseExceptionGroup("remote train operations failed", failures)
                forward_result, optimizer_result, published = results
                if not (
                    isinstance(forward_result, ForwardBackwardResult)
                    and isinstance(optimizer_result, OptimStepResult)
                    and isinstance(published, tuple)
                    and len(published) == 2
                    and isinstance(published[0], SamplerWeightsResult)
                ):
                    raise TypeError("remote train returned an invalid result type")
                sampler_result, publication_metrics = published
            except BaseException as primary:
                if release is not None:
                    try:
                        await complete_task(release)
                    except BaseException as cleanup:
                        raise BaseExceptionGroup(
                            "remote train and trajectory release failed",
                            [primary, cleanup],
                        ) from None
                raise
            if release is not None:
                _, cancelled = await complete_task(release)
                if cancelled is not None:
                    raise cancelled
            _attach_packing_shapes(groups, forward_result.packing.group_shapes)
            metrics = aggregate_rl_training_metrics(
                training_metrics=[
                    {
                        **merge_gradient_step_metrics(
                            forward_result.metrics, optimizer_result.metrics
                        ),
                        **sampler_result.publication_metrics,
                        **(publication_metrics or {}),
                        **_packing_outcome_metrics(forward_result.packing),
                        "time/step_remote_forward_submit_s": forward_submit_s,
                    }
                ],
                trajectory_groups=groups,
                trainer_started=started,
            )
            return ServerlessTrainResult(
                step=step,
                metrics=metrics,
                checkpoint_id=sampler_result.checkpoint.checkpoint_id,
            )

        return _PendingServerlessTrain(
            step=step,
            completion=asyncio.create_task(
                complete(), name=f"remote-pipeline-train-{client.run_id}-{step}"
            ),
        )

    def supports_async_pipeline_packing(self, model: AnyTrainableModel) -> bool:
        del model
        return True

    async def stage_pipeline_group(
        self,
        model: AnyTrainableModel,
        queue: DistributedTrajectoryQueue,
        ref: TrajectoryGroupRef,
    ) -> None:
        client = await self.training_client(model)
        key = (client.run_id, ref.result_id)
        if key in self._staged_pipeline_groups:
            raise RuntimeError("trajectory group is already staged for remote training")
        self._staged_pipeline_groups[key] = asyncio.create_task(
            self._stage_pipeline_group(client, queue, ref),
            name=f"remote-training-stage-{client.run_id}-{ref.result_id}",
        )

    async def _stage_pipeline_group(
        self,
        client: RemoteTrainingClient,
        queue: DistributedTrajectoryQueue,
        ref: TrajectoryGroupRef,
    ) -> _StagedPipelineGroup:
        started = time.monotonic()
        receive_started = time.monotonic()
        bundle = await self._receive_pipeline_group(queue, ref)
        receive_s = time.monotonic() - receive_started
        encode_started = time.monotonic()
        encoded = await asyncio.to_thread(
            encode_trajectory_group,
            bundle,
            object_id=hashlib.sha256(
                (
                    f"{client.run_id}:{ref.owner_actor_id}:{ref.result_id}:"
                    f"{ref.lease_id}"
                ).encode()
            ).hexdigest(),
        )
        encode_s = time.monotonic() - encode_started
        upload_started = time.monotonic()
        await client.stage_rl_group(encoded)
        return _StagedPipelineGroup(
            remote=encoded.remote,
            byte_stream_receive_s=receive_s,
            encode_s=encode_s,
            upload_s=time.monotonic() - upload_started,
            wall_s=time.monotonic() - started,
        )

    async def _receive_pipeline_group(
        self, queue: DistributedTrajectoryQueue, ref: TrajectoryGroupRef
    ) -> TrajectoryGroupBundle:
        return await queue.receive_bundle(ref)

    async def discard_pipeline_group(
        self, model: AnyTrainableModel, group: TrajectoryGroup
    ) -> None:
        selection = group._distributed_lease
        if not isinstance(selection, DistributedTrajectorySelection):
            raise RuntimeError("remote pipeline group has no distributed selection")
        client = await self.training_client(model)
        key = (client.run_id, selection.lease.item.ref.result_id)
        task = self._staged_pipeline_groups.pop(key, None)
        if task is None:
            raise RuntimeError("remote pipeline group was not staged")
        staged = await task
        await self._release_staged_group(client.run_id, staged)

    async def _discard_all_staged_pipeline_groups(self) -> list[BaseException]:
        entries = tuple(self._staged_pipeline_groups.items())
        self._staged_pipeline_groups.clear()
        return await self._discard_staged_pipeline_entries(entries)

    async def _discard_staged_pipeline_run(self, run_id: str) -> list[BaseException]:
        entries = tuple(
            (key, self._staged_pipeline_groups.pop(key))
            for key in tuple(self._staged_pipeline_groups)
            if key[0] == run_id
        )
        return await self._discard_staged_pipeline_entries(entries)

    async def _discard_staged_pipeline_entries(
        self,
        entries: tuple[tuple[tuple[str, str], asyncio.Task[_StagedPipelineGroup]], ...],
    ) -> list[BaseException]:
        results = await asyncio.gather(
            *(task for _, task in entries), return_exceptions=True
        )
        failures = [value for value in results if isinstance(value, BaseException)]
        deletes = await asyncio.gather(
            *(
                self._release_staged_group(key[0], value)
                for (key, _), value in zip(entries, results, strict=True)
                if isinstance(value, _StagedPipelineGroup)
            ),
            return_exceptions=True,
        )
        failures.extend(value for value in deletes if isinstance(value, BaseException))
        return failures

    async def _release_staged_group(
        self, run_id: str, value: _StagedPipelineGroup
    ) -> None:
        operations = [self._service.delete_training_data(run_id, value.remote.data)]
        refs = tuple(route.ref for route in value.remote.routes)
        if refs:
            operations.append(
                self._service.release_route_objects(
                    run_id, ReleaseRouteObjectsRequest(refs=refs)
                )
            )
        results = await asyncio.gather(*operations, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("staged remote group cleanup failed", failures)

    async def prepare_pipeline_batch(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool = True,
        train_kwargs: dict[str, Any],
        learner_parent_version: int,
    ) -> dict[str, float]:
        if normalize_advantages != train_kwargs.get("normalize_advantages", True):
            raise ValueError("pipeline reward normalization configuration changed")
        settings = _ServerlessTrainSettings.model_validate(train_kwargs)
        _, loss = settings.resolve()
        client = await self.training_client(model)
        if client.projected_learner_version != learner_parent_version:
            raise RuntimeError(
                "remote pipeline parent changed before F/B admission: "
                f"expected={learner_parent_version}, "
                f"projected={client.projected_learner_version}"
            )
        started = time.monotonic()
        if any(
            group._prepared_training_batch is not None for group in trajectory_groups
        ):
            raise RuntimeError("trajectory group already owns a prepared batch")
        selections = tuple(group._distributed_lease for group in trajectory_groups)
        selected = tuple(
            selection
            for selection in selections
            if isinstance(selection, DistributedTrajectorySelection)
        )
        if len(selected) != len(trajectory_groups):
            raise RuntimeError("remote pipeline batches require distributed groups")
        queue = selected[0].queue
        if any(selection.queue is not queue for selection in selected):
            raise RuntimeError("remote batch spans distributed trajectory queues")
        for group in trajectory_groups:
            group._distributed_lease = None
        marked_packed = False
        generation_id = uuid.uuid4().hex
        forward: TrainingOperation[ForwardBackwardResult] | None = None
        forward_submit_s = 0.0
        staged: tuple[_StagedPipelineGroup, ...] = ()
        stage_keys = tuple(
            (client.run_id, selection.lease.item.ref.result_id)
            for selection in selected
        )
        adopted = False
        try:
            tasks = tuple(self._staged_pipeline_groups.get(key) for key in stage_keys)
            if any(task is None for task in tasks):
                raise RuntimeError("remote pipeline group staging was not started")
            stage_tasks = tuple(task for task in tasks if task is not None)
            stage_wait_started = time.monotonic()
            staged = tuple(await asyncio.gather(*stage_tasks))
            stage_wait_s = time.monotonic() - stage_wait_started
            min_source_version, max_source_version = _source_version_range(
                trajectory_groups, client.projected_learner_version
            )
            batch = RemoteRlBatchRef(
                groups=tuple(
                    value.remote.model_copy(
                        update={"annotations": selection.lease.item.annotations}
                    )
                    for value, selection in zip(staged, selected, strict=True)
                ),
                min_source_version=min_source_version,
                max_source_version=max_source_version,
            )
            generation_id = hashlib.sha256(batch.model_dump_json().encode()).hexdigest()
            _, cancelled = await complete_task(
                asyncio.create_task(queue.mark_packed(selected, generation_id))
            )
            marked_packed = True
            if cancelled is not None:
                raise cancelled
            for key, task in zip(stage_keys, stage_tasks, strict=True):
                if self._staged_pipeline_groups.pop(key, None) is not task:
                    raise RuntimeError(
                        "remote pipeline group staging ownership changed"
                    )
            adopted = True
            submit_started = time.monotonic()
            forward = await client.forward_backward_refs(
                request_id=uuid.uuid4().hex,
                batch=batch,
                loss=loss,
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in trajectory_groups
                ),
            )
            forward_submit_s = time.monotonic() - submit_started
            if forward.ref.learner_parent_version != learner_parent_version:
                raise RuntimeError(
                    "remote service admitted F/B against another learner"
                )
        except BaseException as primary:
            cleanup = [forward.cancel()] if forward is not None else []
            cleanup.append(
                queue.release_selections(
                    selected,
                    disposition="discarded",
                    generation_id=generation_id if marked_packed else None,
                )
            )
            stage_failures: list[BaseException] = []
            if not adopted:
                entries = tuple(
                    (key, task)
                    for key in stage_keys
                    if (task := self._staged_pipeline_groups.pop(key, None)) is not None
                )
                stage_failures = await self._discard_staged_pipeline_entries(entries)
            failures = [
                result
                for result in await asyncio.gather(*cleanup, return_exceptions=True)
                if isinstance(result, BaseException)
            ]
            failures.extend(error for error in stage_failures if error is not primary)
            if failures:
                raise BaseExceptionGroup(
                    "remote packing and source cleanup failed", [primary, *failures]
                ) from None
            raise
        assert forward is not None
        prepared = _RemotePipelineBatch(
            batch=batch,
            forward=forward,
            loss=loss,
            forward_submit_s=forward_submit_s,
            groups=tuple(trajectory_groups),
            selections=selected,
            generation_id=generation_id,
        )
        for group in trajectory_groups:
            group._prepared_training_batch = prepared
        return {
            "time/step_prepare_remote_batch_s": time.monotonic() - started,
            "time/step_remote_group_stage_wait_s": stage_wait_s,
            "time/step_remote_group_receive_max_s": max(
                value.byte_stream_receive_s for value in staged
            ),
            "time/step_remote_group_encode_max_s": max(
                value.encode_s for value in staged
            ),
            "time/step_remote_group_upload_max_s": max(
                value.upload_s for value in staged
            ),
            "time/step_remote_forward_submit_s": forward_submit_s,
            "data/step_remote_batch_bytes": float(
                sum(
                    value.remote.data.byte_count
                    + sum(route.ref.byte_count for route in value.remote.routes)
                    for value in staged
                )
            ),
        }

    async def discard_pipeline_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> None:
        prepared = trajectory_groups[0]._prepared_training_batch
        if not isinstance(prepared, _RemotePipelineBatch) or any(
            group._prepared_training_batch is not prepared
            for group in trajectory_groups
        ):
            raise RuntimeError("remote pipeline batch is not prepared")
        for group in trajectory_groups:
            group._prepared_training_batch = None
        results = await asyncio.gather(
            prepared.forward.cancel(),
            self._release_remote_pipeline_batch(prepared, disposition="discarded"),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("remote prepared batch discard failed", failures)

    @staticmethod
    async def _release_remote_pipeline_batch(
        prepared: _RemotePipelineBatch,
        *,
        disposition: Literal["consumed", "discarded"],
    ) -> None:
        if not prepared.selections:
            return
        queue = prepared.selections[0].queue
        _, cancelled = await complete_task(
            asyncio.create_task(
                queue.release_selections(
                    prepared.selections,
                    disposition=disposition,
                    generation_id=prepared.generation_id,
                )
            )
        )
        if cancelled is not None:
            raise cancelled

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        del dev_config, verbose
        self._raise_background_failures()
        from art.utils.sft import resolve_sft_batch_size

        values = list(trajectories)
        if not values:
            return
        batch_size = resolve_sft_batch_size(
            batch_size=config.batch_size, default_batch_size=2
        )
        batches = [
            values[index : index + batch_size]
            for index in range(0, len(values), batch_size)
        ]
        rates = (
            config.learning_rate
            if isinstance(config.learning_rate, list)
            else [config.learning_rate] * len(batches)
        )
        if len(rates) != len(batches):
            raise ValueError("SFT learning-rate schedule must match batch count")
        client = await self.training_client(model)
        pending: dict[str, float] | None = None
        for batch, learning_rate in zip(batches, rates, strict=True):
            sequence = client.next_sequence_id
            forward = await client.forward_backward(
                ForwardBackwardRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence,
                    batch=SupervisedTrajectoryBatch(
                        trajectories=tuple(batch),
                        assistant_turns=config.assistant_turns,
                    ),
                    loss=LossConfig(name="cross_entropy"),
                )
            )
            optimizer = await client.optim_step(
                OptimStepRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence + 1,
                    optimizer=AdamConfig(learning_rate=learning_rate),
                )
            )
            forward_result, optimizer_result = await asyncio.gather(
                forward.result(), optimizer.result()
            )
            if pending is not None:
                yield pending
            pending = {
                **merge_gradient_step_metrics(
                    forward_result.metrics, optimizer_result.metrics
                ),
                **_packing_outcome_metrics(forward_result.packing),
                "data/step_num_trajectories": float(len(values)),
                "data/step_num_dropped_trajectories": 0.0,
                TRAIN_GRADIENT_STEPS_KEY: float(len(batches)),
            }
        assert pending is not None
        sequence = client.next_sequence_id
        step = client.projected_learner_version
        sampler = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(
                    mode="in_flight_lora", model_alias=model.name
                ),
            )
        )
        state = await client.save_state(
            SaveStateRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence + 1,
                checkpoint_name=f"step-{step}",
            )
        )
        sampler_result, state_result = await asyncio.gather(
            sampler.result(), state.result()
        )
        publication_metrics = await self._publish_active_sampler(model, sampler_result)
        yield {
            **pending,
            **sampler_result.publication_metrics,
            **state_result.metrics,
            **(publication_metrics or {}),
        }

    def _track(self, awaitable: Coroutine[Any, Any, Any], name: str) -> None:
        task = asyncio.create_task(awaitable, name=name)
        self._background.add(task)

        def completed(value: asyncio.Task[Any]) -> None:
            self._background.discard(value)
            if not value.cancelled() and (error := value.exception()) is not None:
                self._background_failures.append(error)

        task.add_done_callback(completed)

    async def _drain_background(self) -> None:
        if self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)
        self._raise_background_failures()

    def _raise_background_failures(self) -> None:
        if self._background_failures:
            failures, self._background_failures = self._background_failures, []
            raise BaseExceptionGroup("remote background operations failed", failures)

    @staticmethod
    def _model_key(model: AnyTrainableModel) -> tuple[str | None, str, str, str]:
        return model.entity, model.project, model._storage_name(), model.base_model


def _adapter_spec(model: AnyTrainableModel) -> AdapterSpec:
    from art.megatron.lora_config import LORA_ALPHA, default_lora_rank_for_handler
    from art.megatron.model_support import (
        default_target_modules_for_model,
        get_model_support_handler_for_spec,
        get_model_support_spec,
    )

    configured = model.lora_config or {}
    allow_unvalidated = bool(
        (model._internal_config or {}).get("allow_unvalidated_arch")
    )
    support = get_model_support_spec(
        model.base_model, allow_unvalidated_arch=allow_unvalidated
    )
    handler = get_model_support_handler_for_spec(support)
    alpha = int(configured.get("alpha", LORA_ALPHA))
    if alpha != LORA_ALPHA:
        raise ValueError(f"Megatron LoRA requires alpha={LORA_ALPHA}")
    return AdapterSpec(
        rank=int(configured.get("rank", default_lora_rank_for_handler(handler))),
        alpha=alpha,
        target_modules=tuple(
            configured.get("target_modules")
            or default_target_modules_for_model(
                model.base_model, allow_unvalidated_arch=allow_unvalidated
            )
        ),
    )


def _attach_packing_shapes(
    groups: list[TrajectoryGroup], shapes: tuple[Any, ...]
) -> None:
    if not any(group._collect_packing_shape for group in groups):
        return
    if len(groups) != len(shapes):
        raise RuntimeError("remote packing shapes do not match trajectory groups")
    for group, shape in zip(groups, shapes, strict=True):
        group._packed_group_shape = shape


def _source_version_range(
    groups: Iterable[TrajectoryGroup], default: int
) -> tuple[int, int]:
    versions = [
        version
        for group in groups
        for trajectory in group.trajectories
        for version in (
            trajectory.initial_policy_version,
            trajectory.final_policy_version,
        )
        if version is not None
    ]
    return min(versions, default=default), max(versions, default=default)


def _packing_outcome_metrics(packing: PackingOutcome) -> dict[str, float]:
    return {
        "pipeline/packed_sequence_length": float(packing.packed_sequence_length),
        "pipeline/target_packed_sequences": float(packing.target_packed_sequences),
        "data/step_packed_sequences": float(packing.packed_sequences),
        "data/step_physical_tokens": float(packing.physical_tokens),
        "data/step_trainable_assistant_tokens": float(
            packing.trainable_assistant_tokens
        ),
    }


def _inference_root_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value
