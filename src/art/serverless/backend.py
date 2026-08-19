from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterable
from contextlib import asynccontextmanager
import os
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    model_validator,
)

from art._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
    merge_gradient_step_metrics,
    should_save_optimizer_state,
)
from art.adapter_leases import pin_inference_target, pinned_inference_name
from art.backend import AnyModel, AnyTrainableModel
from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    DistributedTrajectorySelection,
)
from art.distributed.trajectory_store import TrajectoryGroupRef
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.serving_capabilities import discover_serving_capabilities
from art.training.client import TrainingOperation
from art.training.contracts import (
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
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    SupervisedTrajectoryBatch,
)
from art.trajectories import Trajectory, TrajectoryGroup
from art.types import ServerlessTrainResult, TrainConfig, TrainSFTConfig
from art.utils.lifecycle import (
    complete_task,
    consume_future_exception,
    process_shutdown_timeout,
)

from .. import dev
from .client import (
    RemoteTrainingClient,
    RemoteTrainingServiceClient,
)
from .contracts import (
    MAX_CHECKPOINT_RETENTION_ITEMS,
    AdapterSpec,
    ApplyCheckpointRetentionRequest,
    CheckpointRevision,
    CreateTrainingRunRequest,
    TrainingRunSpec,
)

if TYPE_CHECKING:
    from art.model import Model, TrainableModel
    from art.pipeline_trainer.checkpoint_retention import (
        CheckpointRetentionPlan,
        CheckpointRetentionStrategy,
    )


_SERVERLESS_KL_REFERENCE_BLOCKER = (
    "serverless KL references require a remote checkpoint-to-reference-adapter "
    "contract. The current training command schema only carries named RL losses "
    "('cispo'/'ppo') plus scalar KL fields and does not expose a named KL loss or "
    "reference checkpoint resolver."
)
_EXACT_ADAPTER_REMOVE_ATTEMPTS = 2
_ModelKey = tuple[str | None, str, str, str]
_SamplerKey = tuple[_ModelKey, int]


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


class _RemotePipelineCommandContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    backend: SkipValidation[Any] = Field(exclude=True)
    model: SkipValidation[Any] = Field(exclude=True)
    client: SkipValidation[RemoteTrainingClient] = Field(exclude=True)
    groups: tuple[TrajectoryGroup, ...]
    selections: tuple[Any, ...]
    generation_id: str = Field(min_length=1)
    forward_request: ForwardBackwardRequest
    settings: _ServerlessTrainSettings
    preparation_metrics: dict[str, float]
    started: float = Field(ge=0)
    _sampler_pending: _PendingSamplerPublication | None = PrivateAttr(default=None)
    _publication: asyncio.Task[dict[str, float]] | None = PrivateAttr(default=None)
    _released: bool = PrivateAttr(default=False)

    def optimizer_request(self, sequence_id: int) -> OptimStepRequest:
        return OptimStepRequest(
            run_id=self.client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence_id,
            optimizer=AdamConfig(learning_rate=self.settings.learning_rate),
        )

    async def sampler_request(
        self, step: int, sequence_id: int
    ) -> SaveWeightsForSamplerRequest:
        object.__setattr__(
            self,
            "_sampler_pending",
            await self.backend._reserve_sampler_publication(self.model, step),
        )
        return SaveWeightsForSamplerRequest(
            run_id=self.client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence_id,
            checkpoint_name=f"step-{step}",
            publication=SamplerPublication(
                mode="in_flight_lora", model_alias=self.model.name
            ),
        )

    def state_request(self, step: int, sequence_id: int) -> SaveStateRequest | None:
        config, _ = self.settings.resolve()
        if not should_save_optimizer_state(step, config):
            return None
        return SaveStateRequest(
            run_id=self.client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence_id,
            checkpoint_name=f"step-{step}",
        )

    async def commands_admitted(
        self,
        *,
        forward: TrainingOperation[ForwardBackwardResult],
        optimizer: TrainingOperation[OptimStepResult],
        sampler: TrainingOperation[SamplerWeightsResult],
        state: TrainingOperation[SaveStateResult] | None,
    ) -> None:
        del forward, optimizer
        pending = self._require_sampler_pending()
        step = sampler.ref.learner_parent_version
        object.__setattr__(
            self,
            "_publication",
            asyncio.create_task(
                self.backend._complete_sampler_publication(
                    self.model, step, sampler, pending
                ),
                name=f"remote-sampler-publication-{self.client.run_id}-{step}",
            ),
        )
        assert self._publication is not None
        self.backend._track_task(self._publication)
        if state is not None:
            self.backend._track(state.result(), f"remote-state-{step}")
        release = asyncio.create_task(
            self._release("consumed"),
            name=f"remote-pipeline-release-{self.client.run_id}-{step}",
        )
        self.backend._track_task(release)

    async def complete(
        self,
        *,
        step: int,
        forward: TrainingOperation[ForwardBackwardResult],
        optimizer: TrainingOperation[OptimStepResult],
        forward_submit_s: float,
    ) -> ServerlessTrainResult:
        results = await asyncio.gather(
            forward.result(), optimizer.result(), return_exceptions=True
        )
        failures = [value for value in results if isinstance(value, BaseException)]
        if failures:
            raise BaseExceptionGroup("remote train operations failed", failures)
        forward_result, optimizer_result = results
        if not (
            isinstance(forward_result, ForwardBackwardResult)
            and isinstance(optimizer_result, OptimStepResult)
        ):
            raise TypeError("remote train returned an invalid result type")
        _attach_packing_shapes(list(self.groups), forward_result.packing.group_shapes)
        metrics = aggregate_rl_training_metrics(
            training_metrics=[
                {
                    **merge_gradient_step_metrics(
                        forward_result.metrics, optimizer_result.metrics
                    ),
                    **_packing_outcome_metrics(forward_result.packing),
                    "time/step_remote_forward_submit_s": forward_submit_s,
                }
            ],
            trajectory_groups=self.groups,
            trainer_started=self.started,
        )
        policy_counts = forward_result.packing.policy_token_counts
        if policy_counts is None:
            raise RuntimeError("remote RL packing omitted exact policy token counts")
        result = ServerlessTrainResult(
            step=step,
            metrics=metrics,
            packed_policy_token_counts=tuple(
                (value.policy_version, value.trainable_assistant_tokens)
                for value in policy_counts
            ),
        )
        pending = self._require_sampler_pending()
        publication = self._publication
        if publication is None:
            raise RuntimeError("remote sampler publication was not started")
        result.checkpoint_ready = self.backend._sampler_checkpoint_readiness(
            result, pending, publication
        )
        result.publication_metrics_ready = (
            self.backend._sampler_publication_metrics_readiness(publication)
        )
        return result

    async def abort(
        self,
        forward: TrainingOperation[ForwardBackwardResult] | None,
        optimizer: TrainingOperation[OptimStepResult] | None,
        sampler: TrainingOperation[SamplerWeightsResult] | None,
        *,
        optimizer_admitted: bool,
    ) -> None:
        del optimizer
        cleanup: list[Coroutine[Any, Any, None]] = []
        if not optimizer_admitted and forward is not None:
            cleanup.append(forward.cancel())
        pending = self._sampler_pending
        if pending is not None and self._publication is None:
            if sampler is None:
                await self.backend._fail_sampler_result(
                    self.model,
                    self.client.projected_learner_version,
                    pending,
                    RuntimeError("pipeline command admission failed"),
                )
            else:
                step = sampler.ref.learner_parent_version
                object.__setattr__(
                    self,
                    "_publication",
                    asyncio.create_task(
                        self.backend._complete_sampler_publication(
                            self.model, step, sampler, pending
                        ),
                        name=(
                            f"remote-sampler-publication-{self.client.run_id}-{step}"
                        ),
                    ),
                )
                assert self._publication is not None
                self.backend._track_task(self._publication)
        if not self._released:
            cleanup.append(
                self._release("consumed" if optimizer_admitted else "discarded")
            )
        failures = [
            value
            for value in await asyncio.gather(*cleanup, return_exceptions=True)
            if isinstance(value, BaseException)
        ]
        if failures:
            raise BaseExceptionGroup("remote pipeline cleanup failed", failures)

    def _require_sampler_pending(self) -> "_PendingSamplerPublication":
        if self._sampler_pending is None:
            raise RuntimeError("remote sampler publication was not reserved")
        return self._sampler_pending

    async def _release(self, disposition: Literal["consumed", "discarded"]) -> None:
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        if not self.selections:
            return
        queue = self.selections[0].queue
        _, cancelled = await complete_task(
            asyncio.create_task(
                queue.release_selections(
                    self.selections,
                    disposition=disposition,
                    generation_id=self.generation_id,
                )
            )
        )
        if cancelled is not None:
            raise cancelled


class _PendingServerlessTrain(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: int = Field(ge=1)
    completion: SkipValidation[asyncio.Task[ServerlessTrainResult]] = Field(
        exclude=True
    )

    async def result(self) -> ServerlessTrainResult:
        return await self.completion


class _PendingSamplerPublication(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    materialized: SkipValidation[asyncio.Future[SamplerWeightsResult]] = Field(
        exclude=True
    )
    predecessor_settled: SkipValidation[asyncio.Future[None] | None] = Field(
        exclude=True
    )
    activation_settled: SkipValidation[asyncio.Future[None]] = Field(exclude=True)


class _SamplerRetention(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_key: _ModelKey
    predecessor_settled: SkipValidation[asyncio.Future[None] | None] = Field(
        exclude=True
    )
    ready: SkipValidation[asyncio.Future[None]] = Field(exclude=True)
    finish_requested: SkipValidation[asyncio.Future[frozenset[_SamplerKey]]] = Field(
        exclude=True
    )
    settled: SkipValidation[asyncio.Future[None]] = Field(exclude=True)
    forget: SkipValidation[dict[_SamplerKey, SamplerWeightsResult]] = Field(
        default_factory=dict, exclude=True
    )
    failure: SkipValidation[BaseException | None] = Field(default=None, exclude=True)


class _ExactAdapterLeaseState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SkipValidation[Any] = Field(exclude=True)
    publication: SamplerPublication
    lock: SkipValidation[asyncio.Lock] = Field(
        default_factory=asyncio.Lock, exclude=True
    )
    leases: int = Field(default=0, ge=0)
    published: bool = False
    remove_failed: bool = False


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
            raise NotImplementedError(_SERVERLESS_KL_REFERENCE_BLOCKER)
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


_RemotePipelineCommandContext.model_rebuild()


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
        api_key: str | None = None,
        inference_api_key: str | None = None,
        checkpoint: str | None = None,
        restore_optimizer: bool = False,
        enable_expert_replay: bool = True,
        close_timeout_s: float = process_shutdown_timeout(1),
    ) -> None:
        api_key = api_key or os.environ.get("REMOTE_TRAINING_API_KEY")
        if api_key is None:
            raise ValueError(
                "ServerlessBackend requires api_key or REMOTE_TRAINING_API_KEY"
            )
        if not inference_base_url:
            raise ValueError("ServerlessBackend requires an inference_base_url")
        if restore_optimizer and checkpoint is None:
            raise ValueError("restore_optimizer requires checkpoint")
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")
        self._service = RemoteTrainingServiceClient(
            api_key=api_key,
            base_url=training_base_url,
        )
        self._inference_base_url = inference_base_url
        self._inference_api_key = inference_api_key or api_key
        self._checkpoint = checkpoint
        self._restore_optimizer = restore_optimizer
        self._sampler_manager = sampler_manager
        self._enable_expert_replay = enable_expert_replay
        self._close_timeout_s = close_timeout_s
        self._clients: dict[_ModelKey, RemoteTrainingClient] = {}
        self._register_lock = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._background_failures: list[BaseException] = []
        self._published_initial_models: set[tuple[str | None, str, str, str]] = set()
        self._sampler_results: dict[
            tuple[tuple[str | None, str, str, str], int], SamplerWeightsResult
        ] = {}
        self._pending_sampler_publications: dict[
            tuple[tuple[str | None, str, str, str], int],
            _PendingSamplerPublication,
        ] = {}
        self._sampler_publication_tails: dict[
            _ModelKey, tuple[int, asyncio.Future[None]]
        ] = {}
        self._sampler_retention_tails: dict[_ModelKey, _SamplerRetention] = {}
        self._sampler_retention_reservations: dict[_SamplerKey, _SamplerRetention] = {}
        self._exact_adapter_states: dict[_SamplerKey, _ExactAdapterLeaseState] = {}
        self._sampler_state_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    async def register(self, model: "Model") -> None:
        from art.model import TrainableModel

        if not isinstance(model, TrainableModel):
            raise TypeError("ServerlessBackend only supports trainable models")
        key = self._model_key(model)
        async with self._register_lock:
            if self._closing or self._closed:
                raise RuntimeError("ServerlessBackend is closed")
            if key in self._clients:
                raise RuntimeError("model is already registered")
            client = await RemoteTrainingClient.create(
                self._service,
                CreateTrainingRunRequest(
                    spec=TrainingRunSpec(
                        run_name=model.run_name,
                        base_model=model.base_model,
                        adapter=_adapter_spec(model),
                        seed=int((model.lora_config or {}).get("random_state", 3407)),
                        metadata={
                            "project": model.project,
                            "model_alias": model.name,
                            **({"entity": model.entity} if model.entity else {}),
                        },
                    ),
                    checkpoint=self._checkpoint,
                    restore_optimizer=self._restore_optimizer,
                ),
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

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            async with self._register_lock:
                async with self._sampler_state_lock:
                    active = sorted(
                        f"{state.publication.model_alias}={state.leases}"
                        for state in self._exact_adapter_states.values()
                        if state.leases
                    )
                    if active:
                        raise RuntimeError(
                            "cannot close ServerlessBackend with active exact adapter "
                            f"leases: {', '.join(active)}"
                        )
                    self._closing = True

            deadline = asyncio.get_running_loop().time() + self._close_timeout_s
            failures: list[BaseException] = []
            try:
                async with asyncio.timeout_at(deadline):
                    results = await asyncio.gather(
                        *(client.shutdown() for client in self._clients.values()),
                        return_exceptions=True,
                    )
                    failures.extend(
                        result
                        for result in results
                        if isinstance(result, BaseException)
                    )
                    await self._drain_background()
            except BaseException as error:
                failures.append(error)
                try:
                    async with asyncio.timeout_at(deadline):
                        results = await asyncio.gather(
                            *(
                                client.abort_result_waiters()
                                for client in self._clients.values()
                            ),
                            return_exceptions=True,
                        )
                    failures.extend(
                        result
                        for result in results
                        if isinstance(result, BaseException)
                    )
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if failures and self._background:
                tasks = tuple(self._background)
                for task in tasks:
                    task.cancel()
                _, pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
                if pending:
                    failures.append(
                        TimeoutError(
                            f"{len(pending)} remote background operations did not stop"
                        )
                    )

            adapters_drained = False
            try:
                async with asyncio.timeout_at(deadline):
                    await self._drain_exact_adapters()
                adapters_drained = True
            except BaseException as error:
                failures.append(error)
            if adapters_drained:
                try:
                    async with asyncio.timeout_at(deadline):
                        await self._service.close()
                except BaseException as error:
                    failures.append(error)
            await self._clear_sampler_state()
            if failures:
                raise BaseExceptionGroup("ServerlessBackend shutdown failed", failures)
            self._closed = True

    async def delete(self, model: "Model") -> None:
        from art.model import TrainableModel

        if not isinstance(model, TrainableModel):
            raise TypeError("ServerlessBackend only supports trainable models")
        client = await self.training_client(model)
        await client.shutdown()
        async with self._sampler_state_lock:
            tail = self._sampler_publication_tails.get(self._model_key(model))
        if tail is not None:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(tail[1])
            self._raise_background_failures()
        async for page in self._service.iter_checkpoint_pages(client.run_id):
            for checkpoint in page.checkpoints:
                await self._service.delete_checkpoint(
                    client.run_id, checkpoint.checkpoint_id
                )
        model_key = self._model_key(model)
        await self._clear_sampler_state(model_key)
        self._clients.pop(model_key)

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
        state = await self._acquire_exact_adapter(model, key, publication)
        try:
            async with pin_inference_target(
                model.name, step=step, inference_name=inference_name
            ):
                yield
        finally:
            await self._release_exact_adapter(key, state)

    async def _acquire_exact_adapter(
        self,
        model: AnyTrainableModel,
        key: _SamplerKey,
        publication: SamplerPublication,
    ) -> _ExactAdapterLeaseState:
        while True:
            async with self._sampler_state_lock:
                if self._closing or self._closed:
                    raise RuntimeError("ServerlessBackend is closed")
                retention = self._sampler_retention_reservations.get(key)
                if retention is None:
                    weights = self._sampler_results.get(key)
                    pending = self._pending_sampler_publications.get(key)
                    if weights is None and pending is None:
                        raise RuntimeError(
                            "exact sampler weights for learner step "
                            f"{key[1]} are unavailable"
                        )
                    state = self._exact_adapter_states.setdefault(
                        key,
                        _ExactAdapterLeaseState(model=model, publication=publication),
                    )
                    state.leases += 1
                    break
            await asyncio.shield(retention.settled)
        try:
            if weights is None:
                assert pending is not None
                weights = await asyncio.shield(pending.materialized)
            async with state.lock:
                if state.remove_failed:
                    await self._remove_exact_adapter(state)
                if not state.published:
                    await self._sampler_manager.publish(
                        state.model, weights, state.publication
                    )
                    state.published = True
        except BaseException:
            await self._drop_exact_adapter_reservation(key, state)
            raise
        return state

    async def _release_exact_adapter(
        self,
        key: _SamplerKey,
        state: _ExactAdapterLeaseState,
    ) -> None:
        async with state.lock:
            async with self._sampler_state_lock:
                if self._exact_adapter_states.get(key) is not state or not state.leases:
                    raise RuntimeError("exact adapter lease state is inconsistent")
                remove = state.leases == 1
                if not remove:
                    state.leases -= 1
                    return
            try:
                await self._remove_exact_adapter(state)
            finally:
                await self._drop_exact_adapter_reservation(key, state)

    async def _remove_exact_adapter(self, state: _ExactAdapterLeaseState) -> None:
        for attempt in range(_EXACT_ADAPTER_REMOVE_ATTEMPTS):
            try:
                await self._sampler_manager.remove(state.model, state.publication)
            except BaseException as error:
                state.remove_failed = True
                if attempt + 1 == _EXACT_ADAPTER_REMOVE_ATTEMPTS or not isinstance(
                    error, Exception
                ):
                    raise
            else:
                state.published = False
                state.remove_failed = False
                return

    async def _drain_exact_adapters(self) -> None:
        async with self._sampler_state_lock:
            states = tuple(self._exact_adapter_states.items())
        results = await asyncio.gather(
            *(self._drain_exact_adapter(key, state) for key, state in states),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("exact adapter removal failed", failures)

    async def _drain_exact_adapter(
        self,
        key: _SamplerKey,
        state: _ExactAdapterLeaseState,
    ) -> None:
        async with state.lock:
            async with self._sampler_state_lock:
                if (
                    self._exact_adapter_states.get(key) is not state
                    or state.leases
                    or not state.published
                ):
                    raise RuntimeError("exact adapter cleanup state is inconsistent")
            await self._remove_exact_adapter(state)
            async with self._sampler_state_lock:
                if (
                    self._exact_adapter_states.get(key) is not state
                    or state.leases
                    or state.published
                ):
                    raise RuntimeError("exact adapter cleanup state is inconsistent")
                del self._exact_adapter_states[key]

    async def _drop_exact_adapter_reservation(
        self,
        key: _SamplerKey,
        state: _ExactAdapterLeaseState,
    ) -> None:
        async with self._sampler_state_lock:
            if self._exact_adapter_states.get(key) is not state or not state.leases:
                raise RuntimeError("exact adapter lease state is inconsistent")
            state.leases -= 1
            if not state.leases and not state.published:
                del self._exact_adapter_states[key]

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
        operation, pending = await self._start_sampler_publication(
            model, client, step, client.next_sequence_id
        )
        await self._complete_sampler_publication(model, step, operation, pending)
        self._published_initial_models.add(key)

    def _sampler_key(self, model: AnyTrainableModel, step: int) -> _SamplerKey:
        return self._model_key(model), step

    async def _start_sampler_publication(
        self,
        model: AnyTrainableModel,
        client: RemoteTrainingClient,
        step: int,
        sequence_id: int,
    ) -> tuple[TrainingOperation[SamplerWeightsResult], _PendingSamplerPublication]:
        pending = await self._reserve_sampler_publication(model, step)
        try:
            operation = await client.save_weights_for_sampler(
                SaveWeightsForSamplerRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence_id,
                    checkpoint_name=f"step-{step}",
                    publication=SamplerPublication(
                        mode="in_flight_lora", model_alias=model.name
                    ),
                )
            )
        except BaseException as error:
            await self._fail_sampler_result(model, step, pending, error)
            raise
        return operation, pending

    async def _reserve_sampler_publication(
        self, model: AnyTrainableModel, step: int
    ) -> _PendingSamplerPublication:
        async with self._sampler_state_lock:
            key = self._sampler_key(model, step)
            if (
                key in self._sampler_results
                or key in self._pending_sampler_publications
            ):
                raise RuntimeError(
                    f"sampler weights for learner step {step} already exist"
                )
            model_key = key[0]
            predecessor = self._sampler_publication_tails.get(model_key)
            if predecessor is not None and step <= predecessor[0]:
                raise RuntimeError(
                    "sampler learner versions must increase monotonically: "
                    f"previous={predecessor[0]}, next={step}"
                )
            loop = asyncio.get_running_loop()
            materialized: asyncio.Future[SamplerWeightsResult] = loop.create_future()
            materialized.add_done_callback(consume_future_exception)
            activation_settled: asyncio.Future[None] = loop.create_future()
            pending = _PendingSamplerPublication(
                materialized=materialized,
                predecessor_settled=(
                    predecessor[1] if predecessor is not None else None
                ),
                activation_settled=activation_settled,
            )
            self._pending_sampler_publications[key] = pending
            self._sampler_publication_tails[model_key] = (step, activation_settled)
            return pending

    async def _resolve_sampler_result(
        self,
        model: AnyTrainableModel,
        step: int,
        pending: _PendingSamplerPublication,
        result: SamplerWeightsResult,
    ) -> None:
        async with self._sampler_state_lock:
            key = self._sampler_key(model, step)
            if (
                self._pending_sampler_publications.get(key) is not pending
                or pending.materialized.done()
            ):
                raise RuntimeError(
                    "pending sampler materialization state is inconsistent"
                )
            if result.checkpoint.learner_version != step:
                raise RuntimeError(
                    "sampler materialized an unexpected learner version: "
                    f"expected={step}, got={result.checkpoint.learner_version}"
                )
            self._sampler_results[key] = result
            pending.materialized.set_result(result)

    async def _finish_sampler_publication(
        self,
        model: AnyTrainableModel,
        step: int,
        pending: _PendingSamplerPublication,
    ) -> None:
        async with self._sampler_state_lock:
            key = self._sampler_key(model, step)
            if (
                self._pending_sampler_publications.get(key) is not pending
                or not pending.materialized.done()
            ):
                raise RuntimeError("pending sampler publication state is inconsistent")
            del self._pending_sampler_publications[key]
            pending.activation_settled.set_result(None)

    async def _fail_sampler_result(
        self,
        model: AnyTrainableModel,
        step: int,
        pending: _PendingSamplerPublication,
        error: BaseException,
    ) -> None:
        async with self._sampler_state_lock:
            key = self._sampler_key(model, step)
            if self._pending_sampler_publications.get(key) is pending:
                del self._pending_sampler_publications[key]
            if not pending.materialized.done():
                if isinstance(error, asyncio.CancelledError):
                    pending.materialized.cancel()
                else:
                    pending.materialized.set_exception(error)
            if not pending.activation_settled.done():
                pending.activation_settled.set_result(None)

    async def _complete_sampler_publication(
        self,
        model: AnyTrainableModel,
        step: int,
        operation: TrainingOperation[SamplerWeightsResult],
        pending: _PendingSamplerPublication,
    ) -> dict[str, float]:
        try:
            result = await operation.result()
            await self._resolve_sampler_result(model, step, pending, result)
            if pending.predecessor_settled is not None:
                await asyncio.shield(pending.predecessor_settled)
            metrics = dict(result.publication_metrics)
            active_metrics = await self._publish_active_sampler(model, result)
            if active_metrics:
                metrics.update(active_metrics)
            await self._finish_sampler_publication(model, step, pending)
            return metrics
        except BaseException as error:
            await self._fail_sampler_result(model, step, pending, error)
            raise

    def _sampler_checkpoint_readiness(
        self,
        result: ServerlessTrainResult,
        pending: _PendingSamplerPublication,
        publication: asyncio.Task[dict[str, float]],
    ) -> asyncio.Task[None]:
        async def wait() -> None:
            weights = await asyncio.shield(pending.materialized)
            result.checkpoint_id = weights.checkpoint.checkpoint_id
            await asyncio.shield(publication)

        task = asyncio.create_task(
            wait(), name=f"remote-checkpoint-ready-{result.step}"
        )
        task.add_done_callback(consume_future_exception)
        return task

    @staticmethod
    def _sampler_publication_metrics_readiness(
        publication: asyncio.Task[dict[str, float]],
    ) -> asyncio.Future[dict[str, float]]:
        ready = asyncio.shield(publication)
        ready.add_done_callback(consume_future_exception)
        return ready

    def _remember_sampler_result(
        self, model: AnyTrainableModel, result: SamplerWeightsResult
    ) -> None:
        """Seed already-materialized weights before concurrent backend work starts."""
        key = self._sampler_key(model, result.checkpoint.learner_version)
        if (
            self._sampler_state_lock.locked()
            or key[0] in self._sampler_retention_tails
            or key in self._sampler_results
            or key in self._pending_sampler_publications
        ):
            raise RuntimeError("sampler state is busy or weights already exist")
        self._sampler_results[key] = result

    async def _publish_active_sampler(
        self, model: AnyTrainableModel, result: SamplerWeightsResult
    ) -> dict[str, float] | None:
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

    async def _begin_sampler_retention(
        self, model: AnyTrainableModel
    ) -> _SamplerRetention:
        model_key = self._model_key(model)
        loop = asyncio.get_running_loop()
        # The tracked owner serializes retention without holding the state lock over I/O.
        # Callers only signal completion, so their cancellation cannot orphan the tail.
        async with self._sampler_state_lock:
            predecessor = self._sampler_retention_tails.get(model_key)
            retention = _SamplerRetention(
                model_key=model_key,
                predecessor_settled=(
                    predecessor.settled if predecessor is not None else None
                ),
                ready=loop.create_future(),
                finish_requested=loop.create_future(),
                settled=loop.create_future(),
            )
            retention.ready.add_done_callback(consume_future_exception)
            self._sampler_retention_tails[model_key] = retention
            self._track(
                self._own_sampler_retention(retention),
                f"sampler-retention-{model.name}",
            )
        try:
            await asyncio.shield(retention.ready)
        except BaseException:
            self._request_sampler_retention_finish(retention, set())
            raise
        return retention

    async def _own_sampler_retention(self, retention: _SamplerRetention) -> None:
        forgotten: frozenset[_SamplerKey] = frozenset()
        try:
            if retention.predecessor_settled is not None:
                await retention.predecessor_settled
            retention.ready.set_result(None)
            forgotten = await retention.finish_requested
            await self._settle_sampler_retention(retention, forgotten)
            if retention.failure is not None:
                raise retention.failure
        except BaseException as error:
            if not retention.ready.done():
                retention.ready.set_exception(error)
            if not retention.settled.done():
                cleanup = asyncio.create_task(
                    self._settle_sampler_retention(retention, frozenset()),
                    name="sampler-retention-cleanup",
                )
                self._track_task(cleanup)
                try:
                    await asyncio.shield(cleanup)
                except BaseException:
                    # Cleanup remains backend-owned and is bounded by close().
                    pass
            raise

    @staticmethod
    def _request_sampler_retention_finish(
        retention: _SamplerRetention, forgotten: set[_SamplerKey]
    ) -> None:
        requested = frozenset(forgotten)
        if retention.finish_requested.done():
            if (
                not retention.finish_requested.cancelled()
                and retention.finish_requested.exception() is None
                and retention.finish_requested.result() == requested
            ):
                return
            raise RuntimeError("sampler retention finish was already requested")
        retention.finish_requested.set_result(requested)

    async def _reserve_sampler_forgetting(
        self,
        retention: _SamplerRetention,
        *,
        observed_steps: set[int],
        retain_steps: set[int],
    ) -> set[int]:
        async with self._sampler_state_lock:
            if (
                retention.settled.done()
                or retention.predecessor_settled is not None
                and not retention.predecessor_settled.done()
            ):
                raise RuntimeError("sampler retention state is inconsistent")
            retained = set(retain_steps)
            retained.update(self._pending_sampler_steps_locked(retention.model_key))
            self._validate_sampler_retention_locked(retention.model_key, retained)
            # Only entries represented by this catalog snapshot may be forgotten.
            forget = {
                key: result
                for key, result in self._sampler_results.items()
                if key[0] == retention.model_key
                and key[1] in observed_steps
                and key[1] not in retained
            }
            if any(key in self._sampler_retention_reservations for key in forget):
                raise RuntimeError("sampler retention reservation is inconsistent")
            retention.forget.update(forget)
            self._sampler_retention_reservations.update(
                dict.fromkeys(forget, retention)
            )
            return retained

    async def _finish_sampler_retention(
        self, retention: _SamplerRetention, forgotten: set[_SamplerKey]
    ) -> None:
        self._request_sampler_retention_finish(retention, forgotten)
        await asyncio.shield(retention.settled)
        if retention.failure is not None:
            raise retention.failure

    async def _settle_sampler_retention(
        self,
        retention: _SamplerRetention,
        forgotten: frozenset[_SamplerKey],
    ) -> None:
        async with self._sampler_state_lock:
            if retention.settled.done():
                return
            failures: list[BaseException] = []
            valid_forgotten = forgotten.intersection(retention.forget)
            if valid_forgotten != forgotten:
                failures.append(RuntimeError("sampler retention state is inconsistent"))
            for key in valid_forgotten:
                if self._sampler_results.get(key) is retention.forget[key]:
                    del self._sampler_results[key]
                else:
                    failures.append(
                        RuntimeError("sampler result changed during retention")
                    )
            for key in retention.forget:
                if self._sampler_retention_reservations.get(key) is retention:
                    del self._sampler_retention_reservations[key]
                else:
                    failures.append(
                        RuntimeError("sampler retention reservation is inconsistent")
                    )
            if self._sampler_retention_tails.get(retention.model_key) is retention:
                del self._sampler_retention_tails[retention.model_key]
            if failures:
                retention.failure = BaseExceptionGroup(
                    "sampler retention settlement failed", failures
                )
            retention.settled.set_result(None)

    async def _delete_checkpoint_files(
        self, model: AnyTrainableModel, steps_to_keep: list[int]
    ) -> None:
        retention = await self._begin_sampler_retention(model)
        forgotten: set[_SamplerKey] = set()
        try:
            client = await self.training_client(model)
            checkpoints = []
            current_checkpoint_id = None
            async for page in self._service.iter_checkpoint_pages(client.run_id):
                current_checkpoint_id = page.current_checkpoint_id
                checkpoints.extend(page.checkpoints)
            keep = set(steps_to_keep)
            keep.update(
                checkpoint.learner_version
                for checkpoint in checkpoints
                if checkpoint.checkpoint_id == current_checkpoint_id
            )
            retained = await self._reserve_sampler_forgetting(
                retention,
                observed_steps={
                    checkpoint.learner_version for checkpoint in checkpoints
                },
                retain_steps=keep,
            )
            for checkpoint in checkpoints:
                if checkpoint.learner_version in retained:
                    continue
                await self._service.delete_checkpoint(
                    client.run_id, checkpoint.checkpoint_id
                )
                key = (retention.model_key, checkpoint.learner_version)
                if key in retention.forget:
                    forgotten.add(key)
        finally:
            await self._finish_sampler_retention(retention, forgotten)

    async def _forget_sampler_results(
        self, model: AnyTrainableModel, retain_steps: set[int]
    ) -> None:
        async with self._sampler_state_lock:
            model_key = self._model_key(model)
            self._validate_sampler_retention_locked(model_key, retain_steps)
            forgotten = [
                key
                for key in self._sampler_results
                if key[0] == model_key and key[1] not in retain_steps
            ]
            for key in forgotten:
                del self._sampler_results[key]

    async def _validate_sampler_retention(
        self, model: AnyTrainableModel, retain_steps: set[int]
    ) -> None:
        async with self._sampler_state_lock:
            self._validate_sampler_retention_locked(
                self._model_key(model), retain_steps
            )

    def _validate_sampler_retention_locked(
        self, model_key: _ModelKey, retain_steps: set[int]
    ) -> None:
        protected = [
            step
            for (key, step), state in self._exact_adapter_states.items()
            if key == model_key
            and step not in retain_steps
            and (state.leases or state.published)
        ]
        protected.extend(
            step
            for key, step in self._pending_sampler_publications
            if key == model_key and step not in retain_steps
        )
        protected.extend(
            step
            for key, step in self._sampler_retention_reservations
            if key == model_key and step not in retain_steps
        )
        if protected:
            raise RuntimeError(
                "checkpoint retention selected pending or active exact adapter steps "
                f"{sorted(set(protected))}"
            )

    def _pending_sampler_steps_locked(self, model_key: _ModelKey) -> set[int]:
        return {
            step for key, step in self._pending_sampler_publications if key == model_key
        }

    async def _list_checkpoint_infos(self, model: AnyTrainableModel):
        from art.pipeline_trainer.checkpoint_retention import CheckpointInfo

        client = await self.training_client(model)
        checkpoints: list[CheckpointInfo] = []
        async for page in self._service.iter_checkpoint_pages(client.run_id):
            for checkpoint in page.checkpoints:
                if checkpoint.state != "ready":
                    continue
                if len(checkpoints) == MAX_CHECKPOINT_RETENTION_ITEMS:
                    raise RuntimeError(
                        "remote checkpoint retention snapshot exceeds 512 checkpoints"
                    )
                checkpoints.append(
                    CheckpointInfo(
                        step=checkpoint.learner_version,
                        created_at=checkpoint.created_at,
                    )
                )
        return checkpoints

    def default_checkpoint_retention_strategy(
        self,
    ) -> "CheckpointRetentionStrategy | None":
        from art.pipeline_trainer.checkpoint_retention import keep_recent_and_periodic

        return keep_recent_and_periodic()

    async def _apply_checkpoint_retention(
        self, model: AnyTrainableModel, plan: "CheckpointRetentionPlan"
    ) -> None:
        retention = await self._begin_sampler_retention(model)
        forgotten: set[_SamplerKey] = set()
        try:
            client = await self.training_client(model)
            ready = []
            current_checkpoint_id = None
            async for page in self._service.iter_checkpoint_pages(client.run_id):
                current_checkpoint_id = page.current_checkpoint_id
                for checkpoint in page.checkpoints:
                    if checkpoint.state != "ready":
                        continue
                    if len(ready) == MAX_CHECKPOINT_RETENTION_ITEMS:
                        raise RuntimeError(
                            "remote checkpoint retention snapshot exceeds 512 checkpoints"
                        )
                    ready.append(checkpoint)
            actual_steps = {checkpoint.learner_version for checkpoint in ready}
            if actual_steps != plan.observed_steps:
                raise RuntimeError("remote checkpoint catalog changed during retention")
            retained = set(plan.retain_steps)
            retained.update(
                checkpoint.learner_version
                for checkpoint in ready
                if checkpoint.checkpoint_id == current_checkpoint_id
            )
            retained = await self._reserve_sampler_forgetting(
                retention,
                observed_steps=actual_steps,
                retain_steps=retained,
            )
            retain_checkpoint_ids = {
                checkpoint.checkpoint_id
                for checkpoint in ready
                if checkpoint.learner_version in retained
            }
            if current_checkpoint_id is not None:
                retain_checkpoint_ids.add(current_checkpoint_id)
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
            forgotten.update(retention.forget)
        finally:
            await self._finish_sampler_retention(retention, forgotten)

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
        started = time.monotonic()
        settings = _serverless_train_settings(locals())
        groups = list(trajectory_groups)
        result = await (await self._start_train(model, groups, settings)).result()
        checkpoint_ready = result.checkpoint_ready
        publication_metrics_ready = result.publication_metrics_ready
        if checkpoint_ready is None or publication_metrics_ready is None:
            raise RuntimeError("serverless train omitted sampler publication readiness")
        await checkpoint_ready
        result.metrics.update(await publication_metrics_ready)
        result.metrics["time/step_backend_train_s"] = time.monotonic() - started
        result.checkpoint_ready = None
        result.publication_metrics_ready = None
        return result

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
        if any(group._distributed_lease is not None for group in groups):
            raise RuntimeError(
                "distributed trajectory batches require remote pipeline preparation"
            )
        started = time.monotonic()
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

        optimizer: TrainingOperation[OptimStepResult] | None = None
        sampler: TrainingOperation[SamplerWeightsResult] | None = None
        sampler_ready: _PendingSamplerPublication | None = None
        step: int | None = None
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
            sampler, sampler_ready = await self._start_sampler_publication(
                model, client, step, sequence + 1
            )
            if should_save_optimizer_state(step, config):
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
            if sampler_ready is not None:
                assert step is not None
                await self._fail_sampler_result(model, step, sampler_ready, primary)
            cleanup: list[Coroutine[Any, Any, None]] = []
            if optimizer is None:
                cleanup.append(forward.cancel())
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
        assert sampler is not None and sampler_ready is not None
        assert step is not None
        publication = asyncio.create_task(
            self._complete_sampler_publication(model, step, sampler, sampler_ready),
            name=f"remote-sampler-publication-{client.run_id}-{step}",
        )
        self._track_task(publication)

        async def complete() -> ServerlessTrainResult:
            results = await asyncio.gather(
                forward.result(),
                optimizer.result(),
                return_exceptions=True,
            )
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                raise BaseExceptionGroup("remote train operations failed", failures)
            forward_result, optimizer_result = results
            if not (
                isinstance(forward_result, ForwardBackwardResult)
                and isinstance(optimizer_result, OptimStepResult)
            ):
                raise TypeError("remote train returned an invalid result type")
            _attach_packing_shapes(groups, forward_result.packing.group_shapes)
            metrics = aggregate_rl_training_metrics(
                training_metrics=[
                    {
                        **merge_gradient_step_metrics(
                            forward_result.metrics, optimizer_result.metrics
                        ),
                        **_packing_outcome_metrics(forward_result.packing),
                        "time/step_remote_forward_submit_s": forward_submit_s,
                    }
                ],
                trajectory_groups=groups,
                trainer_started=started,
            )
            policy_counts = forward_result.packing.policy_token_counts
            if policy_counts is None:
                raise RuntimeError(
                    "remote RL packing omitted exact policy token counts"
                )
            result = ServerlessTrainResult(
                step=step,
                metrics=metrics,
                packed_policy_token_counts=tuple(
                    (value.policy_version, value.trainable_assistant_tokens)
                    for value in policy_counts
                ),
            )
            result.checkpoint_ready = self._sampler_checkpoint_readiness(
                result, sampler_ready, publication
            )
            result.publication_metrics_ready = (
                self._sampler_publication_metrics_readiness(publication)
            )
            return result

        return _PendingServerlessTrain(
            step=step,
            completion=asyncio.create_task(
                complete(), name=f"remote-pipeline-train-{client.run_id}-{step}"
            ),
        )

    def supports_async_pipeline_packing(self, model: AnyTrainableModel) -> bool:
        del model
        return True

    async def prepare_pipeline_commands(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool = True,
        train_kwargs: dict[str, Any],
        learner_parent_version: int,
    ) -> _RemotePipelineCommandContext:
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
        min_source_version, max_source_version = _source_version_range(
            trajectory_groups, client.projected_learner_version
        )
        bundles = tuple(
            await asyncio.gather(
                *(
                    queue.receive_bundle(selection.lease.item.ref)
                    for selection in selected
                )
            )
        )
        batch = RlTrajectoryBatch.from_group_bundles(
            bundles,
            min_source_version=min_source_version,
            max_source_version=max_source_version,
            groups=trajectory_groups,
            group_annotations=tuple(
                selection.lease.item.annotations for selection in selected
            ),
        )
        for group in trajectory_groups:
            group._distributed_lease = None
        request_id = uuid.uuid4().hex
        generation_id = request_id
        request = ForwardBackwardRequest(
            run_id=client.run_id,
            request_id=request_id,
            sequence_id=client.next_sequence_id,
            batch=batch,
            loss=loss,
            collect_packing_shapes=any(
                group._collect_packing_shape for group in trajectory_groups
            ),
        )
        marked_packed = False
        try:
            _, cancelled = await complete_task(
                asyncio.create_task(queue.mark_packed(selected, generation_id))
            )
            marked_packed = True
            if cancelled is not None:
                raise cancelled
        except BaseException as primary:
            cleanup = [
                asyncio.create_task(
                    queue.release_selections(
                        selected,
                        disposition="discarded",
                        generation_id=generation_id if marked_packed else None,
                    )
                )
            ]
            failures = [
                value
                for value in await asyncio.gather(*cleanup, return_exceptions=True)
                if isinstance(value, BaseException) and value is not primary
            ]
            if failures:
                raise BaseExceptionGroup(
                    "remote packing and source cleanup failed", [primary, *failures]
                ) from None
            raise
        metrics = {
            "time/step_prepare_remote_batch_s": time.monotonic() - started,
        }
        return _RemotePipelineCommandContext(
            backend=self,
            model=model,
            client=client,
            groups=tuple(trajectory_groups),
            selections=selected,
            generation_id=generation_id,
            forward_request=request,
            settings=settings,
            preparation_metrics=metrics,
            started=started,
        )

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
        raw_batches = [
            values[index : index + batch_size]
            for index in range(0, len(values), batch_size)
        ]
        rates = (
            [float(value) for value in config.learning_rate]
            if isinstance(config.learning_rate, list)
            else [float(config.learning_rate)] * len(raw_batches)
        )
        if len(rates) != len(raw_batches):
            raise ValueError("SFT learning-rate schedule must match batch count")
        client = await self.training_client(model)
        rows: list[dict[str, float]] = []
        for batch, learning_rate in zip(raw_batches, rates, strict=True):
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
            forward_result = await forward.result()
            if not forward_result.produced_gradient:
                continue
            optimizer = await client.optim_step(
                OptimStepRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence + 1,
                    optimizer=AdamConfig(learning_rate=learning_rate),
                )
            )
            optimizer_result = await optimizer.result()
            rows.append(
                {
                    **merge_gradient_step_metrics(
                        forward_result.metrics, optimizer_result.metrics
                    ),
                    **_packing_outcome_metrics(forward_result.packing),
                    "data/step_num_trajectories": float(len(batch)),
                }
            )
        if not rows:
            return
        for row in rows:
            row[TRAIN_GRADIENT_STEPS_KEY] = float(len(rows))
        sequence = client.next_sequence_id
        step = client.projected_learner_version
        sampler, sampler_ready = await self._start_sampler_publication(
            model, client, step, sequence
        )
        try:
            state = await client.save_state(
                SaveStateRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence + 1,
                    checkpoint_name=f"step-{step}",
                )
            )
        except BaseException as error:
            await self._fail_sampler_result(model, step, sampler_ready, error)
            raise
        state_result, publication_metrics = await asyncio.gather(
            state.result(),
            self._complete_sampler_publication(model, step, sampler, sampler_ready),
        )
        rows[-1].update(state_result.metrics)
        rows[-1].update(publication_metrics)
        for row in rows:
            yield row

    async def finalize_training_session(
        self, model: AnyTrainableModel
    ) -> dict[str, float]:
        del model
        async with asyncio.timeout(self._close_timeout_s):
            await self._drain_background()
        return {}

    async def _clear_sampler_state(self, model_key: _ModelKey | None = None) -> None:
        async with self._sampler_state_lock:
            active_retentions = [
                key
                for key, retention in self._sampler_retention_tails.items()
                if (model_key is None or key == model_key)
                and not retention.settled.done()
            ]
            if active_retentions:
                raise RuntimeError("cannot clear sampler state during retention")
            pending_keys = tuple(
                key
                for key in self._pending_sampler_publications
                if model_key is None or key[0] == model_key
            )
            for key in pending_keys:
                pending = self._pending_sampler_publications.pop(key)
                if not pending.materialized.done():
                    pending.materialized.cancel()
                if not pending.activation_settled.done():
                    pending.activation_settled.set_result(None)
            result_keys = tuple(
                key
                for key in self._sampler_results
                if model_key is None or key[0] == model_key
            )
            for key in result_keys:
                del self._sampler_results[key]
            if model_key is None:
                self._sampler_publication_tails.clear()
                self._sampler_retention_tails.clear()
                self._sampler_retention_reservations.clear()
                self._published_initial_models.clear()
            else:
                self._sampler_publication_tails.pop(model_key, None)
                self._sampler_retention_tails.pop(model_key, None)
                self._published_initial_models.discard(model_key)

    def _track(self, awaitable: Coroutine[Any, Any, Any], name: str) -> None:
        self._track_task(asyncio.create_task(awaitable, name=name))

    def _track_task(self, task: asyncio.Task[Any]) -> None:
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
    def _model_key(model: AnyTrainableModel) -> _ModelKey:
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
        moe_parameterization=configured.get("moe_parameterization", "per_expert"),
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
