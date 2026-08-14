from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterable
from contextlib import asynccontextmanager
import os
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol
import uuid

from art._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
    merge_gradient_step_metrics,
)
from art._source_revision import art_source_revision
from art.adapter_leases import pin_inference_target, pinned_inference_name
from art.backend import AnyModel, AnyTrainableModel
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.serving_capabilities import discover_serving_capabilities
from art.training.contracts import (
    COMMAND_CONTRACT_VERSION,
    PACKING_CONTRACT_VERSION,
    AdamConfig,
    ForwardBackwardRequest,
    LossConfig,
    OptimStepRequest,
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

from .. import dev
from .client import RemoteTrainingClient, RemoteTrainingServiceClient
from .contracts import (
    AdapterSpec,
    ApplyCheckpointRetentionRequest,
    CheckpointRevision,
    CreateTrainingRunRequest,
    TrainingRunSpec,
)
from .data_plane import EncodedTrainingBatch, prepare_training_batch

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
            await self._service.close()
        if failures:
            raise BaseExceptionGroup("ServerlessBackend shutdown failed", failures)

    async def delete(self, model: "Model") -> None:
        from art.model import TrainableModel

        if not isinstance(model, TrainableModel):
            raise TypeError("ServerlessBackend only supports trainable models")
        client = await self.training_client(model)
        await client.close()
        await client.wait_closed()
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
                if checkpoint.learner_version not in keep:
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
                    retain_checkpoint_ids=tuple(
                        checkpoint.checkpoint_id
                        for checkpoint in ready
                        if checkpoint.learner_version in retained
                    ),
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
        grad_accumulation_sequences: int | None = None,
        verbose: bool = False,
    ) -> ServerlessTrainResult:
        del optimizer_save_interval, verbose
        self._raise_background_failures()
        if loss_fn not in {"cispo", "ppo"}:
            raise ValueError("ServerlessBackend supports only cispo and ppo")
        if loss_fn_config is not None or adam_params is not None:
            raise ValueError("custom loss and optimizer objects are not supported")
        if kl_ref_adapter_path is not None:
            raise ValueError("remote training does not accept client filesystem paths")
        if kl_penalty_reference_step is not None:
            raise NotImplementedError(
                "remote KL checkpoint resolution is not implemented"
            )
        groups = list(trajectory_groups)
        if not groups:
            raise ValueError("trajectory_groups must not be empty")
        if not normalize_advantages:
            scale_rewards = False
        config, values = build_rl_train_configs(
            learning_rate=learning_rate,
            advantage_balance=advantage_balance,
            scale_rewards=scale_rewards,
            importance_sampling_level=importance_sampling_level,
            mask_prob_ratio=mask_prob_ratio,
            ppo=loss_fn == "ppo",
            precalculate_logprobs=precalculate_logprobs,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            max_negative_advantage_importance_sampling_weight=(
                max_negative_advantage_importance_sampling_weight
            ),
            kimi_k2_tau=kimi_k2_tau,
            kl_penalty_coef=kl_penalty_coef,
            kl_penalty_source=kl_penalty_source,
            allow_training_without_logprobs=allow_training_without_logprobs,
            plot_tensors=plot_tensors,
            truncated_importance_sampling=truncated_importance_sampling,
            scale_learning_rate_by_reward_std_dev=(
                scale_learning_rate_by_reward_std_dev
            ),
            logprob_calculation_chunk_size=logprob_calculation_chunk_size,
            packed_sequence_length=packed_sequence_length,
            num_trajectories_learning_rate_multiplier_power=(
                num_trajectories_learning_rate_multiplier_power
            ),
            grad_accumulation_sequences=grad_accumulation_sequences,
        )
        started = time.monotonic()
        client = await self.training_client(model)
        sequence = client.next_sequence_id
        prepared = groups[0]._prepared_training_batch
        if prepared is not None:
            local_groups = (
                prepared.batch.require_local_groups()
                if isinstance(prepared, EncodedTrainingBatch)
                else ()
            )
            if (
                not isinstance(prepared, EncodedTrainingBatch)
                or len(local_groups) != len(groups)
                or any(
                    group._prepared_training_batch is not prepared for group in groups
                )
                or any(
                    actual is not expected
                    for actual, expected in zip(local_groups, groups, strict=True)
                )
            ):
                raise RuntimeError("remote pipeline batch preparation is inconsistent")
        batch = (
            prepared.batch
            if prepared is not None
            else RlTrajectoryBatch.from_groups(
                groups, default_source_version=client.projected_learner_version
            )
        )
        request = ForwardBackwardRequest(
            run_id=client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence,
            batch=batch,
            loss=LossConfig(
                name=loss_fn,
                normalize_advantages=scale_rewards,
                values={
                    **values,
                    "grad_accumulation_sequences": config.grad_accumulation_sequences,
                },
            ),
            collect_packing_shapes=any(
                group._collect_packing_shape for group in groups
            ),
        )
        submit_started = time.monotonic()
        try:
            forward = (
                await client.forward_backward(request)
                if prepared is None
                else await client.forward_backward_prepared(request, prepared)
            )
            submit_s = time.monotonic() - submit_started
        finally:
            if prepared is not None:
                for group in groups:
                    group._prepared_training_batch = None
        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence + 1,
                optimizer=AdamConfig(learning_rate=learning_rate),
            )
        )
        step = optimizer.ref.reserved_output_learner_version
        if step is None:
            raise RuntimeError("optimizer did not reserve a learner version")
        sampler = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence + 2,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(
                    mode="in_flight_lora", model_alias=model.name
                ),
            )
        )
        if save_checkpoint:
            state = await client.save_state(
                SaveStateRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence + 3,
                    checkpoint_name=f"step-{step}",
                )
            )
            self._track(state.result(), f"remote-state-{step}")
        forward_result, optimizer_result, sampler_result = await asyncio.gather(
            forward.result(), optimizer.result(), sampler.result()
        )
        publication_metrics = await self._publish_active_sampler(model, sampler_result)
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
                    "time/step_remote_forward_submit_s": submit_s,
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

    def supports_async_pipeline_packing(self, model: AnyTrainableModel) -> bool:
        return self._model_key(model) in self._clients

    async def prepare_pipeline_batch(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool = True,
    ) -> dict[str, float]:
        del normalize_advantages
        client = await self.training_client(model)
        started = time.monotonic()
        batch = RlTrajectoryBatch.from_groups(
            trajectory_groups,
            default_source_version=client.projected_learner_version,
        )
        prepared = await asyncio.to_thread(prepare_training_batch, batch)
        if any(
            group._prepared_training_batch is not None for group in trajectory_groups
        ):
            raise RuntimeError("trajectory group already owns a prepared batch")
        for group in trajectory_groups:
            group._prepared_training_batch = prepared
        return {
            "time/step_prepare_remote_batch_s": time.monotonic() - started,
            "data/step_remote_batch_bytes": float(prepared.ref.byte_count),
        }

    async def discard_pipeline_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> None:
        prepared = trajectory_groups[0]._prepared_training_batch
        if not isinstance(prepared, EncodedTrainingBatch) or any(
            group._prepared_training_batch is not prepared
            for group in trajectory_groups
        ):
            raise RuntimeError("remote pipeline batch is not prepared")
        for group in trajectory_groups:
            group._prepared_training_batch = None

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
