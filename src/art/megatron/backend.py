import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import sys
import time
from typing import Any, AsyncIterator, Iterable, Literal, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation

from .. import dev, types
from .._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
    merge_gradient_step_metrics,
    should_save_optimizer_state,
)
from ..backend import AnyTrainableModel
from ..distributed.art_runtime import ArtRuntime
from ..local.backend import LocalBackend, _PackedTrainingBatch
from ..local.service import ModelService
from ..metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from ..model import Model, TrainableModel
from ..training import (
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
    TrainingOperation,
)
from ..training.client import (
    admit_and_settle_gradient_step,
    iter_sft_batch_schedule,
)
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import LocalTrainResult, TrainSFTConfig
from ..utils.lifecycle import complete_task
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from ..vllm_runtime import get_external_vllm_runtime_config
from .migrations import apply_megatron_migrations
from .runtime.specs import ResidentLoraInspectionResult, ResidentScoreResult
from .runtime_config import get_megatron_runtime_config


class _DistributedBatchPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    packed: Any
    groups: tuple[Any, ...]
    bundles: tuple[Any, ...]
    selections: tuple[Any, ...]
    generation_id: str = Field(min_length=1)
    runtime: Any


class _PackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advantage_balance: float
    allow_training_without_logprobs: bool
    scale_rewards: bool
    plot_tensors: bool
    packed_sequence_length: int = Field(ge=1)
    logprob_calculation_chunk_size: int = Field(ge=1)
    include_moe_routing: bool
    collect_packing_shapes: bool

    @classmethod
    def from_dev_config(
        cls,
        config: Any,
        *,
        include_moe_routing: bool,
        collect_packing_shapes: bool,
    ) -> "_PackingConfig":
        return cls(
            advantage_balance=config.get("advantage_balance", 0.0),
            allow_training_without_logprobs=config.get(
                "allow_training_without_logprobs", False
            ),
            scale_rewards=config.get("scale_rewards", True),
            plot_tensors=config.get("plot_tensors", False),
            packed_sequence_length=config["packed_sequence_length"],
            logprob_calculation_chunk_size=config.get(
                "logprob_calculation_chunk_size", 1024
            ),
            include_moe_routing=include_moe_routing,
            collect_packing_shapes=collect_packing_shapes,
        )


class _MegatronPipelineCommandContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    backend: SkipValidation[Any] = Field(exclude=True)
    model: SkipValidation[TrainableModel] = Field(exclude=True)
    service: SkipValidation[Any] = Field(exclude=True)
    client: SkipValidation[Any] = Field(exclude=True)
    batch: Any
    groups: tuple[TrajectoryGroup, ...]
    config: types.TrainConfig
    forward_request: ForwardBackwardRequest
    preparation_metrics: dict[str, float]
    expose_checkpoint_path: bool
    started: float = Field(ge=0)
    _finished: bool = PrivateAttr(default=False)

    async def forward_backward(
        self, sequence_id: int
    ) -> TrainingOperation[ForwardBackwardResult]:
        return await self.client.forward_backward(
            self.forward_request.model_copy(update={"sequence_id": sequence_id})
        )

    def optimizer_request(self, sequence_id: int) -> OptimStepRequest:
        return OptimStepRequest(
            run_id=self.client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence_id,
            optimizer=AdamConfig(learning_rate=self.config.learning_rate),
        )

    async def sampler_request(
        self, step: int, sequence_id: int
    ) -> SaveWeightsForSamplerRequest:
        return SaveWeightsForSamplerRequest(
            run_id=self.client.run_id,
            request_id=uuid.uuid4().hex,
            sequence_id=sequence_id,
            checkpoint_name=f"step-{step}",
            publication=SamplerPublication(
                mode=_sampler_publication_mode(self.service),
                model_alias=self.model.name,
            ),
        )

    def state_request(self, step: int, sequence_id: int) -> SaveStateRequest | None:
        if not should_save_optimizer_state(step, self.config):
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
        forward: Any,
        optimizer: Any,
        sampler: Any,
        state: Any,
    ) -> None:
        del forward, optimizer
        self._consume_operation(sampler, "pipeline-sampler")
        if state is not None:
            self._consume_operation(state, "pipeline-state")

    async def complete(
        self,
        *,
        step: int,
        forward: Any,
        optimizer: Any,
        forward_submit_s: float,
    ) -> LocalTrainResult:
        try:
            results = await asyncio.gather(
                forward.result(), optimizer.result(), return_exceptions=True
            )
            failures = [value for value in results if isinstance(value, BaseException)]
            if failures:
                raise BaseExceptionGroup("Megatron train operations failed", failures)
            forward_result, optimizer_result = results
            if not (
                isinstance(forward_result, ForwardBackwardResult)
                and isinstance(optimizer_result, OptimStepResult)
            ):
                raise TypeError("Megatron pipeline returned invalid command results")
            _attach_packing_shapes(
                list(self.groups), forward_result.packing.group_shapes
            )
            metrics = aggregate_rl_training_metrics(
                training_metrics=[
                    {
                        **merge_gradient_step_metrics(
                            forward_result.metrics, optimizer_result.metrics
                        ),
                        **_packing_outcome_metrics(forward_result.packing),
                        "time/step_local_forward_submit_s": forward_submit_s,
                        **self.service.drain_publication_metrics(),
                    }
                ],
                trajectory_groups=self.groups,
                trainer_started=self.started,
            )
            policy_counts = forward_result.packing.policy_token_counts
            if policy_counts is None:
                raise RuntimeError("Megatron RL packing omitted policy token counts")
            if (
                self.config.final_training_step is not None
                and step >= self.config.final_training_step
            ):
                metrics.update(await self.service.finalize_publication_metrics(step))
            result = LocalTrainResult(
                step=step,
                metrics=metrics,
                packed_policy_token_counts=tuple(
                    (value.policy_version, value.trainable_assistant_tokens)
                    for value in policy_counts
                ),
            )
            if self.expose_checkpoint_path:
                result.checkpoint_path = get_step_checkpoint_dir(
                    get_model_dir(model=self.model, art_path=self.backend._path), step
                )
                if not Path(result.checkpoint_path).exists():
                    result.checkpoint_ready = self.service.checkpoint_materialization(
                        step
                    )
            wandb_run = self.model._get_wandb_run()
            if wandb_run is not None:
                self.backend._record_provenance_nonblocking(wandb_run, "local-rl")
            await self._finish(failed=False)
            return result
        except BaseException:
            await self._finish(failed=True)
            raise

    async def abort(
        self,
        forward: Any | None,
        optimizer: Any | None,
        sampler: Any | None,
        *,
        optimizer_admitted: bool,
    ) -> None:
        del sampler
        if not optimizer_admitted:
            if forward is not None:
                await forward.cancel()
            await self._finish(failed=True)
            return
        if forward is None or optimizer is None:
            raise RuntimeError("optimizer admission lost its F/B command")

        async def settle() -> None:
            failed = False
            try:
                await asyncio.gather(forward.result(), optimizer.result())
            except BaseException:
                failed = True
            await self._finish(failed=failed)

        self.backend._track_pipeline_operation(
            asyncio.create_task(settle(), name="pipeline-command-abort-settlement")
        )

    async def _finish(self, *, failed: bool) -> None:
        if self._finished:
            return
        object.__setattr__(self, "_finished", True)
        await self.backend._finish_training_batch(self.batch, failed=failed)

    def _consume_operation(self, operation: Any, name: str) -> None:
        self.backend._track_pipeline_operation(
            asyncio.create_task(operation.result(), name=name)
        )


def _packing_metrics(packed: Any) -> dict[str, float]:
    return {
        "time/step_trajectory_fetch_s": packed.trajectory_fetch_s,
        "time/step_trajectory_receive_s": packed.trajectory_receive_s,
        "time/step_trajectory_build_s": packed.trajectory_build_s,
        "time/step_packing_core_s": packed.packing_core_s,
        "time/step_packing_lock_wait_s": packed.packing_lock_wait_s,
        "time/step_packing_compute_s": packed.packing_compute_s,
        **{
            f"time/step_{name}": value
            for name, value in packed.packing_timings.model_dump().items()
        },
        "time/step_trajectory_log_wait_s": packed.trajectory_log_wait_s,
        "time/step_packed_batch_finalize_s": packed.packed_batch_finalize_s,
        "time/step_packing_rpc_s": packed.packing_rpc_s,
        "time/step_packed_batch_fanout_s": packed.packed_batch_fanout_s,
    }


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


def _attach_packing_shapes(
    groups: list[TrajectoryGroup], shapes: tuple[Any, ...]
) -> None:
    if not any(group._collect_packing_shape for group in groups):
        return
    if len(groups) != len(shapes):
        raise RuntimeError("Megatron packing shapes do not match trajectory groups")
    for group, shape in zip(groups, shapes, strict=True):
        group._packed_group_shape = shape


def _sampler_publication_mode(
    service: Any,
) -> Literal["versioned_lora", "in_flight_lora"]:
    return (
        "in_flight_lora"
        if service.rollout_weight_update_mode == "in_flight_lora"
        else "versioned_lora"
    )


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
        runtime: ArtRuntime | None = None,
    ) -> None:
        if in_process:
            raise ValueError(
                "MegatronBackend(in_process=True) belonged to the removed "
                "filesystem service proxy and cannot represent a multi-rank typed "
                "trainer. Use the default Monarch executor."
            )
        if runtime is not None:
            artifact_root = runtime.topology.cluster.artifact_root
            if artifact_root is None:
                raise ValueError("distributed Megatron requires cluster.artifact_root")
            if (
                path is not None
                and Path(path).resolve() != Path(artifact_root).resolve()
            ):
                raise ValueError("backend path must match cluster.artifact_root")
            path = artifact_root
        super().__init__(
            in_process=False,
            path=path,
            enable_expert_replay=enable_expert_replay,
        )
        self._requires_explicit_packed_sequence_length = True
        self._packed_sequence_length_requires_chunk_alignment = False
        self._supports_result_packing = True
        self._runtime = runtime
        self._owns_runtime = runtime is None
        self._runtime_lock = asyncio.Lock()
        self._service_lock = asyncio.Lock()
        self._training_client_lock = asyncio.Lock()
        self._training_clients: dict[tuple[str, str], Any] = {}
        self._owned_runtimes: dict[tuple[str, str], ArtRuntime] = {}
        from .runtime.local import LocalEndpointAllocator

        self._local_endpoints = LocalEndpointAllocator()
        self._owned_runtime_ports: dict[tuple[str, str], tuple[int, int]] = {}
        self._managed_api_key = secrets.token_urlsafe(32)
        self._batch_release_tasks: set[asyncio.Task[None]] = set()
        self._batch_release_failures: list[BaseException] = []
        self._pipeline_operation_tasks: set[asyncio.Task[Any]] = set()
        self._pipeline_operation_failures: list[BaseException] = []
        self._adapter_prune_requests: dict[
            tuple[str, str], tuple[AnyTrainableModel, set[int]]
        ] = {}
        self._adapter_prune_task: asyncio.Task[None] | None = None
        self._adapter_prune_failures: list[BaseException] = []

    def __enter__(self) -> "MegatronBackend":
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self
        raise RuntimeError(
            "Use 'async with MegatronBackend()' inside an async event loop"
        )

    def _close(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
            return
        raise RuntimeError(
            "MegatronBackend synchronous close cannot run inside an async event loop"
        )

    async def __aenter__(self) -> "MegatronBackend":
        return self

    def _compile_local_topology(
        self,
        model: TrainableModel,
        config: Any,
        *,
        service_ports: tuple[int, int] | None = None,
    ) -> Any:
        import torch

        from .runtime.local import compile_local_runtime_topology

        return compile_local_runtime_topology(
            config,
            model_name=model.name,
            base_model=model.base_model,
            artifact_root=str(Path(self._path).resolve()),
            visible_gpu_count=int(torch.cuda.device_count()),
            service_ports=service_ports,
        )

    def _model_runtime_topology(self, model: TrainableModel) -> Any:
        storage_key = self._model_storage_key(model)
        runtime = self._runtime or self._owned_runtimes.get(storage_key)
        if runtime is not None:
            return runtime.topology
        return self._compile_local_topology(model, model._internal_config or {})

    async def _ensure_runtime(self, model: TrainableModel, config: Any) -> ArtRuntime:
        if self._runtime is not None:
            return self._runtime
        storage_key = self._model_storage_key(model)
        if runtime := self._owned_runtimes.get(storage_key):
            return runtime
        async with self._runtime_lock:
            if storage_key not in self._owned_runtimes:
                ports = self._local_endpoints.reserve()
                try:
                    topology = self._compile_local_topology(
                        model, config, service_ports=ports
                    )
                    if not topology.model_services:
                        self._local_endpoints.release(ports)
                        ports = None
                    placements = _topology_gpu_placements(topology)
                    conflicts = {
                        key: placements & _topology_gpu_placements(runtime.topology)
                        for key, runtime in self._owned_runtimes.items()
                        if placements & _topology_gpu_placements(runtime.topology)
                    }
                    if conflicts:
                        raise ValueError(
                            "backend-owned per-model runtimes require disjoint GPU "
                            f"placements; {storage_key!r} conflicts with {conflicts}"
                        )
                    runtime = await ArtRuntime.start_local(topology)
                except BaseException:
                    if ports is not None:
                        self._local_endpoints.release(ports)
                    raise
                self._owned_runtimes[storage_key] = runtime
                if ports is not None:
                    self._owned_runtime_ports[storage_key] = ports
        return self._owned_runtimes[storage_key]

    async def _configure_owned_api_port(self, model: TrainableModel, port: int) -> None:
        storage_key = self._model_storage_key(model)
        async with self._runtime_lock:
            runtime = self._owned_runtimes.get(storage_key)
            ports = self._owned_runtime_ports.get(storage_key)
            if runtime is None or ports is None:
                raise RuntimeError("owned model service runtime has not started")
            configured = self._local_endpoints.replace_api_port(ports, port)
            try:
                from .runtime.local import with_local_serving_port

                topology = with_local_serving_port(
                    runtime.topology,
                    model_name=model.name,
                    port=configured[0],
                    rendezvous_port=configured[1],
                )
            except BaseException:
                self._local_endpoints.replace_api_port(configured, ports[0])
                raise
            runtime.topology = topology
            self._owned_runtime_ports[storage_key] = configured

    async def register(self, model: Model) -> None:
        await super().register(model)
        if model.trainable:
            apply_megatron_migrations(get_model_dir(model=model, art_path=self._path))

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> LocalTrainResult:
        for removed_kwarg in ("packed_sequence_length", "megatron_topology"):
            if removed_kwarg in kwargs:
                raise TypeError(
                    f"MegatronBackend.train gets {removed_kwarg} from "
                    "art.init_megatron_runtime_config(...)."
                )
        groups = list(trajectory_groups)
        from .distributed_service import DistributedMegatronService

        result = await super().train(
            model,
            groups,
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            **kwargs,
        )
        service = cast(DistributedMegatronService, await self._get_service(model))
        final_step = kwargs.get("final_training_step")
        if final_step is not None and result.step >= final_step:
            result.metrics.update(
                await service.finalize_publication_metrics(result.step)
            )
        await service.wait_for_serving(result.step)
        result.metrics.update(service.drain_publication_metrics())
        if not kwargs.get("save_checkpoint", True):
            return result
        result.checkpoint_path = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self._path), result.step
        )
        if not Path(result.checkpoint_path).exists():
            result.checkpoint_ready = service.checkpoint_materialization(result.step)
        return result

    async def finalize_training_session(
        self, model: AnyTrainableModel
    ) -> dict[str, float]:
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        return await service.finalize_publication_metrics(await self._get_step(model))

    async def inspect_resident_lora(
        self,
        model: AnyTrainableModel,
        *,
        expected_learner_version: int,
    ) -> ResidentLoraInspectionResult:
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        service.prefetch_trainer()
        return await service.inspect_resident_lora(
            expected_learner_version=expected_learner_version
        )

    async def score_resident(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        expected_learner_version: int,
        top_k: int = 20,
        grad_accumulation_sequences: int | None = None,
    ) -> ResidentScoreResult:
        groups = list(trajectory_groups)
        if not groups or not any(group.trajectories for group in groups):
            raise ValueError("resident scoring requires at least one trajectory")
        stale = [
            (group_index, trajectory_index, initial, final)
            for group_index, group in enumerate(groups)
            for trajectory_index, trajectory in enumerate(group.trajectories)
            for initial, final in (
                (
                    trajectory.initial_policy_version,
                    trajectory.final_policy_version,
                ),
            )
            if initial != expected_learner_version or final != expected_learner_version
        ]
        if stale:
            raise ValueError(
                "resident score trajectories must have exact initial/final learner "
                f"provenance {expected_learner_version}; mismatches={stale[:8]}"
            )

        include_moe_routing = self._model_uses_expert_replay(model)
        dev_config = {
            "advantage_balance": 0.0,
            "allow_training_without_logprobs": False,
            "scale_rewards": True,
            "plot_tensors": False,
            "packed_sequence_length": (
                get_megatron_runtime_config().packed_sequence_length
            ),
            "logprob_calculation_chunk_size": 1024,
        }
        batch = await self._prepare_training_batch(
            model,
            groups,
            dev_config,
            include_moe_routing=include_moe_routing,
        )
        if batch is None:
            raise RuntimeError("resident scoring produced no packed batch")

        try:
            from ..distributed.art_runtime import DistributedPackedBatch
            from .distributed_service import DistributedMegatronService

            payload = batch.payload
            if not isinstance(payload, _DistributedBatchPayload):
                raise RuntimeError("resident scoring did not use the typed data plane")
            distributed_batch = cast(DistributedPackedBatch, payload.packed)
            service = cast(
                DistributedMegatronService,
                await self._get_service(model),
            )
            accumulation = await service.resolve_global_grad_accumulation_sequences(
                types.TrainConfig(
                    grad_accumulation_sequences=grad_accumulation_sequences
                )
            )
            return await service.score_resident_packed(
                distributed_batch,
                expected_learner_version=expected_learner_version,
                global_grad_accumulation_sequences=accumulation,
                top_k=top_k,
            )
        finally:
            primary = sys.exception()
            paths = {
                group._prepared_log_path
                for group in groups
                if group._prepared_log_path is not None
            }
            try:
                results = await asyncio.gather(
                    self._release_distributed_batch(
                        batch,
                        disposition="discarded",
                    ),
                    *(
                        asyncio.to_thread(Path(path).unlink, missing_ok=True)
                        for path in paths
                    ),
                    return_exceptions=True,
                )
                for group in groups:
                    group._prepared_log_path = None
                failures = [
                    result for result in results if isinstance(result, BaseException)
                ]
                if failures:
                    raise BaseExceptionGroup(
                        "resident score batch release failed", failures
                    )
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                raise BaseExceptionGroup(
                    "resident score and batch release failed",
                    [primary, cleanup_error],
                ) from None

    def _supports_concurrent_training_and_inference(
        self, model: AnyTrainableModel
    ) -> bool:
        topology = self._model_runtime_topology(cast(TrainableModel, model))
        services = tuple(
            service for service in topology.model_services if service.name == model.name
        )
        if len(services) == 1:
            return not services[0].temporal_gpu_sharing
        if (
            not services
            and get_external_vllm_runtime_config(model._internal_config or {})
            is not None
        ):
            return True
        raise ValueError(
            f"runtime topology must define one model service named {model.name!r}"
        )

    def supports_async_pipeline_packing(self, model: AnyTrainableModel) -> bool:
        return True

    @asynccontextmanager
    async def adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        await service.wait_for_serving(step)
        async with super().adapter_lease(model, step):
            yield

    @asynccontextmanager
    async def exact_adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        await service.wait_for_serving(step)
        async with super().exact_adapter_lease(model, step):
            yield

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config

        storage_key = self._model_storage_key(model)
        if service := self._services.get(storage_key):
            return service
        async with self._service_lock:
            if service := self._services.get(storage_key):
                return service
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
                lora_config=model.lora_config,
            )
            config["init_args"]["model_name"] = (
                (model._internal_config or {})
                .get("init_args", {})
                .get("model_name", model.base_model)
            )
            runtime = await self._ensure_runtime(model, config)
            from .distributed_service import DistributedMegatronService

            service = cast(
                ModelService,
                DistributedMegatronService(
                    model_name=model.name,
                    base_model=model.base_model,
                    config=config,
                    output_dir=get_model_dir(model=model, art_path=self._path),
                    runtime=runtime,
                    enable_expert_replay=self._enable_expert_replay,
                ),
            )
            if not self._owns_runtime:
                runtime.register_closeable(service)
            self._services[storage_key] = service
            return service

    async def training_client(self, model: TrainableModel):
        from .distributed_service import DistributedMegatronService
        from .training import LocalMegatronTrainingClient

        storage_key = self._model_storage_key(model)
        if client := self._training_clients.get(storage_key):
            return client
        async with self._training_client_lock:
            if client := self._training_clients.get(storage_key):
                return client
            service = cast(DistributedMegatronService, await self._get_service(model))
            run_id, learner_version = await service.prepare_command_run()
            client = LocalMegatronTrainingClient(
                run_id=run_id,
                learner_version=learner_version,
                backend=self,
                model=model,
                service=service,
            )
            self._training_clients[storage_key] = client
            return client

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        del dev_config, verbose
        from ..utils.sft import resolve_sft_batch_size

        batch_size = resolve_sft_batch_size(
            batch_size=config.batch_size,
            default_batch_size=self._default_sft_batch_size(),
        )
        client = None
        pending_row: dict[str, float] | None = None
        gradient_steps = 0
        for batch, learning_rate in iter_sft_batch_schedule(
            trajectories, batch_size, config.learning_rate
        ):
            if client is None:
                client = await self.training_client(cast(TrainableModel, model))
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
                    return_token_logprobs=False,
                )
            )
            settled = await admit_and_settle_gradient_step(
                forward,
                lambda: client.optim_step(
                    OptimStepRequest(
                        run_id=client.run_id,
                        request_id=uuid.uuid4().hex,
                        sequence_id=sequence + 1,
                        optimizer=AdamConfig(learning_rate=learning_rate),
                    )
                ),
            )
            if settled is None:
                continue
            forward_result, optimizer_result = settled
            row = merge_gradient_step_metrics(
                forward_result.metrics, optimizer_result.metrics
            )
            row.update(
                {
                    "data/step_num_trajectories": float(len(batch)),
                    "data/step_trainable_assistant_tokens": float(
                        forward_result.packing.trainable_assistant_tokens
                    ),
                    "data/step_num_dropped_trajectories": float(
                        forward_result.metrics.get(
                            "data/step_num_dropped_trajectories", 0.0
                        )
                    ),
                }
            )
            row.pop(TRAIN_GRADIENT_STEPS_KEY, None)
            if pending_row is not None:
                yield pending_row
            pending_row = row
            gradient_steps += 1
        if pending_row is None:
            return
        assert client is not None
        service = cast(
            Any,
            await self._get_service(cast(TrainableModel, model)),
        )
        sequence = client.next_sequence_id
        step = client.projected_learner_version
        sampler = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(
                    mode=_sampler_publication_mode(service),
                    model_alias=model.name,
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
        pending_row[TRAIN_GRADIENT_STEPS_KEY] = float(gradient_steps)
        pending_row.update(sampler_result.publication_metrics)
        pending_row.update(state_result.metrics)
        yield pending_row

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        if get_external_vllm_runtime_config(model._internal_config or {}) is not None:
            service.prefetch_trainer()
            return await super()._prepare_backend_for_training(model, config)
        config_dict = dict(config or {})
        server_args = dict(config_dict.get("server_args", {}))
        server_args.setdefault("api_key", self._managed_api_key)
        if self._owns_runtime and "port" in server_args:
            port = server_args["port"]
            if isinstance(port, bool) or not isinstance(port, int):
                raise TypeError("OpenAI server port must be an integer")
            if (
                service._managed_service_name is not None
                and service.openai_server_port != port
            ):
                raise RuntimeError("cannot change a running OpenAI server port")
            await self._configure_owned_api_port(cast(TrainableModel, model), port)
        if "port" not in server_args and not self._owns_runtime:
            server_args["port"] = service.openai_server_port
        config_dict["server_args"] = server_args
        service.prefetch_trainer()
        return await super()._prepare_backend_for_training(
            model, cast(dev.OpenAIServerConfig, config_dict)
        )

    async def _prepare_training_batch(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        dev_config: Any,
        *,
        include_moe_routing: bool,
    ) -> _PackedTrainingBatch | None:
        collect_packing_shapes = any(
            group._collect_packing_shape for group in trajectory_groups
        )
        from ..distributed.packing import PackingRequest
        from ..distributed.rollout import (
            DistributedTrajectorySelection,
            RolloutModelSpec,
        )
        from ..distributed.trajectory_store import TrajectoryGroupBundle

        selections = tuple(group._distributed_lease for group in trajectory_groups)
        selected = tuple(
            selection
            for selection in selections
            if isinstance(selection, DistributedTrajectorySelection)
        )
        for group, selection in zip(trajectory_groups, selections, strict=True):
            if isinstance(selection, DistributedTrajectorySelection):
                group._distributed_lease = None

        generation_id = uuid.uuid4().hex
        trajectory_log_path: str | None = None
        runtime: ArtRuntime | None = None
        packed: Any = None
        marked_packed = False
        transferred = False
        try:
            packing_config = _PackingConfig.from_dev_config(
                dev_config,
                include_moe_routing=include_moe_routing,
                collect_packing_shapes=collect_packing_shapes,
            )
            if selected and len(selected) != len(trajectory_groups):
                raise RuntimeError(
                    "distributed batch mixes owned and controller groups"
                )
            queue = selected[0].queue if selected else None
            if queue is not None and any(
                selection.queue is not queue for selection in selected
            ):
                raise RuntimeError("distributed batch spans trajectory queues")

            from .distributed_service import DistributedMegatronService

            service = cast(DistributedMegatronService, await self._get_service(model))
            runtime = service.runtime
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
            current_step = min(versions) if versions else await self._get_step(model)
            if selected:
                group_ids = tuple(
                    selection.lease.item.ref.result_id for selection in selected
                )
                record_ids = tuple(
                    record.record_id
                    for selection in selected
                    for record in selection.lease.item.ref.records
                )
                trajectory_log_path = str(
                    Path(get_model_dir(model=model, art_path=self._path))
                    / "trajectories"
                    / ".staging"
                    / f"{generation_id}.parquet"
                )
            else:
                group_ids = tuple(
                    f"{group.metadata.get('scenario_id', 'group')}:{index}"
                    for index, group in enumerate(trajectory_groups)
                )
                record_ids = tuple(
                    f"{group_id}:{trajectory_index}"
                    for group_id, group in zip(
                        group_ids, trajectory_groups, strict=True
                    )
                    for trajectory_index, _ in enumerate(group.trajectories)
                )
            local_selections = tuple(
                selection
                for selection in selected
                if selection.lease.item.ref.transfer is None
            )
            if local_selections and len(local_selections) != len(selected):
                raise RuntimeError("distributed batch mixes local and remote owners")
            local_groups = (
                tuple(
                    await asyncio.gather(
                        *(
                            queue.materialize_selection(selection)
                            for selection in selected
                        )
                    )
                )
                if queue is not None and local_selections
                else ()
            )
            packing_bundles = tuple(
                TrajectoryGroupBundle.from_group(group)
                for group in (local_groups if selected else tuple(trajectory_groups))
            )
            command_bundles = (
                packing_bundles
                if not selected
                else tuple(
                    TrajectoryGroupBundle.from_group(group)
                    for group in trajectory_groups
                )
            )
            request = PackingRequest(
                model=RolloutModelSpec.from_model(model),
                generation_id=generation_id,
                trajectory_groups=packing_bundles,
                trajectory_sources=(
                    ()
                    if local_selections
                    else tuple(selection.lease.item for selection in selected)
                ),
                trajectory_log_path=trajectory_log_path,
                group_ids=group_ids,
                record_ids=record_ids,
                min_source_version=min(versions, default=current_step),
                max_source_version=max(versions, default=current_step),
                **packing_config.model_dump(),
            )
            packed = await runtime.pack(request)
            if packed is None:
                return None
            if queue is not None:
                _, cancelled = await complete_task(
                    asyncio.create_task(queue.mark_packed(selected, generation_id))
                )
                marked_packed = True
                if cancelled is not None:
                    raise cancelled
            shapes = tuple(packed.packed_group_shapes)
            if len(shapes) != len(trajectory_groups):
                raise RuntimeError("packed-group shapes do not match trajectory groups")
            ref = packed.leases.ref
            stats = ref.prefix_tree_packing_stats
            if stats is None:
                raise RuntimeError(
                    "distributed packed batch has no prefix-tree statistics"
                )
            batch = _PackedTrainingBatch(
                payload=_DistributedBatchPayload(
                    packed=packed,
                    groups=tuple(trajectory_groups),
                    bundles=command_bundles,
                    selections=selected,
                    generation_id=generation_id,
                    runtime=runtime,
                ),
                num_sequences=ref.num_sequences,
                sequence_length=ref.sequence_length,
                trainable_assistant_tokens=packed.trainable_assistant_tokens,
                loss_bearing_tokens=packed.loss_bearing_tokens,
                non_padding_tokens=packed.non_padding_tokens,
                logical_tokens=stats.logical_tokens,
                physical_tokens=stats.physical_tokens,
                include_moe_routing=include_moe_routing,
            )
            for group, shape in zip(trajectory_groups, shapes, strict=True):
                if shape is not None:
                    group._packed_group_shape = shape
                if selected:
                    group._prepared_log_path = packed.trajectory_log_path
            transferred = True
            return batch
        finally:
            if not transferred:
                primary = sys.exception()
                try:
                    _, cancelled = await complete_task(
                        asyncio.create_task(
                            self._cleanup_packing_ownership(
                                runtime=runtime,
                                packed=packed,
                                selections=selected,
                                generation_id=(
                                    generation_id if marked_packed else None
                                ),
                                trajectory_log_path=trajectory_log_path,
                            )
                        )
                    )
                    if cancelled is not None:
                        raise cancelled
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    raise BaseExceptionGroup(
                        "packing and source cleanup failed",
                        [primary, cleanup_error],
                    ) from None

    async def _cleanup_packing_ownership(
        self,
        *,
        runtime: ArtRuntime | None,
        packed: Any,
        selections: tuple[Any, ...],
        generation_id: str | None,
        trajectory_log_path: str | None,
    ) -> None:
        paths = {
            path
            for path in (
                trajectory_log_path,
                getattr(packed, "trajectory_log_path", None),
            )
            if path is not None
        }
        releases = [
            *(
                (runtime.release_batch(packed),)
                if runtime is not None and packed is not None
                else ()
            ),
            *(
                selection.queue.release_selection(
                    selection,
                    disposition="discarded",
                    generation_id=generation_id,
                )
                for selection in selections
            ),
            *(asyncio.to_thread(Path(path).unlink, missing_ok=True) for path in paths),
        ]
        results = await asyncio.gather(*releases, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("packing ownership cleanup failed", failures)

    async def prepare_pipeline_commands(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool = True,
        train_kwargs: dict[str, Any],
        learner_parent_version: int,
    ) -> _MegatronPipelineCommandContext | None:
        self._raise_pipeline_operation_failures()
        if normalize_advantages != train_kwargs.get("normalize_advantages", True):
            raise ValueError("pipeline reward normalization configuration changed")
        if train_kwargs.get("loss_fn", "cispo") not in {"cispo", "ppo"}:
            raise ValueError("Megatron pipeline supports only cispo and ppo")
        if train_kwargs.get("loss_fn_config") is not None:
            raise ValueError("Megatron pipeline requires loss_fn_config=None")
        if train_kwargs.get("adam_params") is not None:
            raise ValueError("Megatron pipeline requires adam_params=None")
        kl_reference_step = train_kwargs.get("kl_penalty_reference_step")
        kl_ref_adapter_path = (
            get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), kl_reference_step
            )
            if kl_reference_step is not None
            else None
        )
        config, dev_config = build_rl_train_configs(
            learning_rate=float(train_kwargs.get("learning_rate", 5e-6)),
            scale_rewards=normalize_advantages,
            ppo=train_kwargs.get("loss_fn", "cispo") == "ppo",
            kl_penalty_coef=float(train_kwargs.get("kl_penalty_coef", 0.0)),
            kl_penalty_source=train_kwargs.get("kl_penalty_source", "current_learner"),
            packed_sequence_length=(
                get_megatron_runtime_config().packed_sequence_length
            ),
            kl_ref_adapter_path=kl_ref_adapter_path,
            optimizer_save_interval=int(train_kwargs.get("optimizer_save_interval", 5)),
            final_training_step=train_kwargs.get("final_training_step"),
            grad_accumulation_sequences=train_kwargs.get("grad_accumulation_sequences"),
        )
        client = await self.training_client(model)
        if client.projected_learner_version != learner_parent_version:
            raise RuntimeError(
                "Megatron pipeline parent changed before F/B admission: "
                f"expected={learner_parent_version}, "
                f"projected={client.projected_learner_version}"
            )
        started = time.monotonic()
        include_moe_routing = self._model_uses_expert_replay(model)
        batch = await self._prepare_training_batch(
            model,
            trajectory_groups,
            dev_config,
            include_moe_routing=include_moe_routing,
        )
        if batch is None:
            return None
        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            raise RuntimeError(
                "Megatron pipeline batch did not use the typed data plane"
            )
        distributed = payload.packed
        metrics = _packing_metrics(distributed)
        from .distributed_service import DistributedMegatronService

        service = cast(
            DistributedMegatronService,
            await self._get_service(model),
        )
        try:
            metrics.update(
                await service.prepare_cp_lookahead(
                    distributed,
                    global_grad_accumulation_sequences=(
                        config.grad_accumulation_sequences
                    ),
                )
            )
        except BaseException as primary:
            paths = {
                group._prepared_log_path
                for group in trajectory_groups
                if group._prepared_log_path is not None
            }
            try:
                _, cancelled = await complete_task(
                    asyncio.create_task(
                        self._discard_prepared_resources(
                            batch, trajectory_groups, paths
                        )
                    )
                )
                if cancelled is not None:
                    raise cancelled
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "CP lookahead and packed-batch cleanup failed",
                    [primary, cleanup_error],
                ) from None
            raise
        values = {
            **dev_config,
            "kl_penalty_coef": config.kl_penalty_coef,
            "kl_penalty_source": config.kl_penalty_source,
            "grad_accumulation_sequences": config.grad_accumulation_sequences,
        }
        rl_batch = RlTrajectoryBatch(
            groups=payload.bundles,
            min_source_version=payload.packed.leases.ref.min_source_version,
            max_source_version=payload.packed.leases.ref.max_source_version,
        )
        object.__setattr__(rl_batch, "_local_groups", payload.groups)
        object.__setattr__(rl_batch, "_local_packed_batch", batch)
        return _MegatronPipelineCommandContext(
            backend=self,
            model=model,
            service=service,
            client=client,
            batch=batch,
            groups=tuple(trajectory_groups),
            config=config,
            forward_request=ForwardBackwardRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=client.next_sequence_id,
                batch=rl_batch,
                loss=LossConfig(
                    name="ppo" if dev_config.get("ppo", False) else "cispo",
                    normalize_advantages=bool(dev_config.get("scale_rewards", True)),
                    values=values,
                ),
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in trajectory_groups
                ),
                return_token_logprobs=False,
            ),
            preparation_metrics=metrics,
            expose_checkpoint_path=bool(train_kwargs.get("save_checkpoint", True)),
            started=started,
        )

    async def _discard_prepared_resources(
        self,
        batch: _PackedTrainingBatch,
        trajectory_groups: list[TrajectoryGroup],
        paths: set[str],
    ) -> None:
        results = await asyncio.gather(
            self._release_distributed_batch(batch, disposition="discarded"),
            *(asyncio.to_thread(Path(path).unlink, missing_ok=True) for path in paths),
            return_exceptions=True,
        )
        for group in trajectory_groups:
            group._prepared_log_path = None
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("prepared batch discard failed", failures)

    async def _stream_prepared_training(
        self,
        model: TrainableModel,
        service: ModelService,
        batch: _PackedTrainingBatch,
        config: Any,
        service_dev_config: Any,
        grad_accumulation_sequences: int,
        verbose: bool,
    ) -> AsyncIterator[dict[str, float]]:
        del verbose
        self._raise_pipeline_operation_failures()
        self._collect_batch_release_results()
        self._raise_batch_release_failures()
        self._collect_adapter_prune_result()
        self._raise_adapter_prune_failures()
        from .distributed_service import DistributedMegatronService

        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            raise RuntimeError("Megatron training did not use the typed data plane")
        distributed_service = cast(DistributedMegatronService, service)
        client = await self.training_client(model)
        values = {
            **service_dev_config,
            "kl_penalty_coef": config.kl_penalty_coef,
            "kl_penalty_source": config.kl_penalty_source,
            "grad_accumulation_sequences": grad_accumulation_sequences,
        }
        rl_batch = RlTrajectoryBatch(
            groups=payload.bundles,
            min_source_version=payload.packed.leases.ref.min_source_version,
            max_source_version=payload.packed.leases.ref.max_source_version,
        )
        object.__setattr__(rl_batch, "_local_groups", payload.groups)
        object.__setattr__(rl_batch, "_local_packed_batch", batch)
        sequence_id = client.next_sequence_id
        forward = await client.forward_backward(
            ForwardBackwardRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence_id,
                batch=rl_batch,
                loss=LossConfig(
                    name="ppo" if service_dev_config.get("ppo", False) else "cispo",
                    normalize_advantages=bool(
                        service_dev_config.get("scale_rewards", True)
                    ),
                    values=values,
                ),
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in payload.groups
                ),
                return_token_logprobs=False,
            )
        )
        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence_id + 1,
                optimizer=AdamConfig(learning_rate=config.learning_rate),
            )
        )
        next_step = optimizer.ref.reserved_output_learner_version
        if next_step is None:
            raise RuntimeError("optimizer did not reserve a learner version")
        sampler = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=uuid.uuid4().hex,
                sequence_id=sequence_id + 2,
                checkpoint_name=f"step-{next_step}",
                publication=SamplerPublication(
                    mode=_sampler_publication_mode(distributed_service),
                    model_alias=model.name,
                ),
            )
        )
        self._track_pipeline_operation(
            asyncio.create_task(
                sampler.result(), name=f"sync-sampler-{client.run_id}-{next_step}"
            )
        )
        if should_save_optimizer_state(next_step, config):
            state = await client.save_state(
                SaveStateRequest(
                    run_id=client.run_id,
                    request_id=uuid.uuid4().hex,
                    sequence_id=sequence_id + 3,
                    checkpoint_name=f"step-{next_step}",
                )
            )
            self._track_pipeline_operation(
                asyncio.create_task(
                    state.result(), name=f"sync-state-{client.run_id}-{next_step}"
                )
            )
        forward_result, optimizer_result = await asyncio.gather(
            forward.result(), optimizer.result()
        )
        metrics = {
            **merge_gradient_step_metrics(
                forward_result.metrics, optimizer_result.metrics
            ),
            **distributed_service.drain_publication_metrics(),
        }
        yield metrics

    async def _release_training_batch(self, batch: _PackedTrainingBatch) -> None:
        await self._release_distributed_batch(batch, disposition="consumed")

    async def _release_trajectory_sources(
        self,
        batch: _PackedTrainingBatch,
        payload: _DistributedBatchPayload,
    ) -> None:
        selections = payload.selections
        if not selections:
            return
        distributed = payload.packed
        ref = distributed.leases.ref
        expected_groups = tuple(
            selection.lease.item.ref.result_id for selection in selections
        )
        expected_records = tuple(
            record.record_id
            for selection in selections
            for record in selection.lease.item.ref.records
        )
        versions = [
            version
            for selection in selections
            for initial, final in zip(
                selection.lease.item.ref.descriptor.trajectory_initial_policy_versions,
                selection.lease.item.ref.descriptor.trajectory_final_policy_versions,
                strict=True,
            )
            for version in (
                initial
                if initial is not None
                else selection.lease.item.annotations.initial_policy_version,
                final
                if final is not None
                else selection.lease.item.annotations.final_policy_version,
            )
        ]
        if (
            distributed.packing_generation_id != payload.generation_id
            or ref.group_ids != expected_groups
            or ref.record_ids != expected_records
            or ref.min_source_version != min(versions)
            or ref.max_source_version != max(versions)
        ):
            raise RuntimeError("packed batch policy provenance does not match")
        queue = selections[0].queue
        if any(selection.queue is not queue for selection in selections):
            raise RuntimeError("packed batch contains selections from multiple queues")
        await queue.release_selections(
            selections,
            disposition="consumed",
            generation_id=payload.generation_id,
        )
        batch.payload = payload.model_copy(update={"selections": ()})

    async def _finish_training_batch(
        self, batch: _PackedTrainingBatch, *, failed: bool
    ) -> None:
        if failed:
            await super()._finish_training_batch(batch, failed=failed)
            if self._batch_release_tasks:
                await asyncio.gather(
                    *tuple(self._batch_release_tasks), return_exceptions=True
                )
            self._collect_batch_release_results()
            self._raise_batch_release_failures()
            return
        self._collect_batch_release_results()
        self._raise_batch_release_failures()
        while len(self._batch_release_tasks) >= 2:
            await asyncio.wait(
                self._batch_release_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self._collect_batch_release_results()
            self._raise_batch_release_failures()
        self._batch_release_tasks.add(
            asyncio.create_task(self._release_training_batch(batch))
        )

    def _collect_batch_release_results(self) -> None:
        for task in tuple(self._batch_release_tasks):
            if not task.done():
                continue
            self._batch_release_tasks.remove(task)
            try:
                task.result()
            except BaseException as error:
                self._batch_release_failures.append(error)

    def _raise_batch_release_failures(self) -> None:
        if not self._batch_release_failures:
            return
        failures, self._batch_release_failures = self._batch_release_failures, []
        raise BaseExceptionGroup("distributed training batch release failed", failures)

    def _track_pipeline_operation(self, task: asyncio.Task[Any]) -> None:
        self._pipeline_operation_tasks.add(task)

        def completed(value: asyncio.Task[Any]) -> None:
            self._pipeline_operation_tasks.discard(value)
            if not value.cancelled() and (error := value.exception()) is not None:
                self._pipeline_operation_failures.append(error)

        task.add_done_callback(completed)

    def _raise_pipeline_operation_failures(self) -> None:
        if not self._pipeline_operation_failures:
            return
        failures, self._pipeline_operation_failures = (
            self._pipeline_operation_failures,
            [],
        )
        raise BaseExceptionGroup("Megatron pipeline operations failed", failures)

    async def prune_model_adapters(
        self,
        model: AnyTrainableModel,
        *,
        retain_steps: set[int],
    ) -> None:
        service = await self._get_service(cast(TrainableModel, model))
        if getattr(service, "rollout_weight_update_mode", None) == "in_flight_lora":
            return
        self._collect_adapter_prune_result()
        self._raise_adapter_prune_failures()
        self._adapter_prune_requests[self._model_storage_key(model)] = (
            model,
            set(retain_steps),
        )
        if self._adapter_prune_task is None:
            self._adapter_prune_task = asyncio.create_task(self._prune_adapters())

    async def _prune_adapters(self) -> None:
        while self._adapter_prune_requests:
            requests, self._adapter_prune_requests = self._adapter_prune_requests, {}
            for model, retain_steps in requests.values():
                await super().prune_model_adapters(model, retain_steps=retain_steps)

    def _collect_adapter_prune_result(self) -> None:
        task = self._adapter_prune_task
        if task is None or not task.done():
            return
        self._adapter_prune_task = None
        try:
            task.result()
        except BaseException as error:
            self._adapter_prune_failures.append(error)

    def _raise_adapter_prune_failures(self) -> None:
        if not self._adapter_prune_failures:
            return
        failures, self._adapter_prune_failures = self._adapter_prune_failures, []
        raise BaseExceptionGroup("Megatron adapter pruning failed", failures)

    async def close(self) -> None:
        task = asyncio.create_task(self._close_megatron_backend())
        _, cancelled = await complete_task(task)
        if cancelled is not None:
            raise cancelled

    async def _close_megatron_backend(self) -> None:
        failures: list[BaseException] = []
        if self._pipeline_operation_tasks:
            await asyncio.gather(
                *tuple(self._pipeline_operation_tasks), return_exceptions=True
            )
            await asyncio.sleep(0)
        failures.extend(self._pipeline_operation_failures)
        self._pipeline_operation_failures.clear()
        clients, self._training_clients = tuple(self._training_clients.values()), {}
        failures.extend(
            result
            for result in await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True,
            )
            if isinstance(result, BaseException)
        )
        if self._batch_release_tasks:
            results = await asyncio.gather(
                *self._batch_release_tasks, return_exceptions=True
            )
            self._batch_release_tasks.clear()
            failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
        failures.extend(self._batch_release_failures)
        self._batch_release_failures.clear()
        if self._adapter_prune_task is not None:
            result = await asyncio.gather(
                self._adapter_prune_task, return_exceptions=True
            )
            if isinstance(result[0], BaseException):
                failures.append(result[0])
            self._adapter_prune_task = None
        failures.extend(self._adapter_prune_failures)
        self._adapter_prune_failures.clear()
        services = dict(self._services)
        services_closed = True
        try:
            await super().close()
        except BaseException as error:
            failures.append(error)
            services_closed = False
            for key, service in services.items():
                self._services.setdefault(key, service)
        if services_closed:
            runtimes = tuple(self._owned_runtimes.items())
            results = await asyncio.gather(
                *(runtime.close() for _, runtime in runtimes),
                return_exceptions=True,
            )
            for (key, runtime), result in zip(runtimes, results, strict=True):
                if isinstance(result, BaseException):
                    failures.append(result)
                elif self._owned_runtimes.get(key) is runtime:
                    self._owned_runtimes.pop(key)
                    ports = self._owned_runtime_ports.pop(key, None)
                    if ports is not None:
                        self._local_endpoints.release(ports)
        if failures:
            raise BaseExceptionGroup(
                "distributed Megatron backend close failed", failures
            )

    async def _release_distributed_batch(
        self,
        batch: _PackedTrainingBatch,
        *,
        disposition: Literal["consumed", "discarded"],
    ) -> None:
        from ..distributed.art_runtime import DistributedPackedBatch

        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            raise RuntimeError("Megatron batch has no owning typed runtime")
        runtime = cast(ArtRuntime, payload.runtime)
        distributed_batch = cast(DistributedPackedBatch, payload.packed)
        releases: list[Any] = [runtime.release_batch(distributed_batch)]
        if payload.selections:
            queue = payload.selections[0].queue
            releases.append(
                queue.release_selections(
                    payload.selections,
                    disposition=disposition,
                    generation_id=payload.generation_id,
                )
            )
        results = await asyncio.gather(*releases, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup(
                "distributed training batch release failed", failures
            )

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        from ..local.checkpoints import delete_checkpoints
        from .distributed_service import DistributedMegatronService
        from .optimizer_state import optimizer_retention_lease

        service = cast(DistributedMegatronService, await self._get_service(model))
        output_dir = get_model_dir(model=model, art_path=self._path)
        async with service.checkpoint_retention_lease() as active_steps:

            def delete_retained() -> set[int]:
                retained = set(steps_to_keep) | set(active_steps)
                with optimizer_retention_lease(output_dir, retained) as protected:
                    delete_checkpoints(output_dir, sorted(protected))
                    return set(protected)

            retained = await asyncio.to_thread(delete_retained)
            service.prune_checkpoint_metadata_locked(retain_steps=retained)
            client = self._training_clients.get(self._model_storage_key(model))
            if client is not None:
                client.prune_checkpoints(retain_steps=retained)

    async def _advance_skipped_step(
        self,
        model: TrainableModel,
        service: ModelService,
        current_step: int,
        next_step: int,
    ) -> dict[str, float]:
        from .distributed_service import DistributedMegatronService

        distributed = cast(DistributedMegatronService, service)
        return await distributed.advance_without_training(
            expected_step=current_step,
            learner_version=next_step,
        )

    async def _get_step(self, model: AnyTrainableModel) -> int:
        if not model.trainable:
            return 0
        await self._get_service(cast(TrainableModel, model))
        storage_key = self._model_storage_key(model)
        if storage_key in self._services:
            from .distributed_service import DistributedMegatronService

            service = cast(DistributedMegatronService, self._services[storage_key])
            return await service.prepare_for_packing()
        raise RuntimeError("Megatron model service was not initialized")

    def _default_sft_batch_size(self) -> int:
        import torch

        num_gpus = max(int(torch.cuda.device_count()), 1)
        tensor_parallel_size = min(2, num_gpus)
        return max(num_gpus // tensor_parallel_size, 1)


def _topology_gpu_placements(topology: Any) -> frozenset[tuple[str, int | str]]:
    trainer = () if topology.trainer is None else topology.trainer.ranks
    return frozenset(
        [(rank.host_id, rank.gpu_id) for rank in trainer]
        + [
            (member.host_id, gpu_id)
            for service in topology.model_services
            for member in service.members
            for gpu_id in member.gpu_ids
        ]
    )
