from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import secrets
import sys
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Literal, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .. import dev, types
from .._backend_training import (
    aggregate_rl_training_metrics,
    build_rl_train_configs,
    merge_gradient_step_metrics,
)
from ..backend import AnyTrainableModel
from ..distributed.art_runtime import ArtRuntime
from ..local.backend import LocalBackend, _PackedTrainingBatch
from ..local.service import ModelService
from ..metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from ..model import Model, TrainableModel
from ..preprocessing.pack import DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import LocalTrainResult, TrainSFTConfig
from ..utils.lifecycle import complete_task
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from ..vllm_route_transport import RouteBundleReader, unique_retained_route_bundles
from ..vllm_runtime import get_external_vllm_runtime_config
from .migrations import apply_megatron_migrations
from .runtime.specs import ResidentLoraInspectionResult
from .runtime_config import get_megatron_runtime_config

if TYPE_CHECKING:
    from ..pipeline_tuner import PackedGroupShape
    from ..preprocessing.sft import SftBatchTokenizer
    from ..preprocessing.token_matrix import LoweredTokenMatrixBatch
    from ..preprocessing.tokenize import TokenizedResult
    from ..training import NamedLossRequest, TokenMatrixBatch
    from ..vllm_route_transport import RetainedRouteBundleRef
    from .slot_runtime import (
        MegatronRunBinding,
        MegatronSlotRuntime,
    )
    from .training import LocalMegatronTrainingClient

_CONTEXT_PARALLEL_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH = 256
_PIPELINE_TRAIN_DISPATCH: ContextVar[asyncio.Event | None] = ContextVar(
    "megatron_pipeline_train_dispatch", default=None
)
_PIPELINE_TRAIN_STEP: ContextVar[int | None] = ContextVar(
    "megatron_pipeline_train_step", default=None
)
_BOUND_SAVE_CHECKPOINT: ContextVar[bool] = ContextVar(
    "megatron_bound_save_checkpoint", default=True
)
_COMMITTED_LEARNER_STEP_METRIC = "_art/committed_learner_step"


def _sampler_publication_mode(
    model: TrainableModel,
) -> Literal["versioned_lora", "in_flight_lora"]:
    return (
        "in_flight_lora"
        if (model._internal_config or {}).get("rollout_weight_update_mode")
        == "in_flight_lora"
        else "versioned_lora"
    )


def _should_save_optimizer_state(step: int, config: types.TrainConfig) -> bool:
    return (
        step <= 1
        or step % config.optimizer_save_interval == 0
        or (
            config.final_training_step is not None
            and step >= config.final_training_step
        )
    )


class _DistributedBatchPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    packed: Any
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
    min_prefix_tree_shared_segment_length: int = Field(ge=0)

    @classmethod
    def from_dev_config(
        cls,
        config: Any,
        *,
        include_moe_routing: bool,
        collect_packing_shapes: bool,
    ) -> "_PackingConfig":
        topology = get_megatron_runtime_config().topology
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
            min_prefix_tree_shared_segment_length=(
                _CONTEXT_PARALLEL_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
                if topology.cp > 1
                else DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
            ),
        )


class _PipelinePreparedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    batch: Any
    groups: tuple[Any, ...]
    packing_config: _PackingConfig
    metrics: dict[str, float]


@dataclass(slots=True)
class _BoundPreparedBatch:
    groups: tuple[TrajectoryGroup, ...]
    materialized: tuple[TrajectoryGroup, ...]
    selections: tuple[Any, ...]
    queue: Any | None
    packing_generation: str
    claimed: bool = False
    marked: bool = False
    released: bool = False


class _BoundPipelineCommandContext:
    """One materialized batch retained one command ahead of the slot GPU loop."""

    def __init__(
        self,
        backend: "MegatronBackend",
        model: TrainableModel,
        prepared: _BoundPreparedBatch,
        *,
        train_kwargs: dict[str, Any],
    ) -> None:
        self.preparation_metrics: dict[str, float] = {}
        self._backend = backend
        self._model = model
        self._prepared = prepared
        self._train_kwargs = train_kwargs

    async def complete(
        self,
        next_train_dispatched: asyncio.Event | None,
        next_batch_handoff: asyncio.Event | None,
    ) -> LocalTrainResult:
        if self._prepared.claimed:
            raise RuntimeError("bound pipeline command was consumed twice")
        if next_train_dispatched is not None:
            next_train_dispatched.set()
        if next_batch_handoff is not None:
            next_batch_handoff.set()
        return await self._backend.train(
            self._model,
            list(self._prepared.groups),
            **self._train_kwargs,
        )

    async def abort(self) -> None:
        if self._prepared.claimed:
            return
        self._prepared.claimed = True
        await self._backend._release_bound_prepared(
            self._prepared, disposition="discarded"
        )


class _MegatronPipelineCommandContext:
    """One exact packed command ready for the serialized trainer GPU loop."""

    def __init__(
        self,
        backend: "MegatronBackend",
        model: TrainableModel,
        groups: tuple[TrajectoryGroup, ...],
        *,
        prepared: _PipelinePreparedBatch,
        service: Any,
        batch: _PackedTrainingBatch,
        packed: Any,
        config: types.TrainConfig,
        experimental_config: dev.TrainConfig,
        learner_parent_version: int,
        train_kwargs: dict[str, Any],
        preparation_metrics: dict[str, float],
    ) -> None:
        self.preparation_metrics = preparation_metrics
        self._backend = backend
        self._model = model
        self._groups = groups
        self._prepared = prepared
        self._service = service
        self._batch: _PackedTrainingBatch | None = batch
        self._packed = packed
        self._config = config
        self._experimental_config = experimental_config
        self._learner_parent_version = learner_parent_version
        self._train_kwargs = train_kwargs
        self._claimed = False

    async def complete(
        self,
        next_train_dispatched: asyncio.Event | None,
        next_batch_handoff: asyncio.Event | None,
    ) -> LocalTrainResult:
        if self._claimed:
            raise RuntimeError("pipeline train context was consumed twice")
        self._claimed = True
        trainer_started = time.monotonic()
        if next_train_dispatched is not None:
            next_train_dispatched.set()
        if next_batch_handoff is not None:
            # CPU selection, packing, and materialization for the next command can
            # overlap this command's GPU turn. The prepared queue remains bounded
            # to one item, and only the consumer dispatches trainer work.
            next_batch_handoff.set()
        try:
            forward = await self._service.start_pipeline_forward_backward(
                self._packed,
                self._config,
                self._experimental_config,
                expected_learner_version=self._learner_parent_version,
            )
            optimizer = await self._service.start_pipeline_optimizer(
                forward,
                learning_rate=self._config.learning_rate,
            )
            values = await asyncio.gather(
                forward.completion,
                optimizer.completion,
                return_exceptions=True,
            )
            failures = [value for value in values if isinstance(value, BaseException)]
            if failures:
                raise BaseExceptionGroup("Megatron pipeline commands failed", failures)
            forward_result, optimizer_result = values
            if not isinstance(forward_result, dict):
                raise TypeError("Megatron pipeline F/B returned an invalid result")
            raw_optimizer = getattr(optimizer_result, "raw", None)
            publication_metrics = getattr(optimizer_result, "publication_metrics", None)
            if not isinstance(raw_optimizer, dict) or not isinstance(
                publication_metrics, dict
            ):
                raise TypeError(
                    "Megatron pipeline optimizer returned an invalid result"
                )
            metrics = aggregate_rl_training_metrics(
                training_metrics=[
                    {
                        **merge_gradient_step_metrics(
                            forward_result.get("metrics", {}),
                            raw_optimizer.get("metrics", {}),
                        ),
                        **forward.setup_metrics,
                        **publication_metrics,
                        **self._service.drain_publication_metrics(),
                    }
                ],
                trajectory_groups=self._groups,
                trainer_started=trainer_started,
            )
            batch = cast(_PackedTrainingBatch, self._prepared.batch)
            metrics.update(
                {
                    "data/step_trainable_assistant_tokens": float(
                        batch.trainable_assistant_tokens
                    ),
                    "data/step_packed_sequences": float(batch.num_sequences),
                    "prefix_tree/logical_tokens": float(batch.logical_tokens),
                    "prefix_tree/physical_tokens": float(batch.physical_tokens),
                    "prefix_tree/compression_ratio": (
                        batch.logical_tokens / batch.physical_tokens
                    ),
                }
            )
            step = optimizer.step
            final_step = self._config.final_training_step
            if final_step is not None and step >= final_step:
                metrics.update(await self._service.finalize_publication_metrics(step))
            result = LocalTrainResult(step=step, metrics=metrics)
            if self._train_kwargs.get("save_checkpoint", True):
                result.checkpoint_path = get_step_checkpoint_dir(
                    get_model_dir(model=self._model, art_path=self._backend._path),
                    step,
                )
                if not Path(result.checkpoint_path).exists():
                    result.checkpoint_ready = self._service.checkpoint_materialization(
                        step
                    )
            wandb_run = self._model._get_wandb_run()
            if wandb_run is not None:
                self._backend._record_provenance_nonblocking(wandb_run, "local-rl")
            await self._finish(failed=False)
            return result
        except BaseException:
            await self._finish(failed=True)
            raise
        finally:
            if next_batch_handoff is not None:
                next_batch_handoff.set()

    async def abort(self) -> None:
        if self._claimed:
            return
        self._claimed = True
        await self._finish(failed=True)

    async def _finish(self, *, failed: bool) -> None:
        batch, self._batch = self._batch, None
        if batch is not None:
            await self._backend._finish_training_batch(batch, failed=failed)


class MegatronBackend(LocalBackend):
    supports_pipeline_train_dispatch_fence = True

    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
        runtime: ArtRuntime | None = None,
        route_bundle_reader: RouteBundleReader | None = None,
        training_binding: MegatronRunBinding | None = None,
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
        self._training_binding = training_binding
        self._training_client: LocalMegatronTrainingClient | None = None
        self._training_client_model_key: tuple[str, str] | None = None
        self._training_client_lock = asyncio.Lock()
        from ..preprocessing.sft import SftBatchTokenizer

        self._sft_tokenizer: SftBatchTokenizer = SftBatchTokenizer()
        self._owned_slot_runtime: MegatronSlotRuntime | None = None
        self._owned_slot_ports: tuple[int, int] | None = None
        self._route_bundle_reader = route_bundle_reader
        self._owns_runtime = runtime is None
        self._runtime_lock = asyncio.Lock()
        self._service_lock = asyncio.Lock()
        self._owned_runtimes: dict[tuple[str, str], ArtRuntime] = {}
        from .runtime.local import LocalEndpointAllocator

        self._local_endpoints = LocalEndpointAllocator()
        self._owned_runtime_ports: dict[tuple[str, str], tuple[int, int]] = {}
        self._managed_api_key = secrets.token_urlsafe(32)
        self._batch_release_tasks: set[asyncio.Task[None]] = set()
        self._batch_release_failures: list[BaseException] = []
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
                    runtime = await ArtRuntime.start_local(
                        topology,
                        route_bundle_reader=self._route_bundle_reader,
                    )
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
            binding = self._training_binding
            if binding is not None:
                if model.run_id not in {None, binding.config.run_id}:
                    raise ValueError("model run_id differs from its Megatron binding")
                model.run_id = binding.config.run_id

    async def _bind_owned_training_run(
        self,
        model: TrainableModel,
        openai_config: dev.OpenAIServerConfig | None,
    ) -> None:
        """Start the normal local workflow on the production run-slot boundary."""

        if self._training_binding is not None:
            return
        async with self._training_client_lock:
            if self._training_binding is not None:
                return
            if self._owned_slot_runtime is not None:
                raise RuntimeError("one Megatron backend cannot own multiple slot runs")

            from ..dev.get_model_config import get_model_config
            from ..distributed.rollout import RolloutModelSpec
            from ..training import AdapterSpec, TrainingRunSpec
            from .local_checkpoint import MegatronLocalCheckpointOperations
            from .slot_runtime import (
                MegatronRunBinding,
                MegatronRunBootstrapConfig,
                MegatronSlotLaunchConfig,
                launch_megatron_slot,
                prepare_megatron_run_config,
            )

            output_dir = get_model_dir(model=model, art_path=self._path)
            config: dict[str, Any] = dict(
                get_model_config(
                    base_model=model.base_model,
                    output_dir=output_dir,
                    config=model._internal_config,
                    lora_config=model.lora_config,
                )
            )
            init_args = dict(config.get("init_args", {}))
            init_args["model_name"] = (
                (model._internal_config or {})
                .get("init_args", {})
                .get("model_name", model.base_model)
            )
            config["init_args"] = init_args
            client_config = dict(openai_config or {})
            for key in ("engine_args", "server_args"):
                values = dict(config.get(key, {}))
                values.update(dict(client_config.get(key, {})))
                config[key] = values
            server_args = dict(config.get("server_args", {}))
            server_args.setdefault("api_key", self._managed_api_key)
            config["server_args"] = server_args

            ports = self._local_endpoints.reserve()
            requested_port = server_args.get("port")
            if requested_port is not None:
                if isinstance(requested_port, bool) or not isinstance(
                    requested_port, int
                ):
                    self._local_endpoints.release(ports)
                    raise TypeError("OpenAI server port must be an integer")
                ports = self._local_endpoints.replace_api_port(ports, requested_port)
            slot = None
            try:
                topology = self._compile_local_topology(
                    model, config, service_ports=ports
                )
                slot = await launch_megatron_slot(
                    MegatronSlotLaunchConfig(
                        slot_id=f"local-{uuid.uuid4().hex}",
                        runtime_source_epoch=0,
                        topology=topology,
                        megatron=get_megatron_runtime_config(),
                        base_model=model.base_model,
                        model=dict(config),
                        enable_moe_routing_replay=self._enable_expert_replay,
                    ),
                    route_bundle_reader=self._route_bundle_reader,
                )
                publisher = slot.paired_inference
                if publisher is None:
                    raise RuntimeError(
                        "normal MegatronBackend training requires paired inference"
                    )
                service = publisher.service
                model.inference_base_url = (
                    f"{service.leader_endpoint.url.rstrip('/')}/v1"
                )
                model.inference_api_key = publisher.api_key or "default"
                model.inference_model_name = service.name
                object.__setattr__(
                    model, "_serving_capabilities", publisher.capabilities
                )
                object.__setattr__(
                    model, "_inference_connection_errors_are_fatal", True
                )
                if publisher.capabilities.binary_routed_experts:
                    object.__setattr__(
                        model,
                        "_art_binary_routes_base_url",
                        f"{service.leader_endpoint.url.rstrip('/')}/art/v1",
                    )

                runtime_spec = slot.coordinator.trainer.runtime_spec
                run_id = model.run_id or model.run_name
                model.run_id = run_id
                bootstrap = MegatronRunBootstrapConfig(
                    run_id=run_id,
                    training_session_id=run_id,
                    run=TrainingRunSpec(
                        base_model=model.base_model,
                        adapter=AdapterSpec(
                            rank=runtime_spec.lora_rank,
                            target_modules=runtime_spec.lora_target_modules,
                        ),
                        seed=runtime_spec.random_state,
                        dtype="bfloat16",
                    ),
                    output_dir=output_dir,
                )
                rollout_model = RolloutModelSpec.from_model(model)
                operation_config = await asyncio.to_thread(
                    prepare_megatron_run_config,
                    bootstrap,
                    runtime_spec,
                    rollout_model=rollout_model,
                )
                checkpoints = MegatronLocalCheckpointOperations(
                    slot.coordinator,
                    Path(output_dir) / "checkpoints",
                    run_id=run_id,
                    training_session_id=run_id,
                    output_adapter_root=operation_config.output_adapter_root,
                    optimizer_state_path=operation_config.optimizer_state_path,
                )
                run = await slot.coordinator.register_run(
                    operation_config,
                    checkpoints=checkpoints,
                )
                binding = MegatronRunBinding(
                    run=run,
                    config=operation_config,
                    coordinator=slot.coordinator,
                    publisher=publisher,
                )
            except BaseException:
                if slot is not None:
                    await slot.aclose()
                self._local_endpoints.release(ports)
                raise
            self._training_binding = binding
            self._owned_slot_runtime = slot
            self._owned_slot_ports = ports

    async def training_client(
        self, model: TrainableModel
    ) -> LocalMegatronTrainingClient:
        """Bind this backend to one already-registered physical slot run."""

        binding = self._training_binding
        if binding is None:
            raise RuntimeError("MegatronBackend has no registered training slot run")
        if model.run_id != binding.config.run_id:
            raise RuntimeError("model is not registered to its Megatron run binding")
        key = self._model_storage_key(model)
        if self._training_client is not None:
            if self._training_client_model_key != key:
                raise RuntimeError("one training slot run cannot back multiple models")
            return self._training_client
        async with self._training_client_lock:
            if self._training_client is None:
                from .training import LocalMegatronTrainingClient

                self._training_client = LocalMegatronTrainingClient.from_binding(
                    binding
                )
                self._training_client_model_key = key
            elif self._training_client_model_key != key:
                raise RuntimeError("one training slot run cannot back multiple models")
            return self._training_client

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
        if kwargs.get("loss_fn", "cispo") == "ppo":
            raise ValueError("Megatron TokenMatrix training does not support PPO")
        dispatch_event = kwargs.pop("_pipeline_train_dispatch_event", None)
        if dispatch_event is not None and not isinstance(dispatch_event, asyncio.Event):
            raise TypeError("pipeline train dispatch fence must be an asyncio.Event")
        groups = list(trajectory_groups)
        pipeline_call = bool(
            groups
            and isinstance(
                groups[0]._prepared_training_batch,
                (_PipelinePreparedBatch, _BoundPreparedBatch),
            )
        )
        from .distributed_service import DistributedMegatronService

        if dispatch_event is not None and dispatch_event.is_set():
            raise RuntimeError("pipeline train dispatch fence is already set")
        token = _PIPELINE_TRAIN_DISPATCH.set(dispatch_event)
        step_token = _PIPELINE_TRAIN_STEP.set(None)
        save_token = _BOUND_SAVE_CHECKPOINT.set(
            bool(kwargs.get("save_checkpoint", True))
        )
        try:
            if (
                dispatch_event is not None
                and not pipeline_call
                and self._training_binding is None
            ):
                raise RuntimeError("trainer dispatch fencing requires a prepared batch")
            result = await super().train(
                model,
                groups,
                packed_sequence_length=(
                    get_megatron_runtime_config().packed_sequence_length
                ),
                **kwargs,
            )
        finally:
            _BOUND_SAVE_CHECKPOINT.reset(save_token)
            _PIPELINE_TRAIN_STEP.reset(step_token)
            _PIPELINE_TRAIN_DISPATCH.reset(token)
        if self._training_binding is not None:
            return result
        service = cast(DistributedMegatronService, await self._get_service(model))
        final_step = kwargs.get("final_training_step")
        if final_step is not None and result.step >= final_step:
            result.metrics.update(
                await service.finalize_publication_metrics(result.step)
            )
        if not pipeline_call:
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

    async def _train_model(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: types.TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if self._training_binding is None:
            async for metrics in super()._train_model(
                model, trajectory_groups, config, dev_config, verbose
            ):
                yield metrics
            return
        async for metrics in self._train_bound_rl(
            model, trajectory_groups, config, dev_config, verbose
        ):
            yield metrics

    async def _train_bound_rl(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: types.TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool,
    ) -> AsyncIterator[dict[str, float]]:
        from ..preprocessing.token_matrix import token_matrix_batch_from_art_rollouts
        from ..training import (
            AdamConfig,
            ForwardBackwardRequest,
            ForwardBackwardResult,
            OptimStepRequest,
            OptimStepResult,
            SamplerPublication,
            SamplerWeightsResult,
            SaveStateRequest,
            SaveWeightsForSamplerRequest,
        )

        if not trajectory_groups:
            raise ValueError("Megatron training requires at least one group")
        loss = self._rl_token_matrix_loss(dev_config)
        client = await self.training_client(model)
        prepared = await self._claim_or_prepare_bound_batch(trajectory_groups)
        operations: list[tuple[Any, Any]] = []
        acknowledged: set[str] = set()
        optimizer_committed = False
        optimizer_completed = False
        optimizer_operation_id: str | None = None
        forward_produced_gradient = False
        try:
            materialized = prepared.materialized
            if not any(group.trajectories for group in materialized):
                raise ValueError("Megatron training requires at least one trajectory")
            tokenized = self._tokenize_bound_rollouts(
                model,
                list(materialized),
                dev_config,
            )
            dispatch = _PIPELINE_TRAIN_DISPATCH.get()
            if not tokenized:
                if dispatch is not None:
                    dispatch.set()
                yield {TRAIN_GRADIENT_STEPS_KEY: 0.0}
                return
            lowered = token_matrix_batch_from_art_rollouts(
                tokenized,
                loss="cispo",
                normalize_advantages=bool(dev_config.get("scale_rewards", True)),
                advantage_balance=float(dev_config.get("advantage_balance", 0.0)),
            )
            batch = self._token_matrix_batch_with_routes(
                lowered,
                self._retained_route_bundles(
                    prepared.materialized,
                    prepared.selections,
                ),
                require_routes=self._model_uses_expert_replay(model),
            )
            if dispatch is not None:
                dispatch.set()
            request_id = secrets.token_hex(16)
            forward_request = ForwardBackwardRequest(
                run_id=client.run_id,
                request_id=f"fb-{request_id}",
                sequence_id=client.next_sequence_id,
                batch=batch,
                loss=loss,
                collect_packing_shapes=any(
                    group._collect_packing_shape for group in trajectory_groups
                ),
                return_token_logprobs=False,
            )
            forward = await client.forward_backward(forward_request)
            operations.append((forward_request, forward))
            forward_result = await forward.result()
            if not isinstance(forward_result, ForwardBackwardResult):
                raise TypeError("Megatron F/B returned an invalid result")
            forward_produced_gradient = forward_result.produced_gradient
            await self._acknowledge_bound_operation(
                forward_request, forward, acknowledged
            )
            if forward_result.packing.group_shapes:
                self._attach_rl_packing_shapes(
                    trajectory_groups,
                    batch,
                    forward_result.packing.group_shapes,
                )
            if not forward_result.produced_gradient:
                yield {
                    **forward_result.metrics,
                    "data/step_trainable_assistant_tokens": float(
                        forward_result.training.accepted_trainable_tokens
                    ),
                    TRAIN_GRADIENT_STEPS_KEY: 0.0,
                }
                return

            optimizer_request = OptimStepRequest(
                run_id=client.run_id,
                request_id=f"optim-{request_id}",
                sequence_id=client.next_sequence_id,
                optimizer=AdamConfig(learning_rate=config.learning_rate),
            )
            optimizer = await client.optim_step(optimizer_request)
            operations.append((optimizer_request, optimizer))
            optimizer_operation_id = optimizer.ref.operation_id
            optimizer_result = await optimizer.result()
            if not isinstance(optimizer_result, OptimStepResult):
                raise TypeError("Megatron optimizer returned an invalid result")
            optimizer_completed = True
            await self._acknowledge_bound_operation(
                optimizer_request, optimizer, acknowledged
            )
            optimizer_committed = True
            step = optimizer.ref.reserved_output_learner_version
            if step is None:
                raise RuntimeError("optimizer did not reserve a learner version")
            await self._release_bound_prepared(prepared, disposition="consumed")

            sampler_request = SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=f"publish-{request_id}",
                sequence_id=client.next_sequence_id,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(
                    mode=_sampler_publication_mode(model),
                    model_alias=model.name,
                ),
            )
            sampler = await client.save_weights_for_sampler(sampler_request)
            operations.append((sampler_request, sampler))
            sampler_result = await sampler.result()
            if not isinstance(sampler_result, SamplerWeightsResult):
                raise TypeError("Megatron publication returned an invalid result")
            await self._acknowledge_bound_operation(
                sampler_request, sampler, acknowledged
            )
            if _BOUND_SAVE_CHECKPOINT.get() and _should_save_optimizer_state(
                step, config
            ):
                state_request = SaveStateRequest(
                    run_id=client.run_id,
                    request_id=f"state-{request_id}",
                    sequence_id=client.next_sequence_id,
                    checkpoint_name=f"{step:04d}",
                )
                state = await client.save_state(state_request)
                operations.append((state_request, state))
                await state.result()
                await self._acknowledge_bound_operation(
                    state_request, state, acknowledged
                )
            _PIPELINE_TRAIN_STEP.set(step)
            if verbose:
                print(
                    "Megatron operations: "
                    + ", ".join(
                        operation.ref.operation_id for _request, operation in operations
                    )
                )
            yield {
                **merge_gradient_step_metrics(
                    forward_result.metrics, optimizer_result.metrics
                ),
                **sampler_result.metrics,
                "data/step_trainable_assistant_tokens": float(
                    forward_result.training.accepted_trainable_tokens
                ),
            }
        finally:
            failures: list[BaseException] = []
            for request, operation in operations:
                if operation.ref.operation_id not in acknowledged:
                    try:
                        await self._acknowledge_bound_operation(
                            request, operation, acknowledged
                        )
                    except BaseException as error:
                        failures.append(error)
            if (
                optimizer_completed
                and optimizer_operation_id is not None
                and optimizer_operation_id in acknowledged
            ):
                optimizer_committed = True
            if not prepared.released:
                try:
                    operation_ids = {
                        operation.ref.operation_id for _request, operation in operations
                    }
                    if optimizer_committed:
                        await self._release_bound_prepared(
                            prepared, disposition="consumed"
                        )
                    elif not forward_produced_gradient and operation_ids.issubset(
                        acknowledged
                    ):
                        await self._release_bound_prepared(
                            prepared, disposition="discarded"
                        )
                except BaseException as error:
                    failures.append(error)
            for _request, operation in operations:
                if operation.ref.operation_id not in acknowledged:
                    continue
                try:
                    if not client.retire_operation(operation.ref.operation_id):
                        raise RuntimeError(
                            "completed training evidence still has live command lineage"
                        )
                except BaseException as error:
                    failures.append(error)
            if failures:
                raise BaseExceptionGroup("Megatron training cleanup failed", failures)

    @staticmethod
    def _rl_token_matrix_loss(
        dev_config: dev.TrainConfig,
    ) -> NamedLossRequest:
        from ..training import NamedLossRequest

        if dev_config.get("ppo", False):
            raise ValueError("Megatron TokenMatrix training does not support PPO")
        epsilon = dev_config.get("epsilon")
        epsilon_high = dev_config.get("epsilon_high")
        return NamedLossRequest(
            name="cispo",
            normalize_advantages=False,
            values={
                "clip_low_threshold": 1.0
                - (1.0 if epsilon is None else float(epsilon)),
                "clip_high_threshold": 1.0
                + (4.0 if epsilon_high is None else float(epsilon_high)),
            },
        )

    @staticmethod
    def _retained_route_bundles(
        groups: tuple[TrajectoryGroup, ...],
        selections: tuple[Any, ...],
    ) -> tuple[RetainedRouteBundleRef, ...]:
        from ..vllm_route_transport import retained_route_bundles_from_groups

        if not selections:
            return retained_route_bundles_from_groups(groups)
        return unique_retained_route_bundles(
            ref
            for selection in selections
            for ref in selection.lease.item.ref.descriptor.retained_route_bundles
        )

    @staticmethod
    def _token_matrix_batch_with_routes(
        lowered: LoweredTokenMatrixBatch,
        retained_bundles: tuple[RetainedRouteBundleRef, ...] = (),
        *,
        require_routes: bool = False,
    ) -> TokenMatrixBatch:
        import hashlib
        import json

        from ..preprocessing.moe_routing import MoeRouteSegments
        from ..training import InlineTokenRoutes, RetainedTokenRoutes

        matrix_ids = {matrix.matrix_id for matrix in lowered.batch.matrices}
        routes_by_matrix: dict[str, Any] = {
            route.matrix_id: route for route in lowered.batch.routes
        }
        for matrix_id, route in lowered.resolved_routes.items():
            if matrix_id not in matrix_ids:
                raise ValueError("routing replay references an unknown TokenMatrix")
            if matrix_id in routes_by_matrix:
                raise ValueError("TokenMatrix routes were supplied twice")
            segments = (
                route.segments if isinstance(route, MoeRouteSegments) else (route,)
            )
            routes_by_matrix[matrix_id] = InlineTokenRoutes(
                matrix_id=matrix_id,
                num_experts=route.num_experts,
                shape=route.shape,
                expert_ids=b"".join(segment.tobytes(order="C") for segment in segments),
            )

        for matrix in lowered.batch.matrices:
            if matrix.matrix_id in routes_by_matrix:
                continue
            token_ids = tuple(
                int(value) for value in matrix.row("token_ids").dense_values()
            )
            digest = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest()
            matches = tuple(
                (ref, choice)
                for ref in retained_bundles
                for choice in ref.layout.choices
                if choice.shape[0] == matrix.token_count
                and choice.token_ids_sha256 == digest
            )
            if len(matches) > 1:
                raise RuntimeError(
                    "retained routing replay is ambiguous for one TokenMatrix"
                )
            if matches:
                ref, choice = matches[0]
                routes_by_matrix[matrix.matrix_id] = RetainedTokenRoutes(
                    matrix_id=matrix.matrix_id,
                    bundle=ref.model_dump(mode="json"),
                    choice_index=choice.choice_index,
                )

        if (require_routes or routes_by_matrix) and set(routes_by_matrix) != matrix_ids:
            raise ValueError("routing replay requires routes for every TokenMatrix")
        return lowered.batch.model_copy(
            update={
                "routes": tuple(
                    routes_by_matrix[matrix.matrix_id]
                    for matrix in lowered.batch.matrices
                    if matrix.matrix_id in routes_by_matrix
                )
            }
        )

    @staticmethod
    def _attach_rl_packing_shapes(
        trajectory_groups: list[TrajectoryGroup],
        batch: TokenMatrixBatch,
        shapes: tuple[PackedGroupShape, ...],
    ) -> None:
        leaves_by_matrix = {
            leaf.matrix_id: leaf for shape in shapes for leaf in shape.leaves
        }
        if len(leaves_by_matrix) != sum(len(shape.leaves) for shape in shapes):
            raise RuntimeError("packed-group shapes repeat a TokenMatrix")
        if set(leaves_by_matrix) != {matrix.matrix_id for matrix in batch.matrices}:
            raise RuntimeError("packed-group shapes changed matrix identity")

        from ..pipeline_tuner import PackedGroupShape

        for index, group in enumerate(trajectory_groups):
            leaves = tuple(
                leaves_by_matrix[matrix.matrix_id]
                for matrix in batch.matrices
                if matrix.packing_affinity_id == f"prompt-{index}"
            )
            if leaves:
                group._packed_group_shape = PackedGroupShape(leaves=leaves)

    def _tokenize_bound_rollouts(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        dev_config: dev.TrainConfig,
    ) -> list[TokenizedResult]:
        from ..local.backend import _load_training_tokenizer, _tokenizer_cache_key
        from ..preprocessing.tokenize import tokenize_trajectory_groups
        from ..trajectories._selection import automatic_training_model_selector

        internal_config = cast(dev.InternalModelConfig, model._internal_config or {})
        tokenizer_key = _tokenizer_cache_key(model.base_model, internal_config)
        if tokenizer_key not in self._tokenizers:
            self._tokenizers[tokenizer_key] = self._configure_training_tokenizer(
                _load_training_tokenizer(model.base_model),
                model=model,
                internal_config=internal_config,
            )
        return list(
            tokenize_trajectory_groups(
                self._tokenizers[tokenizer_key],
                trajectory_groups,
                bool(dev_config.get("allow_training_without_logprobs", False)),
                bool(dev_config.get("scale_rewards", True)),
                chat_template_kwargs=internal_config.get("chat_template_kwargs"),
                chat_template_tool_schema_format=(
                    self._chat_template_tool_schema_format(internal_config)
                ),
                model=automatic_training_model_selector(model.get_inference_name()),
                _max_sequence_length=min(
                    self._model_max_sequence_length(model),
                    get_megatron_runtime_config().packed_sequence_length,
                ),
            )
        )

    async def _claim_or_prepare_bound_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> _BoundPreparedBatch:
        retained = tuple(group._prepared_training_batch for group in trajectory_groups)
        if any(item is not None for item in retained):
            prepared = retained[0]
            if (
                not isinstance(prepared, _BoundPreparedBatch)
                or any(item is not prepared for item in retained)
                or prepared.groups != tuple(trajectory_groups)
                or prepared.claimed
            ):
                raise RuntimeError("bound pipeline preparation changed ownership")
        else:
            prepared = await self._prepare_bound_batch(trajectory_groups)
        prepared.claimed = True
        for group in trajectory_groups:
            group._prepared_training_batch = None
        return prepared

    async def _prepare_bound_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> _BoundPreparedBatch:
        from ..distributed.rollout import DistributedTrajectorySelection

        if not trajectory_groups:
            raise ValueError("Megatron training requires at least one group")
        if any(
            group._prepared_training_batch is not None for group in trajectory_groups
        ):
            raise RuntimeError("training batch is already prepared")
        leases = tuple(group._distributed_lease for group in trajectory_groups)
        selected = tuple(
            lease
            for lease in leases
            if isinstance(lease, DistributedTrajectorySelection)
        )
        if selected and len(selected) != len(trajectory_groups):
            raise RuntimeError("training batch mixes owned and controller groups")
        queue = selected[0].queue if selected else None
        if queue is not None and any(item.queue is not queue for item in selected):
            raise RuntimeError("training batch spans trajectory queues")
        for group, lease in zip(trajectory_groups, leases, strict=True):
            if isinstance(lease, DistributedTrajectorySelection):
                group._distributed_lease = None
        try:
            materialized = (
                tuple(
                    await asyncio.gather(
                        *(queue.materialize_selection(item) for item in selected)
                    )
                )
                if queue is not None
                else tuple(trajectory_groups)
            )
        except BaseException:
            if queue is not None:
                await queue.release_selections(selected, disposition="discarded")
            raise
        prepared = _BoundPreparedBatch(
            groups=tuple(trajectory_groups),
            materialized=materialized,
            selections=selected,
            queue=queue,
            packing_generation=uuid.uuid4().hex,
        )
        for group in trajectory_groups:
            group._prepared_training_batch = prepared
        return prepared

    async def _release_bound_prepared(
        self,
        prepared: _BoundPreparedBatch,
        *,
        disposition: Literal["consumed", "discarded"],
    ) -> None:
        if prepared.released:
            return
        queue = prepared.queue
        if queue is not None:
            if disposition == "consumed" and not prepared.marked:
                await queue.mark_packed(
                    prepared.selections, prepared.packing_generation
                )
                prepared.marked = True
            await queue.release_selections(
                prepared.selections,
                disposition=disposition,
                generation_id=(
                    prepared.packing_generation if prepared.marked else None
                ),
            )
        prepared.released = True

    async def _acknowledge_bound_operation(
        self,
        request: Any,
        operation: Any,
        acknowledged: set[str],
    ) -> None:
        operation_id = operation.ref.operation_id
        if operation_id in acknowledged:
            return
        binding = self._training_binding
        if binding is None:
            raise RuntimeError("bound Megatron workflow lost its run binding")
        if binding.outcome_sink is not None:
            await binding.outcome_sink.retain_outcome(
                request, await operation.outcome()
            )
        client = self._training_client
        if client is None or client.run_id != operation.ref.run_id:
            raise RuntimeError("bound Megatron client changed during acknowledgement")
        await client.acknowledge_operation(operation_id)
        acknowledged.add(operation_id)

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if self._training_binding is None:
            async for metrics in super()._train_sft(
                model, trajectories, config, dev_config, verbose
            ):
                yield metrics
            return
        del dev_config
        from ..preprocessing.token_matrix import token_matrix_batch_from_sft
        from ..training import (
            AdamConfig,
            ForwardBackwardRequest,
            ForwardBackwardResult,
            NamedLossRequest,
            OptimStepRequest,
            OptimStepResult,
            SamplerPublication,
            SaveWeightsForSamplerRequest,
        )
        from ..utils.sft import resolve_sft_batch_size

        values = list(trajectories)
        if not values:
            yield {
                "data/step_num_trajectories": 0.0,
                "data/step_trainable_assistant_tokens": 0.0,
                "data/step_num_dropped_trajectories": 0.0,
                "data/sft_zero_work": 1.0,
                TRAIN_GRADIENT_STEPS_KEY: 0.0,
            }
            return
        batch_size = resolve_sft_batch_size(
            batch_size=config.batch_size,
            default_batch_size=self._default_sft_batch_size(),
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
        client = await self.training_client(cast(TrainableModel, model))
        rows: list[dict[str, float]] = []
        final_step: int | None = None
        gradient_steps = 0
        for batch, learning_rate in zip(batches, learning_rates, strict=True):
            request_id = secrets.token_hex(16)
            operations: list[tuple[Any, Any]] = []
            acknowledged: set[str] = set()
            try:
                tokenized = self._sft_tokenizer.tokenize(
                    cast(TrainableModel, model),
                    batch,
                    assistant_turns=config.assistant_turns,
                    learning_rate=learning_rate,
                )
                token_matrix_batch = token_matrix_batch_from_sft(tokenized)
                if token_matrix_batch is None:
                    rows.append(
                        {
                            "data/step_num_trajectories": float(len(batch)),
                            "data/step_trainable_assistant_tokens": 0.0,
                            "data/step_num_dropped_trajectories": float(
                                tokenized.num_dropped_trajectories
                            ),
                            "data/sft_zero_work": 1.0,
                        }
                    )
                    continue
                forward_request = ForwardBackwardRequest(
                    run_id=client.run_id,
                    request_id=f"sft-fb-{request_id}",
                    sequence_id=client.next_sequence_id,
                    batch=token_matrix_batch,
                    loss=NamedLossRequest(
                        name="cross_entropy", normalize_advantages=False
                    ),
                    return_token_logprobs=False,
                )
                forward = await client.forward_backward(forward_request)
                operations.append((forward_request, forward))
                forward_result = await forward.result()
                if not isinstance(forward_result, ForwardBackwardResult):
                    raise TypeError("Megatron SFT F/B returned an invalid result")
                await self._acknowledge_bound_operation(
                    forward_request, forward, acknowledged
                )
                if not forward_result.produced_gradient:
                    rows.append(
                        {
                            **forward_result.metrics,
                            "data/step_num_trajectories": float(len(batch)),
                            "data/step_trainable_assistant_tokens": float(
                                forward_result.training.accepted_trainable_tokens
                            ),
                            "data/step_num_dropped_trajectories": float(
                                tokenized.num_dropped_trajectories
                            ),
                        }
                    )
                    continue
                optimizer_request = OptimStepRequest(
                    run_id=client.run_id,
                    request_id=f"sft-optim-{request_id}",
                    sequence_id=client.next_sequence_id,
                    optimizer=AdamConfig(
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        grad_clip_norm=1.0,
                    ),
                )
                optimizer = await client.optim_step(optimizer_request)
                operations.append((optimizer_request, optimizer))
                optimizer_result = await optimizer.result()
                if not isinstance(optimizer_result, OptimStepResult):
                    raise TypeError("Megatron SFT optimizer returned an invalid result")
                await self._acknowledge_bound_operation(
                    optimizer_request, optimizer, acknowledged
                )
                gradient_steps += 1
                final_step = optimizer_result.checkpoint.learner_version
                rows.append(
                    {
                        **merge_gradient_step_metrics(
                            forward_result.metrics, optimizer_result.metrics
                        ),
                        "data/step_num_trajectories": float(len(batch)),
                        "data/step_trainable_assistant_tokens": float(
                            forward_result.training.accepted_trainable_tokens
                        ),
                        "data/step_num_dropped_trajectories": float(
                            tokenized.num_dropped_trajectories
                        ),
                    }
                )
            finally:
                failures: list[BaseException] = []
                for request, operation in operations:
                    if operation.ref.operation_id not in acknowledged:
                        try:
                            await self._acknowledge_bound_operation(
                                request, operation, acknowledged
                            )
                        except BaseException as error:
                            failures.append(error)
                for _request, operation in operations:
                    if operation.ref.operation_id not in acknowledged:
                        continue
                    if not client.retire_operation(operation.ref.operation_id):
                        failures.append(
                            RuntimeError(
                                "completed SFT evidence still has live command lineage"
                            )
                        )
                if failures:
                    raise BaseExceptionGroup("Megatron SFT cleanup failed", failures)
        if final_step is not None:
            publication_request = SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=f"sft-publish-{secrets.token_hex(16)}",
                sequence_id=client.next_sequence_id,
                checkpoint_name=f"step-{final_step}",
                publication=SamplerPublication(
                    mode=_sampler_publication_mode(cast(TrainableModel, model)),
                    model_alias=model.name,
                ),
            )
            publication = await client.save_weights_for_sampler(publication_request)
            acknowledged: set[str] = set()
            try:
                await publication.result()
                await self._acknowledge_bound_operation(
                    publication_request, publication, acknowledged
                )
            finally:
                if publication.ref.operation_id not in acknowledged:
                    await self._acknowledge_bound_operation(
                        publication_request, publication, acknowledged
                    )
                if not client.retire_operation(publication.ref.operation_id):
                    raise RuntimeError(
                        "completed SFT publication evidence is not retireable"
                    )
            _PIPELINE_TRAIN_STEP.set(final_step)
            if verbose:
                print(f"Megatron SFT committed learner step {final_step}")
        for row in rows:
            row[TRAIN_GRADIENT_STEPS_KEY] = float(gradient_steps)
            yield row

    async def finalize_training_session(
        self, model: AnyTrainableModel
    ) -> dict[str, float]:
        if self._training_binding is not None:
            return {}
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        return await service.finalize_publication_metrics(await self._get_step(model))

    async def inspect_resident_lora(
        self,
        model: AnyTrainableModel,
        *,
        expected_learner_version: int,
    ) -> ResidentLoraInspectionResult:
        if self._training_binding is not None:
            raise RuntimeError(
                "resident inspection has no slot-owned diagnostic command boundary"
            )
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        service.prefetch_trainer()
        return await service.inspect_resident_lora(
            expected_learner_version=expected_learner_version
        )

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
        del model
        return True

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        binding = self._training_binding
        if binding is not None:
            from ..adapter_leases import pinned_inference_name

            if pinned := pinned_inference_name(model.name, step):
                return pinned
            publisher = binding.publisher
            if publisher is None:
                raise RuntimeError("bound Megatron run has no paired publisher")
            receipt = publisher.activated_publication(model.name, step)
            if receipt is not None:
                if step is not None and receipt.publication_mode == "in_flight_lora":
                    raise ValueError(
                        "In-flight LoRA serving requires an exact adapter lease"
                    )
                if receipt.runtime_lora_name is None:
                    raise RuntimeError("paired publication omitted its serving name")
                return receipt.runtime_lora_name
            if step not in (None, 0) or binding.config.source.policy_step != 0:
                raise RuntimeError("requested learner version is not activated")
            name = binding.config.rollout_model.payload.get("inference_model_name")
            if isinstance(name, str) and name:
                return name
        return super()._model_inference_name(model, step)

    @asynccontextmanager
    async def adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        binding = self._training_binding
        if binding is not None:
            publisher = binding.publisher
            if publisher is None:
                raise RuntimeError("bound Megatron run has no paired publisher")
            manager = self._adapter_lease_manager(model)
            if step == 0 and publisher.activated_publication(model.name, step) is None:
                from ..adapter_leases import pin_inference_target

                async with (
                    pin_inference_target(
                        model.name,
                        step=step,
                        inference_name=binding.config.rollout_model.payload.get(
                            "inference_model_name"
                        ),
                    ),
                    manager.lease(step),
                ):
                    yield
                return
            from ..adapter_leases import pin_inference_target

            receipt = publisher.activated_publication(model.name, step)
            if receipt is None or receipt.runtime_lora_name is None:
                raise RuntimeError("paired publication is not activated")
            async with (
                pin_inference_target(
                    model.name,
                    step=step,
                    inference_name=receipt.runtime_lora_name,
                ),
                manager.lease(step),
            ):
                yield
            return
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
        binding = self._training_binding
        if binding is not None:
            publisher = binding.publisher
            if publisher is None:
                raise RuntimeError("bound Megatron run has no paired publisher")
            from ..adapter_leases import pin_inference_target

            async with publisher.exact_publication_lease(model.name, step) as receipt:
                if receipt.runtime_lora_name is None:
                    raise RuntimeError(
                        "paired exact publication omitted its serving name"
                    )
                async with (
                    pin_inference_target(
                        model.name,
                        step=step,
                        inference_name=receipt.runtime_lora_name,
                    ),
                    self._adapter_lease_manager(model).lease(step),
                ):
                    yield
            return
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        await service.wait_for_serving(step)
        async with super().exact_adapter_lease(model, step):
            yield

    async def _get_service(self, model: TrainableModel) -> ModelService:
        if self._training_binding is not None:
            raise RuntimeError(
                "a bound Megatron run cannot enter DistributedMegatronService"
            )
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

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        binding = self._training_binding
        if binding is not None:
            del config
            rollout_model = binding.config.rollout_model.build()
            if (
                model.run_id != binding.config.run_id
                or rollout_model.base_model != model.base_model
            ):
                raise RuntimeError("bound rollout model changed training identity")
            base_url = rollout_model.inference_base_url
            api_key = rollout_model.inference_api_key
            if not base_url or not api_key:
                raise RuntimeError("bound rollout model has no inference endpoint")
            model._serving_capabilities = rollout_model._serving_capabilities
            model._art_binary_routes_base_url = (
                rollout_model._art_binary_routes_base_url
            )
            return base_url, api_key
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
        loss = self._rl_token_matrix_loss(cast(dev.TrainConfig, dev_config))
        prepared = tuple(group._prepared_training_batch for group in trajectory_groups)
        collect_packing_shapes = any(
            group._collect_packing_shape for group in trajectory_groups
        )
        if any(value is not None for value in prepared):
            first = prepared[0]
            if (
                not isinstance(first, _PipelinePreparedBatch)
                or any(value is not first for value in prepared)
                or first.groups != tuple(trajectory_groups)
            ):
                raise RuntimeError("pipeline prepared batch does not match training")
            packing_config = _PackingConfig.from_dev_config(
                dev_config,
                include_moe_routing=include_moe_routing,
                collect_packing_shapes=collect_packing_shapes,
            )
            if first.packing_config != packing_config:
                mismatch = RuntimeError(
                    "pipeline prepared batch packing configuration does not match "
                    "training"
                )
                try:
                    await self.discard_pipeline_batch(trajectory_groups)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "prepared batch mismatch cleanup failed",
                        [mismatch, cleanup_error],
                    ) from None
                raise mismatch
            for group in trajectory_groups:
                group._prepared_training_batch = None
            return cast(_PackedTrainingBatch, first.batch)
        from ..distributed.packing import (
            PackingRequest,
            retained_route_bundles_from_token_matrix_batch,
        )
        from ..distributed.rollout import DistributedTrajectorySelection
        from ..preprocessing.token_matrix import token_matrix_batch_from_art_rollouts

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
        runtime: ArtRuntime | None = None
        packed: Any = None
        marked_packed = False
        mark_packed_task: asyncio.Task[None] | None = None
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

            service = await self._get_service(model)
            runtime = cast(Any, service).runtime
            materialized = (
                tuple(
                    await asyncio.gather(
                        *(
                            queue.materialize_selection(selection)
                            for selection in selected
                        )
                    )
                )
                if queue is not None
                else tuple(trajectory_groups)
            )
            tokenized = self._tokenize_bound_rollouts(
                model,
                list(materialized),
                cast(dev.TrainConfig, dev_config),
            )
            if not tokenized:
                return None
            lowered = token_matrix_batch_from_art_rollouts(
                tokenized,
                loss="cispo",
                normalize_advantages=packing_config.scale_rewards,
                advantage_balance=packing_config.advantage_balance,
            )
            token_matrix_batch = self._token_matrix_batch_with_routes(
                lowered,
                self._retained_route_bundles(materialized, selected),
                require_routes=include_moe_routing,
            )
            request = PackingRequest(
                batch=token_matrix_batch,
                loss=loss,
                return_token_logprobs=False,
                generation_id=generation_id,
                retained_route_bundles=(
                    retained_route_bundles_from_token_matrix_batch(token_matrix_batch)
                ),
                packed_sequence_length=packing_config.packed_sequence_length,
                collect_packing_shapes=packing_config.collect_packing_shapes,
                min_prefix_tree_shared_segment_length=(
                    packing_config.min_prefix_tree_shared_segment_length
                ),
            )
            if queue is not None:
                mark_packed_task = asyncio.create_task(
                    queue.mark_packed(selected, generation_id)
                )
            try:
                packed = await runtime.pack(request)
            except BaseException as packing_error:
                if mark_packed_task is not None:
                    try:
                        _, cancelled = await complete_task(mark_packed_task)
                        marked_packed = True
                    except BaseException as marking_error:
                        raise BaseExceptionGroup(
                            "packing and trajectory lease marking failed",
                            [packing_error, marking_error],
                        ) from None
                    if cancelled is not None:
                        packing_error.add_note(
                            "trajectory lease marking observed cancellation"
                        )
                raise
            if mark_packed_task is not None:
                _, cancelled = await complete_task(mark_packed_task)
                marked_packed = True
                if cancelled is not None:
                    raise cancelled
            if packed is None:
                return None
            shapes = tuple(packed.packed_group_shapes)
            ref = packed.leases.ref
            stats = ref.prefix_tree_packing_stats
            if stats is None:
                raise RuntimeError(
                    "distributed packed batch has no prefix-tree statistics"
                )
            batch = _PackedTrainingBatch(
                payload=_DistributedBatchPayload(
                    packed=packed,
                    selections=selected,
                    generation_id=generation_id,
                    runtime=runtime,
                ),
                num_sequences=ref.num_sequences,
                sequence_length=ref.sequence_length,
                trainable_assistant_tokens=(
                    ref.training_outcome.accepted_trainable_tokens
                ),
                loss_bearing_tokens=ref.logical_loss_terms,
                non_padding_tokens=stats.physical_tokens,
                logical_tokens=stats.logical_tokens,
                physical_tokens=stats.physical_tokens,
                include_moe_routing=bool(token_matrix_batch.routes),
            )
            if shapes:
                self._attach_rl_packing_shapes(
                    trajectory_groups,
                    token_matrix_batch,
                    shapes,
                )
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
    ) -> None:
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
        ]
        results = await asyncio.gather(*releases, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("packing ownership cleanup failed", failures)

    async def prepare_pipeline_batch(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool = True,
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        grad_accumulation_sequences: int | None = None,
    ) -> dict[str, float] | None:
        include_moe_routing = self._model_uses_expert_replay(model)
        dev_config = {
            "advantage_balance": advantage_balance,
            "allow_training_without_logprobs": allow_training_without_logprobs,
            "scale_rewards": scale_rewards and normalize_advantages,
            "plot_tensors": plot_tensors,
            "packed_sequence_length": get_megatron_runtime_config().packed_sequence_length,
            "logprob_calculation_chunk_size": logprob_calculation_chunk_size,
        }
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
        metrics = {
            "time/step_batch_fetch_s": distributed.batch_fetch_s,
            "time/step_route_fetch_s": distributed.route_fetch_s,
            "time/step_packing_core_s": distributed.packing_core_s,
            "time/step_packed_batch_finalize_s": distributed.packed_batch_finalize_s,
            "time/step_packing_rpc_s": distributed.packing_rpc_s,
            "time/step_packed_batch_fanout_s": distributed.packed_batch_fanout_s,
        }
        from .distributed_service import DistributedMegatronService

        service = cast(
            DistributedMegatronService,
            await self._get_service(model),
        )
        packing_config = _PackingConfig.from_dev_config(
            dev_config,
            include_moe_routing=include_moe_routing,
            collect_packing_shapes=any(
                group._collect_packing_shape for group in trajectory_groups
            ),
        )
        try:
            metrics.update(
                await service.prepare_cp_lookahead(
                    distributed,
                    global_grad_accumulation_sequences=grad_accumulation_sequences,
                )
            )
        except BaseException as primary:
            cleanup_prepared = _PipelinePreparedBatch(
                batch=batch,
                groups=tuple(trajectory_groups),
                packing_config=packing_config,
                metrics=metrics,
            )
            paths = {
                group._prepared_log_path
                for group in trajectory_groups
                if group._prepared_log_path is not None
            }
            try:
                _, cancelled = await complete_task(
                    asyncio.create_task(
                        self._discard_prepared_resources(
                            cleanup_prepared, trajectory_groups, paths
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
        prepared = _PipelinePreparedBatch(
            batch=batch,
            groups=tuple(trajectory_groups),
            packing_config=packing_config,
            metrics=metrics,
        )
        for group in trajectory_groups:
            group._prepared_training_batch = prepared
        return metrics

    async def prepare_pipeline_commands(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        *,
        learner_parent_version: int,
        train_kwargs: dict[str, Any],
    ) -> _MegatronPipelineCommandContext | _BoundPipelineCommandContext | None:
        supported = {
            "adam_params",
            "grad_accumulation_sequences",
            "kl_penalty_coef",
            "kl_penalty_reference_step",
            "kl_penalty_source",
            "learning_rate",
            "loss_fn",
            "loss_fn_config",
            "normalize_advantages",
            "optimizer_save_interval",
            "save_checkpoint",
        }
        if unexpected := train_kwargs.keys() - supported:
            raise TypeError(f"unsupported Megatron pipeline options: {unexpected}")
        if train_kwargs.get("loss_fn", "cispo") != "cispo":
            raise ValueError("Megatron TokenMatrix pipeline supports only cispo")
        if train_kwargs.get("loss_fn_config") is not None:
            raise ValueError("Megatron pipeline requires loss_fn_config=None")
        if train_kwargs.get("adam_params") is not None:
            raise ValueError("Megatron pipeline requires adam_params=None")
        if self._training_binding is not None:
            client = await self.training_client(model)
            if client.projected_learner_version != learner_parent_version:
                raise RuntimeError(
                    "pipeline preparation learner lineage changed before materialization"
                )
            prepared = await self._prepare_bound_batch(trajectory_groups)
            return _BoundPipelineCommandContext(
                self,
                model,
                prepared,
                train_kwargs=dict(train_kwargs),
            )
        normalize_advantages = bool(train_kwargs.get("normalize_advantages", True))
        kl_reference_step = cast(
            int | None, train_kwargs.get("kl_penalty_reference_step")
        )
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
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            kl_ref_adapter_path=kl_ref_adapter_path,
            optimizer_save_interval=int(train_kwargs.get("optimizer_save_interval", 5)),
            grad_accumulation_sequences=cast(
                int | None, train_kwargs.get("grad_accumulation_sequences")
            ),
        )
        metrics = await self.prepare_pipeline_batch(
            model,
            trajectory_groups,
            normalize_advantages=normalize_advantages,
            advantage_balance=0.0,
            scale_rewards=normalize_advantages,
            allow_training_without_logprobs=False,
            plot_tensors=False,
            logprob_calculation_chunk_size=1024,
            grad_accumulation_sequences=config.grad_accumulation_sequences,
        )
        if metrics is None:
            return None
        prepared = trajectory_groups[0]._prepared_training_batch
        if not isinstance(prepared, _PipelinePreparedBatch) or any(
            group._prepared_training_batch is not prepared
            for group in trajectory_groups
        ):
            raise RuntimeError("pipeline batch preparation changed ownership")
        for group in trajectory_groups:
            group._prepared_training_batch = None
        batch = cast(_PackedTrainingBatch, prepared.batch)
        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            await self._finish_training_batch(batch, failed=True)
            raise RuntimeError("Megatron pipeline batch lost its typed data plane")
        from .distributed_service import DistributedMegatronService

        service = cast(
            DistributedMegatronService,
            await self._get_service(model),
        )
        return _MegatronPipelineCommandContext(
            self,
            model,
            tuple(trajectory_groups),
            prepared=prepared,
            service=service,
            batch=batch,
            packed=payload.packed,
            config=config,
            experimental_config=dev_config,
            learner_parent_version=learner_parent_version,
            train_kwargs=dict(train_kwargs),
            preparation_metrics=metrics,
        )

    async def discard_pipeline_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> None:
        prepared = trajectory_groups[0]._prepared_training_batch
        if isinstance(prepared, _BoundPreparedBatch):
            if any(
                group._prepared_training_batch is not prepared
                for group in trajectory_groups
            ):
                raise RuntimeError("bound pipeline batch preparation changed ownership")
            prepared.claimed = True
            for group in trajectory_groups:
                group._prepared_training_batch = None
            await self._release_bound_prepared(prepared, disposition="discarded")
            return
        if not isinstance(prepared, _PipelinePreparedBatch) or any(
            group._prepared_training_batch is not prepared
            for group in trajectory_groups
        ):
            raise RuntimeError("pipeline batch is not prepared")
        for group in trajectory_groups:
            group._prepared_training_batch = None
        paths = {
            group._prepared_log_path
            for group in trajectory_groups
            if group._prepared_log_path is not None
        }
        _, cancelled = await complete_task(
            asyncio.create_task(
                self._discard_prepared_resources(prepared, trajectory_groups, paths)
            )
        )
        if cancelled is not None:
            raise cancelled

    async def _discard_prepared_resources(
        self,
        prepared: _PipelinePreparedBatch,
        trajectory_groups: list[TrajectoryGroup],
        paths: set[str],
    ) -> None:
        results = await asyncio.gather(
            self._release_distributed_batch(prepared.batch, disposition="discarded"),
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
        self._collect_batch_release_results()
        self._raise_batch_release_failures()
        self._collect_adapter_prune_result()
        self._raise_adapter_prune_failures()
        from ..distributed.art_runtime import DistributedPackedBatch
        from .distributed_service import DistributedMegatronService

        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            raise RuntimeError("Megatron training did not use the typed data plane")
        distributed_batch = cast(
            DistributedPackedBatch,
            payload.packed,
        )
        if distributed_batch.packing_generation_id != payload.generation_id:
            raise RuntimeError("packed batch generation identity does not match")
        distributed_service = cast(DistributedMegatronService, service)
        async for result in distributed_service.train_packed(
            distributed_batch,
            config,
            service_dev_config,
            dispatch_event=_PIPELINE_TRAIN_DISPATCH.get(),
        ):
            committed_step = result.pop(_COMMITTED_LEARNER_STEP_METRIC, None)
            if committed_step is not None:
                exact_step = int(committed_step)
                if float(exact_step) != committed_step:
                    raise RuntimeError("trainer committed a non-integral learner step")
                _PIPELINE_TRAIN_STEP.set(exact_step)
            yield {
                **result,
                **distributed_service.drain_publication_metrics(),
            }

    async def _release_training_batch(self, batch: _PackedTrainingBatch) -> None:
        await self._release_distributed_batch(batch, disposition="consumed")

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

    async def prune_model_adapters(
        self,
        model: AnyTrainableModel,
        *,
        retain_steps: set[int],
    ) -> None:
        binding = self._training_binding
        if binding is not None:
            publisher = binding.publisher
            if publisher is None:
                raise RuntimeError("bound Megatron run has no paired publisher")
            manager = self._adapter_lease_manager(model)
            async with manager.prune_guard() as leased_steps:
                await publisher.prune_versioned_adapters(
                    model.name,
                    retain_steps=set(retain_steps) | leased_steps,
                )
            return
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
        if self._training_client is not None:
            try:
                await self._training_client.close()
            except BaseException as error:
                failures.append(error)
            self._training_client = None
            self._training_client_model_key = None
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
            slot = self._owned_slot_runtime
            if slot is not None:
                try:
                    await slot.aclose()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._owned_slot_runtime = None
                    ports = self._owned_slot_ports
                    self._owned_slot_ports = None
                    if ports is not None:
                        self._local_endpoints.release(ports)
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

            def delete_retained() -> None:
                retained = set(steps_to_keep) | set(active_steps)
                with optimizer_retention_lease(output_dir, retained) as protected:
                    delete_checkpoints(output_dir, sorted(protected))

            await asyncio.to_thread(delete_retained)

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
        if (pipeline_step := _PIPELINE_TRAIN_STEP.get()) is not None:
            return pipeline_step
        if self._training_binding is not None:
            client = self._training_client
            if client is not None:
                return client.projected_learner_version
            return self._training_binding.config.source.policy_step
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
