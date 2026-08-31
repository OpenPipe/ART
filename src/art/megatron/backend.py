import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
import secrets
import sys
from typing import Any, AsyncIterator, Iterable, Literal, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .. import dev, types
from ..backend import AnyTrainableModel
from ..distributed.art_runtime import ArtRuntime
from ..local.backend import LocalBackend, _PackedTrainingBatch
from ..local.service import ModelService
from ..model import Model, TrainableModel
from ..preprocessing.pack import DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult
from ..utils.lifecycle import complete_task
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from ..vllm_route_transport import RouteBundleReader, unique_retained_route_bundles
from ..vllm_runtime import get_external_vllm_runtime_config
from .migrations import apply_megatron_migrations
from .runtime.specs import ResidentLoraInspectionResult, ResidentScoreResult
from .runtime_config import get_megatron_runtime_config

_CONTEXT_PARALLEL_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH = 256
_PIPELINE_TRAIN_DISPATCH: ContextVar[asyncio.Event | None] = ContextVar(
    "megatron_pipeline_train_dispatch", default=None
)
_PIPELINE_TRAIN_EXECUTION_GATE: ContextVar[asyncio.Future[bool] | None] = ContextVar(
    "megatron_pipeline_train_execution_gate", default=None
)
_PIPELINE_TRAIN_STEP: ContextVar[int | None] = ContextVar(
    "megatron_pipeline_train_step", default=None
)
_COMMITTED_LEARNER_STEP_METRIC = "_art/committed_learner_step"


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


def _consume_pipeline_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


class _MegatronPipelineCommandContext:
    """One packed batch whose existing service transaction is queued ahead."""

    def __init__(
        self,
        backend: "MegatronBackend",
        model: TrainableModel,
        groups: tuple[TrajectoryGroup, ...],
        *,
        learner_parent_version: int,
        train_kwargs: dict[str, Any],
        preparation_metrics: dict[str, float],
    ) -> None:
        self.preparation_metrics = preparation_metrics
        self._backend = backend
        self._model = model
        self._groups = groups
        self._learner_parent_version = learner_parent_version
        self._train_kwargs = train_kwargs
        self._execution_allowed = asyncio.get_running_loop().create_future()
        self._dispatched = asyncio.Event()
        self._claimed = False
        self._task = asyncio.create_task(
            self._run(),
            name=f"megatron-pipeline-train-{learner_parent_version + 1}",
        )
        self._task.add_done_callback(_consume_pipeline_task_exception)

    async def _run(self) -> LocalTrainResult:
        result = await self._backend.train(
            self._model,
            self._groups,
            _pipeline_train_execution_gate=self._execution_allowed,
            _pipeline_train_dispatch_event=self._dispatched,
            **self._train_kwargs,
        )
        expected = self._learner_parent_version + 1
        if result.step != expected:
            raise RuntimeError(
                "ready-ahead trainer advanced another learner: "
                f"expected={expected}, actual={result.step}"
            )
        return result

    async def complete(
        self, next_train_dispatched: asyncio.Event | None
    ) -> LocalTrainResult:
        if self._claimed:
            raise RuntimeError("pipeline train context was consumed twice")
        self._claimed = True
        self._execution_allowed.set_result(True)

        relay: asyncio.Task[None] | None = None
        if next_train_dispatched is not None:

            async def relay_dispatch() -> None:
                await self._dispatched.wait()
                next_train_dispatched.set()

            relay = asyncio.create_task(
                relay_dispatch(),
                name=f"megatron-pipeline-dispatch-{self._learner_parent_version + 1}",
            )
        try:
            try:
                return await self._task
            except BaseException:
                await self._discard_unconsumed()
                raise
        finally:
            if next_train_dispatched is not None and self._dispatched.is_set():
                next_train_dispatched.set()
            if relay is not None and not relay.done():
                relay.cancel()
                await asyncio.gather(relay, return_exceptions=True)

    async def abort(self) -> None:
        if self._claimed:
            return
        self._claimed = True
        if self._task.done() and not self._task.cancelled():
            error = self._task.exception()
            if error is None:
                raise RuntimeError("cannot discard a completed pipeline train")
        elif not self._task.done():
            self._execution_allowed.set_result(False)
            await asyncio.gather(self._task, return_exceptions=True)
        await self._discard_unconsumed()

    async def _discard_unconsumed(self) -> None:
        prepared = self._groups[0]._prepared_training_batch
        if isinstance(prepared, _PipelinePreparedBatch) and all(
            group._prepared_training_batch is prepared for group in self._groups
        ):
            await self._backend.discard_pipeline_batch(list(self._groups))


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
        dispatch_event = kwargs.pop("_pipeline_train_dispatch_event", None)
        if dispatch_event is not None and not isinstance(dispatch_event, asyncio.Event):
            raise TypeError("pipeline train dispatch fence must be an asyncio.Event")
        execution_gate = kwargs.pop("_pipeline_train_execution_gate", None)
        if execution_gate is not None and not isinstance(
            execution_gate, asyncio.Future
        ):
            raise TypeError("pipeline train execution gate must be an asyncio.Future")
        groups = list(trajectory_groups)
        pipeline_call = bool(
            groups
            and isinstance(groups[0]._prepared_training_batch, _PipelinePreparedBatch)
        )
        from .distributed_service import DistributedMegatronService

        if dispatch_event is not None and dispatch_event.is_set():
            raise RuntimeError("pipeline train dispatch fence is already set")
        dispatch_token = _PIPELINE_TRAIN_DISPATCH.set(dispatch_event)
        execution_token = _PIPELINE_TRAIN_EXECUTION_GATE.set(execution_gate)
        step_token = _PIPELINE_TRAIN_STEP.set(None)
        try:
            if (dispatch_event is not None or execution_gate is not None) and not (
                pipeline_call
            ):
                raise RuntimeError("trainer dispatch controls require a prepared batch")
            result = await super().train(
                model,
                groups,
                packed_sequence_length=(
                    get_megatron_runtime_config().packed_sequence_length
                ),
                **kwargs,
            )
        finally:
            _PIPELINE_TRAIN_STEP.reset(step_token)
            _PIPELINE_TRAIN_EXECUTION_GATE.reset(execution_token)
            _PIPELINE_TRAIN_DISPATCH.reset(dispatch_token)
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
        from ..distributed.packing import PackingRequest
        from ..distributed.rollout import (
            DistributedTrajectorySelection,
            RolloutModelSpec,
        )
        from ..distributed.trajectory_store import (
            TrajectoryGroupBundle,
            retained_route_bundles_from_bundles,
        )

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
            bundles = tuple(
                TrajectoryGroupBundle.from_group(group)
                for group in (local_groups if selected else tuple(trajectory_groups))
            )
            sources = (
                ()
                if local_selections
                else tuple(selection.lease.item for selection in selected)
            )
            retained_routes = (
                unique_retained_route_bundles(
                    (
                        ref
                        for source in sources
                        for ref in source.ref.descriptor.retained_route_bundles
                    ),
                )
                if sources
                else retained_route_bundles_from_bundles(bundles)
            )
            request = PackingRequest(
                model=RolloutModelSpec.from_model(model),
                generation_id=generation_id,
                trajectory_groups=bundles,
                trajectory_sources=sources,
                retained_route_bundles=retained_routes,
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
            "time/step_trajectory_fetch_s": distributed.trajectory_fetch_s,
            "time/step_route_fetch_s": distributed.route_fetch_s,
            "time/step_packing_core_s": distributed.packing_core_s,
            "time/step_trajectory_log_wait_s": distributed.trajectory_log_wait_s,
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
    ) -> _MegatronPipelineCommandContext | None:
        metrics = await self.prepare_pipeline_batch(
            model,
            trajectory_groups,
            normalize_advantages=bool(train_kwargs.get("normalize_advantages", True)),
            advantage_balance=float(train_kwargs.get("advantage_balance", 0.0)),
            scale_rewards=bool(train_kwargs.get("scale_rewards", True)),
            allow_training_without_logprobs=bool(
                train_kwargs.get("allow_training_without_logprobs", False)
            ),
            plot_tensors=bool(train_kwargs.get("plot_tensors", False)),
            logprob_calculation_chunk_size=int(
                train_kwargs.get("logprob_calculation_chunk_size", 1024)
            ),
            grad_accumulation_sequences=cast(
                int | None, train_kwargs.get("grad_accumulation_sequences")
            ),
        )
        if metrics is None:
            return None
        return _MegatronPipelineCommandContext(
            self,
            model,
            tuple(trajectory_groups),
            learner_parent_version=learner_parent_version,
            train_kwargs=dict(train_kwargs),
            preparation_metrics=metrics,
        )

    async def discard_pipeline_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> None:
        prepared = trajectory_groups[0]._prepared_training_batch
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
        if payload.selections:
            ref = distributed_batch.leases.ref
            expected_groups = tuple(
                selection.lease.item.ref.result_id for selection in payload.selections
            )
            expected_records = tuple(
                record.record_id
                for selection in payload.selections
                for record in selection.lease.item.ref.records
            )
            versions = []
            for selection in payload.selections:
                item = selection.lease.item
                descriptor = item.ref.descriptor
                versions.extend(
                    version
                    for initial, final in zip(
                        descriptor.trajectory_initial_policy_versions,
                        descriptor.trajectory_final_policy_versions,
                        strict=True,
                    )
                    for version in (
                        initial
                        if initial is not None
                        else item.annotations.initial_policy_version,
                        final
                        if final is not None
                        else item.annotations.final_policy_version,
                    )
                )
            if (
                distributed_batch.packing_generation_id != payload.generation_id
                or ref.group_ids != expected_groups
                or ref.record_ids != expected_records
                or ref.min_source_version != min(versions)
                or ref.max_source_version != max(versions)
            ):
                raise RuntimeError("packed batch policy provenance does not match")
        distributed_service = cast(DistributedMegatronService, service)
        async for result in distributed_service.train_packed(
            distributed_batch,
            config,
            service_dev_config,
            dispatch_event=_PIPELINE_TRAIN_DISPATCH.get(),
            execution_gate=_PIPELINE_TRAIN_EXECUTION_GATE.get(),
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
