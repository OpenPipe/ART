from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import os
import socket
from threading import BoundedSemaphore, Event, Lock, Thread
import time
import traceback
from typing import Any, Callable

import monarch.actor as monarch_actor
from monarch.actor import Actor, Channel, MeshFailure, Port, ProcMesh, endpoint
from monarch.spmd import SPMDActor
from pydantic import BaseModel, ConfigDict

from art.distributed.data_plane import PackedBatchLeaseSet
from art.distributed.monarch_bootstrap import activate_cuda_device
from art.distributed.specs import GpuId
from art.megatron.weights.external_lora_publish import (
    ExternalLoraPlan,
    ExternalLoraPublication,
    ExternalLoraSinkSpec,
    ExternalLoraTarget,
)
from art.training import (
    CommandExecutionUsage,
    OperationExecutionError,
    OperationRef,
    TokenLogprobs,
    UsageMeasurement,
)
from art.utils.cache_dirs import configure_model_cache_env
from art.utils.lifecycle import (
    cleanup_after_failure,
    consume_future_exception,
    process_shutdown_timeout,
)

from .data_plane import InMemoryPackedBatch, SFTBatchData
from .numerical_capture import (
    ForwardBackwardNumericalCaptureReceipt,
    ForwardBackwardNumericalRankReceipt,
    build_forward_backward_capture,
)
from .portable_snapshot import (
    PortableSnapshotArchive,
    PortableSnapshotExportReceipt,
    PortableSnapshotGeneration,
    PortableSnapshotInstallReceipt,
    PortableSnapshotLoadReceipt,
    PortableSnapshotRankReceipt,
    PortableSnapshotReadReceipt,
    PortableSnapshotTensorOwner,
    build_portable_snapshot_archive,
    portable_snapshot_sink_from_local_runtime,
    portable_snapshot_source_from_local_runtime,
)
from .publication import (
    TRAINER_PUBLICATION_EVENT_ADAPTER,
    TrainerPublicationEvent,
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .specs import (
    TRAIN_EVENT_ADAPTER,
    AdapterReady,
    CommandPublicationSpec,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    HybridEpRuntimeSpec,
    OptimizerJobSpec,
    ResidentLoraExport,
    ResidentLoraInspectionResult,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentLoraRankSummary,
    ResidentScoreJobSpec,
    ResidentScoreResult,
    ResidentScoreShard,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    SFTJobSpec,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerCommandRunState,
    TrainerGeneration,
    TrainerJobSpec,
    TrainerRuntimeSpec,
    TrainEvent,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)

_MAX_CACHED_COMMAND_OPERATIONS = 65
_MAX_DEFERRED_COMMAND_RESULTS = 8
_DEFERRED_RESULT_JOIN_TIMEOUT_S = process_shutdown_timeout(2)
type _DeferredCommandJob = (
    ForwardJobSpec
    | ForwardBackwardJobSpec
    | SftForwardJobSpec
    | SftForwardBackwardJobSpec
)


class _CommandReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    operation_id: str
    learner_version: int
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


def _response_exception(error: BaseException) -> Exception:
    if isinstance(error, Exception):
        return error
    return RuntimeError(f"{type(error).__name__}: {error}")


def _rank_command_failure(rank: int, error: BaseException) -> dict[str, Any]:
    source = getattr(error, "source", error)
    gpu_service_ns = getattr(error, "gpu_service_ns", None)
    message = str(source).strip() or type(source).__name__
    return {
        "command_status": "failed",
        "rank": rank,
        "error_type": type(source).__name__,
        "message": message[:2048],
        "cancelled": type(source).__name__
        in {
            "CancelledError",
            "TrainingCancelledError",
        },
        "gpu_service_ns": (int(gpu_service_ns) if gpu_service_ns is not None else None),
    }


class _ActorEventSink:
    def __init__(self, port: Port[dict[str, Any]], *, coordinator: bool) -> None:
        self._port = port
        self._coordinator = coordinator

    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None:
        if self._coordinator:
            self._port.send(
                {
                    "kind": "progress",
                    "step_index": step_index,
                    "num_steps": num_steps,
                    "metrics": metrics,
                }
            )

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None:
        if self._coordinator:
            self._port.send(
                {
                    "kind": "adapter_ready",
                    "learner_version": learner_version,
                    "adapter_path": adapter_path,
                }
            )

    def publication(self, event: TrainerPublicationEvent) -> None:
        self._port.send(event.model_dump(mode="json"))


_SUPERVISION_LOCK = Lock()
_SUPERVISION_HANDLERS: dict[str, "MonarchTrainerSupervision"] = {}
_SUPERVISION_MESHES: dict[str, "MonarchTrainerSupervision"] = {}
_PREVIOUS_FAULT_HOOK: Callable[[MeshFailure], None] | None = None


def _configure_hybrid_ep_env(
    spec: HybridEpRuntimeSpec, *, run_id: str | None = None
) -> None:
    os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(
        spec.ranks_per_nvlink_domain
    )
    transport = spec.nixl_transport
    metadata_store = transport.metadata_store if transport is not None else None
    nixl_paths = None
    if transport is not None:
        if metadata_store is None:
            raise RuntimeError(
                "NIXL metadata store was not resolved before trainer launch"
            )
        from art.distributed.nixl_runtime import configure_nixl_environment

        nixl_paths = configure_nixl_environment()
    values = {
        "HYBRID_EP_MULTINODE": "1" if transport else None,
        "USE_NIXL": "1" if transport else None,
        "DEEPEP_NIXL_RUN_ID": (run_id or spec.run_id) if transport else None,
        "NIXL_ETCD_ENDPOINTS": metadata_store.url if metadata_store else None,
        "NIXL_HOME": transport.nixl_home if transport else None,
        "UCX_HOME": transport.ucx_home if transport else None,
        "NIXL_PLUGIN_DIR": (
            transport.nixl_plugin_dir or str(nixl_paths.plugin_dir)
            if transport and nixl_paths
            else None
        ),
        "UCX_MODULE_DIR": (
            transport.ucx_module_dir or str(nixl_paths.ucx_module_dir)
            if transport and nixl_paths
            else None
        ),
        "UCX_NET_DEVICES": transport.ucx_net_devices if transport else None,
        "UCX_TLS": transport.ucx_tls if transport else None,
        "UCX_IB_GDA_RETAIN_INACTIVE_CTX": "yes" if transport else None,
        "UCX_CUDA_COPY_ENABLE_FABRIC": (
            "yes" if transport and transport.enable_cuda_fabric else "no"
        )
        if transport
        else None,
    }
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _build_training_runtime(
    spec: TrainerRuntimeSpec, *, rank: int, distributed_timeout_s: float
) -> Any:
    import torch

    from art.megatron.train import build_training_runtime

    if distributed_timeout_s <= 0:
        raise ValueError("distributed timeout must be positive")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl", timeout=timedelta(seconds=distributed_timeout_s)
        )

    return build_training_runtime(
        model_identifier=spec.model_source,
        model_initialization=spec.model_initialization,
        provider_torch_dtype={
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[spec.dtype],
        print_env=rank == 0,
        model_support_key=spec.model_support_key,
        snapshot_pool_capacity=spec.snapshot_pool_capacity,
    )


class _TrainerRankReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    host_id: str
    gpu_id: GpuId
    hostname: str
    process_id: int


class _CpLookaheadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    batch_id: str
    planned_sequences: int = 0
    elapsed_s: float = 0.0
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


def _dispatch_trainer_fault(failure: MeshFailure) -> None:
    message = str(failure)
    with _SUPERVISION_LOCK:
        owner = _SUPERVISION_MESHES.get(failure.mesh_name)
        handlers = (
            (owner,)
            if owner is not None
            else tuple(
                handler
                for token, handler in _SUPERVISION_HANDLERS.items()
                if token in message
            )
        )
        previous = _PREVIOUS_FAULT_HOOK
    if handlers:
        for handler in handlers:
            handler.notify(message)
        return
    if previous is not None:
        previous(failure)


class MonarchTrainerSupervision:
    """Route one owned trainer mesh failure without masking unrelated faults."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.token = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        self._loop = asyncio.get_running_loop()
        self._failure: asyncio.Future[str] = self._loop.create_future()
        self._mesh_names: set[str] = set()
        self._closed = False
        global _PREVIOUS_FAULT_HOOK
        with _SUPERVISION_LOCK:
            if self.token in _SUPERVISION_HANDLERS:
                raise RuntimeError(f"trainer run {run_id!r} is already supervised")
            if not _SUPERVISION_HANDLERS:
                _PREVIOUS_FAULT_HOOK = monarch_actor.unhandled_fault_hook
                setattr(
                    monarch_actor,
                    "unhandled_fault_hook",
                    _dispatch_trainer_fault,
                )
            _SUPERVISION_HANDLERS[self.token] = self

    def own_mesh(self, mesh_name: str) -> None:
        if not mesh_name:
            raise ValueError("trainer mesh name must not be empty")
        with _SUPERVISION_LOCK:
            if self._closed:
                raise RuntimeError(f"trainer run {self.run_id!r} is closed")
            owner = _SUPERVISION_MESHES.get(mesh_name)
            if owner is not None and owner is not self:
                raise RuntimeError(f"Monarch mesh {mesh_name!r} already has an owner")
            self._mesh_names.add(mesh_name)
            _SUPERVISION_MESHES[mesh_name] = self

    def notify(self, failure: str) -> None:
        def set_failure() -> None:
            if not self._failure.done():
                self._failure.set_result(failure)

        self._loop.call_soon_threadsafe(set_failure)

    async def wait(self) -> str:
        return await asyncio.shield(self._failure)

    def close(self) -> None:
        global _PREVIOUS_FAULT_HOOK
        with _SUPERVISION_LOCK:
            if self._closed:
                return
            self._closed = True
            if _SUPERVISION_HANDLERS.get(self.token) is self:
                _SUPERVISION_HANDLERS.pop(self.token)
            for mesh_name in self._mesh_names:
                if _SUPERVISION_MESHES.get(mesh_name) is self:
                    _SUPERVISION_MESHES.pop(mesh_name)
            if not _SUPERVISION_HANDLERS:
                if monarch_actor.unhandled_fault_hook is _dispatch_trainer_fault:
                    assert _PREVIOUS_FAULT_HOOK is not None
                    setattr(
                        monarch_actor,
                        "unhandled_fault_hook",
                        _PREVIOUS_FAULT_HOOK,
                    )
                _PREVIOUS_FAULT_HOOK = None


class _TrainerSPMDActor(SPMDActor):
    """Own the rendezvous store until the warm trainer mesh is stopped."""

    def __init__(self) -> None:
        super().__init__()
        self._store: Any = None

    @endpoint
    def start_store(self, _request: None) -> tuple[str, int]:
        if self._store is not None:
            raise RuntimeError("trainer rendezvous store is already running")
        from torch.distributed import TCPStore

        hostname = socket.gethostname()
        self._store = TCPStore(
            hostname,
            0,
            self.world_size,
            True,
            wait_for_workers=False,
        )
        return hostname, int(self._store.port)

    @endpoint
    def setup_agent_store_env(self, master_addr: str, master_port: int) -> None:
        self._setup_env(master_addr, master_port)
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "True"

    def __cleanup__(self, exc: Exception | None) -> None:
        del exc
        self._store = None


class MonarchTrainerActor(Actor):
    """One warm Megatron rank, spawned once on every trainer ProcMesh process."""

    def __init__(
        self,
        runtime_spec_json: str,
        run_id: str,
        distributed_timeout_s: float,
    ) -> None:
        runtime_spec = TrainerRuntimeSpec.model_validate_json(runtime_spec_json)
        topology = runtime_spec.trainer_mesh.topology
        cache_root = configure_model_cache_env(cache_root=runtime_spec.cache_root)
        os.environ.update(
            {
                "MODEL_IDENTIFIER": runtime_spec.model_source,
                "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE": str(topology.tp),
                "ART_MEGATRON_CONTEXT_PARALLEL_SIZE": str(topology.cp),
                "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE": str(topology.ep),
                "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE": str(topology.pp),
                "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE": str(topology.etp),
                "ART_MEGATRON_LORA_RANK": str(runtime_spec.lora_rank),
                "ART_MEGATRON_LORA_TARGET_MODULES": json.dumps(
                    runtime_spec.lora_target_modules
                ),
                "ART_DISABLE_MEGATRON_COMPILE": (
                    "0" if runtime_spec.compile_enabled else "1"
                ),
                "ART_MEGATRON_ALLOW_UNVALIDATED_ARCH": str(
                    int(runtime_spec.allow_unvalidated_arch)
                ),
                "ART_MEGATRON_ENABLE_MOE_ROUTING_REPLAY": str(
                    int(runtime_spec.enable_moe_routing_replay)
                ),
                "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD": str(
                    int(runtime_spec.streaming_weight_offload)
                ),
                "ART_MEGATRON_OFFLOAD_BETWEEN_JOBS": str(
                    int(runtime_spec.offload_between_jobs)
                ),
            }
        )
        if runtime_spec.random_state is not None:
            os.environ["ART_MEGATRON_RANDOM_STATE"] = str(runtime_spec.random_state)
        if topology.vpp is not None:
            os.environ["ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE"] = str(
                topology.vpp
            )
        if topology.vpp_microbatch_group_size is not None:
            os.environ["ART_MEGATRON_VPP_MICROBATCH_GROUP_SIZE"] = str(
                topology.vpp_microbatch_group_size
            )
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != len(runtime_spec.trainer_mesh.ranks):
            raise RuntimeError(
                "Monarch ProcMesh world does not match TrainerRuntimeSpec: "
                f"{world_size} != {len(runtime_spec.trainer_mesh.ranks)}"
            )

        rank = int(os.environ["RANK"])
        placement = runtime_spec.trainer_mesh.ranks[rank]
        self._host_id = placement.host_id
        self._gpu_id = placement.gpu_id
        local_rank = activate_cuda_device(placement.gpu_id)
        os.environ["LOCAL_RANK"] = str(local_rank)

        import torch

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.cuda.set_device(local_rank)
        self._compile_cache = None
        self._compile_cache_metrics: dict[str, float] = {}
        if runtime_spec.compile_cache:
            from .compile_cache import TrainerCompileCache

            self._compile_cache = TrainerCompileCache(
                runtime_spec, rank=rank, cache_root=cache_root
            )
            event = self._compile_cache.load()
            self._compile_cache_metrics.update(
                {
                    "hit": float(event.status == "hit"),
                    "load_s": event.elapsed_s,
                    "artifact_bytes": float(event.artifact_bytes),
                }
            )
        if topology.ep > 1:
            from art.megatron.hybrid_ep_setup import validate_hybrid_ep

            hybrid_ep = runtime_spec.hybrid_ep
            if hybrid_ep is None:
                raise RuntimeError(
                    "expert parallelism requires a HybridEP runtime spec"
                )
            group_index = rank // (topology.etp * topology.ep)
            _configure_hybrid_ep_env(
                hybrid_ep,
                run_id=f"{hybrid_ep.run_id}-{run_id}-g{group_index}",
            )
            validate_hybrid_ep(require_multinode=hybrid_ep.multinode)
        self._runtime = _build_training_runtime(
            runtime_spec,
            rank=rank,
            distributed_timeout_s=distributed_timeout_s,
        )
        self._runtime.resident_run_id = run_id
        if self._runtime.model_support_handler.key != runtime_spec.handler_name:
            raise RuntimeError(
                "resolved model-support handler does not match TrainerRuntimeSpec: "
                f"{self._runtime.model_support_handler.key!r} != "
                f"{runtime_spec.handler_name!r}"
            )
        from art.megatron.training.streaming_weight_offload import (
            streaming_weight_offload_config_from_env,
        )
        from art.megatron.training.weight_offload import WeightOffloadManager

        from .executor import MCoreRunSlotExecutor, MegatronTrainJobExecutor

        self._executor = MegatronTrainJobExecutor(
            self._runtime,
            accumulator_l1_budget_bytes=runtime_spec.accumulator_l1_budget_bytes,
        )
        self._run_slot_executor = MCoreRunSlotExecutor(
            self._runtime,
            accumulator_l1_budget_bytes=runtime_spec.accumulator_l1_budget_bytes,
            portable_snapshot_source=portable_snapshot_source_from_local_runtime(
                run_id=run_id,
                rank=rank,
            ),
            portable_snapshot_sink=portable_snapshot_sink_from_local_runtime(
                run_id=run_id,
                rank=rank,
            ),
            publisher=self._executor.publisher,
        )
        self._weight_offload = WeightOffloadManager.from_config(
            model=self._runtime.model,
            rank=self._runtime.rank,
            compile_enabled=self._runtime.transformer_layers_compiled,
            offload_between_jobs=runtime_spec.offload_between_jobs,
            streaming_config=streaming_weight_offload_config_from_env(),
        )
        self._weight_offload.install()
        self._command_job_open = False
        self._deferred_result_lock = Lock()
        self._deferred_result_slots = BoundedSemaphore(_MAX_DEFERRED_COMMAND_RESULTS)
        self._deferred_result_threads: set[Thread] = set()
        self._deferred_result_stopping = False
        self._migration_release_receipts: dict[str, tuple[str, ...]] = {}
        self._cp_preplanner = None
        self._cp_lookahead_port = None
        self._cp_lookahead_thread = None
        if topology.cp > 1:
            from art.megatron.training.microbatches import CpBatchPreplanner

            self._cp_preplanner = CpBatchPreplanner.from_runtime(
                self._runtime,
                device=torch.device("cuda", local_rank),
            )
            if self._cp_preplanner is None:
                raise RuntimeError("CP trainer did not create a batch preplanner")
            self._cp_lookahead_port, receiver = Channel.open()
            self._cp_lookahead_thread = Thread(
                target=self._run_cp_lookahead,
                args=(receiver,),
                name=f"art-cp-lookahead-rank-{rank}",
                daemon=True,
            )
            self._cp_lookahead_thread.start()
        self._valid = True

    def _defer_result(
        self,
        response_port: Port[dict[str, Any]],
        materialize: Callable[[], dict[str, Any]],
        *,
        name: str,
    ) -> None:
        self._deferred_result_slots.acquire()
        with self._deferred_result_lock:
            if self._deferred_result_stopping:
                self._deferred_result_slots.release()
                raise RuntimeError("trainer actor is stopping")

            def settle() -> None:
                try:
                    response_port.send(materialize())
                except BaseException as error:
                    self._valid = False
                    try:
                        response_port.exception(_response_exception(error))
                    except BaseException:
                        pass
                finally:
                    with self._deferred_result_lock:
                        self._deferred_result_threads.discard(thread)
                    self._deferred_result_slots.release()

            thread = Thread(target=settle, name=name, daemon=True)
            self._deferred_result_threads.add(thread)
            thread.start()

    def _stop_deferred_results(self) -> None:
        with self._deferred_result_lock:
            self._deferred_result_stopping = True
            threads = tuple(self._deferred_result_threads)
        deadline = time.monotonic() + _DEFERRED_RESULT_JOIN_TIMEOUT_S
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError(
                "trainer result settlement did not stop within "
                f"{_DEFERRED_RESULT_JOIN_TIMEOUT_S:g}s: {alive}"
            )

    def _defer_command_launch(
        self,
        response_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
        job: _DeferredCommandJob,
        launch: Any,
        *,
        label: str,
    ) -> None:
        ready_port.send(
            _CommandReady(
                rank=self._runtime.rank,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            ).model_dump(mode="json")
        )

        def materialize() -> dict[str, Any]:
            result = launch.materialize()
            coordinator = self._runtime.rank == 0
            return {
                **result,
                "command_status": "succeeded",
                "rank": self._runtime.rank,
                "metrics": result["metrics"] if coordinator else {},
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }

        self._defer_result(
            response_port,
            materialize,
            name=f"art-{label}-result-{job.operation_id}",
        )

    def _fail_command_launch(
        self,
        response_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
        job: _DeferredCommandJob | None,
        error: BaseException,
    ) -> None:
        if job is None:
            response_port.exception(_response_exception(error))
            return
        ready_port.send(
            _CommandReady(
                rank=self._runtime.rank,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
                error_type=type(error).__name__,
                message=str(error),
                traceback_text=traceback.format_exc(),
            ).model_dump(mode="json")
        )
        response_port.send(_rank_command_failure(self._runtime.rank, error))

    def _run_cp_lookahead(self, receiver: Any) -> None:
        while (request := receiver.recv().get()) is not None:
            batch_json, batch_id, accumulation, reply = request
            batch = None
            started = time.perf_counter()
            try:
                leases = PackedBatchLeaseSet.model_validate_json(batch_json)
                if leases.ref.batch_id != batch_id:
                    raise RuntimeError("CP lookahead request batch ID mismatch")
                if self._cp_preplanner is None:
                    raise RuntimeError("CP lookahead preplanner is unavailable")
                batch = InMemoryPackedBatch.open(
                    leases.ref, leases.host_refs[self._host_id]
                )
                planned = self._cp_preplanner.preplan(
                    batch.tensors,
                    global_grad_accumulation_sequences=accumulation,
                )
                result = _CpLookaheadResult(
                    rank=self._runtime.rank,
                    batch_id=batch_id,
                    planned_sequences=planned,
                    elapsed_s=time.perf_counter() - started,
                )
            except BaseException as error:
                result = _CpLookaheadResult(
                    rank=self._runtime.rank,
                    batch_id=batch_id,
                    elapsed_s=time.perf_counter() - started,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
            finally:
                if batch is not None:
                    batch.close()
            reply.send(result.model_dump(mode="json"))

    def _stop_cp_lookahead(self) -> None:
        thread, self._cp_lookahead_thread = self._cp_lookahead_thread, None
        if thread is None:
            return
        port = self._cp_lookahead_port
        if port is None:
            raise RuntimeError("CP lookahead thread has no request port")
        port.send(None)
        thread.join(timeout=30.0)
        if thread.is_alive():
            raise RuntimeError("CP lookahead service did not stop within 30 seconds")

    def _publish_compile_cache(self) -> None:
        if self._compile_cache is None or "publish_s" in self._compile_cache_metrics:
            return
        event = self._compile_cache.publish()
        self._compile_cache_metrics.update(
            {
                "publish_s": event.elapsed_s,
                "published": float(event.status == "published"),
                "artifact_bytes": float(event.artifact_bytes),
            }
        )

    @endpoint
    def ready(self) -> dict[str, Any]:
        return _TrainerRankReady(
            rank=self._runtime.rank,
            host_id=self._host_id,
            gpu_id=self._gpu_id,
            hostname=socket.gethostname(),
            process_id=os.getpid(),
        ).model_dump(mode="json")

    @endpoint
    def cp_lookahead_port(self) -> dict[str, Any] | None:
        if self._cp_lookahead_port is None:
            return None
        return {"rank": self._runtime.rank, "port": self._cp_lookahead_port}

    @endpoint
    def register_command_run(self, run_spec_json: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        run_spec = TrainingRunSpec.model_validate_json(run_spec_json)
        opened_job = False
        try:
            if not self._command_job_open:
                self._weight_offload.before_job()
                opened_job = True
            portable_read = self._run_slot_executor.register_run(run_spec)
            return {
                "rank": self._runtime.rank,
                "run_id": run_spec.run_id,
                "lora_rank": run_spec.lora_rank,
                "lora_target_modules": run_spec.lora_target_modules,
                "portable_read": (
                    None
                    if portable_read is None
                    else portable_read.model_dump(mode="json")
                ),
            }
        finally:
            if opened_job:
                self._weight_offload.after_job()

    @endpoint
    def export_command_run_checkpoint(
        self,
        run_id: str,
        generation_json: str,
        export_id: str,
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        generation = TrainerGeneration.model_validate_json(generation_json)
        receipt = self._run_slot_executor.export_run_checkpoint(
            run_id, generation, export_id
        )
        return {
            "rank": self._runtime.rank,
            "run_id": run_id,
            "receipt": None if receipt is None else receipt.model_dump(mode="json"),
            "tensor_owners": self._run_slot_executor.run_tensor_owners(run_id),
        }

    @endpoint
    def install_command_run_checkpoint(
        self,
        run_id: str,
        generation_json: str,
        archive_json: str,
        restore_optimizer: bool,
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        generation = TrainerGeneration.model_validate_json(generation_json)
        archive = PortableSnapshotArchive.model_validate_json(archive_json)
        receipt = self._run_slot_executor.install_run_checkpoint(
            run_id,
            generation,
            archive,
            restore_optimizer=restore_optimizer,
        )
        return {
            "rank": self._runtime.rank,
            "run_id": run_id,
            "receipt": receipt.model_dump(mode="json"),
        }

    @endpoint
    def execute(
        self,
        job_json: str,
        batch_json: str,
        event_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = TrainJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            coordinator = self._runtime.rank == 0
            with self._weight_offload.job():
                metrics = self._executor.execute(
                    job,
                    batch,
                    _ActorEventSink(event_port, coordinator=coordinator),
                    Event(),
                )
                self._publish_compile_cache()
            if coordinator:
                event_port.send({"kind": "actor_completed", "metrics": metrics})
            return {
                "rank": self._runtime.rank,
                "learner_version": job.learner_version,
                "metrics": metrics if coordinator else {},
                "compile_cache": self._compile_cache_metrics,
            }
        except BaseException as error:
            self._valid = False
            event_port.send(
                {
                    "kind": "rank_failed",
                    "rank": self._runtime.rank,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def execute_forward_backward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardBackwardJobSpec.model_validate_json(job_json)
            self._migration_release_receipts.pop(job.run_id, None)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            execute_job = (
                job
                if self._runtime.rank == 0
                else job.model_copy(update={"return_token_logprobs": False})
            )
            launch = self._run_slot_executor.execute_forward_backward(
                execute_job, batch, Event()
            )
            batch.close()
            batch = None
            self._defer_command_launch(
                response_port, ready_port, job, launch, label="fb"
            )
        except BaseException as error:
            self._valid = False
            try:
                self._abort_command_job()
            except BaseException as cleanup_error:
                error.add_note(
                    "F/B actor cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            self._fail_command_launch(response_port, ready_port, job, error)
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def execute_sft_forward_backward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch: SFTBatchData,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = SftForwardBackwardJobSpec.model_validate_json(job_json)
            self._migration_release_receipts.pop(job.run_id, None)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            execute_job = (
                job
                if self._runtime.rank == 0
                else job.model_copy(update={"return_token_logprobs": False})
            )
            launch = self._run_slot_executor.execute_sft_forward_backward(
                execute_job, batch, Event()
            )
            self._defer_command_launch(
                response_port, ready_port, job, launch, label="sft-fb"
            )
        except BaseException as error:
            self._valid = False
            try:
                self._abort_command_job()
            except BaseException as cleanup_error:
                error.add_note(
                    "SFT F/B actor cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            self._fail_command_launch(response_port, ready_port, job, error)

    @endpoint(explicit_response_port=True)
    def execute_sft_forward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch: SFTBatchData,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        opened_job = False
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = SftForwardJobSpec.model_validate_json(job_json)
            self._migration_release_receipts.pop(job.run_id, None)
            if not self._command_job_open:
                self._weight_offload.before_job()
                opened_job = True
            execute_job = (
                job
                if self._runtime.rank == 0
                else job.model_copy(update={"return_token_logprobs": False})
            )
            launch = self._run_slot_executor.execute_sft_forward(
                execute_job, batch, Event()
            )
            self._defer_command_launch(
                response_port, ready_port, job, launch, label="sft-forward"
            )
        except BaseException as error:
            self._valid = False
            self._fail_command_launch(response_port, ready_port, job, error)
        finally:
            if opened_job:
                self._weight_offload.after_job()

    @endpoint
    def capture_forward_backward_numerics(
        self,
        run_id: str,
        operation_id: str,
        batch_json: str,
        token_logprobs_json: tuple[str, ...],
        root: str,
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        leases = PackedBatchLeaseSet.model_validate_json(batch_json)
        batch = InMemoryPackedBatch.open(leases.ref, leases.host_refs[self._host_id])
        try:
            receipt = self._run_slot_executor.capture_forward_backward_numerics(
                run_id=run_id,
                operation_id=operation_id,
                batch=batch,
                token_logprobs=tuple(
                    TokenLogprobs.model_validate_json(value)
                    for value in token_logprobs_json
                ),
                root=root,
            )
            return receipt.model_dump(mode="json")
        finally:
            batch.close()

    @endpoint(explicit_response_port=True)
    def execute_forward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        opened_job = False
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardJobSpec.model_validate_json(job_json)
            self._migration_release_receipts.pop(job.run_id, None)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                opened_job = True
            execute_job = (
                job
                if self._runtime.rank == 0
                else job.model_copy(update={"return_token_logprobs": False})
            )
            launch = self._run_slot_executor.execute_forward(
                execute_job, batch, Event()
            )
            batch.close()
            batch = None
            self._defer_command_launch(
                response_port, ready_port, job, launch, label="forward"
            )
        except BaseException as error:
            self._valid = False
            self._fail_command_launch(response_port, ready_port, job, error)
        finally:
            if batch is not None:
                batch.close()
            if opened_job:
                self._weight_offload.after_job()

    @endpoint
    def execute_optimizer(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            if not self._command_job_open:
                raise RuntimeError("optimizer has no open F/B interval")
            job = OptimizerJobSpec.model_validate_json(job_json)
            self._migration_release_receipts.pop(job.run_id, None)
            result = self._run_slot_executor.execute_optimizer(job)
            return {
                **result,
                "command_status": "succeeded",
                "rank": self._runtime.rank,
                "metrics": result["metrics"] if self._runtime.rank == 0 else {},
            }
        except BaseException as error:
            self._valid = False
            try:
                self._run_slot_executor.discard_open_gradients()
            except BaseException as cleanup_error:
                error.add_note(
                    "optimizer actor cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            return _rank_command_failure(self._runtime.rank, error)
        finally:
            if (
                self._command_job_open
                and not self._run_slot_executor.has_open_gradients
            ):
                self._weight_offload.after_job()
                self._command_job_open = False

    @endpoint
    def publish_command_generation(self, spec_json: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        spec = CommandPublicationSpec.model_validate_json(spec_json)
        return self._run_slot_executor.publish_generation(spec)

    @endpoint
    def release_command_run_for_migration(
        self,
        run_id: str,
        expected_operation_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        prior = self._migration_release_receipts.get(run_id)
        if prior is not None:
            if prior != expected_operation_ids:
                raise RuntimeError("migration release identity changed")
            contributions = prior
        else:
            if (
                self._run_slot_executor.run_gradient_ids(run_id)
                != expected_operation_ids
            ):
                raise RuntimeError("migration release gradients changed")
            contributions = self._run_slot_executor.discard_run_gradients(run_id)
            self._run_slot_executor.release_run(run_id)
            self._migration_release_receipts[run_id] = contributions
            while len(self._migration_release_receipts) > (
                _MAX_CACHED_COMMAND_OPERATIONS
            ):
                self._migration_release_receipts.pop(
                    next(iter(self._migration_release_receipts))
                )
        if self._command_job_open and not self._run_slot_executor.has_open_gradients:
            self._weight_offload.after_job()
            self._command_job_open = False
        return {
            "rank": self._runtime.rank,
            "run_id": run_id,
            "discarded_forward_backward_operation_ids": contributions,
        }

    @endpoint
    def drain_command_run(self, run_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        if self._run_slot_executor.run_gradient_ids(run_id):
            raise RuntimeError("cannot drain a run with open gradients")
        self._run_slot_executor.release_run(run_id)
        return {"rank": self._runtime.rank, "run_id": run_id}

    @endpoint
    def execute_sft(
        self,
        job_json: str,
        batches: tuple[SFTBatchData, ...],
        event_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = SFTJobSpec.model_validate_json(job_json)
            coordinator = self._runtime.rank == 0
            with self._weight_offload.job():
                metrics = self._executor.execute_sft(
                    job,
                    batches,
                    _ActorEventSink(event_port, coordinator=coordinator),
                    Event(),
                )
                self._publish_compile_cache()
            if coordinator:
                event_port.send({"kind": "actor_completed", "metrics": metrics})
            return {
                "rank": self._runtime.rank,
                "learner_version": job.learner_version,
                "metrics": metrics if coordinator else {},
                "compile_cache": self._compile_cache_metrics,
            }
        except BaseException as error:
            self._valid = False
            event_port.send(
                {
                    "kind": "rank_failed",
                    "rank": self._runtime.rank,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            raise

    @endpoint
    def score(self, job_json: str, batch_json: str) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ResidentScoreJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            with self._weight_offload.job():
                result = self._executor.score(job, batch)
            return result.model_dump(mode="json")
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint
    def inspect_resident_lora(self, request_json: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        request = ResidentLoraInspectionSpec.model_validate_json(request_json)
        with self._weight_offload.job():
            result = self._executor.inspect_resident_lora(request)
        return result.model_dump(mode="json")

    @endpoint
    def publish_external_lora(
        self,
        target_json: str,
        sink_spec_json: str,
        source_topology: str,
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        target = ExternalLoraTarget.model_validate_json(target_json)
        sink = ExternalLoraSinkSpec.model_validate_json(sink_spec_json).create()
        submitted = False
        try:
            with self._weight_offload.job():
                plan, future, metrics = self._executor.publish_external_lora(
                    target,
                    sink,
                    source_topology=source_topology,
                )
                submitted = True
                publication = future.result()
        finally:
            if not submitted:
                close = getattr(sink, "close", None)
                if close is not None:
                    close()
        return {
            "rank": self._runtime.rank,
            "plan": plan.model_dump(mode="json"),
            "publication": (
                None if publication is None else publication.model_dump(mode="json")
            ),
            "metrics": metrics,
        }

    @endpoint
    def close(self) -> None:
        self._stop_cp_lookahead()
        self._stop_deferred_results()
        self._abort_command_job()
        self._run_slot_executor.close()
        self._executor.close()
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _abort_command_job(self) -> None:
        if not self._command_job_open:
            return
        try:
            self._executor.discard_open_gradients()
            self._run_slot_executor.discard_open_gradients()
        finally:
            self._weight_offload.after_job()
            self._command_job_open = False

    @endpoint
    def advance_without_training(
        self,
        source_json: str,
        output_json: str,
        optimizer_state_path: str,
        adapter_json: str | None,
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        from art.megatron.optimizer_state import OptimizerAdapter

        source = TrainerGeneration.model_validate_json(source_json)
        output = TrainerGeneration.model_validate_json(output_json)
        adapter = (
            None
            if adapter_json is None
            else OptimizerAdapter.model_validate_json(adapter_json)
        )
        try:
            with self._weight_offload.job():
                metrics = self._executor.advance_without_training(
                    source=source,
                    output=output,
                    optimizer_state_path=optimizer_state_path,
                    adapter=adapter,
                )
            return {
                "rank": self._runtime.rank,
                "learner_version": output.policy_step,
                "metrics": metrics,
            }
        except BaseException:
            self._valid = False
            raise

    def __cleanup__(self, exc: Exception | None) -> None:
        if exc is not None:
            self._valid = False
        self._stop_cp_lookahead()
        self._stop_deferred_results()
        self._run_slot_executor.close()
        self._executor.close()


async def spawn_monarch_trainer_actors(
    proc_mesh: ProcMesh,
    runtime_spec: TrainerRuntimeSpec,
    supervision: MonarchTrainerSupervision,
    distributed_timeout_s: float,
) -> tuple[Any, tuple[_TrainerRankReady, ...], tuple[Port[Any], ...]]:
    """Configure torch-elastic first, then initialize exactly one actor per rank."""
    spmd: Any = proc_mesh.spawn(
        f"art_torch_elastic_{supervision.token}", _TrainerSPMDActor
    )
    supervision.own_mesh(await spmd._name)
    first_rank = dict.fromkeys(proc_mesh._labels, 0)
    master_addr, master_port = await spmd.slice(**first_rank).start_store.call_one(None)
    await spmd.setup_agent_store_env.call(master_addr, master_port)
    actors: Any = proc_mesh.spawn(
        f"art_megatron_trainer_{supervision.token}",
        MonarchTrainerActor,
        runtime_spec.model_dump_json(),
        supervision.run_id,
        distributed_timeout_s,
    )
    supervision.own_mesh(await actors._name)
    await actors.initialized
    values = await actors.ready.call()
    ready = tuple(
        sorted(
            (_TrainerRankReady.model_validate(value) for value in values.values()),
            key=lambda value: value.rank,
        )
    )
    placements = runtime_spec.trainer_mesh.ranks
    if len(ready) != len(placements) or any(
        (value.rank, value.host_id, value.gpu_id)
        != (rank, placement.host_id, placement.gpu_id)
        for rank, (value, placement) in enumerate(zip(ready, placements, strict=True))
    ):
        raise RuntimeError(
            "trainer startup did not return the configured rank placement"
        )
    port_values = await actors.cp_lookahead_port.call()
    lookahead_ports = tuple(
        value["port"]
        for value in sorted(
            (value for value in port_values.values() if value is not None),
            key=lambda value: value["rank"],
        )
    )
    expected_port_count = (
        len(placements) if runtime_spec.trainer_mesh.topology.cp > 1 else 0
    )
    if len(lookahead_ports) != expected_port_count:
        raise RuntimeError("trainer ranks returned an incomplete CP lookahead service")
    return actors, ready, lookahead_ports


class _PublicationState:
    __slots__ = (
        "active_waiters",
        "drain_done",
        "future",
        "generation_id",
        "late_waitable",
        "outcome_observed",
        "records",
        "train_done",
    )

    def __init__(
        self,
        generation_id: str,
        future: asyncio.Future[tuple[TrainerRankPublication, ...]],
    ) -> None:
        self.generation_id = generation_id
        self.future = future
        self.records: dict[int, TrainerRankPublication] = {}
        self.train_done = False
        self.drain_done = True
        self.active_waiters = 0
        self.late_waitable = True
        self.outcome_observed = False


def _merge_resident_score_shards(
    shards: tuple[ResidentScoreShard, ...],
    *,
    job: ResidentScoreJobSpec,
    world_size: int,
) -> ResidentScoreResult:
    by_rank = {shard.rank: shard for shard in shards}
    expected_ranks = set(range(world_size))
    if len(by_rank) != len(shards) or set(by_rank) != expected_ranks:
        raise RuntimeError("resident score did not return exactly one shard per rank")
    ordered = tuple(by_rank[rank] for rank in range(world_size))
    first = ordered[0]
    for shard in ordered:
        if (
            shard.job_id != job.job_id
            or shard.run_id != job.run_id
            or shard.learner != job.learner
            or shard.batch_id != job.batch.batch_id
            or shard.batch_fingerprint != first.batch_fingerprint
            or shard.top_k != job.top_k
            or shard.expected_score_count != first.expected_score_count
            or shard.routing_replay_packed_tokens != first.routing_replay_packed_tokens
        ):
            raise RuntimeError("resident score rank shards disagree on provenance")
    expected_replay_tokens = (
        0
        if job.batch.moe_routing_replay is None
        else job.batch.moe_routing_replay.packed_tokens
    )
    if first.routing_replay_packed_tokens != expected_replay_tokens:
        raise RuntimeError("resident score routing replay does not match packed data")

    scores: dict[tuple[int, int], Any] = {}
    for shard in ordered:
        for score in shard.scores:
            key = score.sample_index, score.logit_index
            previous = scores.get(key)
            if previous is not None and previous != score:
                raise RuntimeError(
                    f"resident score replicas disagree at coordinate {key}"
                )
            scores[key] = score
    merged = tuple(scores[key] for key in sorted(scores))
    if len(merged) != first.expected_score_count:
        raise RuntimeError(
            "resident score did not cover every packed target: "
            f"expected={first.expected_score_count}, got={len(merged)}"
        )
    return ResidentScoreResult(
        job_id=job.job_id,
        run_id=job.run_id,
        learner=job.learner,
        batch_id=job.batch.batch_id,
        batch_fingerprint=first.batch_fingerprint,
        ranks=tuple(range(world_size)),
        top_k=job.top_k,
        expected_score_count=first.expected_score_count,
        routing_replay_packed_tokens=first.routing_replay_packed_tokens,
        scores=merged,
    )


def _merge_resident_lora_shards(
    shards: tuple[ResidentLoraInspectionShard, ...],
    *,
    request: ResidentLoraInspectionSpec,
    world_size: int,
) -> ResidentLoraInspectionResult:
    by_rank = {shard.rank: shard for shard in shards}
    expected_ranks = set(range(world_size))
    if len(by_rank) != len(shards) or set(by_rank) != expected_ranks:
        raise RuntimeError("resident LoRA inspection did not return one shard per rank")
    ordered = tuple(by_rank[rank] for rank in range(world_size))
    for shard in ordered:
        if (
            shard.request_id != request.request_id
            or shard.run_id != request.run_id
            or shard.learner != request.learner
            or shard.target_modules != request.target_modules
        ):
            raise RuntimeError("resident LoRA rank shards disagree on provenance")

    exports: dict[str, set[str | None]] = {}
    for shard in ordered:
        for export in shard.exports:
            exports.setdefault(export.base_name, set()).update(export.adapter_keys)
    return ResidentLoraInspectionResult(
        request_id=request.request_id,
        run_id=request.run_id,
        learner=request.learner,
        target_modules=request.target_modules,
        rank_summaries=tuple(
            ResidentLoraRankSummary(
                rank=shard.rank,
                module_count=shard.module_count,
                trainable_parameter_count=len(shard.trainable_lora_parameter_names),
                trainable_numel=shard.trainable_numel,
            )
            for shard in ordered
        ),
        wrapped_adapter_prefixes=tuple(
            sorted(
                {
                    prefix
                    for shard in ordered
                    for prefix in shard.wrapped_adapter_prefixes
                }
            )
        ),
        exports=tuple(
            ResidentLoraExport(
                base_name=base_name,
                adapter_keys=tuple(
                    sorted(
                        adapter_keys,
                        key=lambda value: "" if value is None else value,
                    )
                ),
            )
            for base_name, adapter_keys in sorted(exports.items())
        ),
        trainable_lora_parameter_names=tuple(
            sorted(
                {
                    name
                    for shard in ordered
                    for name in shard.trainable_lora_parameter_names
                }
            )
        ),
        unexpected_trainable_parameter_names=tuple(
            sorted(
                {
                    name
                    for shard in ordered
                    for name in shard.unexpected_trainable_parameter_names
                }
            )
        ),
    )


@dataclass(slots=True)
class _CommandRunState:
    spec: TrainingRunSpec
    learner_version: int
    next_operation_sequence: int = 0
    open_forward_backward_ids: list[str] = field(default_factory=list)
    portable_install: PortableSnapshotInstallReceipt | None = None


@dataclass(frozen=True, slots=True)
class TrainerCommandLaunch:
    """The trainer GPU turn is complete; bounded result settlement remains."""

    completion: asyncio.Future[dict[str, Any]]


def _command_run_identity(spec: TrainingRunSpec) -> tuple[Any, ...]:
    return (
        spec.run_id,
        spec.runtime_fingerprint,
        spec.training_session_id,
        spec.initial_learner_version,
        spec.initial_generation_id,
        spec.initial_operation_sequence,
        spec.lora_rank,
        spec.lora_target_modules,
        spec.initial_adapter_path,
        spec.optimizer_state_path,
        (
            None
            if spec.initial_portable_snapshot is None
            else spec.initial_portable_snapshot.archive_sha256
        ),
    )


class MonarchTrainerRun:
    def __init__(
        self,
        runtime_spec: TrainerRuntimeSpec,
        run_spec: TrainingRunSpec,
        actors: Any,
        proc_mesh: ProcMesh,
        supervision: MonarchTrainerSupervision,
        rank_processes: tuple[_TrainerRankReady, ...],
        cp_lookahead_ports: tuple[Port[Any], ...],
        *,
        register_initial_run: bool = True,
    ) -> None:
        if run_spec.runtime_fingerprint != runtime_spec.fingerprint:
            raise ValueError(
                "training run does not match the trainer runtime fingerprint"
            )
        self.runtime_spec = runtime_spec
        self.run_spec = run_spec
        self._actors = actors
        self._proc_mesh = proc_mesh
        self._supervision = supervision
        self._rank_processes = rank_processes
        self._cp_lookahead_ports = cp_lookahead_ports
        self._learner_version = run_spec.initial_learner_version
        self._command_runs = (
            {
                run_spec.run_id: _CommandRunState(
                    spec=run_spec,
                    learner_version=run_spec.initial_learner_version,
                    next_operation_sequence=run_spec.initial_operation_sequence,
                )
            }
            if register_initial_run
            else {}
        )
        self._jobs: dict[str, tuple[str, tuple[TrainEvent, ...]]] = {}
        self._operations: dict[str, tuple[str, dict[str, Any]]] = {}
        self._command_launches: dict[str, tuple[str, TrainerCommandLaunch]] = {}
        self._checkpoint_exports: dict[
            str, tuple[tuple[str, TrainerGeneration], PortableSnapshotExportReceipt]
        ] = {}
        self._checkpoint_loads: dict[str, tuple[str, PortableSnapshotLoadReceipt]] = {}
        self._numerical_captures: dict[
            str, tuple[tuple[str, str], ForwardBackwardNumericalCaptureReceipt]
        ] = {}
        self._command_mode = False
        self._lock = asyncio.Lock()
        self._cp_lookahead_lock = asyncio.Lock()
        self._active_job_id: str | None = None
        self._active_collective: asyncio.Future[Any] | None = None
        self._active_receive: asyncio.Future[Any] | None = None
        self._publications: dict[str, _PublicationState] = {}
        self._publication_drains: set[asyncio.Task[None]] = set()
        self._stop_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._valid = True

    @property
    def learner_version(self) -> int:
        state = self._command_runs.get(self.run_spec.run_id)
        return self._learner_version if state is None else state.learner_version

    @property
    def valid(self) -> bool:
        return self._valid

    async def register_command_run(
        self, run_spec: TrainingRunSpec
    ) -> PortableSnapshotInstallReceipt | None:
        if run_spec.runtime_fingerprint != self.runtime_spec.fingerprint:
            raise ValueError(
                "training run does not match the trainer runtime fingerprint"
            )
        if self._closed or not self._valid:
            raise RuntimeError("trainer runtime is invalid")
        prior = self._command_runs.get(run_spec.run_id)
        if prior is not None:
            if _command_run_identity(prior.spec) != _command_run_identity(run_spec):
                raise RuntimeError("run_id was reused with different trainer state")
            return prior.portable_install
        if self._jobs:
            raise RuntimeError("command runs cannot be registered after fused jobs")
        async with self._lock:
            prior = self._command_runs.get(run_spec.run_id)
            if prior is not None:
                if _command_run_identity(prior.spec) != _command_run_identity(run_spec):
                    raise RuntimeError("run_id was reused with different trainer state")
                return prior.portable_install
            try:
                values = await asyncio.wait_for(
                    self._actors.register_command_run.call(run_spec.model_dump_json()),
                    timeout=run_spec.event_timeout_s,
                )
                results = list(values.values())
                if (
                    {result["rank"] for result in results}
                    != set(range(len(self.runtime_spec.trainer_mesh.ranks)))
                    or {result["run_id"] for result in results} != {run_spec.run_id}
                    or {result["lora_rank"] for result in results}
                    != {run_spec.lora_rank}
                    or {tuple(result["lora_target_modules"]) for result in results}
                    != {run_spec.lora_target_modules}
                ):
                    raise RuntimeError("trainer ranks disagreed on run registration")
                raw_reads = tuple(result["portable_read"] for result in results)
                if run_spec.initial_portable_snapshot is None:
                    if any(value is not None for value in raw_reads):
                        raise RuntimeError(
                            "ordinary run returned portable read evidence"
                        )
                    portable_install = None
                else:
                    if any(value is None for value in raw_reads):
                        raise RuntimeError("portable run omitted rank read evidence")
                    reads = tuple(
                        sorted(
                            (
                                PortableSnapshotReadReceipt.model_validate(value)
                                for value in raw_reads
                            ),
                            key=lambda value: value.destination_rank,
                        )
                    )
                    portable_install = PortableSnapshotInstallReceipt(
                        archive_sha256=(
                            run_spec.initial_portable_snapshot.archive_sha256
                        ),
                        runtime_fingerprint=self.runtime_spec.fingerprint,
                        restore_optimizer=True,
                        ranks=reads,
                    )
                    portable_install.validate_archive(
                        run_spec.initial_portable_snapshot
                    )
            except BaseException as error:
                try:
                    await asyncio.wait_for(
                        self._actors.drain_command_run.call(run_spec.run_id),
                        timeout=run_spec.event_timeout_s,
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "portable registration rollback also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    await self._invalidate_after_command_failure(error)
                raise
            self._command_runs[run_spec.run_id] = _CommandRunState(
                spec=run_spec,
                learner_version=run_spec.initial_learner_version,
                next_operation_sequence=run_spec.initial_operation_sequence,
                portable_install=portable_install,
            )
            return portable_install

    async def command_run_state(self, run_id: str) -> TrainerCommandRunState:
        async with self._lock:
            state = self._command_runs.get(run_id)
            if state is None:
                raise KeyError(f"trainer command run {run_id!r} is absent")
            return TrainerCommandRunState(
                run_id=run_id,
                training_session_id=state.spec.training_session_id,
                learner_version=state.learner_version,
                next_operation_sequence=state.next_operation_sequence,
                open_forward_backward_operation_ids=tuple(
                    state.open_forward_backward_ids
                ),
            )

    async def capture_forward_backward_numerics(
        self,
        run_id: str,
        operation_id: str,
        batch: PackedBatchLeaseSet,
        root: str,
    ) -> ForwardBackwardNumericalCaptureReceipt:
        identity = (run_id, os.path.abspath(root))
        prior = self._numerical_captures.get(operation_id)
        if prior is not None:
            if prior[0] != identity:
                raise RuntimeError("numerical capture operation was reused")
            return prior[1]
        async with self._lock:
            prior = self._numerical_captures.get(operation_id)
            if prior is not None:
                if prior[0] != identity:
                    raise RuntimeError("numerical capture operation was reused")
                return prior[1]
            state = self._command_runs.get(run_id)
            cached = self._operations.get(operation_id)
            if (
                state is None
                or not state.open_forward_backward_ids
                or state.open_forward_backward_ids[-1] != operation_id
                or cached is None
            ):
                raise RuntimeError("numerical capture is not the open F/B suffix")
            result = cached[1]
            token_logprobs = tuple(
                TokenLogprobs.model_validate(value).model_dump_json()
                for value in result.get("token_logprobs", ())
            )
            values = await asyncio.wait_for(
                self._actors.capture_forward_backward_numerics.call(
                    run_id,
                    operation_id,
                    batch.model_dump_json(),
                    token_logprobs,
                    identity[1],
                ),
                timeout=state.spec.event_timeout_s,
            )
            receipt = build_forward_backward_capture(
                tuple(
                    ForwardBackwardNumericalRankReceipt.model_validate(value)
                    for value in values.values()
                )
            )
            self._numerical_captures[operation_id] = (identity, receipt)
            while len(self._numerical_captures) > _MAX_CACHED_COMMAND_OPERATIONS:
                self._numerical_captures.pop(next(iter(self._numerical_captures)))
            return receipt

    async def export_command_run_checkpoint(
        self,
        run_id: str,
        generation: TrainerGeneration,
        export_id: str,
    ) -> PortableSnapshotExportReceipt:
        identity = (run_id, generation)
        prior = self._checkpoint_exports.get(export_id)
        if prior is not None:
            if prior[0] != identity:
                raise RuntimeError("checkpoint export ID was reused")
            return prior[1]
        async with self._lock:
            prior = self._checkpoint_exports.get(export_id)
            if prior is not None:
                if prior[0] != identity:
                    raise RuntimeError("checkpoint export ID was reused")
                return prior[1]
            state = self._command_runs.get(run_id)
            if (
                state is None
                or state.open_forward_backward_ids
                or state.spec.training_session_id != generation.training_session_id
                or state.learner_version != generation.policy_step
            ):
                raise RuntimeError("checkpoint export generation is not quiescent")
            values = await asyncio.wait_for(
                self._actors.export_command_run_checkpoint.call(
                    run_id, generation.model_dump_json(), export_id
                ),
                timeout=state.spec.event_timeout_s,
            )
            results = list(values.values())
            world_size = len(self.runtime_spec.trainer_mesh.ranks)
            if {result["rank"] for result in results} != set(range(world_size)) or {
                result["run_id"] for result in results
            } != {run_id}:
                raise RuntimeError("trainer ranks disagreed on checkpoint export")
            raw_receipts = tuple(
                result["receipt"] for result in results if result["receipt"] is not None
            )
            if len(raw_receipts) != 1:
                raise RuntimeError("checkpoint export requires one committed inventory")
            rank_receipt = PortableSnapshotRankReceipt.model_validate(raw_receipts[0])
            if rank_receipt.rank != 0:
                raise RuntimeError(
                    "checkpoint export inventory is not coordinator-owned"
                )
            owners: dict[tuple[str, int], int] = {}
            for result in results:
                rank = int(result["rank"])
                for name, shard_rank in result["tensor_owners"]:
                    key = (str(name), int(shard_rank))
                    owners[key] = min(rank, owners.get(key, rank))
            portable_generation = PortableSnapshotGeneration(
                training_session_id=generation.training_session_id,
                policy_step=generation.policy_step,
                generation_id=generation.generation_id,
            )
            archive = build_portable_snapshot_archive(
                generation=portable_generation,
                checkpoint_digest=rank_receipt.checkpoint_digest,
                ranks=(rank_receipt,),
            )
            receipt = PortableSnapshotExportReceipt(
                export_id=export_id,
                generation=portable_generation,
                archive=archive,
                tensor_owners=tuple(
                    PortableSnapshotTensorOwner(
                        tensor_name=name,
                        shard_rank=shard_rank,
                        rank=rank,
                    )
                    for (name, shard_rank), rank in sorted(owners.items())
                ),
            )
            self._checkpoint_exports[export_id] = (identity, receipt)
            while len(self._checkpoint_exports) > 128:
                self._checkpoint_exports.pop(next(iter(self._checkpoint_exports)))
            return receipt

    async def install_command_run_checkpoint(
        self,
        operation: OperationRef,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotLoadReceipt:
        fingerprint = hashlib.sha256(
            (
                operation.model_dump_json()
                + "\0"
                + generation.model_dump_json()
                + "\0"
                + archive.archive_sha256
                + "\0"
                + json.dumps(restore_optimizer)
            ).encode()
        ).hexdigest()
        prior = self._checkpoint_loads.get(operation.operation_id)
        if prior is not None:
            if prior[0] != fingerprint:
                raise RuntimeError("checkpoint load operation was reused")
            return prior[1]
        async with self._lock:
            prior = self._checkpoint_loads.get(operation.operation_id)
            if prior is not None:
                if prior[0] != fingerprint:
                    raise RuntimeError("checkpoint load operation was reused")
                return prior[1]
            state = self._command_runs.get(operation.run_id)
            if (
                operation.kind != "load_state"
                or operation.reserved_output_learner_version != generation.policy_step
                or state is None
                or state.open_forward_backward_ids
                or state.learner_version != operation.learner_parent_version
                or state.next_operation_sequence != operation.sequence_id
                or state.spec.training_session_id != generation.training_session_id
            ):
                raise RuntimeError("checkpoint load command changed run state")
            values = await asyncio.wait_for(
                self._actors.install_command_run_checkpoint.call(
                    operation.run_id,
                    generation.model_dump_json(),
                    archive.model_dump_json(),
                    restore_optimizer,
                ),
                timeout=state.spec.event_timeout_s,
            )
            reads = tuple(
                sorted(
                    (
                        PortableSnapshotReadReceipt.model_validate(result["receipt"])
                        for result in values.values()
                    ),
                    key=lambda item: item.destination_rank,
                )
            )
            install = PortableSnapshotInstallReceipt(
                archive_sha256=archive.archive_sha256,
                runtime_fingerprint=self.runtime_spec.fingerprint,
                restore_optimizer=restore_optimizer,
                ranks=reads,
            )
            install.validate_archive(archive)
            receipt = PortableSnapshotLoadReceipt(
                operation_id=operation.operation_id,
                generation=PortableSnapshotGeneration(
                    training_session_id=generation.training_session_id,
                    policy_step=generation.policy_step,
                    generation_id=generation.generation_id,
                ),
                install=install,
            )
            self._checkpoint_loads[operation.operation_id] = (fingerprint, receipt)
            while len(self._checkpoint_loads) > 128:
                self._checkpoint_loads.pop(next(iter(self._checkpoint_loads)))
            return receipt

    async def record_control_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None:
        async with self._lock:
            state = self._command_runs.get(operation.run_id)
            if state is None:
                raise KeyError(f"trainer command run {operation.run_id!r} is absent")
            if operation.sequence_id != state.next_operation_sequence:
                raise RuntimeError("control command sequence changed")
            if operation.learner_parent_version != state.learner_version:
                raise RuntimeError("control command learner parent changed")
            if operation.kind == "load_state":
                if state.open_forward_backward_ids:
                    raise RuntimeError("load_state cannot discard open gradients")
                if operation.reserved_output_learner_version != learner_version:
                    raise RuntimeError("load_state output learner version changed")
                state.learner_version = learner_version
            elif (
                operation.kind not in {"save_sampler", "save_state"}
                or operation.reserved_output_learner_version is not None
                or learner_version != state.learner_version
            ):
                raise RuntimeError("invalid non-GPU control command transition")
            state.next_operation_sequence += 1

    async def record_no_work_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None:
        async with self._lock:
            state = self._command_runs.get(operation.run_id)
            if state is None:
                raise KeyError(f"trainer command run {operation.run_id!r} is absent")
            if (
                operation.kind not in {"forward", "forward_backward"}
                or operation.sequence_id != state.next_operation_sequence
                or operation.learner_parent_version != state.learner_version
                or operation.reserved_output_learner_version is not None
                or learner_version != state.learner_version
            ):
                raise RuntimeError("invalid zero-work command transition")
            self._command_mode = True
            state.next_operation_sequence += 1

    async def release_command_run_for_migration(self, run_id: str) -> None:
        async with self._lock:
            state = self._command_runs.get(run_id)
            if state is None:
                return
            values = await asyncio.wait_for(
                self._actors.release_command_run_for_migration.call(
                    run_id,
                    tuple(state.open_forward_backward_ids),
                ),
                timeout=state.spec.event_timeout_s,
            )
            results = list(values.values())
            expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
            expected_contributions = tuple(state.open_forward_backward_ids)
            if (
                {result["rank"] for result in results} != expected_ranks
                or {result["run_id"] for result in results} != {run_id}
                or {
                    tuple(result["discarded_forward_backward_operation_ids"])
                    for result in results
                }
                != {expected_contributions}
            ):
                raise RuntimeError("trainer ranks disagreed while releasing a run")
            self._command_runs.pop(run_id)

    async def drain_command_run(self, run_id: str) -> None:
        async with self._lock:
            state = self._command_runs.get(run_id)
            if state is None:
                return
            if state.open_forward_backward_ids:
                raise RuntimeError("cannot drain a run with open F/B contributions")
            values = await asyncio.wait_for(
                self._actors.drain_command_run.call(run_id),
                timeout=state.spec.event_timeout_s,
            )
            results = list(values.values())
            if {result["rank"] for result in results} != set(
                range(len(self.runtime_spec.trainer_mesh.ranks))
            ) or {result["run_id"] for result in results} != {run_id}:
                raise RuntimeError("trainer ranks disagreed while draining a run")
            self._command_runs.pop(run_id)

    async def forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        launch = await self.start_forward_backward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> TrainerCommandLaunch:
        def validate() -> None:
            if batch.ref != job.batch:
                raise ValueError("F/B batch ref does not match its leases")
            if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
                raise ValueError("F/B batch length does not match the trainer runtime")
            if bool(job.batch.moe_routing_replay) != bool(
                self.runtime_spec.enable_moe_routing_replay
            ):
                raise ValueError(
                    "F/B routing replay does not match the trainer runtime"
                )

        return await self._start_command(
            job,
            lambda ready: self._actors.execute_forward_backward.call(
                job.model_dump_json(), batch.model_dump_json(), ready
            ),
            validate=validate,
            label="F/B",
            backward=True,
        )

    async def forward(
        self,
        job: ForwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        launch = await self.start_forward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_forward(
        self,
        job: ForwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> TrainerCommandLaunch:
        def validate() -> None:
            if batch.ref != job.batch:
                raise ValueError("forward batch ref does not match its leases")
            if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
                raise ValueError("forward batch length does not match trainer runtime")
            if bool(job.batch.moe_routing_replay) != bool(
                self.runtime_spec.enable_moe_routing_replay
            ):
                raise ValueError(
                    "forward routing replay does not match the trainer runtime"
                )

        return await self._start_command(
            job,
            lambda ready: self._actors.execute_forward.call(
                job.model_dump_json(), batch.model_dump_json(), ready
            ),
            validate=validate,
            label="forward",
            backward=False,
        )

    async def sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        launch = await self.start_sft_forward_backward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
    ) -> TrainerCommandLaunch:
        def validate() -> None:
            if batch.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT F/B payload fingerprint differs from its command")

        return await self._start_command(
            job,
            lambda ready: self._actors.execute_sft_forward_backward.call(
                job.model_dump_json(), batch, ready
            ),
            validate=validate,
            label="SFT F/B",
            backward=True,
        )

    async def sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        launch = await self.start_sft_forward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
    ) -> TrainerCommandLaunch:
        def validate() -> None:
            if batch.fingerprint != job.batch_fingerprint:
                raise ValueError(
                    "SFT forward payload fingerprint differs from its command"
                )

        return await self._start_command(
            job,
            lambda ready: self._actors.execute_sft_forward.call(
                job.model_dump_json(), batch, ready
            ),
            validate=validate,
            label="SFT forward",
            backward=False,
        )

    async def _start_command(
        self,
        job: _DeferredCommandJob,
        invoke: Callable[[Port[dict[str, Any]]], Awaitable[Any]],
        *,
        validate: Callable[[], None],
        label: str,
        backward: bool,
    ) -> TrainerCommandLaunch:
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different command"
                    )
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached[1])
                return TrainerCommandLaunch(completion=completion)
            inflight = self._command_launches.get(job.operation_id)
            if inflight is not None:
                if inflight[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different command"
                    )
                return inflight[1]
            state = self._validate_command(job)
            validate()
            ready_port, receiver = Channel.open()
            rank_call = asyncio.ensure_future(invoke(ready_port))
            readiness = asyncio.create_task(
                self._collect_command_ready(receiver, job, label=label),
                name=f"megatron-{label.lower().replace(' ', '-')}-ready-"
                f"{job.operation_id}",
            )
            supervision = asyncio.create_task(self._supervision.wait())
            self._active_job_id = job.operation_id
            self._active_collective = rank_call
            deadline = asyncio.get_running_loop().time() + state.spec.event_timeout_s
            try:
                done, _ = await asyncio.wait(
                    {rank_call, readiness, supervision},
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError(f"trainer ranks did not reach {label} readiness")
                if supervision in done:
                    raise RuntimeError("trainer mesh failed: " + supervision.result())
                if readiness not in done:
                    await rank_call
                    raise RuntimeError(
                        f"trainer {label} returned before every rank reported ready"
                    )
                readiness.result()
                self._command_mode = True
                if backward:
                    state.open_forward_backward_ids.append(job.operation_id)
                state.next_operation_sequence += 1
                self._clear_active(job.operation_id)
                completion = asyncio.create_task(
                    self._complete_command(
                        job,
                        rank_call,
                        deadline=deadline,
                        label=label,
                        backward=backward,
                    ),
                    name=f"megatron-{label.lower().replace(' ', '-')}-result-"
                    f"{job.operation_id}",
                )
                completion.add_done_callback(consume_future_exception)
                launch = TrainerCommandLaunch(completion=completion)
                self._command_launches[job.operation_id] = (job.fingerprint, launch)
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                await self._invalidate_after_command_failure(error)
                raise
            finally:
                supervision.cancel()
                supervision.add_done_callback(consume_future_exception)
                self._clear_active(job.operation_id)

    async def _collect_command_ready(
        self,
        receiver: Any,
        job: _DeferredCommandJob,
        *,
        label: str,
    ) -> None:
        results = [
            _CommandReady.model_validate(await receiver.recv())
            for _ in self.runtime_spec.trainer_mesh.ranks
        ]
        if {item.rank for item in results} != set(range(len(results))) or any(
            (item.operation_id, item.learner_version)
            != (job.operation_id, job.expected_learner_version)
            for item in results
        ):
            raise RuntimeError(f"trainer {label} readiness changed rank identity")
        failures = [item for item in results if item.error_type is not None]
        if failures:
            details = "\n".join(
                f"rank {item.rank}: {item.error_type}: {item.message}\n"
                f"{item.traceback_text or ''}"
                for item in failures
            )
            raise RuntimeError(f"trainer {label} failed before GPU release:\n{details}")

    async def _complete_command(
        self,
        job: _DeferredCommandJob,
        rank_call: asyncio.Future[Any],
        *,
        deadline: float,
        label: str,
        backward: bool,
    ) -> dict[str, Any]:
        supervision = asyncio.create_task(self._supervision.wait())
        try:
            done, _ = await asyncio.wait(
                {rank_call, supervision},
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(f"trainer {label} result settlement timed out")
            if supervision in done:
                raise RuntimeError("trainer mesh failed: " + supervision.result())
            values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._raise_command_failure(results)
            self._validate_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            if backward and {
                result["loss_bearing_token_count"] for result in results
            } != {job.expected_global_loss_bearing_tokens}:
                raise RuntimeError(
                    f"trainer ranks disagree on {label} token provenance"
                )
            usage = {
                (
                    *((result["completed_gradient_steps"],) if backward else ()),
                    result["logical_nonpadding_tokens"],
                    result["executed_token_equivalents"],
                )
                for result in results
            }
            if len(usage) != 1:
                raise RuntimeError(f"trainer ranks disagree on {label} execution usage")
            result = dict(next(item for item in results if item["rank"] == 0))
            result["gpu_service_ns"], result["gpu_count"] = self._aggregate_gpu_service(
                results
            )
            self._cache_operation(job.operation_id, job.fingerprint, result)
            return result
        except BaseException as error:
            await self._invalidate_after_command_failure(error)
            raise
        finally:
            supervision.cancel()
            supervision.add_done_callback(consume_future_exception)
            self._command_launches.pop(job.operation_id, None)

    async def optim_step(self, job: OptimizerJobSpec) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different optimizer"
                    )
                return cached[1]
            state = self._validate_command(job)
            contributions = tuple(state.open_forward_backward_ids)
            if job.contributing_forward_backward_operation_ids != contributions:
                raise RuntimeError("optimizer does not seal the exact open F/B set")
            try:
                values = await self._run_resident_collective(
                    job.operation_id,
                    self._actors.execute_optimizer.call(job.model_dump_json()),
                    invalidate_on_error=True,
                )
                results = list(values.values())
                self._raise_command_failure(results)
                self._validate_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                contribution_sets = {
                    tuple(result["contributing_forward_backward_operation_ids"])
                    for result in results
                }
                if contribution_sets != {contributions}:
                    raise RuntimeError(
                        "trainer ranks consumed different F/B operations"
                    )
            except BaseException as error:
                await self._invalidate_after_command_failure(error)
                raise
            result = next(result for result in results if result["rank"] == 0)
            result["gpu_service_ns"], result["gpu_count"] = self._aggregate_gpu_service(
                results
            )
            state.open_forward_backward_ids.clear()
            state.learner_version = job.learner_version
            state.next_operation_sequence += 1
            if job.run_id == self.run_spec.run_id:
                self._learner_version = job.learner_version
            self._cache_operation(job.operation_id, job.fingerprint, result)
            return result

    async def publish_command_generation(
        self, spec: CommandPublicationSpec
    ) -> tuple[tuple[TrainerRankPublication, ...], dict[str, float]]:
        async with self._lock:
            if self._closed or not self._valid:
                raise RuntimeError("trainer runtime is invalid")
            state = self._command_runs.get(spec.run_id)
            if state is None or state.learner_version != spec.generation.policy_step:
                raise RuntimeError("published generation is not resident")
            if state.spec.training_session_id != spec.generation.training_session_id:
                raise RuntimeError("published generation belongs to another session")
            if state.open_forward_backward_ids:
                raise RuntimeError("cannot publish a generation with open gradients")
            try:
                values = await self._run_resident_collective(
                    spec.generation.generation_id,
                    self._actors.publish_command_generation.call(
                        spec.model_dump_json()
                    ),
                    invalidate_on_error=True,
                )
                results = list(values.values())
                expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
                if (
                    {result["rank"] for result in results} != expected_ranks
                    or {result["run_id"] for result in results} != {spec.run_id}
                    or {result["generation_id"] for result in results}
                    != {spec.generation.generation_id}
                ):
                    raise RuntimeError("trainer ranks disagreed on command publication")
                records = tuple(
                    sorted(
                        (
                            TrainerRankPublication.model_validate(result["record"])
                            for result in results
                        ),
                        key=lambda record: record.rank,
                    )
                )
                metric_names = {
                    name for result in results for name in result["metrics"]
                }
                metrics = {
                    name: max(
                        float(result["metrics"].get(name, 0.0)) for result in results
                    )
                    for name in metric_names
                }
                return records, metrics
            except BaseException as error:
                await self._invalidate_after_command_failure(error)
                raise

    def _cache_operation(
        self, operation_id: str, fingerprint: str, result: dict[str, Any]
    ) -> None:
        self._operations[operation_id] = (fingerprint, result)
        while len(self._operations) > _MAX_CACHED_COMMAND_OPERATIONS:
            self._operations.pop(next(iter(self._operations)))

    async def _invalidate_after_command_failure(self, error: BaseException) -> None:
        if not self._valid:
            return
        self._valid = False
        self._closed = True
        self._cancel_active()
        await cleanup_after_failure(
            error,
            self._force_stop,
            message="command result validation and trainer cleanup failed",
        )

    def _validate_command(
        self,
        job: (
            ForwardJobSpec
            | ForwardBackwardJobSpec
            | SftForwardJobSpec
            | SftForwardBackwardJobSpec
            | OptimizerJobSpec
        ),
    ) -> _CommandRunState:
        if self._closed or not self._valid:
            raise RuntimeError("trainer runtime is invalid")
        if self._active_job_id is not None:
            raise RuntimeError("trainer has an active operation")
        if self._jobs:
            raise RuntimeError("fused jobs cannot be mixed with command operations")
        state = self._command_runs.get(job.run_id)
        if state is None:
            raise ValueError("operation run_id is not registered on this trainer slot")
        if job.training_session_id != state.spec.training_session_id:
            raise ValueError("operation training session does not match this run")
        if job.sequence_id != state.next_operation_sequence:
            raise RuntimeError(
                "trainer operation sequence must be gapless: "
                f"expected={state.next_operation_sequence}, got={job.sequence_id}"
            )
        if job.expected_learner_version != state.learner_version:
            raise ValueError(
                "operation learner parent mismatch: "
                f"operation={job.expected_learner_version}, "
                f"runtime={state.learner_version}"
            )
        return state

    def _validate_command_results(
        self,
        results: list[dict[str, Any]],
        *,
        operation_id: str,
        learner_version: int,
    ) -> None:
        expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
        if (
            {result["rank"] for result in results} != expected_ranks
            or {result["operation_id"] for result in results} != {operation_id}
            or {result["learner_version"] for result in results} != {learner_version}
        ):
            raise RuntimeError("trainer ranks disagree on command completion")

    def _raise_command_failure(self, results: list[dict[str, Any]]) -> None:
        failures = [
            result for result in results if result.get("command_status") != "succeeded"
        ]
        if not failures:
            return
        expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
        complete_rank_set = {result.get("rank") for result in results} == expected_ranks
        measured = [result.get("gpu_service_ns") for result in results]
        complete_timing = complete_rank_set and all(
            isinstance(value, int) and value >= 0 for value in measured
        )
        usage = CommandExecutionUsage(
            logical_nonpadding_tokens=UsageMeasurement.unknown(),
            executed_token_equivalents=UsageMeasurement.unknown(),
            gpu_count=(
                UsageMeasurement.complete(len(expected_ranks))
                if complete_rank_set
                else UsageMeasurement.unknown()
            ),
            gpu_service_ns=(
                UsageMeasurement.exact_partial(max(measured))
                if complete_timing
                else UsageMeasurement.unknown()
            ),
        )
        code = (
            "cancelled"
            if failures and all(result.get("cancelled") is True for result in failures)
            else "execution_failed"
        )
        details = "; ".join(
            f"rank {result.get('rank')}: {result.get('error_type')}: "
            f"{result.get('message')}"
            for result in failures[:4]
        )
        raise OperationExecutionError(code, details, usage=usage)

    def _aggregate_gpu_service(self, results: list[dict[str, Any]]) -> tuple[int, int]:
        expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
        if {result.get("rank") for result in results} != expected_ranks:
            raise OperationExecutionError(
                "execution_failed",
                "trainer rank coverage is incomplete for GPU service",
            )
        measured = [result.get("gpu_service_ns") for result in results]
        if not all(isinstance(value, int) and value >= 0 for value in measured):
            raise OperationExecutionError(
                "execution_failed",
                "trainer rank GPU service timing is incomplete",
            )
        return max(measured), len(expected_ranks)

    async def prepare_cp_lookahead(
        self,
        batch: PackedBatchLeaseSet,
        *,
        global_grad_accumulation_sequences: int | None,
    ) -> dict[str, float]:
        if not self._cp_lookahead_ports:
            return {}
        async with self._cp_lookahead_lock:
            if self._closed or not self._valid:
                raise RuntimeError("trainer run is not available for CP lookahead")
            reply, receiver = Channel.open()
            request = (
                batch.model_dump_json(),
                batch.ref.batch_id,
                global_grad_accumulation_sequences,
                reply,
            )
            started = time.perf_counter()
            for port in self._cp_lookahead_ports:
                port.send(request)
            async with asyncio.timeout(self.run_spec.event_timeout_s):
                results = []
                for _ in self._cp_lookahead_ports:
                    results.append(
                        _CpLookaheadResult.model_validate(await receiver.recv())
                    )
            expected_ranks = set(range(len(self._cp_lookahead_ports)))
            if {result.rank for result in results} != expected_ranks or any(
                result.batch_id != batch.ref.batch_id for result in results
            ):
                raise RuntimeError("CP lookahead returned mismatched rank or batch IDs")
            failures = [result for result in results if result.error_type is not None]
            if failures:
                details = "\n".join(
                    f"rank {result.rank}: {result.error_type}: {result.message}\n"
                    f"{result.traceback_text or ''}"
                    for result in failures
                )
                raise RuntimeError(f"CP lookahead failed:\n{details}")
            return {
                "time/step_cp_lookahead_wait_s": time.perf_counter() - started,
                "time/step_cp_lookahead_rank_max_s": max(
                    result.elapsed_s for result in results
                ),
                "data/step_cp_preplanned_sequences_rank_max": float(
                    max(result.planned_sequences for result in results)
                ),
            }

    async def score(
        self,
        job: ResidentScoreJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> ResidentScoreResult:
        async with self._lock:
            if error := self._validate_resident_score(job, batch):
                raise error
            values = await self._run_resident_collective(
                job.job_id,
                self._actors.score.call(job.model_dump_json(), batch.model_dump_json()),
                invalidate_on_error=True,
            )
            shards = tuple(
                ResidentScoreShard.model_validate(value) for value in values.values()
            )
            return _merge_resident_score_shards(
                shards,
                job=job,
                world_size=len(self.runtime_spec.trainer_mesh.ranks),
            )

    async def inspect_resident_lora(
        self,
        request: ResidentLoraInspectionSpec,
    ) -> ResidentLoraInspectionResult:
        async with self._lock:
            if error := self._validate_resident_inspection(request):
                raise error
            values = await self._run_resident_collective(
                request.request_id,
                self._actors.inspect_resident_lora.call(request.model_dump_json()),
                invalidate_on_error=False,
            )
            shards = tuple(
                ResidentLoraInspectionShard.model_validate(value)
                for value in values.values()
            )
            return _merge_resident_lora_shards(
                shards,
                request=request,
                world_size=len(self.runtime_spec.trainer_mesh.ranks),
            )

    async def publish_external_lora(
        self,
        target: ExternalLoraTarget,
        sink_spec: ExternalLoraSinkSpec,
        *,
        source_topology: str,
    ) -> tuple[ExternalLoraPlan, ExternalLoraPublication, dict[str, float]]:
        """Publish rank-owned adapter views and settle one manifest last."""

        async with self._lock:
            values = await self._run_resident_collective(
                target.operation_id,
                self._actors.publish_external_lora.call(
                    target.model_dump_json(),
                    sink_spec.model_dump_json(),
                    source_topology,
                ),
                invalidate_on_error=False,
            )
            results = list(values.values())
            expected_ranks = set(range(len(self.runtime_spec.trainer_mesh.ranks)))
            if {result["rank"] for result in results} != expected_ranks:
                raise RuntimeError("external LoRA publication omitted a trainer rank")
            plans = tuple(
                ExternalLoraPlan.model_validate(result["plan"]) for result in results
            )
            if any(plan != plans[0] for plan in plans[1:]):
                raise RuntimeError("trainer ranks returned different external plans")
            publications = tuple(
                (result["rank"], ExternalLoraPublication.model_validate(value))
                for result in results
                if (value := result["publication"]) is not None
            )
            if (
                len(publications) != 1
                or publications[0][0] != plans[0].coordinator_rank
            ):
                raise RuntimeError(
                    "external LoRA publication requires one coordinator result"
                )
            metric_names = {name for result in results for name in result["metrics"]}
            metrics = {
                name: max(float(result["metrics"].get(name, 0.0)) for result in results)
                for name in metric_names
            }
            return plans[0], publications[0][1], metrics

    async def _run_resident_collective(
        self,
        request_id: str,
        operation: Awaitable[Any],
        *,
        invalidate_on_error: bool,
    ) -> Any:
        collective = asyncio.ensure_future(operation)
        supervision = asyncio.create_task(self._supervision.wait())
        self._active_job_id = request_id
        self._active_collective = collective
        try:
            done, _ = await asyncio.wait(
                {collective, supervision},
                timeout=self.run_spec.event_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(
                    "trainer ranks produced no resident diagnostic result for "
                    f"{self.run_spec.event_timeout_s:g}s"
                )
            if supervision in done:
                raise RuntimeError("trainer mesh failed: " + supervision.result())
            return await collective
        except BaseException as exc:
            if invalidate_on_error or not collective.done() or supervision.done():
                self._valid = False
                self._closed = True
                self._cancel_active()
                await cleanup_after_failure(
                    exc,
                    self._force_stop,
                    message="resident diagnostic and trainer cleanup failed",
                )
            raise
        finally:
            supervision.cancel()
            supervision.add_done_callback(consume_future_exception)
            self._clear_active(request_id)

    async def train(
        self,
        job: TrainJobSpec,
        batch: PackedBatchLeaseSet,
        *,
        on_dispatched: Callable[[], None] | None = None,
    ) -> AsyncIterator[TrainEvent]:
        async for event in self._train(
            job,
            lambda port: self._actors.execute.call(
                job.model_dump_json(), batch.model_dump_json(), port
            ),
            lambda: self._validate_rl(job, batch),
            on_dispatched=on_dispatched,
        ):
            yield event

    async def train_sft(
        self, job: SFTJobSpec, batches: tuple[SFTBatchData, ...]
    ) -> AsyncIterator[TrainEvent]:
        async for event in self._train(
            job,
            lambda port: self._actors.execute_sft.call(
                job.model_dump_json(), batches, port
            ),
            lambda: self._validate_sft(job, batches),
        ):
            yield event

    async def _train(
        self,
        job: TrainerJobSpec,
        start: Callable[[Port[dict[str, Any]]], Awaitable[Any]],
        validate: Callable[[], BaseException | None],
        *,
        on_dispatched: Callable[[], None] | None = None,
    ) -> AsyncIterator[TrainEvent]:
        def signal_dispatched() -> None:
            nonlocal on_dispatched
            callback, on_dispatched = on_dispatched, None
            if callback is not None:
                callback()

        cached = self._jobs.get(job.job_id)
        if cached is not None and cached[0] == job.fingerprint:
            signal_dispatched()
            for event in cached[1]:
                yield event
            return

        async with self._lock:
            cached = self._jobs.get(job.job_id)
            if cached is not None:
                if cached[0] == job.fingerprint:
                    signal_dispatched()
                    for event in cached[1]:
                        yield event
                    return
                yield TrainAccepted(
                    job_id=job.job_id,
                    run_id=job.run_id,
                    sequence=0,
                    expected_learner_version=job.expected_learner_version,
                )
                yield self._failed(
                    job,
                    1,
                    RuntimeError("job_id was already used with a different job"),
                    False,
                )
                return
            events: list[TrainEvent] = []

            def emit(event: TrainEvent) -> TrainEvent:
                events.append(event)
                return event

            yield emit(
                TrainAccepted(
                    job_id=job.job_id,
                    run_id=job.run_id,
                    sequence=0,
                    expected_learner_version=job.expected_learner_version,
                )
            )
            error = validate()
            if error is not None:
                yield emit(self._failed(job, len(events), error, not self._valid))
                return

            publication = asyncio.get_running_loop().create_future()
            publication.add_done_callback(consume_future_exception)
            generation_id = job.output_generation_id
            if generation_id in self._publications:
                raise RuntimeError(
                    f"publication generation already exists: {generation_id}"
                )
            self._expire_prior_publications()
            publication_state = _PublicationState(generation_id, publication)
            self._publications[generation_id] = publication_state
            supervision: asyncio.Task[str] | None = None
            try:
                send_port, receiver = Channel[dict[str, Any]].open()
                dispatch_started = time.perf_counter()
                final_progress_received: float | None = None
                collective = asyncio.ensure_future(start(send_port))
                signal_dispatched()
                receive = asyncio.ensure_future(receiver.recv())
                supervision = asyncio.create_task(self._supervision.wait())
                self._active_job_id = job.job_id
                self._active_collective = collective
                self._active_receive = receive
                while True:
                    waiters = {receive, supervision}
                    if not collective.done():
                        waiters.add(collective)
                    event_timeout_s = (
                        self.run_spec.initial_event_timeout_s
                        if len(events) == 1
                        and self.run_spec.initial_event_timeout_s is not None
                        else self.run_spec.event_timeout_s
                    )
                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=event_timeout_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        raise TimeoutError(
                            f"trainer ranks produced no event for {event_timeout_s:g}s"
                        )
                    if supervision in done:
                        raise RuntimeError(
                            "trainer mesh failed: " + supervision.result()
                        )
                    if collective in done:
                        await collective
                        if receive not in done:
                            continue
                    payload = receive.result()
                    if payload["kind"] in {
                        "publication_succeeded",
                        "publication_failed",
                    }:
                        self._record_publication(payload)
                        receive = asyncio.ensure_future(receiver.recv())
                        self._active_receive = receive
                        continue
                    if payload["kind"] == "rank_failed":
                        raise RuntimeError(
                            f"trainer rank {payload['rank']} failed: "
                            f"{payload['error_type']}: {payload['message']}\n"
                            f"{payload['traceback']}"
                        )
                    if payload["kind"] == "progress":
                        if payload["step_index"] + 1 == payload["num_steps"]:
                            final_progress_received = time.perf_counter()
                        event = TrainProgress(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            step_index=payload["step_index"],
                            num_steps=payload["num_steps"],
                            metrics=payload["metrics"],
                        )
                    elif payload["kind"] == "adapter_ready":
                        event = AdapterReady(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            learner_version=payload["learner_version"],
                            adapter_path=payload["adapter_path"],
                        )
                    elif payload["kind"] == "actor_completed":
                        actor_completed_received = time.perf_counter()
                        values = await collective
                        collective_completed = time.perf_counter()
                        results = list(values.values())
                        versions = {result["learner_version"] for result in results}
                        ranks = {result["rank"] for result in results}
                        expected_ranks = set(
                            range(len(self.runtime_spec.trainer_mesh.ranks))
                        )
                        if versions != {job.learner_version} or ranks != expected_ranks:
                            raise RuntimeError(
                                "trainer ranks did not agree on job completion"
                            )
                        metrics = dict(payload["metrics"])
                        cache_metrics = [
                            result.get("compile_cache", {}) for result in results
                        ]
                        if any(cache_metrics):
                            metrics.update(
                                {
                                    "trainer/compile_cache_hit_fraction": sum(
                                        value.get("hit", 0.0) for value in cache_metrics
                                    )
                                    / len(cache_metrics),
                                    "trainer/compile_cache_published_fraction": sum(
                                        value.get("published", 0.0)
                                        for value in cache_metrics
                                    )
                                    / len(cache_metrics),
                                    "trainer/compile_cache_artifact_bytes_max": max(
                                        value.get("artifact_bytes", 0.0)
                                        for value in cache_metrics
                                    ),
                                    "time/trainer_compile_cache_load_max_s": max(
                                        value.get("load_s", 0.0)
                                        for value in cache_metrics
                                    ),
                                    "time/trainer_compile_cache_publish_max_s": max(
                                        value.get("publish_s", 0.0)
                                        for value in cache_metrics
                                    ),
                                }
                            )
                        if final_progress_received is not None:
                            metrics.update(
                                {
                                    "time/step_monarch_dispatch_to_progress_s": (
                                        final_progress_received - dispatch_started
                                    ),
                                    "time/step_monarch_progress_to_completed_s": (
                                        actor_completed_received
                                        - final_progress_received
                                    ),
                                }
                            )
                        metrics["time/step_monarch_collective_tail_s"] = (
                            collective_completed - actor_completed_received
                        )
                        completed = TrainCompleted(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            learner_version=job.learner_version,
                            metrics=metrics,
                        )
                        if not publication.done():
                            publication_state.drain_done = False
                            drain = asyncio.create_task(
                                self._drain_publication(receiver, publication_state)
                            )
                            self._publication_drains.add(drain)
                            drain.add_done_callback(self._publication_drains.discard)
                            drain.add_done_callback(consume_future_exception)
                        yield completed
                        self._learner_version = job.learner_version
                        emit(completed)
                        self._clear_active(job.job_id)
                        break
                    else:
                        raise RuntimeError(
                            f"trainer rank sent unknown event {payload['kind']!r}"
                        )
                    yield emit(TRAIN_EVENT_ADAPTER.validate_python(event))
                    receive = asyncio.ensure_future(receiver.recv())
                    self._active_receive = receive
            except BaseException as exc:
                if not publication.done():
                    publication.set_exception(exc)
                    publication_state.records.clear()
                closed_by_caller = self._closed
                self._valid = False
                self._closed = True
                self._cancel_active()
                await cleanup_after_failure(
                    exc,
                    self._force_stop,
                    message="training and forced trainer ProcMesh cleanup failed",
                )
                caller_cancelled = isinstance(exc, GeneratorExit) or (
                    isinstance(exc, asyncio.CancelledError)
                    and _current_task_is_cancelling()
                )
                if caller_cancelled or (
                    isinstance(exc, asyncio.CancelledError) and closed_by_caller
                ):
                    cancelled = TrainCancelled(
                        job_id=job.job_id,
                        run_id=job.run_id,
                        sequence=len(events),
                        reason="train stream was cancelled",
                    )
                    events.append(cancelled)
                    if caller_cancelled:
                        raise
                    yield cancelled
                    return
                failure = self._failed(job, len(events), exc, True)
                events.append(failure)
                yield failure
            finally:
                if supervision is not None:
                    supervision.cancel()
                    supervision.add_done_callback(consume_future_exception)
                self._clear_active(job.job_id)
                # Older jobs cannot be retried after the sequential learner advances.
                self._jobs = {job.job_id: (job.fingerprint, tuple(events))}
                publication_state.train_done = True
                self._retire_publication(publication_state)

    def wait_for_publication(
        self, generation_id: str
    ) -> Awaitable[tuple[TrainerRankPublication, ...]]:
        state = self._publications.get(generation_id)
        if state is None:
            raise RuntimeError(f"trainer has no publication {generation_id}")
        if not state.late_waitable:
            raise RuntimeError(
                f"trainer publication {generation_id} is no longer waitable"
            )
        # Reserve before returning control; the next train may expire late waiters
        # without yielding to the task which awaits this publication.
        state.active_waiters += 1
        return self._await_publication(state)

    async def _await_publication(
        self, state: "_PublicationState"
    ) -> tuple[TrainerRankPublication, ...]:
        observed = False
        try:
            result = await asyncio.shield(state.future)
            observed = True
            return result
        except asyncio.CancelledError:
            observed = state.future.cancelled()
            raise
        except BaseException:
            observed = True
            raise
        finally:
            state.active_waiters -= 1
            state.outcome_observed |= observed
            self._retire_publication(state)

    def _record_publication(self, payload: dict[str, Any]) -> None:
        event = TRAINER_PUBLICATION_EVENT_ADAPTER.validate_python(payload)
        generation_id = (
            event.record.generation.generation_id
            if isinstance(event, TrainerPublicationSucceeded)
            else event.generation_id
        )
        state = self._publications.get(generation_id)
        if state is None:
            raise RuntimeError(
                f"trainer rank reported unknown publication {generation_id}"
            )
        future = state.future
        if future.done():
            if not future.cancelled() and future.exception() is not None:
                return
            raise RuntimeError(
                f"trainer publication {generation_id} is already terminal"
            )
        if isinstance(event, TrainerPublicationFailed):
            future.set_exception(
                RuntimeError(
                    f"trainer rank {event.rank} publication failed "
                    f"({event.error_type}): {event.message}"
                )
            )
            state.records.clear()
            self._retire_publication(state)
            return
        record = event.record
        world_size = len(self.runtime_spec.trainer_mesh.ranks)
        if record.rank >= world_size:
            raise RuntimeError(f"publication reported invalid rank {record.rank}")
        records = state.records
        if record.rank in records:
            raise RuntimeError(
                f"trainer rank {record.rank} published {generation_id} twice"
            )
        records[record.rank] = record
        if len(records) == world_size:
            future.set_result(tuple(records[rank] for rank in range(world_size)))
            records.clear()
            self._retire_publication(state)

    async def _drain_publication(
        self, receiver: Any, state: "_PublicationState"
    ) -> None:
        publication = state.future
        supervision = asyncio.create_task(self._supervision.wait())
        receive: asyncio.Future[Any] | None = None
        try:
            while not publication.done():
                receive = asyncio.ensure_future(receiver.recv())
                done, _ = await asyncio.wait(
                    {receive, supervision},
                    timeout=self.run_spec.shutdown_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError(
                        f"trainer ranks produced no publication event for "
                        f"{self.run_spec.shutdown_timeout_s:g}s"
                    )
                if supervision in done:
                    raise RuntimeError("trainer mesh failed: " + supervision.result())
                payload = receive.result()
                if payload["kind"] == "rank_failed":
                    raise RuntimeError(
                        f"trainer rank {payload['rank']} failed after training: "
                        f"{payload['error_type']}: {payload['message']}"
                    )
                self._record_publication(payload)
        except BaseException as exc:
            if not publication.done():
                publication.set_exception(exc)
                state.records.clear()
            raise
        finally:
            supervision.cancel()
            supervision.add_done_callback(consume_future_exception)
            if receive is not None and not receive.done():
                receive.cancel()
                receive.add_done_callback(consume_future_exception)
            state.drain_done = True
            self._retire_publication(state)

    def _expire_prior_publications(self) -> None:
        for state in tuple(self._publications.values()):
            state.late_waitable = False
            self._retire_publication(state)

    def _retire_publication(self, state: "_PublicationState") -> None:
        # A waiter can observe a terminal event before the train/drain producer exits.
        if not (
            state.future.done()
            and (state.outcome_observed or not state.late_waitable)
            and state.active_waiters == 0
            and state.train_done
            and state.drain_done
        ):
            return
        if self._publications.get(state.generation_id) is state:
            self._publications.pop(state.generation_id)

    async def advance_without_training(
        self,
        *,
        source: TrainerGeneration,
        output: TrainerGeneration,
        optimizer_state_path: str,
        adapter: Any | None,
    ) -> dict[str, float]:
        async with self._lock:
            if self._closed or not self._valid:
                raise RuntimeError("trainer runtime is invalid")
            if self._active_job_id is not None:
                raise RuntimeError("trainer has an active job")
            if self._command_mode:
                raise RuntimeError(
                    "legacy learner transitions cannot follow command operations"
                )
            if (
                source.training_session_id != self.run_spec.training_session_id
                or source.policy_step != self._learner_version
            ):
                raise ValueError(
                    "expected learner version mismatch: "
                    f"transition={source.policy_step}, "
                    f"runtime={self._learner_version}"
                )
            if (
                output.training_session_id != source.training_session_id
                or output.policy_step != source.policy_step + 1
            ):
                raise ValueError("a no-op transition must advance exactly one step")
            try:
                values = await asyncio.wait_for(
                    self._actors.advance_without_training.call(
                        source.model_dump_json(),
                        output.model_dump_json(),
                        optimizer_state_path,
                        None if adapter is None else adapter.model_dump_json(),
                    ),
                    timeout=self.run_spec.event_timeout_s,
                )
                results = list(values.values())
                if {result["rank"] for result in results} != set(
                    range(len(self.runtime_spec.trainer_mesh.ranks))
                ) or {result["learner_version"] for result in results} != {
                    output.policy_step
                }:
                    raise RuntimeError("trainer ranks rejected no-op transition")
            except BaseException as exc:
                self._valid = False
                self._closed = True
                await cleanup_after_failure(
                    exc,
                    self._force_stop,
                    message="no-op transition and trainer cleanup failed",
                )
                raise
            self._learner_version = output.policy_step
            return next(result["metrics"] for result in results if result["rank"] == 0)

    def _validate_resident_learner(
        self,
        *,
        run_id: str,
        learner: TrainerGeneration,
    ) -> BaseException | None:
        if self._closed or not self._valid:
            return RuntimeError("trainer runtime is invalid")
        if self._active_job_id is not None:
            return RuntimeError("trainer has an active job")
        if any(
            state.open_forward_backward_ids for state in self._command_runs.values()
        ):
            return RuntimeError("diagnostics cannot run with open gradients")
        state = self._command_runs.get(run_id)
        if state is None:
            return ValueError(
                "diagnostic run_id is not registered on this trainer slot"
            )
        if learner.training_session_id != state.spec.training_session_id:
            return ValueError("diagnostic learner does not match the training session")
        if learner.policy_step != state.learner_version:
            return ValueError(
                "diagnostic learner version mismatch: "
                f"request={learner.policy_step}, runtime={state.learner_version}"
            )
        return None

    def _validate_resident_score(
        self,
        job: ResidentScoreJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> BaseException | None:
        if error := self._validate_resident_learner(
            run_id=job.run_id,
            learner=job.learner,
        ):
            return error
        if batch.ref != job.batch:
            return ValueError("resident score batch ref does not match its leases")
        if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
            return ValueError(
                "resident score batch length does not match the trainer runtime"
            )
        if bool(job.batch.moe_routing_replay) != bool(
            self.runtime_spec.enable_moe_routing_replay
        ):
            return ValueError(
                "resident score routing replay does not match the trainer runtime"
            )
        return None

    def _validate_resident_inspection(
        self,
        request: ResidentLoraInspectionSpec,
    ) -> BaseException | None:
        if error := self._validate_resident_learner(
            run_id=request.run_id,
            learner=request.learner,
        ):
            return error
        if request.target_modules != self.runtime_spec.lora_target_modules:
            return ValueError("resident LoRA targets do not match the trainer runtime")
        return None

    def _validate_common(self, job: TrainerJobSpec) -> BaseException | None:
        if self._closed:
            return RuntimeError("trainer run is closed")
        if not self._valid:
            return RuntimeError("trainer runtime is invalid")
        if self._command_mode:
            return RuntimeError("fused jobs cannot be mixed with command operations")
        if job.job_id in self._jobs:
            return RuntimeError("job_id was already used with a different job")
        if job.run_id != self.run_spec.run_id:
            return ValueError("job run_id does not match this training run")
        if job.training_session_id != self.run_spec.training_session_id:
            return ValueError(
                "job training_session_id does not match this training run"
            )
        if job.output.optimizer_state_path != self.run_spec.optimizer_state_path:
            return ValueError(
                "job optimizer state path does not match this training run"
            )
        if job.expected_learner_version != self._learner_version:
            return ValueError(
                "expected learner version mismatch: "
                f"job={job.expected_learner_version}, runtime={self._learner_version}"
            )
        return None

    def _validate_rl(
        self, job: TrainJobSpec, batch: PackedBatchLeaseSet
    ) -> BaseException | None:
        if error := self._validate_common(job):
            return error
        if batch.ref != job.batch:
            return ValueError("job batch ref does not match supplied packed batch")
        if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
            return ValueError(
                "packed batch sequence length does not match the trainer runtime"
            )
        return None

    def _validate_sft(
        self, job: SFTJobSpec, batches: tuple[SFTBatchData, ...]
    ) -> BaseException | None:
        if error := self._validate_common(job):
            return error
        if len(batches) != job.num_batches:
            return ValueError("SFT job batch count does not match its payload")
        return None

    @staticmethod
    def _failed(
        job: TrainerJobSpec,
        sequence: int,
        exc: BaseException,
        invalidated: bool,
    ) -> TrainFailed:
        return TrainFailed(
            job_id=job.job_id,
            run_id=job.run_id,
            sequence=sequence,
            error_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
            runtime_invalidated=invalidated,
        )

    async def close(self) -> None:
        if self._close_task is not None and self._close_task.done():
            try:
                self._close_task.result()
            except BaseException:
                self._close_task = None
        if self._close_task is None:
            graceful = self._valid and self._active_job_id is None
            self._closed = True
            self._valid = False
            self._cancel_active()
            self._close_task = asyncio.create_task(self._close(graceful))
            self._close_task.add_done_callback(consume_future_exception)
        await asyncio.shield(self._close_task)

    async def _close(self, graceful: bool) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.run_spec.shutdown_timeout_s
        primary: BaseException | None = None
        if graceful:
            publications = tuple(self._publications.values())
            try:
                async with asyncio.timeout(self.run_spec.shutdown_timeout_s / 2):
                    await asyncio.gather(
                        _remote_teardown(self._actors.close.call()),
                        *(
                            self._await_publication(publication)
                            for publication in publications
                        ),
                        *tuple(self._publication_drains),
                    )
            except BaseException as exc:
                primary = exc
        try:
            await self._force_stop(max(0.0, deadline - loop.time()))
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(
                    f"trainer ProcMesh cleanup failed: {type(exc).__name__}: {exc}"
                )
        if primary is not None:
            raise primary

    async def _force_stop(self, timeout_s: float | None = None) -> None:
        if self._stop_task is not None and self._stop_task.done():
            try:
                self._stop_task.result()
            except BaseException:
                self._stop_task = None
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(
                _remote_teardown(self._proc_mesh.stop())
            )

            def stopped(task: asyncio.Task[None]) -> None:
                if not task.cancelled() and task.exception() is None:
                    self._supervision.close()

            self._stop_task.add_done_callback(stopped)
        await asyncio.wait_for(
            asyncio.shield(self._stop_task),
            self.run_spec.shutdown_timeout_s if timeout_s is None else timeout_s,
        )

    def _cancel_active(self) -> None:
        # Monarch 0.2 only cancels these local waiters; ProcMesh.stop invalidates ranks.
        for future in (self._active_receive, self._active_collective):
            if future is not None and not future.done():
                future.cancel()
            if future is not None:
                future.add_done_callback(consume_future_exception)

    def _clear_active(self, job_id: str) -> None:
        if self._active_job_id == job_id:
            self._active_job_id = None
            self._active_collective = None
            self._active_receive = None


async def _remote_teardown(operation: Awaitable[Any]) -> None:
    try:
        await operation
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise


def _current_task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and bool(task.cancelling())
