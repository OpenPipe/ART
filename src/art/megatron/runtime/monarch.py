from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from threading import Event, Lock, Thread, current_thread
import time
import traceback
from typing import Any, Callable
import uuid

import monarch.actor as monarch_actor
from monarch.actor import (
    Actor,
    Channel,
    MeshFailure,
    Port,
    ProcMesh,
    endpoint,
)
from monarch.spmd import SPMDActor
from pydantic import BaseModel, ConfigDict, SkipValidation

from art.distributed.data_plane import PackedBatchLeaseSet, SftBatchLeaseSet
from art.distributed.monarch_bootstrap import activate_cuda_device
from art.distributed.specs import GpuId
from art.training.contracts import OperationRef
from art.utils.cache_dirs import configure_model_cache_env
from art.utils.lifecycle import (
    cleanup_after_failure,
    consume_future_exception,
    process_shutdown_timeout,
)

from .data_plane import InMemoryPackedBatch, SFTBatchData
from .preschedule_trace import (
    start_preschedule_watchdog,
    stop_preschedule_watchdog,
    trace_preschedule,
)
from .publication import (
    TRAINER_PUBLICATION_EVENT_ADAPTER,
    SnapshotRankWritePlan,
    SnapshotWriteGrant,
    SnapshotWritePlan,
    TrainerPublicationEvent,
    TrainerPublicationFailed,
    TrainerPublicationProgress,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
    build_snapshot_write_plan,
)
from .specs import (
    TRAIN_EVENT_ADAPTER,
    AdapterReady,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    GenerationSnapshotJobSpec,
    HybridEpRuntimeSpec,
    KlReferenceAcquisition,
    KlReferenceSpec,
    LoadStateJobSpec,
    OptimizerJobSpec,
    RankLocalOptimizerWorkSummary,
    ResidentLoraExport,
    ResidentLoraInspectionResult,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentLoraRankSummary,
    ResidentScoreJobSpec,
    ResidentScoreResult,
    ResidentScoreShard,
    RunOptimizerWorkSummary,
    RunSlotRegistration,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    SFTJobSpec,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerGeneration,
    TrainerJobSpec,
    TrainerRuntimeSpec,
    TrainEvent,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)


def _coordinator_command_result(
    results: list[dict[str, Any]],
    *,
    expected_token_count: int,
    aggregate_telemetry: bool,
) -> dict[str, Any]:
    result = dict(next(item for item in results if item["rank"] == 0))
    payloads = [item["_rank_telemetry"] for item in results]
    if aggregate_telemetry:
        from ..training.command_telemetry import aggregate_rank_command_telemetry

        if any(payload is None for payload in payloads):
            raise RuntimeError("trainer rank omitted required command telemetry")
        result["metrics"] = aggregate_rank_command_telemetry(
            payloads,
            expected_token_count=expected_token_count,
        )
    elif any(payload is not None for payload in payloads):
        raise RuntimeError("trainer rank returned unexpected command telemetry")
    result.pop("_rank_telemetry")
    return result


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
_EXPECTED_STOPPED_MESH_FAILURES: dict[str, float] = {}
_PREVIOUS_FAULT_HOOK: Callable[[MeshFailure], None] | None = None
_MAX_PENDING_RUN_CLEANUPS = 8


def _response_exception(error: BaseException) -> Exception:
    return (
        error
        if isinstance(error, Exception)
        else RuntimeError(f"{type(error).__name__}: {error}")
    )


def _rank_process_group_is_initialized() -> bool:
    import torch

    return torch.distributed.is_initialized()


def _destroy_rank_process_group() -> None:
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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


def _build_training_runtime(spec: TrainerRuntimeSpec, *, rank: int) -> Any:
    import torch

    from art.megatron.train import build_training_runtime

    residency = spec.run_residency
    if residency is not None:
        residency = residency.model_copy(
            update={
                "nvme": residency.nvme.model_copy(
                    update={"root": str(Path(residency.nvme.root) / f"rank-{rank}")}
                )
            }
        )
    return build_training_runtime(
        model_identifier=spec.model_identifier,
        model_initialization=spec.model_initialization,
        provider_torch_dtype={
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[spec.dtype],
        print_env=rank == 0,
        model_support_key=spec.model_support_key,
        snapshot_pool_capacity=spec.snapshot_pool_capacity,
        run_residency_config=residency,
        optimizer_layout_fingerprint=spec.optimizer_layout_fingerprint,
        optimizer_semantic_sha256=spec.optimizer_semantic_fingerprint,
        provider_configure=lambda provider: setattr(
            provider,
            "_art_lora_moe_parameterization",
            spec.lora_moe_parameterization,
        ),
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


class _ResidencyPrefetchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    operation_id: str | None = None
    run_id: str
    command_kind: str
    learner_version: int
    kl_reference_checkpoint_id: str | None = None
    admitted: bool = True
    elapsed_s: float = 0.0
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


class _CommandReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    operation_id: str
    learner_version: int
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


class _SnapshotRankPhase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    operation_id: str
    phase: str
    thread_name: str
    cuda_device: int
    monotonic_ns: int
    error: str | None = None


class _KlReferenceReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    run_id: str
    checkpoint_id: str
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


class _RunRegistrationReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    run_id: str
    optimizer_work: RankLocalOptimizerWorkSummary | None = None
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


async def _prepare_run_residency(
    ports: tuple[Port[Any], ...],
    lock: asyncio.Lock,
    operation_id: str | None,
    run_id: str,
    command_kind: str,
    learner_version: int,
    kl_reference_checkpoint_id: str | None,
    *,
    timeout_s: float,
) -> dict[str, float]:
    async with lock:
        reply, receiver = Channel.open()
        request = (
            operation_id,
            run_id,
            command_kind,
            learner_version,
            kl_reference_checkpoint_id,
            reply,
        )
        started = time.perf_counter()
        for port in ports:
            port.send(request)
        async with asyncio.timeout(timeout_s):
            results = [
                _ResidencyPrefetchResult.model_validate(await receiver.recv())
                for _ in ports
            ]
        if {result.rank for result in results} != set(range(len(ports))) or any(
            (
                result.run_id,
                result.operation_id,
                result.command_kind,
                result.learner_version,
                result.kl_reference_checkpoint_id,
            )
            != (
                run_id,
                operation_id,
                command_kind,
                learner_version,
                kl_reference_checkpoint_id,
            )
            for result in results
        ):
            raise RuntimeError("residency prefetch returned mismatched rank identity")
        failures = [result for result in results if result.error_type is not None]
        if failures:
            details = "\n".join(
                f"rank {result.rank}: {result.error_type}: {result.message}\n"
                f"{result.traceback_text or ''}"
                for result in failures
            )
            raise RuntimeError(f"residency prefetch failed:\n{details}")
        return {
            "time/residency_prefetch_wait_s": time.perf_counter() - started,
            "time/residency_prefetch_rank_max_s": max(
                result.elapsed_s for result in results
            ),
            "residency/prefetch_admitted": float(
                all(result.admitted for result in results)
            ),
        }


async def _prepare_cp_lookahead(
    ports: tuple[Port[Any], ...],
    lock: asyncio.Lock,
    batch: PackedBatchLeaseSet,
    *,
    global_grad_accumulation_sequences: int | None,
    timeout_s: float,
) -> dict[str, float]:
    if not ports:
        return {}
    async with lock:
        reply, receiver = Channel.open()
        request = (
            batch.model_dump_json(),
            batch.ref.batch_id,
            global_grad_accumulation_sequences,
            reply,
        )
        started = time.perf_counter()
        for port in ports:
            port.send(request)
        async with asyncio.timeout(timeout_s):
            results = [
                _CpLookaheadResult.model_validate(await receiver.recv()) for _ in ports
            ]
        if {result.rank for result in results} != set(range(len(ports))) or any(
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


def _dispatch_trainer_fault(failure: MeshFailure) -> None:
    message = str(failure)
    with _SUPERVISION_LOCK:
        now = time.monotonic()
        for mesh_name, deadline in tuple(_EXPECTED_STOPPED_MESH_FAILURES.items()):
            if deadline <= now:
                _EXPECTED_STOPPED_MESH_FAILURES.pop(mesh_name)
        expected_stop = failure.mesh_name in _EXPECTED_STOPPED_MESH_FAILURES
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
        _restore_fault_hook_locked()
    if expected_stop:
        return
    if handlers:
        for handler in handlers:
            handler.notify(message)
        return
    if previous is not None:
        previous(failure)


def _restore_fault_hook_locked() -> None:
    global _PREVIOUS_FAULT_HOOK
    if _SUPERVISION_HANDLERS or _EXPECTED_STOPPED_MESH_FAILURES:
        return
    if monarch_actor.unhandled_fault_hook is _dispatch_trainer_fault:
        assert _PREVIOUS_FAULT_HOOK is not None
        setattr(monarch_actor, "unhandled_fault_hook", _PREVIOUS_FAULT_HOOK)
    _PREVIOUS_FAULT_HOOK = None


def _expire_expected_stopped_mesh_failures() -> None:
    with _SUPERVISION_LOCK:
        now = time.monotonic()
        for mesh_name, deadline in tuple(_EXPECTED_STOPPED_MESH_FAILURES.items()):
            if deadline <= now:
                _EXPECTED_STOPPED_MESH_FAILURES.pop(mesh_name)
        _restore_fault_hook_locked()


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
            if monarch_actor.unhandled_fault_hook is not _dispatch_trainer_fault:
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

    def close(self, *, suppress_owned_mesh_faults_s: float = 0.0) -> None:
        if suppress_owned_mesh_faults_s < 0:
            raise ValueError("mesh-fault suppression duration must be non-negative")
        with _SUPERVISION_LOCK:
            if self._closed:
                return
            self._closed = True
            if _SUPERVISION_HANDLERS.get(self.token) is self:
                _SUPERVISION_HANDLERS.pop(self.token)
            for mesh_name in self._mesh_names:
                if _SUPERVISION_MESHES.get(mesh_name) is self:
                    _SUPERVISION_MESHES.pop(mesh_name)
                if suppress_owned_mesh_faults_s:
                    _EXPECTED_STOPPED_MESH_FAILURES[mesh_name] = (
                        time.monotonic() + suppress_owned_mesh_faults_s
                    )
            _restore_fault_hook_locked()
        if suppress_owned_mesh_faults_s:
            self._loop.call_later(
                suppress_owned_mesh_faults_s,
                _expire_expected_stopped_mesh_failures,
            )


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
    ) -> None:
        self._teardown_lock = Lock()
        self._teardown_complete = False
        self._teardown_poisoned = False
        self._shutdown_timeout_s = process_shutdown_timeout(2)
        self._valid = False
        self._executor: Any | None = None
        self._run_slot_executor: Any | None = None
        self._weight_offload: Any | None = None
        self._command_job_open = False
        self._deferred_response_lock = Lock()
        self._deferred_response_threads: set[Thread] = set()
        self._deferred_response_stopping = False
        self._snapshot_phase_lock = Lock()
        self._snapshot_phases: dict[str, _SnapshotRankPhase] = {}
        self._cp_preplanner = None
        self._cp_lookahead_port = None
        self._cp_lookahead_thread: Thread | None = None
        self._residency_prefetch_port = None
        self._residency_prefetch_thread: Thread | None = None
        runtime_spec = TrainerRuntimeSpec.model_validate_json(runtime_spec_json)
        topology = runtime_spec.trainer_mesh.topology
        cache_root = configure_model_cache_env(cache_root=runtime_spec.cache_root)
        os.environ.update(
            {
                "MODEL_IDENTIFIER": runtime_spec.model_identifier,
                "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE": str(topology.tp),
                "ART_MEGATRON_CONTEXT_PARALLEL_SIZE": str(topology.cp),
                "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE": str(topology.ep),
                "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE": str(topology.pp),
                "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE": str(topology.etp),
                "ART_MEGATRON_LORA_RANK": str(runtime_spec.lora_rank),
                "ART_MEGATRON_LORA_TARGET_MODULES": json.dumps(
                    runtime_spec.lora_target_modules
                ),
                "ART_MEGATRON_LORA_MOE_PARAMETERIZATION": (
                    runtime_spec.lora_moe_parameterization
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
        self._runtime = _build_training_runtime(runtime_spec, rank=rank)
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

        self._executor = MegatronTrainJobExecutor(self._runtime)
        self._run_slot_executor = (
            MCoreRunSlotExecutor(self._runtime)
            if self._runtime.run_residency_config is not None
            else None
        )
        self._weight_offload = WeightOffloadManager.from_config(
            model=self._runtime.model,
            rank=self._runtime.rank,
            compile_enabled=self._runtime.transformer_layers_compiled,
            offload_between_jobs=runtime_spec.offload_between_jobs,
            streaming_config=streaming_weight_offload_config_from_env(),
        )
        self._weight_offload.install()
        self._compile_cache_publish_lock = Lock()
        if self._run_slot_executor is not None:
            self._residency_prefetch_port, receiver = Channel.open()
            self._residency_prefetch_thread = Thread(
                target=self._run_residency_prefetch,
                args=(receiver,),
                name=f"art-residency-prefetch-rank-{rank}",
                daemon=True,
            )
            self._residency_prefetch_thread.start()
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

    def _require_run_slot_executor(self) -> Any:
        if self._run_slot_executor is None:
            raise RuntimeError("trainer actor is not configured for multi-run slots")
        return self._run_slot_executor

    def _record_snapshot_phase(
        self, operation_id: str, phase: str, *, error: BaseException | None = None
    ) -> None:
        import torch

        record = _SnapshotRankPhase(
            rank=self._runtime.rank,
            operation_id=operation_id,
            phase=phase,
            thread_name=current_thread().name,
            cuda_device=torch.cuda.current_device(),
            monotonic_ns=time.monotonic_ns(),
            error=(None if error is None else f"{type(error).__name__}: {error}"),
        )
        if os.environ.get("ART_DEBUG_SNAPSHOT_RANK_PHASES") == "1":
            print(f"ART_SNAPSHOT_RANK_PHASE {record.model_dump_json()}", flush=True)
        with self._snapshot_phase_lock:
            if operation_id not in self._snapshot_phases:
                while len(self._snapshot_phases) >= 16:
                    self._snapshot_phases.pop(next(iter(self._snapshot_phases)))
            self._snapshot_phases[operation_id] = record

    def _snapshot_phase_report(self, operation_id: str) -> dict[str, Any]:
        with self._snapshot_phase_lock:
            phase = self._snapshot_phases.get(operation_id)
        thread_name = f"art-snapshot-prepare-{operation_id}"
        with self._deferred_response_lock:
            thread = next(
                (
                    candidate
                    for candidate in self._deferred_response_threads
                    if candidate.name == thread_name
                ),
                None,
            )
        frame = (
            sys._current_frames().get(thread.ident)
            if thread is not None and thread.ident is not None
            else None
        )
        return {
            "rank": self._runtime.rank,
            "operation_id": operation_id,
            "expected_cuda_device": int(os.environ["LOCAL_RANK"]),
            "phase": None if phase is None else phase.model_dump(mode="json"),
            "thread_alive": thread is not None and thread.is_alive(),
            "thread_stack": (
                None
                if frame is None
                else "".join(traceback.format_stack(frame, limit=32))
            ),
        }

    @endpoint
    def inspect_snapshot_phase(self, operation_id: str) -> dict[str, Any]:
        return self._snapshot_phase_report(operation_id)

    def _defer_response(
        self,
        port: Port[dict[str, Any]],
        materialize: Callable[[], dict[str, Any]],
        *,
        name: str,
        invalidate_on_error: bool,
    ) -> None:
        import torch

        cuda_device = torch.cuda.current_device()
        with self._deferred_response_lock:
            if self._deferred_response_stopping:
                raise RuntimeError("trainer actor is stopping")

            def settle() -> None:
                try:
                    with torch.cuda.device(cuda_device):
                        port.send(materialize())
                except BaseException as error:
                    if invalidate_on_error:
                        self._valid = False
                    try:
                        port.exception(_response_exception(error))
                    except BaseException:
                        pass
                finally:
                    with self._deferred_response_lock:
                        self._deferred_response_threads.discard(thread)

            thread = Thread(target=settle, name=name, daemon=True)
            self._deferred_response_threads.add(thread)
            thread.start()

    def _stop_deferred_responses(self, deadline: float) -> None:
        with self._deferred_response_lock:
            self._deferred_response_stopping = True
            threads = tuple(self._deferred_response_threads)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError(
                "trainer response settlement did not stop within "
                f"the shutdown deadline: {alive}"
            )

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

    def _run_residency_prefetch(self, receiver: Any) -> None:
        while (request := receiver.recv().get()) is not None:
            (
                operation_id,
                run_id,
                command_kind,
                learner_version,
                kl_reference_checkpoint_id,
                reply,
            ) = request
            started = time.perf_counter()
            trace_id = operation_id or f"prefetch:{run_id}:{learner_version}"
            trace_preschedule(
                self._runtime.rank,
                trace_id,
                "residency_request_enter",
                command_kind=command_kind,
                admitted=operation_id is not None,
            )
            try:
                executor = self._require_run_slot_executor()
                admitted = (
                    executor.prefetch_residency(
                        run_id,
                        command_kind,
                        learner_version,
                        kl_reference_checkpoint_id,
                    )
                    if operation_id is None
                    else executor.admit_residency(
                        operation_id,
                        run_id,
                        command_kind,
                        learner_version,
                        kl_reference_checkpoint_id,
                    )
                )
                result = _ResidencyPrefetchResult(
                    rank=self._runtime.rank,
                    operation_id=operation_id,
                    run_id=run_id,
                    command_kind=command_kind,
                    learner_version=learner_version,
                    kl_reference_checkpoint_id=kl_reference_checkpoint_id,
                    admitted=admitted,
                    elapsed_s=time.perf_counter() - started,
                )
            except BaseException as error:
                result = _ResidencyPrefetchResult(
                    rank=self._runtime.rank,
                    operation_id=operation_id,
                    run_id=run_id,
                    command_kind=command_kind,
                    learner_version=learner_version,
                    kl_reference_checkpoint_id=kl_reference_checkpoint_id,
                    elapsed_s=time.perf_counter() - started,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
            trace_preschedule(
                self._runtime.rank,
                trace_id,
                "residency_request_exit",
                command_kind=command_kind,
                admitted=result.admitted,
                error_type=result.error_type,
                elapsed_s=result.elapsed_s,
            )
            reply.send(result.model_dump(mode="json"))

    def _stop_cp_lookahead(self, deadline: float) -> None:
        thread = self._cp_lookahead_thread
        if thread is None:
            return
        port = self._cp_lookahead_port
        if port is None:
            raise RuntimeError("CP lookahead thread has no request port")
        port.send(None)
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise RuntimeError("CP lookahead service exceeded shutdown deadline")
        self._cp_lookahead_thread = None

    def _stop_residency_prefetch(self, deadline: float) -> None:
        thread = self._residency_prefetch_thread
        if thread is None:
            return
        port = self._residency_prefetch_port
        if port is None:
            raise RuntimeError("residency prefetch thread has no request port")
        port.send(None)
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise RuntimeError("residency prefetch service exceeded shutdown deadline")
        self._residency_prefetch_thread = None

    def _publish_compile_cache(self) -> None:
        with self._compile_cache_publish_lock:
            if self._compile_cache is None:
                return
            event = self._compile_cache.publish()
            self._compile_cache_metrics.update(
                {
                    "publish_s": self._compile_cache_metrics.get("publish_s", 0.0)
                    + event.elapsed_s,
                    "published": max(
                        self._compile_cache_metrics.get("published", 0.0),
                        float(event.status == "published"),
                    ),
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
    def residency_prefetch_port(self) -> dict[str, Any] | None:
        if self._residency_prefetch_port is None:
            return None
        return {"rank": self._runtime.rank, "port": self._residency_prefetch_port}

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

    @endpoint
    def execute_forward_backward(
        self,
        job_json: str,
        batch_json: str,
    ) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardBackwardJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            result = self._executor.execute_forward_backward(job, batch, Event())
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def start_forward_backward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        ready = False
        job = None
        try:
            job = ForwardBackwardJobSpec.model_validate_json(job_json)
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            launch = self._executor.start_forward_backward(job, batch, Event())
            batch.close()
            batch = None
            ready_port.send(
                _CommandReady(
                    rank=self._runtime.rank,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                ).model_dump(mode="json")
            )
            ready = True

            def materialize() -> dict[str, Any]:
                result = launch.materialize()
                self._publish_compile_cache()
                coordinator = self._runtime.rank == 0
                return {
                    "rank": self._runtime.rank,
                    "operation_id": job.operation_id,
                    "learner_version": job.expected_learner_version,
                    "token_count": result["token_count"],
                    "metrics": result["metrics"] if coordinator else {},
                    "_rank_telemetry": result["_rank_telemetry"],
                    "token_logprobs": (result["token_logprobs"] if coordinator else ()),
                }

            self._defer_response(
                response_port,
                materialize,
                name=f"art-fb-result-{job.operation_id}",
                invalidate_on_error=True,
            )
        except BaseException as error:
            self._valid = False
            if not ready and job is not None:
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
            response_port.exception(_response_exception(error))
        finally:
            if batch is not None:
                batch.close()

    @endpoint
    def execute_forward(self, job_json: str, batch_json: str) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            result = self._executor.execute_forward(job, batch, Event())
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "metrics": result["metrics"] if coordinator else {},
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint
    def execute_sft_forward_backward(
        self,
        job_json: str,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = SftForwardBackwardJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            result = self._executor.execute_sft_forward_backward(job, batch, Event())
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def execute_sft_forward(
        self,
        job_json: str,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = SftForwardJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            result = self._executor.execute_sft_forward(job, batch, Event())
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def execute_optimizer(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = OptimizerJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                raise RuntimeError("optimizer has no resident F/B command interval")
            result = self._executor.execute_optimizer(job)
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "contributing_forward_backward_operation_ids": result[
                    "contributing_forward_backward_operation_ids"
                ],
                "metrics": result["metrics"] if coordinator else {},
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def execute_load_state(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = LoadStateJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            return {
                "rank": self._runtime.rank,
                **self._executor.execute_load_state(job),
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def start_prepare_run_slot_registration(
        self,
        registration_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        registration = RunSlotRegistration.model_validate_json(registration_json)
        future = self._require_run_slot_executor().start_prepare_run_registration(
            registration
        )

        def ready(completed: Any) -> None:
            try:
                prepared = completed.result()
                result = _RunRegistrationReady(
                    rank=self._runtime.rank,
                    run_id=registration.run_id,
                    optimizer_work=prepared.optimizer_work,
                )
            except BaseException as error:
                result = _RunRegistrationReady(
                    rank=self._runtime.rank,
                    run_id=registration.run_id,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback_text="".join(traceback.format_exception(error)),
                )
            ready_port.send(result.model_dump(mode="json"))

        future.add_done_callback(ready)
        return {"rank": self._runtime.rank, "run_id": registration.run_id}

    @endpoint
    def finish_prepare_run_slot_registration(
        self, registration_json: str
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        registration = RunSlotRegistration.model_validate_json(registration_json)
        executor = self._require_run_slot_executor()
        executor.finish_prepared_run_registration(registration)
        executor.complete_run_registration(registration.run_id)
        return {"rank": self._runtime.rank, "run_id": registration.run_id}

    @endpoint
    def discard_run_slot_registration(self, run_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        self._require_run_slot_executor().discard_prepared_run_registration(run_id)
        return {"rank": self._runtime.rank, "run_id": run_id}

    @endpoint
    def start_unregister_run_slot(self, run_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        self._require_run_slot_executor().start_unregister_run(run_id)
        return {"rank": self._runtime.rank, "run_id": run_id}

    @endpoint
    def release_run_slot_residency_admission(self, operation_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        self._require_run_slot_executor().release_residency_admission(operation_id)
        return {"rank": self._runtime.rank, "operation_id": operation_id}

    @endpoint(explicit_response_port=True)
    def finish_unregister_run_slot(
        self, response_port: Port[dict[str, Any]], run_id: str
    ) -> None:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            executor = self._require_run_slot_executor()

            def finish() -> dict[str, Any]:
                executor.finish_unregister_run(run_id)
                return {"rank": self._runtime.rank, "run_id": run_id}

            self._defer_response(
                response_port,
                finish,
                name=f"art-unregister-run-{run_id}",
                invalidate_on_error=False,
            )
        except BaseException as error:
            response_port.exception(_response_exception(error))

    @endpoint
    def execute_run_slot_forward_backward(
        self,
        job_json: str,
        batch_json: str,
    ) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardBackwardJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            launch = self._require_run_slot_executor().start_forward_backward(
                job,
                batch,
                Event(),
                coordinator=self._runtime.rank == 0,
            )
            result = launch.materialize()
            response = {
                "rank": self._runtime.rank,
                "operation_id": result["operation_id"],
                "learner_version": result["learner_version"],
                "token_count": result["token_count"],
                "metrics": result["metrics"],
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"],
            }
            self._publish_compile_cache()
            return response
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def start_run_slot_forward_backward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        ready = False
        job = None
        try:
            job = ForwardBackwardJobSpec.model_validate_json(job_json)
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            launch = self._require_run_slot_executor().start_forward_backward(
                job,
                batch,
                Event(),
                coordinator=self._runtime.rank == 0,
            )
            batch.close()
            batch = None
            ready_port.send(
                _CommandReady(
                    rank=self._runtime.rank,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                ).model_dump(mode="json")
            )
            ready = True

            def materialize() -> dict[str, Any]:
                result = launch.materialize()
                self._publish_compile_cache()
                return {
                    "rank": self._runtime.rank,
                    "operation_id": result["operation_id"],
                    "learner_version": result["learner_version"],
                    "token_count": result["token_count"],
                    "metrics": result["metrics"],
                    "_rank_telemetry": result["_rank_telemetry"],
                    "token_logprobs": result["token_logprobs"],
                }

            self._defer_response(
                response_port,
                materialize,
                name=f"art-fb-result-{job.operation_id}",
                invalidate_on_error=True,
            )
        except BaseException as error:
            self._valid = False
            if not ready and job is not None:
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
            response_port.exception(_response_exception(error))
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def start_run_slot_forward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        ready = False
        job = None
        watchdog = None
        try:
            job = ForwardJobSpec.model_validate_json(job_json)
            rank = self._runtime.rank
            watchdog = start_preschedule_watchdog(rank, job.operation_id)
            trace_preschedule(rank, job.operation_id, "endpoint_enter")
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            trace_preschedule(rank, job.operation_id, "lease_validated")
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            tensors = batch.tensors
            replay = tensors.get("moe_routing_replay")
            trace_preschedule(
                rank,
                job.operation_id,
                "batch_opened",
                batch_id=job.batch.batch_id,
                token_shape=tuple(tensors["tokens"].shape),
                replay_shape=(
                    None if replay is None else tuple(replay.expert_indices.shape)
                ),
            )
            if not self._command_job_open:
                trace_preschedule(rank, job.operation_id, "before_job_enter")
                self._weight_offload.before_job()
                self._command_job_open = True
                trace_preschedule(rank, job.operation_id, "before_job_exit")
            trace_preschedule(rank, job.operation_id, "executor_enter")
            launch = self._require_run_slot_executor().start_forward(
                job,
                batch,
                Event(),
                coordinator=self._runtime.rank == 0,
            )
            trace_preschedule(rank, job.operation_id, "executor_exit")
            batch.close()
            batch = None
            ready_port.send(
                _CommandReady(
                    rank=self._runtime.rank,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                ).model_dump(mode="json")
            )
            ready = True

            def materialize() -> dict[str, Any]:
                result = {"rank": self._runtime.rank, **launch.materialize()}
                self._publish_compile_cache()
                return result

            self._defer_response(
                response_port,
                materialize,
                name=f"art-forward-result-{job.operation_id}",
                invalidate_on_error=True,
            )
        except BaseException as error:
            self._valid = False
            if not ready and job is not None:
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
            response_port.exception(_response_exception(error))
        finally:
            stop_preschedule_watchdog(watchdog)
            if batch is not None:
                batch.close()

    @endpoint
    def execute_run_slot_forward(
        self,
        job_json: str,
        batch_json: str,
    ) -> dict[str, Any]:
        batch = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = ForwardJobSpec.model_validate_json(job_json)
            leases = PackedBatchLeaseSet.model_validate_json(batch_json)
            batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            result = self._require_run_slot_executor().execute_forward(
                job, batch, Event()
            )
            coordinator = self._runtime.rank == 0
            response = {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "metrics": result["metrics"] if coordinator else {},
                "_rank_telemetry": result["_rank_telemetry"],
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
            self._publish_compile_cache()
            return response
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def start_run_slot_sft_forward_backward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        ready = False
        job = None
        try:
            job = SftForwardBackwardJobSpec.model_validate_json(job_json)
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            leases = SftBatchLeaseSet.model_validate_json(batch_json)
            if leases.manifest.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT F/B lease differs from its job")
            batch = SFTBatchData.open(leases.manifest, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            launch = self._require_run_slot_executor().start_sft_forward_backward(
                job,
                batch,
                Event(),
                coordinator=self._runtime.rank == 0,
            )
            batch.close()
            batch = None
            ready_port.send(
                _CommandReady(
                    rank=self._runtime.rank,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                ).model_dump(mode="json")
            )
            ready = True

            def materialize() -> dict[str, Any]:
                result = {"rank": self._runtime.rank, **launch.materialize()}
                self._publish_compile_cache()
                return result

            self._defer_response(
                response_port,
                materialize,
                name=f"art-sft-fb-result-{job.operation_id}",
                invalidate_on_error=True,
            )
        except BaseException as error:
            self._valid = False
            if not ready and job is not None:
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
            response_port.exception(_response_exception(error))
        finally:
            if batch is not None:
                batch.close()

    @endpoint(explicit_response_port=True)
    def start_run_slot_sft_forward(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        batch_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> None:
        batch = None
        ready = False
        job = None
        try:
            job = SftForwardJobSpec.model_validate_json(job_json)
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            leases = SftBatchLeaseSet.model_validate_json(batch_json)
            if leases.manifest.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT forward lease differs from its job")
            batch = SFTBatchData.open(leases.manifest, leases.host_refs[self._host_id])
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            launch = self._require_run_slot_executor().start_sft_forward(
                job,
                batch,
                Event(),
                coordinator=self._runtime.rank == 0,
            )
            batch.close()
            batch = None
            ready_port.send(
                _CommandReady(
                    rank=self._runtime.rank,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                ).model_dump(mode="json")
            )
            ready = True

            def materialize() -> dict[str, Any]:
                result = {"rank": self._runtime.rank, **launch.materialize()}
                self._publish_compile_cache()
                return result

            self._defer_response(
                response_port,
                materialize,
                name=f"art-sft-forward-result-{job.operation_id}",
                invalidate_on_error=True,
            )
        except BaseException as error:
            self._valid = False
            if not ready and job is not None:
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
            response_port.exception(_response_exception(error))
        finally:
            if batch is not None:
                batch.close()

    @endpoint
    def execute_run_slot_optimizer(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = OptimizerJobSpec.model_validate_json(job_json)
            result = self._require_run_slot_executor().execute_optimizer(job)
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "contributing_forward_backward_operation_ids": result[
                    "contributing_forward_backward_operation_ids"
                ],
                "metrics": result["metrics"] if coordinator else {},
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def start_prepare_run_slot_load_state(
        self,
        job_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = LoadStateJobSpec.model_validate_json(job_json)
            future = self._require_run_slot_executor().start_prepare_load_state(job)

            def ready(completed: Any) -> None:
                try:
                    completed.result()
                    result = _CommandReady(
                        rank=self._runtime.rank,
                        operation_id=job.operation_id,
                        learner_version=job.learner_version,
                    )
                except BaseException as error:
                    result = _CommandReady(
                        rank=self._runtime.rank,
                        operation_id=job.operation_id,
                        learner_version=job.learner_version,
                        error_type=type(error).__name__,
                        message=str(error),
                        traceback_text="".join(traceback.format_exception(error)),
                    )
                ready_port.send(result.model_dump(mode="json"))

            future.add_done_callback(ready)
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def execute_run_slot_load_state(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = LoadStateJobSpec.model_validate_json(job_json)
            result = self._require_run_slot_executor().finish_prepared_load_state(job)
            return {"rank": self._runtime.rank, **result}
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def discard_run_slot_load_state(self, operation_id: str) -> None:
        self._require_run_slot_executor().discard_prepared_load_state(operation_id)

    @endpoint
    def start_prepare_run_slot_kl_reference(
        self,
        spec_json: str,
        ready_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            spec = KlReferenceSpec.model_validate_json(spec_json)
            future = self._require_run_slot_executor().start_prepare_kl_reference(spec)

            def ready(completed: Any) -> None:
                try:
                    completed.result()
                    result = _KlReferenceReady(
                        rank=self._runtime.rank,
                        run_id=spec.run_id,
                        checkpoint_id=spec.checkpoint_id,
                    )
                except BaseException as error:
                    result = _KlReferenceReady(
                        rank=self._runtime.rank,
                        run_id=spec.run_id,
                        checkpoint_id=spec.checkpoint_id,
                        error_type=type(error).__name__,
                        message=str(error),
                        traceback_text="".join(traceback.format_exception(error)),
                    )
                ready_port.send(result.model_dump(mode="json"))

            future.add_done_callback(ready)
            return {
                "rank": self._runtime.rank,
                "run_id": spec.run_id,
                "checkpoint_id": spec.checkpoint_id,
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def acquire_run_slot_kl_reference(
        self, spec_json: str, acquisition_id: str
    ) -> dict[str, Any]:
        spec = KlReferenceSpec.model_validate_json(spec_json)
        return {
            "rank": self._runtime.rank,
            **self._require_run_slot_executor().finish_prepared_kl_reference(
                spec, acquisition_id
            ),
        }

    @endpoint
    def discard_run_slot_kl_reference(self, run_id: str, checkpoint_id: str) -> None:
        self._require_run_slot_executor().discard_prepared_kl_reference(
            run_id, checkpoint_id
        )

    @endpoint
    def abort_run_slot_kl_reference_acquisition(
        self, run_id: str, checkpoint_id: str, acquisition_id: str
    ) -> None:
        self._require_run_slot_executor().abort_kl_reference_acquisition(
            run_id, checkpoint_id, acquisition_id
        )

    @endpoint
    def release_run_slot_kl_reference(
        self, run_id: str, checkpoint_id: str, acquisition_id: str
    ) -> None:
        self._require_run_slot_executor().release_kl_reference(
            run_id, checkpoint_id, acquisition_id
        )

    @endpoint(explicit_response_port=True)
    def execute_run_slot_snapshot(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        event_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
    ) -> None:
        staged = Event()
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = GenerationSnapshotJobSpec.model_validate_json(job_json)
            coordinator = self._runtime.rank == 0
            self._record_snapshot_phase(job.operation_id, "endpoint_validated")

            def mark_staged() -> None:
                self._record_snapshot_phase(
                    job.operation_id, "mutation_fence_complete"
                )
                ready_port.send(
                    _CommandReady(
                        rank=self._runtime.rank,
                        operation_id=job.operation_id,
                        learner_version=job.learner_version,
                    ).model_dump(mode="json")
                )
                staged.set()
                self._record_snapshot_phase(job.operation_id, "readiness_sent")

            def prepare() -> dict[str, Any]:
                try:
                    self._record_snapshot_phase(
                        job.operation_id, "deferred_prepare_started"
                    )
                    self._record_snapshot_phase(job.operation_id, "mutation_fence_enter")
                    result = self._require_run_slot_executor().execute_snapshot(
                        job,
                        _ActorEventSink(event_port, coordinator=coordinator),
                        mark_staged,
                    )
                    self._record_snapshot_phase(job.operation_id, "plan_ready")
                    return {
                        "rank": self._runtime.rank,
                        "operation_id": job.operation_id,
                        "learner_version": job.learner_version,
                        "rank_write_plan": result["rank_write_plan"],
                        "metrics": result["metrics"] if coordinator else {},
                    }
                except BaseException as error:
                    self._record_snapshot_phase(job.operation_id, "failed", error=error)
                    if not staged.is_set():
                        ready_port.send(
                            _CommandReady(
                                rank=self._runtime.rank,
                                operation_id=job.operation_id,
                                learner_version=job.learner_version,
                                error_type=type(error).__name__,
                                message=str(error),
                                traceback_text=traceback.format_exc(),
                            ).model_dump(mode="json")
                        )
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

            self._defer_response(
                response_port,
                prepare,
                name=f"art-snapshot-prepare-{job.operation_id}",
                invalidate_on_error=True,
            )
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
            response_port.exception(_response_exception(error))

    @endpoint
    def authorize_run_slot_snapshot(
        self, plan_json: str, grant_json: str
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            plan = SnapshotWritePlan.model_validate_json(plan_json)
            grant = SnapshotWriteGrant.model_validate_json(grant_json)
            metrics = self._require_run_slot_executor().authorize_snapshot(plan, grant)
            return {
                "rank": self._runtime.rank,
                "operation_id": plan.operation_id,
                "metrics": metrics,
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def discard_run_slot_snapshot(self, operation_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        self._require_run_slot_executor().discard_prepared_snapshot(operation_id)
        return {"rank": self._runtime.rank, "operation_id": operation_id}

    @endpoint(explicit_response_port=True)
    def execute_snapshot(
        self,
        response_port: Port[dict[str, Any]],
        job_json: str,
        event_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
    ) -> None:
        staged = Event()
        job = None
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = GenerationSnapshotJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            coordinator = self._runtime.rank == 0
            self._record_snapshot_phase(job.operation_id, "endpoint_validated")

            def mark_staged() -> None:
                self._record_snapshot_phase(
                    job.operation_id, "mutation_fence_complete"
                )
                ready_port.send(
                    _CommandReady(
                        rank=self._runtime.rank,
                        operation_id=job.operation_id,
                        learner_version=job.learner_version,
                    ).model_dump(mode="json")
                )
                staged.set()
                self._record_snapshot_phase(job.operation_id, "readiness_sent")

            def prepare() -> dict[str, Any]:
                try:
                    self._record_snapshot_phase(
                        job.operation_id, "deferred_prepare_started"
                    )
                    self._record_snapshot_phase(job.operation_id, "mutation_fence_enter")
                    result = self._executor.execute_snapshot(
                        job,
                        _ActorEventSink(event_port, coordinator=coordinator),
                        mark_staged,
                    )
                    self._record_snapshot_phase(job.operation_id, "plan_ready")
                    return {
                        "rank": self._runtime.rank,
                        "operation_id": job.operation_id,
                        "learner_version": job.learner_version,
                        "rank_write_plan": result["rank_write_plan"],
                        "metrics": result["metrics"] if coordinator else {},
                    }
                except BaseException as error:
                    self._record_snapshot_phase(job.operation_id, "failed", error=error)
                    if not staged.is_set():
                        ready_port.send(
                            _CommandReady(
                                rank=self._runtime.rank,
                                operation_id=job.operation_id,
                                learner_version=job.learner_version,
                                error_type=type(error).__name__,
                                message=str(error),
                                traceback_text=traceback.format_exc(),
                            ).model_dump(mode="json")
                        )
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

            self._defer_response(
                response_port,
                prepare,
                name=f"art-snapshot-prepare-{job.operation_id}",
                invalidate_on_error=True,
            )
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
            response_port.exception(_response_exception(error))

    @endpoint
    def authorize_snapshot(self, plan_json: str, grant_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            plan = SnapshotWritePlan.model_validate_json(plan_json)
            grant = SnapshotWriteGrant.model_validate_json(grant_json)
            metrics = self._executor.authorize_snapshot(plan, grant)
            return {
                "rank": self._runtime.rank,
                "operation_id": plan.operation_id,
                "metrics": metrics,
            }
        except BaseException:
            self._valid = False
            raise

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

    def _rank_teardown(self, shutdown_timeout_s: float) -> None:
        with self._teardown_lock:
            if self._teardown_complete:
                return
            self._valid = False
            deadline = time.monotonic() + max(0.0, shutdown_timeout_s)
            failures: list[BaseException] = []

            def attempt(name: str, action: Callable[[], None]) -> None:
                try:
                    action()
                except BaseException as error:
                    error.add_note(f"Megatron rank teardown phase failed: {name}")
                    failures.append(error)

            try:
                attempt(
                    "deferred_responses",
                    lambda: self._stop_deferred_responses(deadline),
                )
                if self._command_job_open:
                    if self._executor is not None:
                        attempt(
                            "discard_open_gradients",
                            self._executor.discard_open_gradients,
                        )
                    if self._weight_offload is not None:
                        attempt(
                            "finish_weight_offload_job",
                            self._weight_offload.after_job,
                        )
                    self._command_job_open = False
                attempt(
                    "residency_prefetch",
                    lambda: self._stop_residency_prefetch(deadline),
                )
                if self._run_slot_executor is not None:
                    attempt(
                        "run_slot_executor",
                        lambda: self._run_slot_executor.close(deadline=deadline),
                    )
                attempt("cp_lookahead", lambda: self._stop_cp_lookahead(deadline))
                if self._executor is not None:
                    attempt(
                        "base_executor",
                        lambda: self._executor.close(deadline=deadline),
                    )
            finally:
                attempt("process_group", _destroy_rank_process_group)

            unsafe = []
            with self._deferred_response_lock:
                if any(thread.is_alive() for thread in self._deferred_response_threads):
                    unsafe.append("deferred responses")
            for name, thread in (
                ("residency prefetch", self._residency_prefetch_thread),
                ("CP lookahead", self._cp_lookahead_thread),
            ):
                if thread is not None and thread.is_alive():
                    unsafe.append(name)
            if (
                self._run_slot_executor is not None
                and not self._run_slot_executor.closed
            ):
                unsafe.append("run-slot executor")
            if self._executor is not None and not self._executor.closed:
                unsafe.append("base executor")
            if _rank_process_group_is_initialized():
                unsafe.append("distributed process group")
            if unsafe:
                failures.append(
                    TimeoutError(
                        "Megatron rank retained unsafe resources after shutdown "
                        f"deadline: {', '.join(unsafe)}"
                    )
                )
            elif time.monotonic() > deadline:
                failures.append(
                    TimeoutError("Megatron rank exceeded shutdown deadline")
                )
            self._teardown_complete = not unsafe
            self._teardown_poisoned = bool(failures)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("Megatron rank teardown failed", failures)

    @endpoint
    def close(self, shutdown_timeout_s: float | None = None) -> None:
        self._rank_teardown(
            self._shutdown_timeout_s
            if shutdown_timeout_s is None
            else shutdown_timeout_s
        )

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
        self._rank_teardown(self._shutdown_timeout_s)


async def spawn_monarch_trainer_actors(
    proc_mesh: ProcMesh,
    runtime_spec: TrainerRuntimeSpec,
    supervision: MonarchTrainerSupervision,
) -> tuple[
    Any,
    tuple[_TrainerRankReady, ...],
    tuple[Port[Any], ...],
    tuple[Port[Any], ...],
]:
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
    residency_values = await actors.residency_prefetch_port.call()
    residency_ports = tuple(
        value["port"]
        for value in sorted(
            (value for value in residency_values.values() if value is not None),
            key=lambda value: value["rank"],
        )
    )
    expected_residency_count = len(placements) if runtime_spec.run_residency else 0
    if len(residency_ports) != expected_residency_count:
        raise RuntimeError(
            "trainer ranks returned an incomplete residency prefetch service"
        )
    return actors, ready, lookahead_ports, residency_ports


class _PublicationState:
    __slots__ = (
        "active_waiters",
        "authorized",
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
        self.authorized = asyncio.Event()
        self.records: dict[int, TrainerRankPublication] = {}
        self.train_done = False
        self.drain_done = True
        self.active_waiters = 0
        self.late_waitable = True
        self.outcome_observed = False


class ForwardBackwardCommandLaunch(BaseModel):
    """All ranks own the gradients; only API result materialization remains."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    completion: SkipValidation[asyncio.Future[dict[str, Any]]]


class ForwardCommandLaunch(BaseModel):
    """All ranks released the GPU turn; API result materialization is pending."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    completion: SkipValidation[asyncio.Future[dict[str, Any]]]


class SnapshotPrepareCommandLaunch(BaseModel):
    """Ranks fenced mutation; the plan is pending and publication is reserved."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    completion: SkipValidation[asyncio.Future[dict[str, Any]]]
    publication: SkipValidation[asyncio.Future[tuple[TrainerRankPublication, ...]]]


async def _collect_command_ready(
    receiver: Any,
    job: (
        ForwardJobSpec
        | ForwardBackwardJobSpec
        | SftForwardJobSpec
        | SftForwardBackwardJobSpec
        | GenerationSnapshotJobSpec
    ),
    rank_processes: tuple[_TrainerRankReady, ...],
    *,
    label: str,
    progress: set[int] | None = None,
) -> None:
    learner_version = (
        job.learner_version
        if isinstance(job, GenerationSnapshotJobSpec)
        else job.expected_learner_version
    )
    ready_state = (
        "gradient-ready"
        if label.endswith("F/B")
        else "mutation-fenced"
        if isinstance(job, GenerationSnapshotJobSpec)
        else "GPU-ready"
    )
    expected_ranks = set(range(len(rank_processes)))
    received_ranks: set[int] = set()
    for _ in rank_processes:
        result = _CommandReady.model_validate(await receiver.recv())
        if (
            result.rank not in expected_ranks
            or result.rank in received_ranks
            or (result.operation_id, result.learner_version)
            != (job.operation_id, learner_version)
        ):
            raise RuntimeError(
                f"trainer {label} readiness has mismatched rank identity"
            )
        received_ranks.add(result.rank)
        if progress is not None:
            progress.add(result.rank)
        if result.error_type is not None:
            raise RuntimeError(
                f"trainer {label} failed before {ready_state}:\n"
                f"rank {result.rank}: {result.error_type}: {result.message}\n"
                f"{result.traceback_text or ''}"
            )
    if received_ranks != expected_ranks:
        raise RuntimeError(f"trainer {label} readiness has mismatched rank identity")


async def _await_command_readiness(
    rank_call: asyncio.Future[Any],
    readiness: asyncio.Future[None],
    deadline: float,
    *,
    timeout_message: str,
) -> None:
    loop = asyncio.get_running_loop()
    done, _ = await asyncio.wait(
        {rank_call, readiness},
        timeout=max(0.0, deadline - loop.time()),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not done:
        raise TimeoutError(timeout_message)
    if readiness in done:
        readiness.result()
    if rank_call in done:
        rank_call.result()
    if not readiness.done():
        done, _ = await asyncio.wait(
            {readiness}, timeout=max(0.0, deadline - loop.time())
        )
        if not done:
            raise TimeoutError(timeout_message)
    readiness.result()


async def _snapshot_readiness_timeout(
    actors: Any,
    job: GenerationSnapshotJobSpec,
    received_ranks: set[int],
) -> TimeoutError:
    try:
        values = await asyncio.wait_for(
            actors.inspect_snapshot_phase.call(job.operation_id), timeout=2.0
        )
        phases: Any = sorted(values.values(), key=lambda value: value["rank"])
    except Exception as error:
        phases = {
            "diagnostic_error": f"{type(error).__name__}: {error}",
        }
    return TimeoutError(
        "trainer ranks did not fence snapshot mutation: "
        f"received_ranks={sorted(received_ranks)} "
        f"rank_phases={json.dumps(phases, sort_keys=True)}"
    )


async def _await_snapshot_readiness(
    rank_call: asyncio.Future[Any],
    readiness: asyncio.Future[None],
    actors: Any,
    job: GenerationSnapshotJobSpec,
    received_ranks: set[int],
    deadline: float,
) -> None:
    loop = asyncio.get_running_loop()
    done, _ = await asyncio.wait(
        {rank_call, readiness},
        timeout=max(0.0, deadline - loop.time()),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not done:
        raise await _snapshot_readiness_timeout(actors, job, received_ranks)
    if readiness in done:
        readiness.result()
    if rank_call in done:
        rank_call.result()
    if not readiness.done():
        done, _ = await asyncio.wait(
            {readiness}, timeout=max(0.0, deadline - loop.time())
        )
        if not done:
            raise await _snapshot_readiness_timeout(actors, job, received_ranks)
    readiness.result()


class MonarchTrainerSlot:
    """One persistent Megatron mesh serving many sequenced training runs."""

    def __init__(
        self,
        runtime_spec: TrainerRuntimeSpec,
        actors: Any,
        proc_mesh: ProcMesh,
        supervision: MonarchTrainerSupervision,
        rank_processes: tuple[_TrainerRankReady, ...],
        cp_lookahead_ports: tuple[Port[Any], ...],
        residency_prefetch_ports: tuple[Port[Any], ...],
        *,
        command_timeout_s: float,
        shutdown_timeout_s: float,
    ) -> None:
        self.runtime_spec = runtime_spec
        self._actors = actors
        self._proc_mesh = proc_mesh
        self._supervision = supervision
        self._rank_processes = rank_processes
        self._cp_lookahead_ports = cp_lookahead_ports
        self._residency_prefetch_ports = residency_prefetch_ports
        self._command_timeout_s = command_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._cp_lookahead_lock = asyncio.Lock()
        self._residency_prefetch_lock = asyncio.Lock()
        self._kl_reference_lock = asyncio.Lock()
        self._registration_lock = asyncio.Lock()
        self._closed = False
        self._valid = True
        self._stop_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._publications: dict[
            str, asyncio.Task[tuple[TrainerRankPublication, ...]]
        ] = {}
        self._publication_authorizations: dict[str, asyncio.Event] = {}
        self._publication_predecessors: dict[str, asyncio.Event | None] = {}
        self._publication_authorization_tail: asyncio.Event | None = None
        self._registrations: dict[str, tuple[str, RunOptimizerWorkSummary]] = {}
        self._registration_tasks: dict[
            str, tuple[str, asyncio.Task[RunOptimizerWorkSummary]]
        ] = {}
        self._removal_tasks: dict[str, asyncio.Task[None]] = {}
        self._cleanup_slots = asyncio.BoundedSemaphore(_MAX_PENDING_RUN_CLEANUPS)
        self._operations: dict[str, tuple[str, dict[str, Any]]] = {}
        self._snapshot_launches: dict[
            str, tuple[str, SnapshotPrepareCommandLaunch]
        ] = {}
        self._snapshot_tasks: set[asyncio.Task[dict[str, Any]]] = set()
        self._forward_backward_launches: dict[
            str, tuple[str, ForwardBackwardCommandLaunch]
        ] = {}
        self._forward_launches: dict[str, tuple[str, ForwardCommandLaunch]] = {}

    @property
    def valid(self) -> bool:
        return self._valid

    async def register_run(
        self, registration: RunSlotRegistration
    ) -> RunOptimizerWorkSummary:
        fingerprint = registration.model_dump_json()
        async with self._registration_lock:
            self._require_open()
            if prior := self._registrations.get(registration.run_id):
                if prior[0] != fingerprint:
                    raise RuntimeError("run_id was reused for another registration")
                return prior[1]
            if registration.run_id in self._removal_tasks:
                raise RuntimeError("run_id is still being removed")
            pending = self._registration_tasks.get(registration.run_id)
            if pending is not None:
                if pending[0] != fingerprint:
                    raise RuntimeError("run_id was reused for another registration")
                task = pending[1]
            else:
                task = asyncio.create_task(
                    self._register_prepared_run(registration, fingerprint),
                    name=f"megatron-register-{registration.run_id}",
                )
                task.add_done_callback(consume_future_exception)
                self._registration_tasks[registration.run_id] = (fingerprint, task)
        return await asyncio.shield(task)

    async def _register_prepared_run(
        self, registration: RunSlotRegistration, fingerprint: str
    ) -> RunOptimizerWorkSummary:
        payload = registration.model_dump_json()
        expected_ranks = set(range(len(self._rank_processes)))

        def validate(results: list[dict[str, Any]]) -> None:
            if {result["rank"] for result in results} != expected_ranks or {
                result["run_id"] for result in results
            } != {registration.run_id}:
                raise RuntimeError("trainer ranks disagree on run registration")

        try:
            ready_port, receiver = Channel.open()
            started = list(
                (
                    await asyncio.wait_for(
                        self._actors.start_prepare_run_slot_registration.call(
                            payload, ready_port
                        ),
                        timeout=self._command_timeout_s,
                    )
                ).values()
            )
            validate(started)
            deadline = asyncio.get_running_loop().time() + self._command_timeout_s
            ready = [
                _RunRegistrationReady.model_validate(
                    await asyncio.wait_for(
                        receiver.recv(),
                        timeout=max(
                            deadline - asyncio.get_running_loop().time(), 0.001
                        ),
                    )
                )
                for _ in self._rank_processes
            ]
            if {item.rank for item in ready} != expected_ranks or {
                item.run_id for item in ready
            } != {registration.run_id}:
                raise RuntimeError("trainer ranks disagree on run preparation")
            failures = [item for item in ready if item.error_type is not None]
            if failures:
                details = "\n".join(
                    f"rank {item.rank}: {item.error_type}: {item.message}\n"
                    f"{item.traceback_text or ''}"
                    for item in failures
                )
                raise RuntimeError(f"run registration preparation failed:\n{details}")
            rank_work = tuple(
                item.optimizer_work
                for item in sorted(ready, key=lambda item: item.rank)
                if item.optimizer_work is not None
            )
            if len(rank_work) != len(self._rank_processes):
                raise RuntimeError("trainer rank omitted exact optimizer work")
            optimizer_work = RunOptimizerWorkSummary(
                run_id=registration.run_id,
                ranks=rank_work,
            )
            finished = list(
                (
                    await asyncio.wait_for(
                        self._actors.finish_prepare_run_slot_registration.call(payload),
                        timeout=max(
                            deadline - asyncio.get_running_loop().time(), 0.001
                        ),
                    )
                ).values()
            )
            validate(finished)
            self._registrations[registration.run_id] = (
                fingerprint,
                optimizer_work,
            )
            return optimizer_work
        except BaseException as error:
            if self.valid:
                try:
                    await asyncio.wait_for(
                        self._actors.discard_run_slot_registration.call(
                            registration.run_id
                        ),
                        timeout=self._command_timeout_s,
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "registration discard failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                await self._invalidate(error, "run registration and cleanup failed")
            raise
        finally:
            pending = self._registration_tasks.get(registration.run_id)
            if pending is not None and pending[1] is asyncio.current_task():
                self._registration_tasks.pop(registration.run_id, None)

    async def unregister_run(self, run_id: str) -> None:
        async with self._registration_lock:
            self._require_open()
            task = self._removal_tasks.get(run_id)
            if task is None and run_id not in self._registrations:
                raise RuntimeError(f"training run is not registered: {run_id!r}")
            if task is None:
                task = asyncio.create_task(
                    self._unregister_run(run_id),
                    name=f"megatron-unregister-{run_id}",
                )
                task.add_done_callback(consume_future_exception)
                self._removal_tasks[run_id] = task
        await asyncio.shield(task)

    async def _unregister_run(self, run_id: str) -> None:
        await self._cleanup_slots.acquire()
        try:
            try:
                async with self._lock:
                    self._require_open()
                    values = await asyncio.wait_for(
                        self._actors.start_unregister_run_slot.call(run_id),
                        timeout=self._command_timeout_s,
                    )
                    self._validate_run_removal(values, run_id)
                values = await asyncio.wait_for(
                    self._actors.finish_unregister_run_slot.call(run_id),
                    timeout=self._command_timeout_s,
                )
                self._validate_run_removal(values, run_id)
                self._registrations.pop(run_id)
            except BaseException as error:
                await self._invalidate(error, "run removal and cleanup failed")
                raise
        finally:
            self._cleanup_slots.release()
            self._removal_tasks.pop(run_id, None)

    def _validate_run_removal(self, values: Any, run_id: str) -> None:
        results = list(values.values())
        if {result["rank"] for result in results} != set(
            range(len(self._rank_processes))
        ) or {result["run_id"] for result in results} != {run_id}:
            raise RuntimeError("trainer ranks disagree on run removal")

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
    ) -> ForwardBackwardCommandLaunch:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached)
                return ForwardBackwardCommandLaunch(completion=completion)
            if inflight := self._forward_backward_launches.get(job.operation_id):
                if inflight[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different F/B command"
                    )
                return inflight[1]
            if batch.ref != job.batch:
                raise ValueError("F/B batch ref does not match supplied packed batch")
            deadline = asyncio.get_running_loop().time() + self._command_timeout_s
            ready_port, receiver = Channel.open()
            rank_call = asyncio.ensure_future(
                self._actors.start_run_slot_forward_backward.call(
                    job.model_dump_json(), batch.model_dump_json(), ready_port
                )
            )
            readiness = asyncio.create_task(
                self._collect_command_ready(receiver, job, label="F/B"),
                name=f"megatron-fb-ready-{job.operation_id}",
            )
            try:
                await _await_command_readiness(
                    rank_call,
                    readiness,
                    deadline,
                    timeout_message="trainer ranks did not reach F/B gradient-ready",
                )
                completion = asyncio.create_task(
                    self._complete_forward_backward(job, rank_call, deadline),
                    name=f"megatron-fb-result-{job.operation_id}",
                )
                launch = ForwardBackwardCommandLaunch(completion=completion)
                self._forward_backward_launches[job.operation_id] = (
                    job.fingerprint,
                    launch,
                )
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                await self._invalidate(error, "F/B operation and cleanup failed")
                raise

    async def _collect_command_ready(
        self,
        receiver: Any,
        job: (
            ForwardJobSpec
            | ForwardBackwardJobSpec
            | SftForwardJobSpec
            | SftForwardBackwardJobSpec
        ),
        *,
        label: str,
    ) -> None:
        await _collect_command_ready(receiver, job, self._rank_processes, label=label)

    async def _complete_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        rank_call: asyncio.Future[Any],
        deadline: float,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            if {result["token_count"] for result in results} != {
                job.trainable_token_count
            }:
                raise RuntimeError(
                    "trainer F/B token count differs from packed provenance"
                )
            result = _coordinator_command_result(
                results,
                expected_token_count=job.trainable_token_count,
                aggregate_telemetry=True,
            )
            self._operations[job.operation_id] = (job.fingerprint, result)
            return result
        except BaseException as error:
            await self._invalidate(error, "late F/B result and cleanup failed")
            raise
        finally:
            self._forward_backward_launches.pop(job.operation_id, None)

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
    ) -> ForwardCommandLaunch:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached)
                return ForwardCommandLaunch(completion=completion)
            if inflight := self._forward_launches.get(job.operation_id):
                if inflight[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different forward command"
                    )
                return inflight[1]
            if batch.ref != job.batch:
                raise ValueError(
                    "forward batch ref does not match supplied packed batch"
                )
            deadline = asyncio.get_running_loop().time() + self._command_timeout_s
            ready_port, receiver = Channel.open()
            rank_call = asyncio.ensure_future(
                self._actors.start_run_slot_forward.call(
                    job.model_dump_json(), batch.model_dump_json(), ready_port
                )
            )
            readiness = asyncio.create_task(
                self._collect_command_ready(receiver, job, label="forward"),
                name=f"megatron-forward-ready-{job.operation_id}",
            )
            try:
                await _await_command_readiness(
                    rank_call,
                    readiness,
                    deadline,
                    timeout_message="trainer ranks did not reach forward GPU-ready",
                )
                completion = asyncio.create_task(
                    self._complete_forward(job, rank_call, deadline),
                    name=f"megatron-forward-result-{job.operation_id}",
                )
                launch = ForwardCommandLaunch(completion=completion)
                self._forward_launches[job.operation_id] = (job.fingerprint, launch)
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                await self._invalidate(error, "forward operation and cleanup failed")
                raise

    async def _complete_forward(
        self,
        job: ForwardJobSpec,
        rank_call: asyncio.Future[Any],
        deadline: float,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            result = _coordinator_command_result(
                results,
                expected_token_count=job.trainable_token_count,
                aggregate_telemetry=job.loss is not None,
            )
            self._operations[job.operation_id] = (job.fingerprint, result)
            return result
        except BaseException as error:
            await self._invalidate(error, "late forward result and cleanup failed")
            raise
        finally:
            self._forward_launches.pop(job.operation_id, None)

    async def sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SftBatchLeaseSet,
    ) -> dict[str, Any]:
        launch = await self.start_sft_forward_backward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SftBatchLeaseSet,
    ) -> ForwardBackwardCommandLaunch:
        launch = await self._start_sft_command(job, batch, backward=True)
        assert isinstance(launch, ForwardBackwardCommandLaunch)
        return launch

    async def sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SftBatchLeaseSet,
    ) -> dict[str, Any]:
        launch = await self.start_sft_forward(job, batch)
        return await asyncio.shield(launch.completion)

    async def start_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SftBatchLeaseSet,
    ) -> ForwardCommandLaunch:
        launch = await self._start_sft_command(job, batch, backward=False)
        assert isinstance(launch, ForwardCommandLaunch)
        return launch

    async def _start_sft_command(
        self,
        job: SftForwardBackwardJobSpec | SftForwardJobSpec,
        batch: SftBatchLeaseSet,
        *,
        backward: bool,
    ) -> ForwardBackwardCommandLaunch | ForwardCommandLaunch:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            launch_type = (
                ForwardBackwardCommandLaunch if backward else ForwardCommandLaunch
            )
            if cached is not None:
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached)
                return launch_type(completion=completion)
            launches = (
                self._forward_backward_launches if backward else self._forward_launches
            )
            if inflight := launches.get(job.operation_id):
                if inflight[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for a different SFT command"
                    )
                return inflight[1]
            if batch.manifest.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT payload fingerprint differs from its job")
            if batch.manifest.num_trainable_tokens != job.trainable_token_count:
                raise ValueError("SFT trainable-token count differs from its job")
            deadline = asyncio.get_running_loop().time() + self._command_timeout_s
            ready_port, receiver = Channel.open()
            endpoint = (
                self._actors.start_run_slot_sft_forward_backward
                if backward
                else self._actors.start_run_slot_sft_forward
            )
            label = "SFT F/B" if backward else "SFT forward"
            rank_call = asyncio.ensure_future(
                endpoint.call(
                    job.model_dump_json(), batch.model_dump_json(), ready_port
                )
            )
            readiness = asyncio.create_task(
                self._collect_command_ready(receiver, job, label=label),
                name=f"megatron-sft-ready-{job.operation_id}",
            )
            try:
                await _await_command_readiness(
                    rank_call,
                    readiness,
                    deadline,
                    timeout_message=f"trainer ranks did not reach {label} readiness",
                )
                completion = asyncio.create_task(
                    self._complete_sft_command(job, rank_call, deadline, backward),
                    name=f"megatron-sft-result-{job.operation_id}",
                )
                launch = launch_type(completion=completion)
                launches[job.operation_id] = (job.fingerprint, launch)
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                await self._invalidate(error, f"{label} operation and cleanup failed")
                raise

    async def _complete_sft_command(
        self,
        job: SftForwardBackwardJobSpec | SftForwardJobSpec,
        rank_call: asyncio.Future[Any],
        deadline: float,
        backward: bool,
    ) -> dict[str, Any]:
        launches = (
            self._forward_backward_launches if backward else self._forward_launches
        )
        label = "SFT F/B" if backward else "SFT forward"
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            if {result["token_count"] for result in results} != {
                job.trainable_token_count
            }:
                raise RuntimeError(
                    f"trainer {label} token count differs from its payload"
                )
            result = _coordinator_command_result(
                results,
                expected_token_count=job.trainable_token_count,
                aggregate_telemetry=True,
            )
            self._operations[job.operation_id] = (job.fingerprint, result)
            return result
        except BaseException as error:
            await self._invalidate(error, f"late {label} result and cleanup failed")
            raise
        finally:
            launches.pop(job.operation_id, None)

    async def optim_step(self, job: OptimizerJobSpec) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_optimizer.call(job.model_dump_json()),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                self._validate_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                contributions = {
                    tuple(result["contributing_forward_backward_operation_ids"])
                    for result in results
                }
                if contributions != {job.contributing_forward_backward_operation_ids}:
                    raise RuntimeError(
                        "trainer ranks consumed different F/B contributions"
                    )
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "optimizer operation and cleanup failed")
                raise

    async def prepare_load_state(self, job: LoadStateJobSpec) -> None:
        self._require_open()
        payload = job.model_dump_json()
        try:
            ready_port, receiver = Channel.open()
            started = list(
                (
                    await asyncio.wait_for(
                        self._actors.start_prepare_run_slot_load_state.call(
                            payload, ready_port
                        ),
                        timeout=self._command_timeout_s,
                    )
                ).values()
            )
            self._validate_command_results(
                started,
                operation_id=job.operation_id,
                learner_version=job.learner_version,
            )
            deadline = asyncio.get_running_loop().time() + self._command_timeout_s
            ready = [
                _CommandReady.model_validate(
                    await asyncio.wait_for(
                        receiver.recv(),
                        timeout=max(
                            deadline - asyncio.get_running_loop().time(), 0.001
                        ),
                    )
                )
                for _ in self._rank_processes
            ]
            self._validate_command_results(
                [item.model_dump(mode="json") for item in ready],
                operation_id=job.operation_id,
                learner_version=job.learner_version,
            )
            failures = [item for item in ready if item.error_type is not None]
            if failures:
                details = "\n".join(
                    f"rank {item.rank}: {item.error_type}: {item.message}\n"
                    f"{item.traceback_text or ''}"
                    for item in failures
                )
                raise RuntimeError(f"load preparation failed:\n{details}")
        except BaseException as error:
            await self._invalidate(error, "load preparation and cleanup failed")
            raise

    async def discard_prepared_load_state(self, operation_id: str) -> None:
        if not self.valid:
            return
        await asyncio.wait_for(
            self._actors.discard_run_slot_load_state.call(operation_id),
            timeout=self._command_timeout_s,
        )

    async def acquire_kl_reference(
        self, spec: KlReferenceSpec
    ) -> KlReferenceAcquisition:
        async with self._kl_reference_lock:
            self._require_open()
            payload = spec.model_dump_json()
            acquisition_id = uuid.uuid4().hex
            ready_port, receiver = Channel.open()
            started_at = time.perf_counter()
            try:
                started = list(
                    (
                        await asyncio.wait_for(
                            self._actors.start_prepare_run_slot_kl_reference.call(
                                payload, ready_port
                            ),
                            timeout=self._command_timeout_s,
                        )
                    ).values()
                )
                if {item["rank"] for item in started} != set(
                    range(len(self._rank_processes))
                ) or any(
                    (item["run_id"], item["checkpoint_id"])
                    != (spec.run_id, spec.checkpoint_id)
                    for item in started
                ):
                    raise RuntimeError(
                        "KL preparation started with mismatched identity"
                    )
                deadline = asyncio.get_running_loop().time() + self._command_timeout_s
                ready = [
                    _KlReferenceReady.model_validate(
                        await asyncio.wait_for(
                            receiver.recv(),
                            timeout=max(
                                deadline - asyncio.get_running_loop().time(), 0.001
                            ),
                        )
                    )
                    for _ in self._rank_processes
                ]
                if {item.rank for item in ready} != set(
                    range(len(self._rank_processes))
                ) or any(
                    (item.run_id, item.checkpoint_id)
                    != (spec.run_id, spec.checkpoint_id)
                    for item in ready
                ):
                    raise RuntimeError(
                        "KL preparation completed with mismatched identity"
                    )
                failures = [item for item in ready if item.error_type is not None]
                if failures:
                    details = "\n".join(
                        f"rank {item.rank}: {item.error_type}: {item.message}\n"
                        f"{item.traceback_text or ''}"
                        for item in failures
                    )
                    raise RuntimeError(f"KL reference preparation failed:\n{details}")
                acquired = list(
                    (
                        await asyncio.wait_for(
                            self._actors.acquire_run_slot_kl_reference.call(
                                payload, acquisition_id
                            ),
                            timeout=self._command_timeout_s,
                        )
                    ).values()
                )
                if {item["rank"] for item in acquired} != set(
                    range(len(self._rank_processes))
                ) or any(
                    (item["run_id"], item["checkpoint_id"])
                    != (spec.run_id, spec.checkpoint_id)
                    for item in acquired
                ):
                    raise RuntimeError("KL reference acquired with mismatched identity")
                return KlReferenceAcquisition(
                    run_id=spec.run_id,
                    checkpoint_id=spec.checkpoint_id,
                    acquisition_id=acquisition_id,
                    metrics={
                        "time/kl_reference_prepare_s": (
                            time.perf_counter() - started_at
                        ),
                        "kl_reference/rank_bytes": float(
                            sum(item["byte_count"] for item in acquired)
                        ),
                    },
                )
            except BaseException as error:
                try:
                    await asyncio.wait_for(
                        self._actors.abort_run_slot_kl_reference_acquisition.call(
                            spec.run_id, spec.checkpoint_id, acquisition_id
                        ),
                        timeout=self._command_timeout_s,
                    )
                except BaseException as cleanup_error:
                    failure = BaseExceptionGroup(
                        "KL reference acquisition and rollback failed",
                        [error, cleanup_error],
                    )
                    await self._invalidate(
                        failure, "KL reference acquisition rollback failed"
                    )
                    raise failure
                raise

    async def release_kl_reference(
        self, run_id: str, checkpoint_id: str, acquisition_id: str
    ) -> None:
        async with self._kl_reference_lock:
            self._require_open()
            try:
                await asyncio.wait_for(
                    self._actors.release_run_slot_kl_reference.call(
                        run_id, checkpoint_id, acquisition_id
                    ),
                    timeout=self._command_timeout_s,
                )
            except BaseException as error:
                try:
                    await asyncio.wait_for(
                        self._actors.release_run_slot_kl_reference.call(
                            run_id, checkpoint_id, acquisition_id
                        ),
                        timeout=self._command_timeout_s,
                    )
                except BaseException as retry_error:
                    failure = BaseExceptionGroup(
                        "KL reference release retry failed", [error, retry_error]
                    )
                    await self._invalidate(failure, "KL reference release failed")
                    raise failure

    async def load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        async with self._control_lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_load_state.call(
                        job.model_dump_json()
                    ),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                self._validate_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                if {result["optimizer_restored"] for result in results} != {
                    job.restore_optimizer
                }:
                    raise RuntimeError(
                        "trainer ranks disagree on optimizer restoration"
                    )
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "load operation and cleanup failed")
                raise

    async def prepare_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotWritePlan:
        launch = await self.start_prepare_snapshot(job)
        result = await asyncio.shield(launch.completion)
        return SnapshotWritePlan.model_validate(result["write_plan"])

    async def start_prepare_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotPrepareCommandLaunch:
        async with self._control_lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                publication = self._publications.get(job.operation_id)
                if publication is None:
                    raise RuntimeError("completed snapshot has no publication owner")
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached)
                return SnapshotPrepareCommandLaunch(
                    completion=completion, publication=publication
                )
            if inflight := self._snapshot_launches.get(job.operation_id):
                if inflight[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another snapshot")
                return inflight[1]
            operation_id = job.operation_id
            if operation_id in self._publications:
                raise RuntimeError(f"publication already exists: {operation_id}")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._command_timeout_s
            event_port, event_receiver = Channel[dict[str, Any]].open()
            ready_port, ready_receiver = Channel[dict[str, Any]].open()
            rank_call = asyncio.ensure_future(
                self._actors.execute_run_slot_snapshot.call(
                    job.model_dump_json(), event_port, ready_port
                )
            )
            readiness_progress: set[int] = set()
            readiness = asyncio.create_task(
                _collect_command_ready(
                    ready_receiver,
                    job,
                    self._rank_processes,
                    label="snapshot",
                    progress=readiness_progress,
                ),
                name=f"megatron-slot-snapshot-ready-{operation_id}",
            )
            try:
                await _await_snapshot_readiness(
                    rank_call,
                    readiness,
                    self._actors,
                    job,
                    readiness_progress,
                    deadline,
                )
                authorization = asyncio.Event()
                publication = asyncio.create_task(
                    self._collect_publication(
                        event_receiver,
                        job.generation,
                        authorization=authorization,
                    ),
                    name=f"megatron-slot-publish-{operation_id}",
                )
                publication.add_done_callback(consume_future_exception)
                self._publications[operation_id] = publication
                self._publication_authorizations[operation_id] = authorization
                self._publication_predecessors[operation_id] = (
                    self._publication_authorization_tail
                )
                self._publication_authorization_tail = authorization
                completion = asyncio.create_task(
                    self._complete_snapshot_prepare(
                        job, rank_call, publication, deadline
                    ),
                    name=f"megatron-slot-snapshot-plan-{operation_id}",
                )
                launch = SnapshotPrepareCommandLaunch(
                    completion=completion, publication=publication
                )
                self._snapshot_launches[operation_id] = (job.fingerprint, launch)
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                publication = self._publications.get(operation_id)
                if publication is not None:
                    await self._abort_snapshot_publication(operation_id, publication)
                await self._invalidate(error, "snapshot operation and cleanup failed")
                raise

    async def _complete_snapshot_prepare(
        self,
        job: GenerationSnapshotJobSpec,
        rank_call: asyncio.Future[Any],
        publication: asyncio.Task[tuple[TrainerRankPublication, ...]],
        deadline: float,
    ) -> dict[str, Any]:
        operation_id = job.operation_id
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_command_results(
                results,
                operation_id=operation_id,
                learner_version=job.learner_version,
            )
            rank_plans = tuple(
                SnapshotRankWritePlan.model_validate(result["rank_write_plan"])
                for result in results
            )
            plan = build_snapshot_write_plan(
                operation_id=operation_id,
                generation=job.generation,
                ranks=rank_plans,
            )
            result = next(result for result in results if result["rank"] == 0)
            result = {**result, "write_plan": plan.model_dump(mode="json")}
            self._operations[operation_id] = (job.fingerprint, result)
            return result
        except BaseException as error:
            await self._abort_snapshot_publication(operation_id, publication)
            await self._invalidate(error, "snapshot operation and cleanup failed")
            raise
        finally:
            self._snapshot_launches.pop(operation_id, None)

    async def _abort_snapshot_publication(
        self,
        operation_id: str,
        publication: asyncio.Task[tuple[TrainerRankPublication, ...]],
    ) -> None:
        authorization = self._publication_authorizations.get(operation_id)
        if authorization is not None:
            authorization.set()
        if not publication.done():
            publication.cancel()
        await asyncio.gather(publication, return_exceptions=True)
        if self._publications.get(operation_id) is publication:
            self._publications.pop(operation_id)
        predecessor = self._publication_predecessors.pop(operation_id, None)
        self._publication_authorizations.pop(operation_id, None)
        if self._publication_authorization_tail is authorization:
            self._publication_authorization_tail = predecessor

    async def snapshot(self, job: GenerationSnapshotJobSpec) -> dict[str, Any]:
        launch = await self.start_snapshot(job)
        return await asyncio.shield(launch.completion)

    async def start_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotPrepareCommandLaunch:
        prepared = await self.start_prepare_snapshot(job)

        async def complete() -> dict[str, Any]:
            result = await asyncio.shield(prepared.completion)
            plan = SnapshotWritePlan.model_validate(result["write_plan"])
            metrics = await self.authorize_snapshot(
                plan, SnapshotWriteGrant.local(plan)
            )
            return {**result, "metrics": {**result["metrics"], **metrics}}

        completion = asyncio.create_task(
            complete(), name=f"megatron-slot-snapshot-{job.operation_id}"
        )
        self._snapshot_tasks.add(completion)
        completion.add_done_callback(self._snapshot_tasks.discard)
        completion.add_done_callback(consume_future_exception)
        return SnapshotPrepareCommandLaunch(
            completion=completion, publication=prepared.publication
        )

    async def authorize_snapshot(
        self, plan: SnapshotWritePlan, grant: SnapshotWriteGrant
    ) -> dict[str, float]:
        grant.validate_plan(plan)
        if plan.operation_id not in self._publication_predecessors:
            raise RuntimeError(
                f"trainer has no prepared publication {plan.operation_id}"
            )
        predecessor = self._publication_predecessors[plan.operation_id]
        if predecessor is not None:
            await asyncio.wait_for(predecessor.wait(), timeout=self._command_timeout_s)
        async with self._control_lock:
            prepared = self._operations.get(plan.operation_id)
            if (
                prepared is None
                or SnapshotWritePlan.model_validate(prepared[1]["write_plan"]) != plan
            ):
                raise RuntimeError(
                    "snapshot authorization differs from its prepared plan"
                )
            authorization = self._publication_authorizations.get(plan.operation_id)
            if authorization is None:
                raise RuntimeError(
                    f"trainer has no prepared publication {plan.operation_id}"
                )
            if authorization.is_set():
                return {}
            values = await asyncio.wait_for(
                self._actors.authorize_run_slot_snapshot.call(
                    plan.model_dump_json(), grant.model_dump_json()
                ),
                timeout=self._command_timeout_s,
            )
            results = list(values.values())
            expected_ranks = set(range(len(self._rank_processes)))
            if {result["rank"] for result in results} != expected_ranks or {
                result["operation_id"] for result in results
            } != {plan.operation_id}:
                raise RuntimeError("trainer ranks returned an invalid authorization")
            authorization.set()
            return next(result for result in results if result["rank"] == 0)["metrics"]

    async def discard_prepared_snapshot(self, operation_id: str) -> None:
        if operation_id not in self._publication_predecessors:
            return
        predecessor = self._publication_predecessors[operation_id]
        if predecessor is not None:
            await asyncio.wait_for(predecessor.wait(), timeout=self._command_timeout_s)
        async with self._control_lock:
            authorization = self._publication_authorizations.get(operation_id)
            publication = self._publications.get(operation_id)
            if authorization is None or publication is None:
                return
            if authorization.is_set():
                raise RuntimeError("cannot discard an authorized snapshot write")
            values = await asyncio.wait_for(
                self._actors.discard_run_slot_snapshot.call(operation_id),
                timeout=self._command_timeout_s,
            )
            results = list(values.values())
            if {result["rank"] for result in results} != set(
                range(len(self._rank_processes))
            ) or {result["operation_id"] for result in results} != {operation_id}:
                raise RuntimeError("trainer ranks discarded another snapshot")
            authorization.set()
            outcome = (await asyncio.gather(publication, return_exceptions=True))[0]
            if not isinstance(outcome, BaseException):
                raise RuntimeError("discarded snapshot unexpectedly published")
            self._publications.pop(operation_id, None)
            self._publication_authorizations.pop(operation_id, None)
            self._publication_predecessors.pop(operation_id, None)
            self._operations.pop(operation_id, None)

    def wait_for_publication(
        self, operation_id: str
    ) -> Awaitable[tuple[TrainerRankPublication, ...]]:
        try:
            publication = self._publications[operation_id]
        except KeyError as exc:
            raise RuntimeError(f"trainer has no publication {operation_id}") from exc
        return asyncio.shield(publication)

    def retire_operation(self, operation_id: str) -> None:
        self._snapshot_launches.pop(operation_id, None)
        publication = self._publications.get(operation_id)
        if publication is not None:
            if not publication.done():
                raise RuntimeError("cannot retire an active trainer publication")
            self._publications.pop(operation_id)
            self._publication_authorizations.pop(operation_id, None)
            self._publication_predecessors.pop(operation_id, None)
        self._operations.pop(operation_id, None)

    async def prepare_cp_lookahead(
        self,
        batch: PackedBatchLeaseSet,
        *,
        global_grad_accumulation_sequences: int | None,
    ) -> dict[str, float]:
        self._require_open()
        return await _prepare_cp_lookahead(
            self._cp_lookahead_ports,
            self._cp_lookahead_lock,
            batch,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            timeout_s=self._command_timeout_s,
        )

    async def prepare_residency(
        self,
        run_id: str,
        command_kind: str,
        learner_version: int,
        kl_reference_checkpoint_id: str | None = None,
    ) -> dict[str, float]:
        self._require_open()
        if not self._residency_prefetch_ports:
            raise RuntimeError("trainer slot has no residency prefetch service")
        return await _prepare_run_residency(
            self._residency_prefetch_ports,
            self._residency_prefetch_lock,
            None,
            run_id,
            command_kind,
            learner_version,
            kl_reference_checkpoint_id,
            timeout_s=self._command_timeout_s,
        )

    async def admit_residency(
        self,
        operation_id: str,
        run_id: str,
        command_kind: str,
        learner_version: int,
        kl_reference_checkpoint_id: str | None = None,
    ) -> dict[str, float]:
        self._require_open()
        try:
            metrics = await _prepare_run_residency(
                self._residency_prefetch_ports,
                self._residency_prefetch_lock,
                operation_id,
                run_id,
                command_kind,
                learner_version,
                kl_reference_checkpoint_id,
                timeout_s=self._command_timeout_s,
            )
        except BaseException as error:
            try:
                await self.release_residency_admission(operation_id)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "residency admission and rollback failed", [error, cleanup_error]
                ) from None
            raise
        if metrics["residency/prefetch_admitted"] == 0.0:
            await self.release_residency_admission(operation_id)
        return metrics

    async def release_residency_admission(self, operation_id: str) -> None:
        self._require_open()
        values = await asyncio.wait_for(
            self._actors.release_run_slot_residency_admission.call(operation_id),
            timeout=self._command_timeout_s,
        )
        results = list(values.values())
        if {result["rank"] for result in results} != set(
            range(len(self._rank_processes))
        ) or {result["operation_id"] for result in results} != {operation_id}:
            raise RuntimeError("trainer ranks released another residency admission")

    def _cached_operation(
        self, operation_id: str, fingerprint: str
    ) -> dict[str, Any] | None:
        cached = self._operations.get(operation_id)
        if cached is None:
            return None
        if cached[0] != fingerprint:
            raise RuntimeError("operation_id was reused for a different command")
        return cached[1]

    async def _collect_publication(
        self,
        receiver: Any,
        generation: TrainerGeneration,
        *,
        authorization: asyncio.Event,
    ) -> tuple[TrainerRankPublication, ...]:
        records: dict[int, TrainerRankPublication] = {}
        progress: dict[int, str] = {}
        await authorization.wait()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._command_timeout_s
        supervision = asyncio.create_task(self._supervision.wait())
        receive: asyncio.Future[Any] | None = None
        try:
            while len(records) < len(self._rank_processes):
                receive = asyncio.ensure_future(receiver.recv())
                done, _ = await asyncio.wait(
                    {receive, supervision},
                    timeout=max(0.0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError(
                        "trainer publication exceeded command timeout; "
                        f"last rank phases: {progress}"
                    )
                if supervision in done:
                    raise RuntimeError("trainer mesh failed: " + supervision.result())
                payload = receive.result()
                if payload["kind"] == "rank_failed":
                    raise RuntimeError(
                        f"trainer rank {payload['rank']} publication failed: "
                        f"{payload['error_type']}: {payload['message']}\n"
                        f"{payload['traceback']}"
                    )
                event = TRAINER_PUBLICATION_EVENT_ADAPTER.validate_python(payload)
                if isinstance(event, TrainerPublicationProgress):
                    if event.generation_id != generation.generation_id:
                        raise RuntimeError("trainer rank progressed another generation")
                    if event.rank >= len(self._rank_processes):
                        raise RuntimeError(
                            "trainer publication progress has unknown rank"
                        )
                    progress[event.rank] = event.phase
                    continue
                if isinstance(event, TrainerPublicationFailed):
                    raise RuntimeError(
                        f"trainer rank {event.rank} publication failed "
                        f"({event.error_type}): {event.message}"
                    )
                record = event.record
                if record.generation != generation:
                    raise RuntimeError("trainer rank published another generation")
                if record.rank in records:
                    raise RuntimeError("trainer rank published a generation twice")
                records[record.rank] = record
            return tuple(records[rank] for rank in range(len(self._rank_processes)))
        finally:
            supervision.cancel()
            supervision.add_done_callback(consume_future_exception)
            if receive is not None and not receive.done():
                receive.cancel()
                receive.add_done_callback(consume_future_exception)

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is not None and self._close_task.done():
            try:
                self._close_task.result()
            except BaseException:
                self._close_task = None
        if self._close_task is None:
            self._valid = False
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(consume_future_exception)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        loop = asyncio.get_running_loop()
        shutdown_timeout_s = min(self._shutdown_timeout_s, process_shutdown_timeout(1))
        deadline = loop.time() + shutdown_timeout_s
        graceful_deadline = min(deadline, loop.time() + process_shutdown_timeout(2))
        primary: BaseException | None = None
        try:
            async with asyncio.timeout_at(graceful_deadline):
                for authorization in self._publication_authorizations.values():
                    authorization.set()
                registrations = tuple(
                    value[1] for value in self._registration_tasks.values()
                )
                for task in registrations:
                    task.cancel()
                if registrations:
                    await asyncio.gather(*registrations, return_exceptions=True)
                if self._snapshot_tasks:
                    await asyncio.gather(*tuple(self._snapshot_tasks))
                removals = tuple(self._removal_tasks.values())
                outcomes = await asyncio.gather(*removals, return_exceptions=True)
                failures = [
                    outcome
                    for outcome in outcomes
                    if isinstance(outcome, BaseException)
                ]
                await _remote_teardown(
                    self._actors.close.call(max(0.0, graceful_deadline - loop.time()))
                )
                await asyncio.gather(*self._publications.values())
                if failures:
                    raise BaseExceptionGroup("run cleanup failed", failures)
        except BaseException as error:
            primary = error
        try:
            await self._force_stop(max(0.0, deadline - loop.time()))
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    f"trainer slot cleanup failed: {type(error).__name__}: {error}"
                )
        else:
            self._closed = True
        if primary is not None:
            raise primary

    def _require_open(self) -> None:
        if self._closed or not self._valid:
            raise RuntimeError("trainer slot is invalid")

    def _validate_command_results(
        self,
        results: list[dict[str, Any]],
        *,
        operation_id: str,
        learner_version: int,
    ) -> None:
        if (
            {result["rank"] for result in results}
            != set(range(len(self._rank_processes)))
            or {result["operation_id"] for result in results} != {operation_id}
            or {result["learner_version"] for result in results} != {learner_version}
        ):
            raise RuntimeError("trainer ranks disagree on command completion")

    async def _invalidate(self, error: BaseException, message: str) -> None:
        self._valid = False

        async def stop() -> None:
            await self._force_stop()
            self._closed = True

        await cleanup_after_failure(error, stop, message=message)

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
                self._supervision.close(
                    suppress_owned_mesh_faults_s=self._shutdown_timeout_s
                )
                if not task.cancelled():
                    task.exception()

            self._stop_task.add_done_callback(stopped)
        await asyncio.wait_for(
            asyncio.shield(self._stop_task),
            min(self._shutdown_timeout_s, process_shutdown_timeout(1))
            if timeout_s is None
            else timeout_s,
        )


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
        self._jobs: dict[str, tuple[str, tuple[TrainEvent, ...]]] = {}
        self._operations: dict[str, tuple[str, dict[str, Any]]] = {}
        self._forward_backward_launches: dict[
            str, tuple[str, ForwardBackwardCommandLaunch]
        ] = {}
        self._snapshot_launches: dict[
            str, tuple[str, SnapshotPrepareCommandLaunch]
        ] = {}
        self._snapshot_tasks: set[asyncio.Task[dict[str, Any]]] = set()
        self._operation_sequence_ids: dict[str, int] = {}
        self._cancelled_operations: dict[str, OperationRef] = {}
        self._next_operation_sequence = 0
        self._open_forward_backward_ids: list[str] = []
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

    async def consume_cancelled_operation(self, ref: OperationRef) -> None:
        prior = self._cancelled_operations.get(ref.operation_id)
        if prior is not None:
            if prior != ref:
                raise RuntimeError(
                    "cancelled operation_id was reused for a different command"
                )
            return
        async with self._lock:
            prior = self._cancelled_operations.get(ref.operation_id)
            if prior is not None:
                if prior != ref:
                    raise RuntimeError(
                        "cancelled operation_id was reused for a different command"
                    )
                return
            if ref.operation_id in self._operation_sequence_ids:
                raise RuntimeError("executed operation cannot be consumed as cancelled")
            if self._closed or not self._valid:
                raise RuntimeError("trainer runtime is invalid")
            if self._jobs:
                raise RuntimeError(
                    "fused train jobs cannot be mixed with command operations"
                )
            if ref.run_id != self.run_spec.run_id:
                raise ValueError("operation run_id does not match this training run")
            if ref.sequence_id != self._next_operation_sequence:
                raise RuntimeError(
                    "trainer cancelled operation sequence must be gapless: "
                    f"expected={self._next_operation_sequence}, "
                    f"got={ref.sequence_id}"
                )
            if ref.learner_parent_version != self._learner_version:
                raise ValueError(
                    "cancelled operation learner parent mismatch: "
                    f"operation={ref.learner_parent_version}, "
                    f"runtime={self._learner_version}"
                )
            if ref.reserved_output_learner_version is not None:
                raise RuntimeError("learner-transition command cannot be cancelled")
            self._cancelled_operations[ref.operation_id] = ref
            self._operation_sequence_ids[ref.operation_id] = ref.sequence_id
            self._next_operation_sequence += 1

    @property
    def learner_version(self) -> int:
        return self._learner_version

    @property
    def valid(self) -> bool:
        return self._valid

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
    ) -> ForwardBackwardCommandLaunch:
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for a different F/B")
                completion = asyncio.get_running_loop().create_future()
                completion.set_result(cached[1])
                return ForwardBackwardCommandLaunch(completion=completion)
            if inflight := self._forward_backward_launches.get(job.operation_id):
                if inflight[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for a different F/B")
                return inflight[1]
            self._validate_operation(job)
            if batch.ref != job.batch:
                raise ValueError("F/B batch ref does not match supplied packed batch")
            if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
                raise ValueError(
                    "packed batch sequence length does not match the trainer runtime"
                )
            return await self._start_forward_backward(job, batch)

    async def forward(
        self,
        job: ForwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another forward")
                return cached[1]
            self._validate_operation(job)
            if batch.ref != job.batch:
                raise ValueError(
                    "forward batch ref does not match supplied packed batch"
                )
            if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
                raise ValueError(
                    "packed batch sequence length does not match the trainer runtime"
                )
            return await self._run_forward(job, batch)

    async def _run_forward(
        self,
        job: ForwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        collective = asyncio.ensure_future(
            self._actors.execute_forward.call(
                job.model_dump_json(), batch.model_dump_json()
            )
        )
        self._active_job_id = job.operation_id
        self._active_collective = collective
        try:
            values = await asyncio.wait_for(
                collective,
                timeout=self._command_timeout_s(),
            )
            results = list(values.values())
            self._validate_rank_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            result = _coordinator_command_result(
                results,
                expected_token_count=job.trainable_token_count,
                aggregate_telemetry=job.loss is not None,
            )
            self._next_operation_sequence += 1
            self._operations[job.operation_id] = (job.fingerprint, result)
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            return result
        except BaseException as error:
            await self._invalidate_command(
                error, "forward operation and cleanup failed"
            )
            raise
        finally:
            if self._active_job_id == job.operation_id:
                self._active_job_id = None
                self._active_collective = None

    async def _start_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> ForwardBackwardCommandLaunch:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._command_timeout_s()
        ready_port, receiver = Channel.open()
        rank_call = asyncio.ensure_future(
            self._actors.start_forward_backward.call(
                job.model_dump_json(), batch.model_dump_json(), ready_port
            )
        )
        readiness = asyncio.create_task(
            _collect_command_ready(
                receiver,
                job,
                self._rank_processes,
                label="F/B",
            ),
            name=f"megatron-fb-ready-{job.operation_id}",
        )
        self._active_job_id = job.operation_id
        self._active_collective = rank_call
        try:
            await _await_command_readiness(
                rank_call,
                readiness,
                deadline,
                timeout_message="trainer ranks did not reach F/B gradient-ready",
            )
            self._open_forward_backward_ids.append(job.operation_id)
            self._next_operation_sequence += 1
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            completion = asyncio.create_task(
                self._complete_forward_backward(job, rank_call, deadline),
                name=f"megatron-fb-result-{job.operation_id}",
            )
            launch = ForwardBackwardCommandLaunch(completion=completion)
            self._forward_backward_launches[job.operation_id] = (
                job.fingerprint,
                launch,
            )
            return launch
        except BaseException as error:
            for task in (readiness, rank_call):
                if not task.done():
                    task.cancel()
            await asyncio.gather(readiness, rank_call, return_exceptions=True)
            await self._invalidate_command(error, "F/B operation and cleanup failed")
            raise
        finally:
            if self._active_job_id == job.operation_id:
                self._active_job_id = None
                self._active_collective = None

    async def _complete_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        rank_call: asyncio.Future[Any],
        deadline: float,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_rank_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.expected_learner_version,
            )
            if {result["token_count"] for result in results} != {
                job.trainable_token_count
            }:
                raise RuntimeError(
                    "trainer F/B token count differs from packed policy provenance"
                )
            result = _coordinator_command_result(
                results,
                expected_token_count=job.trainable_token_count,
                aggregate_telemetry=True,
            )
            self._operations[job.operation_id] = (job.fingerprint, result)
            return result
        except BaseException as error:
            await self._invalidate_command(error, "late F/B result and cleanup failed")
            raise
        finally:
            self._forward_backward_launches.pop(job.operation_id, None)

    async def sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        return await self._sft_forward(job, batch, backward=True)

    async def sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        return await self._sft_forward(job, batch, backward=False)

    async def _sft_forward(
        self,
        job: SftForwardBackwardJobSpec | SftForwardJobSpec,
        batch: SFTBatchData,
        *,
        backward: bool,
    ) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError(
                        "operation_id was reused for another SFT command"
                    )
                return cached[1]
            self._validate_operation(job)
            if batch.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT payload fingerprint differs from its job")
            collective = asyncio.ensure_future(
                (
                    self._actors.execute_sft_forward_backward
                    if backward
                    else self._actors.execute_sft_forward
                ).call(job.model_dump_json(), batch)
            )
            self._active_job_id = job.operation_id
            self._active_collective = collective
            try:
                values = await asyncio.wait_for(
                    collective,
                    timeout=self._command_timeout_s(),
                )
                results = list(values.values())
                self._validate_rank_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                )
                if {result["token_count"] for result in results} != {
                    job.trainable_token_count
                }:
                    raise RuntimeError(
                        "trainer SFT token count differs from its payload"
                    )
                result = _coordinator_command_result(
                    results,
                    expected_token_count=job.trainable_token_count,
                    aggregate_telemetry=True,
                )
                if backward:
                    self._open_forward_backward_ids.append(job.operation_id)
                self._next_operation_sequence += 1
                self._operations[job.operation_id] = (job.fingerprint, result)
                self._operation_sequence_ids[job.operation_id] = job.sequence_id
                return result
            except BaseException as error:
                await self._invalidate_command(error, "SFT command and cleanup failed")
                raise
            finally:
                if self._active_job_id == job.operation_id:
                    self._active_job_id = None
                    self._active_collective = None

    async def optim_step(self, job: OptimizerJobSpec) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another optimizer")
                return cached[1]
            self._validate_operation(job)
            contributions = tuple(self._open_forward_backward_ids)
            if job.contributing_forward_backward_operation_ids != contributions:
                raise RuntimeError(
                    "optimizer does not seal the exact open F/B operations"
                )
            return await self._run_optimizer(job)

    async def _run_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        collective = asyncio.ensure_future(
            self._actors.execute_optimizer.call(job.model_dump_json())
        )
        self._active_job_id = job.operation_id
        self._active_collective = collective
        try:
            values = await asyncio.wait_for(
                collective,
                timeout=self._command_timeout_s(),
            )
            results = list(values.values())
            self._validate_rank_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.learner_version,
            )
            contribution_sets = {
                tuple(result["contributing_forward_backward_operation_ids"])
                for result in results
            }
            if contribution_sets != {job.contributing_forward_backward_operation_ids}:
                raise RuntimeError("trainer ranks consumed different F/B contributions")
            result = next(result for result in results if result["rank"] == 0)
            self._open_forward_backward_ids.clear()
            self._learner_version = job.learner_version
            self._next_operation_sequence += 1
            for operation_id in job.contributing_forward_backward_operation_ids:
                self._operations.pop(operation_id, None)
                self._operation_sequence_ids.pop(operation_id, None)
            self._operations[job.operation_id] = (job.fingerprint, result)
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            return result
        except BaseException as error:
            await self._invalidate_command(
                error, "optimizer operation and cleanup failed"
            )
            raise
        finally:
            if self._active_job_id == job.operation_id:
                self._active_job_id = None
                self._active_collective = None

    async def load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another load")
                return cached[1]
            self._validate_operation(job)
            if self._open_forward_backward_ids:
                raise RuntimeError("load_state cannot discard open F/B contributions")
            collective = asyncio.ensure_future(
                self._actors.execute_load_state.call(job.model_dump_json())
            )
            self._active_job_id = job.operation_id
            self._active_collective = collective
            try:
                values = await asyncio.wait_for(
                    collective,
                    timeout=self._command_timeout_s(),
                )
                results = list(values.values())
                self._validate_rank_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                if {result["optimizer_restored"] for result in results} != {
                    job.restore_optimizer
                }:
                    raise RuntimeError(
                        "trainer ranks disagree on optimizer restoration"
                    )
                result = next(result for result in results if result["rank"] == 0)
                self._learner_version = job.learner_version
                self._next_operation_sequence += 1
                self._operations[job.operation_id] = (job.fingerprint, result)
                self._operation_sequence_ids[job.operation_id] = job.sequence_id
                return result
            except BaseException as error:
                await self._invalidate_command(
                    error, "load operation and cleanup failed"
                )
                raise
            finally:
                if self._active_job_id == job.operation_id:
                    self._active_job_id = None
                    self._active_collective = None

    async def prepare_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotWritePlan:
        launch = await self.start_prepare_snapshot(job)
        result = await asyncio.shield(launch.completion)
        return SnapshotWritePlan.model_validate(result["write_plan"])

    async def start_prepare_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotPrepareCommandLaunch:
        if launch := self._snapshot_launches.get(job.operation_id):
            if launch[0] != job.fingerprint:
                raise RuntimeError("operation_id was reused for another snapshot")
            return launch[1]
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            raise RuntimeError("completed snapshot has no publication owner")
        async with self._lock:
            if launch := self._snapshot_launches.get(job.operation_id):
                if launch[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another snapshot")
                return launch[1]
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another snapshot")
                raise RuntimeError("completed snapshot has no publication owner")
            self._validate_operation(job)
            generation_id = job.generation.generation_id
            if generation_id in self._publications:
                raise RuntimeError(
                    f"publication generation already exists: {generation_id}"
                )
            self._expire_prior_publications()
            publication = asyncio.get_running_loop().create_future()
            publication.add_done_callback(consume_future_exception)
            state = _PublicationState(generation_id, publication)
            self._publications[generation_id] = state
            event_port, event_receiver = Channel[dict[str, Any]].open()
            ready_port, ready_receiver = Channel[dict[str, Any]].open()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._command_timeout_s()
            rank_call = asyncio.ensure_future(
                self._actors.execute_snapshot.call(
                    job.model_dump_json(), event_port, ready_port
                )
            )
            readiness_progress: set[int] = set()
            readiness = asyncio.create_task(
                _collect_command_ready(
                    ready_receiver,
                    job,
                    self._rank_processes,
                    label="snapshot",
                    progress=readiness_progress,
                ),
                name=f"megatron-snapshot-ready-{job.operation_id}",
            )
            self._active_job_id = job.operation_id
            self._active_collective = rank_call
            try:
                await _await_snapshot_readiness(
                    rank_call,
                    readiness,
                    self._actors,
                    job,
                    readiness_progress,
                    deadline,
                )
                if job.sequence_continuation_of is None:
                    self._next_operation_sequence += 1
                self._operation_sequence_ids[job.operation_id] = job.sequence_id
                completion = asyncio.create_task(
                    self._complete_snapshot_prepare(
                        job, rank_call, event_receiver, state, deadline
                    ),
                    name=f"megatron-snapshot-plan-{job.operation_id}",
                )
                launch = SnapshotPrepareCommandLaunch(
                    completion=completion, publication=publication
                )
                self._snapshot_launches[job.operation_id] = (
                    job.fingerprint,
                    launch,
                )
                return launch
            except BaseException as error:
                for task in (readiness, rank_call):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(readiness, rank_call, return_exceptions=True)
                if not publication.done():
                    publication.set_exception(error)
                state.records.clear()
                state.train_done = True
                await self._invalidate_command(error, "snapshot and cleanup failed")
                raise
            finally:
                if self._active_job_id == job.operation_id:
                    self._active_job_id = None
                    self._active_collective = None
                self._retire_publication(state)

    async def snapshot(self, job: GenerationSnapshotJobSpec) -> dict[str, Any]:
        launch = await self.start_snapshot(job)
        return await asyncio.shield(launch.completion)

    async def start_snapshot(
        self, job: GenerationSnapshotJobSpec
    ) -> SnapshotPrepareCommandLaunch:
        prepared = await self.start_prepare_snapshot(job)

        async def complete() -> dict[str, Any]:
            result = await asyncio.shield(prepared.completion)
            plan = SnapshotWritePlan.model_validate(result["write_plan"])
            metrics = await self.authorize_snapshot(
                plan, SnapshotWriteGrant.local(plan)
            )
            return {**result, "metrics": {**result["metrics"], **metrics}}

        completion = asyncio.create_task(
            complete(), name=f"megatron-snapshot-{job.operation_id}"
        )
        self._snapshot_tasks.add(completion)
        completion.add_done_callback(self._snapshot_tasks.discard)
        completion.add_done_callback(consume_future_exception)
        return SnapshotPrepareCommandLaunch(
            completion=completion, publication=prepared.publication
        )

    async def _complete_snapshot_prepare(
        self,
        job: GenerationSnapshotJobSpec,
        rank_call: asyncio.Future[Any],
        receiver: Any,
        state: _PublicationState,
        deadline: float,
    ) -> dict[str, Any]:
        publication = state.future
        try:
            async with asyncio.timeout_at(deadline):
                values = await asyncio.shield(rank_call)
            results = list(values.values())
            self._validate_rank_command_results(
                results,
                operation_id=job.operation_id,
                learner_version=job.learner_version,
            )
            rank_plans = tuple(
                SnapshotRankWritePlan.model_validate(result["rank_write_plan"])
                for result in results
            )
            plan = build_snapshot_write_plan(
                operation_id=job.operation_id,
                generation=job.generation,
                ranks=rank_plans,
            )
            result = next(result for result in results if result["rank"] == 0)
            state.train_done = True
            state.drain_done = False
            drain = asyncio.create_task(self._drain_publication(receiver, state))
            self._publication_drains.add(drain)
            drain.add_done_callback(self._publication_drains.discard)
            drain.add_done_callback(consume_future_exception)
            result = {**result, "write_plan": plan.model_dump(mode="json")}
            self._operations[job.operation_id] = (job.fingerprint, result)
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            return result
        except BaseException as error:
            if not publication.done():
                publication.set_exception(error)
            state.records.clear()
            state.train_done = True
            self._snapshot_launches.pop(job.operation_id, None)
            await self._invalidate_command(error, "snapshot and cleanup failed")
            raise
        finally:
            self._retire_publication(state)

    async def authorize_snapshot(
        self, plan: SnapshotWritePlan, grant: SnapshotWriteGrant
    ) -> dict[str, float]:
        grant.validate_plan(plan)
        prepared = self._operations.get(plan.operation_id)
        if (
            prepared is None
            or SnapshotWritePlan.model_validate(prepared[1]["write_plan"]) != plan
        ):
            raise RuntimeError("snapshot authorization differs from its prepared plan")
        state = self._publications.get(plan.generation.generation_id)
        if state is None:
            raise RuntimeError(
                f"trainer has no prepared publication {plan.operation_id}"
            )
        if state.authorized.is_set():
            return {}
        values = await asyncio.wait_for(
            self._actors.authorize_snapshot.call(
                plan.model_dump_json(), grant.model_dump_json()
            ),
            timeout=self._command_timeout_s(),
        )
        results = list(values.values())
        if {result["rank"] for result in results} != set(
            range(len(self._rank_processes))
        ) or {result["operation_id"] for result in results} != {plan.operation_id}:
            raise RuntimeError("trainer ranks returned an invalid authorization")
        state.authorized.set()
        return next(result for result in results if result["rank"] == 0)["metrics"]

    def _validate_operation(
        self,
        job: (
            ForwardJobSpec
            | ForwardBackwardJobSpec
            | SftForwardBackwardJobSpec
            | SftForwardJobSpec
            | OptimizerJobSpec
            | LoadStateJobSpec
            | GenerationSnapshotJobSpec
        ),
    ) -> None:
        if self._closed or not self._valid:
            raise RuntimeError("trainer runtime is invalid")
        if self._jobs:
            raise RuntimeError(
                "fused train jobs cannot be mixed with command operations"
            )
        if job.run_id != self.run_spec.run_id:
            raise ValueError("operation run_id does not match this training run")
        if job.training_session_id != self.run_spec.training_session_id:
            raise ValueError("operation training_session_id does not match this run")
        continuation = (
            job.sequence_continuation_of
            if isinstance(job, GenerationSnapshotJobSpec)
            else None
        )
        expected_sequence = (
            self._operation_sequence_ids.get(continuation)
            if continuation is not None
            else self._next_operation_sequence
        )
        if expected_sequence is None:
            raise RuntimeError(
                f"snapshot continuation parent is not retained: {continuation}"
            )
        if (
            continuation is not None
            and expected_sequence != self._next_operation_sequence - 1
        ):
            raise RuntimeError(
                "snapshot must continue the immediately preceding command"
            )
        if job.sequence_id != expected_sequence:
            raise RuntimeError(
                "trainer operation sequence must be gapless: "
                f"expected={expected_sequence}, got={job.sequence_id}"
            )
        learner_parent = (
            job.learner_version
            if isinstance(job, GenerationSnapshotJobSpec)
            else job.expected_learner_version
        )
        if learner_parent != self._learner_version:
            raise ValueError(
                "operation learner parent mismatch: "
                f"operation={learner_parent}, "
                f"runtime={self._learner_version}"
            )

    def _command_timeout_s(self) -> float:
        if (
            self._next_operation_sequence == 0
            and self.run_spec.initial_event_timeout_s is not None
        ):
            return self.run_spec.initial_event_timeout_s
        return self.run_spec.event_timeout_s

    def _validate_rank_command_results(
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

    async def _invalidate_command(self, error: BaseException, message: str) -> None:
        self._valid = False
        self._closed = True
        await cleanup_after_failure(error, self._force_stop, message=message)

    async def prepare_cp_lookahead(
        self,
        batch: PackedBatchLeaseSet,
        *,
        global_grad_accumulation_sequences: int | None,
    ) -> dict[str, float]:
        if self._closed or not self._valid:
            raise RuntimeError("trainer run is not available for CP lookahead")
        return await _prepare_cp_lookahead(
            self._cp_lookahead_ports,
            self._cp_lookahead_lock,
            batch,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            timeout_s=self.run_spec.event_timeout_s,
        )

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
                        "publication_progress",
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

    def retire_operation(self, operation_id: str) -> None:
        self._snapshot_launches.pop(operation_id, None)
        self._operations.pop(operation_id, None)
        self._operation_sequence_ids.pop(operation_id, None)
        self._cancelled_operations.pop(operation_id, None)

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
        if isinstance(event, TrainerPublicationProgress):
            return
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
        await state.authorized.wait()
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
                        f"{payload['error_type']}: {payload['message']}\n"
                        f"{payload['traceback']}"
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
        if run_id != self.run_spec.run_id:
            return ValueError("diagnostic run_id does not match this training run")
        if learner.training_session_id != self.run_spec.training_session_id:
            return ValueError("diagnostic learner does not match the training session")
        if learner.policy_step != self._learner_version:
            return ValueError(
                "diagnostic learner version mismatch: "
                f"request={learner.policy_step}, runtime={self._learner_version}"
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
        if (
            job.batch.moe_routing_replay is not None
            and not self.runtime_spec.enable_moe_routing_replay
        ):
            return ValueError(
                "resident score requires unsupported MoE routing replay"
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
        if self._next_operation_sequence:
            return RuntimeError(
                "fused train jobs cannot be mixed with command operations"
            )
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
            graceful = (
                self._valid
                and self._active_job_id is None
                and not self._forward_backward_launches
            )
            self._valid = False
            self._cancel_active()
            self._close_task = asyncio.create_task(self._close(graceful))
            self._close_task.add_done_callback(consume_future_exception)
        await asyncio.shield(self._close_task)

    async def _close(self, graceful: bool) -> None:
        loop = asyncio.get_running_loop()
        shutdown_timeout_s = min(
            self.run_spec.shutdown_timeout_s, process_shutdown_timeout(1)
        )
        deadline = loop.time() + shutdown_timeout_s
        primary: BaseException | None = None
        if graceful:
            publications = tuple(self._publications.values())
            for publication in publications:
                publication.authorized.set()
            graceful_deadline = min(deadline, loop.time() + process_shutdown_timeout(2))
            try:
                async with asyncio.timeout_at(graceful_deadline):
                    if self._snapshot_tasks:
                        await asyncio.gather(*tuple(self._snapshot_tasks))
                    await asyncio.gather(
                        _remote_teardown(
                            self._actors.close.call(
                                max(0.0, graceful_deadline - loop.time())
                            )
                        ),
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
        else:
            self._closed = True
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
                self._supervision.close(
                    suppress_owned_mesh_faults_s=self.run_spec.shutdown_timeout_s
                )
                if not task.cancelled():
                    task.exception()

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
