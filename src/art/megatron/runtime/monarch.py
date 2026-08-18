from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
import hashlib
import json
import os
from pathlib import Path
import socket
from threading import Event, Lock, Thread
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
from art.utils.cache_dirs import configure_model_cache_env
from art.utils.lifecycle import cleanup_after_failure, consume_future_exception

from .data_plane import InMemoryPackedBatch, SFTBatchData
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
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    GenerationSnapshotJobSpec,
    HybridEpRuntimeSpec,
    LoadStateJobSpec,
    OptimizerJobSpec,
    ResidentLoraExport,
    ResidentLoraInspectionResult,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentLoraRankSummary,
    ResidentScoreJobSpec,
    ResidentScoreResult,
    ResidentScoreShard,
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


def _configure_hybrid_ep_env(
    spec: HybridEpRuntimeSpec, *, run_id: str | None = None
) -> None:
    os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(
        spec.ranks_per_nvlink_domain
    )
    transport = spec.nixl_transport
    values = {
        "HYBRID_EP_MULTINODE": "1" if transport else None,
        "USE_NIXL": "1" if transport else None,
        "DEEPEP_NIXL_RUN_ID": (run_id or spec.run_id) if transport else None,
        "NIXL_ETCD_ENDPOINTS": transport.metadata_store.url if transport else None,
        "NIXL_HOME": transport.nixl_home if transport else None,
        "UCX_HOME": transport.ucx_home if transport else None,
        "NIXL_PLUGIN_DIR": transport.nixl_plugin_dir if transport else None,
        "UCX_MODULE_DIR": transport.ucx_module_dir if transport else None,
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
    run_id: str
    command_kind: str
    admitted: bool = True
    elapsed_s: float = 0.0
    error_type: str | None = None
    message: str | None = None
    traceback_text: str | None = None


async def _prepare_run_residency(
    ports: tuple[Port[Any], ...],
    lock: asyncio.Lock,
    run_id: str,
    command_kind: str,
    *,
    timeout_s: float,
) -> dict[str, float]:
    async with lock:
        reply, receiver = Channel.open()
        request = run_id, command_kind, reply
        started = time.perf_counter()
        for port in ports:
            port.send(request)
        async with asyncio.timeout(timeout_s):
            results = [
                _ResidencyPrefetchResult.model_validate(await receiver.recv())
                for _ in ports
            ]
        if {result.rank for result in results} != set(range(len(ports))) or any(
            (result.run_id, result.command_kind) != (run_id, command_kind)
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
        self._command_job_open = False
        self._cp_preplanner = None
        self._cp_lookahead_port = None
        self._cp_lookahead_thread = None
        self._residency_prefetch_port = None
        self._residency_prefetch_thread = None
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
            run_id, command_kind, reply = request
            started = time.perf_counter()
            try:
                admitted = self._require_run_slot_executor().prepare_residency(
                    run_id, command_kind
                )
                result = _ResidencyPrefetchResult(
                    rank=self._runtime.rank,
                    run_id=run_id,
                    command_kind=command_kind,
                    admitted=admitted,
                    elapsed_s=time.perf_counter() - started,
                )
            except BaseException as error:
                result = _ResidencyPrefetchResult(
                    rank=self._runtime.rank,
                    run_id=run_id,
                    command_kind=command_kind,
                    elapsed_s=time.perf_counter() - started,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
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

    def _stop_residency_prefetch(self) -> None:
        thread, self._residency_prefetch_thread = (
            self._residency_prefetch_thread,
            None,
        )
        if thread is None:
            return
        port = self._residency_prefetch_port
        if port is None:
            raise RuntimeError("residency prefetch thread has no request port")
        port.send(None)
        thread.join(timeout=30.0)
        if thread.is_alive():
            raise RuntimeError(
                "residency prefetch service did not stop within 30 seconds"
            )

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
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise
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
    def register_run_slot(self, registration_json: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        registration = RunSlotRegistration.model_validate_json(registration_json)
        executor = self._require_run_slot_executor()
        executor.register_run(
            tenant_id=registration.tenant_id,
            run_id=registration.run_id,
            training_session_id=registration.training_session_id,
            learner_version=registration.learner_version,
            generation_id=registration.generation_id,
            adapter_path=registration.adapter_path,
        )
        if registration.initial_optimizer_state_path is not None:
            assert registration.initial_optimizer_generation_id is not None
            from art.megatron.optimizer_state import (
                load_trainer_rank_optimizer_state,
            )

            optimizer_state = load_trainer_rank_optimizer_state(
                self._runtime,
                optimizer_state_path=registration.initial_optimizer_state_path,
                adapter_path=registration.adapter_path,
                adapter_step=registration.adapter_step,
                optimizer_generation_id=registration.initial_optimizer_generation_id,
                layout=executor.optimizer_layout(registration.run_id),
            )
            executor.restore_optimizer_state(registration.run_id, optimizer_state)
        executor.complete_run_registration(registration.run_id)
        return {"rank": self._runtime.rank, "run_id": registration.run_id}

    @endpoint
    def unregister_run_slot(self, run_id: str) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        self._require_run_slot_executor().unregister_run(run_id)
        return {"rank": self._runtime.rank, "run_id": run_id}

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
            result = self._require_run_slot_executor().execute_forward_backward(
                job, batch, Event()
            )
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise
        finally:
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
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "metrics": result["metrics"] if coordinator else {},
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise
        finally:
            if batch is not None:
                batch.close()

    @endpoint
    def execute_run_slot_sft_forward_backward(
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
            result = self._require_run_slot_executor().execute_sft_forward_backward(
                job, batch, Event()
            )
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def execute_run_slot_sft_forward(
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
            result = self._require_run_slot_executor().execute_sft_forward(
                job, batch, Event()
            )
            coordinator = self._runtime.rank == 0
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "token_count": result["token_count"],
                "metrics": result["metrics"] if coordinator else {},
                "token_logprobs": result["token_logprobs"] if coordinator else (),
            }
        except BaseException:
            self._valid = False
            raise

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
    def start_prepare_run_slot_load_state(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = LoadStateJobSpec.model_validate_json(job_json)
            self._require_run_slot_executor().start_prepare_load_state(job)
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
            }
        except BaseException:
            self._valid = False
            raise

    @endpoint
    def poll_prepare_run_slot_load_state(self, job_json: str) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = LoadStateJobSpec.model_validate_json(job_json)
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "ready": self._require_run_slot_executor().load_state_prepared(job),
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
    def execute_run_slot_snapshot(
        self,
        job_json: str,
        event_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = GenerationSnapshotJobSpec.model_validate_json(job_json)
            coordinator = self._runtime.rank == 0
            result = self._require_run_slot_executor().execute_snapshot(
                job,
                _ActorEventSink(event_port, coordinator=coordinator),
            )
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "metrics": result["metrics"] if coordinator else {},
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
    def execute_snapshot(
        self,
        job_json: str,
        event_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not self._valid:
                raise RuntimeError("trainer actor runtime is invalid")
            job = GenerationSnapshotJobSpec.model_validate_json(job_json)
            if not self._command_job_open:
                self._weight_offload.before_job()
                self._command_job_open = True
            coordinator = self._runtime.rank == 0
            result = self._executor.execute_snapshot(
                job,
                _ActorEventSink(event_port, coordinator=coordinator),
            )
            return {
                "rank": self._runtime.rank,
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "metrics": result["metrics"] if coordinator else {},
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
    def close(self) -> None:
        if self._command_job_open:
            self._executor.discard_open_gradients()
            self._weight_offload.after_job()
            self._command_job_open = False
        self._stop_residency_prefetch()
        if self._run_slot_executor is not None:
            self._run_slot_executor.close()
        self._stop_cp_lookahead()
        self._executor.close()
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

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
        if self._command_job_open:
            self._executor.discard_open_gradients()
            self._weight_offload.after_job()
            self._command_job_open = False
        self._stop_residency_prefetch()
        if self._run_slot_executor is not None:
            self._run_slot_executor.close()
        self._stop_cp_lookahead()
        self._executor.close()


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
        self._closed = False
        self._valid = True
        self._stop_task: asyncio.Task[None] | None = None
        self._publications: dict[
            str, asyncio.Task[tuple[TrainerRankPublication, ...]]
        ] = {}
        self._registrations: dict[str, str] = {}
        self._removed_runs: set[str] = set()
        self._operations: dict[str, tuple[str, dict[str, Any]]] = {}

    @property
    def valid(self) -> bool:
        return self._valid

    async def register_run(self, registration: RunSlotRegistration) -> None:
        async with self._lock:
            self._require_open()
            fingerprint = registration.model_dump_json()
            if prior := self._registrations.get(registration.run_id):
                if prior != fingerprint:
                    raise RuntimeError("run_id was reused for another registration")
                return
            if registration.run_id in self._removed_runs:
                raise RuntimeError("removed run_id cannot be registered again")
            try:
                values = await asyncio.wait_for(
                    self._actors.register_run_slot.call(registration.model_dump_json()),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                if {result["rank"] for result in results} != set(
                    range(len(self._rank_processes))
                ) or {result["run_id"] for result in results} != {registration.run_id}:
                    raise RuntimeError("trainer ranks disagree on run registration")
                self._registrations[registration.run_id] = fingerprint
            except BaseException as error:
                await self._invalidate(error, "run registration and cleanup failed")
                raise

    async def unregister_run(self, run_id: str) -> None:
        async with self._lock:
            self._require_open()
            if run_id in self._removed_runs:
                return
            if run_id not in self._registrations:
                raise RuntimeError(f"training run is not registered: {run_id!r}")
            try:
                values = await asyncio.wait_for(
                    self._actors.unregister_run_slot.call(run_id),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                if {result["rank"] for result in results} != set(
                    range(len(self._rank_processes))
                ) or {result["run_id"] for result in results} != {run_id}:
                    raise RuntimeError("trainer ranks disagree on run removal")
                self._registrations.pop(run_id)
                self._removed_runs.add(run_id)
            except BaseException as error:
                await self._invalidate(error, "run removal and cleanup failed")
                raise

    async def forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            if batch.ref != job.batch:
                raise ValueError("F/B batch ref does not match supplied packed batch")
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_forward_backward.call(
                        job.model_dump_json(), batch.model_dump_json()
                    ),
                    timeout=self._command_timeout_s,
                )
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
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "F/B operation and cleanup failed")
                raise

    async def forward(
        self,
        job: ForwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            if batch.ref != job.batch:
                raise ValueError(
                    "forward batch ref does not match supplied packed batch"
                )
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_forward.call(
                        job.model_dump_json(), batch.model_dump_json()
                    ),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                self._validate_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.expected_learner_version,
                )
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "forward operation and cleanup failed")
                raise

    async def sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            if batch.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT F/B payload fingerprint differs from its job")
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_sft_forward_backward.call(
                        job.model_dump_json(), batch
                    ),
                    timeout=self._command_timeout_s,
                )
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
                        "trainer SFT F/B token count differs from its payload"
                    )
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "SFT F/B operation and cleanup failed")
                raise

    async def sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
    ) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            if batch.fingerprint != job.batch_fingerprint:
                raise ValueError("SFT forward payload fingerprint differs from its job")
            try:
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_sft_forward.call(
                        job.model_dump_json(), batch
                    ),
                    timeout=self._command_timeout_s,
                )
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
                        "trainer SFT forward token count differs from its payload"
                    )
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(
                    error, "SFT forward operation and cleanup failed"
                )
                raise

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
            started = list(
                (
                    await asyncio.wait_for(
                        self._actors.start_prepare_run_slot_load_state.call(payload),
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
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("load preparation exceeded command timeout")
                polled = list(
                    (
                        await asyncio.wait_for(
                            self._actors.poll_prepare_run_slot_load_state.call(payload),
                            timeout=remaining,
                        )
                    ).values()
                )
                self._validate_command_results(
                    polled,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                if all(bool(result["ready"]) for result in polled):
                    return
                await asyncio.sleep(min(0.01, remaining))
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

    async def snapshot(self, job: GenerationSnapshotJobSpec) -> dict[str, Any]:
        async with self._control_lock:
            self._require_open()
            cached = self._cached_operation(job.operation_id, job.fingerprint)
            if cached is not None:
                return cached
            operation_id = job.operation_id
            if operation_id in self._publications:
                raise RuntimeError(f"publication already exists: {operation_id}")
            try:
                send_port, receiver = Channel[dict[str, Any]].open()
                values = await asyncio.wait_for(
                    self._actors.execute_run_slot_snapshot.call(
                        job.model_dump_json(), send_port
                    ),
                    timeout=self._command_timeout_s,
                )
                results = list(values.values())
                self._validate_command_results(
                    results,
                    operation_id=job.operation_id,
                    learner_version=job.learner_version,
                )
                publication = asyncio.create_task(
                    self._collect_publication(receiver, job.generation),
                    name=f"megatron-slot-publish-{operation_id}",
                )
                publication.add_done_callback(consume_future_exception)
                self._publications[operation_id] = publication
                result = next(result for result in results if result["rank"] == 0)
                self._operations[job.operation_id] = (job.fingerprint, result)
                return result
            except BaseException as error:
                await self._invalidate(error, "snapshot operation and cleanup failed")
                raise

    def wait_for_publication(
        self, operation_id: str
    ) -> Awaitable[tuple[TrainerRankPublication, ...]]:
        try:
            publication = self._publications[operation_id]
        except KeyError as exc:
            raise RuntimeError(f"trainer has no publication {operation_id}") from exc
        return asyncio.shield(publication)

    def retire_operation(self, operation_id: str) -> None:
        publication = self._publications.get(operation_id)
        if publication is not None:
            if not publication.done():
                raise RuntimeError("cannot retire an active trainer publication")
            self._publications.pop(operation_id)
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
        self, run_id: str, command_kind: str
    ) -> dict[str, float]:
        self._require_open()
        if not self._residency_prefetch_ports:
            raise RuntimeError("trainer slot has no residency prefetch service")
        return await _prepare_run_residency(
            self._residency_prefetch_ports,
            self._residency_prefetch_lock,
            run_id,
            command_kind,
            timeout_s=self._command_timeout_s,
        )

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
    ) -> tuple[TrainerRankPublication, ...]:
        records: dict[int, TrainerRankPublication] = {}
        supervision = asyncio.create_task(self._supervision.wait())
        receive: asyncio.Future[Any] | None = None
        try:
            while len(records) < len(self._rank_processes):
                receive = asyncio.ensure_future(receiver.recv())
                done, _ = await asyncio.wait(
                    {receive, supervision},
                    timeout=self._shutdown_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError("trainer ranks produced no publication event")
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
        self._closed = True
        self._valid = False
        primary: BaseException | None = None
        try:
            async with asyncio.timeout(self._shutdown_timeout_s * 0.8):
                await _remote_teardown(self._actors.close.call())
                await asyncio.gather(*self._publications.values())
        except BaseException as error:
            primary = error
        try:
            await self._force_stop()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    f"trainer slot cleanup failed: {type(error).__name__}: {error}"
                )
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
        self._closed = True
        await cleanup_after_failure(error, self._force_stop, message=message)

    async def _force_stop(self) -> None:
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
            asyncio.shield(self._stop_task), self._shutdown_timeout_s
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
        self._operation_sequence_ids: dict[str, int] = {}
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
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for a different F/B")
                return cached[1]
            self._validate_operation(job)
            if batch.ref != job.batch:
                raise ValueError("F/B batch ref does not match supplied packed batch")
            if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
                raise ValueError(
                    "packed batch sequence length does not match the trainer runtime"
                )
            return await self._run_forward_backward(job, batch)

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
            result = next(result for result in results if result["rank"] == 0)
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

    async def _run_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: PackedBatchLeaseSet,
    ) -> dict[str, Any]:
        collective = asyncio.ensure_future(
            self._actors.execute_forward_backward.call(
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
            if {result["token_count"] for result in results} != {
                job.trainable_token_count
            }:
                raise RuntimeError(
                    "trainer F/B token count differs from packed policy provenance"
                )
            result = next(result for result in results if result["rank"] == 0)
            self._open_forward_backward_ids.append(job.operation_id)
            self._next_operation_sequence += 1
            self._operations[job.operation_id] = (job.fingerprint, result)
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            return result
        except BaseException as error:
            await self._invalidate_command(error, "F/B operation and cleanup failed")
            raise
        finally:
            if self._active_job_id == job.operation_id:
                self._active_job_id = None
                self._active_collective = None

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
                result = next(item for item in results if item["rank"] == 0)
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

    async def snapshot(self, job: GenerationSnapshotJobSpec) -> dict[str, Any]:
        cached = self._operations.get(job.operation_id)
        if cached is not None and cached[0] == job.fingerprint:
            return cached[1]
        async with self._lock:
            cached = self._operations.get(job.operation_id)
            if cached is not None:
                if cached[0] != job.fingerprint:
                    raise RuntimeError("operation_id was reused for another snapshot")
                return cached[1]
            self._validate_operation(job)
            return await self._run_snapshot(job)

    async def _run_snapshot(self, job: GenerationSnapshotJobSpec) -> dict[str, Any]:
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
        send_port, receiver = Channel[dict[str, Any]].open()
        collective = asyncio.ensure_future(
            self._actors.execute_snapshot.call(job.model_dump_json(), send_port)
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
            result = next(result for result in results if result["rank"] == 0)
            state.train_done = True
            state.drain_done = False
            drain = asyncio.create_task(self._drain_publication(receiver, state))
            self._publication_drains.add(drain)
            drain.add_done_callback(self._publication_drains.discard)
            drain.add_done_callback(consume_future_exception)
            if job.sequence_continuation_of is None:
                self._next_operation_sequence += 1
            self._operations[job.operation_id] = (job.fingerprint, result)
            self._operation_sequence_ids[job.operation_id] = job.sequence_id
            return result
        except BaseException as error:
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
        self._operations.pop(operation_id, None)
        self._operation_sequence_ids.pop(operation_id, None)

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
