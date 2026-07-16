from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import wraps
import gc
from multiprocessing import resource_tracker, shared_memory
import os
import socket
import time
import traceback
from typing import Any, cast

# This module is imported only by explicit distributed runtime construction.
from monarch.actor import Actor, endpoint  # ty: ignore[unresolved-import]

from art.utils.lifecycle import complete_task

from .artifact_preflight import (
    ArtifactProbeCommand,
    ArtifactProbeResult,
    execute_artifact_probe,
)
from .data_plane import (
    PackedBatchCapacityError,
    PackedBatchInbox,
    PackedBatchLeaseError,
    PackedBatchRef,
)
from .host_admission import (
    HostAdmissionReport,
    HostAdmissionRequest,
    inspect_host,
)
from .monarch_runtime import RemoteCallError, RemoteCallResult
from .nccl_preflight import (
    NcclPreflightSessionRequest,
    NcclProbeRequest,
    NcclProbeResult,
    NcclRendezvous,
    NcclRendezvousRequest,
    NcclRendezvousResult,
    run_nccl_probe,
    start_nccl_rendezvous,
)
from .packing import PackingRequest, PackingResult
from .rollout import RolloutInvocation, RolloutResult
from .specs import HostServiceHealth
from .trajectory_store import (
    TrajectoryCapacityError,
    TrajectoryEnqueueResult,
    TrajectoryGroupRef,
    TrajectoryLeaseError,
    TrajectoryQueueItem,
    TrajectoryQueueResize,
    TrajectoryQueueSnapshot,
    TrajectoryQueueStore,
    TrajectoryQueueTake,
)
from .vllm_replica import HostMemberLaunchRequest


def resilient_endpoint(function: Any) -> Any:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> RemoteCallResult:
        try:
            return RemoteCallResult(value=await function(*args, **kwargs))
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as error:
            from art.errors import LocalServingUnavailableError

            if isinstance(error, asyncio.CancelledError):
                kind = "cancelled"
            elif isinstance(error, LocalServingUnavailableError):
                kind = "serving"
            elif isinstance(error, PackedBatchCapacityError | TrajectoryCapacityError):
                kind = "capacity"
            elif isinstance(error, PackedBatchLeaseError | TrajectoryLeaseError):
                kind = "lease"
            elif isinstance(error, (TypeError, ValueError)):
                kind = "input"
            else:
                kind = "internal"
            return RemoteCallResult(
                error=RemoteCallError(
                    kind=kind,
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    traceback=traceback.format_exc(),
                )
            )

    return endpoint(wrapped)


class RolloutHostService(Actor):
    """One packed-batch and managed-service owner per host."""

    def __init__(
        self,
        admission_json: str,
        packed_batch_capacity_bytes: int,
        vllm_output_root: str = "/tmp/art-vllm",
    ) -> None:
        admission = HostAdmissionRequest.model_validate_json(admission_json)
        self.host_id = admission.host_id
        self._admission = admission
        self._admission_report: HostAdmissionReport | None = None
        self.inbox = PackedBatchInbox(
            host_id=self.host_id, capacity_bytes=packed_batch_capacity_bytes
        )
        self._rdma_batches: dict[str, tuple[Any, Any, shared_memory.SharedMemory]] = {}
        self._trajectory_queues: dict[str, TrajectoryQueueStore] = {}
        self._packer = None
        self._vllm_output_root = vllm_output_root
        self._vllm_launcher = None
        self._nccl_cleanups: dict[str, asyncio.Task[None]] = {}
        self._nccl_rendezvous: dict[str, NcclRendezvous] = {}
        self._nccl_sessions: dict[str, tuple[float, asyncio.Task[None]]] = {}
        self._nccl_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancelled_nccl_probes: set[str] = set()

    @resilient_endpoint
    async def admission(self) -> HostAdmissionReport:
        if self._admission_report is None:
            self._admission_report = await asyncio.to_thread(
                inspect_host, self._admission
            )
        return self._admission_report

    @resilient_endpoint
    async def health(self) -> HostServiceHealth:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        return HostServiceHealth(
            host_id=self.host_id,
            hostname=socket.gethostname(),
            process_id=os.getpid(),
        )

    @resilient_endpoint
    async def artifact_root_probe(
        self, command: ArtifactProbeCommand
    ) -> ArtifactProbeResult:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        return await asyncio.to_thread(execute_artifact_probe, self.host_id, command)

    @resilient_endpoint
    async def start_nccl_preflight_session(
        self, request: NcclPreflightSessionRequest
    ) -> None:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        if request.probe_id in self._cancelled_nccl_probes:
            raise asyncio.CancelledError
        if request.probe_id in self._nccl_sessions:
            raise RuntimeError(f"NCCL probe {request.probe_id!r} is already admitted")
        deadline = time.monotonic() + request.lease_s
        reaper = asyncio.create_task(
            self._expire_nccl_preflight_session(request.probe_id, deadline)
        )
        self._nccl_sessions[request.probe_id] = (deadline, reaper)

    @resilient_endpoint
    async def nccl_preflight_rendezvous(
        self, request: NcclRendezvousRequest
    ) -> NcclRendezvousResult:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        deadline = await self._require_nccl_probe(request.probe_id)
        if request.probe_id in self._nccl_rendezvous:
            raise RuntimeError(f"NCCL probe {request.probe_id!r} already has a store")
        task = asyncio.create_task(start_nccl_rendezvous(request, deadline_s=deadline))
        self._nccl_tasks[request.probe_id] = task
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                rendezvous = await task
        finally:
            if self._nccl_tasks.get(request.probe_id) is task:
                self._nccl_tasks.pop(request.probe_id)
        if request.probe_id in self._cancelled_nccl_probes:
            await rendezvous.close()
            raise asyncio.CancelledError
        self._nccl_rendezvous[request.probe_id] = rendezvous
        return NcclRendezvousResult(host_id=self.host_id, port=rendezvous.port)

    @resilient_endpoint
    async def nccl_preflight(self, request: NcclProbeRequest) -> NcclProbeResult:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        deadline = await self._require_nccl_probe(request.probe_id)
        task = asyncio.create_task(run_nccl_probe(self.host_id, request))
        self._nccl_tasks[request.probe_id] = task
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                return await task
        finally:
            if self._nccl_tasks.get(request.probe_id) is task:
                self._nccl_tasks.pop(request.probe_id)

    @resilient_endpoint
    async def cancel_nccl_preflight(self, probe_id: str) -> None:
        await self._cancel_nccl_preflight(probe_id)

    @resilient_endpoint
    async def close(self) -> None:
        probe_ids = tuple(
            {
                *self._nccl_cleanups,
                *self._nccl_sessions,
                *self._nccl_tasks,
                *self._nccl_rendezvous,
            }
        )
        await asyncio.gather(
            *(self._cancel_nccl_preflight(probe_id) for probe_id in probe_ids)
        )
        for queue in self._trajectory_queues.values():
            queue.close()
        self._trajectory_queues.clear()
        for batch_id in tuple(self._rdma_batches):
            await self._drop_batch(batch_id)
        if self._packer is not None:
            await self._packer.close()
            self._packer = None
        if self._vllm_launcher is not None:
            await self._vllm_launcher.close()
            self._vllm_launcher = None
        self.inbox.store.close()

    async def _require_nccl_probe(self, probe_id: str) -> float:
        if probe_id in self._cancelled_nccl_probes:
            raise asyncio.CancelledError
        session = self._nccl_sessions.get(probe_id)
        if session is None:
            raise RuntimeError(f"NCCL probe {probe_id!r} has no active session")
        deadline, _ = session
        if time.monotonic() >= deadline:
            await self._cancel_nccl_preflight(probe_id)
            raise TimeoutError(f"NCCL probe {probe_id!r} session expired")
        if probe_id in self._nccl_tasks:
            raise RuntimeError(f"NCCL probe {probe_id!r} is already active")
        return deadline

    async def _cancel_nccl_preflight(self, probe_id: str) -> None:
        self._cancelled_nccl_probes.add(probe_id)
        cleanup = self._nccl_cleanups.get(probe_id)
        if cleanup is None:
            cleanup = asyncio.create_task(
                self._cleanup_nccl_preflight(probe_id, asyncio.current_task())
            )
            self._nccl_cleanups[probe_id] = cleanup
        try:
            _, cancelled = await complete_task(cleanup)
        finally:
            if cleanup.done() and self._nccl_cleanups.get(probe_id) is cleanup:
                self._nccl_cleanups.pop(probe_id)
        if cancelled is not None:
            raise cancelled

    async def _cleanup_nccl_preflight(
        self, probe_id: str, owner: asyncio.Task[Any] | None
    ) -> None:
        session = self._nccl_sessions.pop(probe_id, None)
        if session is not None and session[1] is not owner:
            session[1].cancel()
            await asyncio.gather(session[1], return_exceptions=True)
        task = self._nccl_tasks.pop(probe_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        rendezvous = self._nccl_rendezvous.pop(probe_id, None)
        if rendezvous is not None:
            await rendezvous.close()

    async def _expire_nccl_preflight_session(
        self, probe_id: str, deadline: float
    ) -> None:
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        if self._nccl_sessions.get(probe_id, (None,))[0] != deadline:
            return
        self._cancelled_nccl_probes.add(probe_id)
        if probe_id in self._nccl_cleanups:
            return
        cleanup = asyncio.current_task()
        assert cleanup is not None
        self._nccl_cleanups[probe_id] = cleanup
        try:
            await self._cleanup_nccl_preflight(probe_id, cleanup)
        finally:
            if self._nccl_cleanups.get(probe_id) is cleanup:
                self._nccl_cleanups.pop(probe_id)

    @resilient_endpoint
    async def create_trajectory_queue(
        self,
        queue_id: str,
        max_ready_groups: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None:
        if queue_id in self._trajectory_queues:
            raise ValueError(f"trajectory queue {queue_id!r} already exists")
        self._trajectory_queues[queue_id] = TrajectoryQueueStore(
            max_ready_groups=max_ready_groups,
            capacity_records=capacity_records,
            capacity_bytes=capacity_bytes,
        )

    @resilient_endpoint
    async def resize_trajectory_queue(self, operation: TrajectoryQueueResize) -> None:
        self._trajectory_queue(operation.queue_id).resize(
            maxsize=operation.maxsize, generation=operation.generation
        )

    @resilient_endpoint
    async def enqueue_trajectory(
        self, queue_id: str, item: TrajectoryQueueItem
    ) -> TrajectoryEnqueueResult:
        return self._trajectory_queue(queue_id).enqueue(item)

    @resilient_endpoint
    async def take_trajectory(
        self, queue_id: str, consumer_id: str
    ) -> TrajectoryQueueTake:
        return self._trajectory_queue(queue_id).take(consumer_id)

    @resilient_endpoint
    async def acknowledge_trajectory(
        self, queue_id: str, result_id: str, consumer_id: str
    ) -> None:
        self._trajectory_queue(queue_id).acknowledge(result_id, consumer_id)

    @resilient_endpoint
    async def finish_trajectory_queue(self, queue_id: str) -> None:
        self._trajectory_queue(queue_id).finish()

    @resilient_endpoint
    async def trajectory_queue_snapshot(self, queue_id: str) -> TrajectoryQueueSnapshot:
        return self._trajectory_queue(queue_id).snapshot()

    @resilient_endpoint
    async def close_trajectory_queue(
        self, queue_id: str
    ) -> tuple[TrajectoryGroupRef, ...]:
        queue = self._trajectory_queues.pop(queue_id, None)
        return () if queue is None else queue.close()

    def _trajectory_queue(self, queue_id: str) -> TrajectoryQueueStore:
        try:
            return self._trajectory_queues[queue_id]
        except KeyError:
            raise ValueError(f"unknown trajectory queue {queue_id!r}") from None

    def _launcher(self):
        if self._vllm_launcher is None:
            from .vllm_replica import ManagedVllmHostLauncher

            self._vllm_launcher = ManagedVllmHostLauncher(
                self._vllm_output_root,
                install_parent_cleanup=lambda: None,
            )
        return self._vllm_launcher

    @resilient_endpoint
    async def start_vllm_member(self, request: HostMemberLaunchRequest):
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        return await self._launcher().start_member(request)

    @resilient_endpoint
    async def vllm_member_state(self, replica_id: str, member_id: str, generation: int):
        return await self._launcher().member_state(replica_id, member_id, generation)

    @resilient_endpoint
    async def stop_vllm_member(
        self, replica_id: str, member_id: str, generation: int
    ) -> None:
        if self._vllm_launcher is not None:
            await self._vllm_launcher.stop_member(replica_id, member_id, generation)

    @resilient_endpoint
    async def pack_batch(self, request: PackingRequest, batch_id: str) -> PackingResult:
        if self._packer is None:
            from art.megatron.backend import MegatronBackend

            self._packer = MegatronBackend(
                in_process=True,
                path=f"/tmp/art-packing-{os.getpid()}",
                enable_expert_replay=request.include_moe_routing,
            )
        groups = [payload.build() for payload in request.trajectory_groups]
        packed = self._packer._get_packed_tensors(
            request.model.build(),
            groups,
            advantage_balance=request.advantage_balance,
            allow_training_without_logprobs=request.allow_training_without_logprobs,
            scale_rewards=request.scale_rewards,
            plot_tensors=request.plot_tensors,
            packed_sequence_length=request.packed_sequence_length,
            logprob_calculation_chunk_size=request.logprob_calculation_chunk_size,
            include_moe_routing=request.include_moe_routing,
        )
        shapes = tuple(group._packed_group_shape for group in groups)
        if packed is None:
            return PackingResult(ref=None, packed_group_shapes=shapes)
        trainable_assistant_tokens = int(packed["assistant_mask"].sum().item())
        non_padding_tokens = int((packed["group_ids"] != -1).sum().item())
        ref = self.inbox.store.create(
            packed,
            batch_id=batch_id,
            group_ids=request.group_ids,
            record_ids=request.record_ids,
            min_source_version=request.min_source_version,
            max_source_version=request.max_source_version,
        )
        return PackingResult(
            ref=ref,
            packed_group_shapes=shapes,
            trainable_assistant_tokens=trainable_assistant_tokens,
            non_padding_tokens=non_padding_tokens,
        )

    @resilient_endpoint
    async def publish_batch(self, ref: PackedBatchRef):
        if ref.batch_id in self._rdma_batches:
            raise RuntimeError(f"packed batch {ref.batch_id!r} is already published")
        from monarch.rdma import RDMABuffer  # ty: ignore[unresolved-import]
        import torch

        shm = shared_memory.SharedMemory(name=ref.shared_memory_name)
        if ref.owner_process_id != os.getpid():
            resource_tracker.unregister(cast(Any, shm)._name, "shared_memory")
        try:
            tensor = torch.frombuffer(
                shm.buf, dtype=torch.uint8, count=ref.storage_byte_count
            )
            handle = RDMABuffer(tensor)
        except BaseException:
            shm.close()
            raise
        self._rdma_batches[ref.batch_id] = (handle, tensor, shm)
        return handle

    @resilient_endpoint
    async def drop_batch(self, batch_id: str) -> None:
        await self._drop_batch(batch_id)

    @resilient_endpoint
    async def note_batch_transmitted(self, byte_count: int) -> None:
        self.inbox.store.note_transmitted(byte_count)

    async def _drop_batch(self, batch_id: str) -> bool:
        published = self._rdma_batches.pop(batch_id, None)
        if published is None:
            return False
        handle, tensor, shm = published
        try:
            await handle.drop()
        finally:
            del tensor
            gc.collect()
            shm.close()
        return True

    @resilient_endpoint
    async def receive_rdma_batch(
        self, ref: PackedBatchRef, rdma_buffer: Any, timeout_s: float
    ) -> PackedBatchRef:
        return await self.inbox.receive_rdma(ref, rdma_buffer, timeout_s=timeout_s)

    @resilient_endpoint
    async def drop_batch_ref(self, ref: PackedBatchRef) -> None:
        await self.inbox.drop(ref)

    @resilient_endpoint
    async def reclaim_batch(self, batch_id: str, fence: bool) -> bool:
        published = False
        failure: BaseException | None = None
        try:
            published = await self._drop_batch(batch_id)
        except BaseException as error:
            failure = error
        reclaimed = self.inbox.store.reclaim(batch_id, fence=fence)
        if failure is not None:
            raise failure
        return published or reclaimed

    @resilient_endpoint
    async def stats(self):
        return self.inbox.store.stats()


class RolloutWorkerService(Actor):
    """One process-isolated CPU rollout slot."""

    def __init__(self, capacity_records: int, capacity_bytes: int) -> None:
        from .trajectory_store import TrajectoryRecordStore

        self._models = OrderedDict()
        self._results = TrajectoryRecordStore(
            owner_actor_id=f"rollout:{socket.gethostname()}:{os.getpid()}",
            capacity_records=capacity_records,
            capacity_bytes=capacity_bytes,
        )

    @resilient_endpoint
    async def run(self, invocation: RolloutInvocation):
        from art.metrics import MetricsBuilder

        key = invocation.model.cache_key
        model = self._models.get(key)
        if model is None:
            model = invocation.model.build()
            self._models[key] = model
            if len(self._models) > 16:
                _, evicted = self._models.popitem(last=False)
                await evicted._reset_inference_runtime()
        else:
            self._models.move_to_end(key)
        builder = MetricsBuilder(cost_context="train")
        token = builder.activate()
        try:
            value = await invocation.callable.resolve()(
                model, invocation.scenario, invocation.config
            )
        finally:
            token.var.reset(token)
        if invocation.store_result:
            from art import TrajectoryGroup

            if isinstance(value, TrajectoryGroup):
                value = self._results.put(value)
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    @resilient_endpoint
    async def materialize_result(self, ref: TrajectoryGroupRef):
        return self._results.payload(ref)

    @resilient_endpoint
    async def drop_result(self, ref: TrajectoryGroupRef) -> None:
        self._results.drop(ref)

    @resilient_endpoint
    async def close(self) -> None:
        for model in self._models.values():
            await model._reset_inference_runtime()
        self._models.clear()
        self._results.close()
