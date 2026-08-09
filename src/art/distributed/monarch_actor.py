from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import wraps
import json
import os
from pathlib import Path
import socket
import time
import traceback
from typing import Any, Literal
from urllib.request import urlopen

# This module is imported only by explicit distributed runtime construction.
from monarch.actor import Actor, endpoint  # ty: ignore[unresolved-import]

from art.utils.lifecycle import complete_task

from .adapter_transport import AdapterSnapshotReceiver
from .artifact_preflight import (
    ArtifactProbeCommand,
    ArtifactProbeResult,
    execute_artifact_probe,
)
from .data_plane import (
    ByteStreamServerLoop,
    PackedBatchCapacityError,
    PackedBatchInbox,
    PackedBatchLeaseError,
    PackedBatchPublisher,
    PackedBatchRef,
    PackedBatchTransfer,
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
    TrajectoryQueuePacking,
    TrajectoryQueueRelease,
    TrajectoryQueueResize,
    TrajectoryQueueSnapshot,
    TrajectoryQueueStore,
    TrajectoryQueueTake,
    publish_trajectory_bundles,
)
from .vllm_replica import HostMemberLaunchRequest


def _require_etcd_health(url: str, timeout_s: float) -> None:
    with urlopen(f"{url}/health", timeout=timeout_s) as response:
        health = json.load(response).get("health")
    if health not in (True, "true"):
        raise RuntimeError(f"etcd health check failed: {health!r}")


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


class ArtHostService(Actor):
    """One ART control and data-plane service per host."""

    def __init__(
        self,
        admission_json: str,
        packed_batch_capacity_bytes: int,
        vllm_output_root: str = "/tmp/art-vllm",
        data_plane_host: str | None = None,
    ) -> None:
        admission = HostAdmissionRequest.model_validate_json(admission_json)
        self.host_id = admission.host_id
        self._admission = admission
        self._admission_report: HostAdmissionReport | None = None
        self._packed_batches = PackedBatchInbox(
            host_id=self.host_id, capacity_bytes=packed_batch_capacity_bytes
        )
        self._batch_publishers: dict[str, PackedBatchPublisher] = {}
        self._data_plane_host = data_plane_host or socket.gethostbyname(
            socket.gethostname()
        )
        self._trajectory_queues: dict[str, TrajectoryQueueStore] = {}
        self._packer = None
        self._vllm_output_root = vllm_output_root
        self._vllm_launcher = None
        self._adapter_receiver: AdapterSnapshotReceiver | None = None
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
    async def nixl_metadata_store_health(self, url: str, timeout_s: float) -> str:
        if self._admission_report is None:
            raise RuntimeError("host has not passed ART runtime admission")
        try:
            await asyncio.to_thread(_require_etcd_health, url, timeout_s)
        except BaseException as error:
            raise RuntimeError(
                f"host {self.host_id!r} cannot reach healthy NIXL metadata store {url}"
            ) from error
        return self.host_id

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
        for batch_id in tuple(self._batch_publishers):
            await self._drop_batch(batch_id)
        if self._packer is not None:
            await self._packer.close()
            self._packer = None
        if self._vllm_launcher is not None:
            await self._vllm_launcher.close()
            self._vllm_launcher = None
        if self._adapter_receiver is not None:
            await asyncio.to_thread(self._adapter_receiver.close)
            self._adapter_receiver = None
        self._packed_batches.store.close()

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
        self, queue_id: str, consumer_id: str, count: int
    ) -> TrajectoryQueueTake:
        return self._trajectory_queue(queue_id).take(consumer_id, count)

    @resilient_endpoint
    async def mark_trajectories_packed(self, operation: TrajectoryQueuePacking) -> None:
        self._trajectory_queue(operation.queue_id).mark_packed(operation)

    @resilient_endpoint
    async def release_trajectory(self, operation: TrajectoryQueueRelease) -> None:
        self._trajectory_queue(operation.queue_id).release(operation)

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

    def _receiver(self) -> AdapterSnapshotReceiver:
        if self._adapter_receiver is None:
            self._adapter_receiver = AdapterSnapshotReceiver(
                self.host_id, self._vllm_output_root
            )
        return self._adapter_receiver

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
    async def prepare_adapter_receive(
        self,
        generation_id: str,
        template_path: str,
        tensor_dtype: str,
        timeout_s: float,
        transport: Literal["local", "nixl"],
    ):
        return await asyncio.to_thread(
            self._receiver().prepare,
            generation_id,
            template_path,
            tensor_dtype,
            timeout_s,
            transport,
        )

    @resilient_endpoint
    async def poll_adapter_receive(self, generation_id: str):
        return await asyncio.to_thread(self._receiver().poll, generation_id)

    @resilient_endpoint
    async def release_adapter_receive(self, generation_id: str) -> None:
        if self._adapter_receiver is not None:
            await asyncio.to_thread(self._adapter_receiver.release, generation_id)

    @resilient_endpoint
    async def pack_batch(
        self, request: PackingRequest, batch_id: str, transfer_timeout_s: float
    ) -> PackingResult:
        if self._packer is None:
            from art.megatron.backend import MegatronBackend

            self._packer = MegatronBackend(
                path=f"/tmp/art-packing-{os.getpid()}",
                enable_expert_replay=request.include_moe_routing,
            )
        fetch_started = time.monotonic()
        if request.trajectory_sources:
            groups = list(
                await asyncio.gather(
                    *(
                        source.receive(timeout_s=transfer_timeout_s)
                        for source in request.trajectory_sources
                    )
                )
            )
        elif request.trajectory_transfer is None:
            groups = [payload.build() for payload in request.trajectory_groups]
        else:
            if request.trajectory_groups:
                raise ValueError("packing request has inline and streamed trajectories")
            if request.trajectory_transfer.stream.stream_id != batch_id:
                raise ValueError("packing request has the wrong trajectory stream")
            groups = list(
                await request.trajectory_transfer.receive_groups(
                    timeout_s=transfer_timeout_s
                )
            )
        trajectory_fetch_s = time.monotonic() - fetch_started
        if request.collect_packing_shapes:
            for group in groups:
                group._collect_packing_shape = True
        log_future = None
        if request.trajectory_log_path is not None:
            from art.utils.trajectory_logging import write_trajectory_groups_parquet

            path = Path(request.trajectory_log_path)

            def write_log() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_trajectory_groups_parquet(groups, str(path))

            log_future = asyncio.get_running_loop().run_in_executor(None, write_log)
        packing_started = time.monotonic()
        try:
            packed = self._packer._get_packed_tensors(
                request.model.build(),
                groups,
                advantage_balance=request.advantage_balance,
                allow_training_without_logprobs=(
                    request.allow_training_without_logprobs
                ),
                scale_rewards=request.scale_rewards,
                plot_tensors=request.plot_tensors,
                packed_sequence_length=request.packed_sequence_length,
                logprob_calculation_chunk_size=request.logprob_calculation_chunk_size,
                include_moe_routing=request.include_moe_routing,
            )
        except BaseException as error:
            if log_future is not None:

                async def finish_log() -> None:
                    await log_future

                try:
                    _, cancelled = await complete_task(
                        asyncio.create_task(finish_log())
                    )
                    if cancelled is not None:
                        error.add_note("trajectory logging observed cancellation")
                except BaseException as log_error:
                    error.add_note(
                        "trajectory logging also failed: "
                        f"{type(log_error).__name__}: {log_error}"
                    )
            raise
        packing_core_s = time.monotonic() - packing_started
        log_wait_started = time.monotonic()
        if log_future is not None:
            await log_future
        trajectory_log_wait_s = time.monotonic() - log_wait_started
        shapes = tuple(group._packed_group_shape for group in groups)
        if packed is None:
            if request.trajectory_log_path is not None:
                await asyncio.to_thread(Path(request.trajectory_log_path).unlink)
            return PackingResult(
                ref=None,
                packed_group_shapes=shapes,
                generation_id=request.generation_id,
                trajectory_fetch_s=trajectory_fetch_s,
                packing_core_s=packing_core_s,
                trajectory_log_wait_s=trajectory_log_wait_s,
            )
        trainable_assistant_tokens = int(packed["assistant_mask"].sum().item())
        loss_bearing_tokens = int(packed["assistant_mask"][:, 1:].sum().item())
        non_padding_tokens = int((packed["group_ids"] != -1).sum().item())
        finalize_started = time.monotonic()
        ref = self._packed_batches.store.create(
            packed,
            batch_id=batch_id,
            group_ids=request.group_ids,
            record_ids=request.record_ids,
            min_source_version=request.min_source_version,
            max_source_version=request.max_source_version,
        )
        packed_batch_finalize_s = time.monotonic() - finalize_started
        return PackingResult(
            ref=ref,
            packed_group_shapes=shapes,
            generation_id=request.generation_id,
            trainable_assistant_tokens=trainable_assistant_tokens,
            loss_bearing_tokens=loss_bearing_tokens,
            non_padding_tokens=non_padding_tokens,
            trajectory_log_path=request.trajectory_log_path,
            trajectory_fetch_s=trajectory_fetch_s,
            packing_core_s=packing_core_s,
            trajectory_log_wait_s=trajectory_log_wait_s,
            packed_batch_finalize_s=packed_batch_finalize_s,
        )

    @resilient_endpoint
    async def publish_batch(self, ref: PackedBatchRef) -> PackedBatchTransfer:
        if ref.batch_id in self._batch_publishers:
            raise RuntimeError(f"packed batch {ref.batch_id!r} is already published")
        publisher = await PackedBatchPublisher.create(
            ref, advertise_host=self._data_plane_host
        )
        try:
            transfer = publisher.transfer
        except BaseException:
            await publisher.close()
            raise
        self._batch_publishers[ref.batch_id] = publisher
        return transfer

    @resilient_endpoint
    async def drop_batch(self, batch_id: str) -> None:
        await self._drop_batch(batch_id)

    @resilient_endpoint
    async def note_batch_transmitted(self, byte_count: int) -> None:
        self._packed_batches.store.note_transmitted(byte_count)

    async def _drop_batch(self, batch_id: str) -> bool:
        publisher = self._batch_publishers.pop(batch_id, None)
        if publisher is None:
            return False
        await publisher.close()
        return True

    @resilient_endpoint
    async def receive_batch(
        self, ref: PackedBatchRef, transfer: PackedBatchTransfer, timeout_s: float
    ) -> PackedBatchRef:
        return await self._packed_batches.receive(ref, transfer, timeout_s=timeout_s)

    @resilient_endpoint
    async def drop_batch_ref(self, ref: PackedBatchRef) -> None:
        await self._packed_batches.drop(ref)

    @resilient_endpoint
    async def reclaim_batch(self, batch_id: str, fence: bool) -> bool:
        published = False
        failure: BaseException | None = None
        try:
            published = await self._drop_batch(batch_id)
        except BaseException as error:
            failure = error
        reclaimed = self._packed_batches.store.reclaim(batch_id, fence=fence)
        if failure is not None:
            raise failure
        return published or reclaimed

    @resilient_endpoint
    async def stats(self):
        return self._packed_batches.store.stats()


class RolloutWorkerService(Actor):
    """One process-isolated CPU rollout slot."""

    def __init__(
        self, capacity_records: int, capacity_bytes: int, data_plane_host: str
    ) -> None:
        from .trajectory_store import TrajectoryRecordStore

        self._models = OrderedDict()
        self._results = TrajectoryRecordStore(
            owner_actor_id=f"rollout:{socket.gethostname()}:{os.getpid()}",
            capacity_records=capacity_records,
            capacity_bytes=capacity_bytes,
        )
        self._data_plane_host = data_plane_host
        self._byte_stream_loop = ByteStreamServerLoop()
        self._trajectory_publishers = {}

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
                ref = self._results.put(value)
                try:
                    transfer, publisher = await publish_trajectory_bundles(
                        (self._results.bundle(ref),),
                        stream_id=ref.result_id,
                        advertise_host=self._data_plane_host,
                        server_loop=self._byte_stream_loop,
                    )
                except BaseException:
                    self._results.drop(ref)
                    raise
                self._trajectory_publishers[ref.result_id] = publisher
                value = ref.model_copy(update={"transfer": transfer})
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    async def _release_trajectory(self, ref: TrajectoryGroupRef) -> None:
        self._results.drop(ref)
        publisher = self._trajectory_publishers.pop(ref.result_id, None)
        if publisher is not None:
            await publisher.close()

    @resilient_endpoint
    async def drop_result(self, ref: TrajectoryGroupRef) -> None:
        await self._release_trajectory(ref)

    @resilient_endpoint
    async def close(self) -> None:
        try:
            await asyncio.gather(
                *(
                    publisher.close()
                    for publisher in tuple(self._trajectory_publishers.values())
                )
            )
        finally:
            self._trajectory_publishers.clear()
            await self._byte_stream_loop.close()
        for model in self._models.values():
            await model._reset_inference_runtime()
        self._models.clear()
        self._results.close()
