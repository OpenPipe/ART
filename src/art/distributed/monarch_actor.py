from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import wraps
import gc
from multiprocessing import resource_tracker, shared_memory
import os
import socket
import traceback
from typing import Any, cast

# This module is imported only by explicit distributed runtime construction.
from monarch.actor import Actor, endpoint  # ty: ignore[unresolved-import]

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
from .packing import PackingRequest, PackingResult
from .rollout import RolloutInvocation, RolloutResult
from .specs import HostServiceHealth
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
            elif isinstance(error, PackedBatchCapacityError):
                kind = "capacity"
            elif isinstance(error, PackedBatchLeaseError):
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
        self._packer = None
        self._vllm_output_root = vllm_output_root
        self._vllm_launcher = None

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
    async def close(self) -> None:
        for batch_id in tuple(self._rdma_batches):
            await self._drop_batch(batch_id)
        if self._packer is not None:
            await self._packer.close()
            self._packer = None
        if self._vllm_launcher is not None:
            await self._vllm_launcher.close()
            self._vllm_launcher = None
        self.inbox.store.close()

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
    async def allocate_port(self) -> int:
        with socket.socket() as listener:
            listener.bind(("", 0))
            return int(listener.getsockname()[1])

    @resilient_endpoint
    async def pack_batch(self, request: PackingRequest) -> PackingResult:
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

    async def _drop_batch(self, batch_id: str) -> None:
        try:
            handle, tensor, shm = self._rdma_batches.pop(batch_id)
        except KeyError:
            raise RuntimeError(f"packed batch {batch_id!r} is not published") from None
        await handle.drop()
        del tensor
        gc.collect()
        shm.close()

    @resilient_endpoint
    async def receive_rdma_batch(
        self, ref: PackedBatchRef, rdma_buffer: Any, timeout_s: float
    ) -> PackedBatchRef:
        return await self.inbox.receive_rdma(ref, rdma_buffer, timeout_s=timeout_s)

    @resilient_endpoint
    async def drop_batch_ref(self, ref: PackedBatchRef) -> None:
        await self.inbox.drop(ref)

    @resilient_endpoint
    async def stats(self):
        return self.inbox.store.stats()


class RolloutWorkerService(Actor):
    """One process-isolated CPU rollout slot."""

    def __init__(self) -> None:
        self._models = OrderedDict()

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
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    @resilient_endpoint
    async def close(self) -> None:
        for model in self._models.values():
            await model._reset_inference_runtime()
        self._models.clear()
