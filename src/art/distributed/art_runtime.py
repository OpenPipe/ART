from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
import logging
import time
from typing import Any, Literal
from urllib.parse import urlparse
import uuid

from pydantic import BaseModel, ConfigDict

from art.megatron.runtime.specs import TrainerRuntimeSpec, TrainingRunSpec
from art.utils.lifecycle import complete_task

from .artifact_preflight import (
    ArtifactProbeCommand,
    ArtifactProbeOperation,
    ArtifactProbeResult,
    ArtifactProbeSpec,
    ArtifactRootPreflightError,
)
from .data_plane import PackedBatchLeaseSet, fanout_packed_batch
from .host_admission import (
    HostAdmissionReport,
    HostAdmissionRequest,
    RuntimeFingerprint,
    build_runtime_fingerprint,
    runtime_package_names,
    validate_host_admission,
)
from .monarch_bootstrap import (
    _start_worker,
    _stop_worker,
    activate_cpu_child_virtualenv,
    activate_trainer_child_virtualenv,
    attach_controller,
    monarch_identifier,
    require_local_worker_address,
)
from .monarch_runtime import (
    MonarchPackedBatchInbox,
    MonarchPackedBatchSource,
    MonarchPackingEndpoint,
    MonarchRolloutWorkerEndpoint,
    MonarchTrajectoryQueueEndpoint,
    MonarchVllmHostLauncher,
    call_remote,
)
from .nccl_preflight import (
    NcclPreflightSessionRequest,
    NcclProbeRequest,
    NcclProbeResult,
    NcclRendezvousRequest,
    NcclRendezvousResult,
)
from .packing import PackingRequest, PackingResult
from .rollout import DistributedRolloutExecutor, InstalledAsyncCallable
from .specs import (
    ArtRuntimeConfig,
    GpuId,
    GpuPlacement,
    HostServiceHealth,
    ModelServiceSpec,
    RuntimeTopology,
)
from .vllm_replica import (
    ReplicaFailure,
    ReplicaLaunchTemplate,
    ReplicaManager,
    ReplicaState,
)

logger = logging.getLogger(__name__)


class DistributedPackedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    leases: PackedBatchLeaseSet
    packed_group_shapes: tuple[Any, ...]
    trainable_assistant_tokens: int
    loss_bearing_tokens: int
    non_padding_tokens: int
    trajectory_log_path: str | None = None
    packing_rpc_s: float = 0.0
    trajectory_fetch_s: float = 0.0
    packing_core_s: float = 0.0
    trajectory_log_wait_s: float = 0.0
    packed_batch_finalize_s: float = 0.0
    packed_batch_fanout_s: float = 0.0
    packing_generation_id: str


class ArtRuntime:
    """Run-scoped owner of ART host services, trainer meshes, and vLLM services."""

    def __init__(
        self,
        host_mesh: Any,
        topology: RuntimeTopology,
        *,
        config: ArtRuntimeConfig | None = None,
        owns_host_mesh: bool = False,
    ) -> None:
        self.host_mesh = host_mesh
        self.topology = topology
        self.config = config or ArtRuntimeConfig()
        self.owns_host_mesh = owns_host_mesh
        self.runtime_id = uuid.uuid4().hex
        self._host_procs: dict[str, Any] = {}
        self._host_services: dict[str, Any] = {}
        self._adapter_procs: dict[str, Any] = {}
        self._adapter_services: dict[str, Any] = {}
        self._rollout_procs: dict[str, Any] = {}
        self._rollout_actors: dict[str, Any] = {}
        self._trainer_runs: set[Any] = set()
        self._live_batches: dict[str, tuple[str, ...]] = {}
        self._model_services: dict[str, ReplicaManager] = {}
        self._closeables: set[Any] = set()
        self._next_packing_host = 0
        self._nccl_preflight_lock = asyncio.Lock()
        self._nccl_preflights: set[tuple[str, tuple[tuple[str, GpuId], ...], str]] = (
            set()
        )
        self._runtime_packages = runtime_package_names(
            trainer=topology.trainer is not None
        )
        self._controller_fingerprint: RuntimeFingerprint
        self._admitted_hosts: dict[str, HostAdmissionReport] = {}
        self._artifact_probe = (
            ArtifactProbeSpec(
                artifact_root=topology.cluster.artifact_root,
                runtime_id=self.runtime_id,
                host_ids=tuple(host.host_id for host in topology.cluster.hosts),
            )
            if topology.cluster.artifact_root is not None
            else None
        )
        self._close_task: asyncio.Task[None] | None = None
        self._local_worker: Any | None = None
        self._started = False
        self._closed = False

    @classmethod
    async def start(
        cls,
        host_mesh: Any,
        topology: RuntimeTopology,
        *,
        config: ArtRuntimeConfig | None = None,
        owns_host_mesh: bool = False,
    ) -> "ArtRuntime":
        runtime = cls(
            host_mesh,
            topology,
            config=config,
            owns_host_mesh=owns_host_mesh,
        )
        return await runtime._start()

    @classmethod
    async def start_local(
        cls,
        topology: RuntimeTopology,
        *,
        config: ArtRuntimeConfig | None = None,
    ) -> "ArtRuntime":
        requested_address = require_local_worker_address(
            tuple(host.worker_address for host in topology.cluster.hosts)
        )
        worker = _start_worker(
            requested_address,
            startup_timeout_s=topology.cluster.startup_timeout_s,
        )
        address = worker.address
        if address != requested_address:
            host = topology.cluster.hosts[0].model_copy(
                update={"worker_address": address}
            )
            cluster = topology.cluster.model_copy(update={"hosts": (host,)})
            topology = RuntimeTopology(
                cluster=cluster,
                rollout_host_ids=topology.rollout_host_ids,
                trainer=topology.trainer,
                model_services=topology.model_services,
            )
        try:
            host_mesh = await attach_controller(
                (address,),
                name=f"art_local_{uuid.uuid4().hex}",
                startup_timeout_s=topology.cluster.startup_timeout_s,
                owned_workers=(worker,),
            )
        except BaseException as startup_error:
            try:
                await asyncio.to_thread(_stop_worker, worker)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "local ART runtime startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
        try:
            runtime = cls(host_mesh, topology, config=config, owns_host_mesh=True)
        except BaseException as startup_error:
            try:
                await asyncio.wait_for(
                    host_mesh.shutdown(), topology.cluster.rpc_timeout_s
                )
                await asyncio.to_thread(_stop_worker, worker)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "local ART runtime construction and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
        runtime._local_worker = worker
        return await runtime._start()

    async def _start(self) -> "ArtRuntime":
        try:
            await self._start_host_services()
        except BaseException as startup_error:
            try:
                await self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "ART runtime startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
        return self

    async def _start_host_services(self) -> None:
        from .monarch_actor import AdapterTransferHostService, ArtHostService

        async with asyncio.timeout(self.topology.cluster.startup_timeout_s):
            for index, host in enumerate(self.topology.cluster.hosts):
                data_plane_host = urlparse(host.worker_address).hostname
                host_mesh = self.host_mesh.slice(hosts=index)
                proc = host_mesh.spawn_procs(
                    per_host={"service": 1},
                    bootstrap=activate_cpu_child_virtualenv,
                    name=monarch_identifier(
                        f"art_host_{self.runtime_id}_{host.host_id}"
                    ),
                )
                self._host_procs[host.host_id] = proc
                actor = proc.spawn(
                    monarch_identifier(f"art_service_{self.runtime_id}_{host.host_id}"),
                    ArtHostService,
                    HostAdmissionRequest(
                        host_id=host.host_id,
                        node_rank=host.node_rank,
                        expected_gpu_ids=host.gpu_ids,
                        runtime_packages=self._runtime_packages,
                    ).model_dump_json(),
                    self.config.packed_batch_capacity_bytes,
                    self.config.vllm_output_root,
                    data_plane_host,
                )
                self._host_services[host.host_id] = actor
                adapter_proc = host_mesh.spawn_procs(
                    per_host={"adapter": 1},
                    bootstrap=activate_cpu_child_virtualenv,
                    name=monarch_identifier(
                        f"art_adapter_host_{self.runtime_id}_{host.host_id}"
                    ),
                )
                self._adapter_procs[host.host_id] = adapter_proc
                adapter_actor = adapter_proc.spawn(
                    monarch_identifier(f"art_adapter_{self.runtime_id}_{host.host_id}"),
                    AdapterTransferHostService,
                    host.host_id,
                    self.config.vllm_output_root,
                )
                self._adapter_services[host.host_id] = adapter_actor
            await asyncio.gather(
                *(actor.initialized for actor in self._host_services.values()),
                *(actor.initialized for actor in self._adapter_services.values()),
            )
            self._controller_fingerprint, reports = await asyncio.gather(
                asyncio.to_thread(build_runtime_fingerprint, self._runtime_packages),
                asyncio.gather(
                    *(
                        call_remote(actor.admission)
                        for actor in self._host_services.values()
                    )
                ),
            )
            self._admitted_hosts = validate_host_admission(
                self.topology.cluster.hosts,
                reports,
                expected_runtime=self._controller_fingerprint,
            )
            self._validate_nccl_transport_environment()
            await self._preflight_artifact_root()
            await self._preflight_nixl_metadata_store()
        self._started = True
        for report in self._admitted_hosts.values():
            gpus = ",".join(
                f"{gpu.index}={gpu.uuid}@{gpu.pci_bus_id}"
                for gpu in report.assigned_gpus
            )
            logger.info(
                "admitted ART host %s hostname=%s boot_id=%s gpus=[%s] runtime=%s",
                report.host_id,
                report.hostname,
                report.boot_id,
                gpus,
                report.runtime.sha256,
            )

    async def health(self) -> dict[str, HostServiceHealth]:
        async with asyncio.timeout(self.topology.cluster.rpc_timeout_s):
            values = await asyncio.gather(
                *(call_remote(actor.health) for actor in self._host_services.values())
            )
        health = {value.host_id: value for value in values}
        if len(health) != len(values) or health.keys() != self._admitted_hosts.keys():
            raise RuntimeError("host-service liveness membership changed")
        for host_id, value in health.items():
            admitted = self._admitted_hosts[host_id]
            if (value.hostname, value.process_id) != (
                admitted.hostname,
                admitted.process_id,
            ):
                raise RuntimeError(f"host service {host_id!r} identity changed")
        return health

    async def _preflight_launch(
        self,
        *,
        runtime_kind: Literal["trainer", "vllm"],
        placements: tuple[GpuPlacement, ...],
        master_addr: str | None = None,
    ) -> None:
        selected = tuple(
            next(value for value in placements if value.host_id == host_id)
            for host_id in dict.fromkeys(value.host_id for value in placements)
        )
        if len(selected) < 2:
            await self.health()
            return
        transport = self.topology.cluster.nccl_transport
        if transport is None:
            raise RuntimeError("multi-host GPU launch has no NCCL transport contract")
        key = (
            runtime_kind,
            tuple((value.host_id, value.gpu_id) for value in selected),
            transport.net_name,
        )
        deadline = (
            asyncio.get_running_loop().time() + self.topology.cluster.startup_timeout_s
        )
        cleanup_budget_s = min(10.0, self.topology.cluster.startup_timeout_s * 0.1)
        operation_deadline = deadline - cleanup_budget_s
        async with asyncio.timeout_at(deadline):
            await self._nccl_preflight_lock.acquire()
        try:
            async with asyncio.timeout_at(operation_deadline):
                await self.health()
            if key in self._nccl_preflights:
                return
            probe_id = uuid.uuid4().hex
            failure: BaseException | None = None
            try:
                async with asyncio.timeout_at(operation_deadline):
                    leader = selected[0]
                    if master_addr is None:
                        worker_address = self._host(leader.host_id).worker_address
                        parsed = urlparse(worker_address)
                        if parsed.scheme != "tcp" or parsed.hostname is None:
                            raise ValueError(
                                f"NCCL preflight requires a TCP worker address, got "
                                f"{worker_address!r}"
                            )
                        master_addr = parsed.hostname
                    phase_timeout_s = max(
                        0.001,
                        (operation_deadline - asyncio.get_running_loop().time()) * 0.45,
                    )
                    session = NcclPreflightSessionRequest(
                        probe_id=probe_id,
                        lease_s=max(
                            0.001,
                            operation_deadline - asyncio.get_running_loop().time(),
                        ),
                    )
                    session_results = await asyncio.gather(
                        *(
                            call_remote(
                                self._host_services[
                                    placement.host_id
                                ].start_nccl_preflight_session,
                                session,
                            )
                            for placement in selected
                        ),
                        return_exceptions=True,
                    )
                    session_failures = [
                        result
                        for result in session_results
                        if isinstance(result, BaseException)
                    ]
                    if session_failures:
                        raise BaseExceptionGroup(
                            "NCCL preflight session admission failed",
                            session_failures,
                        )
                    rendezvous = await call_remote(
                        self._host_services[leader.host_id].nccl_preflight_rendezvous,
                        NcclRendezvousRequest(
                            probe_id=probe_id,
                            runtime_kind=runtime_kind,
                            master_addr=master_addr,
                            timeout_s=phase_timeout_s,
                        ),
                    )
                    if not isinstance(rendezvous, NcclRendezvousResult):
                        raise RuntimeError("NCCL preflight returned an invalid store")
                    requests = tuple(
                        NcclProbeRequest(
                            probe_id=probe_id,
                            runtime_kind=runtime_kind,
                            rank=rank,
                            world_size=len(selected),
                            master_addr=master_addr,
                            master_port=rendezvous.port,
                            gpu_id=placement.gpu_id,
                            net_name=transport.net_name,
                            timeout_s=phase_timeout_s,
                        )
                        for rank, placement in enumerate(selected)
                    )
                    results = await asyncio.gather(
                        *(
                            call_remote(
                                self._host_services[placement.host_id].nccl_preflight,
                                request,
                            )
                            for placement, request in zip(
                                selected, requests, strict=True
                            )
                        ),
                        return_exceptions=True,
                    )
                    failures = [
                        result
                        for result in results
                        if isinstance(result, BaseException)
                    ]
                    if failures:
                        raise BaseExceptionGroup(
                            f"{runtime_kind} NCCL transport preflight failed", failures
                        )
                    reports = tuple(
                        result
                        for result in results
                        if isinstance(result, NcclProbeResult)
                    )
                    expected = tuple(
                        (placement.host_id, rank, transport.net_name)
                        for rank, placement in enumerate(selected)
                    )
                    if (
                        tuple(
                            (report.host_id, report.rank, report.net_name)
                            for report in reports
                        )
                        != expected
                    ):
                        raise RuntimeError(
                            "NCCL preflight returned inconsistent membership"
                        )
            except BaseException as error:
                failure = error
            cleanup_failures, cleanup_cancelled = await complete_task(
                asyncio.create_task(
                    self._cancel_nccl_preflight(
                        selected,
                        probe_id,
                        timeout_s=max(
                            0.001, deadline - asyncio.get_running_loop().time()
                        ),
                    )
                )
            )
            if cleanup_cancelled is not None:
                if failure is not None:
                    cleanup_cancelled.add_note(f"NCCL preflight also failed: {failure}")
                raise cleanup_cancelled
            if failure is not None:
                if cleanup_failures:
                    raise BaseExceptionGroup(
                        f"{runtime_kind} NCCL preflight and cleanup failed",
                        [failure, *cleanup_failures],
                    ) from None
                raise failure
            if cleanup_failures:
                raise BaseExceptionGroup(
                    f"{runtime_kind} NCCL preflight cleanup failed",
                    cleanup_failures,
                )
            self._nccl_preflights.add(key)
        finally:
            self._nccl_preflight_lock.release()

    async def _cancel_nccl_preflight(
        self,
        placements: tuple[GpuPlacement, ...],
        probe_id: str,
        *,
        timeout_s: float,
    ) -> list[BaseException]:
        try:
            async with asyncio.timeout(timeout_s):
                results = await asyncio.gather(
                    *(
                        call_remote(
                            self._host_services[
                                placement.host_id
                            ].cancel_nccl_preflight,
                            probe_id,
                        )
                        for placement in placements
                    ),
                    return_exceptions=True,
                )
        except BaseException as error:
            return [error]
        return [result for result in results if isinstance(result, BaseException)]

    def _validate_nccl_transport_environment(self) -> None:
        transport = self.topology.cluster.nccl_transport
        if transport is None:
            return
        mismatches = {
            host_id: dict(report.runtime.environment).get("NCCL_NET")
            for host_id, report in self._admitted_hosts.items()
            if dict(report.runtime.environment).get("NCCL_NET") != transport.net_name
        }
        if mismatches:
            raise RuntimeError(
                f"NCCL_NET must equal {transport.net_name!r} on every host: "
                f"{mismatches}"
            )

    async def _preflight_nixl_metadata_store(self) -> None:
        transport = self.topology.cluster.nixl_transport
        if transport is None:
            return
        host_ids = tuple(self._host_services)
        probe_timeout_s = min(5.0, self.topology.cluster.rpc_timeout_s)
        async with asyncio.timeout(self.topology.cluster.rpc_timeout_s):
            results = await asyncio.gather(
                *(
                    call_remote(
                        self._host_services[host_id].nixl_metadata_store_health,
                        transport.metadata_store.url,
                        probe_timeout_s,
                    )
                    for host_id in host_ids
                )
            )
        if tuple(results) != host_ids:
            raise RuntimeError("NIXL metadata-store preflight membership changed")

    async def _preflight_artifact_root(self) -> None:
        if self._artifact_probe is None:
            return
        try:
            await self._artifact_probe_phase("initialize", owner_only=True)
            contenders = self._artifact_probe.host_ids[1:]
            if contenders:
                await self._artifact_probe_phase("hold_lock", owner_only=True)
                await self._artifact_probe_phase("check_lock_held", host_ids=contenders)
                await self._artifact_probe_phase("release_lock", owner_only=True)
                for host_id in contenders:
                    await self._artifact_probe_phase(
                        "check_lock_released", host_ids=(host_id,)
                    )
            for operation in (
                "create",
                "read_created",
                "rename",
                "read_renamed",
                "delete",
            ):
                await self._artifact_probe_phase(operation)
            await self._artifact_probe_phase("finalize", owner_only=True)
        except BaseException as preflight_error:
            cleanup_failures = await self._cleanup_artifact_probe()
            if cleanup_failures:
                raise BaseExceptionGroup(
                    "artifact_root preflight and cleanup failed",
                    [preflight_error, *cleanup_failures],
                ) from None
            raise

    async def _artifact_probe_phase(
        self,
        operation: ArtifactProbeOperation,
        *,
        owner_only: bool = False,
        host_ids: tuple[str, ...] | None = None,
    ) -> None:
        if self._artifact_probe is None:
            return
        if host_ids is None:
            host_ids = (
                self._artifact_probe.host_ids[:1]
                if owner_only
                else self._artifact_probe.host_ids
            )
        command = ArtifactProbeCommand(spec=self._artifact_probe, operation=operation)
        async with asyncio.timeout(self.topology.cluster.rpc_timeout_s):
            results: list[ArtifactProbeResult] = await asyncio.gather(
                *(
                    call_remote(
                        self._host_services[host_id].artifact_root_probe, command
                    )
                    for host_id in host_ids
                )
            )
        for host_id, result in zip(host_ids, results, strict=True):
            if result.error_type is not None:
                raise ArtifactRootPreflightError(result)
            if result.host_id != host_id or result.operation != operation:
                raise RuntimeError(
                    f"invalid artifact_root preflight response from host {host_id!r}"
                )

    async def _cleanup_artifact_probe(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for operation, owner_only in (("cleanup", False), ("finalize", True)):
            try:
                await self._artifact_probe_phase(operation, owner_only=owner_only)
            except BaseException as error:
                if not (
                    operation == "finalize"
                    and isinstance(error, ArtifactRootPreflightError)
                    and error.result.error_type == "FileNotFoundError"
                ):
                    failures.append(error)
        return failures

    def rollout_executor(
        self,
        rollout_callable: InstalledAsyncCallable,
        *,
        target_workers: int,
    ) -> DistributedRolloutExecutor:
        self._require_open()
        self._start_rollout_workers()
        hosts = {
            host_id: tuple(
                MonarchRolloutWorkerEndpoint(
                    actor.slice(rollout=slot),
                    timeout_s=self.topology.cluster.rpc_timeout_s,
                )
                for slot in range(self._host(host_id).cpu_slots)
            )
            for host_id, actor in self._rollout_actors.items()
        }
        return DistributedRolloutExecutor(
            callable=rollout_callable,
            hosts=hosts,
            target_workers=target_workers,
            queue_endpoint=MonarchTrajectoryQueueEndpoint(
                self._host_services[self.topology.cluster.controller_host_id]
            ),
            trajectory_capacity_records=self.config.trajectory_capacity_records,
            trajectory_capacity_bytes=self.config.trajectory_capacity_bytes,
        )

    def _start_rollout_workers(self) -> None:
        if self._rollout_actors:
            return
        from .monarch_actor import RolloutWorkerService

        for index, host in enumerate(self.topology.cluster.hosts):
            if host.host_id not in self.topology.rollout_host_ids:
                continue
            data_plane_host = urlparse(host.worker_address).hostname
            if data_plane_host is None:
                raise ValueError(f"host {host.host_id!r} has no routable address")
            proc = self.host_mesh.slice(hosts=index).spawn_procs(
                per_host={"rollout": host.cpu_slots},
                bootstrap=activate_cpu_child_virtualenv,
                name=monarch_identifier(
                    f"art_rollout_{self.runtime_id}_{host.host_id}"
                ),
            )
            actor = proc.spawn(
                monarch_identifier(
                    f"art_rollout_worker_{self.runtime_id}_{host.host_id}"
                ),
                RolloutWorkerService,
                self.config.trajectory_capacity_records,
                self.config.trajectory_capacity_bytes,
                data_plane_host,
            )
            self._rollout_procs[host.host_id] = proc
            self._rollout_actors[host.host_id] = actor

    def _host(self, host_id: str) -> Any:
        return next(
            host for host in self.topology.cluster.hosts if host.host_id == host_id
        )

    async def pack(self, request: PackingRequest) -> DistributedPackedBatch | None:
        self._require_open()
        trainer = self.topology.trainer
        if trainer is None:
            raise RuntimeError("runtime topology has no trainer mesh")
        trainer_hosts = tuple(dict.fromkeys(rank.host_id for rank in trainer.ranks))
        source_host = trainer_hosts[self._next_packing_host % len(trainer_hosts)]
        self._next_packing_host += 1
        source_service = self._host_services[source_host]
        batch_id = uuid.uuid4().hex
        self._live_batches[batch_id] = trainer_hosts
        try:
            publisher = None
            wire_request = request
            if request.trajectory_groups:
                from .trajectory_store import publish_trajectory_bundles

                controller = self._host(self.topology.cluster.controller_host_id)
                data_plane_host = urlparse(controller.worker_address).hostname
                if data_plane_host is None:
                    raise ValueError("controller has no routable address")
                transfer, publisher = await publish_trajectory_bundles(
                    request.trajectory_groups,
                    stream_id=batch_id,
                    advertise_host=data_plane_host,
                )
                wire_request = request.model_copy(
                    update={"trajectory_groups": (), "trajectory_transfer": transfer}
                )
            try:
                packing_rpc_started = time.monotonic()
                result: PackingResult = await MonarchPackingEndpoint(
                    source_service
                ).pack(
                    wire_request,
                    batch_id,
                    transfer_timeout_s=self.topology.cluster.rpc_timeout_s,
                )
                packing_rpc_s = time.monotonic() - packing_rpc_started
            finally:
                if publisher is not None:
                    await publisher.close()
            if result.ref is None:
                self._live_batches.pop(batch_id)
                return None
            if result.generation_id != request.generation_id:
                raise RuntimeError("packing host returned the wrong generation ID")
            if result.ref.batch_id != batch_id:
                raise RuntimeError("packing host returned the wrong batch ID")
            host_refs = {source_host: result.ref}
            destinations = {
                host_id: MonarchPackedBatchInbox(self._host_services[host_id])
                for host_id in trainer_hosts
                if host_id != source_host
            }
            fanout_started = time.monotonic()
            if destinations:
                host_refs.update(
                    await fanout_packed_batch(
                        ref=result.ref,
                        source_endpoint=MonarchPackedBatchSource(source_service),
                        inboxes=destinations,
                        timeout_s=self.topology.cluster.rpc_timeout_s,
                    )
                )
            packed_batch_fanout_s = time.monotonic() - fanout_started
            leases = PackedBatchLeaseSet(ref=result.ref, host_refs=host_refs)
        except BaseException as error:
            await self._reclaim_after_failure(batch_id, error)
            raise
        return DistributedPackedBatch(
            leases=leases,
            packed_group_shapes=result.packed_group_shapes,
            trainable_assistant_tokens=result.trainable_assistant_tokens,
            loss_bearing_tokens=result.loss_bearing_tokens,
            non_padding_tokens=result.non_padding_tokens,
            trajectory_log_path=result.trajectory_log_path,
            packing_rpc_s=packing_rpc_s,
            trajectory_fetch_s=result.trajectory_fetch_s,
            packing_core_s=result.packing_core_s,
            trajectory_log_wait_s=result.trajectory_log_wait_s,
            packed_batch_finalize_s=result.packed_batch_finalize_s,
            packed_batch_fanout_s=packed_batch_fanout_s,
            packing_generation_id=result.generation_id,
        )

    async def release_batch(self, batch: DistributedPackedBatch) -> None:
        await self._reclaim_batch(batch.leases.ref.batch_id, fence=False)

    async def _reclaim_batch(self, batch_id: str, *, fence: bool) -> None:
        hosts = self._live_batches.get(batch_id)
        if hosts is None:
            return

        async def reclaim(host_id: str) -> None:
            inbox = MonarchPackedBatchInbox(self._host_services[host_id])
            await inbox.reclaim(batch_id, fence=fence)

        results = await asyncio.gather(
            *(reclaim(host_id) for host_id in hosts),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("failed to reclaim packed batch", failures)
        if self._live_batches.get(batch_id) == hosts:
            self._live_batches.pop(batch_id)

    async def _reclaim_after_failure(
        self, batch_id: str, primary: BaseException
    ) -> None:
        try:
            _, cancelled = await complete_task(
                asyncio.create_task(self._reclaim_batch(batch_id, fence=True))
            )
            if cancelled is not None:
                primary.add_note("packed-batch reclamation observed cancellation")
        except BaseException as cleanup_error:
            primary.add_note(
                "packed-batch reclamation also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    async def start_trainer(
        self, runtime_spec: TrainerRuntimeSpec, run_spec: TrainingRunSpec
    ) -> Any:
        self._require_open()
        if self.topology.trainer is None:
            raise RuntimeError("runtime topology has no trainer mesh")
        if runtime_spec.trainer_mesh != self.topology.trainer:
            raise ValueError("trainer runtime mesh does not match compiled topology")
        host_ids = [rank.host_id for rank in runtime_spec.trainer_mesh.ranks]
        counts = Counter(host_ids)
        if len(set(counts.values())) != 1:
            raise ValueError("Monarch trainer hosts require equal ranks per host")
        ordered_hosts = tuple(dict.fromkeys(host_ids))
        expected = tuple(
            host.host_id
            for host in self.topology.cluster.hosts
            if host.host_id in counts
        )
        if ordered_hosts != expected:
            raise ValueError("trainer ranks must use cluster host order")
        indices = [
            index
            for index, host in enumerate(self.topology.cluster.hosts)
            if host.host_id in counts
        ]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("trainer hosts must be contiguous in the cluster mesh")
        if runtime_spec.hybrid_ep is not None and runtime_spec.hybrid_ep.multinode:
            await self._preflight_nixl_metadata_store()
        await self._preflight_launch(
            runtime_kind="trainer", placements=runtime_spec.trainer_mesh.ranks
        )
        selected = self.host_mesh.slice(hosts=slice(indices[0], indices[-1] + 1))
        from art.megatron.runtime.monarch import (
            MonarchTrainerRun,
            MonarchTrainerSupervision,
            spawn_monarch_trainer_actors,
        )

        supervision = MonarchTrainerSupervision(run_spec.run_id)
        proc = None
        try:
            proc = selected.spawn_procs(
                per_host={"trainer": next(iter(counts.values()))},
                bootstrap=activate_trainer_child_virtualenv,
                name=monarch_identifier(
                    f"art_trainer_{supervision.token}_{self.runtime_id}"
                ),
            )
            async with asyncio.timeout(self.topology.cluster.startup_timeout_s):
                (
                    actors,
                    rank_processes,
                    cp_lookahead_ports,
                ) = await spawn_monarch_trainer_actors(proc, runtime_spec, supervision)
        except BaseException as startup_error:
            try:
                if proc is not None:
                    async with asyncio.timeout(self.topology.cluster.rpc_timeout_s):
                        await proc.stop()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "trainer startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            finally:
                supervision.close()
            raise
        run = MonarchTrainerRun(
            runtime_spec,
            run_spec,
            actors,
            proc,
            supervision,
            rank_processes,
            cp_lookahead_ports,
        )
        self._trainer_runs.add(run)
        return run

    async def stop_trainer(self, run: Any) -> None:
        await run.close()
        self._trainer_runs.discard(run)

    def register_closeable(self, closeable: Any) -> None:
        self._require_open()
        self._closeables.add(closeable)

    async def start_model_service(
        self,
        spec: ModelServiceSpec,
        template: ReplicaLaunchTemplate,
        *,
        on_failure: Callable[[ReplicaFailure], Awaitable[None]] | None = None,
    ) -> ReplicaState:
        self._require_open()
        configured = {service.name: service for service in self.topology.model_services}
        if configured.get(spec.name) != spec:
            raise ValueError(
                "model service does not match the compiled runtime topology"
            )
        if spec.name in self._model_services:
            raise RuntimeError(f"model service {spec.name!r} is already managed")
        await self._preflight_launch(
            runtime_kind="vllm",
            placements=tuple(
                GpuPlacement(host_id=member.host_id, gpu_id=member.gpu_ids[0])
                for member in spec.members
            ),
            master_addr=spec.rendezvous.host,
        )
        launchers = {
            member.host_id: MonarchVllmHostLauncher(
                self._host_services[member.host_id],
                self._adapter_services[member.host_id],
            )
            for member in spec.members
        }
        manager = ReplicaManager(
            spec,
            launchers,
            template,
            on_failure=on_failure,
            startup_timeout_s=self.topology.cluster.startup_timeout_s,
            rpc_timeout_s=self.topology.cluster.rpc_timeout_s,
        )
        self._model_services[spec.name] = manager
        return await manager.start()

    def model_service(self, name: str) -> ReplicaManager:
        try:
            return self._model_services[name]
        except KeyError:
            raise RuntimeError(f"model service {name!r} is not managed") from None

    async def stop_model_service(self, name: str) -> ReplicaState:
        manager = self.model_service(name)
        state = await manager.stop()
        self._model_services.pop(name, None)
        return state

    async def close(self) -> None:
        if self._close_task is not None and self._close_task.done():
            try:
                self._close_task.result()
            except BaseException:
                self._close_task = None
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        failures: list[BaseException] = []

        async def collect(name: str, *awaitables: Any) -> bool:
            if not awaitables:
                return True
            tasks = {asyncio.ensure_future(awaitable) for awaitable in awaitables}
            try:
                done, pending = await asyncio.wait(
                    tasks, timeout=self.topology.cluster.rpc_timeout_s
                )
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            group_failed = bool(pending)
            if pending:
                failures.append(
                    TimeoutError(
                        f"{name} exceeded {self.topology.cluster.rpc_timeout_s}s"
                    )
                )
            for task in done:
                try:
                    task.result()
                except BaseException as error:
                    failures.append(error)
                    group_failed = True
            return not group_failed

        if await collect(
            "dependent shutdown", *(value.aclose() for value in self._closeables)
        ):
            self._closeables.clear()
        await collect(
            "model-service shutdown",
            *(self.stop_model_service(name) for name in tuple(self._model_services)),
        )
        if await collect(
            "trainer shutdown", *(run.close() for run in self._trainer_runs)
        ):
            self._trainer_runs.clear()
        await collect(
            "packed batch reclamation",
            *(
                self._reclaim_batch(batch_id, fence=True)
                for batch_id in tuple(self._live_batches)
            ),
        )
        if await collect(
            "rollout actor shutdown",
            *(
                call_remote(actor.slice(rollout=slot).close)
                for host_id, actor in self._rollout_actors.items()
                for slot in range(self._host(host_id).cpu_slots)
            ),
        ):
            self._rollout_actors.clear()
        if await collect(
            "rollout process shutdown",
            *(proc.stop() for proc in self._rollout_procs.values()),
        ):
            self._rollout_procs.clear()
        if await collect(
            "adapter transfer service shutdown",
            *(call_remote(actor.close) for actor in self._adapter_services.values()),
        ):
            self._adapter_services.clear()
        if await collect(
            "adapter transfer process shutdown",
            *(proc.stop() for proc in self._adapter_procs.values()),
        ):
            self._adapter_procs.clear()
        if await collect(
            "host service shutdown",
            *(call_remote(actor.close) for actor in self._host_services.values()),
        ):
            self._host_services.clear()
        if await collect(
            "host process shutdown",
            *(proc.stop() for proc in self._host_procs.values()),
        ):
            self._host_procs.clear()
        if self.owns_host_mesh and await collect(
            "host mesh shutdown", self.host_mesh.shutdown()
        ):
            self.owns_host_mesh = False
        if self._local_worker is not None:
            try:
                await asyncio.to_thread(_stop_worker, self._local_worker)
            except BaseException as error:
                failures.append(error)
            else:
                self._local_worker = None
        if failures:
            raise BaseExceptionGroup("ART runtime teardown failed", failures)

    async def __aenter__(self) -> "ArtRuntime":
        self._require_open()
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()

    def _require_open(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("ART runtime is not active")
