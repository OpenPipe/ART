from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
import logging
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict

from .artifact_preflight import (
    ArtifactProbeCommand,
    ArtifactProbeOperation,
    ArtifactProbeResult,
    ArtifactProbeSpec,
    ArtifactRootPreflightError,
)
from .data_plane import PackedBatchLeaseSet, fanout_rdma_packed_batch
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
    activate_child_virtualenv,
    activate_cpu_child_virtualenv,
    attach_controller,
    monarch_identifier,
    require_local_worker_address,
)
from .monarch_runtime import (
    MonarchPackedBatchInbox,
    MonarchPackedBatchSource,
    MonarchPackingEndpoint,
    MonarchRolloutHostEndpoint,
    MonarchVllmHostLauncher,
    call_remote,
)
from .packing import PackingRequest, PackingResult
from .rollout import DistributedRolloutExecutor, InstalledAsyncCallable
from .specs import (
    ArtRuntimeConfig,
    HostServiceHealth,
    ModelServiceReplicaSpec,
    RuntimeTopology,
)
from .vllm_replica import ReplicaLaunchTemplate, ReplicaManager, ReplicaState

logger = logging.getLogger(__name__)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class DistributedPackedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    leases: PackedBatchLeaseSet
    packed_group_shapes: tuple[Any, ...]
    trainable_assistant_tokens: int
    non_padding_tokens: int


class ArtRuntime:
    """Run-scoped owner of ART host services, trainer meshes, and vLLM replicas."""

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
        self._host_actors: dict[str, Any] = {}
        self._rollout_procs: dict[str, Any] = {}
        self._rollout_actors: dict[str, Any] = {}
        self._trainer_runs: set[Any] = set()
        self._replicas: dict[str, ReplicaManager] = {}
        self._closeables: set[Any] = set()
        self._next_packing_host = 0
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
        address = require_local_worker_address(
            tuple(host.worker_address for host in topology.cluster.hosts)
        )
        worker = _start_worker(address)
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
        from .monarch_actor import RolloutHostService

        async with asyncio.timeout(self.topology.cluster.startup_timeout_s):
            for index, host in enumerate(self.topology.cluster.hosts):
                proc = self.host_mesh.slice(hosts=index).spawn_procs(
                    per_host={"service": 1},
                    bootstrap=activate_cpu_child_virtualenv,
                    name=monarch_identifier(
                        f"art_host_{self.runtime_id}_{host.host_id}"
                    ),
                )
                actor = proc.spawn(
                    monarch_identifier(f"art_service_{self.runtime_id}_{host.host_id}"),
                    RolloutHostService,
                    HostAdmissionRequest(
                        host_id=host.host_id,
                        node_rank=host.node_rank,
                        expected_gpu_ids=host.gpu_ids,
                        runtime_packages=self._runtime_packages,
                    ).model_dump_json(),
                    self.config.packed_batch_capacity_bytes,
                    self.config.vllm_output_root,
                )
                self._host_procs[host.host_id] = proc
                self._host_actors[host.host_id] = actor
            await asyncio.gather(
                *(actor.initialized for actor in self._host_actors.values())
            )
            self._controller_fingerprint, reports = await asyncio.gather(
                asyncio.to_thread(build_runtime_fingerprint, self._runtime_packages),
                asyncio.gather(
                    *(
                        call_remote(actor.admission)
                        for actor in self._host_actors.values()
                    )
                ),
            )
            self._admitted_hosts = validate_host_admission(
                self.topology.cluster.hosts,
                reports,
                expected_runtime=self._controller_fingerprint,
            )
            await self._preflight_artifact_root()
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
                *(call_remote(actor.health) for actor in self._host_actors.values())
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

    async def _preflight_launch(self) -> None:
        await self.health()

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
                    call_remote(self._host_actors[host_id].artifact_root_probe, command)
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
                MonarchRolloutHostEndpoint(actor.slice(rollout=slot))
                for slot in range(self._host(host_id).cpu_slots)
            )
            for host_id, actor in self._rollout_actors.items()
        }
        return DistributedRolloutExecutor(
            callable=rollout_callable,
            hosts=hosts,
            target_workers=target_workers,
        )

    def _start_rollout_workers(self) -> None:
        if self._rollout_actors:
            return
        from .monarch_actor import RolloutWorkerService

        for index, host in enumerate(self.topology.cluster.hosts):
            if host.host_id not in self.topology.rollout_host_ids:
                continue
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
        source_actor = self._host_actors[source_host]
        result: PackingResult = await MonarchPackingEndpoint(source_actor).pack(request)
        if result.ref is None:
            return None
        host_refs = {source_host: result.ref}
        destinations = {
            host_id: MonarchPackedBatchInbox(self._host_actors[host_id])
            for host_id in trainer_hosts
            if host_id != source_host
        }
        try:
            if destinations:
                host_refs.update(
                    await fanout_rdma_packed_batch(
                        ref=result.ref,
                        source_endpoint=MonarchPackedBatchSource(source_actor),
                        inboxes=destinations,
                        timeout_s=self.topology.cluster.rpc_timeout_s,
                    )
                )
            leases = PackedBatchLeaseSet(ref=result.ref, host_refs=host_refs)
        except BaseException:
            await self._release_refs(host_refs)
            raise
        return DistributedPackedBatch(
            leases=leases,
            packed_group_shapes=result.packed_group_shapes,
            trainable_assistant_tokens=result.trainable_assistant_tokens,
            non_padding_tokens=result.non_padding_tokens,
        )

    async def release_batch(self, batch: DistributedPackedBatch) -> None:
        await self._release_refs(batch.leases.host_refs)

    async def _release_refs(self, refs: Mapping[str, Any]) -> None:
        async def release(host_id: str, ref: Any) -> None:
            inbox = MonarchPackedBatchInbox(self._host_actors[host_id])
            await inbox.drop(ref)

        results = await asyncio.gather(
            *(release(host_id, ref) for host_id, ref in refs.items()),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("failed to release packed batch", failures)

    async def start_trainer(self, runtime_spec: Any, run_spec: Any) -> Any:
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
        await self._preflight_launch()
        selected = self.host_mesh.slice(hosts=slice(indices[0], indices[-1] + 1))
        proc = selected.spawn_procs(
            per_host={"trainer": next(iter(counts.values()))},
            bootstrap=activate_child_virtualenv,
            name=monarch_identifier(f"art_trainer_{self.runtime_id}_{run_spec.run_id}"),
        )
        from art.megatron.runtime.monarch import (
            MonarchTrainerRun,
            spawn_monarch_trainer_actors,
        )

        try:
            async with asyncio.timeout(self.topology.cluster.startup_timeout_s):
                actors = await spawn_monarch_trainer_actors(proc, runtime_spec)
                await actors.initialized
        except BaseException as startup_error:
            try:
                async with asyncio.timeout(self.topology.cluster.rpc_timeout_s):
                    await proc.stop()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "trainer startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
        run = MonarchTrainerRun(runtime_spec, run_spec, actors, proc)
        self._trainer_runs.add(run)
        return run

    def register_closeable(self, closeable: Any) -> None:
        self._require_open()
        self._closeables.add(closeable)

    async def start_replica(
        self,
        spec: ModelServiceReplicaSpec,
        template: ReplicaLaunchTemplate,
    ) -> ReplicaState:
        self._require_open()
        configured = {
            replica.replica_id: replica
            for service in self.topology.model_services
            for replica in service.replicas
        }
        if configured.get(spec.replica_id) != spec:
            raise ValueError("replica does not match the compiled runtime topology")
        if spec.replica_id in self._replicas:
            raise RuntimeError(f"replica {spec.replica_id!r} is already managed")
        await self._preflight_launch()
        launchers = {
            member.host_id: MonarchVllmHostLauncher(self._host_actors[member.host_id])
            for member in spec.members
        }
        endpoint_hosts = {
            spec.leader_endpoint.host: next(
                member.host_id for member in spec.members if member.leader
            ),
            spec.rendezvous.host: next(
                member.host_id for member in spec.members if member.node_rank == 0
            ),
        }

        async def allocate(host: str) -> int:
            try:
                host_id = endpoint_hosts[host]
            except KeyError:
                raise ValueError(
                    f"no replica member owns endpoint host {host!r}"
                ) from None
            return await launchers[host_id].allocate_port()

        manager = ReplicaManager(
            spec,
            launchers,
            template,
            port_allocator=allocate,
        )
        self._replicas[spec.replica_id] = manager
        try:
            return await manager.start()
        except BaseException:
            self._replicas.pop(spec.replica_id, None)
            raise

    def replica(self, replica_id: str) -> ReplicaManager:
        try:
            return self._replicas[replica_id]
        except KeyError:
            raise RuntimeError(f"replica {replica_id!r} is not managed") from None

    async def stop_replica(self, replica_id: str) -> ReplicaState:
        manager = self.replica(replica_id)
        try:
            return await manager.stop()
        finally:
            self._replicas.pop(replica_id, None)

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        failures: list[BaseException] = []

        async def collect(name: str, *awaitables: Any) -> None:
            if not awaitables:
                return
            tasks = {asyncio.ensure_future(awaitable) for awaitable in awaitables}
            try:
                done, pending = await asyncio.wait(
                    tasks, timeout=self.topology.cluster.rpc_timeout_s
                )
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise
            for task in pending:
                task.cancel()
                task.add_done_callback(_consume_task_result)
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

        await collect(
            "dependent shutdown", *(value.aclose() for value in self._closeables)
        )
        self._closeables.clear()
        await collect("replica shutdown", *(m.stop() for m in self._replicas.values()))
        self._replicas.clear()
        await collect("trainer shutdown", *(run.close() for run in self._trainer_runs))
        self._trainer_runs.clear()
        await collect(
            "rollout actor shutdown",
            *(
                call_remote(actor.slice(rollout=slot).close)
                for host_id, actor in self._rollout_actors.items()
                for slot in range(self._host(host_id).cpu_slots)
            ),
        )
        await collect(
            "rollout process shutdown",
            *(proc.stop() for proc in self._rollout_procs.values()),
        )
        self._rollout_actors.clear()
        self._rollout_procs.clear()
        await collect(
            "host actor shutdown",
            *(call_remote(actor.close) for actor in self._host_actors.values()),
        )
        await collect(
            "host process shutdown",
            *(proc.stop() for proc in self._host_procs.values()),
        )
        self._host_actors.clear()
        self._host_procs.clear()
        if self.owns_host_mesh:
            await collect("host mesh shutdown", self.host_mesh.shutdown())
        worker, self._local_worker = self._local_worker, None
        if worker is not None:
            try:
                await asyncio.to_thread(_stop_worker, worker)
            except BaseException as error:
                failures.append(error)
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
