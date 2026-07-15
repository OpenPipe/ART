from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict

from .data_plane import PackedBatchLeaseSet, fanout_rdma_packed_batch
from .monarch_bootstrap import monarch_identifier
from .monarch_runtime import (
    MonarchPackedBatchInbox,
    MonarchPackedBatchSource,
    MonarchPackingEndpoint,
    MonarchRolloutHostEndpoint,
    MonarchVllmHostLauncher,
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
        self._trainer_runs: set[Any] = set()
        self._replicas: dict[str, ReplicaManager] = {}
        self._next_packing_host = 0
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
        try:
            await runtime._start_host_services()
        except BaseException:
            await runtime.close()
            raise
        return runtime

    async def _start_host_services(self) -> None:
        from .monarch_actor import RolloutHostService

        for index, host in enumerate(self.topology.cluster.hosts):
            proc = self.host_mesh.slice(hosts=index).spawn_procs(
                per_host={"service": 1},
                name=monarch_identifier(f"art_host_{self.runtime_id}_{host.host_id}"),
            )
            actor = proc.spawn(
                monarch_identifier(f"art_service_{self.runtime_id}_{host.host_id}"),
                RolloutHostService,
                host.host_id,
                self.config.packed_batch_capacity_bytes,
                self.config.vllm_output_root,
            )
            self._host_procs[host.host_id] = proc
            self._host_actors[host.host_id] = actor
        await asyncio.gather(
            *(actor.initialized for actor in self._host_actors.values())
        )
        health = await self.health()
        if set(health) != {host.host_id for host in self.topology.cluster.hosts}:
            raise RuntimeError(
                "host-service membership does not match runtime topology"
            )
        self._started = True

    async def health(self) -> dict[str, HostServiceHealth]:
        values = await asyncio.gather(
            *(actor.health.call_one() for actor in self._host_actors.values())
        )
        health = {value.host_id: value for value in values}
        if len(health) != len(values):
            raise RuntimeError("host services returned duplicate identities")
        return health

    def rollout_executor(
        self,
        rollout_callable: InstalledAsyncCallable,
        *,
        target_workers: int,
    ) -> DistributedRolloutExecutor:
        self._require_open()
        hosts = {
            host_id: MonarchRolloutHostEndpoint(self._host_actors[host_id])
            for host_id in self.topology.rollout_host_ids
        }
        slots = {
            host.host_id: host.cpu_slots
            for host in self.topology.cluster.hosts
            if host.host_id in hosts
        }
        return DistributedRolloutExecutor(
            callable=rollout_callable,
            hosts=hosts,
            host_slots=slots,
            target_workers=target_workers,
        )

    async def pack(self, request: PackingRequest) -> DistributedPackedBatch:
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
            raise ValueError("packing produced no trainable batch")
        host_refs = {source_host: result.ref}
        destinations = {
            host_id: MonarchPackedBatchInbox(self._host_actors[host_id])
            for host_id in trainer_hosts
            if host_id != source_host
        }
        try:
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
            await inbox.release(ref.lease_id)
            await inbox.unlink(ref.batch_id)

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
        selected = self.host_mesh.slice(hosts=slice(indices[0], indices[-1] + 1))
        proc = selected.spawn_procs(
            per_host={"trainer": next(iter(counts.values()))},
            name=monarch_identifier(f"art_trainer_{self.runtime_id}_{run_spec.run_id}"),
        )
        from art.megatron.runtime.monarch import (
            MonarchTrainerRun,
            spawn_monarch_trainer_actors,
        )

        try:
            actors = await spawn_monarch_trainer_actors(proc, runtime_spec)
            await actors.initialized
        except BaseException:
            await proc.stop()
            raise
        run = MonarchTrainerRun(runtime_spec, run_spec, actors, proc)
        self._trainer_runs.add(run)
        return run

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
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for manager in tuple(self._replicas.values()):
            try:
                await manager.stop()
            except BaseException as error:
                failures.append(error)
        self._replicas.clear()
        for run in tuple(self._trainer_runs):
            try:
                await run.close()
            except BaseException as error:
                failures.append(error)
        self._trainer_runs.clear()
        results = await asyncio.gather(
            *(actor.close.call_one() for actor in self._host_actors.values()),
            return_exceptions=True,
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        results = await asyncio.gather(
            *(proc.stop() for proc in self._host_procs.values()),
            return_exceptions=True,
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        self._host_actors.clear()
        self._host_procs.clear()
        if self.owns_host_mesh:
            try:
                await self.host_mesh.shutdown()
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
