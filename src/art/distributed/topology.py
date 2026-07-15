from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from .specs import (
    ClusterSpec,
    GpuPlacement,
    HostSpec,
    ModelServiceSpec,
    RuntimeTopology,
    TrainerMeshSpec,
)


def compile_topology(
    *,
    cluster: ClusterSpec,
    rollout_host_ids: tuple[str, ...] | None = None,
    trainer: TrainerMeshSpec | None = None,
    model_services: tuple[ModelServiceSpec, ...] = (),
) -> RuntimeTopology:
    hosts = {host.host_id: host for host in cluster.hosts}
    rollout_hosts = rollout_host_ids or tuple(hosts)
    _require_known_hosts("rollout_host_ids", rollout_hosts, hosts)
    if len(set(rollout_hosts)) != len(rollout_hosts):
        raise ValueError("rollout_host_ids must be unique")

    placements: list[tuple[str, int, str]] = []
    if trainer is not None:
        _validate_placements("trainer", trainer.ranks, hosts)
        placements.extend(
            (placement.host_id, placement.gpu_id, "trainer")
            for placement in trainer.ranks
        )

    service_names = [service.name for service in model_services]
    if len(set(service_names)) != len(service_names):
        raise ValueError("model service names must be unique")
    endpoints: list[tuple[str, int, str]] = []
    for service in model_services:
        for replica in service.replicas:
            member_hosts = tuple(member.host_id for member in replica.members)
            _require_known_hosts(f"replica {replica.replica_id}", member_hosts, hosts)
            _validate_placements(replica.replica_id, replica.gpu_placements, hosts)
            placements.extend(
                (placement.host_id, placement.gpu_id, replica.replica_id)
                for placement in replica.gpu_placements
            )
            endpoints.extend(
                (
                    endpoint.host,
                    endpoint.port,
                    f"{service.name}/{replica.replica_id}/{kind}",
                )
                for kind, endpoint in (
                    ("leader", replica.leader_endpoint),
                    ("rendezvous", replica.rendezvous),
                )
            )

    duplicate_gpus = [
        (host_id, gpu_id)
        for (host_id, gpu_id), count in Counter(
            (host_id, gpu_id) for host_id, gpu_id, _ in placements
        ).items()
        if count > 1
    ]
    if duplicate_gpus:
        owners = [
            owner
            for host_id, gpu_id in duplicate_gpus
            for candidate_host, candidate_gpu, owner in placements
            if (candidate_host, candidate_gpu) == (host_id, gpu_id)
        ]
        raise ValueError(f"GPU placements overlap at {duplicate_gpus}; owners={owners}")

    duplicate_endpoints = [
        endpoint
        for endpoint, count in Counter(
            (host, port) for host, port, _ in endpoints
        ).items()
        if count > 1
    ]
    if duplicate_endpoints:
        raise ValueError(f"model-service ports overlap at {duplicate_endpoints}")

    return RuntimeTopology(
        cluster=cluster,
        rollout_host_ids=rollout_hosts,
        trainer=trainer,
        model_services=model_services,
    )


def _require_known_hosts(
    field: str, host_ids: tuple[str, ...], hosts: Mapping[str, object]
) -> None:
    unknown = sorted(set(host_ids) - hosts.keys())
    if unknown:
        raise ValueError(f"{field} references unknown hosts: {unknown}")


def _validate_placements(
    name: str,
    placements: Sequence[GpuPlacement],
    hosts: Mapping[str, HostSpec],
) -> None:
    for placement in placements:
        host = hosts.get(placement.host_id)
        if host is None:
            raise ValueError(f"{name} references unknown host {placement.host_id!r}")
        if placement.gpu_id not in host.gpu_ids:
            raise ValueError(
                f"{name} requests unavailable GPU "
                f"{placement.host_id}:{placement.gpu_id}"
            )
