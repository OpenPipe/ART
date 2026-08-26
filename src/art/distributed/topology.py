from __future__ import annotations

from .specs import (
    ClusterSpec,
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
    return RuntimeTopology(
        cluster=cluster,
        rollout_host_ids=(
            tuple(host.host_id for host in cluster.hosts)
            if rollout_host_ids is None
            else rollout_host_ids
        ),
        trainer=trainer,
        model_services=model_services,
    )
