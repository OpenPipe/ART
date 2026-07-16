from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .art_runtime import ArtRuntime, DistributedPackedBatch
    from .data_plane import PackedBatchRef, TensorSpec
    from .packing import PackingRequest
    from .rollout import (
        DistributedRolloutExecutor,
        InProcessRolloutHost,
        InstalledAsyncCallable,
        LocalRolloutExecutor,
        RolloutExecutor,
    )
    from .specs import (
        ArtRuntimeConfig,
        ClusterSpec,
        EndpointSpec,
        GpuPlacement,
        HostServiceHealth,
        HostSpec,
        ModelServiceMemberSpec,
        ModelServiceReplicaSpec,
        ModelServiceSpec,
        RuntimeTopology,
        TrainerMeshSpec,
        VllmParallelSpec,
    )
    from .topology import compile_topology
    from .vllm_gateway import VllmGateway
    from .vllm_replica import (
        HostMemberLaunchRequest,
        HostMemberState,
        ManagedVllmHostLauncher,
        ReplicaFailure,
        ReplicaHostLauncher,
        ReplicaLaunchTemplate,
        ReplicaManager,
        ReplicaState,
    )
    from .vllm_router import (
        KvCacheEvent,
        PolicyGenerationCommitError,
        PrefixBlockHashes,
        PreparedRoutingTable,
        ReplicaRouter,
        ReplicaTelemetry,
        ReplicaUpdateReport,
        RoutableReplica,
        RouteReservation,
        RoutingDeadlineExceededError,
        RoutingInput,
        RoutingQueueFullError,
        RoutingTable,
        RoutingUnavailableError,
    )

_EXPORTS = {
    "ArtRuntime": ".art_runtime",
    "ArtRuntimeConfig": ".specs",
    "ClusterSpec": ".specs",
    "DistributedPackedBatch": ".art_runtime",
    "DistributedRolloutExecutor": ".rollout",
    "EndpointSpec": ".specs",
    "GpuPlacement": ".specs",
    "HostMemberLaunchRequest": ".vllm_replica",
    "HostMemberState": ".vllm_replica",
    "HostServiceHealth": ".specs",
    "HostSpec": ".specs",
    "InProcessRolloutHost": ".rollout",
    "InstalledAsyncCallable": ".rollout",
    "KvCacheEvent": ".vllm_router",
    "LocalRolloutExecutor": ".rollout",
    "ManagedVllmHostLauncher": ".vllm_replica",
    "ModelServiceMemberSpec": ".specs",
    "ModelServiceReplicaSpec": ".specs",
    "ModelServiceSpec": ".specs",
    "PackingRequest": ".packing",
    "PackedBatchRef": ".data_plane",
    "PolicyGenerationCommitError": ".vllm_router",
    "PrefixBlockHashes": ".vllm_router",
    "PreparedRoutingTable": ".vllm_router",
    "ReplicaHostLauncher": ".vllm_replica",
    "ReplicaFailure": ".vllm_replica",
    "ReplicaLaunchTemplate": ".vllm_replica",
    "ReplicaManager": ".vllm_replica",
    "ReplicaRouter": ".vllm_router",
    "ReplicaState": ".vllm_replica",
    "ReplicaTelemetry": ".vllm_router",
    "ReplicaUpdateReport": ".vllm_replica",
    "RoutableReplica": ".vllm_router",
    "RouteReservation": ".vllm_router",
    "RoutingDeadlineExceededError": ".vllm_router",
    "RoutingInput": ".vllm_router",
    "RoutingQueueFullError": ".vllm_router",
    "RoutingTable": ".vllm_router",
    "RoutingUnavailableError": ".vllm_router",
    "RolloutExecutor": ".rollout",
    "RuntimeTopology": ".specs",
    "TensorSpec": ".data_plane",
    "TrainerMeshSpec": ".specs",
    "VllmParallelSpec": ".specs",
    "VllmGateway": ".vllm_gateway",
    "compile_topology": ".topology",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
