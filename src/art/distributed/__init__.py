from importlib import import_module
from typing import Any

_EXPORTS = {
    "ArtRuntime": ".art_runtime",
    "ArtRuntimeConfig": ".specs",
    "ClusterSpec": ".specs",
    "EndpointSpec": ".specs",
    "GpuPlacement": ".specs",
    "HostMemberLaunchRequest": ".vllm_replica",
    "HostMemberState": ".vllm_replica",
    "HostSpec": ".specs",
    "InstalledAsyncCallable": ".rollout",
    "KvCacheEvent": ".vllm_router",
    "LocalRolloutExecutor": ".rollout",
    "ManagedVllmHostLauncher": ".vllm_replica",
    "ModelServiceMemberSpec": ".specs",
    "ModelServiceReplicaSpec": ".specs",
    "ModelServiceSpec": ".specs",
    "PackedBatchRef": ".data_plane",
    "PolicyGenerationCommitError": ".vllm_router",
    "PrefixBlockHashes": ".vllm_router",
    "PreparedRoutingTable": ".vllm_router",
    "ReplicaHostLauncher": ".vllm_replica",
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
