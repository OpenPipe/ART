from typing import Any

__all__ = [
    "MegatronBackend",
    "MegatronGateCheckpointOperations",
    "MegatronGateEvidenceRecorder",
    "MegatronOperationConfig",
    "MegatronOperationHandler",
    "MegatronOperationRuntime",
    "MegatronPolicyActivationTiming",
    "MegatronArtifactResourcePlan",
    "MegatronRetainedState",
    "MegatronSamplerPublicationReceipt",
    "POLICY_ACTIVATION_LAG_METRIC",
    "MegatronMigrationContribution",
    "MegatronMigrationFence",
    "MegatronMigrationReplay",
    "MegatronSlotCoordinator",
    "MegatronSlotResourceManager",
    "MegatronSlotResourceRequest",
    "MegatronSlotRun",
    "MegatronSlotScheduleConfig",
    "MegatronSlotLaunchConfig",
    "MegatronRunBinding",
    "MegatronRunBootstrapConfig",
    "MegatronSlotRuntime",
    "MegatronSlotRuntimeDescriptor",
    "RouteBundleOwnershipHandle",
    "RouteBundleOwnershipProvider",
    "RouteBundleOwnershipTransfer",
    "bootstrap_megatron_operation_worker",
    "launch_megatron_slot",
    "prepare_megatron_run_config",
]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .backend import MegatronBackend

        return MegatronBackend
    if name in {
        "MegatronGateCheckpointOperations",
        "MegatronGateEvidenceRecorder",
    }:
        from . import gate_evidence

        return getattr(gate_evidence, name)
    if name in {
        "MegatronOperationConfig",
        "MegatronOperationHandler",
        "MegatronOperationRuntime",
        "MegatronPolicyActivationTiming",
        "MegatronArtifactResourcePlan",
        "MegatronRetainedState",
        "MegatronSamplerPublicationReceipt",
        "POLICY_ACTIVATION_LAG_METRIC",
        "bootstrap_megatron_operation_worker",
    }:
        from . import operation_handler

        return getattr(operation_handler, name)
    if name in {
        "MegatronMigrationContribution",
        "MegatronMigrationFence",
        "MegatronMigrationReplay",
        "MegatronSlotCoordinator",
        "MegatronSlotResourceManager",
        "MegatronSlotResourceRequest",
        "MegatronSlotRun",
        "MegatronSlotScheduleConfig",
    }:
        from . import slot_coordinator

        return getattr(slot_coordinator, name)
    if name in {
        "MegatronSlotLaunchConfig",
        "MegatronRunBinding",
        "MegatronRunBootstrapConfig",
        "MegatronSlotRuntime",
        "MegatronSlotRuntimeDescriptor",
        "launch_megatron_slot",
        "prepare_megatron_run_config",
    }:
        from . import slot_runtime

        return getattr(slot_runtime, name)
    if name in {
        "RouteBundleOwnershipHandle",
        "RouteBundleOwnershipProvider",
        "RouteBundleOwnershipTransfer",
    }:
        from . import route_retention

        return getattr(route_retention, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
