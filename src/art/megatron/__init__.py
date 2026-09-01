from typing import Any

__all__ = [
    "MegatronBackend",
    "MegatronGateCheckpointOperations",
    "MegatronGateAttemptPlan",
    "MegatronGateCommand",
    "MegatronGateEvidenceRecorder",
    "MegatronGateRunPlan",
    "MegatronGateTurn",
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
    "MegatronOperationEvidenceSink",
    "MegatronOperationResidencySummary",
    "MegatronSlotCoordinator",
    "MegatronSlotResourceManager",
    "MegatronSlotResourceRequest",
    "MegatronSlotRun",
    "MegatronSlotScheduleConfig",
    "TrainerMegatronSlotResources",
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
    "run_megatron_gate_attempt",
]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .backend import MegatronBackend

        return MegatronBackend
    if name in {
        "MegatronGateCheckpointOperations",
        "MegatronGateAttemptPlan",
        "MegatronGateCommand",
        "MegatronGateEvidenceRecorder",
        "MegatronGateRunPlan",
        "MegatronGateTurn",
        "run_megatron_gate_attempt",
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
        "MegatronOperationEvidenceSink",
        "MegatronOperationResidencySummary",
        "MegatronSlotCoordinator",
        "MegatronSlotResourceManager",
        "MegatronSlotResourceRequest",
        "MegatronSlotRun",
        "MegatronSlotScheduleConfig",
        "TrainerMegatronSlotResources",
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
