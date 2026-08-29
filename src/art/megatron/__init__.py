from typing import Any

__all__ = [
    "MegatronBackend",
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
    "MegatronSlotRuntime",
    "MegatronSlotRuntimeDescriptor",
    "bootstrap_megatron_operation_worker",
    "launch_megatron_slot",
]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .backend import MegatronBackend

        return MegatronBackend
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
        "MegatronSlotRuntime",
        "MegatronSlotRuntimeDescriptor",
        "launch_megatron_slot",
    }:
        from . import slot_runtime

        return getattr(slot_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
