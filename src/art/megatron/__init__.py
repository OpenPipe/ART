from typing import Any

__all__ = [
    "MegatronBackend",
    "MegatronOperationConfig",
    "MegatronOperationHandler",
    "MegatronOperationRuntime",
    "MegatronArtifactResourcePlan",
    "MegatronRetainedState",
    "MegatronSamplerPublicationReceipt",
    "MegatronMigrationContribution",
    "MegatronMigrationFence",
    "MegatronMigrationReplay",
    "MegatronSlotCoordinator",
    "MegatronSlotResourceManager",
    "MegatronSlotResourceRequest",
    "MegatronSlotRun",
    "MegatronSlotScheduleConfig",
    "bootstrap_megatron_operation_worker",
]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .backend import MegatronBackend

        return MegatronBackend
    if name in {
        "MegatronOperationConfig",
        "MegatronOperationHandler",
        "MegatronOperationRuntime",
        "MegatronArtifactResourcePlan",
        "MegatronRetainedState",
        "MegatronSamplerPublicationReceipt",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
