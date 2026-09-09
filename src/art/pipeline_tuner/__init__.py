from typing import TYPE_CHECKING, Any

from .config import (
    PackedGroupObservation,
    PackedGroupShape,
    PackingLeafShape,
    PipelineAutotuneConfig,
    PipelineAutotunerProfile,
    PipelineMetric,
    PipelineRuntimeConfig,
    PipelineTuneSettings,
)
from .worker_controller import RolloutWorkerController

if TYPE_CHECKING:
    from .attachment import PipelineAutotunerAttachment


def __getattr__(name: str) -> Any:
    if name == "PipelineAutotunerAttachment":
        from .attachment import PipelineAutotunerAttachment

        return PipelineAutotunerAttachment
    raise AttributeError(name)


__all__ = [
    "PackedGroupObservation",
    "PackedGroupShape",
    "PackingLeafShape",
    "PipelineAutotuneConfig",
    "PipelineAutotunerAttachment",
    "PipelineAutotunerProfile",
    "PipelineMetric",
    "PipelineRuntimeConfig",
    "PipelineTuneSettings",
    "RolloutWorkerController",
]
