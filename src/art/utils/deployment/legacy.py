"""Legacy exports for backwards compatibility."""

from enum import Enum

from pydantic import BaseModel

from .together import TogetherJobStatus


class LoRADeploymentProvider(str, Enum):
    """Legacy enum for deployment providers."""

    TOGETHER = "together"
    WANDB = "wandb"


class LoRADeploymentJob(BaseModel):
    """Legacy result class for deployment jobs."""

    status: TogetherJobStatus
    job_id: str
    model_name: str
    failure_reason: str | None
