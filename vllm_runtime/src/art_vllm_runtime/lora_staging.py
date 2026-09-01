"""Bounded descriptor for transport-owned immutable LoRA staging."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MAX_ADAPTER_BYTES = 8 << 30
_MAX_ADAPTER_CONFIG_BYTES = 64 << 10


class StagedLoraDescriptor(BaseModel):
    """Small control descriptor for an already validated local adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=4096)
    source_identity: str = Field(min_length=1, max_length=512)
    layout: Literal["peft_safetensors_v1"] = "peft_safetensors_v1"
    model_bytes: int = Field(gt=0)
    config_bytes: int = Field(gt=0, le=_MAX_ADAPTER_CONFIG_BYTES)

    @property
    def total_bytes(self) -> int:
        return self.model_bytes + self.config_bytes

    @model_validator(mode="after")
    def _validate_total_bytes(self) -> "StagedLoraDescriptor":
        if self.total_bytes > _MAX_ADAPTER_BYTES:
            raise ValueError("LoRA adapter exceeds the runtime byte bound")
        return self
