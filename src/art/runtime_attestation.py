from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed.host_admission import GpuIdentity, HostAdmissionReport

_SHA256 = r"^[0-9a-f]{64}$"


class RuntimeArchitectureAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_kind: Literal["trainer", "inference"]
    base_model: str = Field(min_length=1, max_length=512)
    model_source: str = Field(min_length=1, max_length=2048)
    model_revision: str = Field(min_length=1, max_length=512)
    model_support_key: str = Field(min_length=1, max_length=255)
    handler_name: str = Field(min_length=1, max_length=255)
    canonical_config_sha256: str = Field(pattern=_SHA256)
    loaded_layer_count: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    context_parallel_size: int = Field(ge=1)
    pipeline_parallel_size: int = Field(ge=1)
    expert_parallel_size: int = Field(ge=1)
    data_parallel_size: int = Field(ge=1)
    world_size: int = Field(ge=1)
    runtime_identity: str = Field(min_length=1, max_length=512)
    architecture_sha256: str = Field(pattern=_SHA256)

    @classmethod
    def create(cls, **values: Any) -> Self:
        return cls(**values, architecture_sha256=_architecture_digest(values))

    @model_validator(mode="after")
    def _validate_fingerprint(self) -> Self:
        values = self.model_dump(mode="json", exclude={"architecture_sha256"})
        if self.architecture_sha256 != _architecture_digest(values):
            raise ValueError("architecture fingerprint does not match its fields")
        return self


class RuntimeHostAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host_id: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    boot_id: str = Field(min_length=1)
    runtime_sha256: str = Field(pattern=_SHA256)
    assigned_gpus: tuple[GpuIdentity, ...] = Field(min_length=1)

    @classmethod
    def from_admission(cls, report: HostAdmissionReport) -> Self:
        return cls(
            host_id=report.host_id,
            hostname=report.hostname,
            boot_id=report.boot_id,
            runtime_sha256=report.runtime.sha256,
            assigned_gpus=report.assigned_gpus,
        )


class PairedSlotRuntimeAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trainer: RuntimeArchitectureAttestation
    inference: RuntimeArchitectureAttestation
    trainer_hosts: tuple[RuntimeHostAttestation, ...] = Field(min_length=1)
    inference_hosts: tuple[RuntimeHostAttestation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_roles(self) -> Self:
        if self.trainer.runtime_kind != "trainer":
            raise ValueError("trainer attestation has another runtime kind")
        if self.inference.runtime_kind != "inference":
            raise ValueError("inference attestation has another runtime kind")
        return self


def canonical_config_sha256(config: Any) -> str:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise RuntimeError("resolved model config does not expose to_dict()")
    values = to_dict()
    if not isinstance(values, dict):
        raise RuntimeError("resolved model config did not produce an object")
    return _digest(values)


def _digest(values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _architecture_digest(values: dict[str, Any]) -> str:
    return _digest(
        {key: value for key, value in values.items() if key != "runtime_identity"}
    )
