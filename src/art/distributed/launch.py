from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .specs import (
    ClusterSpec,
    GpuId,
    HostSpec,
    NcclTransportSpec,
    NixlTransportSpec,
)


class ArtLaunchContext(BaseModel):
    """Provider-neutral resources owned by an ``art-monarch`` invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    host_mesh: Any = Field(exclude=True)
    worker_addresses: tuple[str, ...]
    controller_rank: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_hosts(self) -> "ArtLaunchContext":
        if not self.worker_addresses:
            raise ValueError("worker_addresses must not be empty")
        if len(set(self.worker_addresses)) != len(self.worker_addresses):
            raise ValueError("worker_addresses must be unique")
        if self.controller_rank >= len(self.worker_addresses):
            raise ValueError("controller_rank must identify a worker address")
        return self

    @property
    def host_count(self) -> int:
        return len(self.worker_addresses)

    def homogeneous_cluster(
        self,
        *,
        cpu_slots: int,
        gpu_ids: tuple[GpuId, ...] = (),
        artifact_root: str | None = None,
        cache_root: str | None = None,
        nccl_transport: NcclTransportSpec | None = None,
        nixl_transport: NixlTransportSpec | None = None,
        startup_timeout_s: float = 600.0,
        rpc_timeout_s: float = 60.0,
    ) -> ClusterSpec:
        host_ids = tuple(f"host{rank}" for rank in range(self.host_count))
        return ClusterSpec(
            hosts=tuple(
                HostSpec(
                    host_id=host_id,
                    node_rank=rank,
                    worker_address=address,
                    cpu_slots=cpu_slots,
                    gpu_ids=gpu_ids,
                )
                for rank, (host_id, address) in enumerate(
                    zip(host_ids, self.worker_addresses, strict=True)
                )
            ),
            controller_host_id=host_ids[self.controller_rank],
            artifact_root=artifact_root,
            cache_root=cache_root,
            nccl_transport=nccl_transport,
            nixl_transport=nixl_transport,
            startup_timeout_s=startup_timeout_s,
            rpc_timeout_s=rpc_timeout_s,
        )
