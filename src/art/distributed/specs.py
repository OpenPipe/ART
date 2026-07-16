from __future__ import annotations

from collections import Counter
from ipaddress import ip_address
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import MegatronTopologyConfig


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HostSpec(_Spec):
    host_id: str = Field(min_length=1)
    node_rank: int = Field(ge=0)
    worker_address: str = Field(min_length=1)
    cpu_slots: int = Field(ge=1)
    gpu_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate_gpu_ids(self) -> "HostSpec":
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must be non-negative")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must be unique within a host")
        return self


class NcclTransportSpec(_Spec):
    net_name: str = Field(min_length=1, pattern=r"^[^\x00\r\n]+$")

    @model_validator(mode="after")
    def _validate_net_name(self) -> "NcclTransportSpec":
        if self.net_name != self.net_name.strip():
            raise ValueError(
                "NCCL network name must not contain surrounding whitespace"
            )
        if self.net_name.casefold() == "socket":
            raise ValueError("multi-host GPU workloads may not use NCCL Socket")
        return self


class ClusterSpec(_Spec):
    hosts: tuple[HostSpec, ...]
    controller_host_id: str
    artifact_root: str | None = None
    nccl_transport: NcclTransportSpec | None = None
    startup_timeout_s: float = Field(default=300.0, gt=0)
    rpc_timeout_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _validate_hosts(self) -> "ClusterSpec":
        if not self.hosts:
            raise ValueError("hosts must not be empty")
        host_ids = [host.host_id for host in self.hosts]
        node_ranks = [host.node_rank for host in self.hosts]
        addresses = [host.worker_address for host in self.hosts]
        if len(set(host_ids)) != len(host_ids):
            raise ValueError("host_id values must be unique")
        if node_ranks != list(range(len(self.hosts))):
            raise ValueError("hosts must be ordered by contiguous node_rank from zero")
        if len(set(addresses)) != len(addresses):
            raise ValueError("worker_address values must be unique")
        if self.controller_host_id not in host_ids:
            raise ValueError("controller_host_id must identify a configured host")
        return self


class GpuPlacement(_Spec):
    host_id: str = Field(min_length=1)
    gpu_id: int = Field(ge=0)


class TrainerMeshSpec(_Spec):
    ranks: tuple[GpuPlacement, ...]
    topology: MegatronTopologyConfig
    coordinator_rank: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_world(self) -> "TrainerMeshSpec":
        if not self.ranks:
            raise ValueError("trainer ranks must not be empty")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("trainer GPU placements must be unique")
        world_size = len(self.ranks)
        topology = self.topology
        if world_size % (topology.tp * topology.cp * topology.pp):
            raise ValueError("trainer world size must be divisible by TP * CP * PP")
        if world_size % (topology.etp * topology.ep * topology.pp):
            raise ValueError("trainer world size must be divisible by ETP * EP * PP")
        return self


class EndpointSpec(_Spec):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_loopback(self) -> bool:
        if self.host.lower() == "localhost":
            return True
        try:
            return ip_address(self.host.strip("[]")).is_loopback
        except ValueError:
            return False

    @property
    def is_routable(self) -> bool:
        if self.host.lower() == "localhost":
            return False
        try:
            address = ip_address(self.host.strip("[]"))
        except ValueError:
            return self.host not in {"0.0.0.0", "::"}
        return not (
            address.is_loopback
            or address.is_unspecified
            or address.is_link_local
            or address.is_multicast
        )


class VllmParallelSpec(_Spec):
    tp: int = Field(default=1, ge=1)
    pp: int = Field(default=1, ge=1)
    dp: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp


class ModelServiceMemberSpec(_Spec):
    member_id: str = Field(min_length=1)
    host_id: str = Field(min_length=1)
    node_rank: int = Field(ge=0)
    gpu_ids: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_gpu_ids(self) -> "ModelServiceMemberSpec":
        if not self.gpu_ids:
            raise ValueError("model-service members require at least one GPU")
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must be non-negative")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("member gpu_ids must be unique")
        return self


class ModelServiceSpec(_Spec):
    name: str = Field(min_length=1)
    capabilities: frozenset[str] = frozenset()
    members: tuple[ModelServiceMemberSpec, ...]
    leader_endpoint: EndpointSpec
    rendezvous: EndpointSpec
    base_model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    runtime_fingerprint: str = Field(min_length=1)
    parallel: VllmParallelSpec
    update_mode: Literal["lora", "merged"]

    @model_validator(mode="after")
    def _validate_members(self) -> "ModelServiceSpec":
        if not self.members:
            raise ValueError("model service members must not be empty")
        member_ids = [member.member_id for member in self.members]
        node_ranks = [member.node_rank for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("member_id values must be unique within a model service")
        if node_ranks != list(range(len(self.members))):
            raise ValueError(
                "members must be ordered by contiguous node_rank from zero"
            )
        if len({member.host_id for member in self.members}) != len(self.members):
            raise ValueError("native vLLM members must occupy distinct hosts")
        local_world_sizes = {len(member.gpu_ids) for member in self.members}
        if len(local_world_sizes) != 1:
            raise ValueError("native vLLM members must have equal local world sizes")
        if (
            sum(len(member.gpu_ids) for member in self.members)
            != self.parallel.world_size
        ):
            raise ValueError("vLLM TP * PP * DP must equal the service GPU count")
        if len(self.members) > 1 and not self.rendezvous.is_routable:
            raise ValueError("multi-host vLLM rendezvous must be routable")
        local_world_size = len(self.members[0].gpu_ids)
        world_size_within_dp = self.parallel.tp * self.parallel.pp
        if (
            local_world_size >= world_size_within_dp
            and local_world_size % world_size_within_dp
        ) or (
            local_world_size < world_size_within_dp
            and world_size_within_dp % local_world_size
        ):
            raise ValueError(
                "native vLLM DP groups must pack evenly within or span whole members"
            )
        if self.leader_endpoint.port == self.rendezvous.port:
            raise ValueError("model-service API and rendezvous ports must not overlap")
        return self

    @property
    def gpu_placements(self) -> tuple[GpuPlacement, ...]:
        return tuple(
            GpuPlacement(host_id=member.host_id, gpu_id=gpu_id)
            for member in self.members
            for gpu_id in member.gpu_ids
        )


class RuntimeTopology(_Spec):
    cluster: ClusterSpec
    rollout_host_ids: tuple[str, ...]
    trainer: TrainerMeshSpec | None = None
    model_services: tuple[ModelServiceSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_runtime(self) -> "RuntimeTopology":
        hosts = {host.host_id: host for host in self.cluster.hosts}
        if len(set(self.rollout_host_ids)) != len(self.rollout_host_ids):
            raise ValueError("rollout_host_ids must be unique")
        unknown_rollout_hosts = sorted(set(self.rollout_host_ids) - hosts.keys())
        if unknown_rollout_hosts:
            raise ValueError(
                f"rollout_host_ids references unknown hosts: {unknown_rollout_hosts}"
            )

        placements: list[tuple[str, int, str]] = []
        if self.trainer is not None:
            trainer_hosts = tuple(rank.host_id for rank in self.trainer.ranks)
            unknown_trainer_hosts = sorted(set(trainer_hosts) - hosts.keys())
            if unknown_trainer_hosts:
                raise ValueError(
                    f"trainer references unknown hosts: {unknown_trainer_hosts}"
                )
            counts = Counter(trainer_hosts)
            if len(set(counts.values())) != 1:
                raise ValueError("Monarch trainer hosts require equal ranks per host")
            selected_hosts = tuple(
                host.host_id for host in self.cluster.hosts if host.host_id in counts
            )
            selected_indices = tuple(
                index
                for index, host in enumerate(self.cluster.hosts)
                if host.host_id in counts
            )
            if selected_indices != tuple(
                range(selected_indices[0], selected_indices[-1] + 1)
            ):
                raise ValueError("trainer hosts must be contiguous in the cluster mesh")
            ranks_per_host = next(iter(counts.values()))
            expected_rank_hosts = tuple(
                host_id for host_id in selected_hosts for _ in range(ranks_per_host)
            )
            if trainer_hosts != expected_rank_hosts:
                raise ValueError(
                    "trainer ranks must be host-major in cluster host order"
                )
            placements.extend(
                (rank.host_id, rank.gpu_id, "trainer") for rank in self.trainer.ranks
            )

        service_names = [service.name for service in self.model_services]
        if len(set(service_names)) != len(service_names):
            raise ValueError("model service names must be unique")

        endpoints: list[tuple[str, int, str]] = []
        for service in self.model_services:
            placements.extend(
                (placement.host_id, placement.gpu_id, service.name)
                for placement in service.gpu_placements
            )
            endpoints.extend(
                (
                    (
                        service.members[0].host_id,
                        service.leader_endpoint.port,
                        "leader",
                    ),
                    (
                        service.members[0].host_id,
                        service.rendezvous.port,
                        "rendezvous",
                    ),
                )
            )
        spans_hosts = (
            self.trainer is not None
            and len({rank.host_id for rank in self.trainer.ranks}) > 1
        ) or any(len(service.members) > 1 for service in self.model_services)
        if spans_hosts and self.cluster.nccl_transport is None:
            raise ValueError("multi-host GPU workloads require nccl_transport")
        for host_id, gpu_id, owner in placements:
            host = hosts.get(host_id)
            if host is None:
                raise ValueError(f"{owner} references unknown host {host_id!r}")
            if gpu_id not in host.gpu_ids:
                raise ValueError(f"{owner} requests unavailable GPU {host_id}:{gpu_id}")
        duplicate_gpus = [
            placement
            for placement, count in Counter(
                (host_id, gpu_id) for host_id, gpu_id, _ in placements
            ).items()
            if count > 1
        ]
        if duplicate_gpus:
            owners = [
                owner
                for duplicate in duplicate_gpus
                for host_id, gpu_id, owner in placements
                if (host_id, gpu_id) == duplicate
            ]
            raise ValueError(
                f"GPU placements overlap at {duplicate_gpus}; owners={owners}"
            )
        seen: dict[tuple[str, int], str] = {}
        for host_id, port, kind in endpoints:
            key = (host_id, port)
            if previous := seen.get(key):
                raise ValueError(
                    f"model-service port {host_id}:{port} overlaps "
                    f"{previous} and {kind}"
                )
            seen[key] = kind
        return self


class ArtRuntimeConfig(_Spec):
    packed_batch_capacity_bytes: int = Field(default=2 << 30, ge=1)
    trajectory_capacity_records: int = Field(default=16_384, ge=1)
    trajectory_capacity_bytes: int = Field(default=4 << 30, ge=1)
    vllm_output_root: str = "/tmp/art-vllm"


class HostServiceHealth(_Spec):
    host_id: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    process_id: int = Field(ge=1)
