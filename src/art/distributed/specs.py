from __future__ import annotations

import hashlib
from ipaddress import ip_address
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import MegatronTopologyConfig

VLLM_KV_EVENT_BUFFER_STEPS = 16_384
VLLM_KV_EVENT_HWM = 16_384
VLLM_KV_EVENT_SCHEMA_VERSION = "vllm-0.23/sha256-pickle-v5/bytes"
VLLM_PREFIX_HASH_BLOCK_SIZE = 16
VLLM_PREFIX_HASH_SEED = "0"


def vllm_kv_event_topic(
    replica_id: str, generation: int, generation_digest: str
) -> str:
    identity = f"{replica_id}\0{generation}\0{generation_digest}".encode()
    return f"art.kv.{hashlib.sha256(identity).hexdigest()}"


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


class ClusterSpec(_Spec):
    hosts: tuple[HostSpec, ...]
    controller_host_id: str
    artifact_root: str | None = None
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
        if sorted(node_ranks) != list(range(len(self.hosts))):
            raise ValueError("host node_rank values must be contiguous from zero")
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
    leader: bool = False

    @model_validator(mode="after")
    def _validate_gpu_ids(self) -> "ModelServiceMemberSpec":
        if not self.gpu_ids:
            raise ValueError("model-service members require at least one GPU")
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must be non-negative")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("member gpu_ids must be unique")
        return self


class ModelServiceReplicaSpec(_Spec):
    service_name: str = Field(min_length=1)
    replica_id: str = Field(min_length=1)
    generation: int = Field(default=0, ge=0)
    members: tuple[ModelServiceMemberSpec, ...]
    leader_endpoint: EndpointSpec
    rendezvous: EndpointSpec
    base_model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    runtime_fingerprint: str = Field(min_length=1)
    parallel: VllmParallelSpec
    update_mode: Literal["lora", "merged"]
    kv_event_port: int | None = Field(default=None, ge=1, le=65535)
    kv_replay_port: int | None = Field(default=None, ge=1, le=65535)

    @model_validator(mode="after")
    def _validate_members(self) -> "ModelServiceReplicaSpec":
        if not self.members:
            raise ValueError("replica members must not be empty")
        member_ids = [member.member_id for member in self.members]
        node_ranks = [member.node_rank for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("member_id values must be unique within a replica")
        if sorted(node_ranks) != list(range(len(self.members))):
            raise ValueError("member node_rank values must be contiguous from zero")
        if len({member.host_id for member in self.members}) != len(self.members):
            raise ValueError("native vLLM members must occupy distinct hosts")
        leaders = [member for member in self.members if member.leader]
        if len(leaders) != 1 or leaders[0].node_rank != 0:
            raise ValueError("exactly node_rank 0 must be the replica leader")
        local_world_sizes = {len(member.gpu_ids) for member in self.members}
        if len(local_world_sizes) != 1:
            raise ValueError("native vLLM members must have equal local world sizes")
        if (
            sum(len(member.gpu_ids) for member in self.members)
            != self.parallel.world_size
        ):
            raise ValueError("vLLM TP * PP * DP must equal the replica GPU count")
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
        event_ports = range(
            self.kv_event_base_port,
            self.kv_event_base_port + self.parallel.dp,
        )
        replay_ports = range(
            self.kv_replay_base_port,
            self.kv_replay_base_port + self.parallel.dp,
        )
        if event_ports.stop > 65536 or replay_ports.stop > 65536:
            raise ValueError("vLLM KV event port range exceeds 65535")
        endpoints = [
            (leaders[0].host_id, self.leader_endpoint.port),
            (leaders[0].host_id, self.rendezvous.port),
            *(
                (self.kv_event_member(rank).host_id, port)
                for rank, port in enumerate(event_ports)
            ),
            *(
                (self.kv_event_member(rank).host_id, port)
                for rank, port in enumerate(replay_ports)
            ),
        ]
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("replica control and KV event ports must not overlap")
        return self

    @property
    def kv_event_base_port(self) -> int:
        return self.kv_event_port or self.leader_endpoint.port + 1

    @property
    def kv_replay_base_port(self) -> int:
        return self.kv_replay_port or self.kv_event_base_port + self.parallel.dp

    def kv_event_member(self, publisher_rank: int) -> ModelServiceMemberSpec:
        if not 0 <= publisher_rank < self.parallel.dp:
            raise ValueError("publisher_rank is outside the vLLM DP world")
        local_world_size = len(self.members[0].gpu_ids)
        world_size_within_dp = self.parallel.tp * self.parallel.pp
        if local_world_size >= world_size_within_dp:
            local_dp = local_world_size // world_size_within_dp
            node_rank = publisher_rank // local_dp
        else:
            nodes_per_dp = world_size_within_dp // local_world_size
            node_rank = publisher_rank * nodes_per_dp
        return next(member for member in self.members if member.node_rank == node_rank)

    @property
    def gpu_placements(self) -> tuple[GpuPlacement, ...]:
        return tuple(
            GpuPlacement(host_id=member.host_id, gpu_id=gpu_id)
            for member in self.members
            for gpu_id in member.gpu_ids
        )


class ModelServiceSpec(_Spec):
    name: str = Field(min_length=1)
    capabilities: frozenset[str] = frozenset()
    replicas: tuple[ModelServiceReplicaSpec, ...]

    @model_validator(mode="after")
    def _validate_replicas(self) -> "ModelServiceSpec":
        if not self.replicas:
            raise ValueError("model services require at least one replica")
        replica_ids = [replica.replica_id for replica in self.replicas]
        if len(set(replica_ids)) != len(replica_ids):
            raise ValueError("replica_id values must be unique within a service")
        if any(replica.service_name != self.name for replica in self.replicas):
            raise ValueError("replica service_name must match its model service")
        return self


class RuntimeTopology(_Spec):
    cluster: ClusterSpec
    rollout_host_ids: tuple[str, ...]
    trainer: TrainerMeshSpec | None = None
    model_services: tuple[ModelServiceSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_model_service_ports(self) -> "RuntimeTopology":
        endpoints: list[tuple[str, int, str]] = []
        for service in self.model_services:
            for replica in service.replicas:
                leader = next(member for member in replica.members if member.leader)
                endpoints.extend(
                    (
                        (leader.host_id, replica.leader_endpoint.port, "leader"),
                        (leader.host_id, replica.rendezvous.port, "rendezvous"),
                    )
                )
                for rank in range(replica.parallel.dp):
                    host_id = replica.kv_event_member(rank).host_id
                    endpoints.extend(
                        (
                            (host_id, replica.kv_event_base_port + rank, "kv_event"),
                            (host_id, replica.kv_replay_base_port + rank, "kv_replay"),
                        )
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
    vllm_output_root: str = "/tmp/art-vllm"
    gateway_bind_host: str = "0.0.0.0"
    gateway_advertise_host: str | None = None
    gateway_max_queued: int = Field(default=128, ge=0)
    gateway_route_timeout_s: float = Field(default=1200.0, gt=0)


class HostServiceHealth(_Spec):
    host_id: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    process_id: int = Field(ge=1)
