from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PartitionKind = Literal["head_sharded_full_tree", "token_sharded_chain"]
ShardAxis = Literal["head", "group"]
ReplicationKind = Literal["none", "group_to_heads", "all_partitions"]


class ProjectedStreamSpec(BaseModel):
    """One named slice of a recurrent layer's input projection."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    width: int = Field(gt=0)
    shard_axis: ShardAxis | None = None
    shard_count: int | None = Field(default=None, gt=0)
    replication: ReplicationKind = "none"
    replication_factor: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _validate_partition(self) -> ProjectedStreamSpec:
        if self.shard_axis is None and self.shard_count is not None:
            raise ValueError("shard_count requires a shard_axis")
        if self.shard_count is not None and self.width % self.shard_count:
            raise ValueError("projected stream width must divide evenly across shards")
        if self.replication == "group_to_heads" and self.shard_axis != "group":
            raise ValueError("group_to_heads replication requires group sharding")
        if self.replication == "none" and self.replication_factor != 1:
            raise ValueError("unreplicated streams require replication_factor=1")
        return self


class RecurrentStateSpec(BaseModel):
    """One ordered state tensor carried across prefix-tree segments."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_shape(self) -> RecurrentStateSpec:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("recurrent state shapes must contain positive dimensions")
        return self


class ChainCostSpec(BaseModel):
    """Calibrated costs used to select token-sharded chain work."""

    model_config = ConfigDict(frozen=True)

    summary_bytes_per_segment: int = Field(ge=0)
    summary_exchange_count: int = Field(ge=0)
    summary_bandwidth_bytes_per_ms: float = Field(gt=0)
    summary_compute_segments_per_ms: float = Field(gt=0)
    suffix_scan_latency_ms: float = Field(ge=0)
    suffix_scan_segments_per_ms: float = Field(gt=0)


class TokenShardedChainSpec(BaseModel):
    """Family-owned legality, algebra, and cost contract for chain CP."""

    model_config = ConfigDict(frozen=True)

    alignment: int = Field(gt=0)
    legality_key: str = Field(min_length=1)
    summary_implementation_id: str = Field(min_length=1)
    cost: ChainCostSpec


class HeadShardedFullTreePlannerConfig(BaseModel):
    """CPU bucket policy for head-sharded full-tree execution."""

    model_config = ConfigDict(frozen=True)

    max_padded_tokens: int = Field(default=262_144, gt=0)


class LinearRecurrentContract(BaseModel):
    """Immutable model-family identity for recurrent planning."""

    model_config = ConfigDict(frozen=True)

    schema_identity: Literal["art.linear_recurrent.contract.v1"] = (
        "art.linear_recurrent.contract.v1"
    )
    family_key: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    partition_kind: PartitionKind
    projected_streams: tuple[ProjectedStreamSpec, ...]
    states: tuple[RecurrentStateSpec, ...]
    convolution_width: int = Field(gt=0)
    activation: str = Field(min_length=1)
    local_kernel_implementation_id: str = Field(min_length=1)
    layout_compatibility_key: str = Field(min_length=1)
    chain: TokenShardedChainSpec | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> LinearRecurrentContract:
        stream_names = tuple(stream.name for stream in self.projected_streams)
        state_names = tuple(state.name for state in self.states)
        if not stream_names or len(stream_names) != len(set(stream_names)):
            raise ValueError("projected stream names must be non-empty and unique")
        if not state_names or len(state_names) != len(set(state_names)):
            raise ValueError("recurrent state names must be non-empty and unique")
        if (self.chain is None) == (self.partition_kind == "token_sharded_chain"):
            raise ValueError(
                "token_sharded_chain requires chain metadata and "
                "head_sharded_full_tree forbids it"
            )
        return self

    @property
    def planning_identity(self) -> tuple[str, str, str, str, str]:
        """Stable family identity to combine with workload and planner config."""

        digest = hashlib.sha256(self.model_dump_json().encode()).hexdigest()
        return (
            self.schema_identity,
            self.family_key,
            self.contract_version,
            self.partition_kind,
            digest,
        )
