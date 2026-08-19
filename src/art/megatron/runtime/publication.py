from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.megatron.optimizer_state import (
    OptimizerAdapter,
    OptimizerGenerationManifest,
    OptimizerShard,
    OptimizerTopology,
    build_optimizer_manifest,
    commit_optimizer_generation,
    optimizer_generation_nbytes,
    read_committed_optimizer_pointer,
)

from .specs import TrainerGeneration


class _PublicationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SnapshotRankWritePlan(_PublicationModel):
    rank: int = Field(ge=0)
    generation: TrainerGeneration
    adapter: OptimizerAdapter | None = None
    transport_adapter: OptimizerAdapter | None = None
    optimizer_shard: OptimizerShard | None = None
    runtime_sha256: str | None = None
    topology: OptimizerTopology | None = None
    saves_optimizer: bool

    @model_validator(mode="after")
    def _validate_payload(self) -> "SnapshotRankWritePlan":
        optimizer_values = (
            self.optimizer_shard,
            self.runtime_sha256,
            self.topology,
        )
        if self.saves_optimizer != all(value is not None for value in optimizer_values):
            raise ValueError("optimizer write-plan fields must be present together")
        if not self.saves_optimizer and any(
            value is not None for value in optimizer_values
        ):
            raise ValueError("non-optimizer write plan contains optimizer fields")
        if self.rank == 0:
            if self.adapter is None and self.transport_adapter is None:
                raise ValueError("rank-zero write plan requires an adapter")
            for adapter in (self.adapter, self.transport_adapter):
                if adapter is not None and (
                    adapter.training_session_id,
                    adapter.step,
                    adapter.generation_id,
                ) != (
                    self.generation.training_session_id,
                    self.generation.policy_step,
                    self.generation.generation_id,
                ):
                    raise ValueError("adapter write plan identifies another generation")
            if (
                self.adapter is not None
                and self.transport_adapter is not None
                and self.adapter.files != self.transport_adapter.files
            ):
                raise ValueError("local and transport adapter bytes differ")
        elif self.adapter is not None or self.transport_adapter is not None:
            raise ValueError("only rank zero may plan adapter writes")
        if self.optimizer_shard is not None and self.optimizer_shard.rank != self.rank:
            raise ValueError("optimizer write plan identifies another rank")
        return self


class SnapshotWritePlan(_PublicationModel):
    operation_id: str = Field(min_length=1)
    generation: TrainerGeneration
    ranks: tuple[SnapshotRankWritePlan, ...] = Field(min_length=1)
    adapter: OptimizerAdapter
    optimizer_manifest: OptimizerGenerationManifest | None = None
    adapter_bytes: int = Field(gt=0)
    optimizer_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_plan(self) -> "SnapshotWritePlan":
        ordered = tuple(sorted(self.ranks, key=lambda rank: rank.rank))
        if tuple(rank.rank for rank in ordered) != tuple(range(len(ordered))):
            raise ValueError("write plan must cover every rank exactly once")
        if ordered != self.ranks:
            raise ValueError("write-plan ranks must be ordered")
        if {rank.generation for rank in ordered} != {self.generation}:
            raise ValueError("write-plan ranks identify another generation")
        rank_zero_adapter = ordered[0].adapter or ordered[0].transport_adapter
        if rank_zero_adapter != self.adapter:
            raise ValueError("logical adapter differs from rank-zero write plan")
        if self.adapter_bytes != sum(file.size_bytes for file in self.adapter.files):
            raise ValueError("adapter byte count differs from its exact files")
        saves_optimizer = {rank.saves_optimizer for rank in ordered}
        if len(saves_optimizer) != 1:
            raise ValueError("write-plan ranks disagree on optimizer persistence")
        if saves_optimizer == {False}:
            if self.optimizer_manifest is not None or self.optimizer_bytes:
                raise ValueError("adapter-only write plan contains optimizer bytes")
            return self
        manifest = self.optimizer_manifest
        if manifest is None:
            raise ValueError("optimizer write plan has no final manifest")
        if (
            manifest.generation,
            manifest.step,
            manifest.adapter,
        ) != (
            self.generation.generation_id,
            self.generation.policy_step,
            self.adapter,
        ):
            raise ValueError("optimizer manifest identifies another generation")
        shards = tuple(
            rank.optimizer_shard for rank in ordered if rank.optimizer_shard is not None
        )
        if manifest.shards != shards:
            raise ValueError("optimizer manifest differs from exact rank plans")
        if self.optimizer_bytes != optimizer_generation_nbytes(manifest):
            raise ValueError("optimizer byte count differs from its exact manifest")
        return self

    @property
    def digest(self) -> str:
        return snapshot_write_plan_digest(self)


class SnapshotWriteGrant(_PublicationModel):
    operation_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def local(cls, plan: SnapshotWritePlan) -> "SnapshotWriteGrant":
        return cls(
            operation_id=plan.operation_id,
            generation_id=plan.generation.generation_id,
            plan_digest=plan.digest,
        )

    def validate_plan(self, plan: SnapshotWritePlan) -> None:
        if (
            self.operation_id,
            self.generation_id,
            self.plan_digest,
        ) != (
            plan.operation_id,
            plan.generation.generation_id,
            plan.digest,
        ):
            raise RuntimeError("snapshot write grant does not match its exact plan")


class PreparedSave(_PublicationModel):
    operation_id: str = Field(min_length=1)
    kind: Literal["sampler_weights", "state"]
    generation: TrainerGeneration
    plan: SnapshotWritePlan
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_plan(self) -> "PreparedSave":
        if (
            self.operation_id,
            self.generation,
            self.plan_digest,
        ) != (
            self.plan.operation_id,
            self.plan.generation,
            self.plan.digest,
        ):
            raise ValueError("prepared save identity differs from its write plan")
        return self


def snapshot_write_plan_digest(plan: SnapshotWritePlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_snapshot_write_plan(
    *,
    operation_id: str,
    generation: TrainerGeneration,
    ranks: tuple[SnapshotRankWritePlan, ...],
) -> SnapshotWritePlan:
    ordered = tuple(sorted(ranks, key=lambda rank: rank.rank))
    if not ordered:
        raise RuntimeError("snapshot write plan has no trainer ranks")
    adapter = ordered[0].adapter or ordered[0].transport_adapter
    if adapter is None:
        raise RuntimeError("snapshot write plan has no rank-zero adapter")
    saves_optimizer = {rank.saves_optimizer for rank in ordered}
    if len(saves_optimizer) != 1:
        raise RuntimeError("trainer ranks disagree on optimizer persistence")
    manifest = None
    if saves_optimizer == {True}:
        runtime_ids = {rank.runtime_sha256 for rank in ordered}
        topologies = {rank.topology for rank in ordered}
        if len(runtime_ids) != 1 or len(topologies) != 1:
            raise RuntimeError("trainer ranks planned incompatible optimizer archives")
        runtime_sha256 = runtime_ids.pop()
        topology = topologies.pop()
        if runtime_sha256 is None or topology is None:
            raise RuntimeError("optimizer write plan is incomplete")
        manifest = build_optimizer_manifest(
            generation=generation.generation_id,
            step=generation.policy_step,
            adapter=adapter,
            runtime_sha256=runtime_sha256,
            world_size=len(ordered),
            shards=[
                rank.optimizer_shard
                for rank in ordered
                if rank.optimizer_shard is not None
            ],
            topology=topology,
        )
    return SnapshotWritePlan(
        operation_id=operation_id,
        generation=generation,
        ranks=ordered,
        adapter=adapter,
        optimizer_manifest=manifest,
        adapter_bytes=sum(file.size_bytes for file in adapter.files),
        optimizer_bytes=(
            0 if manifest is None else optimizer_generation_nbytes(manifest)
        ),
    )


class TrainerRankPublication(_PublicationModel):
    generation: TrainerGeneration
    rank: int = Field(ge=0)
    plan: SnapshotRankWritePlan
    grant: SnapshotWriteGrant
    adapter: OptimizerAdapter | None = None
    transport_adapter: OptimizerAdapter | None = None
    shard: OptimizerShard | None = None
    runtime_sha256: str | None = None
    topology: OptimizerTopology | None = None
    saves_optimizer: bool
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "TrainerRankPublication":
        if self.plan.rank != self.rank or self.plan.generation != self.generation:
            raise ValueError("rank publication identifies another prepared plan")
        if self.grant.generation_id != self.generation.generation_id:
            raise ValueError("rank publication carries another write grant")
        optimizer_values = (self.shard, self.runtime_sha256, self.topology)
        if (
            self.saves_optimizer
            and not all(value is not None for value in optimizer_values)
        ) or (
            not self.saves_optimizer
            and any(value is not None for value in optimizer_values)
        ):
            raise ValueError("optimizer publication fields must be present together")
        if self.rank == 0:
            if self.adapter is None:
                raise ValueError("rank zero publication requires an adapter")
            for adapter in (self.adapter, self.transport_adapter):
                if adapter is not None and (
                    adapter.training_session_id,
                    adapter.step,
                    adapter.generation_id,
                ) != (
                    self.generation.training_session_id,
                    self.generation.policy_step,
                    self.generation.generation_id,
                ):
                    raise ValueError("adapter and trainer generation identities differ")
        elif self.adapter is not None or self.transport_adapter is not None:
            raise ValueError("only rank zero may publish adapter manifests")
        if self.shard is not None and self.shard.rank != self.rank:
            raise ValueError("optimizer shard identifies another trainer rank")
        if (
            self.adapter,
            self.transport_adapter,
            self.shard,
            self.runtime_sha256,
            self.topology,
            self.saves_optimizer,
        ) != (
            self.plan.adapter or self.plan.transport_adapter,
            self.plan.transport_adapter,
            self.plan.optimizer_shard,
            self.plan.runtime_sha256,
            self.plan.topology,
            self.plan.saves_optimizer,
        ):
            raise ValueError("rank publication differs from its prepared plan")
        return self


class TrainerPublicationSucceeded(_PublicationModel):
    kind: Literal["publication_succeeded"] = "publication_succeeded"
    record: TrainerRankPublication


class TrainerPublicationFailed(_PublicationModel):
    kind: Literal["publication_failed"] = "publication_failed"
    generation_id: str = Field(min_length=1)
    rank: int = Field(ge=0)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


TrainerPublicationEvent = Annotated[
    TrainerPublicationSucceeded | TrainerPublicationFailed,
    Field(discriminator="kind"),
]
TRAINER_PUBLICATION_EVENT_ADAPTER = TypeAdapter(TrainerPublicationEvent)


class DurableTrainerPublication(_PublicationModel):
    adapter: OptimizerAdapter
    transport_adapter: OptimizerAdapter | None = None
    resume_step: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)
    optimizer_bytes: int | None = Field(default=None, gt=0)


def commit_trainer_publication(
    optimizer_state_path: str,
    generation: TrainerGeneration,
    records: tuple[TrainerRankPublication, ...],
    *,
    plan: SnapshotWritePlan | None = None,
    grant: SnapshotWriteGrant | None = None,
) -> DurableTrainerPublication:
    ordered = tuple(sorted(records, key=lambda record: record.rank))
    if tuple(record.rank for record in ordered) != tuple(range(len(ordered))):
        raise RuntimeError("trainer publication does not cover every rank exactly once")
    if not ordered or {record.generation for record in ordered} != {generation}:
        raise RuntimeError("trainer ranks published another generation")
    if len({record.saves_optimizer for record in ordered}) != 1:
        raise RuntimeError("trainer ranks disagree on optimizer persistence")
    embedded_grant = ordered[0].grant
    if any(record.grant != embedded_grant for record in ordered[1:]):
        raise RuntimeError("trainer ranks published under different write grants")
    embedded_plan = build_snapshot_write_plan(
        operation_id=embedded_grant.operation_id,
        generation=generation,
        ranks=tuple(record.plan for record in ordered),
    )
    if plan is None:
        plan = embedded_plan
    elif plan != embedded_plan:
        raise RuntimeError("committed write plan differs from rank authorization")
    if grant is None:
        grant = embedded_grant
    elif grant != embedded_grant:
        raise RuntimeError("committed write grant differs from rank authorization")
    grant.validate_plan(plan)
    if plan.generation != generation:
        raise RuntimeError("snapshot write plan identifies another generation")
    for record, expected in zip(ordered, plan.ranks, strict=True):
        if (
            record.adapter,
            record.transport_adapter,
            record.shard,
            record.runtime_sha256,
            record.topology,
            record.saves_optimizer,
        ) != (
            expected.adapter or expected.transport_adapter,
            expected.transport_adapter,
            expected.optimizer_shard,
            expected.runtime_sha256,
            expected.topology,
            expected.saves_optimizer,
        ):
            raise RuntimeError("trainer publication differs from its authorized plan")
    adapter = ordered[0].adapter
    if adapter is None:
        raise RuntimeError("trainer publication has no rank-zero adapter")
    saves_optimizer = ordered[0].saves_optimizer
    optimizer_bytes = None
    if saves_optimizer:
        manifest = plan.optimizer_manifest
        if manifest is None:
            raise RuntimeError("authorized optimizer publication has no manifest")
        expected = read_committed_optimizer_pointer(optimizer_state_path)
        optimizer_bytes = optimizer_generation_nbytes(manifest)
        commit_optimizer_generation(
            optimizer_state_path,
            manifest,
            expected_pointer=expected,
        )
    committed = read_committed_optimizer_pointer(optimizer_state_path)
    optimizer_step = 0 if committed is None else committed.step
    return DurableTrainerPublication(
        adapter=adapter,
        transport_adapter=ordered[0].transport_adapter,
        resume_step=generation.policy_step if saves_optimizer else optimizer_step,
        optimizer_step=optimizer_step,
        optimizer_bytes=optimizer_bytes,
    )
