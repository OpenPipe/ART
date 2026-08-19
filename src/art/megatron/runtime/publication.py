from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.megatron.optimizer_state import (
    OptimizerAdapter,
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


class TrainerRankPublication(_PublicationModel):
    generation: TrainerGeneration
    rank: int = Field(ge=0)
    adapter: OptimizerAdapter | None = None
    transport_adapter: OptimizerAdapter | None = None
    shard: OptimizerShard | None = None
    runtime_sha256: str | None = None
    topology: OptimizerTopology | None = None
    saves_optimizer: bool
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "TrainerRankPublication":
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
) -> DurableTrainerPublication:
    ordered = tuple(sorted(records, key=lambda record: record.rank))
    if tuple(record.rank for record in ordered) != tuple(range(len(ordered))):
        raise RuntimeError("trainer publication does not cover every rank exactly once")
    if not ordered or {record.generation for record in ordered} != {generation}:
        raise RuntimeError("trainer ranks published another generation")
    if len({record.saves_optimizer for record in ordered}) != 1:
        raise RuntimeError("trainer ranks disagree on optimizer persistence")
    adapter = ordered[0].adapter
    if adapter is None:
        raise RuntimeError("trainer publication has no rank-zero adapter")
    saves_optimizer = ordered[0].saves_optimizer
    optimizer_bytes = None
    if saves_optimizer:
        runtime_ids = {record.runtime_sha256 for record in ordered}
        topologies = {record.topology for record in ordered}
        if len(runtime_ids) != 1 or len(topologies) != 1:
            raise RuntimeError(
                "trainer ranks produced incompatible optimizer snapshots"
            )
        runtime_sha256 = runtime_ids.pop()
        topology = topologies.pop()
        if runtime_sha256 is None or topology is None:
            raise RuntimeError("optimizer publication metadata is incomplete")
        expected = read_committed_optimizer_pointer(optimizer_state_path)
        manifest = build_optimizer_manifest(
            generation=generation.generation_id,
            step=generation.policy_step,
            adapter=adapter,
            runtime_sha256=runtime_sha256,
            world_size=len(ordered),
            shards=[record.shard for record in ordered if record.shard is not None],
            topology=topology,
        )
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
