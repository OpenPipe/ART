"""Canonical, checksum-verified prepared distillation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Self

from .types import (
    CurrentStep,
    Frozen,
    ImmutableModel,
    PreparationReport,
    PreparedConstraints,
    StudentOnPolicy,
    TopKTargetRow,
    TrainingGroupSnapshot,
    canonical_json,
)

PREPARED_BATCH_SCHEMA_VERSION = 1


class PreparedPayload(ImmutableModel):
    schema_version: Literal[1] = PREPARED_BATCH_SCHEMA_VERSION
    groups: tuple[TrainingGroupSnapshot, ...]
    targets: tuple[TopKTargetRow, ...]
    report: PreparationReport
    constraints: PreparedConstraints

    def model_post_init(self, __context: Any) -> None:
        if (
            self.report.selected_generations == 0
            or self.report.selected_tokens == 0
            or self.report.prepared_tokens == 0
        ):
            raise ValueError(
                "prepared distillation batches require selected and prepared tokens"
            )
        if len(self.targets) != self.report.prepared_tokens:
            raise ValueError("prepared token count must equal target row count")
        positions = [(target.generation_id, target.position) for target in self.targets]
        if len(set(positions)) != len(positions):
            raise ValueError("prepared targets must have unique generation positions")
        generations = {}
        group_ids: set[str] = set()
        trajectory_fingerprints: set[str] = set()
        total_generation_tokens = 0
        for group in self.groups:
            if group.group_id in group_ids:
                raise ValueError("group IDs must be unique within a prepared batch")
            group_ids.add(group.group_id)
            for trajectory in group.trajectories:
                if trajectory.trajectory_fingerprint in trajectory_fingerprints:
                    raise ValueError(
                        "trajectory fingerprints must be unique within a prepared batch"
                    )
                trajectory_fingerprints.add(trajectory.trajectory_fingerprint)
                for generation in trajectory.generations:
                    if generation.generation_id in generations:
                        raise ValueError(
                            "generation IDs must be unique within a prepared batch"
                        )
                    generations[generation.generation_id] = generation
                    total_generation_tokens += len(generation.continuation_token_ids)
        if self.report.selected_generations > len(generations):
            raise ValueError("selected generation count exceeds captured generations")
        if self.report.selected_tokens > total_generation_tokens:
            raise ValueError("selected token count exceeds captured generation tokens")
        targeted_generations = {target.generation_id for target in self.targets}
        if len(targeted_generations) > self.report.selected_generations:
            raise ValueError("targeted generations exceed selected generation count")
        if (
            isinstance(self.constraints.consistency, CurrentStep)
            and self.constraints.consistency.revision
            != self.constraints.learner_revision
        ):
            raise ValueError(
                "current-step teacher revision must equal learner revision"
            )
        for target in self.targets:
            generation = generations.get(target.generation_id)
            if generation is None:
                raise ValueError("prepared target references an unknown generation")
            if target.position >= len(generation.continuation_token_ids):
                raise ValueError("prepared target position exceeds generation bounds")
            if (
                target.sampled_token_id
                != generation.continuation_token_ids[target.position]
            ):
                raise ValueError("prepared target sampled token does not match capture")
            if target.forced_token_sha256 != generation.continuation_sha256:
                raise ValueError(
                    "prepared target forced-token hash does not match capture"
                )
            rollout_spans = [
                span
                for span in generation.rollout_spans
                if span.start <= target.position < span.end
            ]
            if len(rollout_spans) != 1:
                raise ValueError(
                    "prepared target lacks exact rollout revision provenance"
                )
            if (
                isinstance(self.constraints.rollout_requirement, StudentOnPolicy)
                or isinstance(self.constraints.consistency, CurrentStep)
            ) and rollout_spans[0].revision != self.constraints.learner_revision:
                raise ValueError(
                    "prepared target rollout revision does not match learner revision"
                )
            if (
                target.token_space_fingerprint
                != self.constraints.token_space_fingerprint
            ):
                raise ValueError(
                    "prepared target token space does not match constraints"
                )
            if target.logical_vocab_size != self.constraints.logical_vocab_size:
                raise ValueError(
                    "prepared target vocabulary does not match constraints"
                )
            consistency = self.constraints.consistency
            expected_revision = (
                consistency.revision_for(target.teacher_name)
                if isinstance(consistency, Frozen)
                else consistency.revision
            )
            if target.teacher_revision != expected_revision:
                raise ValueError(
                    "prepared target teacher revision does not match consistency"
                )


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    """Opaque immutable envelope around canonical prepared bytes."""

    schema_version: int
    batch_id: str
    preparation_id: str
    payload: bytes
    payload_sha256: str

    @classmethod
    def create(
        cls,
        *,
        groups: tuple[TrainingGroupSnapshot, ...],
        targets: tuple[TopKTargetRow, ...],
        report: PreparationReport,
        constraints: PreparedConstraints,
    ) -> Self:
        payload_model = PreparedPayload(
            groups=groups,
            targets=targets,
            report=report,
            constraints=constraints,
        )
        payload = canonical_json(
            payload_model.model_dump(mode="json", round_trip=True)
        ).encode()
        group_projection = canonical_json(
            [group.model_dump(mode="json", round_trip=True) for group in groups]
        ).encode()
        batch_id = hashlib.sha256(
            b"art-distill-batch-v1\0" + group_projection
        ).hexdigest()
        preparation_id = hashlib.sha256(
            b"art-distill-preparation-v1\0" + payload
        ).hexdigest()
        return cls(
            schema_version=PREPARED_BATCH_SCHEMA_VERSION,
            batch_id=batch_id,
            preparation_id=preparation_id,
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def parsed_payload(self) -> PreparedPayload:
        self._validate_envelope()
        return PreparedPayload.model_validate_json(self.payload)

    @property
    def report(self) -> PreparationReport:
        return self.parsed_payload().report

    @property
    def constraints(self) -> PreparedConstraints:
        return self.parsed_payload().constraints

    def to_bytes(self) -> bytes:
        self._validate_envelope()
        envelope = {
            "batch_id": self.batch_id,
            "payload": self.payload.decode(),
            "payload_sha256": self.payload_sha256,
            "preparation_id": self.preparation_id,
            "schema_version": self.schema_version,
        }
        return canonical_json(envelope).encode()

    @classmethod
    def from_bytes(cls, value: bytes) -> Self:
        try:
            raw: Any = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prepared batch envelope must be valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("prepared batch envelope must be a JSON object")
        expected_keys = {
            "batch_id",
            "payload",
            "payload_sha256",
            "preparation_id",
            "schema_version",
        }
        if set(raw) != expected_keys:
            raise ValueError("prepared batch envelope has unexpected fields")
        if not isinstance(raw["payload"], str):
            raise ValueError("prepared batch payload must be a JSON string")
        artifact = cls(
            schema_version=raw["schema_version"],
            batch_id=raw["batch_id"],
            preparation_id=raw["preparation_id"],
            payload=raw["payload"].encode(),
            payload_sha256=raw["payload_sha256"],
        )
        artifact._validate_envelope()
        return artifact

    def _validate_envelope(self) -> None:
        if self.schema_version != PREPARED_BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported prepared batch schema version {self.schema_version}"
            )
        if hashlib.sha256(self.payload).hexdigest() != self.payload_sha256:
            raise ValueError("prepared batch payload checksum mismatch")
        parsed = PreparedPayload.model_validate_json(self.payload)
        canonical_payload = canonical_json(
            parsed.model_dump(mode="json", round_trip=True)
        ).encode()
        if canonical_payload != self.payload:
            raise ValueError("prepared batch payload is not canonical")
        expected_batch_id = hashlib.sha256(
            b"art-distill-batch-v1\0"
            + canonical_json(
                [
                    group.model_dump(mode="json", round_trip=True)
                    for group in parsed.groups
                ]
            ).encode()
        ).hexdigest()
        if self.batch_id != expected_batch_id:
            raise ValueError("prepared batch ID mismatch")
        expected_preparation_id = hashlib.sha256(
            b"art-distill-preparation-v1\0" + self.payload
        ).hexdigest()
        if self.preparation_id != expected_preparation_id:
            raise ValueError("prepared batch preparation ID mismatch")
