"""Immutable, backend-neutral values used by ART distillation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import hashlib
import json
import math
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

JsonValue = (
    None | bool | int | float | str | tuple["JsonValue", ...] | dict[str, "JsonValue"]
)


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation without non-finite numbers."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationPart(StrEnum):
    REASONING = "reasoning"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_CALL = "tool_call"


class TokenSpan(ImmutableModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("token span end must be greater than start")
        return self


class PartSpan(TokenSpan):
    part: GenerationPart


class RolloutRevisionSpan(TokenSpan):
    revision: Annotated[int, Field(ge=0)]
    inference_name: str | None = None
    update_seq: Annotated[int | None, Field(ge=0)] = None


TeacherProtocol = Literal[
    "chat_completions",
    "completions",
    "responses",
    "messages",
]


class TeacherView(ImmutableModel):
    """A canonical semantic request; helpers must return a new view."""

    protocol: TeacherProtocol
    canonical_request_json: str
    fingerprint: str

    @classmethod
    def from_request(cls, protocol: TeacherProtocol, request: Any) -> Self:
        encoded = canonical_json(request)
        return cls(
            protocol=protocol,
            canonical_request_json=encoded,
            fingerprint=sha256_text(f"{protocol}\0{encoded}"),
        )

    @model_validator(mode="after")
    def _valid_canonical_request(self) -> Self:
        try:
            decoded = json.loads(self.canonical_request_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("teacher view request must be valid JSON") from exc
        canonical = canonical_json(decoded)
        if canonical != self.canonical_request_json:
            raise ValueError("teacher view request must use canonical JSON")
        expected = sha256_text(f"{self.protocol}\0{canonical}")
        if self.fingerprint != expected:
            raise ValueError("teacher view fingerprint does not match its request")
        return self

    def request(self) -> JsonValue:
        """Decode a fresh request value so callers cannot mutate this view."""

        return json.loads(self.canonical_request_json)


class CapturedGeneration(ImmutableModel):
    """One exact model-output event captured from a trajectory."""

    generation_id: Annotated[str, Field(min_length=1)]
    trajectory_fingerprint: Annotated[str, Field(min_length=1)]
    event_index: Annotated[int, Field(ge=0)]
    trajectory_token_start: Annotated[int, Field(ge=0)]
    protocol: TeacherProtocol
    continuation_token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    continuation_sha256: str
    context: TeacherView
    part_spans: tuple[PartSpan, ...]
    rollout_spans: tuple[RolloutRevisionSpan, ...]

    @classmethod
    def create(
        cls,
        *,
        generation_id: str,
        trajectory_fingerprint: str,
        event_index: int,
        trajectory_token_start: int,
        protocol: TeacherProtocol,
        continuation_token_ids: tuple[int, ...],
        context: TeacherView,
        part_spans: tuple[PartSpan, ...],
        rollout_spans: tuple[RolloutRevisionSpan, ...],
    ) -> Self:
        token_json = canonical_json(continuation_token_ids)
        return cls(
            generation_id=generation_id,
            trajectory_fingerprint=trajectory_fingerprint,
            event_index=event_index,
            trajectory_token_start=trajectory_token_start,
            protocol=protocol,
            continuation_token_ids=continuation_token_ids,
            continuation_sha256=sha256_text(token_json),
            context=context,
            part_spans=part_spans,
            rollout_spans=rollout_spans,
        )

    @model_validator(mode="after")
    def _valid_capture(self) -> Self:
        token_count = len(self.continuation_token_ids)
        if token_count == 0:
            raise ValueError("captured generation must contain continuation tokens")
        expected = sha256_text(canonical_json(self.continuation_token_ids))
        if self.continuation_sha256 != expected:
            raise ValueError("continuation hash does not match token IDs")
        if self.protocol != self.context.protocol:
            raise ValueError("capture and teacher context protocols must match")
        _validate_spans(self.part_spans, token_count, "part")
        _validate_spans(self.rollout_spans, token_count, "rollout revision")
        return self


def _validate_spans(
    spans: tuple[TokenSpan, ...],
    token_count: int,
    label: str,
) -> None:
    previous_end = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.end > token_count:
            raise ValueError(f"{label} span exceeds continuation token count")
        if span.start < previous_end:
            raise ValueError(f"{label} spans must not overlap")
        previous_end = span.end


class CanonicalJsonObject(ImmutableModel):
    """An immutable JSON object with mapping-style read access."""

    canonical_json: str

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> Self:
        return cls(canonical_json=canonical_json(value))

    @field_validator("canonical_json")
    @classmethod
    def _valid_json_object(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("value must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("value must be a JSON object")
        if canonical_json(decoded) != value:
            raise ValueError("value must use canonical JSON")
        return value

    def to_dict(self) -> dict[str, JsonValue]:
        return json.loads(self.canonical_json)

    def __getitem__(self, key: str) -> JsonValue:
        return self.to_dict()[key]


class Example(ImmutableModel):
    """Application-materialized input for one selected captured generation."""

    generation: CapturedGeneration
    teacher_view: TeacherView
    parts: tuple[GenerationPart, ...]
    provenance: CanonicalJsonObject = Field(
        default_factory=lambda: CanonicalJsonObject.from_value({})
    )

    @classmethod
    def create(
        cls,
        *,
        generation: CapturedGeneration,
        teacher_view: TeacherView,
        parts: frozenset[GenerationPart] | set[GenerationPart],
        provenance: Any = None,
    ) -> Self:
        return cls(
            generation=generation,
            teacher_view=teacher_view,
            parts=parts,
            provenance={} if provenance is None else provenance,
        )

    @field_validator("parts", mode="before")
    @classmethod
    def _normalize_parts(cls, value: Any) -> tuple[GenerationPart, ...]:
        normalized = tuple(sorted(set(value), key=str))
        if not normalized:
            raise ValueError("example must select at least one generation part")
        return normalized

    @field_validator("provenance", mode="before")
    @classmethod
    def _normalize_provenance(cls, value: Any) -> CanonicalJsonObject:
        if isinstance(value, CanonicalJsonObject):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("example provenance must be a JSON object")
        return CanonicalJsonObject.from_value(value)


class TopK(ImmutableModel):
    kind: Literal["top_k"] = "top_k"
    k: PositiveInt = 32
    temperature: PositiveFloat = 1.0


class ForwardKL(ImmutableModel):
    kind: Literal["forward_kl"] = "forward_kl"


class Loss(ImmutableModel):
    divergence: ForwardKL = Field(default_factory=ForwardKL)
    coefficient: Annotated[float, Field(gt=0)] = 1.0
    compensate_temperature_squared: bool = False


class TrainingObjectives(ImmutableModel):
    policy: Literal["cispo", "ppo"] | None = None
    distillation: Loss | None = None

    @model_validator(mode="after")
    def _has_objective(self) -> Self:
        if self.policy is None and self.distillation is None:
            raise ValueError("at least one training objective is required")
        return self


class StudentOnPolicy(ImmutableModel):
    kind: Literal["student_on_policy"] = "student_on_policy"


class AnyRevision(ImmutableModel):
    kind: Literal["any_revision"] = "any_revision"


RolloutRequirement = StudentOnPolicy | AnyRevision


class TeacherRevision(ImmutableModel):
    teacher_name: Annotated[str, Field(min_length=1)]
    revision: int | str

    @field_validator("revision")
    @classmethod
    def _valid_revision(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 0:
            raise ValueError("revision must be non-negative")
        if isinstance(value, str) and not value:
            raise ValueError("revision must not be empty")
        return value


class Frozen(ImmutableModel):
    kind: Literal["frozen"] = "frozen"
    revision: int | str | None = None
    revisions: tuple[TeacherRevision, ...] = ()

    @model_validator(mode="after")
    def _valid_revisions(self) -> Self:
        if (self.revision is None) == (not self.revisions):
            raise ValueError("frozen consistency requires revision or revisions")
        names = [item.teacher_name for item in self.revisions]
        if len(set(names)) != len(names):
            raise ValueError("frozen teacher names must be unique")
        if names != sorted(names):
            raise ValueError("frozen teacher revisions must be canonically ordered")
        return self

    def revision_for(self, teacher_name: str) -> int | str:
        if self.revision is not None:
            return self.revision
        for item in self.revisions:
            if item.teacher_name == teacher_name:
                return item.revision
        raise ValueError(
            f"teacher {teacher_name!r} is absent from frozen revision constraints"
        )


class CurrentStep(ImmutableModel):
    kind: Literal["current_step"] = "current_step"
    revision: Annotated[int, Field(ge=0)]
    session_id: Annotated[str, Field(min_length=1)]


Consistency = Frozen | CurrentStep


class TrainingTrajectorySnapshot(ImmutableModel):
    """Allowlisted token-level projection of one trainable trajectory."""

    trajectory_fingerprint: Annotated[str, Field(min_length=1)]
    token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    logprobs: tuple[float | None, ...]
    token_flags: tuple[Annotated[int, Field(ge=0)], ...]
    reward: float
    advantage: float
    generations: tuple[CapturedGeneration, ...]

    @model_validator(mode="after")
    def _valid_trajectory(self) -> Self:
        if not self.token_ids:
            raise ValueError("training trajectory snapshot must contain tokens")
        if len(self.logprobs) != len(self.token_ids):
            raise ValueError("trajectory log probabilities must align with tokens")
        if len(self.token_flags) != len(self.token_ids):
            raise ValueError("trajectory token flags must align with tokens")
        finite_values = [
            self.reward,
            self.advantage,
            *(value for value in self.logprobs if value is not None),
        ]
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("trajectory numeric values must be finite")
        generation_ids: set[str] = set()
        occupied_ranges: list[tuple[int, int]] = []
        for generation in self.generations:
            if generation.generation_id in generation_ids:
                raise ValueError("generation IDs must be unique within a trajectory")
            generation_ids.add(generation.generation_id)
            if generation.trajectory_fingerprint != self.trajectory_fingerprint:
                raise ValueError("generation belongs to a different trajectory")
            end = generation.trajectory_token_start + len(
                generation.continuation_token_ids
            )
            if end > len(self.token_ids):
                raise ValueError("generation exceeds trajectory token bounds")
            actual = self.token_ids[generation.trajectory_token_start : end]
            if actual != generation.continuation_token_ids:
                raise ValueError("generation tokens do not match trajectory tokens")
            occupied_ranges.append((generation.trajectory_token_start, end))
        occupied_ranges.sort()
        for previous, current in zip(occupied_ranges, occupied_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("captured generations must not overlap")
        return self


class TrainingGroupSnapshot(ImmutableModel):
    """Allowlisted ordered projection of one ART trajectory group."""

    group_id: Annotated[str, Field(min_length=1)]
    trajectories: tuple[TrainingTrajectorySnapshot, ...]

    @model_validator(mode="after")
    def _valid_group(self) -> Self:
        fingerprints = [
            trajectory.trajectory_fingerprint for trajectory in self.trajectories
        ]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("trajectory fingerprints must be unique within a group")
        return self


class TopKTargetRow(ImmutableModel):
    kind: Literal["top_k"] = "top_k"
    generation_id: Annotated[str, Field(min_length=1)]
    position: Annotated[int, Field(ge=0)]
    sampled_token_id: Annotated[int, Field(ge=0)]
    token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    teacher_logprobs: tuple[float, ...]
    tail_logprob: float | None
    logical_vocab_size: PositiveInt
    temperature: PositiveFloat
    teacher_name: Annotated[str, Field(min_length=1)]
    teacher_revision: int | str
    token_space_fingerprint: Annotated[str, Field(min_length=1)]
    request_id: Annotated[str, Field(min_length=1)]
    forced_token_sha256: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _valid_distribution(self) -> Self:
        if not self.token_ids:
            raise ValueError("top-k target row must contain at least one token")
        if len(self.token_ids) != len(self.teacher_logprobs):
            raise ValueError("token IDs and teacher log probabilities must align")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("top-k token IDs must be unique")
        if any(token_id >= self.logical_vocab_size for token_id in self.token_ids):
            raise ValueError("top-k token ID exceeds logical vocabulary")
        if self.sampled_token_id >= self.logical_vocab_size:
            raise ValueError("sampled token ID exceeds logical vocabulary")
        if len(self.token_ids) > self.logical_vocab_size:
            raise ValueError("top-k width exceeds logical vocabulary")
        if any(not math.isfinite(value) for value in self.teacher_logprobs):
            raise ValueError("teacher log probabilities must be finite")
        if self.tail_logprob is not None and not math.isfinite(self.tail_logprob):
            raise ValueError("tail log probability must be finite when present")
        if len(self.token_ids) == self.logical_vocab_size:
            if self.tail_logprob is not None:
                raise ValueError("full-vocabulary rows must omit the tail")
            total = math.fsum(math.exp(value) for value in self.teacher_logprobs)
        else:
            if self.tail_logprob is None:
                raise ValueError("sparse top-k rows must include a tail")
            total = math.fsum(
                [
                    *(math.exp(value) for value in self.teacher_logprobs),
                    math.exp(self.tail_logprob),
                ]
            )
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("teacher target probabilities must sum to one")
        return self


class PreparationReport(ImmutableModel):
    selected_generations: Annotated[int, Field(ge=0)]
    selected_tokens: Annotated[int, Field(ge=0)]
    prepared_tokens: Annotated[int, Field(ge=0)]
    issue_count: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _valid_counts(self) -> Self:
        if self.prepared_tokens > self.selected_tokens:
            raise ValueError("prepared token count exceeds selected token count")
        return self


class PreparedConstraints(ImmutableModel):
    learner_revision: Annotated[int, Field(ge=0)]
    token_space_fingerprint: Annotated[str, Field(min_length=1)]
    logical_vocab_size: PositiveInt
    rollout_requirement: RolloutRequirement = Field(discriminator="kind")
    consistency: Consistency = Field(discriminator="kind")
