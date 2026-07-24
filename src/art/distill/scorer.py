"""Private, transport-neutral contract for acquiring teacher distributions."""

from __future__ import annotations

import math
from typing import Annotated, Protocol, Self

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from .types import (
    ImmutableModel,
    TeacherView,
    TopK,
    canonical_json,
    sha256_text,
)

_REQUEST_HASH_DOMAIN = "art-distill-teacher-scoring-request-v1"
_REQUEST_ID_PREFIX = "teacher-score-"


class TeacherScoringRequest(ImmutableModel):
    """One idempotent request to score a complete forced continuation."""

    request_id: Annotated[str, Field(min_length=1)]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    generation_id: Annotated[str, Field(min_length=1)]
    teacher_view: TeacherView
    forced_token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    forced_token_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_positions: tuple[Annotated[int, Field(ge=0)], ...]
    teacher_name: Annotated[str, Field(min_length=1)]
    teacher_revision: int | str
    token_space_fingerprint: Annotated[str, Field(min_length=1)]
    logical_vocab_size: PositiveInt
    target: TopK

    @classmethod
    def create(
        cls,
        *,
        generation_id: str,
        teacher_view: TeacherView,
        forced_token_ids: tuple[int, ...],
        selected_positions: tuple[int, ...],
        teacher_name: str,
        teacher_revision: int | str,
        token_space_fingerprint: str,
        logical_vocab_size: int,
        target: TopK,
    ) -> Self:
        """Construct the canonical retry identity from scoring semantics."""

        forced_token_sha256 = sha256_text(canonical_json(forced_token_ids))
        projection = _request_projection(
            generation_id=generation_id,
            teacher_view=teacher_view,
            forced_token_ids=forced_token_ids,
            selected_positions=selected_positions,
            teacher_name=teacher_name,
            teacher_revision=teacher_revision,
            token_space_fingerprint=token_space_fingerprint,
            logical_vocab_size=logical_vocab_size,
            target=target,
        )
        request_sha256 = sha256_text(
            f"{_REQUEST_HASH_DOMAIN}\0{canonical_json(projection)}"
        )
        return cls(
            request_id=f"{_REQUEST_ID_PREFIX}{request_sha256}",
            request_sha256=request_sha256,
            generation_id=generation_id,
            teacher_view=teacher_view,
            forced_token_ids=forced_token_ids,
            forced_token_sha256=forced_token_sha256,
            selected_positions=selected_positions,
            teacher_name=teacher_name,
            teacher_revision=teacher_revision,
            token_space_fingerprint=token_space_fingerprint,
            logical_vocab_size=logical_vocab_size,
            target=target,
        )

    @field_validator("teacher_revision")
    @classmethod
    def _valid_revision(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 0:
            raise ValueError("teacher revision must be non-negative")
        if isinstance(value, str) and not value:
            raise ValueError("teacher revision must not be empty")
        return value

    @model_validator(mode="after")
    def _valid_request(self) -> Self:
        if not self.forced_token_ids:
            raise ValueError("forced continuation must contain at least one token")
        if any(
            token_id >= self.logical_vocab_size for token_id in self.forced_token_ids
        ):
            raise ValueError("forced token ID exceeds logical vocabulary")
        if not self.selected_positions:
            raise ValueError("scoring request must select at least one position")
        if tuple(sorted(set(self.selected_positions))) != self.selected_positions:
            raise ValueError("selected positions must be unique and increasing")
        if self.selected_positions[-1] >= len(self.forced_token_ids):
            raise ValueError("selected position exceeds forced continuation bounds")

        expected_forced_hash = sha256_text(canonical_json(self.forced_token_ids))
        if self.forced_token_sha256 != expected_forced_hash:
            raise ValueError("forced-token hash does not match forced continuation")

        projection = _request_projection(
            generation_id=self.generation_id,
            teacher_view=self.teacher_view,
            forced_token_ids=self.forced_token_ids,
            selected_positions=self.selected_positions,
            teacher_name=self.teacher_name,
            teacher_revision=self.teacher_revision,
            token_space_fingerprint=self.token_space_fingerprint,
            logical_vocab_size=self.logical_vocab_size,
            target=self.target,
        )
        expected_request_hash = sha256_text(
            f"{_REQUEST_HASH_DOMAIN}\0{canonical_json(projection)}"
        )
        if self.request_sha256 != expected_request_hash:
            raise ValueError("request hash does not match scoring semantics")
        if self.request_id != f"{_REQUEST_ID_PREFIX}{expected_request_hash}":
            raise ValueError("request ID does not match request hash")
        return self


def _request_projection(
    *,
    generation_id: str,
    teacher_view: TeacherView,
    forced_token_ids: tuple[int, ...],
    selected_positions: tuple[int, ...],
    teacher_name: str,
    teacher_revision: int | str,
    token_space_fingerprint: str,
    logical_vocab_size: int,
    target: TopK,
) -> dict[str, object]:
    return {
        "forced_token_ids": forced_token_ids,
        "generation_id": generation_id,
        "logical_vocab_size": logical_vocab_size,
        "selected_positions": selected_positions,
        "target": target.model_dump(mode="json", round_trip=True),
        "teacher_name": teacher_name,
        "teacher_revision": teacher_revision,
        "teacher_view": teacher_view.model_dump(mode="json", round_trip=True),
        "token_space_fingerprint": token_space_fingerprint,
    }


class RankedTokenLogprob(ImmutableModel):
    """One explicitly ranked token from a teacher's top-k distribution."""

    rank: Annotated[int, Field(ge=0)]
    token_id: Annotated[int, Field(ge=0)]
    logprob: float

    @field_validator("logprob")
    @classmethod
    def _finite_logprob(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("teacher log probability must be finite")
        return value


class ScoredPosition(ImmutableModel):
    """A full-vocabulary-normalized top-k distribution at one local position."""

    position: Annotated[int, Field(ge=0)]
    forced_token_id: Annotated[int, Field(ge=0)]
    entries: tuple[RankedTokenLogprob, ...]
    tail_logprob: float | None
    logical_vocab_size: PositiveInt
    temperature: PositiveFloat

    @field_validator("tail_logprob")
    @classmethod
    def _finite_tail(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("tail log probability must be finite")
        return value

    @model_validator(mode="after")
    def _valid_distribution(self) -> Self:
        if self.forced_token_id >= self.logical_vocab_size:
            raise ValueError("forced token ID exceeds logical vocabulary")
        if not self.entries:
            raise ValueError("scored position must contain top-k entries")
        expected_ranks = tuple(range(len(self.entries)))
        if tuple(entry.rank for entry in self.entries) != expected_ranks:
            raise ValueError("top-k entries must be in contiguous rank order")
        token_ids = tuple(entry.token_id for entry in self.entries)
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("top-k token IDs must be unique")
        if any(token_id >= self.logical_vocab_size for token_id in token_ids):
            raise ValueError("top-k token ID exceeds logical vocabulary")
        logprobs = tuple(entry.logprob for entry in self.entries)
        if any(left < right for left, right in zip(logprobs, logprobs[1:])):
            raise ValueError("top-k entries must be ordered by descending probability")

        if len(self.entries) == self.logical_vocab_size:
            if self.tail_logprob is not None:
                raise ValueError("full-vocabulary distributions must omit the tail")
            total = math.fsum(math.exp(value) for value in logprobs)
        else:
            if self.tail_logprob is None:
                raise ValueError("sparse top-k distributions must include a tail")
            total = math.fsum(
                [*(math.exp(value) for value in logprobs), math.exp(self.tail_logprob)]
            )
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(
                "teacher distribution must be normalized over the full vocabulary"
            )
        return self


class TeacherScoringResult(ImmutableModel):
    """Validated distributions plus the semantic identity echoed by a scorer."""

    request_id: Annotated[str, Field(min_length=1)]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    generation_id: Annotated[str, Field(min_length=1)]
    teacher_view_fingerprint: Annotated[str, Field(min_length=1)]
    forced_token_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_positions: tuple[Annotated[int, Field(ge=0)], ...]
    teacher_name: Annotated[str, Field(min_length=1)]
    teacher_revision: int | str
    token_space_fingerprint: Annotated[str, Field(min_length=1)]
    logical_vocab_size: PositiveInt
    target: TopK
    positions: tuple[ScoredPosition, ...]

    @classmethod
    def create(
        cls,
        *,
        request: TeacherScoringRequest,
        positions: tuple[ScoredPosition, ...],
    ) -> Self:
        result = cls(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            generation_id=request.generation_id,
            teacher_view_fingerprint=request.teacher_view.fingerprint,
            forced_token_sha256=request.forced_token_sha256,
            selected_positions=request.selected_positions,
            teacher_name=request.teacher_name,
            teacher_revision=request.teacher_revision,
            token_space_fingerprint=request.token_space_fingerprint,
            logical_vocab_size=request.logical_vocab_size,
            target=request.target,
            positions=positions,
        )
        return result.validate_for(request)

    @field_validator("teacher_revision")
    @classmethod
    def _valid_revision(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 0:
            raise ValueError("teacher revision must be non-negative")
        if isinstance(value, str) and not value:
            raise ValueError("teacher revision must not be empty")
        return value

    @model_validator(mode="after")
    def _valid_result(self) -> Self:
        if self.request_id != f"{_REQUEST_ID_PREFIX}{self.request_sha256}":
            raise ValueError("request ID does not match echoed request hash")
        if not self.selected_positions:
            raise ValueError("scoring result must contain selected positions")
        if tuple(sorted(set(self.selected_positions))) != self.selected_positions:
            raise ValueError("selected positions must be unique and increasing")
        actual_positions = tuple(row.position for row in self.positions)
        if actual_positions != self.selected_positions:
            raise ValueError(
                "scored positions must completely match selected positions in order"
            )
        expected_width = min(self.target.k, self.logical_vocab_size)
        for row in self.positions:
            if len(row.entries) != expected_width:
                raise ValueError("scored position does not have the exact top-k width")
            if row.logical_vocab_size != self.logical_vocab_size:
                raise ValueError("scored position logical vocabulary does not match")
            if row.temperature != self.target.temperature:
                raise ValueError("scored position temperature does not match target")
        return self

    def validate_for(self, request: TeacherScoringRequest) -> Self:
        """Reject a valid result if any request identity was echoed incorrectly."""

        echoes: tuple[tuple[str, object, object], ...] = (
            ("request ID", self.request_id, request.request_id),
            ("request hash", self.request_sha256, request.request_sha256),
            ("generation ID", self.generation_id, request.generation_id),
            (
                "teacher-view fingerprint",
                self.teacher_view_fingerprint,
                request.teacher_view.fingerprint,
            ),
            (
                "forced-token hash",
                self.forced_token_sha256,
                request.forced_token_sha256,
            ),
            ("selected positions", self.selected_positions, request.selected_positions),
            ("teacher name", self.teacher_name, request.teacher_name),
            ("teacher revision", self.teacher_revision, request.teacher_revision),
            (
                "token-space fingerprint",
                self.token_space_fingerprint,
                request.token_space_fingerprint,
            ),
            (
                "logical vocabulary",
                self.logical_vocab_size,
                request.logical_vocab_size,
            ),
            ("target", self.target, request.target),
        )
        for label, actual, expected in echoes:
            if actual != expected:
                raise ValueError(f"scoring result {label} does not match request")
        for row in self.positions:
            if row.forced_token_id != request.forced_token_ids[row.position]:
                raise ValueError(
                    "scoring result forced token does not match request continuation"
                )
        return self


class TeacherDistributionScorer(Protocol):
    """Acquire teacher distributions without exposing a transport to preparation."""

    async def score(
        self,
        request: TeacherScoringRequest,
    ) -> TeacherScoringResult: ...
