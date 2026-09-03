from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from art.training.token_matrix import TextDatum

from .packing import PackingRequest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from art.model import TrainableModel
    from art.preprocessing.sft import SftBatchTokenizer


def canonical_manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        + b"\n"
    )


def manifest_value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_manifest_bytes(value)).hexdigest()


class PackingRequestSource(BaseModel):
    """Provenance for an exact pre-pack request, independent of replay placement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1, max_length=128)
    command_index: int | None = Field(default=None, ge=1)
    learner_parent_version: int | None = Field(default=None, ge=0)
    result_learner_version: int | None = Field(default=None, ge=0)
    text_datums: tuple[TextDatum, ...] = ()

    @model_validator(mode="after")
    def _validate_lineage(self) -> Self:
        if (
            self.learner_parent_version is not None
            and self.result_learner_version is not None
            and self.result_learner_version != self.learner_parent_version + 1
        ):
            raise ValueError("observed training input must advance one learner version")
        return self


class PackingRequestArtifact(BaseModel):
    """One content-addressed PackingRequest and the source that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: PackingRequestSource
    request: PackingRequest

    @classmethod
    def capture(
        cls,
        request: PackingRequest,
        *,
        source: PackingRequestSource,
    ) -> Self:
        payload = _request_payload(request=request, source=source)
        return cls(
            input_id=f"sha256:{manifest_value_sha256(payload)}",
            source=source,
            request=request,
        )

    @model_validator(mode="after")
    def _validate_input_id(self) -> Self:
        expected = "sha256:" + manifest_value_sha256(
            _request_payload(request=self.request, source=self.source)
        )
        if self.input_id != expected:
            raise ValueError("packing request input_id does not match its content")
        if self.source.text_datums:
            if self.request.loss.name != "cross_entropy":
                raise ValueError("text input evidence requires cross-entropy")
            datums = self.source.text_datums
            matrices = self.request.batch.matrices
            if tuple(value.datum_id for value in datums) != tuple(
                value.matrix_id for value in matrices
            ):
                raise ValueError("text datum identities do not match request matrices")
            if any(
                datum.packing_affinity_id != matrix.packing_affinity_id
                for datum, matrix in zip(datums, matrices, strict=True)
            ):
                raise ValueError("text datum packing affinities do not match")
            if any(
                not any(
                    float(value) != 0.0
                    for value in matrix.row("loss_weights").dense_values()
                )
                for matrix in matrices
            ):
                raise ValueError("text input evidence has no active target")
        return self


class PackingRequestManifest(BaseModel):
    """Neutral retained input contract shared by workflows and gate consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["art.packing_requests.v1"] = "art.packing_requests.v1"
    context: dict[str, JsonValue]
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: tuple[PackingRequestArtifact, ...] = Field(min_length=1)

    @classmethod
    def create(
        cls,
        *,
        context: dict[str, JsonValue],
        inputs: Sequence[PackingRequestArtifact],
    ) -> Self:
        return cls(
            context=context,
            context_sha256=manifest_value_sha256(context),
            inputs=tuple(inputs),
        )

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.context_sha256 != manifest_value_sha256(self.context):
            raise ValueError("packing request context digest does not match")
        input_ids = tuple(value.input_id for value in self.inputs)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("packing request manifest repeats an input")
        source_commands = tuple(
            (value.source.stage, value.source.command_index)
            for value in self.inputs
            if value.source.command_index is not None
        )
        if len(set(source_commands)) != len(source_commands):
            raise ValueError("packing request manifest repeats a source command")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_manifest_bytes(self.model_dump(mode="json"))


def packing_request_from_text_datums(
    datums: Sequence[TextDatum],
    *,
    model: TrainableModel,
    generation_id: str,
    packed_sequence_length: int,
    tokenizer: SftBatchTokenizer | None = None,
    return_token_logprobs: bool = True,
) -> PackingRequest:
    """Lower exact SFT text through the model's configured canonical tokenizer."""

    from art.preprocessing.sft import SftBatchTokenizer
    from art.preprocessing.token_matrix import token_matrix_batch_from_text_datums
    from art.training.token_matrix import NamedLossRequest

    batch = token_matrix_batch_from_text_datums(
        datums,
        model=model,
        tokenizer=tokenizer or SftBatchTokenizer(),
    )
    return PackingRequest(
        batch=batch,
        loss=NamedLossRequest(name="cross_entropy", normalize_advantages=False),
        return_token_logprobs=return_token_logprobs,
        generation_id=generation_id,
        packed_sequence_length=packed_sequence_length,
    )


def _request_payload(
    *,
    request: PackingRequest,
    source: PackingRequestSource,
) -> dict[str, object]:
    return {
        "source": source.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
    }
