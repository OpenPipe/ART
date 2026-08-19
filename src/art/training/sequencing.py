from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from pydantic import Field

from .contracts import (
    Contract,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    OperationKind,
    OperationRef,
    OptimStepRequest,
    RunCommand,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
)


class CommandAdmission(Contract):
    ref: OperationRef
    contributing_forward_backward_operation_ids: tuple[str, ...] = ()


class _AdmissionRecord(Contract):
    request_fingerprint: str = Field(min_length=1)
    admission: CommandAdmission


class RunCommandLedger:
    """Atomically admit one gapless, idempotent run-local command stream."""

    def __init__(self, run_id: str, *, learner_version: int) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if learner_version < 0:
            raise ValueError("learner_version must be non-negative")
        self.run_id = run_id
        self._next_sequence_id = 0
        self._projected_learner_version = learner_version
        self._open_forward_backward_ids: list[str] = []
        self._records: dict[str, _AdmissionRecord] = {}
        self._lock = asyncio.Lock()

    @property
    def projected_learner_version(self) -> int:
        return self._projected_learner_version

    @property
    def next_sequence_id(self) -> int:
        return self._next_sequence_id

    async def admit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
    ) -> CommandAdmission:
        async with self._lock:
            return self._admit(request, kind=kind)

    def retire(self, request_id: str, admission: CommandAdmission) -> None:
        record = self._records.get(request_id)
        if record is None:
            return
        if record.admission != admission:
            raise RuntimeError("command retirement does not match its admission")
        if admission.ref.operation_id in self._open_forward_backward_ids:
            raise RuntimeError("cannot retire an open F/B command")
        self._records.pop(request_id)

    def _admit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
    ) -> CommandAdmission:
        self._validate_request_kind(request, kind)
        if request.run_id != self.run_id:
            raise ValueError(
                f"command run_id {request.run_id!r} does not match {self.run_id!r}"
            )
        prior = self._records.get(request.request_id)
        request_fingerprint = _request_fingerprint(request)
        if prior is not None:
            if (
                prior.request_fingerprint != request_fingerprint
                or prior.admission.ref.kind != kind
            ):
                raise RuntimeError("request_id was reused for a different command")
            return prior.admission
        if request.sequence_id != self._next_sequence_id:
            raise RuntimeError(
                "command sequence must be gapless: "
                f"expected={self._next_sequence_id}, got={request.sequence_id}"
            )

        operation_id = hashlib.sha256(
            f"{self.run_id}\0{request.request_id}".encode()
        ).hexdigest()
        parent = self._projected_learner_version
        contributions: tuple[str, ...] = ()
        reserved: int | None = None
        if kind == "forward_backward":
            self._open_forward_backward_ids.append(operation_id)
        elif kind == "optim_step":
            if not self._open_forward_backward_ids:
                raise RuntimeError("optimizer requires an open F/B contribution")
            contributions = tuple(self._open_forward_backward_ids)
            self._open_forward_backward_ids.clear()
            reserved = parent + 1
            self._projected_learner_version = reserved
        elif kind == "load_state":
            if self._open_forward_backward_ids:
                raise RuntimeError("load_state cannot discard open gradients")
            reserved = parent + 1
            self._projected_learner_version = reserved

        admission = CommandAdmission(
            ref=OperationRef(
                run_id=self.run_id,
                operation_id=operation_id,
                sequence_id=request.sequence_id,
                learner_parent_version=parent,
                reserved_output_learner_version=reserved,
                kind=kind,
            ),
            contributing_forward_backward_operation_ids=contributions,
        )
        self._records[request.request_id] = _AdmissionRecord(
            request_fingerprint=request_fingerprint, admission=admission
        )
        self._next_sequence_id += 1
        return admission

    @staticmethod
    def _validate_request_kind(request: RunCommand, kind: OperationKind) -> None:
        expected = {
            "forward": ForwardRequest,
            "forward_backward": ForwardBackwardRequest,
            "optim_step": OptimStepRequest,
            "save_sampler": SaveWeightsForSamplerRequest,
            "save_state": SaveStateRequest,
            "load_state": LoadStateRequest,
        }[kind]
        if type(request) is not expected:
            raise TypeError(
                f"{kind} requires {expected.__name__}, got {type(request).__name__}"
            )


def _request_fingerprint(request: RunCommand) -> str:
    digest = hashlib.sha256()
    exclude = (
        {"batch": {"groups"}}
        if isinstance(request, ForwardRequest) and request.batch.kind == "rl"
        else None
    )
    metadata = json.dumps(
        request.model_dump(mode="json", exclude=exclude),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _update_digest(digest, metadata)
    if isinstance(request, ForwardRequest) and request.batch.kind == "rl":
        for group in request.batch.groups:
            _update_digest(digest, group.header)
            for record in group.records:
                _update_digest(digest, record)
    return digest.hexdigest()


def _update_digest(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
