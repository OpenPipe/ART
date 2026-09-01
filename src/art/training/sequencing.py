from __future__ import annotations

import asyncio
import hashlib
import json

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


class CommandAdmissionPolicy(Contract):
    max_command_depth: int = Field(default=128, ge=1)
    max_gradient_contributions: int = Field(default=64, ge=1, le=64)


class _AdmissionRecord(Contract):
    request_fingerprint: str = Field(min_length=64, max_length=64)
    admission: CommandAdmission


class RunCommandLedger:
    """Admit one bounded, gapless, idempotent run-local command stream."""

    def __init__(
        self,
        run_id: str,
        *,
        learner_version: int,
        initial_operation_sequence: int = 0,
        policy: CommandAdmissionPolicy | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if learner_version < 0:
            raise ValueError("learner_version must be non-negative")
        if initial_operation_sequence < 0:
            raise ValueError("initial_operation_sequence must be non-negative")
        self.run_id = run_id
        self._policy = policy or CommandAdmissionPolicy()
        self._next_sequence_id = initial_operation_sequence
        self._projected_learner_version = learner_version
        self._open_forward_backward_ids: list[str] = []
        self._records: dict[str, _AdmissionRecord] = {}
        self._nonterminal_request_ids: set[str] = set()
        self._failure: BaseException | None = None
        self._lock = asyncio.Lock()

    @property
    def projected_learner_version(self) -> int:
        return self._projected_learner_version

    @property
    def next_sequence_id(self) -> int:
        return self._next_sequence_id

    @property
    def open_forward_backward_operation_ids(self) -> tuple[str, ...]:
        return tuple(self._open_forward_backward_ids)

    async def admit(
        self, request: RunCommand, *, kind: OperationKind
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
        self._nonterminal_request_ids.discard(request_id)

    def mark_terminal(
        self,
        request_id: str,
        admission: CommandAdmission,
        *,
        error: BaseException | None,
    ) -> None:
        record = self._records.get(request_id)
        if record is None:
            return
        if record.admission != admission:
            raise RuntimeError("command terminal state does not match its admission")
        self._nonterminal_request_ids.discard(request_id)
        if error is not None and admission.ref.kind in {
            "forward_backward",
            "optim_step",
            "load_state",
        }:
            self._failure = self._failure or error
            self._open_forward_backward_ids.clear()

    def cancel_pending_forward_backward(
        self, request_id: str, admission: CommandAdmission
    ) -> None:
        """Remove an admitted F/B that completed without producing gradients."""

        record = self._records.get(request_id)
        if record is None:
            return
        if record.admission != admission:
            raise RuntimeError("F/B cancellation does not match its admission")
        operation_id = admission.ref.operation_id
        if admission.ref.kind != "forward_backward":
            raise RuntimeError("only a forward_backward admission can be cancelled")
        if operation_id in self._open_forward_backward_ids:
            self._open_forward_backward_ids.remove(operation_id)

    def _admit(self, request: RunCommand, *, kind: OperationKind) -> CommandAdmission:
        self._validate_request_kind(request, kind)
        if request.run_id != self.run_id:
            raise ValueError(
                f"command run_id {request.run_id!r} does not match {self.run_id!r}"
            )
        fingerprint = _request_fingerprint(request)
        prior = self._records.get(request.request_id)
        if prior is not None:
            if (
                prior.request_fingerprint != fingerprint
                or prior.admission.ref.kind != kind
            ):
                raise RuntimeError("request_id was reused for a different command")
            return prior.admission
        if self._failure is not None:
            raise RuntimeError(
                "training run command stream has failed"
            ) from self._failure
        if len(self._nonterminal_request_ids) >= self._policy.max_command_depth:
            raise RuntimeError("training run command depth limit reached")
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
            if len(self._open_forward_backward_ids) >= (
                self._policy.max_gradient_contributions
            ):
                raise RuntimeError("training run gradient contribution limit reached")
            self._open_forward_backward_ids.append(operation_id)
        elif kind == "optim_step":
            if not self._open_forward_backward_ids:
                raise RuntimeError("optimizer requires an open F/B contribution")
            contributions = tuple(self._open_forward_backward_ids)
            self._open_forward_backward_ids.clear()
            reserved = parent + 1
        elif kind == "load_state":
            if self._open_forward_backward_ids:
                raise RuntimeError("load_state cannot discard open gradients")
            reserved = parent + 1

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
            request_fingerprint=fingerprint, admission=admission
        )
        self._nonterminal_request_ids.add(request.request_id)
        self._next_sequence_id += 1
        if reserved is not None:
            self._projected_learner_version = reserved
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
    payload = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
