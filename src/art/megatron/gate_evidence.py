from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed import ArtLaunchContext
from art.training import (
    CheckpointRef,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    LoadStateRequest,
    OperationExecutionOutcome,
    OperationFailed,
    OperationRef,
    OptimStepRequest,
    RunCommand,
    RunCommandLedger,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.training.contracts import OperationKind

from .operation_handler import (
    MegatronArtifactResourcePlan,
    MegatronLoadedState,
    MegatronSamplerPublicationReceipt,
)
from .runtime.portable_snapshot import PortableSnapshotExportReceipt
from .runtime.specs import TrainerGeneration
from .slot_coordinator import MegatronSlotCoordinator, MegatronSlotRun
from .slot_runtime import (
    MegatronRunBinding,
    MegatronRunBootstrapConfig,
    MegatronSlotLaunchConfig,
    launch_megatron_slot,
)

ART_MEGATRON_GATE_PLAN_ENV = "ART_MEGATRON_GATE_PLAN"


class _GateContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MegatronGateCommand(_GateContract):
    kind: Literal[
        "forward",
        "forward_backward",
        "optim_step",
        "save_sampler",
        "save_state",
        "load_state",
    ]
    request: dict[str, Any]
    capture_numerics: bool = False

    @model_validator(mode="after")
    def _validate_capture(self) -> "MegatronGateCommand":
        if self.capture_numerics and self.kind != "forward_backward":
            raise ValueError("only forward_backward accepts numerical capture")
        return self


class MegatronGateRunPlan(_GateContract):
    bootstrap: MegatronRunBootstrapConfig
    commands: tuple[MegatronGateCommand, ...] = Field(min_length=1, max_length=128)


class MegatronGateTurn(_GateContract):
    run_id: str = Field(min_length=1)
    command_count: int = Field(ge=1, le=128)
    capture_isolation: bool = False


class MegatronGateAttemptPlan(_GateContract):
    slot: MegatronSlotLaunchConfig
    attempt_root: str = Field(min_length=1)
    runs: tuple[MegatronGateRunPlan, ...] = Field(min_length=1, max_length=4)
    schedule: tuple[MegatronGateTurn, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def _validate_runs(self) -> "MegatronGateAttemptPlan":
        run_ids = tuple(run.bootstrap.run_id for run in self.runs)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("gate run IDs must be unique")
        if not self.schedule:
            return self
        if len(run_ids) < 2:
            raise ValueError("gate isolation schedule requires multiple resident runs")
        runs = {run.bootstrap.run_id: run for run in self.runs}
        consumed = dict.fromkeys(run_ids, 0)
        captured: set[str] = set()
        for turn in self.schedule:
            run = runs.get(turn.run_id)
            if run is None:
                raise ValueError("gate schedule names an unknown run")
            start = consumed[turn.run_id]
            stop = start + turn.command_count
            if stop > len(run.commands):
                raise ValueError("gate schedule consumes beyond a run command stream")
            commands = run.commands[start:stop]
            if turn.capture_isolation:
                if not any(
                    command.kind in {"optim_step", "load_state"} for command in commands
                ):
                    raise ValueError(
                        "isolation capture must contain a state-changing command"
                    )
                captured.add(turn.run_id)
            consumed[turn.run_id] = stop
        if any(consumed[run_id] != len(runs[run_id].commands) for run_id in run_ids):
            raise ValueError(
                "gate schedule must consume every run command exactly once"
            )
        if captured != set(run_ids):
            raise ValueError("gate schedule must capture an active turn for every run")
        return self


class MegatronGateEvidenceRecorder:
    """Record immutable operation evidence from the live slot coordinator."""

    def __init__(self, coordinator: MegatronSlotCoordinator, root: str | Path) -> None:
        self.coordinator = coordinator
        self.root = Path(root).resolve()

    async def execute(
        self,
        run: MegatronSlotRun,
        ledger: RunCommandLedger,
        request: RunCommand,
        *,
        capture_numerics: bool = False,
    ) -> OperationExecutionOutcome:
        kind = _command_kind(request)
        admission = await ledger.admit(request, kind=kind)
        outcome = await run.worker.execute(
            request,
            admission.ref,
            admission.contributing_forward_backward_operation_ids,
        )
        error = None
        gradient_produced = False
        if isinstance(outcome, OperationFailed):
            error = RuntimeError(f"{outcome.failure.code}: {outcome.failure.message}")
        elif isinstance(request, ForwardBackwardRequest):
            result = outcome.result
            assert isinstance(result, ForwardBackwardResult)
            gradient_produced = result.produced_gradient
            if not gradient_produced:
                ledger.cancel_pending_forward_backward(request.request_id, admission)
        ledger.mark_terminal(request.request_id, admission, error=error)

        _write_json(
            self.root
            / "receipts"
            / "operations"
            / f"{admission.ref.operation_id}.json",
            outcome.model_dump_json(indent=2).encode() + b"\n",
        )
        residency = self.coordinator.residency_evidence(
            request.run_id, admission.ref.operation_id
        )
        if residency is not None:
            _write_json(
                self.root
                / "receipts"
                / "residency"
                / f"{admission.ref.operation_id}.json",
                json.dumps(residency, indent=2, sort_keys=True).encode() + b"\n",
            )
        if gradient_produced and capture_numerics:
            receipt = await self.coordinator.capture_forward_backward_numerics(
                run_id=request.run_id,
                operation_id=admission.ref.operation_id,
                root=str(self.root / "artifacts" / "numerics"),
            )
            _write_json(
                self.root
                / "receipts"
                / "numerics"
                / f"{admission.ref.operation_id}.json",
                receipt.model_dump_json(indent=2).encode() + b"\n",
            )
        return outcome

    async def capture_slot_state(
        self,
        *,
        turn_index: int,
        phase: Literal["before", "after"],
        run_ids: tuple[str, ...],
    ) -> None:
        for run_index, run_id in enumerate(run_ids):
            export_id = hashlib.sha256(
                f"gate-isolation\0{turn_index}\0{phase}\0{run_id}".encode()
            ).hexdigest()
            receipt = await self.coordinator.capture_run_checkpoint(run_id, export_id)
            path = (
                self.root
                / "receipts"
                / "isolation"
                / f"turn-{turn_index:03d}"
                / phase
                / f"run-{run_index:03d}.json"
            )
            _write_json(
                path,
                receipt.model_dump_json(indent=2).encode() + b"\n",
            )


class MegatronGateCheckpointOperations:
    """Gate-only save-state port that records the live portable export receipt."""

    def __init__(self, coordinator: MegatronSlotCoordinator, root: str | Path) -> None:
        self.coordinator = coordinator
        self.root = Path(root).resolve()

    async def save_state(
        self,
        request: SaveStateRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SaveStateResult:
        receipt = await self.coordinator.export_run_checkpoint(operation)
        _validate_export(receipt, operation, generation)
        _write_json(
            self.root / "receipts" / "save-state" / f"{operation.operation_id}.json",
            receipt.model_dump_json(indent=2).encode() + b"\n",
        )
        return SaveStateResult(
            operation_id=operation.operation_id,
            checkpoint=CheckpointRef(
                run_id=operation.run_id,
                learner_version=generation.policy_step,
                checkpoint_id=receipt.archive.receipt_sha256,
            ),
        )

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> MegatronSamplerPublicationReceipt:
        del request, operation, generation
        raise RuntimeError(
            "Gate sampler publication requires the production checkpoint adapter"
        )

    async def load_state(
        self,
        request: LoadStateRequest,
        operation: OperationRef,
    ) -> MegatronLoadedState:
        del request, operation
        raise RuntimeError("Gate load requires the production checkpoint adapter")

    async def plan_artifacts(
        self,
        request: SaveWeightsForSamplerRequest | SaveStateRequest | LoadStateRequest,
        generation: TrainerGeneration,
    ) -> MegatronArtifactResourcePlan:
        del request, generation
        raise RuntimeError(
            "Gate artifact admission requires the production checkpoint adapter"
        )


async def run_megatron_gate_attempt(launch: ArtLaunchContext) -> None:
    """Execute one JSON-planned functional slice on a persistent trainer slot."""

    try:
        plan_path = os.environ[ART_MEGATRON_GATE_PLAN_ENV]
    except KeyError:
        raise RuntimeError(f"{ART_MEGATRON_GATE_PLAN_ENV} is required") from None
    plan = MegatronGateAttemptPlan.model_validate_json(Path(plan_path).read_bytes())
    slot = await launch_megatron_slot(plan.slot, launch=launch)
    recorder = MegatronGateEvidenceRecorder(slot.coordinator, plan.attempt_root)
    try:
        bound: list[tuple[MegatronGateRunPlan, MegatronRunBinding]] = []
        for run_plan in plan.runs:
            binding = await slot.bind_run(
                run_plan.bootstrap,
                checkpoints=MegatronGateCheckpointOperations(
                    slot.coordinator, plan.attempt_root
                ),
            )
            bound.append((run_plan, binding))
        if plan.schedule:
            await _execute_schedule(recorder, plan, bound)
        else:
            await asyncio.gather(
                *(
                    _execute_run_plan(recorder, run_plan, binding)
                    for run_plan, binding in bound
                )
            )
    finally:
        await slot.aclose()


@dataclass(slots=True)
class _GateRunExecution:
    plan: MegatronGateRunPlan
    binding: MegatronRunBinding
    ledger: RunCommandLedger
    cursor: int = 0


async def _execute_schedule(
    recorder: MegatronGateEvidenceRecorder,
    plan: MegatronGateAttemptPlan,
    bound: list[tuple[MegatronGateRunPlan, MegatronRunBinding]],
) -> None:
    runs = {
        run_plan.bootstrap.run_id: _GateRunExecution(
            plan=run_plan,
            binding=binding,
            ledger=RunCommandLedger(
                run_plan.bootstrap.run_id,
                learner_version=binding.config.source.policy_step,
            ),
        )
        for run_plan, binding in bound
    }
    run_ids = tuple(run.bootstrap.run_id for run in plan.runs)
    for turn_index, turn in enumerate(plan.schedule):
        execution = runs[turn.run_id]
        if turn.capture_isolation:
            await recorder.capture_slot_state(
                turn_index=turn_index,
                phase="before",
                run_ids=run_ids,
            )
        stop = execution.cursor + turn.command_count
        await _execute_commands(
            recorder,
            execution.plan.commands[execution.cursor : stop],
            execution.binding,
            execution.ledger,
        )
        execution.cursor = stop
        if turn.capture_isolation:
            await recorder.capture_slot_state(
                turn_index=turn_index,
                phase="after",
                run_ids=run_ids,
            )
    for execution in runs.values():
        if execution.ledger.open_forward_backward_operation_ids:
            raise RuntimeError("gate run ended with open F/B contributions")


async def _execute_run_plan(
    recorder: MegatronGateEvidenceRecorder,
    plan: MegatronGateRunPlan,
    binding: MegatronRunBinding,
) -> None:
    ledger = RunCommandLedger(
        plan.bootstrap.run_id,
        learner_version=binding.config.source.policy_step,
    )
    try:
        await _execute_commands(recorder, plan.commands, binding, ledger)
        if ledger.open_forward_backward_operation_ids:
            raise RuntimeError("gate run ended with open F/B contributions")
    finally:
        await recorder.coordinator.drain_run(plan.bootstrap.run_id)


async def _execute_commands(
    recorder: MegatronGateEvidenceRecorder,
    commands: tuple[MegatronGateCommand, ...],
    binding: MegatronRunBinding,
    ledger: RunCommandLedger,
) -> None:
    for command in commands:
        request = _parse_command(command)
        outcome = await recorder.execute(
            binding.run,
            ledger,
            request,
            capture_numerics=command.capture_numerics,
        )
        if isinstance(outcome, OperationFailed):
            raise RuntimeError(f"{outcome.failure.code}: {outcome.failure.message}")


def _parse_command(command: MegatronGateCommand) -> RunCommand:
    request_type = {
        "forward": ForwardRequest,
        "forward_backward": ForwardBackwardRequest,
        "optim_step": OptimStepRequest,
        "save_sampler": SaveWeightsForSamplerRequest,
        "save_state": SaveStateRequest,
        "load_state": LoadStateRequest,
    }[command.kind]
    return cast(RunCommand, request_type.model_validate(command.request))


def _command_kind(request: RunCommand) -> OperationKind:
    if isinstance(request, ForwardBackwardRequest):
        return "forward_backward"
    if isinstance(request, ForwardRequest):
        return "forward"
    if isinstance(request, OptimStepRequest):
        return "optim_step"
    if isinstance(request, SaveWeightsForSamplerRequest):
        return "save_sampler"
    if isinstance(request, SaveStateRequest):
        return "save_state"
    if isinstance(request, LoadStateRequest):
        return "load_state"
    raise TypeError(f"unsupported command type {type(request).__name__}")


def _validate_export(
    receipt: PortableSnapshotExportReceipt,
    operation: OperationRef,
    generation: TrainerGeneration,
) -> None:
    exported = receipt.generation
    if (
        receipt.export_id != operation.operation_id
        or exported.training_session_id != generation.training_session_id
        or exported.policy_step != generation.policy_step
        or exported.generation_id != generation.generation_id
    ):
        raise RuntimeError("portable export changed the active save-state identity")


def _write_json(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"gate evidence changed: {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        candidate = Path(temporary.name)
    try:
        os.link(candidate, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"gate evidence changed: {path.name}") from None
    finally:
        candidate.unlink(missing_ok=True)
