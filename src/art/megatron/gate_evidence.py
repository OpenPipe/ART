from __future__ import annotations

import os
from pathlib import Path
import tempfile

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
