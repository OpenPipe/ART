from types import SimpleNamespace

import pytest

from art.megatron import gate_evidence
from art.megatron.gate_evidence import MegatronGateAttemptPlan, MegatronGateTurn
from art.megatron.runtime.specs import TrainerGeneration
from art.training import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    OperationRef,
    OperationSucceeded,
    PackingOutcome,
    SamplerPublication,
    SaveWeightsForSamplerRequest,
)


@pytest.mark.asyncio
async def test_gate_checkpoint_adapter_accepts_nonpublishing_sampler_save(
    tmp_path,
) -> None:
    operation = OperationRef(
        run_id="run-1",
        operation_id="save-sampler-1",
        sequence_id=1,
        learner_parent_version=0,
        kind="save_sampler",
    )
    result = await gate_evidence.MegatronGateCheckpointOperations(
        SimpleNamespace(), tmp_path
    ).save_weights_for_sampler(
        SaveWeightsForSamplerRequest(
            run_id="run-1",
            request_id="save-sampler",
            sequence_id=1,
            checkpoint_name="open-accumulator",
            publication=SamplerPublication(mode="none"),
        ),
        operation,
        TrainerGeneration(
            training_session_id="session-1",
            policy_step=0,
            generation_id="step-00000000-00000000000000000000000000000000",
            adapter_path="/tmp/adapter",
        ),
    )

    assert result.operation_id == operation.operation_id
    assert result.checkpoint.checkpoint_id == "open-accumulator"
    assert result.lora == "/tmp/adapter"


@pytest.mark.asyncio
async def test_gate_recorder_materializes_deployed_operation_receipts(tmp_path) -> None:
    operation = OperationRef(
        run_id="run-1",
        operation_id="forward-backward-1",
        sequence_id=4,
        learner_parent_version=2,
        kind="forward_backward",
    )

    class Receipt:
        def model_dump_json(self, *, indent: int) -> str:
            assert indent == 2
            return '{"operation_id":"forward-backward-1"}'

    class Coordinator:
        def residency_evidence(self, run_id: str, operation_id: str):
            assert (run_id, operation_id) == ("run-1", "forward-backward-1")
            return {"run_id": run_id, "operation_id": operation_id}

        async def capture_forward_backward_numerics(
            self, *, run_id: str, operation_id: str, root: str
        ):
            assert (run_id, operation_id) == ("run-1", "forward-backward-1")
            assert root.endswith("artifacts/numerics")
            return Receipt()

    await gate_evidence.MegatronGateEvidenceRecorder(
        Coordinator(),
        tmp_path,  # type: ignore[arg-type]
    ).retain_outcome(
        ForwardBackwardRequest.model_construct(
            run_id="run-1", request_id="request-1", sequence_id=4
        ),
        OperationSucceeded.model_construct(
            operation=operation,
            result=ForwardBackwardResult(
                operation_id=operation.operation_id,
                packing=PackingOutcome(
                    packed_sequence_length=8,
                    packed_sequences=1,
                    target_packed_sequences=1,
                    physical_tokens=8,
                    non_padding_tokens=8,
                    loss_bearing_tokens=4,
                    trainable_assistant_tokens=4,
                ),
                produced_gradient=True,
            ),
        ),
        capture_numerics=True,
    )

    assert (tmp_path / "receipts/operations/forward-backward-1.json").is_file()
    assert (tmp_path / "receipts/residency/forward-backward-1.json").is_file()
    assert (tmp_path / "receipts/numerics/forward-backward-1.json").is_file()


def test_gate_schedule_allows_serial_runs_without_isolation_capture() -> None:
    runs = tuple(
        SimpleNamespace(
            bootstrap=SimpleNamespace(run_id=run_id),
            commands=(SimpleNamespace(kind="optim_step"),),
        )
        for run_id in ("run-1", "run-2")
    )
    capture_free = MegatronGateAttemptPlan.model_construct(
        slot=None,
        attempt_root="/tmp/gate",
        runs=runs,
        schedule=tuple(
            MegatronGateTurn(run_id=run_id, command_count=1)
            for run_id in ("run-1", "run-2")
        ),
    )
    assert capture_free._validate_runs() is capture_free

    partial_capture = capture_free.model_copy(
        update={
            "schedule": (
                MegatronGateTurn(
                    run_id="run-1", command_count=1, capture_isolation=True
                ),
                MegatronGateTurn(run_id="run-2", command_count=1),
            )
        }
    )
    with pytest.raises(ValueError, match="capture an active turn for every run"):
        partial_capture._validate_runs()


@pytest.mark.asyncio
async def test_gate_schedule_reuses_only_adjacent_isolation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class Recorder:
        async def capture_slot_state(
            self, *, turn_index: int, phase: str, run_ids: tuple[str, ...]
        ) -> None:
            calls.append(("capture", turn_index, phase, run_ids))

        def reuse_slot_state(
            self, *, source_turn_index: int, turn_index: int, run_count: int
        ) -> None:
            calls.append(("reuse", source_turn_index, turn_index, run_count))

        def finish(self, client) -> None:
            assert not client.open_forward_backward_operation_ids

    class Client:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            self.open_forward_backward_operation_ids = ()

        async def close(self) -> None:
            return None

    async def execute_commands(*_args: object) -> None:
        return None

    monkeypatch.setattr(gate_evidence, "_execute_commands", execute_commands)
    monkeypatch.setattr(
        gate_evidence.LocalMegatronTrainingClient,
        "from_binding",
        lambda binding: binding.client,
    )
    run_ids = ("run-1", "run-2", "run-3", "run-4")
    runs = tuple(
        SimpleNamespace(
            bootstrap=SimpleNamespace(run_id=run_id),
            commands=(SimpleNamespace(kind="optim_step"),),
        )
        for run_id in run_ids
    )
    plan = SimpleNamespace(
        runs=runs,
        schedule=tuple(
            MegatronGateTurn(
                run_id=run_id,
                command_count=1,
                capture_isolation=index != 2,
            )
            for index, run_id in enumerate(run_ids)
        ),
    )
    bound = [
        (
            run,
            SimpleNamespace(client=Client(run.bootstrap.run_id)),
        )
        for run in runs
    ]

    await gate_evidence._execute_schedule(Recorder(), plan, bound)  # type: ignore[arg-type]

    assert calls == [
        ("capture", 0, "before", run_ids),
        ("capture", 0, "after", run_ids),
        ("reuse", 0, 1, 4),
        ("capture", 1, "after", run_ids),
        ("capture", 3, "before", run_ids),
        ("capture", 3, "after", run_ids),
    ]
