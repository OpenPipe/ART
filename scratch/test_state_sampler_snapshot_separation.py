from types import SimpleNamespace

import pytest

from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.publication import (
    SnapshotRankWritePlan,
    build_snapshot_write_plan,
    build_snapshot_write_reservation_plan,
)
from art.megatron.runtime.specs import TrainerGeneration
from art.megatron.training.slot import MegatronTrainingSlot, _ExistingSnapshot
from art.training.contracts import (
    OperationRef,
    SamplerPublication,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
)


def _adapter() -> OptimizerAdapter:
    return OptimizerAdapter(
        identity="/tmp/adapter",
        training_session_id="session",
        step=3,
        generation_id="step-00000003-0123456789abcdef0123456789abcdef",
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
    )


def _ref(kind: str, operation_id: str) -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id=operation_id,
        sequence_id=0,
        learner_parent_version=3,
        kind=kind,
    )


def _slot(calls: list[tuple[bool, bool]]) -> MegatronTrainingSlot:
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot._results = {}
    slot._pending_results = {}
    slot._prepared_saves = {}
    slot._runs = {
        "run": SimpleNamespace(
            registration=SimpleNamespace(optimizer_state_path="/tmp/optimizer")
        )
    }

    async def start_snapshot(_ref, *, save_optimizer, publish_sampler, **_kwargs):
        calls.append((save_optimizer, publish_sampler))
        adapter = _adapter()
        generation = TrainerGeneration(
            training_session_id=adapter.training_session_id,
            policy_step=adapter.step,
            generation_id=adapter.generation_id,
            adapter_path=adapter.identity,
        )
        plan = build_snapshot_write_plan(
            operation_id=_ref.operation_id,
            generation=generation,
            ranks=(
                SnapshotRankWritePlan(
                    rank=0,
                    generation=generation,
                    adapter=adapter,
                    saves_optimizer=False,
                ),
            ),
        )
        return _ExistingSnapshot(
            adapter=adapter,
            optimizer_bytes=1,
            plan=plan,
            reservation_plan=build_snapshot_write_reservation_plan(plan),
        )

    slot._start_snapshot = start_snapshot
    return slot


@pytest.mark.asyncio
async def test_save_state_and_sampler_publication_are_independent() -> None:
    calls: list[tuple[bool, bool]] = []
    slot = _slot(calls)
    state = await slot.start_save_state(
        _ref("save_state", "save-state"),
        SaveStateRequest(
            run_id="run",
            request_id="save-state",
            sequence_id=0,
            checkpoint_name="state",
        ),
    )
    sampler = await slot.start_save_weights_for_sampler(
        _ref("save_sampler", "save-sampler"),
        SaveWeightsForSamplerRequest(
            run_id="run",
            request_id="save-sampler",
            sequence_id=0,
            checkpoint_name="sampler",
            publication=SamplerPublication(mode="none"),
        ),
    )
    await state
    await sampler
    assert calls == [(True, False), (False, True)]
