from types import SimpleNamespace

import pytest

from art.megatron.runtime.specs import TrainerGeneration
from art.megatron.training.slot import MegatronTrainingSlot


@pytest.mark.asyncio
async def test_recovery_head_prepares_exact_write_before_authorization() -> None:
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=4,
        generation_id="step-00000004-0123456789abcdef0123456789abcdef",
        adapter_path="/tmp/adapter",
    )
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot._runs = {
        "run": SimpleNamespace(
            generation=generation,
            output_dir="/tmp/run",
            registration=SimpleNamespace(
                learner_version=4,
                optimizer_state_path="/tmp/run/optimizer_states",
            ),
        )
    }
    prepared = object()
    captured = []

    async def prepare(ref, request, source):
        captured.append((ref, request, source))
        return prepared

    slot._prepare_save = prepare
    slot.authorize_save = lambda *_args: pytest.fail(
        "recovery preparation authorized its physical write"
    )

    result = await slot.prepare_recovery_head(
        "run", learner_version=4, sequence_id=9
    )

    assert result is prepared
    assert len(captured) == 1
    ref, request, source = captured[0]
    assert ref.operation_id == request.request_id == (
        "art-recovery-step-00000004-0123456789abcdef0123456789abcdef"
    )
    assert ref.kind == "save_state"
    assert ref.sequence_id == request.sequence_id == 9
    assert source.generation == generation
