from types import SimpleNamespace

import pytest

from art.megatron.training.slot import (
    MegatronTrainingSlot,
    PreparedEmptySftForward,
    _ResidentRun,
)
from art.preprocessing.tokenize import SFTBatch
from art.training.contracts import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    LossConfig,
    OperationRef,
    SupervisedTrajectoryBatch,
)
from art.trajectories import Trajectory


@pytest.mark.asyncio
async def test_empty_sft_is_prepared_without_gpu_work_or_gradient() -> None:
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot.runtime_spec = SimpleNamespace(
        trainer_mesh=SimpleNamespace(
            ranks=(0, 1), topology=SimpleNamespace(tp=2, cp=1, pp=1)
        )
    )
    slot._sft_tokenizer = SimpleNamespace(
        tokenize=lambda *_args, **_kwargs: SFTBatch(
            trajectory_tensors=[],
            learning_rate=0.0,
            num_trajectories=0,
            num_tokens=0,
            num_trainable_tokens=0,
            num_dropped_trajectories=1,
        )
    )
    slot._runs = {
        "run": _ResidentRun.model_construct(
            registration=SimpleNamespace(),
            model=SimpleNamespace(build=lambda: object()),
            output_dir="/output",
            generation=SimpleNamespace(),
        )
    }
    request = ForwardBackwardRequest(
        run_id="run",
        request_id="request",
        sequence_id=0,
        batch=SupervisedTrajectoryBatch(trajectories=(Trajectory(),)),
        loss=LossConfig(name="cross_entropy"),
    )
    ref = OperationRef(
        run_id="run",
        operation_id="empty",
        sequence_id=0,
        learner_parent_version=0,
        kind="forward_backward",
    )

    prepared = await slot.prepare_forward_backward(ref, request)
    assert isinstance(prepared, PreparedEmptySftForward)
    result = slot._empty_sft_forward_result(prepared, backward=True)
    assert isinstance(result, ForwardBackwardResult)
    assert not result.produced_gradient
    assert result.packing.physical_tokens == 0
    assert result.loss_fn_outputs == ()
