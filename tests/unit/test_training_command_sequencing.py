import pytest

from art import Trajectory
from art.training import (
    AdamConfig,
    CommandAdmissionPolicy,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    LossConfig,
    OptimStepRequest,
    RunCommandLedger,
    SupervisedTrajectoryBatch,
    TokenLogprobs,
)


def _batch() -> SupervisedTrajectoryBatch:
    return SupervisedTrajectoryBatch(
        trajectories=(
            Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            ),
        )
    )


def _forward(sequence_id: int, request_id: str, *, backward: bool = True):
    request = dict(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        batch=_batch(),
        loss=LossConfig(name="cross_entropy"),
    )
    return ForwardBackwardRequest(**request) if backward else ForwardRequest(**request)


@pytest.mark.asyncio
async def test_optimizer_seals_exact_order_and_projects_one_version() -> None:
    ledger = RunCommandLedger("run", learner_version=7)
    forward_only = await ledger.admit(
        _forward(0, "inspect", backward=False), kind="forward"
    )
    first = await ledger.admit(_forward(1, "fb-1"), kind="forward_backward")
    second = await ledger.admit(_forward(2, "fb-2"), kind="forward_backward")
    optimizer = await ledger.admit(
        OptimStepRequest(
            run_id="run",
            request_id="optim",
            sequence_id=3,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        kind="optim_step",
    )

    assert forward_only.ref.learner_parent_version == 7
    assert optimizer.contributing_forward_backward_operation_ids == (
        first.ref.operation_id,
        second.ref.operation_id,
    )
    assert optimizer.ref.reserved_output_learner_version == 8
    assert ledger.projected_learner_version == 8


@pytest.mark.asyncio
async def test_admission_is_gapless_and_idempotent() -> None:
    ledger = RunCommandLedger("run", learner_version=0)
    request = _forward(0, "fb")
    admission = await ledger.admit(request, kind="forward_backward")
    assert await ledger.admit(request, kind="forward_backward") == admission

    with pytest.raises(RuntimeError, match="different command"):
        await ledger.admit(
            request.model_copy(update={"return_token_logprobs": False}),
            kind="forward_backward",
        )
    with pytest.raises(RuntimeError, match="gapless"):
        await ledger.admit(_forward(2, "gap"), kind="forward_backward")


@pytest.mark.asyncio
async def test_gradient_contribution_limit_is_hard_bounded() -> None:
    ledger = RunCommandLedger(
        "run",
        learner_version=0,
        policy=CommandAdmissionPolicy(max_gradient_contributions=2),
    )
    await ledger.admit(_forward(0, "fb-0"), kind="forward_backward")
    await ledger.admit(_forward(1, "fb-1"), kind="forward_backward")

    with pytest.raises(RuntimeError, match="gradient contribution limit"):
        await ledger.admit(_forward(2, "fb-2"), kind="forward_backward")

    with pytest.raises(ValueError):
        CommandAdmissionPolicy(max_gradient_contributions=65)


@pytest.mark.asyncio
async def test_load_cannot_discard_open_gradients() -> None:
    ledger = RunCommandLedger("run", learner_version=2)
    await ledger.admit(_forward(0, "fb"), kind="forward_backward")

    with pytest.raises(RuntimeError, match="open gradients"):
        await ledger.admit(
            LoadStateRequest(
                run_id="run",
                request_id="load",
                sequence_id=1,
                checkpoint="wandb://checkpoint",
                restore_optimizer=True,
            ),
            kind="load_state",
        )


def test_token_logprobs_preserve_candidate_shape() -> None:
    values = TokenLogprobs.from_values([-0.5, -1.0, -1.5, -2.0], shape=(2, 2))

    assert values.shape == (2, 2)
    assert values.value_count == 4
    assert len(values.data) == 16
